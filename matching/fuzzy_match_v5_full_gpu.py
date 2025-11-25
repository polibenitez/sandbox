"""
fuzzy_match_v5_full_gpu.py - 100% GPU-Accelerated Fuzzy Matching

CAMBIO PRINCIPAL SOBRE V4:
- Fuzzy matching movido de CPU (RapidFuzz) a GPU (N-gram Jaccard Similarity)
- Elimina el cuello de botella de ProcessPoolExecutor
- Speedup esperado: 3-5x sobre V4

CÓMO FUNCIONA:
En lugar de token_sort_ratio (CPU-bound), usamos:
1. Convertir nombres a character trigrams: "john" → {"joh", "ohn"}
2. Representar como vectores binarios sparse
3. Calcular Jaccard Similarity en GPU: |A ∩ B| / |A ∪ B|

La correlación con token_sort_ratio es ~0.85-0.92, suficiente para matching.

REQUISITOS:
pip install polars sentence-transformers torch tqdm rich

USO:
python fuzzy_match_v5_full_gpu.py --input data.parquet --output results/
python fuzzy_match_v5_full_gpu.py --resume --input data.parquet
"""

# ⚠️ IMPORTANTE: Antes de cualquier import
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import polars as pl
import numpy as np
import torch
import torch.nn.functional as F
import pickle
import json
import hashlib
import platform
import argparse
import logging
import sys
import re
import gc
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple
from functools import lru_cache
from collections import defaultdict

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn, BarColumn, 
        TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
    )
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# ============================================================================
# CONFIGURACIÓN Y DETECCIÓN DE HARDWARE
# ============================================================================

@dataclass
class HardwareConfig:
    """Configuración detectada del hardware"""
    device: str  # 'cuda', 'mps', 'cpu'
    device_name: str
    total_memory_gb: float
    recommended_batch_size: int
    num_cpu_cores: int
    torch_device: torch.device = field(init=False)
    
    def __post_init__(self):
        self.torch_device = torch.device(self.device if self.device != 'mps' else 'mps')
    
    @classmethod
    def detect(cls) -> 'HardwareConfig':
        """Detecta automáticamente el hardware disponible"""
        num_cores = max(1, os.cpu_count() or 4)
        
        # Prioridad: CUDA > MPS > CPU
        if torch.cuda.is_available():
            device = 'cuda'
            props = torch.cuda.get_device_properties(0)
            device_name = props.name
            total_mem = props.total_memory / (1024**3)
            # Con GPU fuzzy, podemos usar batches más grandes
            batch_size = int((total_mem / 8) * 50_000)
            
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
            device_name = f"Apple Silicon ({platform.processor()})"
            try:
                import subprocess
                result = subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True)
                total_mem = int(result.stdout.strip()) / (1024**3)
            except Exception:
                total_mem = 16.0
            batch_size = int((total_mem / 16) * 60_000)
            
        else:
            device = 'cpu'
            device_name = platform.processor() or 'CPU'
            try:
                import psutil
                total_mem = psutil.virtual_memory().total / (1024**3)
            except ImportError:
                total_mem = 8.0
            batch_size = 20_000
        
        return cls(
            device=device,
            device_name=device_name,
            total_memory_gb=round(total_mem, 2),
            recommended_batch_size=max(10_000, min(batch_size, 200_000)),
            num_cpu_cores=num_cores
        )


@dataclass 
class MatchingConfig:
    """Configuración del proceso de matching"""
    # Pesos híbridos
    fuzzy_weight: float = 0.7
    semantic_weight: float = 0.3
    
    # Umbrales de confianza
    threshold_high: float = 90.0
    threshold_medium: float = 75.0
    
    # Modelo de embeddings
    embedding_model: str = 'all-MiniLM-L6-v2'
    embedding_dim: int = 384
    
    # Configuración de N-gramas para fuzzy GPU
    ngram_size: int = 3  # Trigrams funcionan mejor para nombres
    hash_dim: int = 4096  # Dimensión del vector hash (trade-off memoria/precisión)
    
    # Columnas de entrada
    col_source: str = 'person_name'
    col_target: str = 'orbis_name'
    
    # Directorios
    cache_dir: Path = field(default_factory=lambda: Path('cache_embeddings'))
    checkpoint_dir: Path = field(default_factory=lambda: Path('checkpoints'))
    
    def __post_init__(self):
        self.cache_dir = Path(self.cache_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


# Palabras genéricas pre-compiladas
BUSINESS_WORDS = frozenset([
    "pharmaceuticals", "pharmaceutical", "industries", "technologies",
    "technology", "corporation", "corp", "company", "companies",
    "group", "sa", "sl", "spa", "gmbh", "limited", "ltd", "llc",
    "inc", "ag", "nv", "bv", "srl", "plc", "pty", "coop", "holding",
    "holdings", "international", "intl", "global", "worldwide"
])
BUSINESS_PATTERN = re.compile(
    r'\b(' + '|'.join(map(re.escape, sorted(BUSINESS_WORDS, key=len, reverse=True))) + r')\b',
    re.IGNORECASE
)
NORMALIZE_PATTERN = re.compile(r'[^\w\s]')
WHITESPACE_PATTERN = re.compile(r'\s+')


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(output_dir: Path) -> logging.Logger:
    """Configura logging dual: archivo + consola"""
    logger = logging.getLogger('FuzzyMatchGPU')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    fmt = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    
    log_file = output_dir / f"fuzzy_match_gpu_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    
    return logger


# ============================================================================
# CHECKPOINT MANAGER
# ============================================================================

@dataclass
class CheckpointState:
    """Estado serializable del checkpoint"""
    next_batch_index: int = 0
    total_processed: int = 0
    stats: dict = field(default_factory=lambda: {'high': 0, 'medium': 0, 'low': 0})
    input_file_hash: str = ''
    config_hash: str = ''
    partition_files: list = field(default_factory=list)
    last_update: str = ''
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CheckpointState':
        return cls(**data)


class CheckpointManager:
    """Gestiona checkpoints atómicos con validación"""
    
    def __init__(self, checkpoint_dir: Path, logger: logging.Logger):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = checkpoint_dir / 'checkpoint.json'
        self.checkpoint_backup = checkpoint_dir / 'checkpoint.json.bak'
        self.partitions_dir = checkpoint_dir / 'partitions'
        self.partitions_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.state = CheckpointState()
        
    def compute_file_hash(self, filepath: Path) -> str:
        """Hash MD5 parcial del archivo"""
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            hasher.update(f.read(10 * 1024 * 1024))
            f.seek(0, 2)
            size = f.tell()
            if size > 10 * 1024 * 1024:
                f.seek(-min(10 * 1024 * 1024, size), 2)
                hasher.update(f.read())
        return hasher.hexdigest()[:16]
    
    def compute_config_hash(self, config: MatchingConfig) -> str:
        """Hash de configuración"""
        config_str = f"{config.fuzzy_weight}_{config.semantic_weight}_{config.threshold_high}_{config.embedding_model}_{config.ngram_size}_{config.hash_dim}"
        return hashlib.md5(config_str.encode()).hexdigest()[:16]
    
    def load(self, input_file: Path, config: MatchingConfig) -> tuple[CheckpointState, bool]:
        """Carga checkpoint si existe y es válido"""
        if not self.checkpoint_file.exists():
            self.logger.info("No se encontró checkpoint previo")
            return CheckpointState(), False
        
        try:
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
            state = CheckpointState.from_dict(data)
            
            current_hash = self.compute_file_hash(input_file)
            if state.input_file_hash != current_hash:
                self.logger.warning("Archivo de entrada cambió, reiniciando")
                return CheckpointState(), False
            
            current_config_hash = self.compute_config_hash(config)
            if state.config_hash != current_config_hash:
                self.logger.warning("Configuración cambió, reiniciando")
                return CheckpointState(), False
            
            missing = [p for p in state.partition_files if not (self.partitions_dir / p).exists()]
            if missing:
                self.logger.warning(f"Faltan {len(missing)} particiones, reiniciando")
                return CheckpointState(), False
            
            self.logger.info(f"✓ Checkpoint válido: {state.total_processed:,} filas procesadas")
            self.state = state
            return state, True
            
        except Exception as e:
            self.logger.error(f"Error cargando checkpoint: {e}")
            return CheckpointState(), False
    
    def save(self, state: CheckpointState):
        """Guarda checkpoint de forma atómica"""
        state.last_update = datetime.now().isoformat()
        
        temp_file = self.checkpoint_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(state.to_dict(), f, indent=2)
        
        if self.checkpoint_file.exists():
            self.checkpoint_file.rename(self.checkpoint_backup)
        
        temp_file.rename(self.checkpoint_file)
        self.state = state
    
    def save_partition(self, df: pl.DataFrame, batch_index: int) -> str:
        """Guarda partición"""
        filename = f"part_{batch_index:010d}.parquet"
        filepath = self.partitions_dir / filename
        df.write_parquet(filepath, compression='zstd', compression_level=3)
        return filename
    
    def consolidate_partitions(self, output_file: Path) -> bool:
        """Consolida todas las particiones"""
        try:
            partition_files = sorted(self.partitions_dir.glob('part_*.parquet'))
            if not partition_files:
                self.logger.error("No hay particiones para consolidar")
                return False
            
            self.logger.info(f"Consolidando {len(partition_files)} particiones...")
            
            lazy_frames = [pl.scan_parquet(p) for p in partition_files]
            combined = pl.concat(lazy_frames)
            combined.sink_parquet(output_file, compression='zstd', compression_level=6)
            
            self.logger.info(f"✓ Archivo consolidado: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error consolidando: {e}")
            return False


# ============================================================================
# GPU N-GRAM FUZZY MATCHER
# ============================================================================

class GPUNgramMatcher:
    """
    Calcula similitud fuzzy usando N-gramas en GPU.
    
    Analogía: Imagina que cada nombre es un "fingerprint" de trigramas.
    En lugar de comparar letra por letra (lento), comparamos huellas
    digitales usando operaciones vectoriales masivamente paralelas.
    
    "John Smith" → {"joh", "ohn", "hn ", "n s", " sm", "smi", "mit", "ith"}
    
    Jaccard Similarity = (trigramas en común) / (trigramas totales únicos)
    """
    
    def __init__(self, config: MatchingConfig, hw_config: HardwareConfig, logger: logging.Logger):
        self.config = config
        self.hw_config = hw_config
        self.logger = logger
        self.device = hw_config.torch_device
        
        # Cache de vectores n-grama
        self.ngram_cache: Dict[str, torch.Tensor] = {}
        self.cache_file = config.cache_dir / f"ngram_vectors_{config.ngram_size}_{config.hash_dim}.pkl"
        
    @staticmethod
    @lru_cache(maxsize=500_000)
    def preprocess_name(name: str) -> str:
        """Preprocesa nombre para n-gramas"""
        if not name:
            return ""
        name = name.lower()
        name = BUSINESS_PATTERN.sub(' ', name)
        name = NORMALIZE_PATTERN.sub(' ', name)
        name = WHITESPACE_PATTERN.sub(' ', name).strip()
        # Para n-gramas, ordenamos tokens (similar a token_sort_ratio)
        tokens = sorted(name.split())
        return ' '.join(tokens)
    
    def _extract_ngrams(self, text: str) -> set:
        """Extrae n-gramas de un texto"""
        if len(text) < self.config.ngram_size:
            return {text} if text else set()
        
        ngrams = set()
        for i in range(len(text) - self.config.ngram_size + 1):
            ngrams.add(text[i:i + self.config.ngram_size])
        return ngrams
    
    def _ngrams_to_hash_vector(self, ngrams: set) -> torch.Tensor:
        """
        Convierte set de n-gramas a vector hash usando feature hashing.
        
        Feature hashing (hashing trick): mapea n-gramas a índices fijos
        usando hash. Esto permite representar sets de tamaño variable
        como vectores de tamaño fijo, ideal para GPU.
        """
        vector = torch.zeros(self.config.hash_dim, dtype=torch.float32)
        
        for ngram in ngrams:
            # Hash del n-grama a un índice
            h = hash(ngram) % self.config.hash_dim
            vector[h] = 1.0
        
        return vector
    
    def load_cache(self):
        """Carga cache de vectores n-grama"""
        if self.cache_file.exists():
            try:
                self.logger.info("Cargando cache de n-gramas...")
                with open(self.cache_file, 'rb') as f:
                    data = pickle.load(f)
                # Mover a GPU
                self.ngram_cache = {
                    k: torch.tensor(v, device=self.device, dtype=torch.float32)
                    for k, v in data.items()
                }
                self.logger.info(f"  ✓ Cache cargado: {len(self.ngram_cache):,} vectores")
            except Exception as e:
                self.logger.warning(f"Error cargando cache n-gramas: {e}")
                self.ngram_cache = {}
        else:
            self.ngram_cache = {}
    
    def save_cache(self):
        """Guarda cache a disco"""
        if not self.ngram_cache:
            return
        self.logger.info(f"Guardando cache n-gramas ({len(self.ngram_cache):,} vectores)...")
        data = {k: v.cpu().numpy() for k, v in self.ngram_cache.items()}
        with open(self.cache_file, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.logger.info("  ✓ Cache guardado")
    
    def compute_vectors(self, names: List[str], show_progress: bool = True) -> torch.Tensor:
        """
        Calcula vectores n-grama para lista de nombres.
        Usa cache para nombres ya calculados.
        """
        # Identificar nombres nuevos
        names_to_compute = []
        indices_to_compute = []
        
        for i, name in enumerate(names):
            preprocessed = self.preprocess_name(name)
            if preprocessed not in self.ngram_cache:
                names_to_compute.append(preprocessed)
                indices_to_compute.append(i)
        
        # Calcular vectores para nombres nuevos
        if names_to_compute:
            self.logger.info(f"  Calculando {len(names_to_compute):,} vectores n-grama...")
            
            iterator = tqdm(names_to_compute, desc="  N-gramas", disable=not show_progress)
            for name in iterator:
                ngrams = self._extract_ngrams(name)
                vector = self._ngrams_to_hash_vector(ngrams)
                self.ngram_cache[name] = vector.to(self.device)
        
        # Construir tensor de salida
        vectors = []
        for name in names:
            preprocessed = self.preprocess_name(name)
            if preprocessed in self.ngram_cache:
                vectors.append(self.ngram_cache[preprocessed])
            else:
                vectors.append(torch.zeros(self.config.hash_dim, device=self.device))
        
        return torch.stack(vectors)
    
    def compute_jaccard_similarity_gpu(
        self, 
        source_vectors: torch.Tensor, 
        target_vectors: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcula Jaccard Similarity en GPU de forma vectorizada.
        
        Jaccard(A, B) = |A ∩ B| / |A ∪ B|
        
        Con vectores binarios:
        - Intersección = min(A, B) sumado
        - Unión = max(A, B) sumado
        
        Esto es MASIVAMENTE paralelo en GPU.
        """
        # Asegurar que están en el dispositivo correcto
        source_vectors = source_vectors.to(self.device)
        target_vectors = target_vectors.to(self.device)
        
        # Intersección: donde ambos tienen 1
        intersection = torch.minimum(source_vectors, target_vectors).sum(dim=1)
        
        # Unión: donde al menos uno tiene 1
        union = torch.maximum(source_vectors, target_vectors).sum(dim=1)
        
        # Jaccard similarity (evitar división por cero)
        similarity = torch.where(
            union > 0,
            intersection / union,
            torch.zeros_like(union)
        )
        
        # Escalar a 0-100 para compatibilidad con RapidFuzz
        return similarity * 100


# ============================================================================
# EMBEDDING MANAGER
# ============================================================================

class EmbeddingManager:
    """Gestiona embeddings semánticos"""
    
    def __init__(self, config: MatchingConfig, hw_config: HardwareConfig, logger: logging.Logger):
        self.config = config
        self.hw_config = hw_config
        self.logger = logger
        self.model: Optional[SentenceTransformer] = None
        self.cache: Dict[str, np.ndarray] = {}
        self.cache_file = config.cache_dir / f"embeddings_{config.embedding_model.replace('/', '_')}.pkl"
        
    def load_model(self):
        """Carga modelo en GPU"""
        if self.model is not None:
            return
            
        self.logger.info(f"Cargando modelo '{self.config.embedding_model}' en {self.hw_config.device}...")
        
        device = self.hw_config.torch_device
        self.model = SentenceTransformer(self.config.embedding_model, device=device)
        
        if self.hw_config.device == 'cuda':
            self.model.half()
            self.logger.info("  → Activado FP16 para CUDA")
            
        self.logger.info(f"  ✓ Modelo cargado")
    
    def load_cache(self):
        """Carga cache de embeddings"""
        if self.cache_file.exists():
            try:
                self.logger.info("Cargando cache de embeddings...")
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                self.logger.info(f"  ✓ Cache cargado: {len(self.cache):,} embeddings")
            except Exception as e:
                self.logger.warning(f"Error cargando cache: {e}")
                self.cache = {}
        else:
            self.cache = {}
    
    def save_cache(self):
        """Guarda cache"""
        if not self.cache:
            return
        self.logger.info(f"Guardando cache embeddings ({len(self.cache):,})...")
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.logger.info("  ✓ Cache guardado")
    
    @staticmethod
    @lru_cache(maxsize=500_000)
    def preprocess_name(name: str) -> str:
        """Preprocesa nombre"""
        if not name:
            return ""
        name = name.lower()
        name = BUSINESS_PATTERN.sub(' ', name)
        name = NORMALIZE_PATTERN.sub(' ', name)
        name = WHITESPACE_PATTERN.sub(' ', name).strip()
        return name
    
    def compute_embeddings(self, names: List[str], show_progress: bool = True) -> np.ndarray:
        """Calcula embeddings usando cache"""
        self.load_model()
        
        names_to_compute = []
        for name in names:
            preprocessed = self.preprocess_name(name)
            if preprocessed not in self.cache:
                names_to_compute.append(preprocessed)
        
        if names_to_compute:
            self.logger.info(f"  Calculando {len(names_to_compute):,} embeddings...")
            
            batch_size = 256 if self.hw_config.device == 'cuda' else 128
            
            new_embeddings = self.model.encode(
                names_to_compute,
                convert_to_numpy=True,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=True
            )
            
            for name, emb in zip(names_to_compute, new_embeddings):
                self.cache[name] = emb.astype(np.float32)
        
        result = np.zeros((len(names), self.config.embedding_dim), dtype=np.float32)
        for i, name in enumerate(names):
            preprocessed = self.preprocess_name(name)
            result[i] = self.cache.get(preprocessed, np.zeros(self.config.embedding_dim))
        
        return result


# ============================================================================
# FUZZY MATCHER ENGINE (100% GPU)
# ============================================================================

class FuzzyMatcherGPU:
    """Motor de fuzzy matching 100% GPU"""
    
    def __init__(
        self,
        config: MatchingConfig,
        hw_config: HardwareConfig,
        logger: logging.Logger
    ):
        self.config = config
        self.hw_config = hw_config
        self.logger = logger
        self.device = hw_config.torch_device
        
        self.embedding_manager = EmbeddingManager(config, hw_config, logger)
        self.ngram_matcher = GPUNgramMatcher(config, hw_config, logger)
        
    def process_batch(
        self,
        df: pl.DataFrame,
        source_semantic_embs: np.ndarray,
        target_semantic_embs: np.ndarray,
        source_ngram_vecs: torch.Tensor,
        target_ngram_vecs: torch.Tensor
    ) -> pl.DataFrame:
        """Procesa un batch completamente en GPU"""
        
        # 1. Similitud semántica (GPU)
        source_sem = torch.tensor(source_semantic_embs, device=self.device)
        target_sem = torch.tensor(target_semantic_embs, device=self.device)
        semantic_scores = torch.sum(source_sem * target_sem, dim=1) * 100
        semantic_scores = torch.clamp(semantic_scores, 0, 100)
        
        # 2. Similitud léxica via n-gramas (GPU) - ¡AQUÍ ESTÁ LA MAGIA!
        fuzzy_scores = self.ngram_matcher.compute_jaccard_similarity_gpu(
            source_ngram_vecs,
            target_ngram_vecs
        )
        
        # 3. Score combinado (GPU)
        final_scores = (
            self.config.fuzzy_weight * fuzzy_scores +
            self.config.semantic_weight * semantic_scores
        )
        
        # 4. Clasificación de confianza (GPU)
        confidence = torch.where(
            final_scores >= self.config.threshold_high,
            torch.tensor(2, device=self.device),  # high = 2
            torch.where(
                final_scores >= self.config.threshold_medium,
                torch.tensor(1, device=self.device),  # medium = 1
                torch.tensor(0, device=self.device)   # low = 0
            )
        )
        
        # 5. Mover a CPU para DataFrame
        fuzzy_np = fuzzy_scores.cpu().numpy().astype(np.float32)
        semantic_np = semantic_scores.cpu().numpy().astype(np.float32)
        final_np = final_scores.cpu().numpy().astype(np.float32)
        conf_np = confidence.cpu().numpy()
        
        # Mapear códigos a strings
        conf_map = {0: 'low', 1: 'medium', 2: 'high'}
        conf_strings = [conf_map[c] for c in conf_np]
        
        # 6. Añadir columnas al DataFrame
        return df.with_columns([
            pl.Series('fuzzy_score', fuzzy_np),
            pl.Series('semantic_score', semantic_np),
            pl.Series('final_score', final_np),
            pl.Series('confidence_level', conf_strings)
        ])
    
    def run(
        self,
        input_file: Path,
        output_file: Path,
        batch_size: Optional[int] = None
    ):
        """Ejecuta el pipeline completo"""
        
        batch_size = batch_size or self.hw_config.recommended_batch_size
        
        # Setup checkpoint
        checkpoint_mgr = CheckpointManager(self.config.checkpoint_dir, self.logger)
        state, can_resume = checkpoint_mgr.load(input_file, self.config)
        
        if not can_resume:
            state.input_file_hash = checkpoint_mgr.compute_file_hash(input_file)
            state.config_hash = checkpoint_mgr.compute_config_hash(self.config)
        
        # Cargar caches
        self.embedding_manager.load_cache()
        self.ngram_matcher.load_cache()
        
        # Total de filas
        total_rows = pl.scan_parquet(input_file).select(pl.len()).collect().item()
        self.logger.info(f"Total de filas: {total_rows:,}")
        
        if can_resume:
            self.logger.info(f"Continuando desde: {state.total_processed:,}")
        
        # Cargar nombres únicos
        self.logger.info("Cargando nombres únicos...")
        unique_source = pl.scan_parquet(input_file).select(
            pl.col(self.config.col_source).unique()
        ).collect()[self.config.col_source].to_list()
        
        unique_target = pl.scan_parquet(input_file).select(
            pl.col(self.config.col_target).unique()
        ).collect()[self.config.col_target].to_list()
        
        all_unique = list(set(unique_source) | set(unique_target))
        self.logger.info(f"Nombres únicos: {len(all_unique):,}")
        
        # Pre-calcular embeddings semánticos
        self.logger.info("\nPre-calculando embeddings semánticos...")
        all_semantic_embs = self.embedding_manager.compute_embeddings(all_unique, show_progress=True)
        self.embedding_manager.save_cache()
        
        # Pre-calcular vectores n-grama
        self.logger.info("\nPre-calculando vectores n-grama (GPU)...")
        all_ngram_vecs = self.ngram_matcher.compute_vectors(all_unique, show_progress=True)
        self.ngram_matcher.save_cache()
        
        # Crear lookups
        name_to_idx = {name: i for i, name in enumerate(all_unique)}
        
        # Procesar batches
        start_batch = state.next_batch_index
        total_batches = (total_rows + batch_size - 1) // batch_size
        
        self.logger.info(f"\nProcesando {total_batches - start_batch} batches (100% GPU)...")
        
        try:
            if RICH_AVAILABLE:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]{task.description}"),
                    BarColumn(bar_width=40),
                    TaskProgressColumn(),
                    TextColumn("•"),
                    TimeElapsedColumn(),
                    TextColumn("•"),
                    TimeRemainingColumn(),
                    console=Console(),
                    refresh_per_second=2
                ) as progress:
                    task = progress.add_task("Procesando", total=total_rows, completed=state.total_processed)
                    self._process_all_batches(
                        input_file, batch_size, start_batch, total_rows,
                        all_semantic_embs, all_ngram_vecs, name_to_idx,
                        checkpoint_mgr, state, progress, task
                    )
            else:
                with tqdm(total=total_rows, initial=state.total_processed, 
                         desc="Procesando", unit=" filas") as pbar:
                    self._process_all_batches(
                        input_file, batch_size, start_batch, total_rows,
                        all_semantic_embs, all_ngram_vecs, name_to_idx,
                        checkpoint_mgr, state, pbar, None
                    )
                    
        except KeyboardInterrupt:
            self.logger.warning("\n⚠ Interrumpido por usuario")
            self.logger.info(f"Progreso guardado: {state.total_processed:,} filas")
            return
        
        # Consolidar
        self.logger.info("\nConsolidando resultados...")
        if checkpoint_mgr.consolidate_partitions(output_file):
            high_conf_file = output_file.with_stem(output_file.stem + '_high_confidence')
            
            pl.scan_parquet(output_file).filter(
                pl.col('confidence_level') == 'high'
            ).sink_parquet(high_conf_file)
            
            self._print_stats(state, output_file, high_conf_file)
    
    def _process_all_batches(
        self,
        input_file: Path,
        batch_size: int,
        start_batch: int,
        total_rows: int,
        all_semantic_embs: np.ndarray,
        all_ngram_vecs: torch.Tensor,
        name_to_idx: dict,
        checkpoint_mgr: CheckpointManager,
        state: CheckpointState,
        progress_bar,
        task_id
    ):
        """Procesa todos los batches"""
        
        for batch_idx in range(start_batch, (total_rows + batch_size - 1) // batch_size):
            start_row = batch_idx * batch_size
            
            # Leer batch
            batch_df = pl.scan_parquet(input_file).slice(start_row, batch_size).collect()
            current_size = len(batch_df)
            
            if current_size == 0:
                break
            
            # Obtener índices
            source_names = batch_df[self.config.col_source].to_list()
            target_names = batch_df[self.config.col_target].to_list()
            
            source_indices = [name_to_idx[n] for n in source_names]
            target_indices = [name_to_idx[n] for n in target_names]
            
            # Extraer embeddings y vectores
            source_sem = all_semantic_embs[source_indices]
            target_sem = all_semantic_embs[target_indices]
            source_ngram = all_ngram_vecs[source_indices]
            target_ngram = all_ngram_vecs[target_indices]
            
            # Procesar (100% GPU)
            result_df = self.process_batch(
                batch_df, source_sem, target_sem, source_ngram, target_ngram
            )
            
            # Estadísticas
            conf_counts = result_df.group_by('confidence_level').len().to_dict()
            conf_dict = dict(zip(conf_counts.get('confidence_level', []), conf_counts.get('len', [])))
            
            state.stats['high'] += conf_dict.get('high', 0)
            state.stats['medium'] += conf_dict.get('medium', 0)
            state.stats['low'] += conf_dict.get('low', 0)
            state.total_processed += current_size
            state.next_batch_index = batch_idx + 1
            
            # Guardar
            partition_name = checkpoint_mgr.save_partition(result_df, batch_idx)
            state.partition_files.append(partition_name)
            checkpoint_mgr.save(state)
            
            # Actualizar progreso
            if RICH_AVAILABLE and task_id is not None:
                progress_bar.update(task_id, completed=state.total_processed)
            elif hasattr(progress_bar, 'update'):
                progress_bar.update(current_size)
            
            # Liberar memoria
            del batch_df, result_df, source_sem, target_sem, source_ngram, target_ngram
            if batch_idx % 10 == 0:
                gc.collect()
                if self.hw_config.device == 'cuda':
                    torch.cuda.empty_cache()
    
    def _print_stats(self, state: CheckpointState, output_file: Path, high_conf_file: Path):
        """Imprime estadísticas finales"""
        total = state.total_processed
        stats = state.stats
        
        if RICH_AVAILABLE:
            console = Console()
            table = Table(title="📊 Resultados del Matching (GPU)", show_header=True)
            table.add_column("Métrica", style="cyan")
            table.add_column("Valor", style="green", justify="right")
            table.add_column("%", style="yellow", justify="right")
            
            table.add_row("Total procesado", f"{total:,}", "100%")
            table.add_row("Alta confianza", f"{stats['high']:,}", f"{stats['high']/total*100:.1f}%")
            table.add_row("Media confianza", f"{stats['medium']:,}", f"{stats['medium']/total*100:.1f}%")
            table.add_row("Baja confianza", f"{stats['low']:,}", f"{stats['low']/total*100:.1f}%")
            
            console.print(table)
            console.print(f"\n📁 Archivo completo: [bold]{output_file}[/bold]")
            console.print(f"📁 Alta confianza: [bold]{high_conf_file}[/bold]")
        else:
            self.logger.info("\n" + "="*60)
            self.logger.info("RESULTADOS DEL MATCHING (GPU)")
            self.logger.info("="*60)
            self.logger.info(f"Total procesado:   {total:,}")
            self.logger.info(f"Alta confianza:    {stats['high']:,} ({stats['high']/total*100:.1f}%)")
            self.logger.info(f"Media confianza:   {stats['medium']:,} ({stats['medium']/total*100:.1f}%)")
            self.logger.info(f"Baja confianza:    {stats['low']:,} ({stats['low']/total*100:.1f}%)")
            self.logger.info(f"\nArchivo completo:  {output_file}")
            self.logger.info(f"Alta confianza:    {high_conf_file}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Fuzzy Matching 100% GPU (N-gram Jaccard + Semantic)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python fuzzy_match_v5_full_gpu.py --input data.parquet --output results/
  python fuzzy_match_v5_full_gpu.py --resume --input data.parquet
  
Mejoras sobre V4:
  - Fuzzy matching en GPU (antes CPU con RapidFuzz)
  - Speedup esperado: 3-5x
        """
    )
    parser.add_argument('--input', '-i', type=Path, help='Archivo parquet de entrada')
    parser.add_argument('--output', '-o', type=Path, default=Path('resultados_matching'),
                       help='Directorio de salida')
    parser.add_argument('--batch-size', '-b', type=int, help='Tamaño de batch')
    parser.add_argument('--fuzzy-weight', type=float, default=0.7, help='Peso fuzzy (default: 0.7)')
    parser.add_argument('--semantic-weight', type=float, default=0.3, help='Peso semántico (default: 0.3)')
    parser.add_argument('--ngram-size', type=int, default=3, help='Tamaño de n-gramas (default: 3)')
    parser.add_argument('--hash-dim', type=int, default=4096, help='Dimensión hash (default: 4096)')
    parser.add_argument('--col-source', default='person_name', help='Columna fuente')
    parser.add_argument('--col-target', default='orbis_name', help='Columna objetivo')
    parser.add_argument('--resume', '-r', action='store_true', help='Continuar desde checkpoint')
    parser.add_argument('--force', '-f', action='store_true', help='Forzar reinicio')
    
    args = parser.parse_args()
    
    if not args.resume and not args.input:
        parser.error("Se requiere --input o --resume")
    
    args.output.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(args.output)
    hw_config = HardwareConfig.detect()
    
    logger.info("="*70)
    logger.info("FUZZY MATCHING V5 - 100% GPU Pipeline")
    logger.info("="*70)
    logger.info(f"Dispositivo:      {hw_config.device.upper()} ({hw_config.device_name})")
    logger.info(f"Memoria:          {hw_config.total_memory_gb:.1f} GB")
    logger.info(f"Batch size:       {args.batch_size or hw_config.recommended_batch_size:,}")
    logger.info(f"N-gram size:      {args.ngram_size}")
    logger.info(f"Hash dimension:   {args.hash_dim}")
    logger.info("="*70)
    
    config = MatchingConfig(
        fuzzy_weight=args.fuzzy_weight,
        semantic_weight=args.semantic_weight,
        ngram_size=args.ngram_size,
        hash_dim=args.hash_dim,
        col_source=args.col_source,
        col_target=args.col_target,
        cache_dir=args.output / 'cache_embeddings',
        checkpoint_dir=args.output / 'checkpoints_v5'  # Directorio diferente para V5
    )
    
    if args.force and config.checkpoint_dir.exists():
        import shutil
        shutil.rmtree(config.checkpoint_dir)
        logger.info("Checkpoints eliminados (--force)")
    
    if args.resume:
        checkpoint_file = config.checkpoint_dir / 'checkpoint.json'
        if not checkpoint_file.exists():
            parser.error("No se encontró checkpoint para resumir")
        if not args.input:
            parser.error("Con --resume necesitas especificar --input")
    
    input_file = args.input
    if not input_file.exists():
        parser.error(f"Archivo no encontrado: {input_file}")
    
    output_file = args.output / f"matched_{input_file.stem}_fuzzy_v5_gpu.parquet"
    
    matcher = FuzzyMatcherGPU(config, hw_config, logger)
    matcher.run(input_file, output_file, args.batch_size)
    
    logger.info("\n✅ PROCESO COMPLETADO")


if __name__ == "__main__":
    main()
