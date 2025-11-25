"""
fuzzy_match_v4_universal.py - GPU-Accelerated Hybrid Matching (CUDA + Apple Silicon)

MEJORAS SOBRE V3:
- Compatibilidad universal: CUDA (Linux/Windows) + MPS (Mac M4) + CPU fallback
- Barra de progreso robusta con tqdm + logging dual (archivo + consola)
- Checkpointing atómico con validación y auto-recuperación
- Detección automática de hardware y configuración óptima
- Mejor gestión de memoria (evita OOM)
- Procesamiento por lotes adaptativo según VRAM/RAM disponible

REQUISITOS:
pip install polars rapidfuzz sentence-transformers faiss-cpu torch tqdm rich

Para CUDA (Linux RTX 3060 Ti):
pip install faiss-gpu torch --index-url https://download.pytorch.org/whl/cu118

Para Mac M4:
pip install faiss-cpu torch  # MPS se activa automáticamente

USO:
python fuzzy_match_v4_universal.py --input data.parquet --output results/
python fuzzy_match_v4_universal.py --resume  # Continuar desde último checkpoint
"""

import polars as pl
import numpy as np
import torch
import faiss
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
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache

from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
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
    faiss_use_gpu: bool
    
    @classmethod
    def detect(cls) -> 'HardwareConfig':
        """Detecta automáticamente el hardware disponible"""
        num_cores = max(1, (hasattr(os, 'cpu_count') and os.cpu_count()) or 4)
        
        # Prioridad: CUDA > MPS > CPU
        if torch.cuda.is_available():
            device = 'cuda'
            props = torch.cuda.get_device_properties(0)
            device_name = props.name
            total_mem = props.total_memory / (1024**3)
            # RTX 3060 Ti tiene 8GB, ser conservador
            batch_size = int((total_mem / 8) * 30_000)
            faiss_gpu = faiss.get_num_gpus() > 0
            
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
            device_name = f"Apple Silicon ({platform.processor()})"
            # Mac M4 Pro tiene memoria unificada, estimar ~16-32GB disponibles
            import subprocess
            try:
                result = subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True)
                total_mem = int(result.stdout.strip()) / (1024**3)
            except Exception:
                total_mem = 16.0  # Default conservador
            # MPS es eficiente pero más lento que CUDA, batches moderados
            batch_size = int((total_mem / 16) * 40_000)
            faiss_gpu = False  # FAISS no soporta MPS
            
        else:
            device = 'cpu'
            device_name = platform.processor() or 'CPU'
            import psutil
            total_mem = psutil.virtual_memory().total / (1024**3) if 'psutil' in sys.modules else 8.0
            batch_size = 10_000
            faiss_gpu = False
        
        return cls(
            device=device,
            device_name=device_name,
            total_memory_gb=round(total_mem, 2),
            recommended_batch_size=max(5_000, min(batch_size, 100_000)),
            num_cpu_cores=num_cores,
            faiss_use_gpu=faiss_gpu
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
# LOGGING CONFIGURACIÓN
# ============================================================================

def setup_logging(output_dir: Path) -> logging.Logger:
    """Configura logging dual: archivo + consola con colores"""
    logger = logging.getLogger('FuzzyMatch')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    # Formato
    fmt = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    
    # Handler archivo
    log_file = output_dir / f"fuzzy_match_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    
    # Handler consola
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    
    return logger


# ============================================================================
# CHECKPOINT MANAGER (Robusto y Atómico)
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
        """Calcula hash MD5 del archivo para detectar cambios"""
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            # Leer solo primeros y últimos 10MB para velocidad
            hasher.update(f.read(10 * 1024 * 1024))
            f.seek(-min(10 * 1024 * 1024, f.seek(0, 2)), 2)
            hasher.update(f.read())
        return hasher.hexdigest()[:16]
    
    def compute_config_hash(self, config: MatchingConfig) -> str:
        """Hash de configuración para detectar cambios de parámetros"""
        config_str = f"{config.fuzzy_weight}_{config.semantic_weight}_{config.threshold_high}_{config.embedding_model}"
        return hashlib.md5(config_str.encode()).hexdigest()[:16]
    
    def load(self, input_file: Path, config: MatchingConfig) -> tuple[CheckpointState, bool]:
        """
        Carga checkpoint si existe y es válido.
        Retorna (state, can_resume)
        """
        if not self.checkpoint_file.exists():
            self.logger.info("No se encontró checkpoint previo, iniciando desde cero")
            return CheckpointState(), False
        
        try:
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
            state = CheckpointState.from_dict(data)
            
            # Validar que el archivo de entrada no cambió
            current_hash = self.compute_file_hash(input_file)
            if state.input_file_hash != current_hash:
                self.logger.warning(f"El archivo de entrada cambió (hash: {current_hash} vs {state.input_file_hash})")
                self.logger.warning("Reiniciando proceso desde cero")
                return CheckpointState(), False
            
            # Validar configuración
            current_config_hash = self.compute_config_hash(config)
            if state.config_hash != current_config_hash:
                self.logger.warning("La configuración cambió, reiniciando proceso")
                return CheckpointState(), False
            
            # Validar que las particiones existen
            missing = [p for p in state.partition_files if not (self.partitions_dir / p).exists()]
            if missing:
                self.logger.warning(f"Faltan {len(missing)} archivos de partición, reiniciando")
                return CheckpointState(), False
            
            self.logger.info(f"✓ Checkpoint válido encontrado: {state.total_processed:,} filas procesadas")
            self.state = state
            return state, True
            
        except Exception as e:
            self.logger.error(f"Error cargando checkpoint: {e}")
            # Intentar backup
            if self.checkpoint_backup.exists():
                try:
                    self.checkpoint_file.write_bytes(self.checkpoint_backup.read_bytes())
                    return self.load(input_file, config)
                except Exception:
                    pass
            return CheckpointState(), False
    
    def save(self, state: CheckpointState):
        """Guarda checkpoint de forma atómica"""
        state.last_update = datetime.now().isoformat()
        
        # Escribir a archivo temporal primero
        temp_file = self.checkpoint_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(state.to_dict(), f, indent=2)
        
        # Backup del anterior si existe
        if self.checkpoint_file.exists():
            self.checkpoint_file.rename(self.checkpoint_backup)
        
        # Mover el nuevo (operación atómica en la mayoría de sistemas)
        temp_file.rename(self.checkpoint_file)
        self.state = state
    
    def save_partition(self, df: pl.DataFrame, batch_index: int) -> str:
        """Guarda partición y retorna nombre del archivo"""
        filename = f"part_{batch_index:010d}.parquet"
        filepath = self.partitions_dir / filename
        df.write_parquet(filepath, compression='zstd', compression_level=3)
        return filename
    
    def consolidate_partitions(self, output_file: Path) -> bool:
        """Consolida todas las particiones en un archivo final"""
        try:
            partition_files = sorted(self.partitions_dir.glob('part_*.parquet'))
            if not partition_files:
                self.logger.error("No hay particiones para consolidar")
                return False
            
            self.logger.info(f"Consolidando {len(partition_files)} particiones...")
            
            # Usar scan para memoria eficiente
            lazy_frames = [pl.scan_parquet(p) for p in partition_files]
            combined = pl.concat(lazy_frames)
            
            # Escribir con streaming
            combined.sink_parquet(
                output_file,
                compression='zstd',
                compression_level=6
            )
            
            self.logger.info(f"✓ Archivo consolidado: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error consolidando: {e}")
            return False


# ============================================================================
# GPU EMBEDDING MANAGER (Universal)
# ============================================================================

class EmbeddingManager:
    """Gestiona embeddings con cache persistente y soporte multi-GPU"""
    
    def __init__(self, config: MatchingConfig, hw_config: HardwareConfig, logger: logging.Logger):
        self.config = config
        self.hw_config = hw_config
        self.logger = logger
        self.model: Optional[SentenceTransformer] = None
        self.cache: dict[str, np.ndarray] = {}
        self.cache_file = config.cache_dir / f"embeddings_{config.embedding_model.replace('/', '_')}.pkl"
        
    def _get_torch_device(self) -> torch.device:
        """Obtiene el dispositivo torch apropiado"""
        if self.hw_config.device == 'cuda':
            return torch.device('cuda:0')
        elif self.hw_config.device == 'mps':
            return torch.device('mps')
        return torch.device('cpu')
    
    def load_model(self):
        """Carga modelo optimizado para el hardware detectado"""
        if self.model is not None:
            return
            
        self.logger.info(f"Cargando modelo '{self.config.embedding_model}' en {self.hw_config.device}...")
        
        device = self._get_torch_device()
        self.model = SentenceTransformer(self.config.embedding_model, device=device)
        
        # Optimizaciones específicas por plataforma
        if self.hw_config.device == 'cuda':
            # Half precision para CUDA (2x throughput, menos VRAM)
            self.model.half()
            self.logger.info("  → Activado FP16 para CUDA")
        elif self.hw_config.device == 'mps':
            # MPS funciona mejor con FP32 actualmente
            pass
            
        self.logger.info(f"  ✓ Modelo cargado en {device}")
    
    def load_cache(self):
        """Carga cache de embeddings desde disco"""
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
        """Persiste cache a disco"""
        if not self.cache:
            return
        self.logger.info(f"Guardando cache ({len(self.cache):,} embeddings)...")
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.logger.info("  ✓ Cache guardado")
    
    @staticmethod
    @lru_cache(maxsize=100_000)
    def preprocess_name(name: str) -> str:
        """Preprocesa nombre para embedding (cacheado en memoria)"""
        if not name:
            return ""
        name = name.lower()
        name = BUSINESS_PATTERN.sub(' ', name)
        name = NORMALIZE_PATTERN.sub(' ', name)
        name = WHITESPACE_PATTERN.sub(' ', name).strip()
        return name
    
    def compute_embeddings(self, names: list[str], show_progress: bool = True) -> np.ndarray:
        """
        Calcula embeddings para lista de nombres.
        Usa cache para nombres ya calculados.
        """
        self.load_model()
        
        # Separar nombres en cache vs nuevos
        names_to_compute = []
        indices_to_compute = []
        
        for i, name in enumerate(names):
            preprocessed = self.preprocess_name(name)
            if preprocessed not in self.cache:
                names_to_compute.append(preprocessed)
                indices_to_compute.append(i)
        
        # Calcular embeddings para nombres nuevos
        if names_to_compute:
            self.logger.info(f"  Calculando {len(names_to_compute):,} embeddings nuevos...")
            
            # Batch size adaptativo
            batch_size = 256 if self.hw_config.device == 'cuda' else 128
            
            new_embeddings = self.model.encode(
                names_to_compute,
                convert_to_numpy=True,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=True  # Pre-normalizar para cosine sim
            )
            
            # Guardar en cache
            for name, emb in zip(names_to_compute, new_embeddings):
                self.cache[name] = emb.astype(np.float32)
        
        # Construir array de salida
        result = np.zeros((len(names), self.config.embedding_dim), dtype=np.float32)
        for i, name in enumerate(names):
            preprocessed = self.preprocess_name(name)
            result[i] = self.cache.get(preprocessed, np.zeros(self.config.embedding_dim))
        
        return result


# ============================================================================
# FUZZY MATCHER ENGINE
# ============================================================================

def compute_fuzzy_scores_batch(args: tuple) -> np.ndarray:
    """Calcula scores fuzzy para un batch (para multiprocessing)"""
    source_names, target_names, start_idx, end_idx = args
    
    scores = np.zeros(end_idx - start_idx, dtype=np.float32)
    for i, (s, t) in enumerate(zip(source_names[start_idx:end_idx], target_names[start_idx:end_idx])):
        # Preprocesar
        s_clean = WHITESPACE_PATTERN.sub(' ', BUSINESS_PATTERN.sub(' ', s.lower())).strip()
        t_clean = WHITESPACE_PATTERN.sub(' ', BUSINESS_PATTERN.sub(' ', t.lower())).strip()
        scores[i] = fuzz.token_sort_ratio(s_clean, t_clean)
    
    return scores


class FuzzyMatcher:
    """Motor principal de fuzzy matching híbrido"""
    
    def __init__(
        self,
        config: MatchingConfig,
        hw_config: HardwareConfig,
        logger: logging.Logger
    ):
        self.config = config
        self.hw_config = hw_config
        self.logger = logger
        self.embedding_manager = EmbeddingManager(config, hw_config, logger)
        
    def process_batch(
        self,
        df: pl.DataFrame,
        source_embeddings: np.ndarray,
        target_embeddings: np.ndarray
    ) -> pl.DataFrame:
        """Procesa un batch de datos"""
        
        source_names = df[self.config.col_source].to_list()
        target_names = df[self.config.col_target].to_list()
        n = len(source_names)
        
        # 1. Similitud semántica (ya normalizada)
        # Dot product de embeddings normalizados = cosine similarity
        semantic_scores = np.sum(source_embeddings * target_embeddings, axis=1) * 100
        semantic_scores = np.clip(semantic_scores, 0, 100)
        
        # 2. Similitud léxica (paralelo en CPU)
        num_workers = max(1, self.hw_config.num_cpu_cores // 2)
        chunk_size = max(100, n // num_workers)
        
        chunks = [
            (source_names, target_names, i, min(i + chunk_size, n))
            for i in range(0, n, chunk_size)
        ]
        
        # Usar ProcessPoolExecutor para aprovechar múltiples cores
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            fuzzy_chunks = list(executor.map(compute_fuzzy_scores_batch, chunks))
        
        fuzzy_scores = np.concatenate(fuzzy_chunks)
        
        # 3. Score combinado
        final_scores = (
            self.config.fuzzy_weight * fuzzy_scores +
            self.config.semantic_weight * semantic_scores
        )
        
        # 4. Clasificación de confianza
        confidence = np.where(
            final_scores >= self.config.threshold_high, 'high',
            np.where(final_scores >= self.config.threshold_medium, 'medium', 'low')
        )
        
        # 5. Añadir columnas al DataFrame
        return df.with_columns([
            pl.Series('fuzzy_score', fuzzy_scores),
            pl.Series('semantic_score', semantic_scores),
            pl.Series('final_score', final_scores),
            pl.Series('confidence_level', confidence)
        ])
    
    def run(
        self,
        input_file: Path,
        output_file: Path,
        batch_size: Optional[int] = None
    ):
        """Ejecuta el pipeline completo de matching"""
        
        batch_size = batch_size or self.hw_config.recommended_batch_size
        
        # Setup checkpoint manager
        checkpoint_mgr = CheckpointManager(self.config.checkpoint_dir, self.logger)
        
        # Intentar cargar checkpoint
        state, can_resume = checkpoint_mgr.load(input_file, self.config)
        if not can_resume:
            state.input_file_hash = checkpoint_mgr.compute_file_hash(input_file)
            state.config_hash = checkpoint_mgr.compute_config_hash(self.config)
        
        # Cargar cache de embeddings
        self.embedding_manager.load_cache()
        
        # Obtener total de filas
        total_rows = pl.scan_parquet(input_file).select(pl.len()).collect().item()
        self.logger.info(f"Total de filas a procesar: {total_rows:,}")
        
        if can_resume:
            self.logger.info(f"Continuando desde fila {state.total_processed:,}")
        
        # Calcular número de batches
        start_batch = state.next_batch_index
        total_batches = (total_rows + batch_size - 1) // batch_size
        
        # Pre-cargar nombres únicos para embeddings
        self.logger.info("Cargando nombres únicos...")
        unique_source = pl.scan_parquet(input_file).select(
            pl.col(self.config.col_source).unique()
        ).collect()[self.config.col_source].to_list()
        
        unique_target = pl.scan_parquet(input_file).select(
            pl.col(self.config.col_target).unique()
        ).collect()[self.config.col_target].to_list()
        
        all_unique = list(set(unique_source) | set(unique_target))
        self.logger.info(f"Nombres únicos: {len(all_unique):,}")
        
        # Pre-calcular todos los embeddings
        self.logger.info("Pre-calculando embeddings globales...")
        all_embeddings = self.embedding_manager.compute_embeddings(all_unique, show_progress=True)
        
        # Crear índice de lookup
        name_to_idx = {name: i for i, name in enumerate(all_unique)}
        
        # Guardar cache
        self.embedding_manager.save_cache()
        
        # Configurar barra de progreso
        if RICH_AVAILABLE:
            progress = Progress(
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
            )
        else:
            progress = None
        
        # Procesar batches
        self.logger.info(f"\nProcesando {total_batches - start_batch} batches restantes...")
        
        try:
            if RICH_AVAILABLE:
                with progress:
                    task = progress.add_task(
                        "Procesando",
                        total=total_rows,
                        completed=state.total_processed
                    )
                    self._process_batches(
                        input_file, batch_size, start_batch, total_rows,
                        all_embeddings, name_to_idx, checkpoint_mgr, state,
                        progress, task
                    )
            else:
                # Fallback a tqdm
                with tqdm(total=total_rows, initial=state.total_processed, 
                         desc="Procesando", unit=" filas") as pbar:
                    self._process_batches(
                        input_file, batch_size, start_batch, total_rows,
                        all_embeddings, name_to_idx, checkpoint_mgr, state,
                        pbar, None
                    )
                    
        except KeyboardInterrupt:
            self.logger.warning("\n⚠ Proceso interrumpido por usuario")
            self.logger.info(f"Progreso guardado: {state.total_processed:,} filas")
            self.logger.info("Ejecuta de nuevo para continuar desde el último checkpoint")
            return
        
        # Consolidar resultados
        self.logger.info("\nConsolidando resultados finales...")
        if checkpoint_mgr.consolidate_partitions(output_file):
            # Generar archivo de alta confianza
            high_conf_file = output_file.with_stem(output_file.stem + '_high_confidence')
            self.logger.info(f"Generando archivo de alta confianza...")
            
            pl.scan_parquet(output_file).filter(
                pl.col('confidence_level') == 'high'
            ).sink_parquet(high_conf_file)
            
            # Estadísticas finales
            self._print_final_stats(state, output_file, high_conf_file)
    
    def _process_batches(
        self,
        input_file: Path,
        batch_size: int,
        start_batch: int,
        total_rows: int,
        all_embeddings: np.ndarray,
        name_to_idx: dict,
        checkpoint_mgr: CheckpointManager,
        state: CheckpointState,
        progress_bar,
        task_id
    ):
        """Procesa todos los batches con checkpointing"""
        
        for batch_idx in range(start_batch, (total_rows + batch_size - 1) // batch_size):
            start_row = batch_idx * batch_size
            
            # Leer batch
            batch_df = pl.scan_parquet(input_file).slice(start_row, batch_size).collect()
            current_size = len(batch_df)
            
            if current_size == 0:
                break
            
            # Obtener embeddings para este batch
            source_names = batch_df[self.config.col_source].to_list()
            target_names = batch_df[self.config.col_target].to_list()
            
            source_indices = [name_to_idx[n] for n in source_names]
            target_indices = [name_to_idx[n] for n in target_names]
            
            source_embs = all_embeddings[source_indices]
            target_embs = all_embeddings[target_indices]
            
            # Procesar batch
            result_df = self.process_batch(batch_df, source_embs, target_embs)
            
            # Actualizar estadísticas
            conf_counts = result_df.group_by('confidence_level').len().to_dict()
            conf_dict = dict(zip(conf_counts.get('confidence_level', []), conf_counts.get('len', [])))
            
            state.stats['high'] += conf_dict.get('high', 0)
            state.stats['medium'] += conf_dict.get('medium', 0)
            state.stats['low'] += conf_dict.get('low', 0)
            state.total_processed += current_size
            state.next_batch_index = batch_idx + 1
            
            # Guardar partición
            partition_name = checkpoint_mgr.save_partition(result_df, batch_idx)
            state.partition_files.append(partition_name)
            
            # Guardar checkpoint
            checkpoint_mgr.save(state)
            
            # Actualizar progreso
            if RICH_AVAILABLE and task_id is not None:
                progress_bar.update(task_id, completed=state.total_processed)
            elif hasattr(progress_bar, 'update'):
                progress_bar.update(current_size)
            
            # Liberar memoria
            del batch_df, result_df, source_embs, target_embs
            if batch_idx % 10 == 0:
                gc.collect()
                if self.hw_config.device == 'cuda':
                    torch.cuda.empty_cache()
    
    def _print_final_stats(self, state: CheckpointState, output_file: Path, high_conf_file: Path):
        """Imprime estadísticas finales"""
        
        total = state.total_processed
        stats = state.stats
        
        if RICH_AVAILABLE:
            console = Console()
            table = Table(title="📊 Resultados del Matching", show_header=True)
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
            self.logger.info("RESULTADOS DEL MATCHING")
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
        description='Fuzzy Matching GPU-Accelerated (CUDA + Apple Silicon)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python fuzzy_match_v4_universal.py --input data.parquet --output results/
  python fuzzy_match_v4_universal.py --input data.parquet --batch-size 20000
  python fuzzy_match_v4_universal.py --resume  # Continuar desde checkpoint
        """
    )
    parser.add_argument('--input', '-i', type=Path, help='Archivo parquet de entrada')
    parser.add_argument('--output', '-o', type=Path, default=Path('resultados_matching'),
                       help='Directorio de salida')
    parser.add_argument('--batch-size', '-b', type=int, help='Tamaño de batch (auto-detectado si no se especifica)')
    parser.add_argument('--fuzzy-weight', type=float, default=0.7, help='Peso del score fuzzy (default: 0.7)')
    parser.add_argument('--semantic-weight', type=float, default=0.3, help='Peso del score semántico (default: 0.3)')
    parser.add_argument('--col-source', default='person_name', help='Columna de nombres fuente')
    parser.add_argument('--col-target', default='orbis_name', help='Columna de nombres objetivo')
    parser.add_argument('--resume', '-r', action='store_true', help='Continuar desde último checkpoint')
    parser.add_argument('--force', '-f', action='store_true', help='Forzar reinicio ignorando checkpoints')
    
    args = parser.parse_args()
    
    # Validar argumentos
    if not args.resume and not args.input:
        parser.error("Se requiere --input o --resume")
    
    # Crear directorio de salida
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.output)
    
    # Detectar hardware
    import os  # Importar aquí para que esté disponible en HardwareConfig
    hw_config = HardwareConfig.detect()
    
    logger.info("="*70)
    logger.info("FUZZY MATCHING V4 - Universal GPU Pipeline")
    logger.info("="*70)
    logger.info(f"Dispositivo:      {hw_config.device.upper()} ({hw_config.device_name})")
    logger.info(f"Memoria:          {hw_config.total_memory_gb:.1f} GB")
    logger.info(f"Batch size:       {args.batch_size or hw_config.recommended_batch_size:,}")
    logger.info(f"CPU cores:        {hw_config.num_cpu_cores}")
    logger.info(f"FAISS GPU:        {'Sí' if hw_config.faiss_use_gpu else 'No'}")
    logger.info("="*70)
    
    # Configuración
    config = MatchingConfig(
        fuzzy_weight=args.fuzzy_weight,
        semantic_weight=args.semantic_weight,
        col_source=args.col_source,
        col_target=args.col_target,
        cache_dir=args.output / 'cache_embeddings',
        checkpoint_dir=args.output / 'checkpoints'
    )
    
    # Si --force, limpiar checkpoints
    if args.force and config.checkpoint_dir.exists():
        import shutil
        shutil.rmtree(config.checkpoint_dir)
        logger.info("Checkpoints eliminados (--force)")
    
    # Determinar archivo de entrada
    if args.resume:
        # Buscar en checkpoint
        checkpoint_file = config.checkpoint_dir / 'checkpoint.json'
        if checkpoint_file.exists():
            with open(checkpoint_file) as f:
                data = json.load(f)
            # El archivo de entrada debe especificarse igual
            if not args.input:
                parser.error("Con --resume necesitas especificar --input también")
        else:
            parser.error("No se encontró checkpoint para resumir")
    
    input_file = args.input
    if not input_file.exists():
        parser.error(f"Archivo no encontrado: {input_file}")
    
    output_file = args.output / f"matched_{input_file.stem}_fuzzy_v4.parquet"
    
    # Ejecutar
    matcher = FuzzyMatcher(config, hw_config, logger)
    matcher.run(input_file, output_file, args.batch_size)
    
    logger.info("\n✅ PROCESO COMPLETADO")


if __name__ == "__main__":
    main()
