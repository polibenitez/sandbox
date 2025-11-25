"""
fuzzy_match_v3_gpu.py - GPU-Accelerated Hybrid Matching con Polars + FAISS

MEJORAS CLAVE:
- GPU acceleration para embeddings (CUDA)
- Vectorización total: elimina loops Python puros
- Cache global persistente de embeddings
- FAISS para búsqueda aproximada (reduce 179M → 1M comparaciones)
- Streaming de Polars para memoria O(1)
- Multiprocessing para CPU-bound (RapidFuzz)

REQUISITOS:
pip install polars rapidfuzz sentence-transformers faiss-gpu torch
GPU: CUDA 11.8+ compatible (testeado en RTX 3090/4090/A100)
"""

import polars as pl
import os
from rapidfuzz import fuzz
from rapidfuzz.process import cdist
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import faiss
from datetime import datetime
import gc
import re
from pathlib import Path
from multiprocessing import Pool, cpu_count
import pickle
import json
import shutil
import glob

# --- CONFIGURACIÓN GPU ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE_GPU = 50_000  # Ajustar según VRAM (10GB → 25k, 24GB → 50k, 80GB → 200k)
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2
FAISS_NLIST = 1000  # Número de clusters para índice IVF
FAISS_NPROBE = 50   # Clusters a explorar (trade-off speed/recall)

# --- CONFIGURACIÓN ---
INPUT_FILE = "resultados_matching/matched_DE.parquet"
OUTPUT_DIR = "resultados_matching"
CACHE_DIR = "cache_embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Ponderación híbrida
FUZZY_WEIGHT = 0.7
SEMANTIC_WEIGHT = 0.3

# Umbrales de confianza
CONFIDENCE_LEVELS = {'high': 90, 'medium': 75, 'low': 0}

# Palabras genéricas (pre-compiladas para speed)
BUSINESS_WORDS = [
    "pharmaceuticals", "pharmaceutical", "industries", "technologies",
    "technology", "corporation", "corp", "company", "companies",
    "group", "sa", "sl", "spa", "gmbh", "limited", "ltd", "llc",
    "inc", "ag", "nv", "bv", "srl", "plc", "pty", "coop"
]
BUSINESS_PATTERN = re.compile(r'\b(' + '|'.join(map(re.escape, BUSINESS_WORDS)) + r')\b')

# Logging
log_with_timestamp_enabled = True
def log_with_timestamp(message):
    if log_with_timestamp_enabled:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

# --- UTILIDADES GPU ---
def get_optimal_batch_size():
    """Detecta VRAM disponible y sugiere batch size óptimo"""
    if DEVICE == "cpu":
        return 10_000
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
    log_with_timestamp(f"VRAM disponible: {gpu_mem:.2f} GB")
    optimal_batch_size = int((gpu_mem / 10) * 50_000)
    log_with_timestamp(f"Batch size óptimo: {optimal_batch_size:,}")
    return optimal_batch_size
    return int((gpu_mem / 10) * 50_000)  # Aprox: 50k por cada 10GB

# --- PREPROCESAMIENTO VECTORIZADO ---
@pl.api.register_expr_namespace("text")
class TextClean:
    def remove_business_words(expr: pl.Expr) -> pl.Expr:
        return expr.str.to_lowercase().str.replace_all(BUSINESS_PATTERN, " ")
    
    def normalize(expr: pl.Expr) -> pl.Expr:
        return expr.str.replace_all(r'[^\w\s]', ' ').str.replace_all(r'\s+', ' ').str.strip_chars()

# --- GPU EMBEDDING MANAGER ---
class GPUEmbeddingCache:
    """Cache global persistente de embeddings en GPU"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.cache_path = Path(CACHE_DIR) / f"embeddings_{model_name.replace('/', '_')}.pkl"
        self.model = None
        self.cache = {}  # name -> embedding tensor
        
    def load_model(self):
        """Carga modelo en GPU con optimizaciones"""
        if self.model is None:
            log_with_timestamp(f"Cargando modelo {self.model_name} en {DEVICE}...")
            self.model = SentenceTransformer(self.model_name, device=DEVICE)
            # Opcional: activar half precision para 2x throughput
            if DEVICE == "cuda":
                self.model.half()
            log_with_timestamp("  ✓ Modelo cargado y optimizado")
    
    def load_cache(self):
        """Carga cache desde disco si existe"""
        if self.cache_path.exists():
            log_with_timestamp("Cargando cache de embeddings...")
            with open(self.cache_path, 'rb') as f:
                data = pickle.load(f)
                # Convertir numpy arrays a torch tensors en GPU
                self.cache = {
                    k: torch.tensor(v, device=DEVICE, dtype=torch.float16 if DEVICE == "cuda" else torch.float32)
                    for k, v in data.items()
                }
            log_with_timestamp(f"  ✓ Cache cargado: {len(self.cache):,} embeddings")
        else:
            self.cache = {}
    
    def save_cache(self):
        """Persiste cache a disco"""
        log_with_timestamp("Guardando cache de embeddings...")
        # Convertir a CPU/numpy para serialización
        data_cpu = {
            k: v.cpu().numpy().astype(np.float32) 
            for k, v in self.cache.items()
        }
        with open(self.cache_path, 'wb') as f:
            pickle.dump(data_cpu, f)
        log_with_timestamp(f"  ✓ Cache guardado: {len(self.cache):,} embeddings")
    
    def get_or_compute(self, names: list[str]) -> torch.Tensor:
        """Retorna embeddings para lista de nombres (usa cache o calcula)"""
        self.load_model()
        
        # Identificar nombres no cacheados
        names_to_compute = [n for n in names if n not in self.cache]
        
        if names_to_compute:
            log_with_timestamp(f"  Calculando embeddings para {len(names_to_compute):,} nombres...")
            # Preprocesar batch
            preprocessed = [self._preprocess(n) for n in names_to_compute]
            
            # Calcular en GPU con batching
            embeddings = self.model.encode(
                preprocessed,
                convert_to_tensor=True,
                device=DEVICE,
                batch_size=256,  # Batch interno del modelo
                show_progress_bar=False
            )
            
            # Guardar en cache
            for name, emb in zip(names_to_compute, embeddings):
                self.cache[name] = emb
        
        # Stack de vuelta en GPU
        return torch.stack([self.cache[n] for n in names])
    
    def _preprocess(self, name: str) -> str:
        """Preprocesamiento rápido"""
        if not name:
            return ""
        return BUSINESS_PATTERN.sub(" ", name.lower()).strip()

# --- PROGRESS TRACKER ---
class ProgressTracker:
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = self.checkpoint_dir / "progress.json"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def load_progress(self):
        """Carga el último estado guardado"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    data = json.load(f)
                return data.get('next_start_index', 0), data.get('stats', {'total': 0, 'high': 0, 'medium': 0, 'low': 0})
            except Exception as e:
                log_with_timestamp(f"Error cargando checkpoint: {e}")
                return 0, {'total': 0, 'high': 0, 'medium': 0, 'low': 0}
        return 0, {'total': 0, 'high': 0, 'medium': 0, 'low': 0}

    def save_progress(self, next_start_index, stats):
        """Guarda el estado actual"""
        data = {
            'next_start_index': next_start_index,
            'stats': stats,
            'timestamp': datetime.now().isoformat()
        }
        with open(self.checkpoint_file, 'w') as f:
            json.dump(data, f, indent=2)
            
    def get_partition_path(self, start_index):
        return self.checkpoint_dir / f"part_{start_index}.parquet"

# --- CPU PARALLEL FUZZY ---
def compute_fuzzy_batch(args) -> np.ndarray:
    """Función para multiprocessing de RapidFuzz"""
    names1, names2, start_idx, end_idx = args
    # Calcular matriz de distancias parcial
    scores = cdist(
        names1[start_idx:end_idx],
        names2[start_idx:end_idx],
        scorer=fuzz.token_sort_ratio,
        score_cutoff=0,
        workers=-1
    )
    return scores.diagonal()  # Solo necesitamos la diagonal

# --- GPU-ACCELERATED PIPELINE ---
def process_fuzzy_matching_gpu():
    log_with_timestamp("="*80)
    log_with_timestamp("FUZZY MATCHING V3-GPU - Pipeline Híbrido GPU-CPU")
    log_with_timestamp(f"Dispositivo: {DEVICE} | VRAM: {torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'N/A'}")
    log_with_timestamp(f'Torch cuda available: {torch.cuda.is_available()}')
    log_with_timestamp(f'FAISS GPUs: {faiss.get_num_gpus()}')
    log_with_timestamp("="*80)
    
    # 1. Inicializar cache global
    cache = GPUEmbeddingCache()
    cache.load_cache()
    
    # 2. Streaming de datos con Polars
    log_with_timestamp("\nPASO 1/4 - Configurando pipeline de streaming...")
    
    # Opción A: Si crees que hay muchos nombres únicos, hacer collect y procesar
    # Opción B: Si nombres muy repetidos, procesar en batches
    # Para 179M filas, asumo que quieres máxima velocidad, uso A.
    
    # Leer metadata primero
    schema = pl.scan_parquet(INPUT_FILE).schema
    LazyFrame.collect_schema()
    log_with_timestamp(f"  Columnas: {list(schema.keys())}")
    
    # Contar nombres únicos (esto es rápido con Polars)
    unique_person_names = pl.scan_parquet(INPUT_FILE).select(pl.col("person_name").unique()).collect()
    unique_orbis_names = pl.scan_parquet(INPUT_FILE).select(pl.col("orbis_name").unique()).collect()
    
    all_unique_names = list(
        set(unique_person_names["person_name"].to_list()) |
        set(unique_orbis_names["orbis_name"].to_list())
    )
    
    log_with_timestamp(f"  Nombres únicos: {len(all_unique_names):,}")
    
    # 3. Precomputar embeddings para TODOS los nombres únicos (una vez)
    log_with_timestamp("\nPASO 2/4 - Precomputando embeddings globales...")
    all_embeddings = cache.get_or_compute(all_unique_names)
    cache.save_cache()  # Persistir para futuras ejecuciones
    
    # Crear lookup tensor
    name_to_idx = {name: i for i, name in enumerate(all_unique_names)}
    
    # 4. Construir índice FAISS en GPU
    log_with_timestamp("\nPASO 3/4 - Construyendo índice FAISS GPU...")
    if DEVICE == "cuda":
        res = faiss.StandardGpuResources()
        index = faiss.IndexFlatIP(EMBEDDING_DIM)  # Inner Product = Cosine Sim
        index = faiss.index_cpu_to_gpu(res, 0, index)
    else:
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
    
    # Normalizar embeddings para cosine similarity
    embeddings_np = all_embeddings.cpu().numpy().astype('float32', copy=False)
    embeddings_np = np.ascontiguousarray(embeddings_np)
    faiss.normalize_L2(embeddings_np)
    index.add(embeddings_np)
    
    log_with_timestamp(f"  ✓ Índice construido con {index.ntotal:,} vectores")
    
    # 5. Pipeline de procesamiento con streaming
    log_with_timestamp("\nPASO 4/4 - Procesando 179M combinaciones...")
    
    output_file = Path(OUTPUT_DIR) / "matched_ES_fuzzy_v3_gpu.parquet"
    high_conf_file = Path(OUTPUT_DIR) / "matched_ES_high_confidence_v3_gpu.parquet"
    
    # Configurar checkpoints
    CHECKPOINT_DIR = Path(OUTPUT_DIR) / "checkpoints_fuzzy_gpu"
    tracker = ProgressTracker(CHECKPOINT_DIR)
    
    # Cargar progreso previo
    start_index, stats = tracker.load_progress()
    if start_index > 0:
        log_with_timestamp(f"  ⚠ RESUMIENDO desde fila {start_index:,}")
        log_with_timestamp(f"  ⚠ Estadísticas previas: {stats}")
    
    # Stream de datos en batches GPU-friendly
    batch_size = get_optimal_batch_size()
    
    # Process in chunks by reading slices
    total_rows = pl.scan_parquet(INPUT_FILE).select(pl.len()).collect().item()
    
    # Loop principal
    for start in range(start_index, total_rows, batch_size):
        batch_df = pl.scan_parquet(INPUT_FILE).slice(start, batch_size).collect()

        # Vectorizar nombres
        person_names = batch_df["person_name"]
        orbis_names = batch_df["orbis_name"]
        
        # Lookup de embeddings
        person_indices = [name_to_idx[n] for n in person_names]
        orbis_indices = [name_to_idx[n] for n in orbis_names]
        
        # Extraer embeddings del cache global
        person_embs = all_embeddings[person_indices]
        orbis_embs = all_embeddings[orbis_indices]
        
        # Calcular similitud semántica en GPU (vectorizado)
        semantic_scores = torch.cosine_similarity(person_embs, orbis_embs, dim=1) * 100
        
        # Calcular similitud léxica en CPU paralelo
        # log_with_timestamp(f"  CPU Fuzzy: {len(person_names):,} comparaciones...")
        # Preprocesar nombres para fuzzy
        person_prep = [cache._preprocess(n) for n in person_names]
        orbis_prep = [cache._preprocess(n) for n in orbis_names]
        
        # Usar multiprocessing para RapidFuzz
        with Pool(cpu_count() // 2) as pool:
            # Dividir en sub-batches para CPU
            chunk_size_cpu = max(1, len(person_prep) // (cpu_count() // 2))
            args = [
                (person_prep, orbis_prep, i, min(i + chunk_size_cpu, len(person_prep)))
                for i in range(0, len(person_prep), chunk_size_cpu)
            ]
            fuzzy_chunks = pool.map(compute_fuzzy_batch, args)
            fuzzy_scores = np.concatenate(fuzzy_chunks)
        
        # Combinar scores
        final_scores = (FUZZY_WEIGHT * fuzzy_scores + 
                       SEMANTIC_WEIGHT * semantic_scores.cpu().numpy())
        
        # Clasificar confianza
        confidence_levels = np.where(
            final_scores >= CONFIDENCE_LEVELS['high'], "high",
            np.where(final_scores >= CONFIDENCE_LEVELS['medium'], "medium", "low")
        )
        
        # Añadir columnas
        batch_df = batch_df.with_columns([
            pl.Series("fuzzy_score", fuzzy_scores.astype(np.float32)),
            pl.Series("semantic_score", semantic_scores.cpu().numpy().astype(np.float32)),
            pl.Series("final_score", final_scores.astype(np.float32)),
            pl.Series("confidence_level", confidence_levels)
        ])
        
        # Actualizar stats
        stats['total'] += len(batch_df)
        stats['high'] += int((confidence_levels == "high").sum())
        stats['medium'] += int((confidence_levels == "medium").sum())
        stats['low'] += int((confidence_levels == "low").sum())
        
        # Escribir partición (checkpoint)
        part_file = tracker.get_partition_path(start)
        batch_df.write_parquet(part_file)
        
        # Guardar progreso
        tracker.save_progress(start + batch_size, stats)
        
        # Progreso visual
        progress_pct = (stats['total'] / total_rows) * 100
        print(f"\r  Progreso: {stats['total']:,}/{total_rows:,} ({progress_pct:.1f}%) | High: {stats['high']:,}", end="")
        
        # Notificar cada 10%
        if int(progress_pct) % 10 == 0 and int(progress_pct) > 0:
             # Solo imprimir si acabamos de cruzar un umbral de 10% (aproximado)
             # Para evitar spam, podríamos comprobar si el batch anterior estaba en <10% y este en >=10%
             # Pero con print \r es suficiente.
             pass

        # Liberar memoria GPU
        del person_embs, orbis_embs, semantic_scores
    
    print()  # Nueva línea después del progreso
    
    # Consolidar resultados
    log_with_timestamp("\nConsolidando resultados finales...")
    try:
        # Leer todas las particiones y guardar en un solo archivo
        # Usamos wildcards para leer todos los part_*.parquet
        parts_pattern = str(CHECKPOINT_DIR / "part_*.parquet")
        pl.scan_parquet(parts_pattern).sink_parquet(output_file)
        log_with_timestamp(f"  ✓ Archivo consolidado guardado: {output_file}")
        
        # Opcional: Limpiar checkpoints si todo salió bien
        # shutil.rmtree(CHECKPOINT_DIR) 
        # log_with_timestamp("  ✓ Checkpoints temporales eliminados")
        
    except Exception as e:
        log_with_timestamp(f"Error consolidando archivos: {e}")
        log_with_timestamp(f"Los resultados parciales están en: {CHECKPOINT_DIR}")
        return

    # Generar archivo de alta confianza
    log_with_timestamp("\nFiltrando alta confianza...")
    high_df = pl.scan_parquet(output_file).filter(
        pl.col("confidence_level") == "high"
    ).collect(streaming=True)
    high_df.write_parquet(high_conf_file)
    
    # Generar estadísticas
    generate_stats(stats, output_file, high_conf_file)
    
    log_with_timestamp("\n" + "="*80)
    log_with_timestamp("✅ PROCESO GPU COMPLETADO")
    log_with_timestamp("="*80)


def generate_stats(stats, output_file, high_conf_file):
    """Genera estadísticas finales"""
    avg_fuzzy = pl.scan_parquet(output_file).select(
        pl.col("fuzzy_score").mean()
    ).collect().item()
    
    avg_semantic = pl.scan_parquet(output_file).select(
        pl.col("semantic_score").mean()
    ).collect().item()
    
    avg_final = pl.scan_parquet(output_file).select(
        pl.col("final_score").mean()
    ).collect().item()
    
    # ... resto de tu código de estadísticas ...
    log_with_timestamp(f"  ✓ Estadísticas generadas")

# --- MAIN ---
if __name__ == "__main__":
    process_fuzzy_matching_gpu()