"""
fuzzy_match_v3.py - Fuzzy Matching con similitud híbrida (léxica + semántica)

MEJORAS RESPECTO A V2:
- Mantiene RapidFuzz con token_sort_ratio para similitud léxica
- Añade similitud semántica usando sentence-transformers (all-MiniLM-L6-v2)
- Score híbrido: 70% léxico + 30% semántico
- Preprocesamiento mejorado: elimina palabras genéricas de negocio
- Cache de embeddings para optimización
- Columnas adicionales: semantic_score, final_score
- Niveles de confianza basados en final_score

ENTRADA:
  v3/resultados_matching/matched_ES.parquet (179,989,635 filas)
  Columnas: person_id, person_name, orbis_id, orbis_name, match_level, etc.

SALIDA:
  v3/resultados_matching/matched_ES_fuzzy_v3.parquet (con scores léxico, semántico y final)
  v3/resultados_matching/matched_ES_high_confidence_v3.parquet (final_score >= 90)
  v3/resultados_matching/fuzzy_statistics_v3.txt (estadísticas)
"""

import polars as pl
import os
from rapidfuzz import fuzz
from datetime import datetime
import gc
from sentence_transformers import SentenceTransformer, util
import torch
import re

# --- CONFIGURACIÓN ---
INPUT_FILE = "v3/resultados_matching/matched_ES.parquet"
OUTPUT_DIR = "v3/resultados_matching"
CHUNK_SIZE = 100_000  # Procesar 100k filas a la vez
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Niveles de confianza (ahora basados en final_score)
CONFIDENCE_LEVELS = {
    'high': 90,      # >= 90: Match muy confiable
    'medium': 75,    # 75-89: Requiere revisión
    'low': 0         # < 75: Probablemente incorrecto
}

# Ponderación para score híbrido
FUZZY_WEIGHT = 0.7
SEMANTIC_WEIGHT = 0.3

# Palabras genéricas a eliminar
BUSINESS_WORDS = [
    "pharmaceuticals", "pharmaceutical", "industries", "technologies",
    "technology", "corporation", "corp", "company", "companies",
    "group", "sa", "sl", "spa", "gmbh", "limited", "ltd", "llc",
    "inc", "ag", "nv", "bv", "srl", "plc", "pty", "coop"
]

# Modelo de embeddings (se carga una sola vez)
log_with_timestamp_enabled = True

def log_with_timestamp(message):
    """Log con timestamp"""
    if log_with_timestamp_enabled:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")


def load_embedding_model():
    """
    Carga el modelo de sentence-transformers una sola vez en el mejor dispositivo disponible:
    - 'cuda' si hay GPU NVIDIA
    - 'mps' si estás en Mac con chip Apple Silicon (M1/M2/M3/M4)
    - 'cpu' si no hay GPU compatible
    Usa all-MiniLM-L6-v2: rápido y eficiente.
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    log_with_timestamp(f"Cargando modelo de embeddings (all-MiniLM-L6-v2) en dispositivo: {device}...")
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    log_with_timestamp("  ✓ Modelo cargado exitosamente")
    return model


def preprocess_name(name: str) -> str:
    """
    Preprocesamiento mejorado de nombres:
    1. Convertir a minúsculas
    2. Eliminar puntuación
    3. Eliminar palabras genéricas de negocio
    4. Normalizar espacios
    """
    if not name or not isinstance(name, str):
        return ""

    # 1. Minúsculas
    name = name.lower()

    # 2. Eliminar puntuación (mantener espacios y letras)
    name = re.sub(r'[^\w\s]', ' ', name)

    # 3. Eliminar palabras genéricas (como palabras completas)
    for word in BUSINESS_WORDS:
        # Usar \b para word boundaries
        name = re.sub(r'\b' + re.escape(word) + r'\b', '', name)

    # 4. Normalizar espacios múltiples y trim
    name = re.sub(r'\s+', ' ', name).strip()

    return name


def clean_name_expression(col_name: str) -> pl.Expr:
    """
    Expresión de Polars para limpiar nombres (versión v2 original).
    Se mantiene para compatibilidad con token_sort_ratio.
    """
    # Lista ampliada de sufijos legales a eliminar
    suffixes = [
        r"\bS\.?L\.?\b", r"\bS\.?A\.?\b", r"\bS\.?L\.?U\.?\b", r"\bS\.?A\.?U\.?\b",
        r"\bSOCIEDAD LIMITADA\b", r"\bSOCIEDAD ANONIMA\b", r"\bSOCIEDAD LIMITADA UNIPERSONAL\b",
        r"\bLLC\b", r"\bINC\.?\b", r"\bCORP\.?\b", r"\bLTD\.?\b", r"\bLIMITED\b",
        r"\bGMBH\b", r"\bAG\b", r"\bN\.?V\.?\b", r"\bB\.?V\.?\b",
        r"\bSPA\b", r"\bS\.?R\.?L\.?\b", r"\bPLC\b", r"\bPTY\b",
        r"\bCOOP\.?\b", r"\bS\.?COOP\.?\b",
        r"[\.,;&\-\(\)\[\]{}]"  # Puntuación
    ]

    expr = pl.col(col_name).str.to_uppercase()

    # Eliminar cada sufijo
    for suffix in suffixes:
        expr = expr.str.replace_all(suffix, " ")

    # Limpiar espacios múltiples y trim
    expr = expr.str.replace_all(r"\s+", " ").str.strip_chars()

    return expr.alias(f"{col_name}_clean")


def build_embedding_cache(df: pl.DataFrame, model: SentenceTransformer) -> dict:
    """
    Construye un cache de embeddings para todos los nombres únicos.
    Esto evita recalcular embeddings para nombres repetidos.
    """
    # Extraer nombres únicos de ambas columnas
    unique_person_names = set(df["person_name"].to_list())
    unique_orbis_names = set(df["orbis_name"].to_list())

    # Combinar y preprocesar
    all_unique_names = list(unique_person_names | unique_orbis_names)
    preprocessed_names = [preprocess_name(name) for name in all_unique_names]

    # Filtrar nombres vacíos
    valid_pairs = [(orig, prep) for orig, prep in zip(all_unique_names, preprocessed_names) if prep]

    if not valid_pairs:
        return {}

    original_names, preprocessed = zip(*valid_pairs)

    # Calcular embeddings en batch (mucho más rápido)
    log_with_timestamp(f"  Calculando embeddings para {len(preprocessed)} nombres únicos...")
    embeddings = model.encode(list(preprocessed), convert_to_tensor=True, show_progress_bar=False)

    # Crear diccionario de cache
    cache = {}
    for orig_name, prep_name, embedding in zip(original_names, preprocessed, embeddings):
        cache[orig_name] = {
            'preprocessed': prep_name,
            'embedding': embedding
        }

    return cache


def hybrid_similarity(person_name: str, orbis_name: str,
                     embedding_cache: dict, model: SentenceTransformer) -> tuple:
    """
    Calcula similitud híbrida entre dos nombres.

    Returns:
        tuple: (fuzzy_score, semantic_score, final_score)
    """
    # Obtener nombres preprocesados y embeddings del cache
    person_data = embedding_cache.get(person_name, None)
    orbis_data = embedding_cache.get(orbis_name, None)

    if not person_data or not orbis_data:
        # Fallback: calcular on-the-fly si no está en cache
        person_prep = preprocess_name(person_name)
        orbis_prep = preprocess_name(orbis_name)

        if not person_prep or not orbis_prep:
            return (0, 0, 0)

        # Calcular fuzzy score
        fuzzy_score = fuzz.token_sort_ratio(person_prep, orbis_prep)

        # Calcular semantic score
        person_emb = model.encode(person_prep, convert_to_tensor=True)
        orbis_emb = model.encode(orbis_prep, convert_to_tensor=True)
        semantic_score = util.cos_sim(person_emb, orbis_emb).item() * 100

    else:
        person_prep = person_data['preprocessed']
        orbis_prep = orbis_data['preprocessed']

        if not person_prep or not orbis_prep:
            return (0, 0, 0)

        # Calcular fuzzy score
        fuzzy_score = fuzz.token_sort_ratio(person_prep, orbis_prep)

        # Calcular semantic score usando embeddings del cache
        semantic_score = util.cos_sim(person_data['embedding'],
                                     orbis_data['embedding']).item() * 100

    # Score final híbrido
    final_score = (FUZZY_WEIGHT * fuzzy_score) + (SEMANTIC_WEIGHT * semantic_score)

    return (fuzzy_score, semantic_score, final_score)


def calculate_similarity_batch(df: pl.DataFrame, model: SentenceTransformer) -> pl.DataFrame:
    """
    Calcula la similitud léxica, semántica e híbrida para un batch.
    Optimizado con cache de embeddings.
    """
    # Construir cache de embeddings para este batch
    log_with_timestamp("  Construyendo cache de embeddings para el batch...")
    embedding_cache = build_embedding_cache(df, model)

    # Calcular similitud para cada fila
    log_with_timestamp("  Calculando similitudes (léxica + semántica)...")
    fuzzy_scores = []
    semantic_scores = []
    final_scores = []

    person_names = df["person_name"].to_list()
    orbis_names = df["orbis_name"].to_list()

    for person, orbis in zip(person_names, orbis_names):
        fuzzy, semantic, final = hybrid_similarity(person, orbis, embedding_cache, model)
        fuzzy_scores.append(fuzzy)
        semantic_scores.append(semantic)
        final_scores.append(final)

    # Añadir columnas de similitud
    df = df.with_columns([
        pl.Series("fuzzy_score", fuzzy_scores),
        pl.Series("semantic_score", semantic_scores),
        pl.Series("final_score", final_scores)
    ])

    # Clasificar por nivel de confianza (basado en final_score)
    df = df.with_columns(
        pl.when(pl.col("final_score") >= CONFIDENCE_LEVELS['high'])
        .then(pl.lit("high"))
        .when(pl.col("final_score") >= CONFIDENCE_LEVELS['medium'])
        .then(pl.lit("medium"))
        .otherwise(pl.lit("low"))
        .alias("confidence_level")
    )

    return df


def process_fuzzy_matching():
    """
    Proceso principal de fuzzy matching con similitud híbrida.
    Procesa el archivo en chunks para manejar el gran volumen.
    """
    log_with_timestamp("="*80)
    log_with_timestamp("FUZZY MATCHING V3 - Análisis Híbrido (Léxico + Semántico)")
    log_with_timestamp("="*80)

    log_with_timestamp(f"\nArchivo de entrada: {INPUT_FILE}")

    # Verificar que existe el archivo
    if not os.path.exists(INPUT_FILE):
        log_with_timestamp(f"❌ ERROR: No se encuentra {INPUT_FILE}")
        log_with_timestamp("   Ejecuta primero: python matching_v3.py")
        return

    # Cargar modelo de embeddings una sola vez
    model = load_embedding_model()

    # Contar filas totales
    log_with_timestamp("\nPASO 1/5 - Analizando archivo de entrada...")
    total_rows = pl.scan_parquet(INPUT_FILE).select(pl.count()).collect().item()
    log_with_timestamp(f"  Total de combinaciones: {total_rows:,}")

    # Calcular número de chunks
    num_chunks = (total_rows // CHUNK_SIZE) + 1
    log_with_timestamp(f"  Procesando en {num_chunks:,} chunks de {CHUNK_SIZE:,} filas")

    # Archivos temporales para cada chunk procesado
    temp_files = []

    # Estadísticas globales
    stats = {
        'total_processed': 0,
        'high_confidence': 0,
        'medium_confidence': 0,
        'low_confidence': 0,
        'fuzzy_score_sum': 0,
        'semantic_score_sum': 0,
        'final_score_sum': 0
    }

    log_with_timestamp("\nPASO 2/5 - Calculando similitud híbrida de nombres...")

    # Procesar por chunks
    lazy_df = pl.scan_parquet(INPUT_FILE)

    for chunk_idx in range(num_chunks):
        offset = chunk_idx * CHUNK_SIZE

        # Leer chunk
        chunk_df = lazy_df.slice(offset, CHUNK_SIZE).collect()

        if len(chunk_df) == 0:
            break

        # Calcular similitud
        chunk_df = calculate_similarity_batch(chunk_df, model)

        # Actualizar estadísticas
        stats['total_processed'] += len(chunk_df)
        stats['high_confidence'] += (chunk_df['confidence_level'] == 'high').sum()
        stats['medium_confidence'] += (chunk_df['confidence_level'] == 'medium').sum()
        stats['low_confidence'] += (chunk_df['confidence_level'] == 'low').sum()
        stats['fuzzy_score_sum'] += chunk_df['fuzzy_score'].sum()
        stats['semantic_score_sum'] += chunk_df['semantic_score'].sum()
        stats['final_score_sum'] += chunk_df['final_score'].sum()

        # Guardar chunk temporal
        temp_file = os.path.join(OUTPUT_DIR, f"temp_fuzzy_v3_{chunk_idx}.parquet")
        chunk_df.write_parquet(temp_file)
        temp_files.append(temp_file)

        # Mostrar progreso
        progress = (stats['total_processed'] / total_rows) * 100
        log_with_timestamp(f"  Progreso: {progress:.1f}% ({stats['total_processed']:,}/{total_rows:,})")

        # Liberar memoria
        del chunk_df
        gc.collect()

    log_with_timestamp("\n  ✓ Similitud híbrida calculada para todas las filas")

    # Consolidar resultados
    log_with_timestamp("\nPASO 3/5 - Consolidando resultados...")

    output_file = os.path.join(OUTPUT_DIR, "matched_ES_fuzzy_v3.parquet")

    # Usar scan para consolidar sin cargar todo en memoria
    combined = pl.scan_parquet(temp_files)
    combined.sink_parquet(output_file)

    log_with_timestamp(f"  ✓ Archivo completo guardado: {output_file}")

    # Crear archivo de alta confianza
    log_with_timestamp("\nPASO 4/5 - Generando archivo de alta confianza...")

    high_conf_file = os.path.join(OUTPUT_DIR, "matched_ES_high_confidence_v3.parquet")
    high_conf = pl.scan_parquet(output_file).filter(
        pl.col("confidence_level") == "high"
    )
    high_conf.sink_parquet(high_conf_file)

    high_count = pl.scan_parquet(high_conf_file).select(pl.count()).collect().item()
    log_with_timestamp(f"  ✓ Matches de alta confianza: {high_count:,}")
    log_with_timestamp(f"  ✓ Archivo guardado: {high_conf_file}")

    # Generar estadísticas
    log_with_timestamp("\nPASO 5/5 - Generando estadísticas...")

    avg_fuzzy = stats['fuzzy_score_sum'] / stats['total_processed'] if stats['total_processed'] > 0 else 0
    avg_semantic = stats['semantic_score_sum'] / stats['total_processed'] if stats['total_processed'] > 0 else 0
    avg_final = stats['final_score_sum'] / stats['total_processed'] if stats['total_processed'] > 0 else 0

    stats_text = f"""
{'='*80}
ESTADÍSTICAS DE FUZZY MATCHING V3 (HÍBRIDO)
{'='*80}

Archivo procesado: {INPUT_FILE}
Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

CONFIGURACIÓN:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Modelo de embeddings: all-MiniLM-L6-v2
  Métrica léxica: RapidFuzz token_sort_ratio
  Ponderación híbrida: {FUZZY_WEIGHT*100:.0f}% léxico + {SEMANTIC_WEIGHT*100:.0f}% semántico
  Palabras genéricas eliminadas: {len(BUSINESS_WORDS)} términos de negocio

RESUMEN GENERAL:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Total de combinaciones procesadas: {stats['total_processed']:,}

  Scores promedios:
    • Similitud léxica (fuzzy):    {avg_fuzzy:.2f}/100
    • Similitud semántica:          {avg_semantic:.2f}/100
    • Score final (híbrido):        {avg_final:.2f}/100

DISTRIBUCIÓN POR NIVEL DE CONFIANZA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Alta confianza (final_score ≥ 90):
    • Combinaciones: {stats['high_confidence']:,}
    • Porcentaje: {stats['high_confidence']/stats['total_processed']*100:.2f}%
    • Interpretación: Match muy confiable, listo para usar

  Confianza media (final_score 75-89):
    • Combinaciones: {stats['medium_confidence']:,}
    • Porcentaje: {stats['medium_confidence']/stats['total_processed']*100:.2f}%
    • Interpretación: Requiere revisión manual

  Baja confianza (final_score < 75):
    • Combinaciones: {stats['low_confidence']:,}
    • Porcentaje: {stats['low_confidence']/stats['total_processed']*100:.2f}%
    • Interpretación: Probablemente incorrecto

MEJORAS EN V3:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ✓ Reducción de falsos positivos semánticos
    Ejemplo: "ESTEVE PHARMACEUTICALS" vs "OTSUKA PHARMACEUTICAL"
    - V2 (solo léxico): Score alto por palabras comunes
    - V3 (híbrido): Score ajustado por contexto semántico diferente

  ✓ Preprocesamiento mejorado
    Eliminación de palabras genéricas: {', '.join(BUSINESS_WORDS[:10])}...

  ✓ Cache de embeddings
    Optimización: embeddings calculados una sola vez por nombre único

RECOMENDACIONES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. Usar archivo de alta confianza para análisis:
     {high_conf_file}

  2. Revisar manualmente una muestra de confianza media

  3. Descartar matches de baja confianza (final_score < 75)

  4. Comparar con V2 para evaluar mejora en precisión

ARCHIVOS GENERADOS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  • {output_file}
    └── Todos los matches con fuzzy_score, semantic_score, final_score y confidence_level

  • {high_conf_file}
    └── Solo matches de alta confianza (final_score ≥ 90)

{'='*80}
"""

    stats_file = os.path.join(OUTPUT_DIR, "fuzzy_statistics_v3.txt")
    with open(stats_file, 'w') as f:
        f.write(stats_text)

    print(stats_text)
    log_with_timestamp(f"Estadísticas guardadas en: {stats_file}")

    # Limpiar archivos temporales
    log_with_timestamp("\nLimpiando archivos temporales...")
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    log_with_timestamp("  ✓ Archivos temporales eliminados")

    log_with_timestamp("\n" + "="*80)
    log_with_timestamp("✅ PROCESO COMPLETADO")
    log_with_timestamp("="*80)


def analyze_top_matches(n=10):
    """
    Analiza los top N matches por score híbrido para verificar calidad.
    """
    log_with_timestamp(f"\nAnalizando top {n} matches por similitud híbrida...")

    output_file = os.path.join(OUTPUT_DIR, "matched_ES_fuzzy_v3.parquet")

    if not os.path.exists(output_file):
        log_with_timestamp("❌ Primero ejecuta process_fuzzy_matching()")
        return

    top_matches = pl.scan_parquet(output_file).sort(
        "final_score", descending=True
    ).head(n).collect()

    print("\n" + "="*80)
    print(f"TOP {n} MATCHES POR SIMILITUD HÍBRIDA:")
    print("="*80)

    for i, row in enumerate(top_matches.iter_rows(named=True), 1):
        print(f"\n{i}. Final Score: {row['final_score']:.2f} (Léxico: {row['fuzzy_score']:.2f}, Semántico: {row['semantic_score']:.2f})")
        print(f"   Patente: {row['person_name']}")
        print(f"   Empresa: {row['orbis_name']}")
        print(f"   Match por: {row['match_level']} ({row['matching_key']})")
        print(f"   IDs: person_id={row['person_id']}, orbis_id={row['orbis_id']}")


def compare_with_v2(n=20):
    """
    Compara resultados de V2 vs V3 para evaluar mejoras.
    Muestra casos donde V3 difiere significativamente de V2.
    """
    v2_file = os.path.join(OUTPUT_DIR, "matched_ES_fuzzy.parquet")
    v3_file = os.path.join(OUTPUT_DIR, "matched_ES_fuzzy_v3.parquet")

    if not os.path.exists(v2_file) or not os.path.exists(v3_file):
        log_with_timestamp("❌ Se requieren ambos archivos (V2 y V3) para comparar")
        return

    log_with_timestamp(f"\nComparando top {n} diferencias entre V2 y V3...")

    # Leer ambos archivos y unirlos por IDs
    v2_df = pl.scan_parquet(v2_file).select([
        "person_id", "orbis_id", "person_name", "orbis_name",
        pl.col("fuzzy_score").alias("v2_score")
    ])

    v3_df = pl.scan_parquet(v3_file).select([
        "person_id", "orbis_id",
        pl.col("fuzzy_score").alias("v3_fuzzy"),
        "semantic_score",
        pl.col("final_score").alias("v3_final")
    ])

    # Join y calcular diferencias
    comparison = v2_df.join(
        v3_df, on=["person_id", "orbis_id"], how="inner"
    ).with_columns(
        (pl.col("v3_final") - pl.col("v2_score")).alias("score_diff")
    ).sort(
        pl.col("score_diff").abs(), descending=True
    ).head(n).collect()

    print("\n" + "="*80)
    print(f"TOP {n} DIFERENCIAS MÁS SIGNIFICATIVAS (V2 vs V3):")
    print("="*80)

    for i, row in enumerate(comparison.iter_rows(named=True), 1):
        diff = row['score_diff']
        direction = "↓ REDUCIDO" if diff < 0 else "↑ AUMENTADO"

        print(f"\n{i}. {direction} en {abs(diff):.2f} puntos")
        print(f"   Patente: {row['person_name']}")
        print(f"   Empresa: {row['orbis_name']}")
        print(f"   V2 Score:      {row['v2_score']:.2f}")
        print(f"   V3 Léxico:     {row['v3_fuzzy']:.2f}")
        print(f"   V3 Semántico:  {row['semantic_score']:.2f}")
        print(f"   V3 Final:      {row['v3_final']:.2f}")
        print(f"   IDs: person_id={row['person_id']}, orbis_id={row['orbis_id']}")


def main():
    """Función principal"""
    # Ejecutar fuzzy matching
    process_fuzzy_matching()

    # Analizar top matches
    analyze_top_matches(10)

    # Opcional: comparar con V2 si existe
    if os.path.exists(os.path.join(OUTPUT_DIR, "matched_ES_fuzzy.parquet")):
        compare_with_v2(10)


if __name__ == "__main__":
    main()
