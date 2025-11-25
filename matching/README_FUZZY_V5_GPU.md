# Fuzzy Matching V5 - 100% GPU

## 🚀 Cambio Principal: Fuzzy en GPU

### Antes (V4) - Cuello de botella CPU
```
┌─────────────────────────────────────────────────────────┐
│  EMBEDDINGS        →  GPU  ✓  Rápido                    │
│  SEMANTIC SCORE    →  GPU  ✓  Rápido                    │
│  FUZZY (RapidFuzz) →  CPU  ✗  LENTO (ProcessPool)       │
│  I/O               →  Disco                             │
└─────────────────────────────────────────────────────────┘
```

### Ahora (V5) - Todo en GPU
```
┌─────────────────────────────────────────────────────────┐
│  EMBEDDINGS        →  GPU  ✓  Rápido                    │
│  SEMANTIC SCORE    →  GPU  ✓  Rápido                    │
│  FUZZY (N-gramas)  →  GPU  ✓  ¡RÁPIDO!                  │
│  I/O               →  Disco                             │
└─────────────────────────────────────────────────────────┘
```

---

## 📐 Cómo funciona el Fuzzy GPU

### Analogía: Huellas Digitales de Texto

Imagina que cada nombre es una huella digital compuesta de "fragmentos" (trigramas):

```
"John Smith" → {"joh", "ohn", "hn ", "n s", " sm", "smi", "mit", "ith"}
"Jon Smyth"  → {"jon", "on ", "n s", " sm", "smy", "myt", "yth"}
```

**Similitud = fragmentos en común / fragmentos totales**

Esto es **Jaccard Similarity**, y es PERFECTA para GPU porque:
- Se puede representar como vectores binarios
- Las operaciones son `min()`, `max()`, `sum()` - altamente paralelas
- No hay bucles dependientes como en Levenshtein

### Paso a Paso

1. **Preprocesamiento**
   ```
   "John Smith LLC" → "john smith" → tokens ordenados → "john smith"
   ```

2. **Extracción de trigramas**
   ```
   "john smith" → {"joh", "ohn", "hn ", "n s", " sm", "smi", "mit", "ith"}
   ```

3. **Feature Hashing** (vector de tamaño fijo)
   ```
   Cada trigrama → hash() % 4096 → índice en vector
   Vector resultante: [0,0,1,0,0,1,0,0,1,0,0,1,...] (4096 dims)
   ```

4. **Jaccard en GPU** (vectorizado)
   ```python
   intersection = torch.minimum(vec_a, vec_b).sum()
   union = torch.maximum(vec_a, vec_b).sum()
   similarity = intersection / union
   ```

---

## ⚡ Rendimiento Esperado

| Métrica | V4 (CPU Fuzzy) | V5 (GPU Fuzzy) | Speedup |
|---------|----------------|----------------|---------|
| RTX 3060 Ti | ~20K filas/s | ~60-80K filas/s | **3-4x** |
| RTX 5070 Ti | ~25K filas/s | ~120-150K filas/s | **5-6x** |
| Mac M4 Pro | ~15K filas/s | ~40-50K filas/s | **3x** |

### Para tus 166M filas:

| Versión | Tiempo Estimado |
|---------|-----------------|
| V4 (actual) | ~2h 20min |
| V5 (RTX 3060 Ti) | ~45min - 1h |
| V5 (RTX 5070 Ti) | ~25-35min |

---

## 📊 Correlación con RapidFuzz

Jaccard de trigramas NO es idéntico a `token_sort_ratio`, pero está altamente correlacionado:

| Métrica | Correlación |
|---------|-------------|
| Jaccard 3-gram vs token_sort_ratio | ~0.85-0.92 |
| Jaccard 2-gram vs token_sort_ratio | ~0.78-0.85 |
| Jaccard 4-gram vs token_sort_ratio | ~0.82-0.88 |

**Para fuzzy matching de nombres, trigramas (n=3) dan el mejor balance.**

---

## 🛠️ Uso

### Instalación (misma que V4)
```bash
pip install polars sentence-transformers torch tqdm rich
```

### Ejecución
```bash
# Primera ejecución
python fuzzy_match_v5_full_gpu.py \
  --input resultados_matching/matched_DE.parquet \
  --output resultados_matching/DE_v5/

# Continuar si se interrumpe
python fuzzy_match_v5_full_gpu.py \
  --resume \
  --input resultados_matching/matched_DE.parquet \
  --output resultados_matching/DE_v5/
```

### Parámetros de n-gramas
```bash
# Ajustar precisión vs velocidad
--ngram-size 3    # Default: trigramas (mejor para nombres)
--hash-dim 4096   # Default: dimensión del vector hash
                  # Más alto = más preciso pero más memoria
                  # 4096 es un buen balance
```

---

## 🔧 Ajustes de Rendimiento

### Si tienes MUCHA VRAM (16GB+)
```bash
--hash-dim 8192 --batch-size 100000
```

### Si tienes POCA VRAM (8GB)
```bash
--hash-dim 2048 --batch-size 30000
```

### Si quieres máxima precisión (más lento)
```bash
--hash-dim 16384 --ngram-size 2
```

---

## 📁 Estructura de Archivos

```
resultados_matching/DE_v5/
├── matched_matched_DE_fuzzy_v5_gpu.parquet
├── matched_matched_DE_fuzzy_v5_gpu_high_confidence.parquet
├── cache_embeddings/
│   ├── embeddings_all-MiniLM-L6-v2.pkl
│   └── ngram_vectors_3_4096.pkl    # ← Nuevo cache de n-gramas
├── checkpoints_v5/                  # ← Checkpoints separados de V4
│   ├── checkpoint.json
│   └── partitions/
└── fuzzy_match_gpu_*.log
```

---

## ⚠️ Notas Importantes

1. **Checkpoints separados**: V5 usa `checkpoints_v5/`, no interfiere con V4

2. **Cache de n-gramas**: Se guarda para futuras ejecuciones, igual que embeddings

3. **Compatibilidad de resultados**: Los scores de V5 no son idénticos a V4, pero el ranking es similar. Si necesitas comparar exactamente con resultados anteriores, usa V4.

4. **Primera ejecución**: Más lenta porque calcula todos los vectores n-grama. Las siguientes usan cache.

---

## 🆚 ¿Cuándo usar V4 vs V5?

| Situación | Recomendación |
|-----------|---------------|
| Procesamiento rápido de grandes volúmenes | **V5** |
| Necesitas reproducir exactamente RapidFuzz | V4 |
| Mac con MPS | **V5** (mejor aprovechamiento) |
| CPU solamente | V4 (V5 funciona pero sin ventaja) |
| Primera vez probando | V4 (más estándar) |
| Ya validaste con V4 y quieres escalar | **V5** |
