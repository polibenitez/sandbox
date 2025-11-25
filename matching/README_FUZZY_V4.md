# Fuzzy Matching V4 - Guía de Mejoras y Uso

## 🚀 Mejoras Principales sobre V3

### 1. **Compatibilidad Universal**
```
Linux + CUDA  →  GPU NVIDIA (RTX 3060 Ti, etc.)
Mac M4 Pro    →  Metal Performance Shaders (MPS)
Cualquier PC  →  CPU fallback automático
```

La detección es automática. No necesitas cambiar código.

### 2. **Checkpointing Robusto**
```
Antes (V3):  Si falla, pierdes todo el progreso
Ahora (V4):  
  - Guardado atómico (sin corrupción)
  - Validación de integridad (hash MD5)
  - Auto-recuperación desde backup
  - Resume con: python script.py --resume -i archivo.parquet
```

### 3. **Progreso Visible**
```
Con rich instalado:
  ⠋ Procesando ████████████░░░░░░░░ 45% • 00:12:34 • 00:15:21

Sin rich:
  Procesando: 45%|████████████░░░░░░░░| 4.5M/10M [12:34<15:21]
```

### 4. **Gestión de Memoria Mejorada**
- Batch size auto-ajustado según VRAM/RAM
- Liberación periódica de memoria GPU
- Embeddings pre-normalizados (menos cálculos)

---

## 📦 Instalación

### Linux con CUDA (RTX 3060 Ti)
```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate

# Dependencias base
pip install polars rapidfuzz sentence-transformers tqdm rich psutil

# PyTorch con CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# FAISS con GPU
pip install faiss-gpu
```

### Mac M4 Pro
```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate

# Dependencias (MPS se activa automáticamente)
pip install polars rapidfuzz sentence-transformers tqdm rich psutil
pip install torch  # Incluye soporte MPS
pip install faiss-cpu  # No hay faiss-gpu para Mac
```

---

## 🎮 Uso

### Ejecución básica
```bash
python fuzzy_match_v4_universal.py \
  --input datos/matched_DE.parquet \
  --output resultados/
```

### Con parámetros personalizados
```bash
python fuzzy_match_v4_universal.py \
  --input datos.parquet \
  --output resultados/ \
  --batch-size 25000 \
  --fuzzy-weight 0.6 \
  --semantic-weight 0.4 \
  --col-source "nombre_persona" \
  --col-target "nombre_empresa"
```

### Continuar después de interrupción
```bash
# Ctrl+C durante ejecución → progreso guardado automáticamente
# Para continuar:
python fuzzy_match_v4_universal.py --resume --input datos.parquet
```

### Forzar reinicio (ignorar checkpoints)
```bash
python fuzzy_match_v4_universal.py --force --input datos.parquet
```

---

## 📊 Archivos de Salida

```
resultados/
├── matched_datos_fuzzy_v4.parquet      # Todos los resultados
├── matched_datos_fuzzy_v4_high_confidence.parquet  # Solo alta confianza
├── cache_embeddings/
│   └── embeddings_all-MiniLM-L6-v2.pkl # Cache reutilizable
├── checkpoints/
│   ├── checkpoint.json                  # Estado actual
│   └── partitions/
│       ├── part_0000000000.parquet
│       └── ...
└── fuzzy_match_YYYYMMDD_HHMMSS.log     # Log completo
```

---

## ⚡ Benchmarks Esperados

| Hardware | Batch Size | Velocidad Aprox. |
|----------|------------|------------------|
| RTX 3060 Ti (8GB) | 30,000 | ~500K filas/min |
| Mac M4 Pro (18GB) | 40,000 | ~350K filas/min |
| CPU (8 cores) | 10,000 | ~50K filas/min |

Para 179M filas:
- GPU: ~6-8 horas
- CPU: ~60 horas

---

## 🔧 Troubleshooting

### "CUDA out of memory"
```bash
# Reducir batch size
python script.py --input datos.parquet --batch-size 15000
```

### "MPS backend not available" en Mac
```bash
# Verificar versión de PyTorch
python -c "import torch; print(torch.backends.mps.is_available())"

# Si es False, actualizar PyTorch
pip install --upgrade torch
```

### Proceso muy lento
1. Verificar que se detecta GPU: mira la línea "Dispositivo:" al inicio
2. Si dice CPU pero tienes GPU, verificar drivers/CUDA
3. El primer batch es lento (carga modelo), después acelera

---

## 🆚 Comparación V3 vs V4

| Aspecto | V3 | V4 |
|---------|----|----|
| Plataformas | Solo CUDA | CUDA + MPS + CPU |
| Checkpointing | Básico | Atómico + validación |
| Progreso | print("\r...") | tqdm/rich con ETA |
| Errores de código | Varios bugs | Corregidos |
| Configuración | Hardcoded | CLI args |
| Logging | Básico | Dual (archivo + consola) |
| Memoria | Manual | Auto-gestionada |
