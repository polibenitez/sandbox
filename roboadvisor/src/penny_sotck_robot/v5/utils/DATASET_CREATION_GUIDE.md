# Guía de Creación del Dataset de Entrenamiento

## 📊 Resumen de Mejoras Implementadas

### ✅ Problemas Corregidos

| Problema Original | Solución Implementada |
|-------------------|----------------------|
| ❌ Funciones duplicadas (`compute_features2`, `fetch_data2`) | ✅ Eliminadas - código limpio y consolidado |
| ❌ ADX calculado incorrectamente (solo volatilidad) | ✅ ADX real implementado con DI+, DI- y True Range |
| ❌ Short float aleatorio (`np.random.uniform`) | ✅ Obtención real desde yfinance API |
| ❌ Faltaban features: `compression_days`, `volume_dry`, `price_range_pct` | ✅ Todos implementados correctamente |
| ❌ RSI con división por cero | ✅ Manejo robusto de división por cero |
| ❌ Criterios de setup muy restrictivos | ✅ Criterios más flexibles para generar suficientes samples |
| ❌ No validaba datos de salida | ✅ Validación completa con estadísticas |
| ❌ Threshold de explosión fijo | ✅ Configurable vía `--explosion-threshold` |
| ❌ Sin manejo de errores robusto | ✅ Try/catch en cada función, mensajes claros |

### 🆕 Nuevas Características

1. **Indicadores Técnicos Mejorados**
   - ADX real (no solo volatilidad)
   - RSI con manejo de división por cero
   - MACD con señal y diferencia
   - Bollinger Bands correctos
   - ATR y ATR ratio

2. **Features Adicionales**
   - `compression_days`: Días consecutivos en compresión (cálculo real)
   - `volume_dry`: Indicador binario de volumen bajo
   - `price_range_pct`: Rango de precio en ventana móvil

3. **Obtención Real de Short Interest**
   - Intenta obtener `shortPercentOfFloat` de yfinance
   - Fallback a `shortRatio` con estimación
   - Valor default 0.15 si no hay datos

4. **Criterios de Setup Flexibles**
   - Detecta compresión por múltiples métricas
   - Genera más samples para mejor entrenamiento
   - Configurable y extensible

5. **Estadísticas Detalladas**
   - Win rate del dataset
   - Ganancia promedio en explosiones vs no explosiones
   - Conteo por símbolo
   - Warnings si pocos samples

6. **Argumentos CLI Completos**
   ```bash
   --months              # Meses de histórico
   --explosion-threshold # % para considerar explosión
   --min-samples         # Mínimo de samples requeridos
   --output              # Ruta de salida personalizada
   ```

---

## 📋 Uso del Script

### Uso Básico

```bash
cd utils
python create_training_dataset.py
```

Esto generará un dataset con:
- 24 meses de histórico (default)
- Threshold de explosión: 15%
- Output: `../data/penny_stock_training.csv`

### Uso Avanzado

```bash
# Más histórico para más datos
python create_training_dataset.py --months 36

# Threshold más agresivo (10% para considerar explosión)
python create_training_dataset.py --explosion-threshold 0.10

# Combinar parámetros
python create_training_dataset.py --months 48 --explosion-threshold 0.20 --min-samples 100

# Salida personalizada
python create_training_dataset.py --output /path/to/custom_dataset.csv
```

---

## 📊 Formato del Dataset Generado

### Columnas

| Columna | Tipo | Descripción | Rango |
|---------|------|-------------|-------|
| `symbol` | str | Ticker del símbolo | - |
| `date` | datetime | Fecha del setup | - |
| `bb_width` | float | Ancho de Bandas de Bollinger | 0.0 - 0.3+ |
| `adx` | float | Average Directional Index | 0 - 100 |
| `vol_ratio` | float | Volumen / Promedio 20d | 0.1 - 10+ |
| `rsi` | float | Relative Strength Index | 0 - 100 |
| `macd_diff` | float | MACD - Signal | -0.5 - +0.5 |
| `atr_ratio` | float | ATR / Precio | 0.01 - 0.10 |
| `short_float` | float | % del float en corto | 0.0 - 0.50 |
| `compression_days` | int | Días consecutivos comprimido | 0 - 20 |
| `volume_dry` | int | 1 si volumen bajo, 0 si no | 0, 1 |
| `price_range_pct` | float | Rango precio últimos 5d (%) | 1 - 20 |
| **`exploded`** | **int** | **TARGET: 1 si explotó, 0 si no** | **0, 1** |
| `gain_pct` | float | Ganancia real en 5d (análisis) | -20 - +50+ |

### Ejemplo de Datos

```csv
symbol,date,bb_width,adx,vol_ratio,rsi,macd_diff,atr_ratio,short_float,compression_days,volume_dry,price_range_pct,exploded,gain_pct
BYND,2024-05-15,0.08,18.3,2.9,62,0.003,0.015,0.12,7,1,6.5,1,22.5
COSM,2024-05-18,0.10,22.1,1.7,48,-0.002,0.025,0.09,5,0,8.2,0,8.3
XAIR,2024-06-10,0.06,15.2,3.2,58,0.005,0.012,0.18,8,1,5.8,1,31.2
```

---

## 🔧 Cálculo de Indicadores

### ADX (Average Directional Index)

**Método correcto implementado:**

```python
# True Range
TR = max(High - Low, abs(High - Close_prev), abs(Low - Close_prev))

# Directional Movement
+DM = High - High_prev (si > 0 y > -DM)
-DM = Low_prev - Low (si > 0 y > +DM)

# Directional Indicators
+DI = 100 * (+DM_smooth / ATR)
-DI = 100 * (-DM_smooth / ATR)

# ADX
DX = 100 * |+DI - -DI| / (+DI + -DI)
ADX = smooth(DX, 14)
```

**Interpretación:**
- ADX < 20: Sin tendencia (ideal para compresión)
- ADX 20-25: Tendencia débil
- ADX > 25: Tendencia fuerte

### Compression Days

**Algoritmo:**

```python
for cada día i:
    días_comprimidos = 0
    mirar_hasta_20_días_atrás:
        rango_precio = (max - min) / min * 100
        if rango_precio <= 8%:
            días_comprimidos += 1
        else:
            break  # Se rompió la compresión
```

Esto cuenta **días consecutivos** donde el precio estuvo comprimido (<8% de rango).

### Short Interest

**Prioridad de obtención:**

1. `shortPercentOfFloat` de yfinance (mejor opción)
2. `shortRatio` * 0.05 (estimación)
3. Default 0.15 (15%)

---

## 🎯 Criterios de Setup

Un día se considera un **setup válido** si cumple:

```python
# Compresión detectada (al menos una)
is_compressed = (bb_width < 0.15) OR (compression_days >= 3)

# Estructura favorable
has_structure = (price_range_pct < 10%)

# No sobrecomprado
not_overbought = (rsi < 75)
```

### Etiquetado (Target)

```python
# Se mira 5 días hacia adelante
exploded = 1 si max(High[i+1:i+6]) >= Close[i] * (1 + explosion_threshold)
exploded = 0 si no
```

**Ejemplo:**
- Si `explosion_threshold = 0.15` (15%)
- Y el precio sube de $2.00 a $2.30+ en los próximos 5 días
- Entonces `exploded = 1`

---

## 📈 Estadísticas Esperadas

### Buen Dataset

Un dataset bien balanceado debería tener:

| Métrica | Valor Ideal | Tu Dataset |
|---------|-------------|------------|
| Total samples | 200+ | ? |
| Win rate | 30-50% | ? |
| Explosiones | 30-50% | ? |
| No explosiones | 50-70% | ? |
| Avg gain (exploded) | 20-30% | ? |
| Avg gain (not exploded) | 0-10% | ? |

### Si tienes pocos samples:

**Opciones:**

1. **Aumentar periodo:**
   ```bash
   python create_training_dataset.py --months 48
   ```

2. **Reducir threshold:**
   ```bash
   python create_training_dataset.py --explosion-threshold 0.10
   ```

3. **Agregar más símbolos:**
   - Edita la `WATCHLIST_SYMBOLS` en el script
   - Agrega penny stocks conocidos

4. **Relajar criterios:**
   - Edita `detect_setups()` línea 284
   - Cambia `bb_width < 0.15` a `bb_width < 0.20`
   - Cambia `compression_days >= 3` a `compression_days >= 2`

---

## 🔍 Validación del Dataset

### Después de Generar

```bash
python create_training_dataset.py --months 24
```

**Revisar:**

1. ✅ **Total samples > 100**: Necesitas suficientes datos
2. ✅ **Win rate 30-50%**: Balance entre explosiones y no explosiones
3. ✅ **Sin NaN**: Todas las columnas tienen valores válidos
4. ✅ **Rango correcto**: Valores dentro de rangos esperados

### Análisis Rápido con Pandas

```python
import pandas as pd

df = pd.read_csv('../data/penny_stock_training.csv')

print(f"Total samples: {len(df)}")
print(f"Win rate: {df['exploded'].mean():.1%}")
print(f"\nDescriptive stats:")
print(df.describe())

print(f"\nDistribución por símbolo:")
print(df.groupby('symbol')['exploded'].agg(['count', 'sum', 'mean']))
```

---

## 🎓 Tips para Mejorar el Dataset

### 1. Diversidad de Símbolos

**Mejor tener:**
- 50+ símbolos con 5 setups cada uno
- Que 10 símbolos con 25 setups cada uno

**Por qué:** Evita overfitting a características específicas de pocos símbolos.

### 2. Balance de Clases

**Ideal:**
- 40-60% explosiones
- 60-40% no explosiones

**Si está desbalanceado:**
```python
# En ml_model_v5.py ya usamos class_weight='balanced'
# Pero puedes ajustar el threshold de explosión
```

### 3. Periodo de Tiempo

**Recomendado:**
- Mínimo: 12 meses (1 año)
- Óptimo: 24-36 meses (2-3 años)
- Máximo: 48 meses (4 años)

**Por qué:** Captura diferentes condiciones de mercado (alcista, bajista, lateral).

### 4. Calidad sobre Cantidad

**Preferir:**
- 200 samples de alta calidad (setups reales)
- Sobre 1000 samples con ruido

**Cómo:** Mantén los criterios de setup razonablemente estrictos.

---

## 🐛 Troubleshooting

### Error: "Sin datos para SYMBOL"

**Causa:** yfinance no tiene datos para ese símbolo

**Solución:**
- Verifica que el ticker sea correcto
- Algunos símbolos antiguos (ej: BBBYQ) pueden no tener datos
- Elimina símbolos problemáticos de la watchlist

### Warning: "Solo X samples generados"

**Causa:** Criterios muy restrictivos o periodo corto

**Solución:**
```bash
# Opción 1: Más tiempo
python create_training_dataset.py --months 36

# Opción 2: Threshold más bajo
python create_training_dataset.py --explosion-threshold 0.10

# Opción 3: Más símbolos (edita el script)
```

### Error: "Columnas faltantes"

**Causa:** yfinance retornó datos incompletos

**Solución:** El script ahora valida automáticamente y salta símbolos con datos incompletos.

### Dataset muy desbalanceado

**Ejemplo:** 90% explosiones o 90% no explosiones

**Causa:** Threshold muy bajo/alto o mercado muy alcista/bajista

**Solución:**
```bash
# Si 90% explotan (threshold muy bajo):
python create_training_dataset.py --explosion-threshold 0.20

# Si 10% explotan (threshold muy alto):
python create_training_dataset.py --explosion-threshold 0.10
```

---

## 📚 Siguiente Paso: Entrenar el Modelo

Una vez tengas un dataset con 100+ samples:

```python
from utils.ml_model_v5 import BreakoutPredictor
import pandas as pd

# Cargar dataset
df = pd.read_csv('data/penny_stock_training.csv')

# Eliminar columnas no usadas por el modelo
df = df.drop(['symbol', 'date', 'gain_pct'], axis=1)

# Entrenar
predictor = BreakoutPredictor()
metrics = predictor.train(df)

print(f"Accuracy: {metrics['accuracy']:.2%}")
print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
```

Ver `INTEGRATION_GUIDE_V5.md` para más detalles sobre el entrenamiento del modelo.

---

## 🎯 Checklist Final

Antes de usar el dataset para entrenar:

- [ ] Total samples > 100
- [ ] Win rate entre 30-60%
- [ ] Al menos 20 símbolos diferentes
- [ ] Sin valores NaN
- [ ] Todas las columnas presentes
- [ ] Short interest real (no todos 0.15)
- [ ] Compression_days varía (no todos 0)
- [ ] Distribución razonable de features

**Si todo está ✅, tu dataset está listo para entrenar el modelo ML!**

---

Generated by Claude Code - Dataset Creation Guide
Version: 5.0
Date: 2025-10-24
