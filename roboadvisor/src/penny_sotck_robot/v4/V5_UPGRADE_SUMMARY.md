# PENNY STOCK ROBOT V4 → V5 UPGRADE SUMMARY

## ✅ Módulos Completados

### 1. **market_data_cache_v5.py**
Sistema de caché con:
- LRU cache en memoria usando `functools.lru_cache`
- Persistencia opcional con pickle
- TTL configurable por tipo de dato
- Thread-safe con locks
- Funciones helper para cálculos repetitivos

**Uso:**
```python
from market_data_cache_v5 import MarketDataCache

cache = MarketDataCache(cache_dir="./cache", enable_persistence=True)
data = cache.get_cached_data('BYND', 'historical', period='2mo', ttl_minutes=60)
```

### 2. **logging_config_v5.py**
Sistema de logging configurable:
- `logging.config.dictConfig()` para configuración centralizada
- Niveles dinámicos (DEBUG, INFO, WARNING, ERROR)
- Múltiples handlers (consola + archivo rotating)
- Context manager para cambios temporales de nivel
- Decorador @log_execution para funciones

**Uso:**
```python
from logging_config_v5 import setup_logging, get_logger

setup_logging(level="INFO", log_to_file=True)
logger = get_logger('penny_stock_v5')
logger.info("Mensaje")
```

### 3. **backtester_v5.py**
Sistema de backtesting paralelo:
- Análisis multi-símbolo con `concurrent.futures.ThreadPoolExecutor`
- Simulación de slippage y costos
- Métricas completas (win rate, profit factor, drawdown)
- Gestión de posiciones con stop loss y take profits
- Generación de reportes

**Uso:**
```python
from backtester_v5 import BacktesterV5

backtester = BacktesterV5(initial_capital=10000)
results = backtester.run_backtest(
    symbols=['BYND', 'COSM'],
    advisor=advisor_instance,
    start_date='2024-01-01',
    end_date='2024-12-31'
)
```

### 4. **ml_model_v5.py**
Modelo de Machine Learning:
- `RandomForestClassifier` de sklearn
- Feature engineering automático
- Training pipeline con train/test split
- Predicción de probabilidad de breakout
- Persistencia del modelo con pickle
- Feature importance analysis

**Dataset esperado:**
```python
# Columnas: bb_width, adx, vol_ratio, rsi, macd_diff, atr_ratio,
#           short_float, compression_days, volume_dry, price_range_pct, exploded
```

**Uso:**
```python
from ml_model_v5 import BreakoutPredictor

predictor = BreakoutPredictor()
metrics = predictor.train(training_df)
prediction = predictor.predict(features_dict)
# Returns: {'prediction': 1, 'probability': 0.85, 'confidence': 'high'}
```

### 5. **optimizer_v5.py**
Optimizador dinámico de thresholds:
- Autoajuste basado en win rate rolling
- Recalibración cada N trades
- Ajuste adaptativo según performance
- Tracking histórico con `collections.deque`

**Lógica:**
- Win rate < 40% → AUMENTAR thresholds (más selectivo)
- Win rate > 70% → DISMINUIR thresholds (capturar más oportunidades)
- Performance estable → Mantener o tender a valores base

**Uso:**
```python
from optimizer_v5 import DynamicOptimizer

optimizer = DynamicOptimizer(window_size=20, recalibration_frequency=10)
optimizer.record_trade(trade_result)
thresholds = optimizer.get_current_thresholds()
```

### 6. **alternative_data_v5.py**
Integración de datos alternativos:
- Reddit sentiment analysis (API opcional o CSV local)
- Short borrow rate tracking
- Score combinado (0-100)
- Fallback automático a datos default

**Estructura de archivos CSV:**
- `./data/reddit_sentiment.csv`: symbol, mentions, sentiment_score, sentiment, trending
- `./data/short_borrow_rates.csv`: symbol, borrow_rate_pct, availability

**Uso:**
```python
from alternative_data_v5 import AlternativeDataProvider

provider = AlternativeDataProvider(use_api=False, local_data_path="./data")
reddit = provider.get_reddit_sentiment('BYND')
short_rate = provider.get_short_borrow_rate('BYND')
combined = provider.get_combined_alternative_data('BYND')
```

### 7. **divergence_detector_v5.py**
Detección de divergencias RSI/MACD:
- Divergencias bajistas RSI (precio sube, RSI baja)
- Divergencias bajistas MACD
- Detección automática de picos con numpy
- Clasificación por fuerza (weak/moderate/strong)
- Recomendaciones de salida

**Uso:**
```python
from divergence_detector_v5 import DivergenceDetector

detector = DivergenceDetector(lookback_window=10)
result = detector.detect_all_divergences(price_history, rsi_history, macd_history)
# Returns: rsi_divergence, macd_divergence, recommendation
```

---

## 📋 Cambios Clave para V5

### Mejoras Implementadas

| Mejora | Implementación | Archivo |
|--------|---------------|---------|
| Caché de datos | MarketDataCache con lru_cache + pickle | `market_data_cache_v5.py` |
| Logging configurable | logging.config.dictConfig() | `logging_config_v5.py` |
| Score normalizado | Todas las fases escaladas 0-100 | Pendiente integrar |
| ATR en compresión | Parámetro atr_ratio < 0.02 | Pendiente integrar |
| Backtesting multi-símbolo | concurrent.futures | `backtester_v5.py` |
| ML supervisado | RandomForestClassifier | `ml_model_v5.py` |
| Autoajuste dinámico | Recalibración thresholds | `optimizer_v5.py` |
| Datos alternativos | Reddit + short rate | `alternative_data_v5.py` |
| Divergencias RSI/MACD | Detección automática | `divergence_detector_v5.py` |

---

## 🔧 Pasos para Completar la Integración

### Paso 1: Actualizar `penny_stock_advisor_v4.py` → V5

**Cambios necesarios:**

1. **Importar nuevos módulos:**
```python
from logging_config_v5 import setup_logging, get_logger
from market_data_cache_v5 import MarketDataCache
from ml_model_v5 import BreakoutPredictor
from alternative_data_v5 import AlternativeDataProvider
from divergence_detector_v5 import DivergenceDetector, calculate_macd
from optimizer_v5 import DynamicOptimizer
```

2. **En `__init__`:**
```python
def __init__(self, config_preset="balanced"):
    # Inicializar logging
    setup_logging(level="INFO", log_to_file=True)
    self.logger = get_logger('penny_stock_v5')

    # Inicializar cache
    self.data_cache = MarketDataCache(enable_persistence=True)

    # Inicializar ML model
    self.ml_predictor = BreakoutPredictor()

    # Inicializar alternative data
    self.alt_data = AlternativeDataProvider(use_api=False)

    # Inicializar divergence detector
    self.divergence_detector = DivergenceDetector()

    # Inicializar optimizer
    self.optimizer = DynamicOptimizer()
```

3. **Agregar ATR ratio a `detect_setup_compression`:**
```python
def detect_setup_compression(self, price_history, volume_history, avg_volume_20d, atr):
    # ... código existente ...

    # NUEVO: ATR ratio check
    current_price = price_history[-1]
    atr_ratio = atr / current_price if current_price > 0 else 0

    if atr_ratio < 0.02:  # ATR muy bajo = compresión extrema
        compression_score += 5
        signals.append(f"ATR ratio bajo: {atr_ratio:.4f} (compresión extrema)")
```

4. **Integrar ML en análisis:**
```python
def analyze_symbol_v5(self, symbol, market_data, historical_data, market_context):
    # ... después de calcular phase1, phase2, phase3 ...

    # NUEVO: Predicción ML
    ml_features = self._extract_ml_features(market_data, historical_data)
    ml_prediction = self.ml_predictor.predict(ml_features)

    # Ajustar score final con ML
    if ml_prediction['probability'] < 0.3:
        penalties['total_penalty'] -= 20  # Penalizar si ML dice baja probabilidad
```

5. **Integrar datos alternativos:**
```python
def calculate_phase3_context_score(self, market_context, symbol):
    # ... código existente de mercado ...

    # NUEVO: Alternative data
    alt_data = self.alt_data.get_combined_alternative_data(symbol)
    alt_score = alt_data['combined_score'] * 0.20  # 20% del score de fase 3

    total_score += alt_score
```

6. **Normalizar scores (0-100):**
```python
# En cada fase, asegurar que el score está normalizado 0-100
def calculate_phase1_setup_score(...):
    # ... calcular total_score ...

    # Normalizar a 0-100
    normalized_score = (total_score / max_possible_score) * 100
    normalized_score = max(0, min(100, normalized_score))
```

7. **Usar caché en `get_enhanced_market_data`:**
```python
def get_enhanced_market_data(self, symbol, period="2mo"):
    # Intentar obtener del caché primero
    cached = self.data_cache.get_cached_data(symbol, 'historical', period)
    if cached:
        return cached

    # Si no está en caché, obtener de yfinance
    market_data, historical_data = self._fetch_from_yfinance(symbol, period)

    # Guardar en caché
    self.data_cache.set_cached_data(symbol, (market_data, historical_data), 'historical', period)

    return market_data, historical_data
```

### Paso 2: Actualizar `integration_v4_trading_manager.py` → V5

**Cambios necesarios:**

1. **Actualizar imports y usar PennyStockAdvisorV5**

2. **Integrar optimizer en el flujo:**
```python
def run_full_analysis(self):
    # ... análisis existente ...

    # NUEVO: Obtener thresholds dinámicos del optimizer
    dynamic_thresholds = self.optimizer.get_current_thresholds()
    self.robot.thresholds = dynamic_thresholds
```

3. **Integrar divergence detection en ExitManager:**
```python
def should_exit_position(self, position, current_data):
    # ... checks existentes ...

    # NUEVO: Check divergencias
    divergences = self.divergence_detector.detect_all_divergences(
        price_history, rsi_history, macd_history
    )

    if divergences['critical_exit_signal']:
        return True, 'Critical divergence detected'
```

---

## 📊 Flujo de Ejecución V5

```
1. Inicializar módulos V5
   ├─ Logging configurable
   ├─ Market data cache
   ├─ ML model (cargar o entrenar)
   ├─ Alternative data provider
   ├─ Divergence detector
   └─ Dynamic optimizer

2. Para cada símbolo:
   ├─ Obtener datos (usar caché)
   ├─ Calcular indicadores (con ATR ratio)
   ├─ FASE 1: Setup compression (normalizado 0-100)
   ├─ FASE 2: Trigger detection (normalizado 0-100)
   ├─ FASE 3: Context + Alternative data (normalizado 0-100)
   ├─ Predicción ML (ajustar score)
   ├─ Aplicar penalizaciones
   ├─ Score final con thresholds dinámicos
   └─ Generar señal BUY/WAIT/SKIP

3. Post-trade:
   ├─ Registrar resultado en optimizer
   ├─ Recalibrar thresholds si necesario
   └─ Detectar divergencias para salidas
```

---

## 🧪 Testing

Cada módulo incluye un bloque `if __name__ == "__main__"` para testing independiente:

```bash
# Test individual de cada módulo
python market_data_cache_v5.py
python logging_config_v5.py
python ml_model_v5.py
python optimizer_v5.py
python alternative_data_v5.py
python divergence_detector_v5.py
python backtester_v5.py
```

---

## 📦 Estructura de Archivos

```
v4/
├── penny_stock_advisor_v4.py         # Base V4 (existente)
├── integration_v4_trading_manager.py # Base V4 (existente)
├── penny_stock_advisor_v5.py         # NUEVO - A crear
├── integration_v5_trading_manager.py # NUEVO - A crear
├── market_data_cache_v5.py           # ✅ Completado
├── logging_config_v5.py              # ✅ Completado
├── backtester_v5.py                  # ✅ Completado
├── ml_model_v5.py                    # ✅ Completado
├── optimizer_v5.py                   # ✅ Completado
├── alternative_data_v5.py            # ✅ Completado
├── divergence_detector_v5.py         # ✅ Completado
├── cache/                            # Caché persistente
├── logs/                             # Archivos de log
├── models/                           # Modelos ML
└── data/                             # Datos alternativos (CSV)
```

---

## 🎯 Próximos Pasos

1. ✅ Crear archivos de módulos auxiliares (COMPLETADO)
2. ⏳ Crear `penny_stock_advisor_v5.py` integrando todos los módulos
3. ⏳ Crear `integration_v5_trading_manager.py` con nuevo flujo
4. ⏳ Preparar dataset de entrenamiento para ML
5. ⏳ Testing completo del sistema V5
6. ⏳ Documentación de uso

---

## 💾 Dataset de Entrenamiento Esperado

Crear `training_data.csv` con este formato:

```csv
symbol,date,bb_width,adx,vol_ratio,rsi,macd_diff,atr_ratio,short_float,compression_days,volume_dry,price_range_pct,exploded
BYND,2024-05-15,0.08,18.3,2.9,62,0.003,0.015,0.12,7,1,6.5,1
COSM,2024-05-18,0.10,22.1,1.7,48,-0.002,0.025,0.09,5,0,8.2,0
...
```

- **exploded**: 1 si el setup resultó en breakout real, 0 si no

---

## 📚 Referencias

- **Logging**: [Python logging.config](https://docs.python.org/3/library/logging.config.html)
- **LRU Cache**: [functools.lru_cache](https://docs.python.org/3/library/functools.html#functools.lru_cache)
- **Random Forest**: [sklearn.RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- **ThreadPoolExecutor**: [concurrent.futures](https://docs.python.org/3/library/concurrent.futures.html)

---

Generated by Claude Code - V5 Upgrade Project
Date: 2025-10-24
