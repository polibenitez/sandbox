# 🚀 QUICK START - PENNY STOCK ROBOT V5

## Inicio Rápido en 3 Pasos

### 1. Preparar el Entorno

```bash
# Instalar dependencias
pip install yfinance pandas numpy scikit-learn tqdm

# Crear directorios necesarios
mkdir -p data cache logs models
```

### 2. Generar Dataset de Entrenamiento (Opcional pero Recomendado)

```bash
cd utils
python create_training_dataset.py --months 24
```

Esto generará `../data/penny_stock_training.csv` con datos reales.

### 3. Ejecutar el Sistema

```bash
# Volver al directorio principal
cd ..

# Ejecutar análisis completo
python integration_v5_trading_manager.py
```

---

## 📊 Uso Completo

### Opción A: Sistema Completo con Trading Manager

```python
from integration_v5_trading_manager import TradingManagerV5

# Crear instancia
manager = TradingManagerV5(
    config_preset="balanced",  # "conservative" | "balanced" | "aggressive"
    enable_backtesting=False
)

# Ejecutar análisis
results, buy_signals = manager.run_full_analysis()

# Ver oportunidades
for signal in buy_signals:
    print(f"{signal['symbol']}: Score {signal['final_score']:.0f}/100")
    print(f"  ML Probability: {signal['ml_adjustment']['probability']:.1%}")
    print(f"  Entry: ${signal['trading_decision']['current_price']:.2f}")
    print(f"  Stop: ${signal['trading_decision']['stop_loss']:.2f}")
    print()
```

### Opción B: Solo Advisor V5

```python
from penny_stock_advisor_v5 import PennyStockAdvisorV5

# Crear advisor
advisor = PennyStockAdvisorV5(config_preset="balanced")

# Analizar un símbolo
market_data, historical_data = advisor.get_enhanced_market_data('BYND')

market_context = {
    'spy_trend': 'neutral',
    'vix': 18.5,
    'market_favorable': True
}

analysis = advisor.analyze_symbol_v5(
    'BYND',
    market_data,
    historical_data,
    market_context
)

print(f"Score: {analysis['final_score']:.0f}/100")
print(f"Action: {analysis['trading_decision']['action']}")
print(f"ML Probability: {analysis['ml_adjustment']['probability']:.1%}")
```

---

## 🧠 Entrenar el Modelo ML

### Primera Vez (Recomendado)

```bash
# 1. Generar dataset con datos reales
cd utils
python create_training_dataset.py --months 36 --explosion-threshold 0.15

# 2. Entrenar modelo
python -c "
from ml_model_v5 import BreakoutPredictor
import pandas as pd

df = pd.read_csv('../data/penny_stock_training.csv')
df = df.drop(['symbol', 'date', 'gain_pct'], axis=1)

predictor = BreakoutPredictor()
metrics = predictor.train(df)

print(f'Accuracy: {metrics[\"accuracy\"]:.2%}')
print(f'ROC-AUC: {metrics[\"roc_auc\"]:.3f}')
"
```

El modelo se guarda automáticamente en `models/breakout_model.pkl` y se cargará automáticamente en futuras ejecuciones.

---

## 🧪 Backtesting

```python
from integration_v5_trading_manager import TradingManagerV5

# Crear manager con backtesting habilitado
manager = TradingManagerV5(
    config_preset="balanced",
    enable_backtesting=True
)

# Ejecutar backtest
symbols = ['BYND', 'COSM', 'XAIR', 'RANI', 'CLOV']
results = manager.run_backtest(
    symbols=symbols,
    start_date='2024-01-01',
    end_date='2024-10-24'
)

# Los resultados se guardan automáticamente en JSON
```

---

## ⚙️ Configuración de Datos Alternativos

### Opción 1: Usar Archivos de Ejemplo

```python
from utils.alternative_data_v5 import create_sample_local_files

# Crea archivos de ejemplo en data/
create_sample_local_files()
```

### Opción 2: Crear tus Propios Archivos

**data/reddit_sentiment.csv:**
```csv
symbol,mentions,sentiment_score,sentiment,trending
BYND,150,0.6,bullish,True
COSM,45,-0.2,bearish,False
```

**data/short_borrow_rates.csv:**
```csv
symbol,borrow_rate_pct,availability
BYND,35.5,hard_to_borrow
COSM,15.2,moderate
```

---

## 📋 Estructura de Salida

### Archivo JSON de Resultados

```json
{
  "version": "V5 - Evolution Edition",
  "timestamp": "2025-10-24T10:30:00",
  "market_context": {
    "spy_trend": "neutral",
    "vix": 18.5,
    "favorable": true
  },
  "buy_signals": [
    {
      "symbol": "BYND",
      "final_score": 85,
      "ml_probability": 0.85,
      "ml_confidence": "high",
      "action": "COMPRA FUERTE",
      "entry_price": 2.15,
      "stop_loss": 1.98,
      "tp1": 2.47,
      "tp2": 2.80
    }
  ]
}
```

### Log de Consola

```
======================================================================
OPORTUNIDADES DE TRADING - V5 EVOLUTION
======================================================================

✅ 3 OPORTUNIDADES DETECTADAS

──────────────────────────────────────────────────────────────────────
1. 📈 BYND - COMPRA FUERTE
──────────────────────────────────────────────────────────────────────
💯 SCORING:
   • Score final: 85/100
   • Score bruto: 90/100
   • Penalizaciones: -5 pts

🧠 MACHINE LEARNING:
   • Probabilidad breakout: 85%
   • Confianza: HIGH
   • Modelo: Disponible

📊 ANÁLISIS POR FASES:
   🔵 Fase 1 (Setup): 38/100
      • Compresión EXTREMA (ATR muy bajo): rango 4.5% en 5d, ATR ratio 0.012
      • SI cualificado: 18.3%, DTC: 2.5

   🟡 Fase 2 (Trigger): 36/100
      • Volumen explosivo TEMPRANO: 3.2x en día 1
      • Breakout limpio: precio > resistencia y SMAs

   🌍 Fase 3 (Context+Alt): 16/100
      • Mercado neutral
      • Alt Data: Reddit bullish (150 mentions), Borrow 35.5%

💰 PLAN DE TRADING:
   • Precio entrada: $2.150
   • Posición: 3% del capital
   • Stop loss: $1.978 (-8.0%)

🎯 TAKE PROFITS:
   • TP1 (30%): $2.473 (+15%)
   • TP2 (30%): $2.795 (+30%)
   • Restante (40%): Trailing stop o divergencia
```

---

## 🔧 Configuraciones Avanzadas

### Ajustar Agresividad

```python
# Conservador (menos trades, más calidad)
manager = TradingManagerV5(config_preset="conservative")

# Balanceado (default)
manager = TradingManagerV5(config_preset="balanced")

# Agresivo (más trades, menos restrictivo)
manager = TradingManagerV5(config_preset="aggressive")
```

### Desactivar Caché (para Testing)

```python
from penny_stock_advisor_v5 import PennyStockAdvisorV5

advisor = PennyStockAdvisorV5(
    config_preset="balanced",
    enable_cache=False  # Desactiva caché
)
```

### Cambiar Nivel de Logging

```python
from utils.logging_config_v5 import setup_logging

# DEBUG: Máximo detalle
setup_logging(level="DEBUG", log_to_file=True)

# INFO: Normal (default)
setup_logging(level="INFO", log_to_file=True)

# WARNING: Solo advertencias
setup_logging(level="WARNING", log_to_file=True)
```

---

## 🎯 Flujo Típico de Trabajo

### Trading Diario

```python
# 1. Ejecutar análisis matutino
from integration_v5_trading_manager import TradingManagerV5

manager = TradingManagerV5(config_preset="balanced")
results, buy_signals = manager.run_full_analysis()

# 2. Revisar oportunidades
if buy_signals:
    print(f"\n✅ {len(buy_signals)} oportunidades detectadas")
    for signal in buy_signals:
        decision = signal['trading_decision']
        print(f"\n{signal['symbol']}:")
        print(f"  Score: {signal['final_score']:.0f}/100")
        print(f"  ML Probability: {signal['ml_adjustment']['probability']:.1%}")
        print(f"  Entry: ${decision['current_price']:.2f}")
        print(f"  Stop: ${decision['stop_loss']:.2f}")
else:
    print("\n⏸️ No hay oportunidades hoy - esperando mejores setups")

# 3. Los resultados se guardan automáticamente en JSON
```

### Análisis de un Símbolo Específico

```python
from penny_stock_advisor_v5 import PennyStockAdvisorV5

advisor = PennyStockAdvisorV5()

# Analizar símbolo específico
symbol = "BYND"
market_data, historical_data = advisor.get_enhanced_market_data(symbol)

if market_data:
    market_context = {
        'spy_trend': 'neutral',
        'vix': 18.5,
        'market_favorable': True
    }

    analysis = advisor.analyze_symbol_v5(
        symbol, market_data, historical_data, market_context
    )

    print(f"\n{symbol} Analysis:")
    print(f"Score: {analysis['final_score']:.0f}/100")
    print(f"Action: {analysis['trading_decision']['action']}")
    print(f"ML Prob: {analysis['ml_adjustment']['probability']:.1%}")

    # Ver detalles por fase
    print(f"\nPhase 1 (Setup): {analysis['phase1_setup']['score']:.0f}/100")
    print(f"Phase 2 (Trigger): {analysis['phase2_trigger']['score']:.0f}/100")
    print(f"Phase 3 (Context): {analysis['phase3_context']['score']:.0f}/100")
```

---

## 📊 Monitoreo de Posiciones

```python
from utils.divergence_detector_v5 import DivergenceDetector

# Crear detector
detector = DivergenceDetector(lookback_window=10)

# Obtener datos de posición abierta
# (price_history, rsi_history, macd_history desde yfinance)

# Detectar divergencias para salida
divergences = detector.detect_all_divergences(
    price_history,
    rsi_history,
    macd_history
)

if divergences['critical_exit_signal']:
    print("⚠️ DIVERGENCIA CRÍTICA - Considerar salida")
    print(divergences['recommendation'])
```

---

## 🐛 Troubleshooting

### Problema: "Modelo ML no entrenado"

**Solución:**
```bash
cd utils
python create_training_dataset.py --months 24
python -c "from ml_model_v5 import BreakoutPredictor; import pandas as pd; \
df = pd.read_csv('../data/penny_stock_training.csv').drop(['symbol','date','gain_pct'], axis=1); \
predictor = BreakoutPredictor(); predictor.train(df)"
```

### Problema: "Alternative data no disponible"

**Solución:**
```python
from utils.alternative_data_v5 import create_sample_local_files
create_sample_local_files()  # Crea archivos de ejemplo
```

### Problema: Caché muy grande

**Solución:**
```python
from utils.market_data_cache_v5 import MarketDataCache
cache = MarketDataCache()
cache.clear_cache()  # Limpia todo el caché
```

---

## 📚 Archivos Principales

```
v4/
├── integration_v5_trading_manager.py  ← EJECUTAR ESTE
├── penny_stock_advisor_v5.py          ← Core del sistema
└── utils/
    ├── market_data_cache_v5.py
    ├── logging_config_v5.py
    ├── ml_model_v5.py
    ├── optimizer_v5.py
    ├── alternative_data_v5.py
    ├── divergence_detector_v5.py
    ├── backtester_v5.py
    └── create_training_dataset.py
```

---

## ✅ Checklist de Primera Ejecución

- [ ] Instalar dependencias: `pip install yfinance pandas numpy scikit-learn tqdm`
- [ ] Crear directorios: `mkdir -p data cache logs models`
- [ ] Generar dataset: `cd utils && python create_training_dataset.py`
- [ ] Entrenar modelo ML (ver sección "Entrenar el Modelo ML")
- [ ] Crear datos alternativos: `python -c "from utils.alternative_data_v5 import create_sample_local_files; create_sample_local_files()"`
- [ ] Ejecutar primera vez: `python integration_v5_trading_manager.py`
- [ ] Revisar logs en `logs/`
- [ ] Revisar resultados JSON generados

---

## 🎯 Próximos Pasos

1. **Acumular datos históricos**: Ejecuta `create_training_dataset.py` periódicamente para recolectar más setups
2. **Re-entrenar modelo**: Cada mes, re-entrena con nuevos datos
3. **Actualizar datos alternativos**: Mantén actualizados Reddit sentiment y short rates
4. **Monitorear optimizer**: Revisa cómo se ajustan los thresholds
5. **Backtesting regular**: Valida estrategias antes de trading real

---

**Sistema V5 listo para usar!** 🚀

Para más detalles, consulta:
- `README_V5.md` - Documentación completa
- `INTEGRATION_GUIDE_V5.md` - Guía de integración
- `V5_UPGRADE_SUMMARY.md` - Resumen de cambios
- `DATASET_CREATION_GUIDE.md` - Guía del dataset

---

Generated by Claude Code - Quick Start V5
Version: 5.0
Date: 2025-10-24
