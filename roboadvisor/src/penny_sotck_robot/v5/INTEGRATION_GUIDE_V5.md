# GUÍA RÁPIDA DE INTEGRACIÓN V5

## 🚀 Cómo Usar los Módulos V5

### 1. Inicialización Básica

```python
from penny_stock_advisor_v5 import PennyStockAdvisorV5
from integration_v5_trading_manager import TradingManagerV5

# Configurar logging primero
from logging_config_v5 import setup_logging
setup_logging(level="INFO", log_to_file=True)

# Crear instancia V5
manager = TradingManagerV5(config_preset="balanced")

# Ejecutar análisis
results, buy_signals = manager.run_full_analysis()
```

### 2. Entrenar Modelo ML (Primera Vez)

```python
from ml_model_v5 import BreakoutPredictor, create_sample_training_data
import pandas as pd

# Opción A: Usar datos de ejemplo
training_data = create_sample_training_data()

# Opción B: Cargar tu propio dataset
training_data = pd.read_csv('training_data.csv')

# Entrenar
predictor = BreakoutPredictor()
metrics = predictor.train(training_data)

print(f"Accuracy: {metrics['accuracy']:.2%}")
print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
```

### 3. Configurar Datos Alternativos

```python
from alternative_data_v5 import create_sample_local_files

# Crear archivos de ejemplo
create_sample_local_files()

# O crear tus propios archivos CSV en ./data/:
# - reddit_sentiment.csv
# - short_borrow_rates.csv
```

### 4. Backtesting

```python
from backtester_v5 import BacktesterV5
from penny_stock_advisor_v5 import PennyStockAdvisorV5

# Inicializar
advisor = PennyStockAdvisorV5(config_preset="balanced")
backtester = BacktesterV5(initial_capital=10000)

# Ejecutar backtest
symbols = ['BYND', 'COSM', 'XAIR', 'RANI']
results = backtester.run_backtest(
    symbols=symbols,
    advisor=advisor,
    start_date='2024-01-01',
    end_date='2024-10-24'
)

# Ver reporte
report = backtester.generate_report(results)
print(report)
```

---

## 🔧 Código de Ejemplo Completo

```python
#!/usr/bin/env python3
"""
Ejemplo de uso completo del sistema V5
"""

from logging_config_v5 import setup_logging, get_logger
from penny_stock_advisor_v5 import PennyStockAdvisorV5
from integration_v5_trading_manager import TradingManagerV5
from ml_model_v5 import BreakoutPredictor
from alternative_data_v5 import AlternativeDataProvider, create_sample_local_files

def main():
    # 1. Configurar logging
    setup_logging(level="INFO", log_to_file=True)
    logger = get_logger('main')

    logger.info("="*70)
    logger.info("PENNY STOCK ROBOT V5 - FULL SYSTEM")
    logger.info("="*70)

    # 2. Preparar datos alternativos (primera vez)
    logger.info("Preparando datos alternativos...")
    create_sample_local_files()

    # 3. Entrenar modelo ML (si no existe)
    logger.info("Verificando modelo ML...")
    predictor = BreakoutPredictor()
    if not predictor.is_trained:
        logger.info("Entrenando modelo ML...")
        from ml_model_v5 import create_sample_training_data
        training_data = create_sample_training_data()

        # Expandir dataset
        import pandas as pd
        import numpy as np
        expanded = []
        for _ in range(20):
            for _, row in training_data.iterrows():
                new_row = row.copy()
                for col in training_data.columns[2:-1]:  # Skip symbol, date, exploded
                    if pd.api.types.is_numeric_dtype(training_data[col]):
                        new_row[col] *= (1 + np.random.uniform(-0.1, 0.1))
                expanded.append(new_row)

        training_df = pd.DataFrame(expanded)
        metrics = predictor.train(training_df)
        logger.info(f"Modelo entrenado - Accuracy: {metrics['accuracy']:.2%}")

    # 4. Crear manager y ejecutar análisis
    logger.info("Inicializando Trading Manager V5...")
    manager = TradingManagerV5(config_preset="balanced")

    logger.info("Ejecutando análisis completo...")
    results, buy_signals = manager.run_full_analysis()

    # 5. Mostrar resultados
    logger.info(f"\n{'='*70}")
    logger.info(f"RESULTADOS")
    logger.info(f"{'='*70}")
    logger.info(f"Total analizado: {len(results)} símbolos")
    logger.info(f"Señales de compra: {len(buy_signals)}")

    if buy_signals:
        logger.info(f"\nTop 3 oportunidades:")
        for i, signal in enumerate(buy_signals[:3], 1):
            logger.info(f"{i}. {signal['symbol']}: Score {signal['final_score']:.0f}/100")

    logger.info(f"\n{'='*70}")
    logger.info("Análisis completado exitosamente")
    logger.info(f"{'='*70}")

if __name__ == "__main__":
    main()
```

---

## 📊 Formato de Salida

### Señal de Compra V5

```python
{
    'symbol': 'BYND',
    'action': 'COMPRA FUERTE',
    'score': 85,
    'raw_score': 90,
    'penalty_applied': -5,

    # Scores por fase (normalizados 0-100)
    'phase1_score': 38,  # de 40 max
    'phase2_score': 36,  # de 40 max
    'phase3_score': 16,  # de 20 max

    # ML prediction
    'ml_prediction': {
        'probability': 0.85,
        'confidence': 'high',
        'model_available': True
    },

    # Alternative data
    'alternative_data_score': 75,

    # Trading plan
    'entry_price': 2.15,
    'stop_loss': 1.98,
    'take_profit_1': 2.47,  # +15%
    'take_profit_2': 2.80,  # +30%
    'position_size_pct': 3,

    # Exit management
    'trailing_stop_trigger': 2.47,
    'divergence_check': True,

    # Warnings
    'warnings': []
}
```

---

## 🎯 Checklist de Implementación

### Pre-requisitos
- [ ] Python 3.8+
- [ ] Dependencias instaladas: `yfinance`, `pandas`, `numpy`, `sklearn`
- [ ] Directorios creados: `./cache/`, `./logs/`, `./models/`, `./data/`

### Primera Ejecución
- [ ] Configurar logging
- [ ] Crear archivos de datos alternativos
- [ ] Entrenar modelo ML con dataset
- [ ] Verificar caché funcionando
- [ ] Ejecutar análisis de prueba

### Operación Continua
- [ ] Actualizar datos alternativos regularmente
- [ ] Re-entrenar modelo ML con nuevos datos
- [ ] Monitorear optimizer y ajustes de thresholds
- [ ] Revisar logs para debugging
- [ ] Limpiar caché periódicamente

---

## ⚙️ Configuración Avanzada

### Ajustar Agresividad

```python
# Conservador (menos trades, mayor calidad)
manager = TradingManagerV5(config_preset="conservative")

# Balanceado (default)
manager = TradingManagerV5(config_preset="balanced")

# Agresivo (más trades, menor calidad)
manager = TradingManagerV5(config_preset="aggressive")
```

### Configurar Optimizer

```python
from optimizer_v5 import DynamicOptimizer

# Optimizer más reactivo
optimizer = DynamicOptimizer(
    window_size=10,  # Ventana más pequeña
    recalibration_frequency=5  # Recalibrar más frecuente
)

# Optimizer más conservador
optimizer = DynamicOptimizer(
    window_size=50,  # Ventana más grande
    recalibration_frequency=20  # Recalibrar menos frecuente
)
```

### Configurar Caché

```python
from market_data_cache_v5 import MarketDataCache

# Caché agresivo (guarda todo)
cache = MarketDataCache(
    cache_dir="./cache",
    enable_persistence=True
)

# Caché en memoria solamente
cache = MarketDataCache(
    enable_persistence=False
)

# Limpiar caché viejo
cache.clear_cache()  # Todo
cache.clear_cache('BYND')  # Solo un símbolo
```

---

## 🐛 Troubleshooting

### Problema: Modelo ML no encuentra archivo
**Solución:** Entrenar modelo primero o verificar path
```python
predictor = BreakoutPredictor(model_path="./models/breakout_model.pkl")
```

### Problema: Datos alternativos no cargan
**Solución:** Crear archivos CSV o usar defaults
```python
provider = AlternativeDataProvider(use_api=False, local_data_path="./data")
# Verifica que existan: reddit_sentiment.csv y short_borrow_rates.csv
```

### Problema: Caché crece mucho
**Solución:** Limpiar periódicamente
```python
cache.clear_cache()  # Limpia todo
```

### Problema: Logs muy grandes
**Solución:** Los logs rotan automáticamente (10MB, 5 backups)
- Cambiar en `logging_config_v5.py`: `maxBytes` y `backupCount`

---

## 📈 Performance Tips

1. **Usar caché agresivamente** - Reduce llamadas a yfinance
2. **Entrenar ML con datos reales** - Mejor accuracy
3. **Actualizar datos alternativos** - Reddit y short rates frescos
4. **Monitorear optimizer** - Ajustar thresholds según performance
5. **Backtesting regular** - Validar antes de trading real

---

## 🔗 Referencias Rápidas

- Logging: `setup_logging(level="DEBUG")` para más detalle
- Caché: `cache.get_cache_stats()` para ver uso
- ML: `predictor.predict(features)` retorna probabilidad
- Optimizer: `optimizer.get_metrics()` para ver performance
- Divergencias: `detector.detect_all_divergences()` para exits

---

Generated by Claude Code - V5 Integration Guide
