# Backtest Engine V2 - Sistema de Backtesting Riguroso

## 📋 Descripción

Sistema profesional de backtesting diseñado para **evitar sesgos estadísticos** y proporcionar resultados realistas y confiables. Implementa las mejores prácticas de ingeniería cuantitativa para validar estrategias de trading.

## 🎯 Objetivos Principales

### Evita los 3 Sesgos Críticos:

1. **Look-ahead Bias**: NO usa datos futuros
   - Features calculados solo con datos disponibles hasta el momento
   - Ejecución de trades al día siguiente (realista)

2. **Survivorship Bias**: NO solo stocks exitosos
   - Framework preparado para incluir stocks delistados
   - Universo amplio y diversificado

3. **Overfitting**: NO sobreoptimización
   - Temporal splits sin shuffle (train/val/test)
   - Walk-forward analysis con re-entrenamiento
   - Validación train vs test

## 🏗️ Arquitectura del Sistema

### 1. Data Management

```
┌─────────────┬──────────┬─────────┐
│   TRAIN     │   VAL    │  TEST   │
│  (60%)      │  (20%)   │  (20%)  │
└─────────────┴──────────┴─────────┘
```

**Funciones:**
- `get_universe()`: Obtiene universo de penny stocks
- `temporal_split()`: Split temporal SIN shuffle
- `walk_forward_split()`: Ventanas walk-forward
- `calculate_features_realtime()`: Features sin look-ahead bias

### 2. Walk-Forward Analysis

```
[Train: 180 días] → [Test: 30 días] → Rodar ventana →
                    [Train: 180 días] → [Test: 30 días] →
```

Previene usar información futura mediante re-entrenamiento periódico.

### 3. Position Management

**Clases:**
- `Position`: Representa posición abierta
- `Trade`: Representa trade cerrado
- `BacktestConfig`: Configuración del backtest

**Funcionalidades:**
- Kelly Criterion para sizing óptimo
- Stop-loss dinámico
- Take-profit por niveles
- Trailing stop
- Gestión de capital realista

### 4. Transaction Costs (REALISTA)

```python
commission = 0.001  # 0.1% por trade
slippage = 0.002    # 0.2% (penny stocks tienen spread alto)

entry_price = open_price * (1 + slippage)
exit_price = close_price * (1 - slippage)
```

### 5. Performance Metrics (COMPLETAS)

**Returns:**
- Total Return (%)
- CAGR (Compound Annual Growth Rate)
- Final Capital vs Initial

**Risk-Adjusted:**
- Sharpe Ratio (risk-adjusted return)
- Sortino Ratio (downside risk)
- Calmar Ratio (CAGR / Max DD)

**Risk:**
- Max Drawdown (%)
- Max Drawdown Duration
- Recovery Time
- Volatility

**Trade Stats:**
- Win Rate (%)
- Profit Factor
- Expectancy
- Average Win/Loss
- Holding Days
- Consecutive Wins/Losses

### 6. Monte Carlo Simulation

Simula 1000+ secuencias aleatorias de trades para:
- Calcular distribución de resultados
- Percentiles de retorno (P5, P25, P50, P75, P95)
- Probabilidad de ruina
- Validar robustez de la estrategia

### 7. Statistical Validation

**Tests implementados:**
- T-test vs Benchmark (SPY)
- Information Ratio
- Beta (sensibilidad al mercado)
- Alpha (excess return ajustado)
- P-value para significancia estadística

### 8. Visualización Avanzada

**Tearsheet completo incluye:**
1. Equity Curve con drawdowns
2. Underwater Plot
3. Distribution of Returns
4. Monthly Returns Heatmap
5. Trade Duration Distribution
6. Monte Carlo Percentiles
7. Key Metrics Table

## 📦 Instalación y Uso

### Requisitos

```bash
pip3 install numpy pandas matplotlib seaborn scipy yfinance scikit-learn
```

### Uso Básico

```python
from backtest_engine_v2 import BacktestEngineV2, BacktestConfig

# 1. Configurar backtest
config = BacktestConfig(
    initial_capital=100000,
    position_size_pct=0.02,  # 2% por trade
    max_positions=10,
    commission=0.001,         # 0.1%
    slippage=0.002,          # 0.2%
    stop_loss_pct=0.15,      # 15%
    take_profit_pct=0.20     # 20%
)

# 2. Crear engine
engine = BacktestEngineV2(config)

# 3. Abrir posición
position = engine.open_position(
    symbol='SNDL',
    entry_price=1.50,
    entry_date=datetime.now()
)

# 4. Actualizar posiciones (cada día)
engine.update_positions(
    current_date=datetime.now(),
    current_prices={'SNDL': 1.75}
)

# 5. Calcular métricas
from backtest_engine_v2 import calculate_all_metrics

metrics = calculate_all_metrics(
    engine.closed_trades,
    engine.equity_curve,
    config,
    period_days=365
)
```

### Ejemplo Completo

Ver `example_backtest_v2.py` para ejemplos completos de uso:

```bash
python example_backtest_v2.py
```

## 📊 Output Esperado

El sistema genera:

1. **data/backtest_trades_v2.csv**
   - Detalle de cada trade
   - Entry/exit dates y prices
   - P&L neto y porcentual
   - Holding days, exit reason

2. **data/equity_curve_v2.csv**
   - Curva de capital día a día
   - Para análisis temporal

3. **data/metrics_v2.json**
   - Todas las métricas agregadas
   - Formato JSON estructurado

4. **reports/tearsheet_YYYYMMDD_HHMMSS.png**
   - Reporte visual completo
   - 8 gráficos en alta resolución

## ⚠️ Validaciones Críticas

El sistema imprime automáticamente:

```
⚠️  VALIDACIONES CRÍTICAS DEL BACKTEST
================================================================================

1. ¿El test set es out-of-sample? → YES

2. ¿Sharpe ratio > 1.0? → YES/NO
   Sharpe: X.XX

3. ¿Outperformance vs SPY es estadísticamente significativo? → YES/NO
   p-value: 0.XXXX
   Alpha anualizado: X.XX%

4. ¿Max Drawdown < 30%? → YES/NO
   Max DD: XX.XX%

5. ¿Train/Test gap < 20%? → YES/NO (detecta overfitting)
   Train return: XX.XX%
   Test return: XX.XX%
   Gap: X.X%

6. ¿Profit Factor > 1.5? → YES/NO
   Profit Factor: X.XX

7. ¿Win Rate razonable (40-70%)? → YES/WARNING
   Win Rate: XX.X%
```

## 🔧 Funciones Principales

### Data Management

```python
# Obtener universo
universe = get_universe(
    min_price=1.0,
    max_price=10.0,
    start_date='2020-01-01',
    end_date='2023-12-31'
)

# Temporal split
train, val, test = temporal_split(
    data,
    train_pct=0.6,
    val_pct=0.2,
    test_pct=0.2
)

# Walk-forward
windows = walk_forward_split(
    data,
    train_days=180,
    test_days=30
)

# Features sin look-ahead
features = calculate_features_realtime(
    df,
    current_idx=100,
    lookback=20
)
```

### Position Management

```python
# Sizing con Kelly Criterion
kelly_pct = kelly_criterion(
    win_rate=0.55,
    avg_win=100,
    avg_loss=50
)

# Abrir posición
position = engine.open_position(
    symbol='SNDL',
    entry_price=1.50,
    entry_date=date
)

# Cerrar posición
trade = engine.close_position(
    position,
    exit_price=1.80,
    exit_date=date,
    exit_reason='Take Profit'
)
```

### Analytics

```python
# Métricas completas
metrics = calculate_all_metrics(trades, equity_curve, config, period_days)

# Monte Carlo
mc_results = monte_carlo_simulation(trades, n_simulations=1000)

# Statistical validation
stat_val = statistical_validation(
    strategy_returns,
    benchmark_returns
)

# Tearsheet
tearsheet_path = generate_tearsheet(
    metrics,
    equity_curve,
    trades,
    mc_results,
    save_path='reports/'
)

# Validaciones críticas
print_critical_validations(metrics, stat_val, train_metrics)
```

## 🎁 Features Adicionales

### Kelly Criterion

Calcula tamaño óptimo de posición basado en win rate y avg win/loss:

```python
config = BacktestConfig(
    use_kelly_criterion=True,  # Activar Kelly
    position_size_pct=0.02     # Fallback si no hay datos
)
```

### Trailing Stop

Stop loss móvil que sigue el precio máximo:

```python
position, should_close = apply_trailing_stop(
    position,
    current_price,
    trailing_pct=0.10  # 10% trailing
)
```

### Max Adverse/Favorable Excursion

Calcula cuánto se movió el precio contra/a favor:

```python
trade.max_adverse_excursion   # MAE
trade.max_favorable_excursion  # MFE
```

## 📈 Benchmarking

Compara contra SPY automáticamente:

```python
# Obtener datos de benchmark
spy_data = fetch_benchmark_data(
    symbol='SPY',
    start_date='2020-01-01',
    end_date='2023-12-31'
)

# Validar estadísticamente
stat_val = statistical_validation(
    strategy_returns,
    spy_returns
)

print(f"Alpha anualizado: {stat_val['alpha_annualized_pct']:.2f}%")
print(f"Beta: {stat_val['beta']:.2f}")
print(f"Information Ratio: {stat_val['information_ratio']:.2f}")
```

## 🚀 Próximos Pasos

Para usar este sistema con tu estrategia:

1. **Integrar con tu lógica de señales**
   ```python
   # En lugar de usar open_position directamente,
   # generar señales con tu PennyStockAdvisorV5
   analysis = advisor.analyze_symbol_v4(...)

   if analysis['trading_decision']['action'] in ['COMPRA FUERTE', 'COMPRA MODERADA']:
       engine.open_position(...)
   ```

2. **Implementar walk-forward completo**
   ```python
   windows = walk_forward_split(data, train_days=180, test_days=30)

   for train_window, test_window in windows:
       # Re-entrenar modelo con train_window
       # Backtest en test_window
       # Agregar resultados
   ```

3. **Agregar stocks delistados**
   ```python
   # Usar base de datos con histórico completo
   # Incluir stocks que fallaron/quebraron
   # Eliminar survivorship bias
   ```

4. **Optimizar hiperparámetros**
   ```python
   # Grid search en validation set
   # Evaluar en test set (una sola vez)
   # Medir train/test gap
   ```

## 📚 Documentación Adicional

### Estructura de Archivos

```
v5/
├── backtest_engine_v2.py       # Motor principal
├── example_backtest_v2.py      # Ejemplos de uso
├── BACKTEST_V2_README.md       # Esta documentación
├── data/                       # Resultados
│   ├── backtest_trades_v2.csv
│   ├── equity_curve_v2.csv
│   └── metrics_v2.json
└── reports/                    # Visualizaciones
    └── tearsheet_*.png
```

### Referencias

- **Walk-Forward Analysis**: [Pardo (2008) - "The Evaluation and Optimization of Trading Strategies"](https://www.wiley.com/en-us/The+Evaluation+and+Optimization+of+Trading+Strategies%2C+2nd+Edition-p-9780470128015)
- **Kelly Criterion**: [Kelly (1956) - "A New Interpretation of Information Rate"](https://en.wikipedia.org/wiki/Kelly_criterion)
- **Sharpe Ratio**: [Sharpe (1994) - "The Sharpe Ratio"](https://web.stanford.edu/~wfsharpe/art/sr/sr.htm)
- **Bias-Free Backtesting**: [Bailey et al. (2014) - "The Deflated Sharpe Ratio"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)

## 🤝 Contribuciones

Este es un framework profesional diseñado para producción. Mejoras sugeridas:

1. Integración con bases de datos de stocks delistados
2. Soporte para múltiples estrategias simultáneas
3. Optimización de hiperparámetros con Optuna
4. Export a QuantStats para análisis avanzado
5. Detección automática de regimen changes
6. Backtesting multi-asset (crypto, forex, etc.)

## 📄 Licencia

Uso interno para trading algorítmico.

---

**Autor**: Quantitative Engineering Team
**Fecha**: 2025
**Versión**: 2.0

Para preguntas o sugerencias, revisar la documentación en el código fuente.
