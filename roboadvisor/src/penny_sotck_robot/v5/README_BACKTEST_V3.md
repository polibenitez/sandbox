# Backtest Engine V3 - Complete Integration Guide

## 🚀 Overview

Sistema completo de backtesting con capacidades adaptativas que integra:

- **BacktestEngineV3**: Motor de backtesting riguroso con features adaptativas
- **PennyStockAdvisorV5**: Sistema de generación de señales con ML y datos alternativos

### Features V3

✅ **Adaptive Thresholds** - Ajuste automático con safeguards anti-overfitting
✅ **Ensemble Models** - RF + XGBoost + LightGBM con voting
✅ **Market Regime Detection** - Bull/Bear/Choppy detection
✅ **Risk Monitoring** - Alertas automáticas de riesgo
✅ **Online Learning** - Re-training automático controlado
✅ **Interactive Dashboards** - Visualizaciones con Plotly
✅ **Meta-Validation** - Backtesting del backtest
✅ **QuantStats Integration** - Reportes profesionales

---

## 📦 Installation

### Required Dependencies

```bash
# Core dependencies (required)
pip install numpy pandas yfinance scipy matplotlib

# Optional but recommended for full features
pip install scikit-learn xgboost lightgbm plotly tqdm quantstats seaborn

# For notifications (optional)
pip install requests  # For Slack webhooks
```

### Verify Installation

```bash
python3 -c "import backtest_engine_v2; print('✓ Backtest Engine V3 loaded')"
```

---

## 🎯 Quick Start

### Example 1: Simple Backtest (Quickest)

```bash
python3 example_quick_backtest.py
```

Este script ejecuta un backtest simplificado en ~3 meses con 3 tickers. Perfecto para:
- Probar que todo funciona
- Entender el flujo básico
- Debugging rápido

**Salida esperada:**
```
📈 PERFORMANCE:
  Total Return:    +15.23%
  Sharpe Ratio:      1.45
  Max Drawdown:     -8.50%

💼 TRADING:
  Total Trades:         12
  Win Rate:          58.33%
  Profit Factor:      2.15
```

### Example 2: Complete Backtest (Full Featured)

```bash
python3 run_complete_backtest_v3.py
```

Este script ejecuta el backtest completo con TODAS las capacidades V3:
- Adaptive thresholds
- Ensemble models (si disponibles)
- Market regime detection
- Risk monitoring y alertas
- Reportes interactivos

**Duración:** ~10-30 minutos dependiendo del universo de tickers

---

## 📚 Usage Guide

### Basic Usage

```python
from backtest_engine_v2 import BacktestEngineV3, BacktestConfig
from penny_stock_advisor_v5 import PennyStockAdvisorV5

# 1. Configuración
config = BacktestConfig(
    initial_capital=100000,
    position_size_pct=0.02,  # 2% per trade
    max_positions=5,
    commission=0.001,
    slippage=0.002,
    stop_loss_pct=0.08,
    take_profit_pct=0.15
)

# 2. Crear engine (modo adaptativo)
engine = BacktestEngineV3(config, adaptive=True, use_ensemble=True)

# 3. Crear advisor para señales
advisor = PennyStockAdvisorV5(config_preset="balanced")

# 4. Ejecutar backtest...
# (Ver example scripts para código completo)
```

### Advanced: Adaptive Mode

```python
# Activar modo adaptativo completo
engine = BacktestEngineV3(config, adaptive=True, use_ensemble=True)

# Entrenar ensemble models
X_train = prepare_features(training_data)
y_train = prepare_labels(training_data)
engine.train_ensemble(X_train, y_train)

# Actualizar régimen de mercado
spy_data = yf.Ticker('SPY').history(period='1y')
engine.update_market_regime(spy_data)

# Durante el backtest:
# - Threshold se ajusta automáticamente
# - Regime detection actualiza estrategia
# - Risk monitoring genera alertas
# - Performance tracking detecta drift
```

### Advanced: Custom Signal Integration

```python
from run_complete_backtest_v3 import SignalAdapter

# Crear adapter personalizado
adapter = SignalAdapter(advisor)

# Evaluar señal para un símbolo
should_enter, signal_data = adapter.should_enter(
    symbol='SNDL',
    date=current_date,
    current_price=1.50
)

if should_enter:
    position = engine.open_position(
        symbol='SNDL',
        entry_price=1.50,
        entry_date=current_date,
        stop_loss_pct=signal_data['stop_loss_pct'],
        take_profit_pct=signal_data['take_profit_pct']
    )
```

---

## 📊 Configuration Options

### BacktestConfig Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_capital` | 100000 | Capital inicial ($) |
| `position_size_pct` | 0.02 | Tamaño de posición (% del capital) |
| `max_positions` | 10 | Máximo posiciones simultáneas |
| `commission` | 0.001 | Comisión por trade (0.1%) |
| `slippage` | 0.002 | Slippage estimado (0.2%) |
| `stop_loss_pct` | 0.15 | Stop loss (15%) |
| `take_profit_pct` | 0.20 | Take profit (20%) |
| `max_holding_days` | 30 | Máximo días en una posición |
| `risk_free_rate` | 0.04 | Tasa libre de riesgo (4% anual) |
| `use_kelly_criterion` | False | Usar Kelly para sizing |

### Engine Modes

```python
# Modo básico (sin adaptive)
engine = BacktestEngineV3(config, adaptive=False, use_ensemble=False)

# Modo adaptativo sin ensemble (si no hay sklearn)
engine = BacktestEngineV3(config, adaptive=True, use_ensemble=False)

# Modo completo (recomendado si tienes dependencias)
engine = BacktestEngineV3(config, adaptive=True, use_ensemble=True)
```

---

## 📈 Output & Reports

### Reports Generated

1. **Tearsheet (PNG)** - Visualización estática completa
   - Location: `reports/tearsheet_YYYYMMDD_HHMMSS.png`
   - Includes: Equity curve, drawdown, returns distribution, monthly heatmap

2. **Interactive Dashboard (HTML)** - Dashboard interactivo con Plotly
   - Location: `reports/dashboard_YYYYMMDD_HHMMSS.html`
   - Open in browser for interactive exploration

3. **QuantStats Report (HTML)** - Reporte profesional completo
   - Location: `reports/quantstats_report_YYYYMMDD_HHMMSS.html`
   - Compatible con QuantStats library

4. **JSON Results** - Datos completos en formato JSON (opcional)
   - All metrics, trades, and configuration

### Key Metrics Explained

#### Performance Metrics

- **Total Return %**: Return total del período
- **CAGR**: Compound Annual Growth Rate (retorno anualizado)
- **Sharpe Ratio**: Return ajustado por riesgo (>1.5 es bueno)
- **Sortino Ratio**: Similar a Sharpe pero solo considera downside
- **Calmar Ratio**: CAGR / Max Drawdown (>2.0 es bueno)
- **Max Drawdown**: Pérdida máxima desde peak (< 25% tolerable)

#### Trading Metrics

- **Win Rate**: % de trades ganadores (40-70% es razonable)
- **Profit Factor**: Gross Profit / Gross Loss (>1.5 es bueno)
- **Expectancy**: Ganancia esperada por trade
- **Avg Holding Days**: Duración promedio de trades

#### Statistical Validation

- **P-value**: Significancia del outperformance vs SPY (< 0.05 es significativo)
- **Alpha**: Excess return ajustado por riesgo vs benchmark
- **Beta**: Sensibilidad al mercado (1.0 = igual que mercado)
- **Information Ratio**: Excess return / Tracking error (>0.5 es bueno)

---

## ⚠️ Important Warnings

### Golden Principles

1. **"Un backtest que se ve demasiado bueno, probablemente lo sea"**
   - Si Win Rate > 75%, Sharpe > 3.0, Max DD < 10%
   - Probablemente hay overfitting u olvidas algo crítico

2. **Solo ajustar en VALIDATION set, NUNCA en test**
   - Test set es sacrosanto
   - Adaptive thresholds se ajustan en validation únicamente

3. **Transaction costs son CRÍTICOS**
   - Penny stocks tienen spreads altos (0.2-0.5%)
   - Slippage + Commission destruyen estrategias marginales

4. **Verificar train/val/test gap**
   - Gap > 25% indica overfitting
   - System auto-detecta y alerta

### Pre-Live Checklist

Antes de usar en trading real:

- [ ] Backtest en out-of-sample completo (min. 2 años)
- [ ] Forward test en paper trading (3 meses mínimo)
- [ ] Transaction costs REALISTAS incluidos
- [ ] Sharpe ratio > 1.5 en forward test
- [ ] Max Drawdown tolerable (< 25%)
- [ ] Train/Val/Test gap < 20% (no overfitting)
- [ ] Validación estadística vs benchmark (p-value < 0.05)
- [ ] Monte Carlo simulation ejecutado
- [ ] Risk monitoring activo
- [ ] Código revisado por otro developer
- [ ] Plan de exit si no funciona
- [ ] Capital de riesgo que puedes perder

---

## 🔧 Troubleshooting

### Common Issues

#### 1. "No trades executed"

**Causa**: Señales muy estrictas o datos insuficientes

**Solución:**
- Reducir thresholds en advisor config
- Ampliar universo de tickers
- Aumentar período de backtest
- Verificar que los tickers tienen datos disponibles

#### 2. "Ensemble models not available"

**Causa**: sklearn/xgboost/lightgbm no instalados

**Solución:**
```bash
pip install scikit-learn xgboost lightgbm
```

O usar modo sin ensemble:
```python
engine = BacktestEngineV3(config, adaptive=True, use_ensemble=False)
```

#### 3. "Interactive dashboard not generated"

**Causa**: Plotly no instalado

**Solución:**
```bash
pip install plotly
```

#### 4. Backtest muy lento

**Solución:**
- Reducir universo de tickers
- Reducir período de backtest
- Desactivar caché del advisor si causa problemas
- Usar modo simple (adaptive=False)

#### 5. Datos de yfinance incompletos

**Causa**: Penny stocks tienen datos limitados o problemas de red

**Solución:**
- Verificar conectividad
- Usar tickers con mejor histórico
- Agregar retry logic
- Usar caché del advisor

---

## 📖 Architecture Overview

```
run_complete_backtest_v3.py
├── BacktestEngineV3 (backtest_engine_v2.py)
│   ├── Position Management
│   ├── Transaction Costs
│   ├── Risk Monitoring
│   ├── Adaptive Thresholds
│   ├── Ensemble Models
│   └── Reporting
│
├── PennyStockAdvisorV5 (penny_stock_advisor_v5.py)
│   ├── Signal Generation
│   ├── ML Prediction
│   ├── Alternative Data
│   └── Scoring System
│
└── SignalAdapter
    ├── Market Context
    ├── Signal Conversion
    └── Feature Preparation
```

### Data Flow

1. **Initialization**
   - Load configuration
   - Create engine (V3) + advisor (V5)
   - Setup signal adapter

2. **Training Phase** (if adaptive)
   - Prepare training data
   - Train ensemble models
   - Set initial thresholds

3. **Backtest Loop**
   - For each day:
     - Update market regime (periodic)
     - Update existing positions (check exits)
     - Evaluate new signals (via advisor)
     - Open positions (if signal + space available)
     - Record equity
     - Check risk alerts
     - Update adaptive threshold (periodic)

4. **Reporting**
   - Calculate metrics
   - Generate visualizations
   - Export reports
   - Print summary

---

## 🎓 Examples

### Example: Monthly Rebalancing

```python
# Rebalance portfolio monthly
month_counter = 0

for date in trading_days:
    month_counter += 1

    # Monthly rebalancing
    if month_counter % 30 == 0:
        # Close all positions
        for position in engine.positions:
            engine.close_position(
                position,
                current_prices[position.symbol],
                date,
                'Monthly Rebalance'
            )

        # Re-evaluate all tickers
        for ticker in tickers:
            should_enter, signal_data = adapter.should_enter(...)
            if should_enter:
                engine.open_position(...)
```

### Example: Custom Exit Logic

```python
# Custom exit based on divergence
from penny_stock_advisor_v5 import DivergenceDetector

divergence_detector = DivergenceDetector()

for position in engine.positions:
    # Check for bearish divergence
    div_result = divergence_detector.detect_divergences(
        prices, rsi_values, macd_values
    )

    if div_result['bearish_divergence_detected']:
        # Exit immediately
        engine.close_position(
            position,
            current_price,
            current_date,
            'Bearish Divergence'
        )
```

### Example: Position Sizing with Kelly

```python
# Enable Kelly Criterion
config = BacktestConfig(
    use_kelly_criterion=True,
    position_size_pct=0.02  # Will be adjusted by Kelly
)

# Kelly will automatically adjust position sizes based on:
# - Historical win rate
# - Average win/loss ratio
# - Cap at max_size to avoid over-leverage
```

---

## 📞 Support & Contributing

### Getting Help

1. Check this README first
2. Review example scripts
3. Check logs for detailed error messages
4. Verify dependencies are installed

### Contributing

Improvements welcome! Focus areas:
- Additional signal sources
- Better regime detection (add VIX integration)
- Improved risk management rules
- More comprehensive reporting

---

## 📝 License

Proprietary - Quantitative Engineering Team © 2025

---

## 🔄 Version History

### V3.0 (Current)
- ✅ Complete adaptive system
- ✅ Ensemble models
- ✅ Risk monitoring
- ✅ Interactive dashboards
- ✅ Meta-validation
- ✅ Integration with PennyStockAdvisorV5

### V2.0
- Basic backtesting framework
- Walk-forward analysis
- Transaction costs
- Monte Carlo simulation

### V1.0
- Initial framework
- Simple position management

---

**Ready to start? Run:**

```bash
# Quick test (3 minutes)
python3 example_quick_backtest.py

# Full backtest (30 minutes)
python3 run_complete_backtest_v3.py
```

🎯 **Happy backtesting!**
