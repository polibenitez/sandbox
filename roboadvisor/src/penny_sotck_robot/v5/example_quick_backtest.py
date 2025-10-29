#!/usr/bin/env python3
"""
QUICK BACKTEST EXAMPLE - Simplified Version
============================================

Ejemplo simplificado para probar rápidamente el sistema V3 integrado
con las señales del Penny Stock Advisor V5.

Este script es más simple y rápido que run_complete_backtest_v3.py
"""

import sys
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

from backtest_engine_v2 import (
    BacktestEngineV3,
    BacktestConfig,
    calculate_all_metrics,
    HAS_SKLEARN
)

from penny_stock_advisor_v5 import PennyStockAdvisorV5

print("="*80)
print("QUICK BACKTEST EXAMPLE - V3 + V5 Integration")
print("="*80)

# ============================================================================
# 1. CONFIGURACIÓN SIMPLE
# ============================================================================

config = BacktestConfig(
    initial_capital=10000,  # Capital más pequeño para ejemplo
    position_size_pct=0.05,  # 5% por trade
    max_positions=3,  # Máximo 3 posiciones simultáneas
    commission=0.001,
    slippage=0.002,
    stop_loss_pct=0.10,  # 10% stop loss
    take_profit_pct=0.20,  # 20% take profit
    max_holding_days=5
)

print("\n📋 Configuration:")
print(f"  Initial Capital: ${config.initial_capital:,.2f}")
print(f"  Position Size: {config.position_size_pct*100:.1f}%")
print(f"  Max Positions: {config.max_positions}")
print(f"  Stop Loss: {config.stop_loss_pct*100:.0f}%")
print(f"  Take Profit: {config.take_profit_pct*100:.0f}%")

# ============================================================================
# 2. CREAR COMPONENTES
# ============================================================================

print("\n🔧 Initializing components...")

# Backtest engine (modo simple sin adaptive)
engine = BacktestEngineV3(config, adaptive=False, use_ensemble=False)
print(f"  ✓ BacktestEngine V3 (simple mode)")

# Signal generator
advisor = PennyStockAdvisorV5(config_preset="balanced", enable_logging=False, enable_cache=False)
print(f"  ✓ PennyStockAdvisor V5")

# ============================================================================
# 3. SÍMBOLOS Y PERÍODO (Pequeño para ejemplo)
# ============================================================================

tickers = ['PLUG', 'RIOT', 'AMC']  # Tickers más activos con mejor data
start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')  # Últimos 6 meses
end_date = datetime.now().strftime('%Y-%m-%d')

print(f"\n📊 Backtest Parameters:")
print(f"  Tickers: {', '.join(tickers)}")
print(f"  Period: {start_date} to {end_date}")

# ============================================================================
# 4. EJECUTAR BACKTEST SIMPLIFICADO
# ============================================================================

print(f"\n⏳ Running simplified backtest...")
print("="*80)

current_date = datetime.strptime(start_date, '%Y-%m-%d')
end_dt = datetime.strptime(end_date, '%Y-%m-%d')

trades_opened = 0
days_processed = 0

while current_date <= end_dt:
    days_processed += 1

    # Obtener precios actuales
    current_prices = {}
    for ticker in tickers:
        try:
            ticker_obj = yf.Ticker(ticker)
            hist = ticker_obj.history(start=current_date - timedelta(days=2),
                                     end=current_date + timedelta(days=1))
            if len(hist) > 0:
                current_prices[ticker] = float(hist['Close'].iloc[-1])
        except:
            continue

    # Actualizar posiciones existentes
    engine.update_positions(current_date, current_prices)

    # Intentar abrir nuevas posiciones
    if engine.can_open_position():
        for ticker in tickers:
            if ticker in current_prices:
                current_price = current_prices[ticker]

                # Obtener señal del advisor (simplificado)
                try:
                    market_data, historical_data = advisor.get_enhanced_market_data(ticker, period="2mo")

                    if market_data and historical_data:
                        # Análisis simple
                        market_context = {'spy_trend': 'neutral', 'vix': 15}
                        analysis = advisor.analyze_symbol_v5(ticker, market_data, historical_data, market_context)
                        decision = analysis['trading_decision']

                        # Señal de compra?
                        should_buy = decision['action'] in ['COMPRA FUERTE', 'COMPRA MODERADA']

                        if should_buy:
                            position = engine.open_position(
                                symbol=ticker,
                                entry_price=current_price,
                                entry_date=current_date
                            )
                            if position:
                                trades_opened += 1
                                print(f"  ✓ Opened: {ticker} @ ${current_price:.2f} (Score: {decision.get('score', 0):.0f})")
                except Exception as e:
                    pass  # Ignorar errores en modo ejemplo

    # Registrar equity
    engine.record_equity(current_date, current_prices)

    # Avanzar día
    current_date += timedelta(days=1)

    # Progress
    if days_processed % 15 == 0:
        print(f"  Progress: Day {days_processed}, Trades: {len(engine.closed_trades)}, " +
              f"Equity: ${engine.get_total_equity(current_prices):,.2f}")

# ============================================================================
# 5. RESULTADOS
# ============================================================================

print("\n" + "="*80)
print("BACKTEST COMPLETED")
print("="*80)

if len(engine.closed_trades) == 0:
    print("\n❌ No trades were executed during the backtest period")
    print("   Possible reasons:")
    print("   - Signal criteria too strict")
    print("   - Insufficient data for the tickers")
    print("   - Market conditions not favorable")
    sys.exit(0)

# Calcular métricas
period_days = (end_dt - datetime.strptime(start_date, '%Y-%m-%d')).days
metrics = calculate_all_metrics(engine.closed_trades, engine.equity_curve, config, period_days)

# Imprimir resultados
print(f"\n📈 PERFORMANCE:")
print(f"  Total Return:    {metrics['total_return_pct']:>8.2f}%")
print(f"  Sharpe Ratio:    {metrics['sharpe_ratio']:>8.2f}")
print(f"  Max Drawdown:    {metrics['max_drawdown_pct']:>8.2f}%")

print(f"\n💼 TRADING:")
print(f"  Total Trades:    {metrics['total_trades']:>8}")
print(f"  Winning:         {metrics['winning_trades']:>8}")
print(f"  Losing:          {metrics['losing_trades']:>8}")
print(f"  Win Rate:        {metrics['win_rate']*100:>8.2f}%")
print(f"  Profit Factor:   {metrics['profit_factor']:>8.2f}")

print(f"\n💰 P&L:")
print(f"  Gross Profit:    ${metrics['gross_profit']:>8,.2f}")
print(f"  Gross Loss:      ${metrics['gross_loss']:>8,.2f}")
print(f"  Net P&L:         ${metrics['total_pnl']:>8,.2f}")

print(f"\n📊 TRADE DETAILS:")
print(f"  Avg Win:         ${metrics['avg_win']:>8,.2f}")
print(f"  Avg Loss:        ${metrics['avg_loss']:>8,.2f}")
print(f"  Expectancy:      ${metrics['expectancy']:>8,.2f}")
print(f"  Avg Holding:     {metrics['avg_holding_days']:>8.1f} days")

# Listar trades
print(f"\n📋 TRADE LOG:")
print("-" * 80)
for i, trade in enumerate(engine.closed_trades, 1):
    pnl_symbol = "✓" if trade.pnl_net > 0 else "✗"
    print(f"  {i:2}. {pnl_symbol} {trade.symbol:6} | " +
          f"Entry: {trade.entry_date.strftime('%Y-%m-%d')} @ ${trade.entry_price:.2f} | " +
          f"Exit: {trade.exit_date.strftime('%Y-%m-%d')} @ ${trade.exit_price:.2f} | " +
          f"P&L: ${trade.pnl_net:>7.2f} ({trade.pnl_pct:>+6.2f}%) | " +
          f"{trade.exit_reason}")

print("\n" + "="*80)
print("✅ Example completed successfully!")
print("="*80)
print("\nTo run the full backtest with all V3 features:")
print("  python run_complete_backtest_v3.py")
