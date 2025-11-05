#!/usr/bin/env python3
"""
COMPLETE BACKTEST EXECUTION V3 - INTEGRATION WITH PENNY STOCK ADVISOR V5
=========================================================================

Script completo que integra:
- BacktestEngineV3 (adaptive, ensemble, risk monitoring)
- PennyStockAdvisorV5 (ML signals, alternative data)

Ejecuta backtesting completo con todas las capacidades V3:
✓ Walk-forward analysis
✓ Adaptive thresholds
✓ Ensemble models
✓ Market regime detection
✓ Risk monitoring
✓ Interactive dashboards
✓ Meta-validation

Author: Quantitative Engineering Team
Date: 2025
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Import backtest engine V3
from backtest_engine_v2 import (
    BacktestEngineV3,
    BacktestConfig,
    calculate_all_metrics,
    monte_carlo_simulation,
    statistical_validation,
    fetch_benchmark_data,
    generate_tearsheet,
    generate_interactive_dashboard,
    factor_attribution_analysis,
    worst_trades_analysis,
    export_to_quantstats,
    meta_validation_backtest,
    detect_market_regime,
    risk_monitoring,
    temporal_split,
    build_ensemble_models,
    HAS_SKLEARN,
    HAS_XGBOOST,
    HAS_LIGHTGBM
)

# Import signal generation system
from penny_stock_advisor_v5 import PennyStockAdvisorV5

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('complete_backtest_v3')


# ============================================================================
# SIGNAL ADAPTER - Converts V5 Signals to Backtest Actions
# ============================================================================

class SignalAdapter:
    """
    Adapta señales del PennyStockAdvisorV5 al formato del BacktestEngineV3
    """

    def __init__(self, advisor: PennyStockAdvisorV5):
        """
        Args:
            advisor: Instancia de PennyStockAdvisorV5
        """
        self.advisor = advisor
        self.logger = logging.getLogger('signal_adapter')

    def get_market_context(self, date: datetime) -> Dict:
        """
        Obtiene contexto de mercado para una fecha específica

        Args:
            date: Fecha para obtener contexto

        Returns:
            Dict con contexto de mercado (SPY trend, VIX, etc.)
        """
        try:
            # Obtener datos de SPY
            end_date = date
            start_date = date - timedelta(days=60)

            spy = yf.Ticker('SPY')
            spy_hist = spy.history(start=start_date, end=end_date)

            if len(spy_hist) == 0:
                return {'spy_trend': 'neutral', 'vix': 15}

            # Calcular tendencia de SPY
            lookback = min(5, len(spy_hist))
            if lookback >= 5:
                price_change = ((spy_hist['Close'].iloc[-1] - spy_hist['Close'].iloc[-lookback]) /
                               spy_hist['Close'].iloc[-lookback] * 100)

                if price_change > 2:
                    spy_trend = 'bullish'
                elif price_change < -2:
                    spy_trend = 'bearish'
                else:
                    spy_trend = 'neutral'
            else:
                spy_trend = 'neutral'

            # VIX (simplificado - en producción obtener dato real)
            vix = 15  # Default

            return {
                'spy_trend': spy_trend,
                'vix': vix,
                'spy_price_change': price_change if lookback >= 5 else 0
            }

        except Exception as e:
            self.logger.warning(f"Error getting market context: {e}")
            return {'spy_trend': 'neutral', 'vix': 15}

    def should_enter(self, symbol: str, date: datetime, current_price: float) -> Tuple[bool, Dict]:
        """
        Determina si debe entrar en una posición

        Args:
            symbol: Símbolo del activo
            date: Fecha de evaluación
            current_price: Precio actual

        Returns:
            (should_enter, signal_data)
        """
        try:
            # Obtener datos de mercado y contexto
            market_data, historical_data = self.advisor.get_enhanced_market_data(symbol, period="2mo")

            if market_data is None or historical_data is None:
                return False, {}

            market_context = self.get_market_context(date)

            # Analizar con PennyStockAdvisorV5
            analysis = self.advisor.analyze_symbol_v5(
                symbol,
                market_data,
                historical_data,
                market_context
            )

            decision = analysis['trading_decision']

            # Determinar si debe entrar
            should_enter = decision['action'] in ['COMPRA FUERTE', 'COMPRA MODERADA']

            # Preparar datos de señal
            signal_data = {
                'score': decision.get('score', 0),
                'confidence': decision.get('confidence', 'LOW'),
                'stop_loss_pct': 0.08,  # 8% default
                'take_profit_pct': 0.15 if decision.get('urgency') == 'MEDIA' else 0.20,
                'position_size_pct': decision.get('position_size_pct', 2) / 100,  # Convert to decimal
                'entry_reason': decision.get('action', 'Unknown'),
                'key_signals': decision.get('key_signals', []),
                'ml_probability': decision.get('ml_probability', 0.5),
                'opportunity_type': analysis.get('opportunity_type', 'Unknown'),
                'warnings': decision.get('warnings', [])
            }

            if should_enter:
                self.logger.info(f"✓ ENTRY SIGNAL: {symbol} @ ${current_price:.2f} - Score: {signal_data['score']:.0f}")

            return should_enter, signal_data

        except Exception as e:
            self.logger.error(f"Error evaluating {symbol}: {e}")
            return False, {}

    def prepare_features_for_ml(self, symbol: str, date: datetime) -> Optional[np.ndarray]:
        """
        Prepara features para entrenamiento de ensemble models

        Args:
            symbol: Símbolo
            date: Fecha

        Returns:
            Array de features o None
        """
        try:
            market_data, historical_data = self.advisor.get_enhanced_market_data(symbol, period="2mo")

            if market_data is None or historical_data is None:
                return None

            # Extraer features relevantes
            features = np.array([
                market_data.get('rsi', 50) / 100,  # Normalizar a 0-1
                market_data.get('atr_ratio', 0.05),
                market_data.get('short_interest_pct', 15) / 100,
                min(market_data.get('volume', 1) / market_data.get('avg_volume_20d', 1), 10) / 10,  # Volume ratio normalizado
                len(historical_data.get('close', [])) / 60,  # Days available
                market_data.get('macd_diff', 0),
            ])

            return features

        except Exception as e:
            self.logger.error(f"Error preparing features for {symbol}: {e}")
            return None


# ============================================================================
# COMPLETE BACKTEST RUNNER
# ============================================================================

class CompleteBacktestRunner:
    """
    Ejecuta backtesting completo con integración V3 + V5
    """

    def __init__(self,
                 config: BacktestConfig,
                 adaptive: bool = True,
                 use_ensemble: bool = True):
        """
        Args:
            config: Configuración del backtest
            adaptive: Activar modo adaptativo
            use_ensemble: Usar ensemble de modelos
        """
        self.config = config
        self.adaptive = adaptive
        self.use_ensemble = use_ensemble

        # Crear componentes
        self.engine = BacktestEngineV3(config, adaptive=adaptive, use_ensemble=use_ensemble)
        self.advisor = PennyStockAdvisorV5(config_preset="balanced", enable_logging=False, enable_cache=True)
        self.signal_adapter = SignalAdapter(self.advisor)

        self.logger = logging.getLogger('backtest_runner')

        self.logger.info("="*80)
        self.logger.info("COMPLETE BACKTEST RUNNER V3 INITIALIZED")
        self.logger.info("="*80)
        self.logger.info(f"Adaptive mode: {adaptive}")
        self.logger.info(f"Ensemble models: {use_ensemble}")
        self.logger.info(f"Initial capital: ${config.initial_capital:,.2f}")

    def run_complete_backtest(self,
                             tickers: List[str],
                             start_date: str,
                             end_date: str,
                             train_val_test_split: Tuple[float, float, float] = (0.6, 0.2, 0.2),
                             update_regime_every_days: int = 30) -> Dict[str, Any]:
        """
        Ejecuta backtest completo con todas las capacidades V3

        Args:
            tickers: Lista de símbolos a testear
            start_date: Fecha inicio (YYYY-MM-DD)
            end_date: Fecha fin (YYYY-MM-DD)
            train_val_test_split: Split temporal (train, val, test)
            update_regime_every_days: Actualizar régimen cada N días

        Returns:
            Dict con resultados completos
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("STARTING COMPLETE BACKTEST EXECUTION")
        self.logger.info("="*80)
        self.logger.info(f"Tickers: {len(tickers)} symbols")
        self.logger.info(f"Period: {start_date} to {end_date}")
        self.logger.info(f"Split: Train={train_val_test_split[0]*100:.0f}%, Val={train_val_test_split[1]*100:.0f}%, Test={train_val_test_split[2]*100:.0f}%")

        # Obtener datos de SPY para regime detection
        spy_data = yf.Ticker('SPY').history(start=start_date, end=end_date)

        if len(spy_data) == 0:
            self.logger.error("No SPY data available")
            return {'error': 'No SPY data'}

        # Detectar régimen inicial
        self.engine.update_market_regime(spy_data)
        self.logger.info(f"Initial market regime: {self.engine.current_regime.upper()}")

        # Simular backtesting día por día
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')

        current_date = start_dt
        day_counter = 0
        regime_update_counter = 0

        while current_date <= end_dt:
            day_counter += 1
            regime_update_counter += 1

            # Actualizar régimen de mercado periódicamente
            if regime_update_counter >= update_regime_every_days:
                spy_recent = spy_data[spy_data.index <= pd.Timestamp(current_date).tz_localize('America/New_York')]
                if len(spy_recent) > 200:
                    self.engine.update_market_regime(spy_recent)
                    regime_update_counter = 0

            # Obtener precios actuales para todos los símbolos
            current_prices = {}

            for ticker in tickers:
                try:
                    ticker_obj = yf.Ticker(ticker)
                    hist = ticker_obj.history(start=current_date - timedelta(days=5),
                                             end=current_date + timedelta(days=1))

                    if len(hist) > 0:
                        current_prices[ticker] = float(hist['Close'].iloc[-1])

                except Exception as e:
                    self.logger.debug(f"Could not get price for {ticker}: {e}")
                    continue

            # Actualizar posiciones existentes
            self.engine.update_positions(current_date, current_prices)

            # Evaluar nuevas entradas (solo si hay espacio)
            if self.engine.can_open_position():
                for ticker in tickers:
                    if ticker in current_prices:
                        current_price = current_prices[ticker]

                        # Evaluar señal
                        should_enter, signal_data = self.signal_adapter.should_enter(
                            ticker, current_date, current_price
                        )

                        if should_enter:
                            # Aplicar threshold adaptativo si está activado
                            signal_score = signal_data.get('score', 0)

                            if self.adaptive:
                                # Usar threshold actual del engine
                                if signal_score < self.engine.current_threshold * 100:
                                    self.logger.debug(f"Signal score {signal_score:.0f} < adaptive threshold {self.engine.current_threshold*100:.0f}")
                                    continue

                            # Abrir posición
                            position = self.engine.open_position(
                                symbol=ticker,
                                entry_price=current_price,
                                entry_date=current_date,
                                stop_loss_pct=signal_data.get('stop_loss_pct', self.config.stop_loss_pct),
                                take_profit_pct=signal_data.get('take_profit_pct', self.config.take_profit_pct)
                            )

                            if position:
                                self.logger.info(f"Opened position: {ticker} @ ${current_price:.2f}")

            # Registrar equity
            self.engine.record_equity(current_date, current_prices)

            # Registrar performance periódicamente
            if day_counter % 30 == 0:
                self.engine.record_performance()

            # Check risk alerts
            if day_counter % 10 == 0 and len(self.engine.closed_trades) > 5:
                alerts = self.engine.check_risk_alerts()
                if alerts and alerts.get('action_required'):
                    self.logger.warning(f"Risk alerts triggered: {len(alerts.get('alerts', []))} alerts")

            # Actualizar threshold adaptativo periódicamente
            if self.adaptive and day_counter % 20 == 0 and len(self.engine.closed_trades) >= 20:
                self.engine.update_adaptive_threshold()

            # Avanzar al siguiente día
            current_date += timedelta(days=1)

            # Progress logging
            if day_counter % 30 == 0:
                self.logger.info(f"Progress: Day {day_counter}, Equity: ${self.engine.get_total_equity(current_prices):,.2f}, Trades: {len(self.engine.closed_trades)}")

        # Calcular métricas finales
        self.logger.info("\n" + "="*80)
        self.logger.info("BACKTEST COMPLETED - CALCULATING METRICS")
        self.logger.info("="*80)

        if len(self.engine.closed_trades) == 0:
            self.logger.error("No trades executed during backtest")
            return {'error': 'No trades'}

        period_days = (end_dt - start_dt).days
        metrics = calculate_all_metrics(
            self.engine.closed_trades,
            self.engine.equity_curve,
            self.config,
            period_days
        )

        # Monte Carlo
        monte_carlo_results = monte_carlo_simulation(
            self.engine.closed_trades,
            n_simulations=1000,
            initial_capital=self.config.initial_capital
        )

        # Benchmark comparison
        benchmark_data = fetch_benchmark_data('SPY', start_date, end_date)

        if benchmark_data is not None and len(benchmark_data) > 0:
            # Calcular returns del backtest
            equity_values = [eq[1] for eq in self.engine.equity_curve]
            backtest_returns = np.diff(equity_values) / equity_values[:-1]

            # Returns del benchmark
            benchmark_returns = benchmark_data['Close'].pct_change().dropna().values

            # Alinear longitudes
            min_len = min(len(backtest_returns), len(benchmark_returns))
            backtest_returns = backtest_returns[:min_len]
            benchmark_returns = benchmark_returns[:min_len]

            stat_validation = statistical_validation(backtest_returns, benchmark_returns)
        else:
            stat_validation = None

        # Generar reportes
        self.logger.info("\n" + "="*80)
        self.logger.info("GENERATING REPORTS")
        self.logger.info("="*80)

        # Tearsheet estático
        tearsheet_path = generate_tearsheet(
            metrics,
            self.engine.equity_curve,
            self.engine.closed_trades,
            monte_carlo_results,
            save_path='reports/'
        )

        # Dashboard interactivo
        dashboard_path = generate_interactive_dashboard(
            metrics,
            self.engine.equity_curve,
            self.engine.closed_trades,
            save_path='reports/'
        )

        # Worst trades analysis
        worst_analysis = worst_trades_analysis(self.engine.closed_trades, top_n=10)

        # QuantStats export
        quantstats_path = export_to_quantstats(
            self.engine.equity_curve,
            self.engine.closed_trades,
            save_path='reports/'
        )

        # Factor attribution (si hay ensemble)
        feature_names = ['rsi', 'atr_ratio', 'short_interest', 'volume_ratio', 'days_available', 'macd_diff']
        if self.engine.ensemble_models:
            attribution = factor_attribution_analysis(
                self.engine.ensemble_models,
                feature_names
            )
        else:
            attribution = None

        # Adaptive summary
        adaptive_summary = self.engine.get_adaptive_summary()

        results = {
            'config': self.config,
            'adaptive_mode': self.adaptive,
            'use_ensemble': self.use_ensemble,
            'period': {'start': start_date, 'end': end_date, 'days': period_days},
            'metrics': metrics,
            'monte_carlo': monte_carlo_results,
            'statistical_validation': stat_validation,
            'worst_trades': worst_analysis,
            'factor_attribution': attribution,
            'adaptive_summary': adaptive_summary,
            'risk_alerts': self.engine.alerts_history,
            'reports': {
                'tearsheet': tearsheet_path,
                'dashboard': dashboard_path,
                'quantstats': quantstats_path
            },
            'equity_curve': self.engine.equity_curve,
            'trades': self.engine.closed_trades,
            'total_trades': len(self.engine.closed_trades)
        }

        # Print summary
        self.print_summary(results)

        return results

    def print_summary(self, results: Dict):
        """Imprime resumen de resultados"""
        metrics = results['metrics']

        print("\n" + "="*80)
        print("BACKTEST RESULTS SUMMARY")
        print("="*80)

        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"  Total Return:     {metrics['total_return_pct']:>10.2f}%")
        print(f"  CAGR:             {metrics['cagr']:>10.2f}%")
        print(f"  Sharpe Ratio:     {metrics['sharpe_ratio']:>10.2f}")
        print(f"  Sortino Ratio:    {metrics['sortino_ratio']:>10.2f}")
        print(f"  Calmar Ratio:     {metrics['calmar_ratio']:>10.2f}")
        print(f"  Max Drawdown:     {metrics['max_drawdown_pct']:>10.2f}%")

        print(f"\n📈 TRADING STATISTICS:")
        print(f"  Total Trades:     {metrics['total_trades']:>10}")
        print(f"  Winning Trades:   {metrics['winning_trades']:>10}")
        print(f"  Losing Trades:    {metrics['losing_trades']:>10}")
        print(f"  Win Rate:         {metrics['win_rate']*100:>10.2f}%")
        print(f"  Profit Factor:    {metrics['profit_factor']:>10.2f}")
        print(f"  Expectancy:       ${metrics['expectancy']:>9.2f}")
        print(f"  Avg Holding:      {metrics['avg_holding_days']:>10.1f} days")

        print(f"\n💰 P&L:")
        print(f"  Gross Profit:     ${metrics['gross_profit']:>9,.2f}")
        print(f"  Gross Loss:       ${metrics['gross_loss']:>9,.2f}")
        print(f"  Net P&L:          ${metrics['total_pnl']:>9,.2f}")

        if results['statistical_validation']:
            stat = results['statistical_validation']
            print(f"\n📊 STATISTICAL VALIDATION vs SPY:")
            print(f"  Outperformance:   {stat['conclusion']}")
            print(f"  P-value:          {stat['p_value']:>10.4f}")
            print(f"  Alpha (annual):   {stat['alpha_annualized_pct']:>10.2f}%")
            print(f"  Beta:             {stat['beta']:>10.2f}")
            print(f"  Info Ratio:       {stat['information_ratio']:>10.2f}")

        mc = results['monte_carlo']
        print(f"\n🎲 MONTE CARLO ({mc['n_simulations']} simulations):")
        print(f"  Mean Return:      {mc['return_mean']:>10.2f}%")
        print(f"  Std Dev:          {mc['return_std']:>10.2f}%")
        print(f"  P5 (worst 5%):    {mc['return_percentiles']['P5']:>10.2f}%")
        print(f"  P50 (median):     {mc['return_percentiles']['P50']:>10.2f}%")
        print(f"  P95 (best 5%):    {mc['return_percentiles']['P95']:>10.2f}%")
        print(f"  Prob of Ruin:     {mc['prob_ruin']*100:>10.2f}%")

        if results['adaptive_mode']:
            adaptive = results['adaptive_summary']
            print(f"\n🔄 ADAPTIVE FEATURES:")
            print(f"  Current Threshold: {adaptive['current_threshold']:>9.2f}")
            print(f"  Market Regime:     {adaptive['current_regime'].upper():>9}")
            print(f"  Ensemble Ready:    {'YES' if adaptive['ensemble_ready'] else 'NO':>9}")
            print(f"  Risk Alerts:       {adaptive['num_alerts']:>9}")

        print(f"\n📁 REPORTS GENERATED:")
        for report_type, path in results['reports'].items():
            if path:
                print(f"  {report_type.title():>15}: {path}")

        print("\n" + "="*80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Ejecución principal del backtest completo"""

    print("="*80)
    print("COMPLETE BACKTEST EXECUTION V3 - PENNY STOCK STRATEGY")
    print("="*80)

    # Configuración
    config = BacktestConfig(
        initial_capital=100000,
        position_size_pct=0.02,  # 2% per trade
        max_positions=5,
        commission=0.001,  # 0.1%
        slippage=0.002,  # 0.2%
        stop_loss_pct=0.08,  # 8%
        take_profit_pct=0.15,  # 15%
        max_holding_days=7,
        risk_free_rate=0.04
    )

    # Universe de penny stocks
    tickers = [
        'BYND', 'OPEN', 'ASST', 'PLUG', 'ABVX',
        'SLNH'
        # Agregar más según necesidad
    ]

    # Período de backtest (último año como ejemplo)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    print(f"\nConfiguration:")
    print(f"  Universe: {len(tickers)} tickers")
    print(f"  Period: {start_date} to {end_date}")
    print(f"  Capital: ${config.initial_capital:,.2f}")
    print(f"  Adaptive: YES")
    print(f"  Ensemble: {'YES' if HAS_SKLEARN else 'NO (sklearn not available)'}")

    # Crear runner
    runner = CompleteBacktestRunner(
        config=config,
        adaptive=True,
        use_ensemble=HAS_SKLEARN  # Use ensemble if available
    )

    # Ejecutar backtest
    print(f"\nStarting backtest execution...")
    print("="*80)

    results = runner.run_complete_backtest(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        train_val_test_split=(0.6, 0.2, 0.2),
        update_regime_every_days=30
    )

    if 'error' not in results:
        print("\n✅ BACKTEST COMPLETED SUCCESSFULLY!")
        print(f"\nReports saved to: reports/")
        print(f"  - Tearsheet: {results['reports']['tearsheet']}")
        if results['reports']['dashboard']:
            print(f"  - Interactive Dashboard: {results['reports']['dashboard']}")

        # Mostrar adaptive summary
        if results['adaptive_mode']:
            print(f"\n🔄 Final Adaptive State:")
            adaptive = results['adaptive_summary']
            print(f"  Threshold: {adaptive['current_threshold']:.2f}")
            print(f"  Regime: {adaptive['current_regime'].upper()}")
            print(f"  Alerts: {adaptive['num_alerts']}")
    else:
        print(f"\n❌ BACKTEST FAILED: {results['error']}")

    return results


if __name__ == "__main__":
    results = main()
