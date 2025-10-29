#!/usr/bin/env python3
"""
BACKTESTER V5
=============
Sistema de backtesting multi-símbolo con análisis paralelo

Features:
- Análisis paralelo con concurrent.futures
- Métricas de rendimiento completas
- Simulación de trades con slippage
- Estadísticas de win rate y drawdown
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import yfinance as yf

logger = logging.getLogger('backtester')


class BacktesterV5:
    """
    Backtester multi-símbolo con ejecución paralela
    """

    def __init__(self, initial_capital: float = 10000,
                 position_size_pct: float = 0.05,
                 max_workers: int = 5):
        """
        Args:
            initial_capital: Capital inicial
            position_size_pct: % del capital por posición
            max_workers: Número de workers para paralelización
        """
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.max_workers = max_workers

        logger.info(f"Backtester V5 inicializado - Capital: ${initial_capital:,.2f}")

    def run_backtest(self, symbols: List[str],
                     advisor,
                     start_date: str,
                     end_date: str) -> Dict:
        """
        Ejecuta backtest paralelo para múltiples símbolos

        Args:
            symbols: Lista de símbolos a testear
            advisor: Instancia de PennyStockAdvisorV5
            start_date: Fecha inicio (YYYY-MM-DD)
            end_date: Fecha fin (YYYY-MM-DD)

        Returns:
            Dict con resultados del backtest
        """
        logger.info(f"Iniciando backtest: {len(symbols)} símbolos")
        logger.info(f"Periodo: {start_date} a {end_date}")

        all_trades = []
        results_by_symbol = {}

        # Ejecutar backtest en paralelo
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_symbol = {
                executor.submit(
                    self._backtest_single_symbol,
                    symbol, advisor, start_date, end_date
                ): symbol for symbol in symbols
            }

            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    trades, stats = future.result()
                    all_trades.extend(trades)
                    results_by_symbol[symbol] = stats
                    logger.info(f"✓ {symbol}: {len(trades)} trades")
                except Exception as e:
                    logger.error(f"✗ {symbol}: Error - {e}")
                    results_by_symbol[symbol] = None

        # Calcular métricas agregadas
        metrics = self._calculate_metrics(all_trades)

        return {
            'trades': all_trades,
            'results_by_symbol': results_by_symbol,
            'metrics': metrics,
            'config': {
                'initial_capital': self.initial_capital,
                'position_size_pct': self.position_size_pct,
                'period': f"{start_date} to {end_date}"
            }
        }

    def _backtest_single_symbol(self, symbol: str, advisor,
                               start_date: str, end_date: str) -> Tuple[List[Dict], Dict]:
        """
        Ejecuta backtest para un símbolo individual

        Returns:
            (trades, stats)
        """
        trades = []

        try:
            # Obtener datos históricos
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)

            if len(hist) == 0:
                return trades, {'error': 'No data'}

            # Simular trading día por día
            position = None

            for i in range(20, len(hist)):  # Necesitamos historial para indicadores
                current_date = hist.index[i]
                current_price = hist['Close'].iloc[i]

                # Preparar datos históricos hasta este día
                hist_subset = hist.iloc[:i+1]

                # Si no hay posición abierta, buscar entrada
                if position is None:
                    # Simular análisis
                    try:
                        market_data = self._prepare_market_data(hist_subset, symbol)
                        historical_data = self._prepare_historical_data(hist_subset)

                        # Contexto de mercado (simplificado para backtest)
                        market_context = {
                            'spy_trend': 'neutral',
                            'vix': 15.0,
                            'market_favorable': True,
                            'sector_sentiment': 'neutral'
                        }

                        # Analizar con advisor
                        analysis = advisor.analyze_symbol_v4(
                            symbol, market_data, historical_data, market_context
                        )

                        # ¿Señal de compra?
                        action = analysis['trading_decision']['action']
                        if action in ['COMPRA FUERTE', 'COMPRA MODERADA']:
                            # Abrir posición
                            entry_price = current_price * 1.001  # Simular slippage
                            shares = int((self.initial_capital * self.position_size_pct) / entry_price)

                            if shares > 0:
                                position = {
                                    'symbol': symbol,
                                    'entry_date': current_date,
                                    'entry_price': entry_price,
                                    'shares': shares,
                                    'stop_loss': analysis['trading_decision']['stop_loss'],
                                    'tp1': analysis['trading_decision']['take_profit_1'],
                                    'tp2': analysis['trading_decision']['take_profit_2'],
                                    'highest_price': entry_price
                                }

                    except Exception as e:
                        logger.debug(f"Error analizando {symbol} en {current_date}: {e}")
                        continue

                # Si hay posición abierta, gestionar salida
                else:
                    position['highest_price'] = max(position['highest_price'], current_price)

                    # Condiciones de salida
                    exit_triggered = False
                    exit_reason = ''
                    exit_price = current_price

                    # 1. Stop loss
                    if current_price <= position['stop_loss']:
                        exit_triggered = True
                        exit_reason = 'Stop Loss'
                        exit_price = position['stop_loss']

                    # 2. Take Profit 1
                    elif current_price >= position['tp1']:
                        exit_triggered = True
                        exit_reason = 'Take Profit 1'
                        exit_price = position['tp1']

                    # 3. Holding time máximo (7 días)
                    elif (current_date - position['entry_date']).days >= 7:
                        exit_triggered = True
                        exit_reason = 'Max Holding Time'

                    if exit_triggered:
                        # Cerrar posición
                        exit_price_adj = exit_price * 0.999  # Simular slippage
                        pnl = (exit_price_adj - position['entry_price']) * position['shares']
                        pnl_pct = ((exit_price_adj - position['entry_price']) / position['entry_price']) * 100

                        trade = {
                            'symbol': symbol,
                            'entry_date': position['entry_date'],
                            'exit_date': current_date,
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price_adj,
                            'shares': position['shares'],
                            'pnl': pnl,
                            'pnl_pct': pnl_pct,
                            'holding_days': (current_date - position['entry_date']).days,
                            'exit_reason': exit_reason,
                            'highest_price': position['highest_price']
                        }

                        trades.append(trade)
                        position = None

            # Stats para este símbolo
            stats = self._calculate_symbol_stats(trades)

        except Exception as e:
            logger.error(f"Error en backtest de {symbol}: {e}")
            stats = {'error': str(e)}

        return trades, stats

    def _prepare_market_data(self, hist: pd.DataFrame, symbol: str) -> Dict:
        """Prepara market_data para el análisis"""
        current_price = hist['Close'].iloc[-1]
        current_volume = hist['Volume'].iloc[-1]
        avg_volume = hist['Volume'].mean()

        # RSI
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_value = rsi.iloc[-1] if len(rsi) > 0 else 50

        return {
            'price': float(current_price),
            'volume': int(current_volume),
            'avg_volume_20d': int(avg_volume),
            'short_interest_pct': 15.0,  # Mock
            'days_to_cover': 2.0,
            'borrow_rate': 25.0,
            'rsi': float(rsi_value),
            'atr_14': current_price * 0.05,
            'bid_ask_spread_pct': 5.0,
            'market_depth_dollars': 10000
        }

    def _prepare_historical_data(self, hist: pd.DataFrame) -> Dict:
        """Prepara historical_data para el análisis"""
        return {
            'close': hist['Close'].values,
            'volume': hist['Volume'].values,
            'high': hist['High'].values,
            'low': hist['Low'].values
        }

    def _calculate_symbol_stats(self, trades: List[Dict]) -> Dict:
        """Calcula estadísticas para un símbolo"""
        if not trades:
            return {'total_trades': 0}

        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['pnl'] > 0])
        losing_trades = total_trades - winning_trades

        total_pnl = sum(t['pnl'] for t in trades)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0

        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl
        }

    def _calculate_metrics(self, trades: List[Dict]) -> Dict:
        """Calcula métricas agregadas del backtest"""
        if not trades:
            return {'error': 'No trades executed'}

        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['pnl'] > 0])
        losing_trades = total_trades - winning_trades

        total_pnl = sum(t['pnl'] for t in trades)
        total_pnl_pct = (total_pnl / self.initial_capital) * 100

        avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] < 0]) if losing_trades > 0 else 0

        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # Profit factor
        gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Drawdown máximo
        equity_curve = [self.initial_capital]
        for trade in trades:
            equity_curve.append(equity_curve[-1] + trade['pnl'])

        max_equity = equity_curve[0]
        max_dd = 0
        for equity in equity_curve:
            max_equity = max(max_equity, equity)
            dd = ((max_equity - equity) / max_equity) * 100
            max_dd = max(max_dd, dd)

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown_pct': max_dd,
            'final_capital': equity_curve[-1],
            'return_pct': ((equity_curve[-1] - self.initial_capital) / self.initial_capital) * 100
        }

    def generate_report(self, backtest_results: Dict) -> str:
        """Genera reporte de backtest"""
        metrics = backtest_results['metrics']

        report = "\n" + "="*70 + "\n"
        report += "BACKTEST RESULTS - V5\n"
        report += "="*70 + "\n\n"

        report += f"📊 RESUMEN DE PERFORMANCE\n"
        report += f"   Total Trades: {metrics['total_trades']}\n"
        report += f"   Win Rate: {metrics['win_rate']:.1f}%\n"
        report += f"   Total P&L: ${metrics['total_pnl']:,.2f} ({metrics['total_pnl_pct']:+.2f}%)\n"
        report += f"   Final Capital: ${metrics['final_capital']:,.2f}\n\n"

        report += f"💰 MÉTRICAS DETALLADAS\n"
        report += f"   Winning Trades: {metrics['winning_trades']}\n"
        report += f"   Losing Trades: {metrics['losing_trades']}\n"
        report += f"   Avg Win: ${metrics['avg_win']:,.2f}\n"
        report += f"   Avg Loss: ${metrics['avg_loss']:,.2f}\n"
        report += f"   Profit Factor: {metrics['profit_factor']:.2f}\n"
        report += f"   Max Drawdown: {metrics['max_drawdown_pct']:.2f}%\n\n"

        report += "="*70 + "\n"

        return report


if __name__ == "__main__":
    # Test básico
    from logging_config_v5 import setup_logging
    setup_logging(level="INFO")

    backtester = BacktesterV5(initial_capital=10000)
    logger.info("Backtester V5 listo para usar")
