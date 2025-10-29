#!/usr/bin/env python3
"""
OPTIMIZER V5 - AUTOAJUSTE DINÁMICO DE THRESHOLDS
=================================================

Features:
- Recalibra thresholds basados en win rate rolling
- Ajuste automático cada N trades
- Tracking de performance histórica
- Optimización adaptativa
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from collections import deque

logger = logging.getLogger('optimizer')


class DynamicOptimizer:
    """
    Optimizador dinámico que ajusta thresholds basados en performance
    """

    def __init__(self, window_size: int = 20, recalibration_frequency: int = 10):
        """
        Args:
            window_size: Ventana para calcular win rate rolling
            recalibration_frequency: Cada cuántos trades recalibrar
        """
        self.window_size = window_size
        self.recalibration_frequency = recalibration_frequency

        # Tracking de trades
        self.trade_history = deque(maxlen=window_size)
        self.trades_since_recalibration = 0

        # Thresholds actuales
        self.current_thresholds = {
            'buy_strong': 70,
            'buy_moderate': 55,
            'watchlist': 40
        }

        # Thresholds base (valores iniciales)
        self.base_thresholds = self.current_thresholds.copy()

        logger.info(f"Optimizer V5 inicializado - Window: {window_size}, Frecuencia: {recalibration_frequency}")

    def record_trade(self, trade_result: Dict):
        """
        Registra resultado de un trade

        Args:
            trade_result: Dict con info del trade
                {
                    'symbol': str,
                    'entry_date': datetime,
                    'exit_date': datetime,
                    'pnl': float,
                    'pnl_pct': float,
                    'score': float,  # Score que generó la señal
                    'action': str  # 'COMPRA FUERTE' o 'COMPRA MODERADA'
                }
        """
        self.trade_history.append(trade_result)
        self.trades_since_recalibration += 1

        logger.debug(f"Trade registrado: {trade_result['symbol']} - P&L: {trade_result['pnl_pct']:.1f}%")

        # ¿Necesitamos recalibrar?
        if self.trades_since_recalibration >= self.recalibration_frequency:
            self.recalibrate_thresholds()
            self.trades_since_recalibration = 0

    def recalibrate_thresholds(self) -> Dict:
        """
        Recalibra thresholds basados en performance reciente

        Returns:
            Dict con nuevos thresholds y métricas
        """
        if len(self.trade_history) < 5:
            logger.info("No hay suficientes trades para recalibrar")
            return {
                'thresholds': self.current_thresholds,
                'win_rate': 0,
                'avg_pnl_pct': 0,
                'adjustment': 'none'
            }

        # Calcular métricas recientes
        win_rate = self._calculate_win_rate()
        avg_pnl_pct = self._calculate_avg_pnl_pct()
        avg_winner_pct = self._calculate_avg_winner_pct()
        avg_loser_pct = self._calculate_avg_loser_pct()

        logger.info(f"Recalibrando - Win Rate: {win_rate:.1f}%, Avg P&L: {avg_pnl_pct:.1f}%")

        # Lógica de ajuste
        adjustment = 'none'
        old_thresholds = self.current_thresholds.copy()

        # CASO 1: Win rate bajo (<40%) → AUMENTAR thresholds (ser más selectivo)
        if win_rate < 40:
            adjustment = 'increase'
            self.current_thresholds['buy_strong'] = min(85, old_thresholds['buy_strong'] + 5)
            self.current_thresholds['buy_moderate'] = min(70, old_thresholds['buy_moderate'] + 5)
            self.current_thresholds['watchlist'] = min(55, old_thresholds['watchlist'] + 5)

            logger.warning(f"⚠️ Win rate bajo ({win_rate:.1f}%) - AUMENTANDO thresholds")

        # CASO 2: Win rate alto (>70%) → DISMINUIR thresholds (capturar más oportunidades)
        elif win_rate > 70:
            adjustment = 'decrease'
            self.current_thresholds['buy_strong'] = max(60, old_thresholds['buy_strong'] - 3)
            self.current_thresholds['buy_moderate'] = max(45, old_thresholds['buy_moderate'] - 3)
            self.current_thresholds['watchlist'] = max(30, old_thresholds['watchlist'] - 3)

            logger.info(f"✅ Win rate alto ({win_rate:.1f}%) - DISMINUYENDO thresholds")

        # CASO 3: Win rate OK pero P&L promedio bajo → Ajuste fino
        elif win_rate >= 40 and avg_pnl_pct < 5:
            adjustment = 'fine_tune_increase'
            self.current_thresholds['buy_strong'] = min(80, old_thresholds['buy_strong'] + 2)
            self.current_thresholds['buy_moderate'] = min(65, old_thresholds['buy_moderate'] + 2)

            logger.info(f"🔧 Win rate OK pero P&L bajo - Ajuste fino")

        # CASO 4: Todo bien → Mantener o ajuste muy pequeño hacia base
        else:
            adjustment = 'maintain'
            # Tender lentamente hacia los thresholds base
            for key in self.current_thresholds:
                current = self.current_thresholds[key]
                base = self.base_thresholds[key]
                if current > base:
                    self.current_thresholds[key] = max(base, current - 1)
                elif current < base:
                    self.current_thresholds[key] = min(base, current + 1)

            logger.info(f"✓ Performance estable - Manteniendo thresholds")

        return {
            'old_thresholds': old_thresholds,
            'new_thresholds': self.current_thresholds.copy(),
            'adjustment': adjustment,
            'metrics': {
                'win_rate': win_rate,
                'avg_pnl_pct': avg_pnl_pct,
                'avg_winner_pct': avg_winner_pct,
                'avg_loser_pct': avg_loser_pct,
                'sample_size': len(self.trade_history)
            }
        }

    def get_current_thresholds(self) -> Dict:
        """Obtiene thresholds actuales"""
        return self.current_thresholds.copy()

    def get_metrics(self) -> Dict:
        """Obtiene métricas actuales"""
        if len(self.trade_history) == 0:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_pnl_pct': 0
            }

        return {
            'total_trades': len(self.trade_history),
            'win_rate': self._calculate_win_rate(),
            'avg_pnl_pct': self._calculate_avg_pnl_pct(),
            'avg_winner_pct': self._calculate_avg_winner_pct(),
            'avg_loser_pct': self._calculate_avg_loser_pct(),
            'current_thresholds': self.current_thresholds.copy()
        }

    def reset_to_defaults(self):
        """Resetea thresholds a valores base"""
        self.current_thresholds = self.base_thresholds.copy()
        logger.info("Thresholds reseteados a valores base")

    def _calculate_win_rate(self) -> float:
        """Calcula win rate"""
        if not self.trade_history:
            return 0.0

        winners = sum(1 for t in self.trade_history if t.get('pnl', 0) > 0)
        return (winners / len(self.trade_history)) * 100

    def _calculate_avg_pnl_pct(self) -> float:
        """Calcula P&L promedio"""
        if not self.trade_history:
            return 0.0

        total_pnl_pct = sum(t.get('pnl_pct', 0) for t in self.trade_history)
        return total_pnl_pct / len(self.trade_history)

    def _calculate_avg_winner_pct(self) -> float:
        """Calcula P&L promedio de winners"""
        winners = [t for t in self.trade_history if t.get('pnl', 0) > 0]
        if not winners:
            return 0.0

        return sum(t.get('pnl_pct', 0) for t in winners) / len(winners)

    def _calculate_avg_loser_pct(self) -> float:
        """Calcula P&L promedio de losers"""
        losers = [t for t in self.trade_history if t.get('pnl', 0) < 0]
        if not losers:
            return 0.0

        return sum(t.get('pnl_pct', 0) for t in losers) / len(losers)


def simulate_optimizer_behavior():
    """Simula el comportamiento del optimizador con trades ficticios"""
    logger.info("\n" + "="*70)
    logger.info("SIMULACIÓN DE OPTIMIZER V5")
    logger.info("="*70)

    optimizer = DynamicOptimizer(window_size=20, recalibration_frequency=10)

    # Simular 50 trades
    np.random.seed(42)

    for i in range(50):
        # Simular resultado de trade
        # 60% de win rate con algunos winners grandes y losers controlados
        is_winner = np.random.random() < 0.6

        if is_winner:
            pnl_pct = np.random.uniform(5, 30)  # Winners: 5-30%
            pnl = pnl_pct * 100  # Simplificado
        else:
            pnl_pct = np.random.uniform(-8, -3)  # Losers: -3% a -8%
            pnl = pnl_pct * 100

        trade = {
            'symbol': f'TEST{i}',
            'entry_date': '2024-01-01',
            'exit_date': '2024-01-02',
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'score': 75,
            'action': 'COMPRA FUERTE'
        }

        optimizer.record_trade(trade)

        # Mostrar estado cada 10 trades
        if (i + 1) % 10 == 0:
            metrics = optimizer.get_metrics()
            logger.info(f"\nTrade {i+1}/50:")
            logger.info(f"  Win Rate: {metrics['win_rate']:.1f}%")
            logger.info(f"  Avg P&L: {metrics['avg_pnl_pct']:.1f}%")
            logger.info(f"  Thresholds: {metrics['current_thresholds']}")

    logger.info("\n" + "="*70)


if __name__ == "__main__":
    from logging_config_v5 import setup_logging
    setup_logging(level="INFO")

    # Ejecutar simulación
    simulate_optimizer_behavior()
