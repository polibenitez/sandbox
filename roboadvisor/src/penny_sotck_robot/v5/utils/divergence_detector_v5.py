#!/usr/bin/env python3
"""
DIVERGENCE DETECTOR V5
======================
Detección de divergencias RSI/MACD para confirmación de salidas

Features:
- Divergencias bajistas RSI (precio sube, RSI baja)
- Divergencias bajistas MACD
- Detección automática con numpy.diff
- Clasificación por fuerza
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger('divergence_detector')


class DivergenceDetector:
    """
    Detector de divergencias técnicas (RSI y MACD)
    """

    def __init__(self, lookback_window: int = 10):
        """
        Args:
            lookback_window: Ventana de lookback para detectar divergencias
        """
        self.lookback_window = lookback_window
        logger.info(f"DivergenceDetector inicializado - Lookback: {lookback_window}")

    def detect_rsi_divergence(self, price_history: np.ndarray,
                               rsi_history: np.ndarray) -> Dict:
        """
        Detecta divergencias bajistas en RSI

        Divergencia bajista:
        - Precio hace higher high
        - RSI hace lower high
        - Señal de debilidad y posible reversión

        Args:
            price_history: Array de precios
            rsi_history: Array de valores RSI

        Returns:
            Dict con información de divergencia
        """
        if price_history is None or rsi_history is None:
            return self._no_divergence()

        if len(price_history) < self.lookback_window or len(rsi_history) < self.lookback_window:
            return self._no_divergence()

        # Tomar ventana reciente
        recent_prices = price_history[-self.lookback_window:]
        recent_rsi = rsi_history[-self.lookback_window:]

        # Encontrar picos (highs locales)
        price_peaks = self._find_peaks(recent_prices)
        rsi_peaks = self._find_peaks(recent_rsi)

        if len(price_peaks) < 2 or len(rsi_peaks) < 2:
            return self._no_divergence()

        # Obtener los 2 picos más recientes
        price_peak1_idx = price_peaks[-2]
        price_peak2_idx = price_peaks[-1]

        rsi_peak1_idx = rsi_peaks[-2]
        rsi_peak2_idx = rsi_peaks[-1]

        price_peak1 = recent_prices[price_peak1_idx]
        price_peak2 = recent_prices[price_peak2_idx]

        rsi_peak1 = recent_rsi[rsi_peak1_idx]
        rsi_peak2 = recent_rsi[rsi_peak2_idx]

        # ¿Divergencia bajista?
        price_higher = price_peak2 > price_peak1
        rsi_lower = rsi_peak2 < rsi_peak1

        has_divergence = price_higher and rsi_lower

        if not has_divergence:
            return self._no_divergence()

        # Calcular fuerza de la divergencia
        price_change_pct = ((price_peak2 - price_peak1) / price_peak1) * 100
        rsi_change = rsi_peak2 - rsi_peak1

        # Clasificar fuerza
        if price_change_pct > 10 and rsi_change < -10:
            strength = 'strong'
            should_exit = True
        elif price_change_pct > 5 and rsi_change < -5:
            strength = 'moderate'
            should_exit = True
        else:
            strength = 'weak'
            should_exit = False

        return {
            'type': 'bearish',
            'indicator': 'RSI',
            'has_divergence': True,
            'strength': strength,
            'should_exit': should_exit,
            'price_peak1': float(price_peak1),
            'price_peak2': float(price_peak2),
            'price_change_pct': float(price_change_pct),
            'rsi_peak1': float(rsi_peak1),
            'rsi_peak2': float(rsi_peak2),
            'rsi_change': float(rsi_change),
            'reason': f'RSI divergencia bajista {strength}: precio +{price_change_pct:.1f}%, RSI {rsi_change:.1f}'
        }

    def detect_macd_divergence(self, price_history: np.ndarray,
                                macd_history: np.ndarray) -> Dict:
        """
        Detecta divergencias bajistas en MACD

        Divergencia bajista:
        - Precio hace higher high
        - MACD hace lower high

        Args:
            price_history: Array de precios
            macd_history: Array de valores MACD

        Returns:
            Dict con información de divergencia
        """
        if price_history is None or macd_history is None:
            return self._no_divergence()

        if len(price_history) < self.lookback_window or len(macd_history) < self.lookback_window:
            return self._no_divergence()

        # Tomar ventana reciente
        recent_prices = price_history[-self.lookback_window:]
        recent_macd = macd_history[-self.lookback_window:]

        # Encontrar picos
        price_peaks = self._find_peaks(recent_prices)
        macd_peaks = self._find_peaks(recent_macd)

        if len(price_peaks) < 2 or len(macd_peaks) < 2:
            return self._no_divergence()

        # Obtener los 2 picos más recientes
        price_peak1 = recent_prices[price_peaks[-2]]
        price_peak2 = recent_prices[price_peaks[-1]]

        macd_peak1 = recent_macd[macd_peaks[-2]]
        macd_peak2 = recent_macd[macd_peaks[-1]]

        # ¿Divergencia bajista?
        price_higher = price_peak2 > price_peak1
        macd_lower = macd_peak2 < macd_peak1

        has_divergence = price_higher and macd_lower

        if not has_divergence:
            return self._no_divergence()

        # Calcular fuerza
        price_change_pct = ((price_peak2 - price_peak1) / price_peak1) * 100
        macd_change_pct = ((macd_peak2 - macd_peak1) / abs(macd_peak1)) * 100 if macd_peak1 != 0 else 0

        # Clasificar fuerza
        if price_change_pct > 10 and macd_change_pct < -15:
            strength = 'strong'
            should_exit = True
        elif price_change_pct > 5 and macd_change_pct < -10:
            strength = 'moderate'
            should_exit = True
        else:
            strength = 'weak'
            should_exit = False

        return {
            'type': 'bearish',
            'indicator': 'MACD',
            'has_divergence': True,
            'strength': strength,
            'should_exit': should_exit,
            'price_peak1': float(price_peak1),
            'price_peak2': float(price_peak2),
            'price_change_pct': float(price_change_pct),
            'macd_peak1': float(macd_peak1),
            'macd_peak2': float(macd_peak2),
            'macd_change_pct': float(macd_change_pct),
            'reason': f'MACD divergencia bajista {strength}: precio +{price_change_pct:.1f}%, MACD {macd_change_pct:.1f}%'
        }

    def detect_all_divergences(self, price_history: np.ndarray,
                                rsi_history: np.ndarray,
                                macd_history: np.ndarray) -> Dict:
        """
        Detecta todas las divergencias (RSI + MACD)

        Returns:
            Dict con todas las divergencias encontradas
        """
        rsi_div = self.detect_rsi_divergence(price_history, rsi_history)
        macd_div = self.detect_macd_divergence(price_history, macd_history)

        # ¿Hay alguna divergencia?
        has_any_divergence = rsi_div['has_divergence'] or macd_div['has_divergence']

        # ¿Señal fuerte de salida? (ambas divergencias presentes)
        strong_exit_signal = (rsi_div['has_divergence'] and rsi_div['should_exit']) or \
                            (macd_div['has_divergence'] and macd_div['should_exit'])

        critical_exit_signal = (rsi_div['has_divergence'] and rsi_div['should_exit']) and \
                              (macd_div['has_divergence'] and macd_div['should_exit'])

        return {
            'has_any_divergence': has_any_divergence,
            'strong_exit_signal': strong_exit_signal,
            'critical_exit_signal': critical_exit_signal,
            'rsi_divergence': rsi_div,
            'macd_divergence': macd_div,
            'recommendation': self._get_recommendation(rsi_div, macd_div, critical_exit_signal)
        }

    def _find_peaks(self, data: np.ndarray) -> List[int]:
        """
        Encuentra índices de picos locales (local maxima)

        Args:
            data: Array de datos

        Returns:
            Lista de índices donde hay picos
        """
        if len(data) < 3:
            return []

        peaks = []

        for i in range(1, len(data) - 1):
            # Un pico es un punto mayor que sus vecinos
            if data[i] > data[i-1] and data[i] > data[i+1]:
                peaks.append(i)

        return peaks

    def _no_divergence(self) -> Dict:
        """Retorna Dict indicando que no hay divergencia"""
        return {
            'has_divergence': False,
            'should_exit': False,
            'strength': 'none',
            'reason': 'No divergence detected'
        }

    def _get_recommendation(self, rsi_div: Dict, macd_div: Dict,
                           critical: bool) -> str:
        """Genera recomendación basada en divergencias"""
        if critical:
            return "EXIT IMMEDIATELY - Divergencias RSI y MACD confirmadas"
        elif rsi_div['should_exit'] or macd_div['should_exit']:
            return "Consider EXIT - Divergencia detectada"
        elif rsi_div['has_divergence'] or macd_div['has_divergence']:
            return "Monitor closely - Divergencia débil"
        else:
            return "HOLD - No divergencias"


def calculate_macd(prices: np.ndarray, fast: int = 12, slow: int = 26,
                   signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula MACD (Moving Average Convergence Divergence)

    Args:
        prices: Array de precios
        fast: Periodo EMA rápida
        slow: Periodo EMA lenta
        signal: Periodo línea de señal

    Returns:
        (macd_line, signal_line, histogram)
    """
    if len(prices) < slow:
        return np.array([]), np.array([]), np.array([])

    # Calcular EMAs
    ema_fast = _calculate_ema(prices, fast)
    ema_slow = _calculate_ema(prices, slow)

    # MACD line
    macd_line = ema_fast - ema_slow

    # Signal line
    signal_line = _calculate_ema(macd_line, signal)

    # Histogram
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def _calculate_ema_deprecate(data: np.ndarray, period: int) -> np.ndarray:
    """Calcula EMA (Exponential Moving Average)"""
    if len(data) < period:
        return np.full(len(data), np.nan)

    multiplier = 2 / (period + 1)
    ema = np.zeros(len(data))

    # Primer valor: SMA
    ema[period-1] = np.mean(data[:period])

    # Resto: EMA
    for i in range(period, len(data)):
        ema[i] = (data[i] * multiplier) + (ema[i-1] * (1 - multiplier))

    # Primeros valores = NaN
    ema[:period-1] = np.nan

    return ema

def _calculate_ema(data: np.ndarray, period: int) -> np.ndarray:
    if len(data) < period:
        return np.full(len(data), np.nan)
    
    series = pd.Series(data)
    ema = series.ewm(span=period, adjust=False).mean().values
    
    # Opcional: forzar NaN en los primeros period-1 valores (como en tu versión)
    ema[:period-1] = np.nan
    
    return ema


if __name__ == "__main__":
    from logging_config_v5 import setup_logging
    setup_logging(level="INFO")

    # Test del detector con datos sintéticos
    logger.info("\n" + "="*70)
    logger.info("TEST DIVERGENCE DETECTOR V5")
    logger.info("="*70)

    # Simular divergencia bajista en RSI
    # Precio sube, pero RSI baja
    price_history = np.array([
        10.0, 10.2, 10.5, 10.3, 10.6,  # Primer pico
        10.4, 10.7, 11.0, 10.8, 11.2   # Segundo pico (más alto)
    ])

    rsi_history = np.array([
        55, 60, 65, 62, 68,  # Primer pico RSI
        64, 66, 62, 60, 58   # Segundo pico RSI (más bajo) - DIVERGENCIA
    ])

    macd_history = np.array([
        0.05, 0.10, 0.15, 0.12, 0.18,
        0.14, 0.16, 0.12, 0.10, 0.08
    ])

    detector = DivergenceDetector(lookback_window=10)

    # Detectar divergencias
    result = detector.detect_all_divergences(price_history, rsi_history, macd_history)

    print("\n📊 RESULTADO:")
    print(f"  Has Divergence: {result['has_any_divergence']}")
    print(f"  Strong Exit Signal: {result['strong_exit_signal']}")
    print(f"  Critical Exit Signal: {result['critical_exit_signal']}")
    print(f"\n  Recommendation: {result['recommendation']}")

    if result['rsi_divergence']['has_divergence']:
        rsi = result['rsi_divergence']
        print(f"\n  RSI Divergence:")
        print(f"    Strength: {rsi['strength']}")
        print(f"    Price: ${rsi['price_peak1']:.2f} → ${rsi['price_peak2']:.2f} (+{rsi['price_change_pct']:.1f}%)")
        print(f"    RSI: {rsi['rsi_peak1']:.0f} → {rsi['rsi_peak2']:.0f} ({rsi['rsi_change']:.0f})")
