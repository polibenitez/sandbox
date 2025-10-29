#!/usr/bin/env python3
"""
PENNY STOCK ROBOT ADVISOR V4 - PARADIGM SHIFT
==============================================

CAMBIO FUNDAMENTAL vs V3:
❌ V3: Compraba la explosión (tarde)
✅ V4: Anticipa la compresión ANTES de la explosión (temprano)

Filosofía: "Comprar el resorte comprimido, no el resorte liberado"

NUEVAS CAPACIDADES:
1. ✅ Sistema de scoring de 3 CAPAS (Setup + Trigger + Context)
2. ✅ Detección de "día de explosión" (¿estamos en día 1, 2 o 3?)
3. ✅ PENALIZACIONES SEVERAS por timing tardío
4. ✅ Análisis de contexto de mercado (SPY/QQQ/VIX)
5. ✅ Gestión de salidas mejorada con divergencias

Basado en el análisis teórico del 23 de Octubre, 2025
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, List, Tuple, Optional

class PennyStockAdvisorV4:
    """
    Robot Advisor V4 - Paradigm Shift Edition

    Ya no compramos cohetes en vuelo.
    Ahora los encontramos en la plataforma de lanzamiento.
    """

    def __init__(self, config_preset="balanced"):
        """
        Inicializa el robot con nueva filosofía de trading

        Args:
            config_preset (str): "conservative", "balanced", "aggressive"
        """
        self.watchlist = []
        self.config_preset = config_preset
        self.explosion_memory = {}  # Nueva: memoria de días de explosión
        self.load_configuration(config_preset)

    def load_configuration(self, preset):
        """Carga configuración con nuevos parámetros V4"""

        configurations = {
            "balanced": {
                "description": "Configuración balanceada V4 - anticipa squeezes",

                # NUEVO: Umbrales de scoring por capa
                "layer_weights": {
                    'phase1_setup': 0.40,      # 40 puntos - SETUP ESTRUCTURAL
                    'phase2_trigger': 0.40,    # 40 puntos - TRIGGER DE ENTRADA
                    'phase3_context': 0.20     # 20 puntos - CONTEXTO DE MERCADO
                },

                "thresholds": {
                    'buy_strong': 70,          # Score >= 70/100
                    'buy_moderate': 55,        # Score >= 55/100
                    'watchlist': 40,           # Score >= 40/100
                    'reject': 40               # Score < 40/100
                },

                # FASE 1: SETUP ESTRUCTURAL (Compresión)
                "setup_params": {
                    'min_compression_days': 5,
                    'max_price_range_pct': 8,           # Rango < 8% = comprimido
                    'bollinger_width_threshold': 0.05,
                    'adx_no_trend_threshold': 20,
                    'min_volume_dry_pct': 0.80,         # Volumen < 80% del promedio
                    'min_short_interest': 15,            # SI > 15%
                    'min_days_to_cover': 2.0,
                    'min_borrow_rate': 30,
                    'price_range_min': 0.50,
                    'price_range_max': 8.00,
                    'max_float_shares': 50_000_000
                },

                # FASE 2: TRIGGER DE ENTRADA (Explosión Inicial)
                "trigger_params": {
                    'min_volume_spike': 2.5,             # Volumen > 2.5x promedio
                    'min_volume_vs_yesterday': 1.5,      # Volumen hoy > 1.5x ayer
                    'min_intraday_acceleration': 2.0,    # Aceleración intradía
                    'max_explosion_day': 2,              # Solo día 1 o 2 del movimiento
                    'rsi_cross_from_below': 55,          # RSI cruzó 55 desde abajo
                    'rsi_not_overbought': 70,
                    'max_gap_up_pct': 10,                # Gap up < 10%
                    'min_close_in_range_pct': 0.30       # Cierre en top 30% del día
                },

                # FASE 3: CONTEXTO DE MERCADO
                "context_params": {
                    'vix_panic_threshold': 25,
                    'spy_trend_lookback': 5,
                    'min_spy_change_bearish': -2.0       # SPY -2% = mercado bajista
                },

                # PENALIZACIONES (CRÍTICO)
                "penalties": {
                    'price_up_15pct_3d': -30,            # Si precio subió 15% en 3d
                    'rsi_overbought': -20,               # RSI > 70
                    'volume_already_exploded': -25,      # Volumen ya explotó ayer
                    'gap_up_excessive': -15,             # Gap up > 10%
                    'late_to_party': -30,                # Día 3+ del movimiento
                    'market_bearish': -15                # Mercado general bajista
                },

                # GESTIÓN DE SALIDAS MEJORADA
                "exit_params": {
                    'stop_loss_initial_pct': 0.08,       # 8% inicial
                    'trailing_stop_trigger_pct': 0.15,   # Activar a +15%
                    'trailing_stop_distance_pct': 0.08,  # 8% desde máximo
                    'tp1_pct': 0.15,                     # +15%
                    'tp2_pct': 0.30,                     # +30%
                    'tp1_size': 0.30,                    # 30% de la posición
                    'tp2_size': 0.30,                    # 30% de la posición
                    'max_holding_days': 7,
                    'check_divergence_after_days': 2
                }
            }
        }

        if preset not in configurations:
            preset = "balanced"

        config = configurations[preset]
        self.layer_weights = config["layer_weights"]
        self.thresholds = config["thresholds"]
        self.setup_params = config["setup_params"]
        self.trigger_params = config["trigger_params"]
        self.context_params = config["context_params"]
        self.penalties = config["penalties"]
        self.exit_params = config["exit_params"]
        self.description = config["description"]

        print(f"🔧 Robot Advisor V4 - PARADIGM SHIFT Edition")
        print(f"📝 {self.description}")
        print(f"🎯 Filosofía: Anticipar compresión, no perseguir explosión")

    # ========================================================================
    # FASE A: NUEVAS FUNCIONES PRINCIPALES
    # ========================================================================

    def detect_setup_compression(self, price_history: np.ndarray,
                                 volume_history: np.ndarray,
                                 avg_volume_20d: float) -> Dict:
        """
        🔵 FASE 1 - SETUP: Detecta si el resorte está COMPRIMIDO

        Analogía: Como un resorte comprimido esperando ser liberado.
        Mientras más comprimido y más tiempo, más explosiva la liberación.

        Returns:
            dict: {
                'is_compressed': bool,
                'compression_score': 0-15 puntos,
                'price_range_pct': float,
                'compression_days': int,
                'volume_dry': bool,
                'reason': str
            }
        """
        if price_history is None or len(price_history) < self.setup_params['min_compression_days']:
            return {
                'is_compressed': False,
                'compression_score': 0,
                'reason': 'Datos insuficientes'
            }

        # 1. COMPRESIÓN DE PRECIO
        lookback = self.setup_params['min_compression_days']
        recent_prices = price_history[-lookback:]
        price_high = np.max(recent_prices)
        price_low = np.min(recent_prices)
        price_range_pct = ((price_high - price_low) / price_low * 100) if price_low > 0 else 100

        # Score por compresión de precio
        max_range = self.setup_params['max_price_range_pct']
        if price_range_pct <= 5:
            compression_score = 15
            level = "EXTREMA"
        elif price_range_pct <= max_range:
            compression_score = 10
            level = "ALTA"
        else:
            compression_score = 0
            level = "NINGUNA"

        is_compressed = price_range_pct <= max_range

        # 2. VOLUMEN SECO (baja liquidez antes de explosión)
        if volume_history is not None and len(volume_history) >= 5:
            recent_volume_avg = np.mean(volume_history[-5:])
            volume_dry = recent_volume_avg < (avg_volume_20d * self.setup_params['min_volume_dry_pct'])

            if volume_dry:
                compression_score += 5  # Bonus
        else:
            volume_dry = False

        # 3. Días en compresión (más tiempo = mejor)
        compression_days = 0
        for i in range(len(price_history) - 1, 0, -1):
            if i < lookback:
                break
            subset = price_history[i-lookback:i]
            range_pct = ((np.max(subset) - np.min(subset)) / np.min(subset) * 100)
            if range_pct <= max_range:
                compression_days += 1
            else:
                break

        return {
            'is_compressed': is_compressed,
            'compression_score': compression_score,
            'price_range_pct': price_range_pct,
            'compression_level': level,
            'compression_days': compression_days,
            'volume_dry': volume_dry,
            'reason': f"Compresión {level}: rango {price_range_pct:.1f}% en {lookback}d, volumen {'seco' if volume_dry else 'normal'}"
        }

    def get_explosion_day_number(self, symbol: str,
                                  volume_history: np.ndarray,
                                  avg_volume_20d: float,
                                  price_history: np.ndarray) -> Dict:
        """
        🟡 CRÍTICO: Determina en qué DÍA de la explosión estamos

        Esta es LA función que evita el error de BYND/AIRE.

        Si estamos en día 1-2: ✅ Bueno - entramos temprano
        Si estamos en día 3+: ❌ Tarde - NO entramos

        Returns:
            dict: {
                'explosion_day': int (1, 2, 3+),
                'is_early_enough': bool,
                'penalty_points': int,
                'consecutive_high_volume_days': int
            }
        """
        if volume_history is None or len(volume_history) < 5:
            return {
                'explosion_day': 0,
                'is_early_enough': False,
                'penalty_points': 0,
                'reason': 'Datos insuficientes'
            }

        # Buscar cuántos días consecutivos de volumen alto
        spike_threshold = self.trigger_params['min_volume_spike']
        consecutive_days = 0

        for i in range(len(volume_history) - 1, -1, -1):
            vol_ratio = volume_history[i] / avg_volume_20d if avg_volume_20d > 0 else 0

            if vol_ratio >= spike_threshold:
                consecutive_days += 1
            else:
                break  # Se acabó la racha

        # Determinar día de explosión
        explosion_day = max(1, consecutive_days)

        # ¿Estamos temprano o tarde?
        max_day = self.trigger_params['max_explosion_day']
        is_early_enough = explosion_day <= max_day

        # Penalización por llegar tarde
        if explosion_day == 1:
            penalty = 0
            status = "PERFECTO - Día 1"
        elif explosion_day == 2:
            penalty = 0
            status = "BUENO - Día 2"
        elif explosion_day == 3:
            penalty = self.penalties['late_to_party']
            status = "TARDE - Día 3"
        else:
            penalty = self.penalties['late_to_party']
            status = f"MUY TARDE - Día {explosion_day}"

        # Guardar en memoria
        self.explosion_memory[symbol] = {
            'day': explosion_day,
            'date': datetime.now(),
            'is_early': is_early_enough
        }

        return {
            'explosion_day': explosion_day,
            'is_early_enough': is_early_enough,
            'penalty_points': penalty,
            'consecutive_high_volume_days': consecutive_days,
            'status': status,
            'reason': f"{status} ({consecutive_days} días consecutivos de volumen alto)"
        }

    def calculate_phase1_setup_score(self, symbol: str,
                                     price_history: np.ndarray,
                                     volume_history: np.ndarray,
                                     market_data: Dict) -> Dict:
        """
        🔵 CAPA 1: SETUP ESTRUCTURAL (40 puntos máximo)

        El resorte debe estar comprimido ANTES de que explote.
        Este es el 40% del score total.

        Criterios:
        - Compresión de precio (15 pts)
        - Volumen seco (10 pts)
        - Short interest alto (10 pts)
        - Estructura favorable (5 pts)
        """
        max_score = 40
        total_score = 0
        signals = []

        # 1. Compresión de precio + volumen seco (25 pts)
        compression = self.detect_setup_compression(
            price_history,
            volume_history,
            market_data.get('avg_volume_20d', 1)
        )
        total_score += compression['compression_score']
        signals.append(compression['reason'])

        # 2. Short Interest cualificado (10 pts)
        short_interest = market_data.get('short_interest_pct', 0)
        days_to_cover = market_data.get('days_to_cover', 0)
        borrow_rate = market_data.get('borrow_rate', 0)

        si_score = 0
        if short_interest >= self.setup_params['min_short_interest'] and \
           days_to_cover >= self.setup_params['min_days_to_cover']:
            si_score = 10
            signals.append(f"SI cualificado: {short_interest:.1f}%, DTC: {days_to_cover:.1f}")
        elif short_interest >= 10:
            si_score = 5
            signals.append(f"SI moderado: {short_interest:.1f}%")

        total_score += si_score

        # 3. Estructura favorable: float + precio (5 pts)
        price = market_data.get('price', 0)
        # float_shares = market_data.get('float_shares', 100_000_000)

        structure_score = 0
        if self.setup_params['price_range_min'] <= price <= self.setup_params['price_range_max']:
            structure_score = 5
            signals.append(f"Precio en sweet spot: ${price:.2f}")

        total_score += structure_score

        # Normalizar a porcentaje del máximo
        score_pct = (total_score / max_score) * self.layer_weights['phase1_setup'] * 100

        return {
            'phase': 'FASE 1: SETUP ESTRUCTURAL',
            'score': score_pct,
            'max_score': self.layer_weights['phase1_setup'] * 100,
            'compression': compression,
            'short_interest_qualified': short_interest >= self.setup_params['min_short_interest'],
            'signals': signals,
            'passed': compression['is_compressed']
        }

    def calculate_phase2_trigger_score(self, symbol: str,
                                       current_price: float,
                                       price_history: np.ndarray,
                                       volume_history: np.ndarray,
                                       market_data: Dict) -> Dict:
        """
        🟡 CAPA 2: TRIGGER DE ENTRADA (40 puntos máximo)

        Confirmación de que el resorte COMIENZA a liberarse (no que ya se liberó).
        Este es el 40% del score total.

        Criterios:
        - Volumen explosivo inicial (15 pts)
        - Breakout técnico limpio (10 pts)
        - Momentum confirmado (10 pts)
        - Confirmación institucional (5 pts)
        """
        max_score = 40
        total_score = 0
        signals = []

        # 1. VOLUMEN EXPLOSIVO - ¿Estamos temprano o tarde? (15 pts)
        current_volume = market_data.get('volume', 0)
        avg_volume = market_data.get('avg_volume_20d', 1)

        explosion_info = self.get_explosion_day_number(
            symbol, volume_history, avg_volume, price_history
        )

        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0

        if explosion_info['is_early_enough'] and volume_ratio >= self.trigger_params['min_volume_spike']:
            volume_score = 15
            signals.append(f"Volumen explosivo TEMPRANO: {volume_ratio:.1f}x en día {explosion_info['explosion_day']}")
        elif volume_ratio >= self.trigger_params['min_volume_spike']:
            volume_score = 5  # Volumen alto pero tardío
            signals.append(f"Volumen alto pero TARDÍO: día {explosion_info['explosion_day']}")
        else:
            volume_score = 0

        total_score += volume_score

        # 2. BREAKOUT TÉCNICO LIMPIO (10 pts)
        if price_history is not None and len(price_history) >= 20:
            resistance = np.max(price_history[-20:-1])  # Máximo previo (sin incluir hoy)
            sma_20 = np.mean(price_history[-20:])
            sma_50 = np.mean(price_history[-50:]) if len(price_history) >= 50 else sma_20

            breakout_clean = current_price > resistance and \
                           current_price > sma_20 and \
                           sma_20 > sma_50

            if breakout_clean:
                breakout_score = 10
                signals.append(f"Breakout limpio: precio > resistencia y SMAs")
            else:
                breakout_score = 0
        else:
            breakout_score = 0

        total_score += breakout_score

        # 3. MOMENTUM CONFIRMADO (10 pts)
        rsi = market_data.get('rsi', 50)
        rsi_crossed_from_below = 55 <= rsi <= self.trigger_params['rsi_not_overbought']

        if rsi_crossed_from_below:
            momentum_score = 10
            signals.append(f"RSI en zona óptima: {rsi:.0f}")
        elif rsi > self.trigger_params['rsi_not_overbought']:
            momentum_score = 0
            signals.append(f"RSI sobrecomprado: {rsi:.0f}")
        else:
            momentum_score = 5

        total_score += momentum_score

        # 4. CONFIRMACIÓN INSTITUCIONAL (5 pts - simplificado)
        if price_history is not None and len(price_history) >= 2:
            # Precio subiendo en últimos días
            price_trend_up = price_history[-1] > price_history[-3]
            if price_trend_up:
                institutional_score = 5
                signals.append("Precio en tendencia alcista")
            else:
                institutional_score = 0
        else:
            institutional_score = 0

        total_score += institutional_score

        # Normalizar
        score_pct = (total_score / max_score) * self.layer_weights['phase2_trigger'] * 100

        return {
            'phase': 'FASE 2: TRIGGER DE ENTRADA',
            'score': score_pct,
            'max_score': self.layer_weights['phase2_trigger'] * 100,
            'explosion_info': explosion_info,
            'volume_ratio': volume_ratio,
            'signals': signals,
            'passed': explosion_info['is_early_enough'] and volume_ratio >= self.trigger_params['min_volume_spike']
        }

    def calculate_phase3_context_score(self, market_context: Dict) -> Dict:
        """
        🌍 CAPA 3: CONTEXTO DE MERCADO (20 puntos máximo)

        No luchas contra el mercado general. Si SPY/QQQ están bajistas,
        incluso el mejor setup de penny stock puede fallar.

        Criterios:
        - Mercado general neutral/alcista (10 pts)
        - VIX bajo (5 pts)
        - Sector no en pánico (5 pts)
        """
        max_score = 20
        total_score = 0
        signals = []

        # 1. Mercado general (SPY/QQQ) (10 pts)
        spy_trend = market_context.get('spy_trend', 'neutral')
        if spy_trend == 'bullish':
            market_score = 10
            signals.append("Mercado alcista")
        elif spy_trend == 'neutral':
            market_score = 10
            signals.append("Mercado neutral")
        else:
            market_score = 0
            signals.append("Mercado bajista")

        total_score += market_score

        # 2. VIX (5 pts)
        vix = market_context.get('vix', 15)
        if vix < self.context_params['vix_panic_threshold']:
            vix_score = 5
            signals.append(f"VIX bajo: {vix:.1f}")
        else:
            vix_score = 0
            signals.append(f"VIX alto: {vix:.1f}")

        total_score += vix_score

        # 3. Sector (5 pts - simplificado)
        sector_sentiment = market_context.get('sector_sentiment', 'neutral')
        if sector_sentiment != 'panic':
            sector_score = 5
            signals.append("Sector estable")
        else:
            sector_score = 0
            signals.append("Sector en pánico")

        total_score += sector_score

        # Normalizar
        score_pct = (total_score / max_score) * self.layer_weights['phase3_context'] * 100

        return {
            'phase': 'FASE 3: CONTEXTO DE MERCADO',
            'score': score_pct,
            'max_score': self.layer_weights['phase3_context'] * 100,
            'signals': signals,
            'market_favorable': spy_trend != 'bearish' and vix < self.context_params['vix_panic_threshold']
        }

    def apply_penalties(self, symbol: str,
                       price_history: np.ndarray,
                       volume_history: np.ndarray,
                       market_data: Dict,
                       explosion_info: Dict,
                       market_context: Dict) -> Dict:
        """
        ⛔ PENALIZACIONES SEVERAS - La clave para evitar BYND/AIRE

        Esto es lo que faltaba en V3. Las penalizaciones pueden llevar
        un score de 75 a 45, rechazando la entrada.

        Returns:
            dict: {
                'total_penalty': int (puntos negativos),
                'penalty_breakdown': list,
                'should_reject': bool
            }
        """
        total_penalty = 0
        penalties_applied = []

        # 1. Precio subió mucho en últimos 3 días (-30 pts) - ¡YA ES TARDE!
        if price_history is not None and len(price_history) >= 3:
            price_3d_ago = price_history[-4]
            current_price = price_history[-1]
            price_change_3d = ((current_price - price_3d_ago) / price_3d_ago * 100) if price_3d_ago > 0 else 0

            if price_change_3d > 15:
                total_penalty += self.penalties['price_up_15pct_3d']
                penalties_applied.append(f"Precio subió {price_change_3d:.1f}% en 3d (TARDE): {self.penalties['price_up_15pct_3d']} pts")

        # 2. RSI sobrecomprado (-20 pts)
        rsi = market_data.get('rsi', 50)
        if rsi > self.trigger_params['rsi_not_overbought']:
            total_penalty += self.penalties['rsi_overbought']
            penalties_applied.append(f"RSI sobrecomprado ({rsi:.0f}): {self.penalties['rsi_overbought']} pts")

        # 3. Volumen ya explotó ayer (-25 pts)
        if volume_history is not None and len(volume_history) >= 2:
            avg_volume = market_data.get('avg_volume_20d', 1)
            yesterday_volume = volume_history[-2]
            yesterday_ratio = yesterday_volume / avg_volume if avg_volume > 0 else 0

            if yesterday_ratio > 4.0:
                total_penalty += self.penalties['volume_already_exploded']
                penalties_applied.append(f"Volumen ayer ya fue {yesterday_ratio:.1f}x: {self.penalties['volume_already_exploded']} pts")

        # 4. Día de explosión tardío (-30 pts)
        if explosion_info['explosion_day'] >= 3:
            total_penalty += explosion_info['penalty_points']
            penalties_applied.append(f"Día {explosion_info['explosion_day']} de explosión: {explosion_info['penalty_points']} pts")

        # 5. Mercado bajista (-15 pts)
        if market_context.get('spy_trend') == 'bearish':
            total_penalty += self.penalties['market_bearish']
            penalties_applied.append(f"Mercado bajista: {self.penalties['market_bearish']} pts")

        # ¿Deberíamos rechazar automáticamente?
        should_reject = total_penalty <= -50  # Si penalización >= 50 puntos, rechazar

        return {
            'total_penalty': total_penalty,
            'penalty_breakdown': penalties_applied,
            'should_reject': should_reject,
            'reason': f"{len(penalties_applied)} penalizaciones aplicadas ({total_penalty} pts)"
        }

    # ========================================================================
    # ANÁLISIS COMPLETO V4
    # ========================================================================

    def analyze_symbol_v4(self, symbol: str, market_data: Dict,
                          historical_data: Dict, market_context: Dict) -> Dict:
        """
        Análisis completo con sistema de 3 capas + penalizaciones

        Args:
            symbol: Ticker
            market_data: Datos actuales
            historical_data: Datos históricos (REQUERIDO en V4)
            market_context: Contexto de mercado (SPY/VIX/etc)

        Returns:
            dict: Análisis completo con score y recomendación
        """
        print(f"\n🔍 Analizando {symbol} (V4 - Paradigm Shift)...")

        # Extraer datos
        current_price = market_data.get('price', 0)
        price_history = historical_data.get('close')
        volume_history = historical_data.get('volume')

        # CAPA 1: SETUP ESTRUCTURAL (40 pts)
        phase1 = self.calculate_phase1_setup_score(
            symbol, price_history, volume_history, market_data
        )

        # CAPA 2: TRIGGER DE ENTRADA (40 pts)
        phase2 = self.calculate_phase2_trigger_score(
            symbol, current_price, price_history, volume_history, market_data
        )

        # CAPA 3: CONTEXTO DE MERCADO (20 pts)
        phase3 = self.calculate_phase3_context_score(market_context)

        # SCORE BRUTO (antes de penalizaciones)
        raw_score = phase1['score'] + phase2['score'] + phase3['score']

        # APLICAR PENALIZACIONES
        penalties = self.apply_penalties(
            symbol, price_history, volume_history, market_data,
            phase2['explosion_info'], market_context
        )

        # SCORE FINAL
        final_score = raw_score + penalties['total_penalty']
        final_score = max(0, final_score)  # No puede ser negativo

        # DECISIÓN
        trading_decision = self.generate_trading_decision_v4(
            symbol, final_score, raw_score, penalties,
            phase1, phase2, phase3, market_data
        )

        return {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'raw_score': raw_score,
            'final_score': final_score,
            'phase1_setup': phase1,
            'phase2_trigger': phase2,
            'phase3_context': phase3,
            'penalties': penalties,
            'trading_decision': trading_decision,
            'market_data': market_data
        }

    def generate_trading_decision_v4(self, symbol: str, final_score: float,
                                     raw_score: float, penalties: Dict,
                                     phase1: Dict, phase2: Dict, phase3: Dict,
                                     market_data: Dict) -> Dict:
        """
        Genera decisión de trading basada en el nuevo sistema V4
        """
        # Rechazar automáticamente si penalizaciones son severas
        if penalties['should_reject']:
            return {
                'symbol': symbol,
                'action': 'RECHAZAR',
                'reason': f'Penalizaciones severas: {penalties["total_penalty"]} puntos',
                'score': final_score,
                'details': penalties['penalty_breakdown']
            }

        # Rechazar si setup no está comprimido
        if not phase1['compression']['is_compressed']:
            return {
                'symbol': symbol,
                'action': 'RECHAZAR',
                'reason': 'No hay compresión previa - no es un setup válido',
                'score': final_score
            }

        # Decisión por score
        if final_score >= self.thresholds['buy_strong']:
            action = 'COMPRA FUERTE'
            urgency = 'ALTA'
            confidence = 'ALTA'
        elif final_score >= self.thresholds['buy_moderate']:
            action = 'COMPRA MODERADA'
            urgency = 'MEDIA'
            confidence = 'MEDIA'
        elif final_score >= self.thresholds['watchlist']:
            action = 'WATCHLIST'
            urgency = 'BAJA'
            confidence = 'MONITOREAR'
            return {
                'symbol': symbol,
                'action': action,
                'reason': f'Score {final_score:.0f} - Setup prometedor pero falta trigger',
                'score': final_score,
                'urgency': urgency
            }
        else:
            action = 'RECHAZAR'
            urgency = 'N/A'
            confidence = 'BAJA'
            return {
                'symbol': symbol,
                'action': action,
                'reason': f'Score {final_score:.0f} < {self.thresholds["watchlist"]}',
                'score': final_score
            }

        # Generar plan de trading completo
        current_price = market_data.get('price', 0)

        # Stop loss y take profits
        stop_loss = current_price * (1 - self.exit_params['stop_loss_initial_pct'])
        tp1 = current_price * (1 + self.exit_params['tp1_pct'])
        tp2 = current_price * (1 + self.exit_params['tp2_pct'])

        # Trailing stop
        trailing_trigger = current_price * (1 + self.exit_params['trailing_stop_trigger_pct'])

        # Tamaño de posición (ejemplo simplificado)
        position_size_pct = 3 if urgency == 'ALTA' else 2

        return {
            'symbol': symbol,
            'action': action,
            'urgency': urgency,
            'confidence': confidence,
            'score': final_score,
            'raw_score': raw_score,
            'penalty_applied': penalties['total_penalty'],
            'current_price': current_price,
            'position_size_pct': position_size_pct,
            'stop_loss': stop_loss,
            'take_profit_1': tp1,
            'take_profit_2': tp2,
            'tp1_allocation': self.exit_params['tp1_size'] * 100,
            'tp2_allocation': self.exit_params['tp2_size'] * 100,
            'trailing_stop_trigger': trailing_trigger,
            'max_holding_days': self.exit_params['max_holding_days'],
            'key_signals': self._summarize_signals(phase1, phase2, phase3),
            'warnings': penalties['penalty_breakdown']
        }

    def _summarize_signals(self, phase1, phase2, phase3):
        """Resume las señales clave"""
        summary = []
        summary.extend(phase1['signals'][:2])
        summary.extend(phase2['signals'][:2])
        summary.extend(phase3['signals'][:1])
        return summary

    # ========================================================================
    # OBTENCIÓN DE DATOS
    # ========================================================================

    def get_enhanced_market_data(self, symbol: str, period="2mo") -> Tuple[Optional[Dict], Optional[Dict]]:
        """Obtiene datos mejorados de yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            info = ticker.info

            if len(hist) == 0:
                return None, None

            # Datos actuales
            current_price = hist['Close'].iloc[-1]
            current_volume = hist['Volume'].iloc[-1]

            # ATR
            high_low = hist['High'] - hist['Low']
            high_close = np.abs(hist['High'] - hist['Close'].shift())
            low_close = np.abs(hist['Low'] - hist['Close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr_14 = tr.rolling(14).mean().iloc[-1] if len(tr) >= 14 else current_price * 0.05

            # RSI
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            rsi_value = rsi.iloc[-1] if len(rsi) > 0 else 50

            market_data = {
                'price': float(current_price),
                'volume': int(current_volume),
                'avg_volume_20d': int(hist['Volume'].mean()),
                'short_interest_pct': info.get('shortPercentOfFloat', 0.1) * 100,
                'days_to_cover': 2.0,
                'borrow_rate': 25.0,
                'rsi': float(rsi_value),
                'atr_14': float(atr_14),
                'bid_ask_spread_pct': 5.0,
                'market_depth_dollars': 10000
            }

            historical_data = {
                'close': hist['Close'].values,
                'volume': hist['Volume'].values,
                'high': hist['High'].values,
                'low': hist['Low'].values
            }

            return market_data, historical_data

        except Exception as e:
            print(f"❌ Error obteniendo datos para {symbol}: {e}")
            return None, None

    def update_watchlist(self, symbols: List[str]):
        """Actualiza watchlist"""
        self.watchlist = [s.upper() for s in symbols]

if __name__ == "__main__":
    print("🚀 PENNY STOCK ADVISOR V4 - PARADIGM SHIFT")
    print("Ya no compramos cohetes en vuelo. Los encontramos en la plataforma de lanzamiento.\n")
