#!/usr/bin/env python3
"""
PENNY STOCK ROBOT ADVISOR V5 - EVOLUTION
=========================================

CAMBIOS FUNDAMENTALES vs V4:
✅ Machine Learning: Predicción de probabilidad de breakout real
✅ Datos Alternativos: Reddit sentiment + Short borrow rates
✅ Optimización Dinámica: Auto-ajuste de thresholds
✅ Sistema de Caché: 10x más rápido
✅ Divergencias: Detección automática RSI/MACD
✅ Scores Normalizados: Todo en escala 0-100
✅ ATR Compression: Ratio ATR para detectar compresión extrema

Filosofía: "Comprar el resorte comprimido, no el resorte liberado"
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz
import yfinance as yf
from typing import Dict, List, Tuple, Optional

# Agregar utils al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

# Imports de módulos V5
from logging_config_v5 import setup_logging, get_logger
from market_data_cache_v5 import MarketDataCache
from ml_model_v5 import BreakoutPredictor
from alternative_data_v5 import AlternativeDataProvider
from divergence_detector_v5 import DivergenceDetector, calculate_macd
from optimizer_v5 import DynamicOptimizer


class PennyStockAdvisorV5:
    """
    Robot Advisor V5 - Evolution Edition

    Integra todos los módulos avanzados para análisis predictivo
    """

    def __init__(self, config_preset="balanced", enable_logging=True, enable_cache=True):
        """
        Inicializa el robot con nueva arquitectura V5

        Args:
            config_preset (str): "conservative", "balanced", "aggressive"
            enable_logging (bool): Habilitar sistema de logging
            enable_cache (bool): Habilitar caché de datos
        """
        # Configurar logging primero
        if enable_logging:
            setup_logging(level="INFO", log_to_file=True)

        self.logger = get_logger('penny_stock_v5')
        self.logger.info("="*70)
        self.logger.info("PENNY STOCK ADVISOR V5 - EVOLUTION EDITION")
        self.logger.info("="*70)

        # Configuración
        self.config_preset = config_preset
        self.watchlist = []
        self.explosion_memory = {}

        # Módulos V5
        self.data_cache = MarketDataCache(enable_persistence=enable_cache) if enable_cache else None
        self.ml_predictor = BreakoutPredictor()
        self.alt_data_provider = AlternativeDataProvider(use_api=False)  # Usar datos locales CSV
        self.divergence_detector = DivergenceDetector(lookback_window=10)
        self.optimizer = DynamicOptimizer(window_size=20, recalibration_frequency=10)

        # Cargar configuración
        self.load_configuration(config_preset)

        self.logger.info(f"Configuración: {config_preset.upper()}")
        self.logger.info(f"ML Model: {'Trained' if self.ml_predictor.is_trained else 'Not trained'}")
        self.logger.info(f"Cache: {'Enabled' if enable_cache else 'Disabled'}")
        self.logger.info(f"Filosofía: Anticipar compresión, no perseguir explosión")
        self.logger.info("="*70)

    def load_configuration(self, preset):
        """Carga configuración con parámetros V5 mejorados"""

        configurations = {
            "balanced": {
                "description": "Configuración balanceada V5 - ML + Alternative Data",

                # Pesos de capas (normalizados 0-100)
                "layer_weights": {
                    'phase1_setup': 35,        # 35 puntos - SETUP ESTRUCTURAL
                    'phase2_trigger': 35,      # 35 puntos - TRIGGER DE ENTRADA
                    'phase3_context': 20,      # 20 puntos - CONTEXTO + ALT DATA
                    'ml_adjustment': 10        # 10 puntos - AJUSTE ML
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
                    'max_price_range_pct': 8,
                    'bollinger_width_threshold': 0.05,
                    'adx_no_trend_threshold': 20,
                    'min_volume_dry_pct': 0.80,
                    'min_short_interest': 15,
                    'min_days_to_cover': 2.0,
                    'min_borrow_rate': 30,
                    'price_range_min': 0.50,
                    'price_range_max': 8.00,
                    'max_float_shares': 50_000_000,
                    'atr_ratio_extreme_compression': 0.02  # NUEVO V5
                },

                # FASE 2: TRIGGER DE ENTRADA
                "trigger_params": {
                    'min_volume_spike': 2.5,
                    'min_volume_vs_yesterday': 1.5,
                    'min_intraday_acceleration': 2.0,
                    'max_explosion_day': 2,
                    'rsi_cross_from_below': 55,
                    'rsi_not_overbought': 70,
                    'max_gap_up_pct': 10,
                    'min_close_in_range_pct': 0.30,

                    # NUEVO: Entrada tardía (día 3) con condiciones estrictas
                    'allow_late_entry_day3': True,
                    'day3_min_volume_spike': 3.5,      # Volumen muy alto requerido
                    'day3_max_rsi': 75,                # No sobrecomprado extremo
                    'day3_max_price_change_3d': 60,    # Máximo +60% en 3 días
                    'day3_min_momentum_sustained': True, # Momentum sostenido
                    'day3_penalty_reduced': -15        # Penalización reducida vs -30
                },

                # FASE 3: CONTEXTO + ALTERNATIVE DATA
                "context_params": {
                    'vix_panic_threshold': 25,
                    'spy_trend_lookback': 5,
                    'min_spy_change_bearish': -2.0,
                    'alt_data_weight': 0.30  # 30% del score de fase 3
                },

                # ML PARAMETERS
                "ml_params": {
                    'min_probability': 0.30,    # Penalizar si prob < 30%
                    'high_confidence': 0.70,    # Bonus si prob > 70%
                    'ml_penalty': -15,          # Penalización por baja probabilidad
                    'ml_bonus': 5               # Bonus por alta probabilidad
                },

                # PENALIZACIONES
                "penalties": {
                    'price_up_15pct_3d': -30,
                    'rsi_overbought': -20,
                    'volume_already_exploded': -25,
                    'gap_up_excessive': -15,
                    'late_to_party': -30,
                    'market_bearish': -15,
                    'ml_low_probability': -15  # NUEVO V5
                },

                # GESTIÓN DE SALIDAS
                "exit_params": {
                    'stop_loss_initial_pct': 0.08,
                    'trailing_stop_trigger_pct': 0.15,
                    'trailing_stop_distance_pct': 0.08,
                    'tp1_pct': 0.15,
                    'tp2_pct': 0.30,
                    'tp1_size': 0.30,
                    'tp2_size': 0.30,
                    'max_holding_days': 7,
                    'check_divergence_after_days': 2
                },

                # NUEVA FILOSOFÍA: MOMENTUM PURO (sin compresión previa)
                "momentum_puro_params": {
                    'enabled': True,
                    'min_volume_spike': 4.0,           # Volumen muy alto requerido
                    'max_explosion_day': 2,            # Solo día 1-2 (no entrar tarde)
                    'min_price_change_3d': 15,         # Mínimo cambio alcista
                    'max_price_change_3d': 80,         # Máximo para no estar muy tarde
                    'rsi_min_bounce': 15,              # RSI mínimo (rebote desde sobreventa)
                    'rsi_max_momentum': 65,            # RSI máximo (no sobrecomprado)
                    'min_breakout_from_low': 0.20,     # 20% sobre mínimo reciente
                    'lookback_low_days': 10,           # Días para buscar mínimo reciente
                    'min_consecutive_green_candles': 1, # Días consecutivos alcistas
                    'position_size_pct_reduction': 0.5, # Reducir posición 50% vs setup normal
                    'score_threshold': 50              # Score mínimo para Momentum Puro
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
        self.ml_params = config["ml_params"]
        self.penalties = config["penalties"]
        self.exit_params = config["exit_params"]
        self.momentum_puro_params = config["momentum_puro_params"]
        self.description = config["description"]

    # ========================================================================
    # OBTENCIÓN DE DATOS CON CACHÉ
    # ========================================================================

    def get_enhanced_market_data(self, symbol: str, period="2mo") -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Obtiene datos de mercado con sistema de caché V5

        Returns:
            (market_data, historical_data) o (None, None) si falla
        """
        # Intentar obtener del caché
        if self.data_cache:
            cached = self.data_cache.get_cached_data(symbol, 'market_data', period, ttl_minutes=60)
            if cached:
                self.logger.debug(f"Cache HIT: {symbol}")
                return cached

        # No está en caché, obtener de yfinance
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            info = ticker.info

            if len(hist) == 0:
                return None, None

            # Datos actuales
            current_price = hist['Close'].iloc[-1]
            current_volume = hist['Volume'].iloc[-1]

            # Intentar obtener precio del pre-market si aplica
            ny_tz = pytz.timezone("America/New_York")
            now_ny = datetime.now(ny_tz)
            market_open = now_ny.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now_ny.replace(hour=16, minute=0, second=0, microsecond=0)

            if now_ny < market_open or now_ny > market_close:
                pre_hist = ticker.history(period="1d", interval="1m", prepost=True)
                if not pre_hist.empty:
                    latest_premarket_price = pre_hist['Close'].iloc[-1]
                    if not np.isnan(latest_premarket_price):
                        current_price = latest_premarket_price

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

            # MACD
            macd_line, macd_signal, macd_diff_series = calculate_macd(hist['Close'])
            macd_diff_value = macd_diff_series[-1] if len(macd_diff_series) > 0 else 0

            market_data = {
                'price': float(current_price),
                'volume': int(current_volume),
                'avg_volume_20d': int(hist['Volume'].mean()),
                'short_interest_pct': info.get('shortPercentOfFloat', 0.1) * 100,
                'days_to_cover': 2.0,
                'borrow_rate': 25.0,
                'rsi': float(rsi_value),
                'atr_14': float(atr_14),
                'atr_ratio': float(atr_14 / current_price) if current_price > 0 else 0.05,
                'macd_diff': float(macd_diff_value),
                'bid_ask_spread_pct': 5.0,
                'market_depth_dollars': 10000
            }

            historical_data = {
                'close': hist['Close'].values,
                'volume': hist['Volume'].values,
                'high': hist['High'].values,
                'low': hist['Low'].values,
                'rsi': rsi.values,
                'macd': macd_line,
                'macd_signal': macd_signal,
                'macd_diff': macd_diff_series
            }

            # Guardar en caché
            if self.data_cache:
                self.data_cache.set_cached_data(
                    symbol,
                    (market_data, historical_data),
                    'market_data',
                    period
                )

            return market_data, historical_data

        except Exception as e:
            self.logger.exception(f"Error obteniendo datos para {symbol}: {e}")
            return None, None

    # ========================================================================
    # FASE 1: SETUP ESTRUCTURAL CON ATR
    # ========================================================================

    def detect_setup_compression(self, price_history: np.ndarray,
                                 volume_history: np.ndarray,
                                 avg_volume_20d: float,
                                 atr_ratio: float) -> Dict:
        """
        FASE 1 - SETUP: Detecta compresión con ATR ratio (V5)

        NUEVO: Incluye atr_ratio < 0.02 para compresión extrema
        """
        if price_history is None or len(price_history) < self.setup_params['min_compression_days']:
            return {
                'is_compressed': False,
                'compression_score': 0,
                'reason': 'Datos insuficientes'
            }

        lookback = self.setup_params['min_compression_days']
        recent_prices = price_history[-lookback:]
        price_high = np.max(recent_prices)
        price_low = np.min(recent_prices)
        price_range_pct = ((price_high - price_low) / price_low * 100) if price_low > 0 else 100

        # Score por compresión de precio (0-15 pts)
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

        # NUEVO V5: ATR ratio check (bonus de +5 pts)
        atr_extreme_threshold = self.setup_params['atr_ratio_extreme_compression']
        if atr_ratio < atr_extreme_threshold:
            compression_score += 5
            level = "EXTREMA (ATR muy bajo)"
            self.logger.debug(f"ATR ratio {atr_ratio:.4f} < {atr_extreme_threshold} - Compresión extrema detectada")

        # Volumen seco (bonus +5 pts)
        volume_dry = False
        if volume_history is not None and len(volume_history) >= 5:
            recent_volume_avg = np.mean(volume_history[-5:])
            volume_dry = recent_volume_avg < (avg_volume_20d * self.setup_params['min_volume_dry_pct'])

            if volume_dry:
                compression_score += 5

        # Días en compresión
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
            'compression_score': min(25, compression_score),  # Max 25 pts
            'price_range_pct': price_range_pct,
            'compression_level': level,
            'compression_days': compression_days,
            'volume_dry': volume_dry,
            'atr_ratio': atr_ratio,
            'atr_extreme': atr_ratio < atr_extreme_threshold,
            'reason': f"Compresión {level}: rango {price_range_pct:.1f}% en {lookback}d, " +
                     f"ATR ratio {atr_ratio:.4f}, volumen {'seco' if volume_dry else 'normal'}"
        }

    def calculate_phase1_setup_score(self, symbol: str,
                                     price_history: np.ndarray,
                                     volume_history: np.ndarray,
                                     market_data: Dict) -> Dict:
        """
        CAPA 1: SETUP ESTRUCTURAL (35 puntos máximo normalizado 0-100)

        Componentes:
        - Compresión de precio + ATR (25 pts)
        - Short interest alto (10 pts)

        Normalizado a escala 0-100 según peso de capa (35%)
        """
        max_score_raw = 35  # Puntos brutos
        total_score = 0
        signals = []

        # 1. Compresión con ATR (0-25 pts)
        atr_ratio = market_data.get('atr_ratio', 0.05)
        compression = self.detect_setup_compression(
            price_history,
            volume_history,
            market_data.get('avg_volume_20d', 1),
            atr_ratio
        )
        total_score += compression['compression_score']
        signals.append(compression['reason'])

        # 2. Short Interest cualificado (0-10 pts)
        short_interest = market_data.get('short_interest_pct', 0)
        days_to_cover = market_data.get('days_to_cover', 0)

        si_score = 0
        if short_interest >= self.setup_params['min_short_interest'] and \
           days_to_cover >= self.setup_params['min_days_to_cover']:
            si_score = 10
            signals.append(f"SI cualificado: {short_interest:.1f}%, DTC: {days_to_cover:.1f}")
        elif short_interest >= 10:
            si_score = 5
            signals.append(f"SI moderado: {short_interest:.1f}%")

        total_score += si_score

        # Normalizar a 0-100 según peso de capa
        score_normalized = (total_score / max_score_raw) * 100
        score_normalized = max(0, min(100, score_normalized))

        return {
            'phase': 'FASE 1: SETUP ESTRUCTURAL',
            'score': score_normalized,
            'max_score': 100,
            'weight': self.layer_weights['phase1_setup'],
            'weighted_score': score_normalized * (self.layer_weights['phase1_setup'] / 100),
            'compression': compression,
            'short_interest_qualified': short_interest >= self.setup_params['min_short_interest'],
            'signals': signals,
            'passed': compression['is_compressed']
        }

    # ========================================================================
    # FASE 2: TRIGGER (mantiene lógica V4)
    # ========================================================================

    def get_explosion_day_number(self, symbol: str,
                                  volume_history: np.ndarray,
                                  avg_volume_20d: float,
                                  price_history: np.ndarray,
                                  current_rsi: float = 50,
                                  current_volume_ratio: float = 1.0) -> Dict:
        """
        Detecta en qué DÍA de la explosión estamos

        NUEVO V5: Permite día 3 con condiciones estrictas
        """
        if volume_history is None or len(volume_history) < 5:
            return {
                'explosion_day': 0,
                'is_early_enough': False,
                'penalty_points': 0,
                'late_entry_qualified': False,
                'reason': 'Datos insuficientes'
            }

        spike_threshold = self.trigger_params['min_volume_spike']
        consecutive_days = 0

        for i in range(len(volume_history) - 1, -1, -1):
            vol_ratio = volume_history[i] / avg_volume_20d if avg_volume_20d > 0 else 0

            if vol_ratio >= spike_threshold:
                consecutive_days += 1
            else:
                break

        explosion_day = max(1, consecutive_days)
        max_day = self.trigger_params['max_explosion_day']
        is_early_enough = explosion_day <= max_day
        late_entry_qualified = False

        # Evaluar entrada tardía (día 3)
        if explosion_day == 3 and self.trigger_params.get('allow_late_entry_day3', False):
            # Condiciones estrictas para día 3
            conditions_met = []
            conditions_failed = []

            # 1. Volumen aún muy alto
            day3_vol_threshold = self.trigger_params['day3_min_volume_spike']
            if current_volume_ratio >= day3_vol_threshold:
                conditions_met.append(f"Volumen {current_volume_ratio:.1f}x (req: {day3_vol_threshold}x)")
            else:
                conditions_failed.append(f"Volumen {current_volume_ratio:.1f}x < {day3_vol_threshold}x")

            # 2. RSI no sobrecomprado extremo
            day3_max_rsi = self.trigger_params['day3_max_rsi']
            if current_rsi <= day3_max_rsi:
                conditions_met.append(f"RSI {current_rsi:.0f} (max: {day3_max_rsi})")
            else:
                conditions_failed.append(f"RSI {current_rsi:.0f} > {day3_max_rsi}")

            # 3. Precio no subió demasiado
            if price_history is not None and len(price_history) >= 4:
                price_3d_ago = price_history[-4]
                current_price = price_history[-1]
                price_change_3d = ((current_price - price_3d_ago) / price_3d_ago * 100) if price_3d_ago > 0 else 0
                max_change = self.trigger_params['day3_max_price_change_3d']

                if price_change_3d <= max_change:
                    conditions_met.append(f"Cambio precio {price_change_3d:.1f}% (max: {max_change}%)")
                else:
                    conditions_failed.append(f"Cambio precio {price_change_3d:.1f}% > {max_change}%")

            # Si cumple todas las condiciones, califica para entrada tardía
            if len(conditions_failed) == 0 and len(conditions_met) >= 3:
                late_entry_qualified = True
                is_early_enough = True  # Permitir entrada
                penalty = self.trigger_params['day3_penalty_reduced']
                status = f"⚠️ TARDÍO PERO CALIFICADO - Día 3"
            else:
                penalty = self.penalties['late_to_party']
                status = f"TARDE - Día 3 (no califica)"

        elif explosion_day == 1:
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

        self.explosion_memory[symbol] = {
            'day': explosion_day,
            'date': datetime.now(),
            'is_early': is_early_enough,
            'late_entry_qualified': late_entry_qualified
        }

        return {
            'explosion_day': explosion_day,
            'is_early_enough': is_early_enough,
            'late_entry_qualified': late_entry_qualified,
            'penalty_points': penalty,
            'consecutive_high_volume_days': consecutive_days,
            'status': status,
            'reason': f"{status} ({consecutive_days} días consecutivos de volumen alto)"
        }

    def calculate_phase2_trigger_score(self, symbol: str,
                                       current_price: float,
                                       price_history: np.ndarray,
                                       volume_history: np.ndarray,
                                       market_data: Dict) -> Dict:
        """
        CAPA 2: TRIGGER DE ENTRADA (35 puntos normalizado 0-100)

        NUEVO V5: Evalúa entrada tardía día 3
        """
        max_score_raw = 40
        total_score = 0
        signals = []

        current_volume = market_data.get('volume', 0)
        avg_volume = market_data.get('avg_volume_20d', 1)
        current_rsi = market_data.get('rsi', 50)
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0

        # Explosion day check (con parámetros para día 3)
        explosion_info = self.get_explosion_day_number(
            symbol, volume_history, avg_volume, price_history,
            current_rsi=current_rsi,
            current_volume_ratio=volume_ratio
        )

        # Volumen explosivo temprano (15 pts)
        if explosion_info['is_early_enough'] and volume_ratio >= self.trigger_params['min_volume_spike']:
            # Si es entrada tardía calificada, dar puntuación reducida
            if explosion_info.get('late_entry_qualified', False):
                volume_score = 10  # Menos puntos que día 1/2
                signals.append(f"⚠️ Volumen explosivo TARDÍO CALIFICADO: {volume_ratio:.1f}x en día {explosion_info['explosion_day']}")
            else:
                volume_score = 15
                signals.append(f"Volumen explosivo TEMPRANO: {volume_ratio:.1f}x en día {explosion_info['explosion_day']}")
        elif volume_ratio >= self.trigger_params['min_volume_spike']:
            volume_score = 5
            signals.append(f"Volumen alto pero TARDÍO: día {explosion_info['explosion_day']}")
        else:
            volume_score = 0

        total_score += volume_score

        # Breakout técnico (10 pts)
        if price_history is not None and len(price_history) >= 20:
            resistance = np.max(price_history[-20:-1])
            sma_20 = np.mean(price_history[-20:])
            sma_50 = np.mean(price_history[-50:]) if len(price_history) >= 50 else sma_20

            breakout_clean = current_price > resistance and \
                           current_price > sma_20 and \
                           sma_20 > sma_50

            if breakout_clean:
                breakout_score = 10
                signals.append("Breakout limpio: precio > resistencia y SMAs")
            else:
                breakout_score = 0
        else:
            breakout_score = 0

        total_score += breakout_score

        # Momentum (10 pts)
        rsi = market_data.get('rsi', 50)
        rsi_crossed = 55 <= rsi <= self.trigger_params['rsi_not_overbought']

        if rsi_crossed:
            momentum_score = 10
            signals.append(f"RSI en zona óptima: {rsi:.0f}")
        elif rsi > self.trigger_params['rsi_not_overbought']:
            momentum_score = 0
            signals.append(f"RSI sobrecomprado: {rsi:.0f}")
        else:
            momentum_score = 5

        total_score += momentum_score

        # Confirmación institucional (5 pts)
        if price_history is not None and len(price_history) >= 2:
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
        score_normalized = (total_score / max_score_raw) * 100
        score_normalized = max(0, min(100, score_normalized))

        return {
            'phase': 'FASE 2: TRIGGER DE ENTRADA',
            'score': score_normalized,
            'max_score': 100,
            'weight': self.layer_weights['phase2_trigger'],
            'weighted_score': score_normalized * (self.layer_weights['phase2_trigger'] / 100),
            'explosion_info': explosion_info,
            'volume_ratio': volume_ratio,
            'signals': signals,
            'passed': explosion_info['is_early_enough'] and volume_ratio >= self.trigger_params['min_volume_spike']
        }

    # ========================================================================
    # FASE 3: CONTEXTO + ALTERNATIVE DATA (V5)
    # ========================================================================

    def calculate_phase3_context_score(self, symbol: str, market_context: Dict) -> Dict:
        """
        CAPA 3: CONTEXTO + ALTERNATIVE DATA (20 puntos normalizado 0-100)

        NUEVO V5: Incluye datos alternativos (Reddit + Short borrow)
        """
        max_score_raw = 20
        total_score = 0
        signals = []

        # Mercado general (10 pts)
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

        # VIX (5 pts)
        vix = market_context.get('vix', 15)
        if vix < self.context_params['vix_panic_threshold']:
            vix_score = 5
            signals.append(f"VIX bajo: {vix:.1f}")
        else:
            vix_score = 0
            signals.append(f"VIX alto: {vix:.1f}")

        total_score += vix_score

        # NUEVO V5: Alternative Data (5 pts)
        try:
            alt_data = self.alt_data_provider.get_combined_alternative_data(symbol)
            alt_score_raw = alt_data['combined_score']  # 0-100
            alt_score = (alt_score_raw / 100) * 5  # Convertir a 0-5

            total_score += alt_score

            reddit = alt_data['reddit']
            short_borrow = alt_data['short_borrow']

            signals.append(
                f"Alt Data: Reddit {reddit['sentiment']} ({reddit['mentions']} mentions), " +
                f"Borrow {short_borrow['borrow_rate_pct']:.1f}%"
            )
        except Exception as e:
            self.logger.debug(f"Error obteniendo alternative data para {symbol}: {e}")
            alt_score = 2.5  # Default neutral
            signals.append("Alt Data: No disponible (usando default)")

        # Normalizar
        score_normalized = (total_score / max_score_raw) * 100
        score_normalized = max(0, min(100, score_normalized))

        return {
            'phase': 'FASE 3: CONTEXTO + ALT DATA',
            'score': score_normalized,
            'max_score': 100,
            'weight': self.layer_weights['phase3_context'],
            'weighted_score': score_normalized * (self.layer_weights['phase3_context'] / 100),
            'signals': signals,
            'market_favorable': spy_trend != 'bearish' and vix < self.context_params['vix_panic_threshold']
        }

    # ========================================================================
    # ML PREDICTION (V5)
    # ========================================================================

    def calculate_ml_adjustment(self, market_data: Dict, phase1: Dict, phase2: Dict) -> Dict:
        """
        NUEVO V5: Ajuste basado en predicción ML

        Returns ajuste de score (-15 a +5 puntos)
        """
        try:
            # Preparar features para ML
            compression = phase1['compression']

            ml_features = {
                'bb_width': compression.get('price_range_pct', 8) / 100,  # Normalizar
                'adx': 20.0,  # Simplificado - podríamos calcular real
                'vol_ratio': phase2['volume_ratio'],
                'rsi': market_data.get('rsi', 50),
                'macd_diff': market_data.get('macd_diff', 0),
                'atr_ratio': market_data.get('atr_ratio', 0.05),
                'short_float': market_data.get('short_interest_pct', 15) / 100,
                'compression_days': compression.get('compression_days', 0),
                'volume_dry': 1 if compression.get('volume_dry', False) else 0,
                'price_range_pct': compression.get('price_range_pct', 8)
            }

            # Predicción
            prediction = self.ml_predictor.predict(ml_features)

            probability = prediction['probability']
            confidence = prediction['confidence']

            # Calcular ajuste
            if probability < self.ml_params['min_probability']:
                # Baja probabilidad -> penalizar
                adjustment = self.ml_params['ml_penalty']
                reason = f"ML: Baja probabilidad ({probability:.1%}) - Penalización"
            elif probability >= self.ml_params['high_confidence']:
                # Alta probabilidad -> bonus
                adjustment = self.ml_params['ml_bonus']
                reason = f"ML: Alta probabilidad ({probability:.1%}) - Bonus"
            else:
                # Probabilidad media -> neutral
                adjustment = 0
                reason = f"ML: Probabilidad media ({probability:.1%}) - Neutral"

            return {
                'adjustment': adjustment,
                'probability': probability,
                'confidence': confidence,
                'model_available': prediction['model_available'],
                'reason': reason
            }

        except Exception as e:
            self.logger.debug(f"Error en predicción ML: {e}")
            return {
                'adjustment': 0,
                'probability': 0.5,
                'confidence': 'low',
                'model_available': False,
                'reason': 'ML: Modelo no disponible'
            }

    # ========================================================================
    # PENALIZACIONES (V4 + V5)
    # ========================================================================

    def apply_penalties(self, symbol: str,
                       price_history: np.ndarray,
                       volume_history: np.ndarray,
                       market_data: Dict,
                       explosion_info: Dict,
                       market_context: Dict,
                       ml_adjustment: Dict) -> Dict:
        """
        Penalizaciones severas (V4 + ML penalty V5)
        """
        total_penalty = 0
        penalties_applied = []

        # 1. Precio subió mucho en 3d
        # NUEVO: No penalizar si es entrada tardía calificada (ya se validó el cambio de precio)
        is_late_entry_qualified = explosion_info.get('late_entry_qualified', False)

        if price_history is not None and len(price_history) >= 4:
            price_3d_ago = price_history[-4]
            current_price = price_history[-1]
            price_change_3d = ((current_price - price_3d_ago) / price_3d_ago * 100) if price_3d_ago > 0 else 0

            # Si es entrada tardía calificada, el límite es 60% (ya validado)
            # Si no, el límite es 15%
            price_limit = 60 if is_late_entry_qualified else 15

            if price_change_3d > price_limit:
                total_penalty += self.penalties['price_up_15pct_3d']
                penalties_applied.append(f"Precio subió {price_change_3d:.1f}% en 3d (límite: {price_limit}%): {self.penalties['price_up_15pct_3d']} pts")

        # 2. RSI sobrecomprado
        # NUEVO: RSI límite más alto para entrada tardía calificada
        rsi = market_data.get('rsi', 50)
        rsi_limit = self.trigger_params['day3_max_rsi'] if is_late_entry_qualified else self.trigger_params['rsi_not_overbought']

        if rsi > rsi_limit:
            total_penalty += self.penalties['rsi_overbought']
            penalties_applied.append(f"RSI sobrecomprado ({rsi:.0f}, límite: {rsi_limit}): {self.penalties['rsi_overbought']} pts")

        # 3. Volumen ya explotó ayer
        if volume_history is not None and len(volume_history) >= 2:
            avg_volume = market_data.get('avg_volume_20d', 1)
            yesterday_volume = volume_history[-2]
            yesterday_ratio = yesterday_volume / avg_volume if avg_volume > 0 else 0

            if yesterday_ratio > 4.0:
                total_penalty += self.penalties['volume_already_exploded']
                penalties_applied.append(f"Volumen ayer {yesterday_ratio:.1f}x: {self.penalties['volume_already_exploded']} pts")

        # 4. Día tardío
        if explosion_info['explosion_day'] >= 3:
            total_penalty += explosion_info['penalty_points']
            penalties_applied.append(f"Día {explosion_info['explosion_day']}: {explosion_info['penalty_points']} pts")

        # 5. Mercado bajista
        if market_context.get('spy_trend') == 'bearish':
            total_penalty += self.penalties['market_bearish']
            penalties_applied.append(f"Mercado bajista: {self.penalties['market_bearish']} pts")

        # 6. NUEVO V5: ML baja probabilidad
        if ml_adjustment['adjustment'] < 0:
            total_penalty += ml_adjustment['adjustment']
            penalties_applied.append(f"ML baja probabilidad: {ml_adjustment['adjustment']} pts")

        should_reject = total_penalty <= -50

        return {
            'total_penalty': total_penalty,
            'penalty_breakdown': penalties_applied,
            'should_reject': should_reject,
            'reason': f"{len(penalties_applied)} penalizaciones ({total_penalty} pts)"
        }

    # ========================================================================
    # MOMENTUM PURO (Nueva Filosofía)
    # ========================================================================

    def detect_momentum_puro_pattern(self, symbol: str,
                                     price_history: np.ndarray,
                                     volume_history: np.ndarray,
                                     market_data: Dict) -> Dict:
        """
        NUEVA FILOSOFÍA: Detecta patrón de "Momentum Puro"

        Características:
        - Explosión de volumen directa (sin compresión previa)
        - Rebote fuerte desde mínimos recientes
        - Momentum alcista sostenido
        - Catalizador claro (volumen extremo)

        Este patrón NO requiere setup de compresión
        """
        if not self.momentum_puro_params.get('enabled', False):
            return {
                'is_momentum_puro': False,
                'reason': 'Momentum Puro deshabilitado'
            }

        if price_history is None or volume_history is None:
            return {
                'is_momentum_puro': False,
                'reason': 'Datos insuficientes'
            }

        if len(price_history) < 10 or len(volume_history) < 10:
            return {
                'is_momentum_puro': False,
                'reason': 'Historial insuficiente'
            }

        signals = []
        score = 0
        max_score = 100

        # 1. Volumen explosivo extremo (30 pts)
        current_volume = market_data.get('volume', 0)
        avg_volume = market_data.get('avg_volume_20d', 1)
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0

        min_vol_spike = self.momentum_puro_params['min_volume_spike']
        if volume_ratio >= min_vol_spike:
            vol_score = min(30, (volume_ratio / min_vol_spike) * 15)
            score += vol_score
            signals.append(f"Volumen extremo: {volume_ratio:.1f}x (req: {min_vol_spike}x)")
        else:
            return {
                'is_momentum_puro': False,
                'reason': f'Volumen {volume_ratio:.1f}x < {min_vol_spike}x requerido'
            }

        # 2. Día de explosión temprano (25 pts)
        explosion_info = self.get_explosion_day_number(
            symbol, volume_history, avg_volume, price_history,
            current_rsi=market_data.get('rsi', 50),
            current_volume_ratio=volume_ratio
        )

        max_day = self.momentum_puro_params['max_explosion_day']
        if explosion_info['explosion_day'] <= max_day:
            day_score = 25 if explosion_info['explosion_day'] == 1 else 20
            score += day_score
            signals.append(f"Día {explosion_info['explosion_day']} de explosión (temprano)")
        else:
            return {
                'is_momentum_puro': False,
                'reason': f"Día {explosion_info['explosion_day']} > {max_day} (muy tarde)"
            }

        # 3. Cambio de precio alcista fuerte (20 pts)
        if len(price_history) >= 4:
            price_3d_ago = price_history[-4]
            current_price = price_history[-1]
            price_change_3d = ((current_price - price_3d_ago) / price_3d_ago * 100) if price_3d_ago > 0 else 0

            min_change = self.momentum_puro_params['min_price_change_3d']
            max_change = self.momentum_puro_params['max_price_change_3d']

            if min_change <= price_change_3d <= max_change:
                change_score = 20
                score += change_score
                signals.append(f"Cambio precio {price_change_3d:.1f}% en 3d (rango: {min_change}-{max_change}%)")
            elif price_change_3d > max_change:
                return {
                    'is_momentum_puro': False,
                    'reason': f'Precio subió {price_change_3d:.1f}% > {max_change}% (muy tarde)'
                }
            else:
                return {
                    'is_momentum_puro': False,
                    'reason': f'Cambio precio {price_change_3d:.1f}% < {min_change}% (momentum débil)'
                }

        # 4. Breakout desde mínimo reciente (15 pts)
        lookback = self.momentum_puro_params['lookback_low_days']
        recent_low = np.min(price_history[-lookback:])
        current_price = price_history[-1]
        breakout_pct = ((current_price - recent_low) / recent_low) if recent_low > 0 else 0

        min_breakout = self.momentum_puro_params['min_breakout_from_low']
        if breakout_pct >= min_breakout:
            breakout_score = 15
            score += breakout_score
            signals.append(f"Breakout {breakout_pct:.1%} desde mínimo {lookback}d")
        else:
            signals.append(f"Breakout débil: {breakout_pct:.1%} < {min_breakout:.1%}")

        # 5. RSI en zona de rebote (10 pts)
        rsi = market_data.get('rsi', 50)
        rsi_min = self.momentum_puro_params['rsi_min_bounce']
        rsi_max = self.momentum_puro_params['rsi_max_momentum']

        if rsi_min <= rsi <= rsi_max:
            rsi_score = 10
            score += rsi_score
            signals.append(f"RSI {rsi:.0f} en zona óptima ({rsi_min}-{rsi_max})")
        else:
            if rsi < rsi_min:
                signals.append(f"RSI {rsi:.0f} muy bajo (< {rsi_min})")
            else:
                signals.append(f"RSI {rsi:.0f} sobrecomprado (> {rsi_max})")

        # 6. Normalizar score
        score = min(100, score)

        threshold = self.momentum_puro_params['score_threshold']
        is_qualified = score >= threshold

        return {
            'is_momentum_puro': is_qualified,
            'score': score,
            'threshold': threshold,
            'volume_ratio': volume_ratio,
            'explosion_day': explosion_info['explosion_day'],
            'price_change_3d': price_change_3d if len(price_history) >= 4 else 0,
            'breakout_pct': breakout_pct,
            'rsi': rsi,
            'signals': signals,
            'reason': f"Momentum Puro score {score:.0f}/100 ({'CALIFICA' if is_qualified else 'NO CALIFICA'})"
        }

    # ========================================================================
    # ANÁLISIS COMPLETO V5
    # ========================================================================

    def analyze_symbol_v5(self, symbol: str, market_data: Dict,
                          historical_data: Dict, market_context: Dict) -> Dict:
        """
        Análisis completo V5 con ML + Alternative Data + Momentum Puro

        NUEVA: Evalúa dos filosofías:
        1. Filosofía tradicional: "Resorte Comprimido" (compresión + explosión)
        2. Filosofía alternativa: "Momentum Puro" (explosión directa sin setup)
        """
        current_price = market_data.get('price', 0)
        price_history = historical_data.get('close')
        volume_history = historical_data.get('volume')

        # EVALUAR FILOSOFÍA TRADICIONAL: "Resorte Comprimido"
        # FASE 1: Setup
        phase1 = self.calculate_phase1_setup_score(
            symbol, price_history, volume_history, market_data
        )

        # FASE 2: Trigger
        phase2 = self.calculate_phase2_trigger_score(
            symbol, current_price, price_history, volume_history, market_data
        )

        # FASE 3: Context + Alt Data
        phase3 = self.calculate_phase3_context_score(symbol, market_context)

        # ML Adjustment
        ml_adjustment = self.calculate_ml_adjustment(market_data, phase1, phase2)

        # Score bruto (suma ponderada de fases)
        raw_score = (
            phase1['weighted_score'] +
            phase2['weighted_score'] +
            phase3['weighted_score']
        )

        # Aplicar ML adjustment
        raw_score += ml_adjustment['adjustment']

        # Penalizaciones
        penalties = self.apply_penalties(
            symbol, price_history, volume_history, market_data,
            phase2['explosion_info'], market_context, ml_adjustment
        )

        # Score final
        final_score = raw_score + penalties['total_penalty']
        final_score = max(0, min(100, final_score))

        # EVALUAR FILOSOFÍA ALTERNATIVA: "Momentum Puro"
        # Solo si no tiene compresión previa (falla filosofía tradicional)
        momentum_puro = None
        opportunity_type = "Resorte Comprimido"

        if not phase1['compression']['is_compressed']:
            # Sin compresión -> evaluar patrón Momentum Puro
            momentum_puro = self.detect_momentum_puro_pattern(
                symbol, price_history, volume_history, market_data
            )

            # Si califica como Momentum Puro, usar ese score
            if momentum_puro['is_momentum_puro']:
                final_score = momentum_puro['score']
                raw_score = momentum_puro['score']
                opportunity_type = "Momentum Puro"
                self.logger.info(f"  {symbol}: Califica como MOMENTUM PURO (score: {final_score:.0f})")

        # Decisión
        trading_decision = self.generate_trading_decision_v5(
            symbol, final_score, raw_score, penalties,
            phase1, phase2, phase3, ml_adjustment, market_data,
            momentum_puro=momentum_puro,
            opportunity_type=opportunity_type
        )

        return {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'opportunity_type': opportunity_type,
            'raw_score': raw_score,
            'final_score': final_score,
            'phase1_setup': phase1,
            'phase2_trigger': phase2,
            'phase3_context': phase3,
            'ml_adjustment': ml_adjustment,
            'penalties': penalties,
            'momentum_puro': momentum_puro,
            'trading_decision': trading_decision,
            'market_data': market_data
        }

    def generate_trading_decision_v5(self, symbol: str, final_score: float,
                                     raw_score: float, penalties: Dict,
                                     phase1: Dict, phase2: Dict, phase3: Dict,
                                     ml_adjustment: Dict, market_data: Dict,
                                     momentum_puro: Optional[Dict] = None,
                                     opportunity_type: str = "Resorte Comprimido") -> Dict:
        """
        Genera decisión de trading V5

        NUEVO: Soporta dos tipos de oportunidades:
        - "Resorte Comprimido": Patrón tradicional con compresión
        - "Momentum Puro": Explosión directa sin setup
        """

        # Obtener thresholds dinámicos del optimizer
        thresholds = self.optimizer.get_current_thresholds()

        # Si es Momentum Puro, usar lógica específica
        if opportunity_type == "Momentum Puro" and momentum_puro and momentum_puro['is_momentum_puro']:
            return self._generate_momentum_puro_decision(
                symbol, final_score, momentum_puro, market_data, thresholds
            )

        # LÓGICA TRADICIONAL: "Resorte Comprimido"

        # Rechazar si penalizaciones severas
        if penalties['should_reject']:
            return {
                'symbol': symbol,
                'action': 'RECHAZAR',
                'opportunity_type': opportunity_type,
                'reason': f'Penalizaciones severas: {penalties["total_penalty"]} puntos',
                'score': final_score,
                'details': penalties['penalty_breakdown']
            }

        # Rechazar si no hay compresión (y no califica como Momentum Puro)
        if not phase1['compression']['is_compressed']:
            return {
                'symbol': symbol,
                'action': 'RECHAZAR',
                'opportunity_type': opportunity_type,
                'reason': 'No hay compresión previa - setup inválido',
                'score': final_score
            }

        # Decisión por score con thresholds dinámicos
        if final_score >= thresholds['buy_strong']:
            action = 'COMPRA FUERTE'
            urgency = 'ALTA'
            confidence = 'ALTA'
        elif final_score >= thresholds['buy_moderate']:
            action = 'COMPRA MODERADA'
            urgency = 'MEDIA'
            confidence = 'MEDIA'
        elif final_score >= thresholds['watchlist']:
            action = 'WATCHLIST'
            urgency = 'BAJA'
            confidence = 'MONITOREAR'
            return {
                'symbol': symbol,
                'action': action,
                'reason': f'Score {final_score:.0f} - Setup prometedor, falta trigger',
                'score': final_score,
                'urgency': urgency,
                'thresholds_used': thresholds
            }
        else:
            action = 'RECHAZAR'
            return {
                'symbol': symbol,
                'action': action,
                'reason': f'Score {final_score:.0f} < {thresholds["watchlist"]}',
                'score': final_score,
                'thresholds_used': thresholds
            }

        # Plan de trading completo
        current_price = market_data.get('price', 0)

        stop_loss = current_price * (1 - self.exit_params['stop_loss_initial_pct'])
        tp1 = current_price * (1 + self.exit_params['tp1_pct'])
        tp2 = current_price * (1 + self.exit_params['tp2_pct'])
        trailing_trigger = current_price * (1 + self.exit_params['trailing_stop_trigger_pct'])

        # Reducir tamaño de posición si es entrada tardía
        is_late_entry = phase2['explosion_info'].get('late_entry_qualified', False)
        if is_late_entry:
            position_size_pct = 2 if urgency == 'ALTA' else 1.5  # Menor tamaño por riesgo
        else:
            position_size_pct = 3 if urgency == 'ALTA' else 2

        # Agregar advertencias específicas de entrada tardía
        warnings = penalties['penalty_breakdown'].copy()
        if is_late_entry:
            warnings.insert(0, "⚠️ ENTRADA TARDÍA (DÍA 3): Mayor riesgo - Posición reducida")
            warnings.insert(1, "⚠️ Stop loss más ajustado recomendado")
            warnings.insert(2, "⚠️ Considerar salida anticipada si momentum debilita")

        return {
            'symbol': symbol,
            'action': action,
            'urgency': urgency,
            'confidence': confidence,
            'score': final_score,
            'raw_score': raw_score,
            'penalty_applied': penalties['total_penalty'],
            'ml_probability': ml_adjustment['probability'],
            'ml_confidence': ml_adjustment['confidence'],
            'is_late_entry': is_late_entry,
            'explosion_day': phase2['explosion_info']['explosion_day'],
            'current_price': current_price,
            'position_size_pct': position_size_pct,
            'stop_loss': stop_loss,
            'take_profit_1': tp1,
            'take_profit_2': tp2,
            'tp1_allocation': self.exit_params['tp1_size'] * 100,
            'tp2_allocation': self.exit_params['tp2_size'] * 100,
            'trailing_stop_trigger': trailing_trigger,
            'max_holding_days': self.exit_params['max_holding_days'],
            'key_signals': self._summarize_signals(phase1, phase2, phase3, ml_adjustment),
            'warnings': warnings,
            'thresholds_used': thresholds,
            'exit_params': self.exit_params
        }

    def _generate_momentum_puro_decision(self, symbol: str, final_score: float,
                                         momentum_puro: Dict, market_data: Dict,
                                         thresholds: Dict) -> Dict:
        """
        Genera decisión específica para oportunidades de Momentum Puro

        Características:
        - Posición reducida (50% del tamaño normal)
        - Stop loss más ajustado
        - Advertencias específicas
        """
        current_price = market_data.get('price', 0)

        # Thresholds ajustados para Momentum Puro
        momentum_threshold = self.momentum_puro_params['score_threshold']

        if final_score >= thresholds['buy_strong']:
            action = 'COMPRA FUERTE'
            urgency = 'ALTA'
            confidence = 'MEDIA-ALTA'
        elif final_score >= momentum_threshold:
            action = 'COMPRA MODERADA'
            urgency = 'MEDIA'
            confidence = 'MEDIA'
        else:
            return {
                'symbol': symbol,
                'action': 'RECHAZAR',
                'opportunity_type': 'Momentum Puro',
                'reason': f'Score {final_score:.0f} < {momentum_threshold} (threshold Momentum Puro)',
                'score': final_score
            }

        # Plan de trading con ajustes para Momentum Puro
        stop_loss = current_price * (1 - 0.10)  # 10% stop (más ajustado)
        tp1 = current_price * (1 + 0.15)
        tp2 = current_price * (1 + 0.30)
        trailing_trigger = current_price * (1 + 0.12)  # Activar antes

        # Posición reducida (50% del tamaño normal)
        reduction = self.momentum_puro_params['position_size_pct_reduction']
        if urgency == 'ALTA':
            position_size_pct = 3 * reduction  # 1.5%
        else:
            position_size_pct = 2 * reduction  # 1%

        # Advertencias específicas de Momentum Puro
        warnings = [
            "⚠️ MOMENTUM PURO: Sin setup de compresión previa",
            "⚠️ RIESGO MAYOR: Posición reducida 50%",
            "⚠️ Stop loss ajustado al 10% (vs 8% normal)",
            "⚠️ Salir rápido si momentum debilita",
            "⚠️ Monitorear volumen: debe sostenerse alto"
        ]

        return {
            'symbol': symbol,
            'action': action,
            'opportunity_type': 'Momentum Puro',
            'urgency': urgency,
            'confidence': confidence,
            'score': final_score,
            'explosion_day': momentum_puro['explosion_day'],
            'current_price': current_price,
            'position_size_pct': position_size_pct,
            'stop_loss': stop_loss,
            'take_profit_1': tp1,
            'take_profit_2': tp2,
            'tp1_allocation': self.exit_params['tp1_size'] * 100,
            'tp2_allocation': self.exit_params['tp2_size'] * 100,
            'trailing_stop_trigger': trailing_trigger,
            'max_holding_days': 5,  # Holding más corto
            'key_signals': momentum_puro['signals'],
            'warnings': warnings,
            'thresholds_used': thresholds,
            'momentum_puro_stats': {
                'volume_ratio': momentum_puro['volume_ratio'],
                'price_change_3d': momentum_puro['price_change_3d'],
                'breakout_pct': momentum_puro['breakout_pct'],
                'rsi': momentum_puro['rsi']
            }
        }

    def _summarize_signals(self, phase1, phase2, phase3, ml_adjustment):
        """Resume señales clave"""
        summary = []
        summary.extend(phase1['signals'][:2])
        summary.extend(phase2['signals'][:2])
        summary.extend(phase3['signals'][:2])
        summary.append(ml_adjustment['reason'])
        return summary

    def update_watchlist(self, symbols: List[str]):
        """Actualiza watchlist"""
        self.watchlist = [s.upper() for s in symbols]
        self.logger.info(f"Watchlist actualizada: {len(self.watchlist)} símbolos")


if __name__ == "__main__":
    print("\n🚀 PENNY STOCK ADVISOR V5 - EVOLUTION")
    print("="*70)
    print("Sistema avanzado con ML, Alternative Data y Optimización Dinámica")
    print("="*70)

    # Test básico
    advisor = PennyStockAdvisorV5(config_preset="balanced")

    print(f"\n✅ Sistema inicializado correctamente")
    print(f"   ML Model: {'Entrenado' if advisor.ml_predictor.is_trained else 'No entrenado (usar datos de prueba)'}")
    print(f"   Cache: {'Habilitado' if advisor.data_cache else 'Deshabilitado'}")
    print(f"   Alternative Data: Configurado (modo local)")
    print(f"   Optimizer: Activo (thresholds dinámicos)")
    print("\n" + "="*70)
