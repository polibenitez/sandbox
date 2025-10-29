#!/usr/bin/env python3
"""
PENNY STOCK ROBOT ADVISOR V3 - ENHANCED MOMENTUM & VOLUME ANALYSIS
=================================================================

MEJORAS CLAVE vs V2:
1. ✅ Detección de EXPLOSIÓN DE VOLUMEN (aceleración intradiaria)
2. ✅ Análisis de BREAKOUT TÉCNICO con volumen
3. ✅ Take Profits DINÁMICOS basados en momentum
4. ✅ Trailing Stop Loss inteligente
5. ✅ Score de URGENCIA para squeeze inminente
6. ✅ Detección de patrón "squeeze setup" (precio comprimido + volumen explosivo)

Caso de uso: BYND subió 150% - Este algoritmo lo habría detectado mejor
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

class PennyStockAdvisorV3Enhanced:
    """
    Robot Advisor V3 - Especializado en detectar squeezes ANTES de que exploten
    
    Piensa en el mercado como un resorte: mientras más comprimido (alto short interest,
    bajo volumen), más violenta es la explosión cuando se suelta (volumen masivo).
    """
    
    def __init__(self, config_preset="balanced"):
        """
        Inicializa el robot con configuración seleccionada
        
        Args:
            config_preset (str): "conservative", "balanced", "aggressive", "very_aggressive"
        """
        self.watchlist = []
        self.config_preset = config_preset
        self.load_configuration(config_preset)
    
    def load_configuration(self, preset):
        """Carga configuración según el preset seleccionado"""
        
        configurations = {
            "conservative": {
                "description": "Configuración conservadora - alta precisión",
                "signals_weights": {
                    'short_interest_qualified': 0.20,
                    'volume_explosion': 0.25,           # NUEVO - Peso alto
                    'momentum_breakout': 0.20,          # NUEVO
                    'price_compression': 0.15,          # NUEVO
                    'liquidity_filter': 0.10,
                    'technical_breakout': 0.10          # NUEVO
                },
                "thresholds": {
                    'buy_strong': 0.80,
                    'buy_moderate': 0.65,
                    'buy_light': 0.50,
                    'urgent_squeeze': 0.85              # NUEVO - Para casos como BYND
                },
                "signal_params": {
                    'min_volume_spike': 3.0,            # 3x volumen promedio
                    'min_volume_acceleration': 1.5,     # 50% aceleración intradiaria
                    'min_price_compression_days': 5,    # Días en rango estrecho
                    'compression_range_pct': 15,        # Rango máximo para "compresión"
                    'breakout_volume_threshold': 2.5,   # Volumen para confirmar breakout
                    'rsi_oversold': 35,
                    'rsi_overbought': 70,
                    'max_spread_pct': 10
                },
                "risk_params": {
                    'base_position_size': 0.02,
                    'max_position_size_urgent': 0.04,   # NUEVO - Para squeezes urgentes
                    'stop_loss_atr_multiplier': 2.0,
                    'trailing_stop_trigger': 0.15,      # NUEVO - Activar trailing a +15%
                    'trailing_stop_distance': 0.08,     # NUEVO - 8% desde máximo
                    # Take profits DINÁMICOS
                    'take_profit_normal': [2, 3, 5],    # Multiplicadores ATR normales
                    'take_profit_squeeze': [3, 5, 8],   # NUEVO - Para squeeze confirmado
                    'take_profit_urgent': [4, 7, 12]    # NUEVO - Para squeeze urgente (caso BYND)
                }
            },
            
            "balanced": {
                "description": "Configuración balanceada - óptimo para penny stocks",
                "signals_weights": {
                    'short_interest_qualified': 0.18,
                    'volume_explosion': 0.28,           # Aumentado
                    'momentum_breakout': 0.22,
                    'price_compression': 0.15,
                    'liquidity_filter': 0.08,
                    'technical_breakout': 0.09
                },
                "thresholds": {
                    'buy_strong': 0.75,
                    'buy_moderate': 0.60,
                    'buy_light': 0.45,
                    'urgent_squeeze': 0.80
                },
                "signal_params": {
                    'min_volume_spike': 2.5,
                    'min_volume_acceleration': 1.3,
                    'min_price_compression_days': 4,
                    'compression_range_pct': 18,
                    'breakout_volume_threshold': 2.0,
                    'rsi_oversold': 30,
                    'rsi_overbought': 75,
                    'max_spread_pct': 12
                },
                "risk_params": {
                    'base_position_size': 0.025,
                    'max_position_size_urgent': 0.05,
                    'stop_loss_atr_multiplier': 1.8,
                    'trailing_stop_trigger': 0.12,
                    'trailing_stop_distance': 0.07,
                    'take_profit_normal': [2, 3, 5],
                    'take_profit_squeeze': [3, 6, 10],
                    'take_profit_urgent': [5, 9, 15]
                }
            },
            
            "aggressive": {
                "description": "Configuración agresiva - detecta más oportunidades early",
                "signals_weights": {
                    'short_interest_qualified': 0.15,
                    'volume_explosion': 0.30,
                    'momentum_breakout': 0.25,
                    'price_compression': 0.15,
                    'liquidity_filter': 0.05,
                    'technical_breakout': 0.10
                },
                "thresholds": {
                    'buy_strong': 0.70,
                    'buy_moderate': 0.55,
                    'buy_light': 0.40,
                    'urgent_squeeze': 0.75
                },
                "signal_params": {
                    'min_volume_spike': 2.0,
                    'min_volume_acceleration': 1.2,
                    'min_price_compression_days': 3,
                    'compression_range_pct': 20,
                    'breakout_volume_threshold': 1.8,
                    'rsi_oversold': 25,
                    'rsi_overbought': 80,
                    'max_spread_pct': 15
                },
                "risk_params": {
                    'base_position_size': 0.03,
                    'max_position_size_urgent': 0.06,
                    'stop_loss_atr_multiplier': 1.6,
                    'trailing_stop_trigger': 0.10,
                    'trailing_stop_distance': 0.06,
                    'take_profit_normal': [2, 3, 5],
                    'take_profit_squeeze': [4, 7, 12],
                    'take_profit_urgent': [6, 12, 20]
                }
            },
            
            "very_aggressive": {
                "description": "Configuración muy agresiva - máxima sensibilidad",
                "signals_weights": {
                    'short_interest_qualified': 0.12,
                    'volume_explosion': 0.35,
                    'momentum_breakout': 0.28,
                    'price_compression': 0.12,
                    'liquidity_filter': 0.05,
                    'technical_breakout': 0.08
                },
                "thresholds": {
                    'buy_strong': 0.65,
                    'buy_moderate': 0.50,
                    'buy_light': 0.35,
                    'urgent_squeeze': 0.70
                },
                "signal_params": {
                    'min_volume_spike': 1.8,
                    'min_volume_acceleration': 1.15,
                    'min_price_compression_days': 3,
                    'compression_range_pct': 22,
                    'breakout_volume_threshold': 1.5,
                    'rsi_oversold': 20,
                    'rsi_overbought': 85,
                    'max_spread_pct': 18
                },
                "risk_params": {
                    'base_position_size': 0.035,
                    'max_position_size_urgent': 0.07,
                    'stop_loss_atr_multiplier': 1.5,
                    'trailing_stop_trigger': 0.08,
                    'trailing_stop_distance': 0.05,
                    'take_profit_normal': [2, 3, 5],
                    'take_profit_squeeze': [5, 9, 15],
                    'take_profit_urgent': [8, 15, 25]
                }
            }
        }
        
        if preset not in configurations:
            preset = "balanced"
        
        config = configurations[preset]
        self.signals_weights = config["signals_weights"]
        self.thresholds = config["thresholds"]
        self.signal_params = config["signal_params"]
        self.risk_params = config["risk_params"]
        self.description = config["description"]
        
        print(f"🔧 Robot Advisor V3 Enhanced cargado: {preset.upper()}")
        print(f"📝 {self.description}")
        print(f"🚀 Nuevas capacidades: Detección de volumen explosivo + breakouts")
    
    def update_watchlist(self, symbols):
        """Actualiza la lista de símbolos a analizar"""
        self.watchlist = [s.upper() for s in symbols]
    
    # ========================================================================
    # NUEVAS SEÑALES - DETECCIÓN DE EXPLOSIÓN Y MOMENTUM
    # ========================================================================
    
    def analyze_volume_explosion(self, symbol, current_volume, avg_volume_20d, 
                                 volume_history=None, price_history=None):
        """
        🚨 SEÑAL CLAVE #1: Detección de EXPLOSIÓN DE VOLUMEN
        
        Analogía: Como detectar cuando una olla a presión está por explotar.
        No solo miramos si hay más volumen, sino si se está ACELERANDO.
        
        Returns:
            dict: {
                'score': 0-1,
                'volume_ratio': float,
                'acceleration': float,
                'is_explosive': bool,
                'reason': str
            }
        """
        
        # Ratio básico de volumen
        volume_ratio = current_volume / avg_volume_20d if avg_volume_20d > 0 else 0
        
        # Score base por volumen
        if volume_ratio >= 5.0:
            base_score = 1.0
            urgency = "CRÍTICA"
        elif volume_ratio >= 3.0:
            base_score = 0.8
            urgency = "ALTA"
        elif volume_ratio >= 2.0:
            base_score = 0.6
            urgency = "MEDIA"
        elif volume_ratio >= 1.5:
            base_score = 0.4
            urgency = "BAJA"
        else:
            base_score = 0.0
            urgency = "NINGUNA"
        
        acceleration_factor = 0
        is_accelerating = False
        
        # ANÁLISIS DE ACELERACIÓN INTRADIARIA (si tenemos datos históricos)
        if volume_history is not None and len(volume_history) >= 2:
            # Comparar volumen reciente vs volumen anterior
            recent_volume = np.mean(volume_history[-3:]) if len(volume_history) >= 3 else current_volume
            older_volume = np.mean(volume_history[-10:-3]) if len(volume_history) >= 10 else avg_volume_20d
            
            if older_volume > 0:
                acceleration_factor = recent_volume / older_volume
                is_accelerating = acceleration_factor >= self.signal_params['min_volume_acceleration']
                
                if is_accelerating:
                    # Bonificación por aceleración
                    base_score = min(1.0, base_score * 1.3)
        
        # Verificar si es explosivo según threshold
        min_spike = self.signal_params['min_volume_spike']
        is_explosive = volume_ratio >= min_spike and base_score >= 0.6
        
        reason = f"Vol: {volume_ratio:.1f}x avg"
        if is_accelerating:
            reason += f" + aceleración {acceleration_factor:.1f}x"
        
        return {
            'score': base_score,
            'volume_ratio': volume_ratio,
            'acceleration': acceleration_factor,
            'is_explosive': is_explosive,
            'urgency': urgency,
            'reason': reason
        }
    
    def analyze_momentum_breakout(self, symbol, current_price, price_history, 
                                  volume_history=None, rsi=50):
        """
        🚨 SEÑAL CLAVE #2: Detección de BREAKOUT con MOMENTUM
        
        Analogía: Como un sprinter que rompe el cordón de salida - no solo
        cruza la línea, sino que lo hace con velocidad explosiva.
        
        Detecta:
        - Ruptura de resistencia clave
        - Con volumen confirmatorio
        - Con momentum técnico (RSI no sobrecomprado aún)
        """
        
        if price_history is None or len(price_history) < 20:
            return {'score': 0, 'is_breakout': False, 'reason': 'Datos insuficientes'}
        
        # Identificar resistencia (máximo reciente)
        resistance = np.max(price_history[-20:])
        support = np.min(price_history[-20:])
        price_range = resistance - support
        
        # ¿Está rompiendo la resistencia?
        distance_from_resistance = (current_price - resistance) / resistance if resistance > 0 else 0
        is_breakout = distance_from_resistance > 0.02  # +2% sobre resistencia
        
        # Score base
        if is_breakout:
            if distance_from_resistance > 0.10:  # +10%
                base_score = 1.0
            elif distance_from_resistance > 0.05:  # +5%
                base_score = 0.8
            else:
                base_score = 0.6
        else:
            # Está cerca de resistencia pero no ha roto
            if distance_from_resistance > -0.02:  # Dentro del 2%
                base_score = 0.4
            else:
                base_score = 0.2
        
        # Ajuste por RSI (evitar comprar en extremo sobrecomprado)
        if rsi > self.signal_params['rsi_overbought']:
            base_score *= 0.5  # Penalización severa
        elif rsi > 70:
            base_score *= 0.8
        elif rsi < self.signal_params['rsi_oversold']:
            base_score *= 1.2  # Bonus por oversold
        
        # Confirmación de volumen (si disponible)
        volume_confirmed = False
        if volume_history is not None and len(volume_history) >= 5:
            recent_volume = np.mean(volume_history[-3:])
            older_volume = np.mean(volume_history[-20:-3])
            if older_volume > 0:
                volume_ratio = recent_volume / older_volume
                if volume_ratio >= self.signal_params['breakout_volume_threshold']:
                    base_score = min(1.0, base_score * 1.2)
                    volume_confirmed = True
        
        reason = f"Precio {distance_from_resistance*100:+.1f}% vs resistencia"
        if volume_confirmed:
            reason += " + volumen confirmatorio"
        
        return {
            'score': base_score,
            'is_breakout': is_breakout,
            'distance_from_resistance_pct': distance_from_resistance * 100,
            'resistance_level': resistance,
            'volume_confirmed': volume_confirmed,
            'reason': reason
        }
    
    def analyze_price_compression(self, symbol, price_history, atr_14=None):
        """
        🚨 SEÑAL CLAVE #3: Detección de COMPRESIÓN DE PRECIO
        
        Analogía: Como un resorte comprimido - mientras más tiempo pase
        en rango estrecho, más explosiva será la ruptura.
        
        Detecta:
        - Precio moviéndose en rango estrecho
        - Baja volatilidad (consolidación)
        - Setup clásico de squeeze
        """
        
        if price_history is None or len(price_history) < 10:
            return {'score': 0, 'is_compressed': False, 'reason': 'Datos insuficientes'}
        
        # Calcular rango de precios reciente
        recent_prices = price_history[-self.signal_params['min_price_compression_days']:]
        price_high = np.max(recent_prices)
        price_low = np.min(recent_prices)
        price_range_pct = ((price_high - price_low) / price_low * 100) if price_low > 0 else 100
        
        # ¿Está comprimido?
        max_compression = self.signal_params['compression_range_pct']
        is_compressed = price_range_pct <= max_compression
        
        # Score basado en qué tan comprimido está
        if price_range_pct <= 5:
            base_score = 1.0
            compression_level = "EXTREMA"
        elif price_range_pct <= 10:
            base_score = 0.8
            compression_level = "ALTA"
        elif price_range_pct <= 15:
            base_score = 0.6
            compression_level = "MODERADA"
        elif price_range_pct <= max_compression:
            base_score = 0.4
            compression_level = "LEVE"
        else:
            base_score = 0.0
            compression_level = "NINGUNA"
        
        # Bonificación si ATR está bajando (volatilidad contrayéndose)
        if atr_14 is not None and len(price_history) >= 20:
            older_atr = np.std(price_history[-20:-10])
            if older_atr > 0 and atr_14 < older_atr * 0.8:
                base_score = min(1.0, base_score * 1.2)
                compression_level += " + Volatilidad decreciente"
        
        return {
            'score': base_score,
            'is_compressed': is_compressed,
            'price_range_pct': price_range_pct,
            'compression_level': compression_level,
            'reason': f"Rango {price_range_pct:.1f}% en {len(recent_prices)}d"
        }
    
    def analyze_technical_breakout(self, symbol, current_price, price_history, 
                                   volume_history=None):
        """
        🚨 SEÑAL CLAVE #4: Análisis técnico de breakout
        
        Combina múltiples indicadores técnicos:
        - Moving averages (cruce alcista)
        - Volumen en el breakout
        - Momentum sostenido
        """
        
        if price_history is None or len(price_history) < 50:
            return {'score': 0, 'signals': [], 'reason': 'Datos insuficientes'}
        
        signals = []
        total_score = 0
        
        # 1. SMA 20/50 Cross (Golden Cross en timeframe corto)
        sma_20 = np.mean(price_history[-20:])
        sma_50 = np.mean(price_history[-50:])
        
        if sma_20 > sma_50:
            cross_score = min(1.0, (sma_20 - sma_50) / sma_50 * 10)
            signals.append(f"SMA20>SMA50 ({cross_score:.2f})")
            total_score += cross_score * 0.4
        
        # 2. Precio sobre medias móviles
        if current_price > sma_20:
            above_sma_score = min(1.0, (current_price - sma_20) / sma_20 * 5)
            signals.append(f"P>SMA20 ({above_sma_score:.2f})")
            total_score += above_sma_score * 0.3
        
        # 3. Momentum alcista (precio subiendo últimos 5 días)
        if len(price_history) >= 5:
            momentum = (price_history[-1] - price_history[-5]) / price_history[-5]
            if momentum > 0:
                momentum_score = min(1.0, momentum * 10)
                signals.append(f"Momentum+ ({momentum_score:.2f})")
                total_score += momentum_score * 0.3
        
        # Normalizar score
        final_score = min(1.0, total_score)
        
        return {
            'score': final_score,
            'signals': signals,
            'reason': ' | '.join(signals) if signals else 'Sin señales técnicas'
        }
    
    # ========================================================================
    # ANÁLISIS COMPLETO CON NUEVAS SEÑALES
    # ========================================================================
    
    def analyze_symbol_enhanced(self, symbol, market_data, historical_data=None):
        """
        Análisis completo de un símbolo con las nuevas señales mejoradas
        
        Args:
            symbol: Ticker del símbolo
            market_data: Datos actuales de mercado
            historical_data: Datos históricos (opcional, mejora el análisis)
        
        Returns:
            dict: Análisis completo con score y recomendación
        """
        
        print(f"\n🔍 Analizando {symbol}...")
        
        # Extraer datos básicos
        current_price = market_data.get('price', 0)
        volume = market_data.get('volume', 0)
        avg_volume = market_data.get('avg_volume_20d', 1)
        short_interest = market_data.get('short_interest_pct', 0)
        rsi = market_data.get('rsi', 50)
        atr = market_data.get('atr_14', current_price * 0.05)
        spread_pct = market_data.get('bid_ask_spread_pct', 5)
        
        # Datos históricos (si están disponibles)
        price_history = None
        volume_history = None
        
        if historical_data is not None:
            price_history = historical_data.get('close', None)
            volume_history = historical_data.get('volume', None)
        
        # =========================================
        # ANÁLISIS DE SEÑALES
        # =========================================
        signals = {}
        
        # Señal 1: Short Interest Cualificado (mantener de V2)
        days_to_cover = market_data.get('days_to_cover', 2)
        borrow_rate = market_data.get('borrow_rate', 20)
        si_analysis = self.analyze_short_interest_qualified(
            symbol, short_interest, days_to_cover, borrow_rate
        )
        signals['short_interest_qualified'] = si_analysis
        
        # Señal 2: EXPLOSIÓN DE VOLUMEN (NUEVO)
        vol_explosion = self.analyze_volume_explosion(
            symbol, volume, avg_volume, volume_history, price_history
        )
        signals['volume_explosion'] = vol_explosion
        
        # Señal 3: BREAKOUT con MOMENTUM (NUEVO)
        momentum_breakout = self.analyze_momentum_breakout(
            symbol, current_price, price_history, volume_history, rsi
        )
        signals['momentum_breakout'] = momentum_breakout
        
        # Señal 4: COMPRESIÓN DE PRECIO (NUEVO)
        compression = self.analyze_price_compression(
            symbol, price_history, atr
        )
        signals['price_compression'] = compression
        
        # Señal 5: Liquidez (mantener de V2)
        liquidity = self.analyze_liquidity(
            symbol, spread_pct, volume, market_data.get('market_depth_dollars', 10000)
        )
        signals['liquidity_filter'] = liquidity
        
        # Señal 6: BREAKOUT TÉCNICO (NUEVO)
        tech_breakout = self.analyze_technical_breakout(
            symbol, current_price, price_history, volume_history
        )
        signals['technical_breakout'] = tech_breakout
        
        # =========================================
        # CÁLCULO DE SCORE COMPUESTO
        # =========================================
        composite_score = 0
        for signal_name, weight in self.signals_weights.items():
            if signal_name in signals:
                composite_score += signals[signal_name]['score'] * weight
        
        # =========================================
        # DETECCIÓN DE SQUEEZE URGENTE (caso BYND)
        # =========================================
        is_urgent_squeeze = (
            composite_score >= self.thresholds['urgent_squeeze'] and
            vol_explosion['is_explosive'] and
            (momentum_breakout['is_breakout'] or compression['is_compressed'])
        )
        
        # =========================================
        # DECISIÓN DE TRADING
        # =========================================
        trading_action = self.generate_trading_action_v3(
            symbol, current_price, composite_score, signals, 
            is_urgent_squeeze, atr, rsi
        )
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'composite_score': composite_score,
            'signals': signals,
            'is_urgent_squeeze': is_urgent_squeeze,
            'trading_action': trading_action,
            'market_data': market_data
        }
    
    def analyze_short_interest_qualified(self, symbol, short_interest_pct, 
                                        days_to_cover=0, borrow_rate=0):
        """Short Interest cualificado (mantenido de V2)"""
        
        if short_interest_pct >= 30:
            base_score = 1.0
        elif short_interest_pct >= 20:
            base_score = 0.8
        elif short_interest_pct >= 15:
            base_score = 0.6
        elif short_interest_pct >= 10:
            base_score = 0.4
        else:
            base_score = 0.2
        
        # Bonificación por días para cubrir
        if days_to_cover >= 5:
            base_score = min(1.0, base_score * 1.3)
        elif days_to_cover >= 3:
            base_score = min(1.0, base_score * 1.15)
        
        # Bonificación por alto costo de préstamo
        if borrow_rate >= 50:
            base_score = min(1.0, base_score * 1.2)
        elif borrow_rate >= 30:
            base_score = min(1.0, base_score * 1.1)
        
        return {
            'score': base_score,
            'short_interest_pct': short_interest_pct,
            'days_to_cover': days_to_cover,
            'borrow_rate': borrow_rate,
            'reason': f"SI: {short_interest_pct:.1f}%, DTC: {days_to_cover:.1f}"
        }
    
    def analyze_liquidity(self, symbol, spread_pct, volume, market_depth):
        """Análisis de liquidez (mantenido de V2)"""
        
        # Filtro duro
        if spread_pct > self.signal_params['max_spread_pct']:
            return {
                'score': 0,
                'passed': False,
                'reason': f'Spread {spread_pct:.1f}% > {self.signal_params["max_spread_pct"]}%'
            }
        
        if volume < 50000:
            return {
                'score': 0,
                'passed': False,
                'reason': f'Volumen {volume:,} < 50,000'
            }
        
        # Score de calidad
        if spread_pct < 2 and volume > 1000000:
            score = 1.0
        elif spread_pct < 5 and volume > 500000:
            score = 0.8
        elif spread_pct < 8 and volume > 200000:
            score = 0.6
        else:
            score = 0.4
        
        return {
            'score': score,
            'passed': True,
            'spread_pct': spread_pct,
            'volume': volume,
            'reason': f'Spread {spread_pct:.1f}%, Vol {volume:,}'
        }
    
    def generate_trading_action_v3(self, symbol, current_price, composite_score, 
                                   signals, is_urgent_squeeze, atr, rsi):
        """
        Genera recomendación de trading con Take Profits DINÁMICOS
        
        CAMBIO CLAVE vs V2: Los take profits se ajustan según el tipo de setup:
        - Normal: Multiplicadores conservadores
        - Squeeze confirmado: Multiplicadores medios
        - Squeeze URGENTE: Multiplicadores agresivos (caso BYND)
        """
        
        # Filtro de liquidez
        if not signals['liquidity_filter']['passed']:
            return {
                'symbol': symbol,
                'action': 'DESCALIFICADA - LIQUIDEZ',
                'reason': signals['liquidity_filter']['reason'],
                'composite_score': composite_score
            }
        
        # Determinar acción base
        if composite_score >= self.thresholds['buy_strong']:
            action = 'COMPRAR FUERTE'
            urgency = 'ALTA'
        elif composite_score >= self.thresholds['buy_moderate']:
            action = 'COMPRAR MODERADO'
            urgency = 'MEDIA'
        elif composite_score >= self.thresholds['buy_light']:
            action = 'COMPRAR LIGERO'
            urgency = 'BAJA'
        else:
            return {
                'symbol': symbol,
                'action': 'ESPERAR',
                'reason': f'Score {composite_score:.3f} < {self.thresholds["buy_light"]}',
                'composite_score': composite_score
            }
        
        # AJUSTE POR SQUEEZE URGENTE
        if is_urgent_squeeze:
            action = '🚨 SQUEEZE URGENTE 🚨'
            urgency = 'CRÍTICA'
        
        # =========================================
        # TAKE PROFITS DINÁMICOS (MEJORA CLAVE)
        # =========================================
        
        # Seleccionar multiplicadores según tipo de setup
        if is_urgent_squeeze:
            tp_multipliers = self.risk_params['take_profit_urgent']
            position_size = self.risk_params['max_position_size_urgent']
        elif signals['volume_explosion']['is_explosive']:
            tp_multipliers = self.risk_params['take_profit_squeeze']
            position_size = min(0.04, self.risk_params['base_position_size'] * 1.5)
        else:
            tp_multipliers = self.risk_params['take_profit_normal']
            position_size = self.risk_params['base_position_size']
        
        # Calcular niveles de take profit
        take_profit_levels = []
        for multiplier in tp_multipliers:
            tp_price = current_price + (atr * multiplier)
            take_profit_levels.append(tp_price)
        
        # Stop loss
        stop_loss = current_price - (atr * self.risk_params['stop_loss_atr_multiplier'])
        stop_distance_pct = (current_price - stop_loss) / current_price
        
        # Risk/Reward
        avg_tp = np.mean(take_profit_levels)
        risk = current_price - stop_loss
        reward = avg_tp - current_price
        risk_reward_ratio = reward / risk if risk > 0 else 0
        
        # Trailing stop (NUEVO)
        trailing_stop_config = {
            'enabled': True,
            'trigger_gain_pct': self.risk_params['trailing_stop_trigger'],
            'trail_distance_pct': self.risk_params['trailing_stop_distance']
        }
        
        return {
            'symbol': symbol,
            'action': action,
            'urgency': urgency,
            'composite_score': composite_score,
            'config_preset': self.config_preset,
            'current_price': current_price,
            'position_size_pct': position_size * 100,
            'stop_loss': stop_loss,
            'stop_method': f'ATR {self.risk_params["stop_loss_atr_multiplier"]}x',
            'stop_distance_pct': stop_distance_pct,
            'take_profit_levels': take_profit_levels,
            'take_profit_multipliers': tp_multipliers,  # Para referencia
            'risk_reward_ratio': risk_reward_ratio,
            'trailing_stop': trailing_stop_config,
            'is_urgent_squeeze': is_urgent_squeeze,
            'key_signals': self._get_key_signals_summary(signals),
            'max_holding_days': 5 if is_urgent_squeeze else 7
        }
    
    def _get_key_signals_summary(self, signals):
        """Resumen de señales clave para mostrar"""
        summary = []
        for name, signal in signals.items():
            if signal.get('score', 0) >= 0.6:
                reason = signal.get('reason', 'N/A')
                summary.append(f"{name}: {reason}")
        return summary
    
    # ========================================================================
    # OBTENCIÓN DE DATOS HISTÓRICOS DE YFINANCE
    # ========================================================================
    
    def get_enhanced_market_data(self, symbol, period="1mo"):
        """
        Obtiene datos enriquecidos de yfinance para análisis mejorado
        
        Returns:
            tuple: (market_data, historical_data)
        """
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            info = ticker.info
            
            if len(hist) == 0:
                return None, None
            
            # Datos actuales
            current_price = hist['Close'].iloc[-1]
            current_volume = hist['Volume'].iloc[-1]
            
            # Cálculo de ATR
            high_low = hist['High'] - hist['Low']
            high_close = np.abs(hist['High'] - hist['Close'].shift())
            low_close = np.abs(hist['Low'] - hist['Close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr_14 = tr.rolling(14).mean().iloc[-1] if len(tr) >= 14 else current_price * 0.05
            
            # RSI simple
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
                'days_to_cover': 2.0,  # Estimado
                'borrow_rate': 25.0,  # Estimado
                'rsi': float(rsi_value),
                'atr_14': float(atr_14),
                'bid_ask_spread_pct': 5.0,  # Estimado
                'market_depth_dollars': 10000,  # Estimado
                'has_delisting_warning': current_price < 1.0
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
    
    def analyze_watchlist_enhanced(self, symbols=None):
        """
        Analiza watchlist completa con datos históricos
        """
        if symbols is None:
            symbols = self.watchlist
        
        results = []
        
        print(f"\n{'='*60}")
        print(f"🤖 ROBOT ADVISOR V3 - ANÁLISIS ENHANCED")
        print(f"{'='*60}")
        print(f"📊 Analizando {len(symbols)} símbolos")
        print(f"🔧 Configuración: {self.config_preset.upper()}")
        print()
        
        for symbol in symbols:
            market_data, historical_data = self.get_enhanced_market_data(symbol)
            
            if market_data is None:
                print(f"⚠️  {symbol}: Sin datos disponibles")
                continue
            
            analysis = self.analyze_symbol_enhanced(symbol, market_data, historical_data)
            results.append(analysis)
        
        # Ordenar por score
        results.sort(key=lambda x: x['composite_score'], reverse=True)
        
        return results
    
    def generate_trading_report_v3(self, results):
        """
        Genera reporte de trading con formato mejorado
        """
        print(f"\n{'='*70}")
        print("🚨 OPORTUNIDADES DE TRADING - V3 ENHANCED")
        print(f"{'='*70}")
        
        buy_signals = [r for r in results if r['trading_action']['action'] not in ['ESPERAR', 'DESCALIFICADA - LIQUIDEZ']]
        
        if not buy_signals:
            print("⏸️  No hay oportunidades hoy")
            print("   Criterios muy exigentes - esperando mejores setups")
            return []
        
        print(f"\n✅ {len(buy_signals)} OPORTUNIDADES DETECTADAS\n")
        
        for i, result in enumerate(buy_signals, 1):
            ta = result['trading_action']
            print(f"\n{'─'*70}")
            print(f"{i}. 📈 {ta['symbol']} - {ta['action']}")
            print(f"{'─'*70}")
            print(f"💰 Precio actual: ${ta['current_price']:.3f}")
            print(f"📊 Score: {ta['composite_score']:.3f}/1.000")
            print(f"⚡ Urgencia: {ta['urgency']}")
            
            if ta['is_urgent_squeeze']:
                print(f"🚨 *** SQUEEZE URGENTE DETECTADO ***")
                print(f"🚨 Este setup es similar al caso BYND (+150%)")
            
            print(f"\n💼 POSICIÓN RECOMENDADA:")
            print(f"   • Tamaño: {ta['position_size_pct']:.1f}% del capital")
            print(f"   • (Ejemplo: $10,000 capital = ${ta['position_size_pct']*100:.0f})")
            
            print(f"\n🛑 STOP LOSS:")
            print(f"   • Precio: ${ta['stop_loss']:.3f}")
            print(f"   • Distancia: -{ta['stop_distance_pct']:.1%}")
            print(f"   • Método: {ta['stop_method']}")
            
            print(f"\n🎯 TAKE PROFITS (escalonados):")
            for j, (tp_price, mult) in enumerate(zip(ta['take_profit_levels'], 
                                                     ta['take_profit_multipliers']), 1):
                gain_pct = (tp_price - ta['current_price']) / ta['current_price']
                print(f"   • TP{j}: ${tp_price:.3f} (+{gain_pct:.1%}) [{mult}x ATR]")
            
            print(f"\n📈 TRAILING STOP:")
            ts = ta['trailing_stop']
            print(f"   • Activar a: +{ts['trigger_gain_pct']:.1%}")
            print(f"   • Distancia: {ts['trail_distance_pct']:.1%} desde máximo")
            
            print(f"\n📊 MÉTRICAS:")
            print(f"   • Risk/Reward: 1:{ta['risk_reward_ratio']:.1f}")
            print(f"   • Max holding: {ta['max_holding_days']} días")
            
            print(f"\n⭐ SEÑALES CLAVE:")
            for signal in ta['key_signals'][:3]:
                print(f"   • {signal}")
        
        print(f"\n{'='*70}")
        print("⚠️  RECORDATORIOS IMPORTANTES:")
        print("   • USAR STOP LOSS OBLIGATORIO en todas las posiciones")
        print("   • ACTIVAR TRAILING STOP cuando alcance el trigger")
        print("   • VENDER POR TRAMOS en los take profits")
        print("   • NO PERSEGUIR el precio si hace gap up")
        print("   • MONITOREAR diariamente las posiciones")
        print(f"{'='*70}\n")
        
        return buy_signals

if __name__ == "__main__":
    print("🚀 PENNY STOCK ADVISOR V3 - ENHANCED")
    print("Versión mejorada con detección de squeezes urgentes\n")
    
