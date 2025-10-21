#!/usr/bin/env python3
"""
PENNY STOCK ROBOT ADVISOR - CLASE PRINCIPAL
==========================================

Clase principal del Robot Advisor con algoritmo mejorado.
Incluye configuraciones ajustables para diferentes niveles de agresividad.

Version: 2.0 Enhanced
Features: Short Interest Cualificado + Momentum + Liquidez
"""

import numpy as np
from datetime import datetime

class PennyStockRobotAdvisor:
    """
    Robot Advisor especializado en penny stocks con alto short interest
    Analiza 5 señales clave para detectar oportunidades de short squeeze
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
                    'short_interest_qualified': 0.25,
                    'momentum_confirmation': 0.25,
                    'delisting_risk': 0.20,
                    'volume_quality': 0.20,
                    'liquidity_filter': 0.10
                },
                "thresholds": {
                    'buy_strong': 0.85,
                    'buy_moderate': 0.70,
                    'buy_light': 0.55
                },
                "signal_params": {
                    'rsi_min': 45, 'rsi_max': 65,
                    'min_volume_ratio': 3, 'max_volume_ratio': 15,
                    'min_price_change': 2, 'max_spread_pct': 10,
                    'min_depth_dollars': 8000, 'min_daily_volume': 100000
                },
                "risk_params": {
                    'max_position_size': 0.02, 'stop_loss_atr_multiplier': 2.0,
                    'take_profit_atr_multipliers': [2, 3, 5], 'max_holding_days': 10
                }
            },
            
            "balanced": {
                "description": "Configuración balanceada - equilibrio oportunidades/riesgo",
                "signals_weights": {
                    'short_interest_qualified': 0.25,
                    'momentum_confirmation': 0.30,  # Aumentado
                    'delisting_risk': 0.20,
                    'volume_quality': 0.20,
                    'liquidity_filter': 0.05       # Reducido
                },
                "thresholds": {
                    'buy_strong': 0.80,    # Reducido
                    'buy_moderate': 0.65,  # Reducido
                    'buy_light': 0.50      # Reducido
                },
                "signal_params": {
                    'rsi_min': 40, 'rsi_max': 70,  # Expandido
                    'min_volume_ratio': 2.5, 'max_volume_ratio': 20,
                    'min_price_change': 1.5, 'max_spread_pct': 12,
                    'min_depth_dollars': 6000, 'min_daily_volume': 75000
                },
                "risk_params": {
                    'max_position_size': 0.025, 'stop_loss_atr_multiplier': 2.0,
                    'take_profit_atr_multipliers': [2, 3, 5], 'max_holding_days': 8
                }
            },
            
            "aggressive": {
                "description": "Configuración agresiva - más oportunidades",
                "signals_weights": {
                    'short_interest_qualified': 0.25,
                    'momentum_confirmation': 0.35,  # Muy aumentado
                    'delisting_risk': 0.15,         # Reducido
                    'volume_quality': 0.20,
                    'liquidity_filter': 0.05       # Muy reducido
                },
                "thresholds": {
                    'buy_strong': 0.75,    # Muy reducido
                    'buy_moderate': 0.60,  # Muy reducido
                    'buy_light': 0.45      # Muy reducido
                },
                "signal_params": {
                    'rsi_min': 35, 'rsi_max': 75,  # Muy expandido
                    'min_volume_ratio': 2, 'max_volume_ratio': 25,
                    'min_price_change': 1, 'max_spread_pct': 15,
                    'min_depth_dollars': 5000, 'min_daily_volume': 50000
                },
                "risk_params": {
                    'max_position_size': 0.03, 'stop_loss_atr_multiplier': 1.8,
                    'take_profit_atr_multipliers': [2, 3, 5], 'max_holding_days': 6
                }
            },
            
            "very_aggressive": {
                "description": "Configuración muy agresiva - máximas oportunidades",
                "signals_weights": {
                    'short_interest_qualified': 0.20,  # Reducido
                    'momentum_confirmation': 0.40,     # Máximo
                    'delisting_risk': 0.15,
                    'volume_quality': 0.20,
                    'liquidity_filter': 0.05          # Mínimo
                },
                "thresholds": {
                    'buy_strong': 0.70,    # Mínimo
                    'buy_moderate': 0.55,  # Mínimo
                    'buy_light': 0.40      # Mínimo
                },
                "signal_params": {
                    'rsi_min': 30, 'rsi_max': 80,  # Máximo expandido
                    'min_volume_ratio': 1.5, 'max_volume_ratio': 30,
                    'min_price_change': 0.5, 'max_spread_pct': 20,
                    'min_depth_dollars': 3000, 'min_daily_volume': 30000
                },
                "risk_params": {
                    'max_position_size': 0.035, 'stop_loss_atr_multiplier': 1.5,
                    'take_profit_atr_multipliers': [1.5, 2.5, 4], 'max_holding_days': 5
                }
            }
        }
        
        if preset not in configurations:
            preset = "balanced"
            print(f"⚠️  Preset '{preset}' no encontrado, usando 'balanced'")
        
        config = configurations[preset]
        self.signals_weights = config["signals_weights"]
        self.thresholds = config["thresholds"]
        self.signal_params = config["signal_params"]
        self.risk_params = config["risk_params"]
        self.description = config["description"]
        
        print(f"🔧 Configuración cargada: {preset.upper()}")
        print(f"📝 {self.description}")
    
    def update_watchlist(self, symbols):
        """Actualiza la lista de símbolos a analizar"""
        self.watchlist = [s.upper() for s in symbols]
        print(f"✅ Watchlist actualizada con {len(symbols)} símbolos: {', '.join(self.watchlist)}")
    
    def analyze_short_interest_qualified(self, symbol, short_interest_pct, 
                                        days_to_cover=0, borrow_rate=0):
        """
        MEJORA 1: Analiza short interest CUALIFICADO
        No basta alto SI, debe ser difícil y caro de cubrir
        """
        # Score base por porcentaje de short interest
        if short_interest_pct >= 30:
            base_score = 1.0
        elif short_interest_pct >= 20:
            base_score = 0.8
        elif short_interest_pct >= 15:
            base_score = 0.6
        elif short_interest_pct >= 10:  # Relajado para configuraciones agresivas
            base_score = 0.4
        else:
            base_score = 0.0
        
        # Bonificación por dificultad de covering (days to cover)
        dtc_bonus = 0
        if days_to_cover >= 5:      # Muy difícil cubrir
            dtc_bonus = 0.3
        elif days_to_cover >= 3:    # Moderadamente difícil  
            dtc_bonus = 0.2
        elif days_to_cover >= 2:    # Algo difícil
            dtc_bonus = 0.1
        
        # Bonificación por costo de borrow (borrow rate)
        borrow_bonus = 0
        if borrow_rate >= 100:      # Extremadamente caro (>100%)
            borrow_bonus = 0.3
        elif borrow_rate >= 50:     # Muy caro (50-100%)
            borrow_bonus = 0.2
        elif borrow_rate >= 20:     # Caro (20-50%)
            borrow_bonus = 0.1
        
        # Score final cualificado (máximo 1.0)
        qualified_score = min(base_score + dtc_bonus + borrow_bonus, 1.0)
        
        # Determina nivel de alerta
        alert_level = "BAJA"
        if qualified_score >= 0.8:
            alert_level = "CRÍTICA"
        elif qualified_score >= 0.6:
            alert_level = "ALTA"
        elif qualified_score >= 0.4:
            alert_level = "MEDIA"
        
        return {
            'signal': 'short_interest_qualified',
            'score': qualified_score,
            'alert_level': alert_level,
            'value': short_interest_pct,
            'description': f"SI: {short_interest_pct:.1f}%, DTC: {days_to_cover:.1f}d, Rate: {borrow_rate:.0f}% - Qualified {alert_level.lower()}"
        }
    
    def analyze_momentum_confirmation(self, symbol, price, vwap, rsi, 
                                    volume_ratio, price_change_pct):
        """
        MEJORA 2: Confirmación de MOMENTUM técnico
        El squeeze debe mostrar momentum real, no solo volumen
        """
        momentum_signals = 0
        max_signals = 5
        signal_details = []
        
        # 1. Precio por encima de VWAP (presión compradora dominante)
        if price > vwap:
            momentum_signals += 1
            signal_details.append("Precio>VWAP")
            
        # 2. RSI en zona configurada (adaptable según preset)
        if self.signal_params['rsi_min'] <= rsi <= self.signal_params['rsi_max']:
            momentum_signals += 1  
            signal_details.append(f"RSI_{rsi:.0f}_ideal")
            
        # 3. Volumen en rango configurado
        if self.signal_params['min_volume_ratio'] <= volume_ratio <= self.signal_params['max_volume_ratio']:
            momentum_signals += 1
            signal_details.append(f"Vol_{volume_ratio:.1f}x_optimo")
            
        # 4. Movimiento de precio positivo configurado
        if price_change_pct > self.signal_params['min_price_change']:
            momentum_signals += 1
            signal_details.append(f"Up_{price_change_pct:.1f}%")
            
        # 5. Subida sustentable, no gap extremo (<25% para agresivos)
        max_change = 25 if self.config_preset in ["aggressive", "very_aggressive"] else 20
        if price_change_pct <= max_change:
            momentum_signals += 1
            signal_details.append("Subida_sustentable")
        
        # Score basado en número de señales confirmadas
        momentum_score = momentum_signals / max_signals
        
        # Determina nivel de alerta
        alert_level = "BAJA"
        if momentum_score >= 0.8:
            alert_level = "CRÍTICA"
        elif momentum_score >= 0.6:
            alert_level = "ALTA"
        elif momentum_score >= 0.4:
            alert_level = "MEDIA"
        
        return {
            'signal': 'momentum_confirmation', 
            'score': momentum_score,
            'alert_level': alert_level,
            'value': momentum_signals,
            'description': f"Momentum: {momentum_signals}/5 ({', '.join(signal_details)})"
        }
    
    def analyze_delisting_risk(self, symbol, price, has_delisting_warning=False):
        """
        Señal 3: Analiza riesgo de delisting y precio (original)
        """
        score = 0
        alert_level = "BAJA"
        
        if price < 0.5 and has_delisting_warning:
            score = 1.0
            alert_level = "CRÍTICA"
        elif price < 1.0 and has_delisting_warning:
            score = 0.8
            alert_level = "ALTA"
        elif price < 1.0:
            score = 0.4
            alert_level = "MEDIA"
        
        return {
            'signal': 'delisting_risk',
            'score': score,
            'alert_level': alert_level,
            'value': price,
            'description': f"Precio ${price:.3f} - Riesgo delisting: {'SÍ' if has_delisting_warning else 'NO'}"
        }
    
    def analyze_volume_quality(self, symbol, current_volume, avg_volume_20d, 
                              price_change_pct, bid_ask_spread_pct):
        """
        MEJORA 3: Analiza CALIDAD del volumen (no solo cantidad)
        """
        if avg_volume_20d == 0:
            return {'signal': 'volume_quality', 'score': 0, 'alert_level': 'BAJA',
                   'description': 'Sin datos de volumen histórico'}
        
        volume_ratio = current_volume / avg_volume_20d
        
        # Evalúa calidad según dirección del precio (adaptado para agresivos)
        if price_change_pct < -3:
            quality_multiplier = 0.2
            quality_desc = "dumping"
        elif price_change_pct < 0:
            quality_multiplier = 0.5 if self.config_preset in ["aggressive", "very_aggressive"] else 0.4
            quality_desc = "debilidad"
        elif price_change_pct >= 10:
            quality_multiplier = 1.3
            quality_desc = "buying_pressure"
        elif price_change_pct >= 5:
            quality_multiplier = 1.1
            quality_desc = "acumulacion"
        elif price_change_pct >= 2:
            quality_multiplier = 1.0
            quality_desc = "neutral_positivo"
        else:
            quality_multiplier = 0.7 if self.config_preset in ["aggressive", "very_aggressive"] else 0.6
            quality_desc = "neutral"
        
        # Penaliza spread alto según configuración
        if bid_ask_spread_pct > self.signal_params['max_spread_pct']:
            spread_penalty = 0.5
        elif bid_ask_spread_pct > self.signal_params['max_spread_pct'] * 0.6:
            spread_penalty = 0.7
        else:
            spread_penalty = 1.0
        
        # Score base normalizado (adaptado según preset)
        volume_threshold = 8 if self.config_preset == "conservative" else 6
        if volume_ratio >= volume_threshold * 1.25:
            base_score = 1.0
        elif volume_ratio >= volume_threshold:
            base_score = 0.9
        elif volume_ratio >= volume_threshold * 0.6:
            base_score = 0.7
        elif volume_ratio >= 2:
            base_score = 0.5 if self.config_preset in ["aggressive", "very_aggressive"] else 0.4
        else:
            base_score = 0.2
        
        # Score final ajustado por calidad
        quality_score = base_score * quality_multiplier * spread_penalty
        quality_score = min(quality_score, 1.0)
        
        # Determina nivel de alerta
        alert_level = "BAJA"
        if quality_score >= 0.8:
            alert_level = "CRÍTICA"
        elif quality_score >= 0.6:
            alert_level = "ALTA"
        elif quality_score >= 0.4:
            alert_level = "MEDIA"
        
        return {
            'signal': 'volume_quality',
            'score': quality_score,
            'alert_level': alert_level,
            'value': volume_ratio,
            'description': f"Vol: {volume_ratio:.1f}x, Calidad: {quality_desc}, Score: {quality_score:.2f}"
        }
    
    def analyze_liquidity_filter(self, symbol, bid_ask_spread_pct, 
                               market_depth_dollars, daily_dollar_volume):
        """
        MEJORA 4: Filtro crítico de LIQUIDEZ (adaptable según preset)
        """
        liquidity_components = []
        
        # Parámetros según configuración
        max_spread = self.signal_params['max_spread_pct']
        min_depth = self.signal_params['min_depth_dollars']
        min_daily_vol = self.signal_params['min_daily_volume']
        
        # 1. Evaluación del bid-ask spread
        if bid_ask_spread_pct <= max_spread * 0.3:
            spread_score = 1.0
            liquidity_components.append("Spread_excelente")
        elif bid_ask_spread_pct <= max_spread * 0.6:
            spread_score = 0.8
            liquidity_components.append("Spread_bueno")
        elif bid_ask_spread_pct <= max_spread:
            spread_score = 0.6
            liquidity_components.append("Spread_aceptable")
        else:
            spread_score = 0.0  # DESCALIFICA
            liquidity_components.append("Spread_prohibitivo")
        
        # 2. Evaluación de profundidad de mercado
        if market_depth_dollars >= min_depth * 2:
            depth_score = 1.0
            liquidity_components.append("Depth_excelente")
        elif market_depth_dollars >= min_depth * 1.5:
            depth_score = 0.8
            liquidity_components.append("Depth_buena")
        elif market_depth_dollars >= min_depth:
            depth_score = 0.6
            liquidity_components.append("Depth_aceptable")
        else:
            depth_score = 0.0  # DESCALIFICA
            liquidity_components.append("Depth_insuficiente")
        
        # 3. Evaluación de volumen en dólares diario
        if daily_dollar_volume >= min_daily_vol * 3:
            dollar_vol_score = 1.0
            liquidity_components.append("DolVol_excelente")
        elif daily_dollar_volume >= min_daily_vol * 2:
            dollar_vol_score = 0.8
            liquidity_components.append("DolVol_bueno")
        elif daily_dollar_volume >= min_daily_vol:
            dollar_vol_score = 0.6
            liquidity_components.append("DolVol_aceptable")
        else:
            dollar_vol_score = 0.0  # DESCALIFICA
            liquidity_components.append("DolVol_insuficiente")
        
        # Score promedio ponderado (spread es más crítico)
        liquidity_score = (spread_score * 0.4 + depth_score * 0.3 + dollar_vol_score * 0.3)
        
        # FILTRO CRÍTICO: Si cualquier componente es 0, descalificar completamente
        if spread_score == 0 or depth_score == 0 or dollar_vol_score == 0:
            liquidity_score = 0.0
        
        # Determina nivel de alerta
        if liquidity_score == 0:
            alert_level = "DESCALIFICADA"
        elif liquidity_score >= 0.8:
            alert_level = "EXCELENTE"
        elif liquidity_score >= 0.6:
            alert_level = "BUENA"
        elif liquidity_score >= 0.4:
            alert_level = "ACEPTABLE"
        else:
            alert_level = "INSUFICIENTE"
        
        return {
            'signal': 'liquidity_filter',
            'score': liquidity_score,
            'alert_level': alert_level,
            'value': liquidity_score,
            'description': f"Liquidez: {alert_level} - Spread: {bid_ask_spread_pct:.1f}%, Depth: ${market_depth_dollars:,.0f}"
        }
    
    def calculate_atr_stop_loss(self, current_price, atr_14, atr_multiplier=None):
        """
        Stop loss inteligente basado en ATR
        """
        if atr_multiplier is None:
            atr_multiplier = self.risk_params['stop_loss_atr_multiplier']
        
        # Stop loss basado en ATR
        atr_stop = current_price - (atr_multiplier * atr_14)
        
        # Límite máximo de pérdida (20% absoluto)
        max_loss_pct = 0.20
        max_stop = current_price * (1 - max_loss_pct)
        
        # Límite mínimo de pérdida (8% mínimo para penny stocks)
        min_loss_pct = 0.08
        min_stop = current_price * (1 - min_loss_pct)
        
        # Usar el stop más conservador pero realista
        if atr_stop >= max_stop:
            intelligent_stop = max_stop
            method = f"Max_loss_{max_loss_pct:.0%}"
        elif atr_stop >= min_stop:
            intelligent_stop = atr_stop
            method = f"ATR_{atr_multiplier}x"
        else:
            intelligent_stop = min_stop
            method = f"Min_loss_{min_loss_pct:.0%}"
        
        stop_distance_pct = (current_price - intelligent_stop) / current_price
        
        return {
            'stop_loss_price': round(intelligent_stop, 3),
            'stop_distance_pct': stop_distance_pct,
            'atr_value': atr_14,
            'method': method,
            'description': f"Stop: ${intelligent_stop:.3f} ({method}) = -{stop_distance_pct:.1%}"
        }
    
    def calculate_composite_score(self, signals):
        """Calcula puntuación compuesta ponderada"""
        total_score = 0
        for signal in signals:
            weight = self.signals_weights.get(signal['signal'], 0)
            total_score += signal['score'] * weight
        
        return min(total_score, 1.0)
    
    def generate_enhanced_trading_action(self, symbol, composite_score, current_price, 
                                       signals, atr_stop):
        """
        Genera acciones de trading con criterios adaptativos según configuración
        """
        action = "ESPERAR"
        position_size = 0
        urgency = "BAJA"
        
        # Umbrales adaptativos según configuración
        if composite_score >= self.thresholds['buy_strong']:
            action = "COMPRAR FUERTE"
            position_size = self.risk_params['max_position_size']
            urgency = "CRÍTICA"
        elif composite_score >= self.thresholds['buy_moderate']:
            action = "COMPRAR MODERADO"
            position_size = self.risk_params['max_position_size'] * 0.7
            urgency = "ALTA"
        elif composite_score >= self.thresholds['buy_light']:
            action = "COMPRAR LIGERO"
            position_size = self.risk_params['max_position_size'] * 0.4
            urgency = "MEDIA"
        
        # Calcula niveles de riesgo con take profit adaptativo
        if action != "ESPERAR":
            atr_value = atr_stop['atr_value']
            tp_multipliers = self.risk_params['take_profit_atr_multipliers']
            
            take_profit_levels = [
                current_price + (mult * atr_value) for mult in tp_multipliers
            ]
        else:
            take_profit_levels = []
        
        return {
            'symbol': symbol,
            'action': action,
            'urgency': urgency,
            'composite_score': composite_score,
            'position_size_pct': position_size * 100,
            'current_price': current_price,
            
            # Stop loss inteligente
            'stop_loss': atr_stop['stop_loss_price'],
            'stop_method': atr_stop['method'],
            'stop_distance_pct': atr_stop['stop_distance_pct'],
            
            # Take profit adaptativo
            'take_profit_levels': take_profit_levels,
            'tp_method': 'ATR-based' if action != "ESPERAR" else 'N/A',
            
            'max_holding_days': self.risk_params['max_holding_days'],
            'signals_summary': [f"{s['signal']}: {s['score']:.2f}" for s in signals],
            
            # Información adicional
            'atr_value': atr_stop['atr_value'],
            'risk_reward_ratio': (take_profit_levels[0] - current_price) / (current_price - atr_stop['stop_loss_price']) if action != "ESPERAR" and atr_stop['stop_loss_price'] < current_price else 0,
            'config_preset': self.config_preset
        }
    
    def analyze_symbol(self, symbol, market_data):
        """
        Análisis completo de un símbolo con configuración adaptativa
        """
        signals = []
        
        # Señal 1: Short Interest CUALIFICADO
        signals.append(self.analyze_short_interest_qualified(
            symbol, 
            market_data.get('short_interest_pct', 0),
            market_data.get('days_to_cover', 0),
            market_data.get('borrow_rate', 0)
        ))
        
        # Señal 2: Confirmación de MOMENTUM
        volume_ratio = market_data.get('volume', 0) / max(market_data.get('avg_volume_20d', 1), 1)
        signals.append(self.analyze_momentum_confirmation(
            symbol,
            market_data.get('price', 0),
            market_data.get('vwap', market_data.get('price', 0)),
            market_data.get('rsi', 50),
            volume_ratio,
            market_data.get('price_change_pct', 0)
        ))
        
        # Señal 3: Delisting Risk
        signals.append(self.analyze_delisting_risk(
            symbol, market_data.get('price', 0), 
            market_data.get('has_delisting_warning', False)
        ))
        
        # Señal 4: Calidad de VOLUMEN
        signals.append(self.analyze_volume_quality(
            symbol, 
            market_data.get('volume', 0),
            market_data.get('avg_volume_20d', 1),
            market_data.get('price_change_pct', 0),
            market_data.get('bid_ask_spread_pct', 10)
        ))
        
        # Señal 5: Filtro de LIQUIDEZ
        liquidity_signal = self.analyze_liquidity_filter(
            symbol,
            market_data.get('bid_ask_spread_pct', 15),
            market_data.get('market_depth_dollars', 0),
            market_data.get('daily_dollar_volume', 0)
        )
        signals.append(liquidity_signal)
        
        # FILTRO CRÍTICO: Si no pasa liquidez, descartar completamente
        if liquidity_signal['score'] == 0:
            return {
                'analysis_timestamp': datetime.now(),
                'symbol': symbol,
                'signals': signals,
                'composite_score': 0,
                'trading_action': {
                    'symbol': symbol,
                    'action': 'DESCALIFICADA - LIQUIDEZ INSUFICIENTE',
                    'urgency': 'N/A',
                    'composite_score': 0,
                    'reason': f"Falla filtro de liquidez: {liquidity_signal['description']}",
                    'config_preset': self.config_preset
                }
            }
        
        # Calcula score compuesto
        composite_score = self.calculate_composite_score(signals)
        
        # Stop loss inteligente basado en ATR
        atr_stop = self.calculate_atr_stop_loss(
            market_data.get('price', 0),
            market_data.get('atr_14', 0.05)
        )
        
        # Genera acción de trading
        trading_action = self.generate_enhanced_trading_action(
            symbol, composite_score, market_data.get('price', 0), 
            signals, atr_stop
        )
        
        return {
            'analysis_timestamp': datetime.now(),
            'symbol': symbol,
            'signals': signals,
            'composite_score': composite_score,
            'trading_action': trading_action
        }
    
    def get_configuration_summary(self):
        """Retorna resumen de la configuración actual"""
        return {
            'preset': self.config_preset,
            'description': self.description,
            'thresholds': self.thresholds,
            'signal_params': self.signal_params,
            'signals_weights': self.signals_weights,
            'risk_params': self.risk_params
        }
    
    def change_configuration(self, new_preset):
        """Cambia la configuración del robot"""
        old_preset = self.config_preset
        self.load_configuration(new_preset)
        print(f"🔄 Configuración cambiada: {old_preset} → {new_preset}")
        return True
    
    def analyze_batch(self, market_data_batch):
        """
        Analiza un lote de símbolos y retorna resultados ordenados
        
        Args:
            market_data_batch (dict): {symbol: market_data}
            
        Returns:
            list: Lista de análisis ordenados por score
        """
        results = []
        
        for symbol in self.watchlist:
            if symbol in market_data_batch:
                analysis = self.analyze_symbol(symbol, market_data_batch[symbol])
                results.append(analysis)
        
        # Ordena por score compuesto (mejores oportunidades primero)
        results.sort(key=lambda x: x['composite_score'], reverse=True)
        
        return results
    
    def generate_daily_report(self, market_data_batch, show_details=True):
        """
        Genera reporte diario completo con estadísticas
        
        Args:
            market_data_batch (dict): Datos de mercado
            show_details (bool): Mostrar detalles de análisis
            
        Returns:
            dict: Resultados del análisis con estadísticas
        """
        if show_details:
            print("🤖 ROBOT ADVISOR - ANÁLISIS DIARIO")
            print("=" * 60)
            print(f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            print(f"📊 Símbolos analizados: {len(market_data_batch)}")
            print(f"🔧 Configuración: {self.config_preset.upper()}")
            print(f"📝 {self.description}")
            print()
        
        # Ejecuta análisis por lotes
        results = self.analyze_batch(market_data_batch)
        
        # Calcula estadísticas
        analyzed = len(results)
        passed_liquidity = sum(1 for r in results if r['composite_score'] > 0)
        buy_signals = sum(1 for r in results if r['trading_action']['action'] not in ['ESPERAR', 'DESCALIFICADA - LIQUIDEZ INSUFICIENTE'])
        
        if show_details:
            print("📈 ESTADÍSTICAS DE FILTRADO:")
            print("-" * 30)
            print(f"   • Símbolos analizados: {analyzed}")
            print(f"   • Pasaron filtro liquidez: {passed_liquidity} ({passed_liquidity/analyzed*100:.1f}%)")
            print(f"   • Generaron señales compra: {buy_signals} ({buy_signals/analyzed*100:.1f}%)")
            print(f"   • Tasa de selectividad: {100-buy_signals/analyzed*100:.1f}% (rechazados)")
        
        # Filtra oportunidades
        buy_opportunities = [r for r in results if r['trading_action']['action'] not in ['ESPERAR', 'DESCALIFICADA - LIQUIDEZ INSUFICIENTE']]
        rejected = [r for r in results if r['trading_action']['action'] in ['ESPERAR', 'DESCALIFICADA - LIQUIDEZ INSUFICIENTE']]
        
        if show_details:
            self._print_opportunities(buy_opportunities)
            self._print_rejected(rejected)
            self._print_configuration_info()
        
        return {
            'timestamp': datetime.now(),
            'config_preset': self.config_preset,
            'statistics': {
                'analyzed': analyzed,
                'passed_liquidity': passed_liquidity,
                'buy_signals': buy_signals,
                'rejection_rate': (analyzed - buy_signals) / analyzed * 100 if analyzed > 0 else 0
            },
            'opportunities': buy_opportunities,
            'rejected': rejected,
            'all_results': results
        }
    
    def _print_opportunities(self, buy_opportunities):
        """Imprime oportunidades de compra"""
        print(f"\n🚨 OPORTUNIDADES DETECTADAS ({len(buy_opportunities)}):")
        print("-" * 50)
        
        if not buy_opportunities:
            print("⏸️  No hay señales de compra hoy")
            print("   ✅ Esto demuestra selectividad del algoritmo")
            print("   ✅ Mejor esperar mejores oportunidades")
        else:
            for i, result in enumerate(buy_opportunities[:5], 1):  # Top 5
                ta = result['trading_action']
                print(f"\n{i}. 📈 {ta['symbol']} - {ta['action']} ({ta['urgency']})")
                print(f"   💰 Precio: ${ta['current_price']:.3f}")
                print(f"   📊 Score: {ta['composite_score']:.3f}/1.000 (Config: {ta['config_preset']})")
                print(f"   💸 Posición: {ta['position_size_pct']:.1f}% del capital")
                print(f"   🛑 Stop Loss: ${ta['stop_loss']:.3f} ({ta['stop_method']}) = -{ta['stop_distance_pct']:.1%}")
                print(f"   🎯 Take Profit: ${ta['take_profit_levels'][0]:.3f} / ${ta['take_profit_levels'][1]:.3f} / ${ta['take_profit_levels'][2]:.3f}")
                print(f"   📊 Risk/Reward: 1:{ta['risk_reward_ratio']:.1f}")
                print(f"   ⏰ Max holding: {ta['max_holding_days']} días")
                
                # Señales más importantes
                key_signals = [s for s in result['signals'] if s['score'] >= 0.6]
                if key_signals:
                    signals_desc = ', '.join([f"{s['signal']}({s['score']:.2f})" for s in key_signals])
                    print(f"   ⭐ Señales clave: {signals_desc}")
    
    def _print_rejected(self, rejected):
        """Imprime símbolos rechazados"""
        if rejected:
            liquidity_fails = [r for r in rejected if 'LIQUIDEZ' in r['trading_action']['action']]
            low_scores = [r for r in rejected if r['trading_action']['action'] == 'ESPERAR']
            
            if liquidity_fails:
                print(f"\n❌ DESCALIFICADOS POR LIQUIDEZ ({len(liquidity_fails)}):")
                for result in liquidity_fails[:3]:  # Solo top 3
                    ta = result['trading_action']
                    print(f"   • {ta['symbol']}: {ta['reason']}")
            
            if low_scores:
                print(f"\n⏸️  SCORE INSUFICIENTE ({len(low_scores)}):")
                for result in low_scores[:3]:  # Solo top 3
                    ta = result['trading_action']
                    print(f"   • {ta['symbol']}: Score {ta['composite_score']:.3f} < {self.thresholds['buy_light']}")
    
    def _print_configuration_info(self):
        """Imprime información de configuración"""
        print("\n" + "=" * 60)
        print("⚠️  CONFIGURACIÓN Y RECORDATORIOS:")
        print(f"• Preset: {self.config_preset.upper()} - {self.description}")
        print(f"• Umbrales: Ligero={self.thresholds['buy_light']}, Moderado={self.thresholds['buy_moderate']}, Fuerte={self.thresholds['buy_strong']}")
        print(f"• RSI rango: {self.signal_params['rsi_min']}-{self.signal_params['rsi_max']}")
        print(f"• Volumen mínimo: {self.signal_params['min_volume_ratio']}x")
        print(f"• Spread máximo: {self.signal_params['max_spread_pct']}%")
        print("• Stop-loss INTELIGENTE basado en volatilidad específica")
        print("• Take-profit DINÁMICO basado en ATR")
        print("• Risk/reward ratio calculado automáticamente")
        print("=" * 60)

# Ejemplo de uso y testing
if __name__ == "__main__":
    print("🧪 TESTING DE LA CLASE PennyStockRobotAdvisor")
    print("=" * 50)
    
    # Test de diferentes configuraciones
    configs_to_test = ["conservative", "balanced", "aggressive", "very_aggressive"]
    
    for config in configs_to_test:
        print(f"\n📊 Testing configuración: {config.upper()}")
        print("-" * 30)
        
        robot = PennyStockRobotAdvisor(config_preset=config)
        summary = robot.get_configuration_summary()
        
        print(f"🎯 Umbrales: {summary['thresholds']}")
        print(f"⚡ RSI: {summary['signal_params']['rsi_min']}-{summary['signal_params']['rsi_max']}")
        print(f"📈 Vol mín: {summary['signal_params']['min_volume_ratio']}x")
        print(f"💧 Spread máx: {summary['signal_params']['max_spread_pct']}%")
    
    print(f"\n✅ Clase lista para importar y usar")
    print(f"💡 Uso recomendado para HTOO/LCFY: 'aggressive' o 'very_aggressive'")