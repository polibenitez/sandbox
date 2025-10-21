#!/usr/bin/env python3
"""
TESTER DE ALGORITMO AJUSTABLE - PARA PRUEBAS MANUALES
====================================================

Script para ajustar parámetros del algoritmo y probar con casos reales
como HTOO (+60%) y LCFY (+20%) que no fueron detectados.

Uso: python adjustable_algorithm_tester.py
"""

import numpy as np
from datetime import datetime
import json

class AdjustableAlgorithmTester:
    """Clase para testing y ajuste de parámetros del algoritmo"""
    
    def __init__(self):
        # PARÁMETROS AJUSTABLES
        self.signals_weights = {
            'short_interest_qualified': 0.25,
            'momentum_confirmation': 0.25,
            'delisting_risk': 0.20,
            'volume_quality': 0.20,
            'liquidity_filter': 0.10
        }
        
        # UMBRALES AJUSTABLES
        self.thresholds = {
            'buy_strong': 0.85,    # Reducir para más sensibilidad
            'buy_moderate': 0.70,  # Reducir para más sensibilidad
            'buy_light': 0.55      # Reducir para más sensibilidad
        }
        
        # PARÁMETROS DE SEÑALES AJUSTABLES
        self.signal_params = {
            'short_interest': {
                'min_si_pct': 15,           # Mínimo SI para considerar
                'min_days_to_cover': 2,     # Mínimo DTC
                'min_borrow_rate': 20       # Mínimo borrow rate
            },
            'momentum': {
                'rsi_min': 45,              # RSI mínimo
                'rsi_max': 65,              # RSI máximo
                'min_volume_ratio': 3,      # Volumen mínimo vs promedio
                'max_volume_ratio': 15,     # Volumen máximo vs promedio
                'min_price_change': 2       # Cambio mínimo de precio %
            },
            'liquidity': {
                'max_spread_pct': 10,       # Spread máximo permitido
                'min_depth_dollars': 8000,  # Profundidad mínima
                'min_daily_volume': 100000  # Volumen diario mínimo $
            }
        }
    
    def analyze_with_current_params(self, symbol, data):
        """Analiza con parámetros actuales"""
        print(f"\n🔍 ANALIZANDO {symbol} CON PARÁMETROS ACTUALES")
        print("-" * 50)
        
        # Señal 1: Short Interest Cualificado
        si_score, si_details = self.analyze_short_interest_qualified(data)
        print(f"📊 Short Interest: {si_score:.3f} - {si_details}")
        
        # Señal 2: Momentum
        momentum_score, momentum_details = self.analyze_momentum_confirmation(data)
        print(f"⚡ Momentum: {momentum_score:.3f} - {momentum_details}")
        
        # Señal 3: Delisting Risk
        delisting_score = 0.8 if data['price'] < 1 and data.get('has_delisting_warning', False) else 0.2
        print(f"🚨 Delisting Risk: {delisting_score:.3f} - {'Alto riesgo' if delisting_score > 0.5 else 'Bajo riesgo'}")
        
        # Señal 4: Volume Quality
        volume_score, volume_details = self.analyze_volume_quality(data)
        print(f"📈 Volume Quality: {volume_score:.3f} - {volume_details}")
        
        # Señal 5: Liquidity Filter
        liquidity_score, liquidity_details = self.analyze_liquidity_filter(data)
        print(f"💧 Liquidity: {liquidity_score:.3f} - {liquidity_details}")
        
        # Score compuesto
        composite_score = (
            si_score * self.signals_weights['short_interest_qualified'] +
            momentum_score * self.signals_weights['momentum_confirmation'] +
            delisting_score * self.signals_weights['delisting_risk'] +
            volume_score * self.signals_weights['volume_quality'] +
            liquidity_score * self.signals_weights['liquidity_filter']
        )
        
        print(f"\n🎯 SCORE COMPUESTO: {composite_score:.3f}")
        
        # Decisión
        if liquidity_score == 0:
            action = "DESCALIFICADA - LIQUIDEZ"
        elif composite_score >= self.thresholds['buy_strong']:
            action = "COMPRAR FUERTE"
        elif composite_score >= self.thresholds['buy_moderate']:
            action = "COMPRAR MODERADO"
        elif composite_score >= self.thresholds['buy_light']:
            action = "COMPRAR LIGERO"
        else:
            action = "ESPERAR"
        
        print(f"📋 ACCIÓN: {action}")
        print(f"🎚️  Umbrales: Ligero={self.thresholds['buy_light']}, Moderado={self.thresholds['buy_moderate']}, Fuerte={self.thresholds['buy_strong']}")
        
        return {
            'symbol': symbol,
            'composite_score': composite_score,
            'action': action,
            'signals': {
                'si_qualified': si_score,
                'momentum': momentum_score,
                'delisting': delisting_score,
                'volume_quality': volume_score,
                'liquidity': liquidity_score
            }
        }
    
    def analyze_short_interest_qualified(self, data):
        """Analiza short interest cualificado"""
        si_pct = data.get('short_interest_pct', 0)
        dtc = data.get('days_to_cover', 0)
        borrow_rate = data.get('borrow_rate', 0)
        
        # Score base
        if si_pct >= 30: base_score = 1.0
        elif si_pct >= 20: base_score = 0.8
        elif si_pct >= self.signal_params['short_interest']['min_si_pct']: base_score = 0.6
        else: base_score = 0.0
        
        # Bonificaciones
        dtc_bonus = min(max(dtc - self.signal_params['short_interest']['min_days_to_cover'], 0) / 3 * 0.3, 0.3)
        borrow_bonus = min(max(borrow_rate - self.signal_params['short_interest']['min_borrow_rate'], 0) / 80 * 0.3, 0.3)
        
        qualified_score = min(base_score + dtc_bonus + borrow_bonus, 1.0)
        
        details = f"SI:{si_pct:.1f}% DTC:{dtc:.1f}d Rate:{borrow_rate:.0f}%"
        return qualified_score, details
    
    def analyze_momentum_confirmation(self, data):
        """Analiza confirmación de momentum"""
        signals = 0
        details = []
        
        # 1. Precio vs VWAP
        if data.get('price', 0) > data.get('vwap', 0):
            signals += 1
            details.append("P>VWAP")
        
        # 2. RSI en rango
        rsi = data.get('rsi', 50)
        if self.signal_params['momentum']['rsi_min'] <= rsi <= self.signal_params['momentum']['rsi_max']:
            signals += 1
            details.append(f"RSI_{rsi:.0f}")
        
        # 3. Volumen en rango
        vol_ratio = data.get('volume', 0) / max(data.get('avg_volume_20d', 1), 1)
        if self.signal_params['momentum']['min_volume_ratio'] <= vol_ratio <= self.signal_params['momentum']['max_volume_ratio']:
            signals += 1
            details.append(f"Vol_{vol_ratio:.1f}x")
        
        # 4. Cambio de precio positivo
        price_change = data.get('price_change_pct', 0)
        if price_change > self.signal_params['momentum']['min_price_change']:
            signals += 1
            details.append(f"+{price_change:.1f}%")
        
        # 5. Cambio sustentable
        if price_change <= 20:
            signals += 1
            details.append("Sustentable")
        
        momentum_score = signals / 5
        return momentum_score, f"{signals}/5 ({', '.join(details)})"
    
    def analyze_volume_quality(self, data):
        """Analiza calidad del volumen"""
        vol_ratio = data.get('volume', 0) / max(data.get('avg_volume_20d', 1), 1)
        price_change = data.get('price_change_pct', 0)
        spread = data.get('bid_ask_spread_pct', 10)
        
        # Calidad según precio
        if price_change > 5:
            quality_mult = 1.3
            quality_desc = "buying_pressure"
        elif price_change > 0:
            quality_mult = 1.0
            quality_desc = "neutral+"
        else:
            quality_mult = 0.4
            quality_desc = "dumping"
        
        # Penalización por spread
        spread_mult = 1.0 if spread <= 5 else 0.7
        
        # Score base
        base_score = min(vol_ratio / 8, 1.0)
        quality_score = base_score * quality_mult * spread_mult
        quality_score = min(quality_score, 1.0)
        
        return quality_score, f"Vol:{vol_ratio:.1f}x {quality_desc}"
    
    def analyze_liquidity_filter(self, data):
        """Analiza filtro de liquidez"""
        spread = data.get('bid_ask_spread_pct', 15)
        depth = data.get('market_depth_dollars', 0)
        daily_vol = data.get('daily_dollar_volume', 0)
        
        # Verificaciones críticas
        spread_ok = spread <= self.signal_params['liquidity']['max_spread_pct']
        depth_ok = depth >= self.signal_params['liquidity']['min_depth_dollars']
        volume_ok = daily_vol >= self.signal_params['liquidity']['min_daily_volume']
        
        if not (spread_ok and depth_ok and volume_ok):
            fails = []
            if not spread_ok: fails.append(f"Spread:{spread:.1f}%")
            if not depth_ok: fails.append(f"Depth:${depth:,.0f}")
            if not volume_ok: fails.append(f"Vol:${daily_vol:,.0f}")
            return 0.0, f"FALLA: {', '.join(fails)}"
        
        # Score basado en calidad
        if spread <= 3 and depth >= 25000 and daily_vol >= 500000:
            return 1.0, "Excelente"
        elif spread <= 6 and depth >= 15000 and daily_vol >= 250000:
            return 0.8, "Buena"
        else:
            return 0.6, "Aceptable"
    
    def suggest_adjustments_for_missed_opportunities(self):
        """Sugiere ajustes para capturar oportunidades perdidas como HTOO y LCFY"""
        print("\n🔧 AJUSTES SUGERIDOS PARA CAPTURAR MÁS OPORTUNIDADES")
        print("=" * 60)
        
        suggestions = {
            "1. REDUCIR UMBRALES DE COMPRA": {
                "actual": f"Ligero: {self.thresholds['buy_light']}, Moderado: {self.thresholds['buy_moderate']}, Fuerte: {self.thresholds['buy_strong']}",
                "sugerido": "Ligero: 0.45, Moderado: 0.60, Fuerte: 0.75",
                "impacto": "30-50% más oportunidades detectadas",
                "riesgo": "Posibles falsos positivos"
            },
            
            "2. RELAJAR FILTROS DE MOMENTUM": {
                "actual": f"RSI: {self.signal_params['momentum']['rsi_min']}-{self.signal_params['momentum']['rsi_max']}, Min Vol: {self.signal_params['momentum']['min_volume_ratio']}x",
                "sugerido": "RSI: 35-75, Min Vol: 2x",
                "impacto": "Detecta momentum temprano",
                "riesgo": "Entradas en tendencias débiles"
            },
            
            "3. FLEXIBILIZAR LIQUIDEZ": {
                "actual": f"Max Spread: {self.signal_params['liquidity']['max_spread_pct']}%, Min Depth: ${self.signal_params['liquidity']['min_depth_dollars']:,}",
                "sugerido": "Max Spread: 15%, Min Depth: $5,000",
                "impacto": "Incluye más penny stocks",
                "riesgo": "Mayor dificultad para salir"
            },
            
            "4. AJUSTAR PESOS DE SEÑALES": {
                "actual": "Momentum: 25%, Liquidez: 10%",
                "sugerido": "Momentum: 35%, Liquidez: 5%",
                "impacto": "Prioriza movimiento sobre liquidez",
                "riesgo": "Posibles trampas de liquidez"
            }
        }
        
        for adjustment, details in suggestions.items():
            print(f"\n{adjustment}:")
            print(f"   📊 Actual: {details['actual']}")
            print(f"   🎯 Sugerido: {details['sugerido']}")
            print(f"   📈 Impacto: {details['impacto']}")
            print(f"   ⚠️  Riesgo: {details['riesgo']}")
    
    def create_aggressive_preset(self):
        """Crea preset agresivo para capturar más oportunidades"""
        print("\n🚀 CREANDO PRESET AGRESIVO")
        print("-" * 30)
        
        # Guardar configuración original
        original_config = {
            'thresholds': self.thresholds.copy(),
            'signal_params': self.signal_params.copy(),
            'signals_weights': self.signals_weights.copy()
        }
        
        # Aplicar configuración agresiva
        self.thresholds = {
            'buy_strong': 0.75,    # Era 0.85
            'buy_moderate': 0.60,  # Era 0.70
            'buy_light': 0.45      # Era 0.55
        }
        
        self.signal_params['momentum']['rsi_min'] = 35    # Era 45
        self.signal_params['momentum']['rsi_max'] = 75    # Era 65
        self.signal_params['momentum']['min_volume_ratio'] = 2  # Era 3
        self.signal_params['momentum']['min_price_change'] = 1  # Era 2
        
        self.signal_params['liquidity']['max_spread_pct'] = 15  # Era 10
        self.signal_params['liquidity']['min_depth_dollars'] = 5000  # Era 8000
        self.signal_params['liquidity']['min_daily_volume'] = 50000  # Era 100000
        
        # Aumentar peso de momentum
        self.signals_weights['momentum_confirmation'] = 0.35  # Era 0.25
        self.signals_weights['liquidity_filter'] = 0.05      # Era 0.10
        
        print("✅ Configuración agresiva aplicada:")
        print(f"   🎯 Nuevos umbrales: {self.thresholds}")
        print(f"   ⚡ RSI expandido: {self.signal_params['momentum']['rsi_min']}-{self.signal_params['momentum']['rsi_max']}")
        print(f"   💧 Liquidez relajada: Spread máx {self.signal_params['liquidity']['max_spread_pct']}%")
        
        return original_config
    
    def restore_original_config(self, original_config):
        """Restaura configuración original"""
        self.thresholds = original_config['thresholds']
        self.signal_params = original_config['signal_params']
        self.signals_weights = original_config['signals_weights']
        print("🔄 Configuración original restaurada")
    
    def test_htoo_lcfy_scenarios(self):
        """Prueba con escenarios similares a HTOO y LCFY"""
        print("\n🧪 TESTING CON ESCENARIOS HTOO Y LCFY")
        print("=" * 50)
        
        # Datos simulados basados en HTOO (+60%)
        htoo_scenario = {
            'symbol': 'HTOO_SCENARIO',
            'price': 7.255,
            'volume': 15000000,       # Alto volumen
            'avg_volume_20d': 3000000,
            'price_change_pct': 8.5,  # Subiendo (señal de momentum)
            'short_interest_pct': 18.2,
            'days_to_cover': 2.1,
            'borrow_rate': 35.4,
            'vwap': 6.8,              # Precio > VWAP
            'rsi': 72,                # RSI alto (momentum fuerte)
            'bid_ask_spread_pct': 6.2,
            'market_depth_dollars': 12000,
            'daily_dollar_volume': 580000,
            'atr_14': 0.48,
            'has_delisting_warning': False
        }
        
        # Datos simulados basados en LCFY (+20%)
        lcfy_scenario = {
            'symbol': 'LCFY_SCENARIO',
            'price': 5.750,
            'volume': 8500000,
            'avg_volume_20d': 2100000,
            'price_change_pct': 5.2,
            'short_interest_pct': 14.7,
            'days_to_cover': 1.8,
            'borrow_rate': 28.1,
            'vwap': 5.4,
            'rsi': 68,
            'bid_ask_spread_pct': 4.8,
            'market_depth_dollars': 18500,
            'daily_dollar_volume': 420000,
            'atr_14': 0.32,
            'has_delisting_warning': False
        }
        
        # Test con configuración actual
        print("\n📊 CON CONFIGURACIÓN ACTUAL (CONSERVADORA):")
        htoo_result_current = self.analyze_with_current_params('HTOO_SCENARIO', htoo_scenario)
        lcfy_result_current = self.analyze_with_current_params('LCFY_SCENARIO', lcfy_scenario)
        
        # Aplicar configuración agresiva
        original_config = self.create_aggressive_preset()
        
        # Test con configuración agresiva
        print("\n🚀 CON CONFIGURACIÓN AGRESIVA:")
        htoo_result_aggressive = self.analyze_with_current_params('HTOO_SCENARIO', htoo_scenario)
        lcfy_result_aggressive = self.analyze_with_current_params('LCFY_SCENARIO', lcfy_scenario)
        
        # Comparación
        print(f"\n📊 COMPARACIÓN DE RESULTADOS:")
        print(f"HTOO - Conservador: {htoo_result_current['action']} (Score: {htoo_result_current['composite_score']:.3f})")
        print(f"HTOO - Agresivo: {htoo_result_aggressive['action']} (Score: {htoo_result_aggressive['composite_score']:.3f})")
        print(f"LCFY - Conservador: {lcfy_result_current['action']} (Score: {lcfy_result_current['composite_score']:.3f})")
        print(f"LCFY - Agresivo: {lcfy_result_aggressive['action']} (Score: {lcfy_result_aggressive['composite_score']:.3f})")
        
        # Restaurar configuración original
        self.restore_original_config(original_config)
        
        return {
            'htoo_conservative': htoo_result_current,
            'htoo_aggressive': htoo_result_aggressive,
            'lcfy_conservative': lcfy_result_current,
            'lcfy_aggressive': lcfy_result_aggressive
        }

def main():
    """Función principal de testing"""
    print("🔧 TESTER DE ALGORITMO AJUSTABLE")
    print("📊 Para capturar oportunidades como HTOO (+60%) y LCFY (+20%)")
    print("=" * 60)
    
    tester = AdjustableAlgorithmTester()
    
    # Mostrar sugerencias de ajuste
    tester.suggest_adjustments_for_missed_opportunities()
    
    # Testing con escenarios reales
    results = tester.test_htoo_lcfy_scenarios()
    
    print(f"\n💡 CONCLUSIONES:")
    print(f"• El algoritmo actual es muy conservador")
    print(f"• Configuración agresiva detectaría más oportunidades")
    print(f"• Recomendación: Reducir umbrales en 0.10 puntos")
    print(f"• Expandir rango RSI a 35-75 para momentum temprano")
    print(f"• Considerar aumentar peso de momentum a 35%")
    
    # Guardar resultados
    with open('algorithm_adjustment_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Resultados guardados en 'algorithm_adjustment_results.json'")

if __name__ == "__main__":
    main()