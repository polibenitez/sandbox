#!/usr/bin/env python3
"""
TESTER DEL ALGORITMO MEJORADO
============================

Script para probar y comparar el algoritmo original vs el mejorado
con diferentes escenarios de prueba.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import json

class AlgorithmTester:
    """Clase para testing del algoritmo mejorado"""
    
    def __init__(self):
        self.test_scenarios = self.create_test_scenarios()
        self.results = []
    
    def create_test_scenarios(self):
        """Crea escenarios de prueba realistas"""
        scenarios = {
            "squeeze_perfecto": {
                "name": "Squeeze Perfecto",
                "description": "Condiciones ideales para short squeeze",
                "data": {
                    'price': 0.75,
                    'volume': 25000000,
                    'avg_volume_20d': 3000000,
                    'price_change_pct': 8.5,
                    'short_interest_pct': 32.1,
                    'days_to_cover': 4.8,
                    'borrow_rate': 85.3,
                    'vwap': 0.71,
                    'rsi': 62,
                    'bid_ask_spread_pct': 2.1,
                    'market_depth_dollars': 45000,
                    'daily_dollar_volume': 1200000,
                    'atr_14': 0.048,
                    'has_delisting_warning': False
                },
                "expected_action": "COMPRAR FUERTE"
            },
            
            "falso_positivo_clasico": {
                "name": "Falso Positivo Clásico",
                "description": "Alto SI pero fácil de cubrir y bajando",
                "data": {
                    'price': 0.18,
                    'volume': 80000000,
                    'avg_volume_20d': 12000000,
                    'price_change_pct': -8.2,
                    'short_interest_pct': 38.7,
                    'days_to_cover': 0.8,  # MUY fácil cubrir
                    'borrow_rate': 15.2,   # Barato mantener
                    'vwap': 0.22,          # Precio BAJO VWAP
                    'rsi': 25,             # Oversold extremo
                    'bid_ask_spread_pct': 12.5,
                    'market_depth_dollars': 3500,
                    'daily_dollar_volume': 180000,
                    'atr_14': 0.023,
                    'has_delisting_warning': True
                },
                "expected_action": "ESPERAR"  # Algoritmo mejorado debe rechazar
            },
            
            "momentum_sin_fundamento": {
                "name": "Momentum Sin Fundamento",
                "description": "Subida técnica pero sin SI significativo",
                "data": {
                    'price': 1.25,
                    'volume': 8000000,
                    'avg_volume_20d': 1500000,
                    'price_change_pct': 12.3,
                    'short_interest_pct': 6.2,  # SI muy bajo
                    'days_to_cover': 2.1,
                    'borrow_rate': 8.5,
                    'vwap': 1.18,
                    'rsi': 68,
                    'bid_ask_spread_pct': 3.8,
                    'market_depth_dollars': 35000,
                    'daily_dollar_volume': 950000,
                    'atr_14': 0.062,
                    'has_delisting_warning': False
                },
                "expected_action": "ESPERAR"  # Sin SI no hay squeeze potential
            },
            
            "trampa_de_liquidez": {
                "name": "Trampa de Liquidez",
                "description": "Buenas señales pero liquidez terrible",
                "data": {
                    'price': 0.45,
                    'volume': 15000000,
                    'avg_volume_20d': 2500000,
                    'price_change_pct': 6.8,
                    'short_interest_pct': 28.3,
                    'days_to_cover': 3.9,
                    'borrow_rate': 65.1,
                    'vwap': 0.42,
                    'rsi': 59,
                    'bid_ask_spread_pct': 18.5,  # SPREAD HORRIBLE
                    'market_depth_dollars': 2100,  # SIN DEPTH
                    'daily_dollar_volume': 85000,  # VOLUMEN $ MÍNIMO
                    'atr_14': 0.035,
                    'has_delisting_warning': False
                },
                "expected_action": "DESCALIFICADA"  # Filtro de liquidez debe rechazar
            },
            
            "oportunidad_moderada": {
                "name": "Oportunidad Moderada",
                "description": "Señales decentes pero no espectaculares",
                "data": {
                    'price': 0.92,
                    'volume': 6000000,
                    'avg_volume_20d': 1800000,
                    'price_change_pct': 4.1,
                    'short_interest_pct': 18.7,
                    'days_to_cover': 2.8,
                    'borrow_rate': 35.2,
                    'vwap': 0.89,
                    'rsi': 56,
                    'bid_ask_spread_pct': 4.2,
                    'market_depth_dollars': 22000,
                    'daily_dollar_volume': 450000,
                    'atr_14': 0.041,
                    'has_delisting_warning': False
                },
                "expected_action": "COMPRAR LIGERO"
            }
        }
        
        return scenarios
    
    def analyze_with_improved_algorithm(self, scenario_data):
        """Simula análisis con algoritmo mejorado"""
        
        # Señal 1: Short Interest Cualificado (25%)
        si_pct = scenario_data['short_interest_pct']
        dtc = scenario_data['days_to_cover']
        borrow_rate = scenario_data['borrow_rate']
        
        si_base = min(si_pct / 30, 1.0)
        dtc_bonus = min(max(dtc - 2, 0) / 3 * 0.3, 0.3)
        borrow_bonus = min(max(borrow_rate - 20, 0) / 80 * 0.3, 0.3)
        si_qualified_score = min(si_base + dtc_bonus + borrow_bonus, 1.0)
        
        # Señal 2: Confirmación de Momentum (25%)  
        momentum_signals = 0
        if scenario_data['price'] > scenario_data['vwap']: momentum_signals += 1
        if 45 <= scenario_data['rsi'] <= 65: momentum_signals += 1
        vol_ratio = scenario_data['volume'] / scenario_data['avg_volume_20d']
        if 3 <= vol_ratio <= 15: momentum_signals += 1
        if scenario_data['price_change_pct'] > 2: momentum_signals += 1
        if scenario_data['price_change_pct'] <= 20: momentum_signals += 1
        momentum_score = momentum_signals / 5
        
        # Señal 3: Delisting Risk (20%)
        delisting_score = 0.8 if scenario_data['price'] < 1 and scenario_data['has_delisting_warning'] else 0.2
        
        # Señal 4: Calidad de Volumen (20%)
        volume_base = min(vol_ratio / 8, 1.0)
        if scenario_data['price_change_pct'] > 5:
            quality_mult = 1.2
        elif scenario_data['price_change_pct'] > 0:
            quality_mult = 1.0
        else:
            quality_mult = 0.4
        
        spread_mult = 1.0 if scenario_data['bid_ask_spread_pct'] <= 5 else 0.7
        volume_quality_score = volume_base * quality_mult * spread_mult
        volume_quality_score = min(volume_quality_score, 1.0)
        
        # Señal 5: Filtro de Liquidez (10%) - CRÍTICO
        liquidity_fail = (scenario_data['bid_ask_spread_pct'] > 10 or 
                         scenario_data['market_depth_dollars'] < 8000 or
                         scenario_data['daily_dollar_volume'] < 100000)
        
        if liquidity_fail:
            return {
                'composite_score': 0,
                'action': 'DESCALIFICADA - LIQUIDEZ INSUFICIENTE',
                'signals': {
                    'si_qualified': si_qualified_score,
                    'momentum': momentum_score, 
                    'delisting': delisting_score,
                    'volume_quality': volume_quality_score,
                    'liquidity': 0
                },
                'liquidity_fail_reason': f"Spread: {scenario_data['bid_ask_spread_pct']:.1f}%, Depth: ${scenario_data['market_depth_dollars']:,.0f}"
            }
        
        # Score compuesto
        composite_score = (
            si_qualified_score * 0.25 +
            momentum_score * 0.25 +
            delisting_score * 0.20 +
            volume_quality_score * 0.20 +
            1.0 * 0.10  # Liquidez OK
        )
        
        # Decisión con umbrales exigentes
        if composite_score >= 0.85:
            action = "COMPRAR FUERTE"
        elif composite_score >= 0.70:
            action = "COMPRAR MODERADO"
        elif composite_score >= 0.55:
            action = "COMPRAR LIGERO"
        else:
            action = "ESPERAR"
        
        return {
            'composite_score': composite_score,
            'action': action,
            'signals': {
                'si_qualified': si_qualified_score,
                'momentum': momentum_score,
                'delisting': delisting_score, 
                'volume_quality': volume_quality_score,
                'liquidity': 1.0
            }
        }
    
    def analyze_with_original_algorithm(self, scenario_data):
        """Simula análisis con algoritmo original (para comparación)"""
        
        # Algoritmo original simplificado
        si_score = min(scenario_data['short_interest_pct'] / 30, 1.0)
        delisting_score = 0.8 if scenario_data['price'] < 1 and scenario_data['has_delisting_warning'] else 0.2
        vol_ratio = scenario_data['volume'] / scenario_data['avg_volume_20d']
        volume_score = min(vol_ratio / 10, 1.0)
        
        # Sin filtros de momentum o liquidez
        original_score = si_score * 0.4 + delisting_score * 0.3 + volume_score * 0.3
        
        # Umbrales originales más permisivos
        if original_score >= 0.8:
            action = "COMPRAR FUERTE"
        elif original_score >= 0.6:
            action = "COMPRAR MODERADO"
        elif original_score >= 0.4:
            action = "COMPRAR LIGERO"
        else:
            action = "ESPERAR"
        
        return {
            'composite_score': original_score,
            'action': action,
            'signals': {
                'si_basic': si_score,
                'delisting': delisting_score,
                'volume_basic': volume_score
            }
        }
    
    def run_comprehensive_test(self):
        """Ejecuta test completo de todos los escenarios"""
        print("🧪 TESTING COMPLETO DEL ALGORITMO MEJORADO")
        print("=" * 55)
        
        results = []
        
        for scenario_id, scenario in self.test_scenarios.items():
            print(f"\n📊 ESCENARIO: {scenario['name']}")
            print(f"📝 {scenario['description']}")
            print("-" * 40)
            
            # Análisis con algoritmo mejorado
            improved_result = self.analyze_with_improved_algorithm(scenario['data'])
            
            # Análisis con algoritmo original 
            original_result = self.analyze_with_original_algorithm(scenario['data'])
            
            # Comparación
            expected = scenario['expected_action']
            improved_correct = improved_result['action'].startswith(expected) or (expected == "DESCALIFICADA" and "DESCALIFICADA" in improved_result['action'])
            
            print(f"🎯 Resultado esperado: {expected}")
            print(f"🚀 Algoritmo mejorado: {improved_result['action']} (Score: {improved_result['composite_score']:.3f})")
            print(f"📊 Algoritmo original: {original_result['action']} (Score: {original_result['composite_score']:.3f})")
            
            if improved_correct:
                print("✅ ALGORITMO MEJORADO: CORRECTO")
            else:
                print("❌ ALGORITMO MEJORADO: INCORRECTO")
            
            # Detalles de señales mejoradas
            if 'liquidity_fail_reason' in improved_result:
                print(f"🚫 Razón descalificación: {improved_result['liquidity_fail_reason']}")
            else:
                signals = improved_result['signals']
                print(f"📋 Señales: SI_Q:{signals['si_qualified']:.2f} Mom:{signals['momentum']:.2f} Vol_Q:{signals['volume_quality']:.2f}")
            
            results.append({
                'scenario': scenario['name'],
                'expected': expected,
                'improved': improved_result,
                'original': original_result,
                'improved_correct': improved_correct
            })
        
        # Resumen final
        self.print_test_summary(results)
        return results
    
    def print_test_summary(self, results):
        """Imprime resumen de resultados del test"""
        print("\n" + "=" * 55)
        print("📊 RESUMEN DE TESTING")
        print("=" * 55)
        
        total_tests = len(results)
        correct_improved = sum(1 for r in results if r['improved_correct'])
        
        print(f"🎯 Tests ejecutados: {total_tests}")
        print(f"✅ Algoritmo mejorado correcto: {correct_improved}/{total_tests} ({correct_improved/total_tests*100:.1f}%)")
        
        # Análisis de mejoras
        print(f"\n🔍 ANÁLISIS COMPARATIVO:")
        
        for result in results:
            improved_action = result['improved']['action']
            original_action = result['original']['action']
            
            if improved_action != original_action:
                print(f"📈 {result['scenario']}:")
                print(f"   Original: {original_action}")
                print(f"   Mejorado: {improved_action}")
                
                if result['improved_correct']:
                    print("   ✅ Mejora correcta")
                else:
                    print("   ⚠️  Cambio cuestionable")
        
        print(f"\n💡 CONCLUSIONES:")
        
        if correct_improved >= total_tests * 0.8:
            print("✅ El algoritmo mejorado funciona correctamente")
            print("🎯 Las mejoras implementadas son efectivas")
            print("🚀 Listo para uso en producción")
        else:
            print("⚠️  El algoritmo necesita ajustes")
            print("🔧 Revisar pesos y umbrales")

def run_performance_comparison():
    """Ejecuta comparación de performance entre algoritmos"""
    print("\n🏁 COMPARACIÓN DE PERFORMANCE")
    print("=" * 35)
    
    # Simulación de 100 stocks aleatorios
    np.random.seed(42)  # Para reproducibilidad
    
    tester = AlgorithmTester()
    improved_trades = 0
    original_trades = 0
    
    print("🔄 Simulando 100 penny stocks aleatorios...")
    
    for i in range(100):
        # Genera datos aleatorios realistas
        price = np.random.uniform(0.05, 2.0)
        
        random_data = {
            'price': price,
            'volume': np.random.randint(500000, 50000000),
            'avg_volume_20d': np.random.randint(200000, 15000000),
            'price_change_pct': np.random.uniform(-15, 25),
            'short_interest_pct': np.random.uniform(3, 50),
            'days_to_cover': np.random.uniform(0.2, 8.0),
            'borrow_rate': np.random.uniform(2, 150),
            'vwap': price * np.random.uniform(0.85, 1.15),
            'rsi': np.random.uniform(10, 90),
            'bid_ask_spread_pct': np.random.uniform(0.5, 25),
            'market_depth_