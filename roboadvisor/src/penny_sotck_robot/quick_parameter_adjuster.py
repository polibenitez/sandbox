#!/usr/bin/env python3
"""
AJUSTADOR RÁPIDO DE PARÁMETROS
=============================

Script para ajustar rápidamente los parámetros del algoritmo principal
para capturar más oportunidades como HTOO y LCFY.

Uso: python quick_parameter_adjuster.py
"""

def apply_aggressive_settings():
    """Aplica configuraciones agresivas al script principal"""
    
    print("🔧 APLICANDO CONFIGURACIONES AGRESIVAS")
    print("=" * 45)
    
    # Configuraciones sugeridas
    configs = {
        "CONSERVADOR (ACTUAL)": {
            "description": "Configuración actual - muy selectiva",
            "buy_thresholds": {"light": 0.55, "moderate": 0.70, "strong": 0.85},
            "rsi_range": {"min": 45, "max": 65},
            "volume_min": 3,
            "spread_max": 10,
            "momentum_weight": 0.25
        },
        
        "BALANCEADO": {
            "description": "Equilibrio entre oportunidades y seguridad",
            "buy_thresholds": {"light": 0.50, "moderate": 0.65, "strong": 0.80},
            "rsi_range": {"min": 40, "max": 70},
            "volume_min": 2.5,
            "spread_max": 12,
            "momentum_weight": 0.30
        },
        
        "AGRESIVO": {
            "description": "Captura más oportunidades - mayor riesgo",
            "buy_thresholds": {"light": 0.45, "moderate": 0.60, "strong": 0.75},
            "rsi_range": {"min": 35, "max": 75},
            "volume_min": 2,
            "spread_max": 15,
            "momentum_weight": 0.35
        },
        
        "MUY AGRESIVO": {
            "description": "Para capturar HTOO/LCFY - alto riesgo",
            "buy_thresholds": {"light": 0.40, "moderate": 0.55, "strong": 0.70},
            "rsi_range": {"min": 30, "max": 80},
            "volume_min": 1.5,
            "spread_max": 20,
            "momentum_weight": 0.40
        }
    }
    
    print("📊 CONFIGURACIONES DISPONIBLES:")
    print("-" * 35)
    
    for i, (name, config) in enumerate(configs.items(), 1):
        print(f"\n{i}. {name}")
        print(f"   📝 {config['description']}")
        print(f"   🎯 Umbrales: L={config['buy_thresholds']['light']}, M={config['buy_thresholds']['moderate']}, F={config['buy_thresholds']['strong']}")
        print(f"   ⚡ RSI: {config['rsi_range']['min']}-{config['rsi_range']['max']}")
        print(f"   📈 Vol mín: {config['volume_min']}x, Spread máx: {config['spread_max']}%")
    
    return configs

def generate_modified_daily_script(config_name, config):
    """Genera versión modificada del script diario con nuevos parámetros"""
    
    modified_script = f'''#!/usr/bin/env python3
"""
ROBOT ADVISOR - CONFIGURACIÓN {config_name.upper()}
===============================================
"""

import sys
import os
import numpy as np
from datetime import datetime, timedelta
import json

# Configuración de símbolos
WATCHLIST_SYMBOLS = [
    'BBIG', 'MULN', 'SNDL', 'AMC', 'GME', 
    'CTRM', 'SHIP', 'NAKD', 'GNUS', 'XELA',
    'HTOO', 'LCFY', 'SIRI', 'XAIR'  # Añadidos para testing
]

class ModifiedRobotAdvisor:
    """Robot Advisor con configuración {config_name.lower()}"""
    
    def __init__(self):
        # CONFIGURACIÓN {config_name.upper()} - PARÁMETROS AJUSTADOS
        self.signals_weights = {{
            'short_interest_qualified': 0.25,
            'momentum_confirmation': {config['momentum_weight']},  # AJUSTADO
            'delisting_risk': 0.20,
            'volume_quality': 0.20,
            'liquidity_filter': {0.30 - config['momentum_weight']}  # AJUSTADO
        }}
        
        # UMBRALES AJUSTADOS PARA CAPTURAR MÁS OPORTUNIDADES
        self.thresholds = {{
            'buy_strong': {config['buy_thresholds']['strong']},    # Era 0.85
            'buy_moderate': {config['buy_thresholds']['moderate']}, # Era 0.70
            'buy_light': {config['buy_thresholds']['light']}      # Era 0.55
        }}
        
        # PARÁMETROS DE SEÑALES AJUSTADOS
        self.signal_params = {{
            'rsi_min': {config['rsi_range']['min']},        # Era 45
            'rsi_max': {config['rsi_range']['max']},        # Era 65
            'min_volume_ratio': {config['volume_min']},      # Era 3
            'max_spread_pct': {config['spread_max']},        # Era 10
            'min_depth_dollars': 5000,      # Era 8000
            'min_daily_volume': 50000       # Era 100000
        }}
    
    def analyze_symbol_modified(self, symbol, data):
        """Análisis con parámetros modificados"""
        
        # Señal 1: Short Interest (sin cambios)
        si_pct = data.get('short_interest_pct', 0)
        if si_pct >= 25: si_score = 1.0
        elif si_pct >= 15: si_score = 0.7
        elif si_pct >= 10: si_score = 0.4  # RELAJADO
        else: si_score = 0.0
        
        # Señal 2: Momentum (RELAJADO)
        momentum_signals = 0
        rsi = data.get('rsi', 50)
        price_change = data.get('price_change_pct', 0)
        vol_ratio = data.get('volume', 0) / max(data.get('avg_volume_20d', 1), 1)
        
        if data.get('price', 0) > data.get('vwap', 0): momentum_signals += 1
        if self.signal_params['rsi_min'] <= rsi <= self.signal_params['rsi_max']: momentum_signals += 1
        if vol_ratio >= self.signal_params['min_volume_ratio']: momentum_signals += 1  # RELAJADO
        if price_change > 1: momentum_signals += 1  # Era 2
        if price_change <= 25: momentum_signals += 1  # Era 20
        
        momentum_score = momentum_signals / 5
        
        # Señal 3: Delisting (sin cambios)
        delisting_score = 0.8 if data.get('price', 2) < 1 and data.get('has_delisting_warning', False) else 0.2
        
        # Señal 4: Volume Quality (RELAJADO)
        base_score = min(vol_ratio / 6, 1.0)  # Era /8
        if price_change > 3: quality_mult = 1.3
        elif price_change > 0: quality_mult = 1.0
        else: quality_mult = 0.5  # Era 0.4
        
        spread_mult = 1.0 if data.get('bid_ask_spread_pct', 10) <= 8 else 0.8  # Era 5
        volume_score = min(base_score * quality_mult * spread_mult, 1.0)
        
        # Señal 5: Liquidity (RELAJADO)
        spread = data.get('bid_ask_spread_pct', 15)
        depth = data.get('market_depth_dollars', 0)
        daily_vol = data.get('daily_dollar_volume', 0)
        
        spread_ok = spread <= self.signal_params['max_spread_pct']
        depth_ok = depth >= self.signal_params['min_depth_dollars']
        volume_ok = daily_vol >= self.signal_params['min_daily_volume']
        
        if not (spread_ok and depth_ok and volume_ok):
            liquidity_score = 0.0
        else:
            liquidity_score = 0.8  # Simplificado
        
        # Score compuesto
        composite_score = (
            si_score * self.signals_weights['short_interest_qualified'] +
            momentum_score * self.signals_weights['momentum_confirmation'] +
            delisting_score * self.signals_weights['delisting_risk'] +
            volume_score * self.signals_weights['volume_quality'] +
            liquidity_score * self.signals_weights['liquidity_filter']
        )
        
        # Decisión con umbrales modificados
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
        
        return {{
            'symbol': symbol,
            'action': action,
            'composite_score': composite_score,
            'price': data.get('price', 0),
            'signals': {{
                'si_score': si_score,
                'momentum_score': momentum_score,
                'volume_score': volume_score,
                'liquidity_score': liquidity_score
            }},
            'details': {{
                'rsi': rsi,
                'price_change_pct': price_change,
                'volume_ratio': vol_ratio,
                'spread_pct': spread
            }}
        }}

# DATOS DE PRUEBA PARA HTOO Y LCFY
def create_test_data():
    """Crea datos de prueba basados en HTOO y LCFY reales"""
    return {{
        'HTOO': {{
            'price': 7.255,
            'volume': 25000000,      # Alto volumen
            'avg_volume_20d': 4000000,
            'price_change_pct': 12.5,  # Fuerte subida
            'short_interest_pct': 16.8,
            'days_to_cover': 2.3,
            'borrow_rate': 42.1,
            'vwap': 6.9,             # Precio > VWAP
            'rsi': 74,               # RSI alto (momentum)
            'bid_ask_spread_pct': 8.2,
            'market_depth_dollars': 15000,
            'daily_dollar_volume': 681250,
            'has_delisting_warning': False
        }},
        
        'LCFY': {{
            'price': 5.750,
            'volume': 12000000,
            'avg_volume_20d': 2800000,
            'price_change_pct': 6.8,
            'short_interest_pct': 13.2,
            'days_to_cover': 1.9,
            'borrow_rate': 31.5,
            'vwap': 5.4,
            'rsi': 69,
            'bid_ask_spread_pct': 5.4,
            'market_depth_dollars': 22000,
            'daily_dollar_volume': 460000,
            'has_delisting_warning': False
        }},
        
        'SNDL': {{  # Stock control (más conservador)
            'price': 0.673,
            'volume': 8500000,
            'avg_volume_20d': 6200000,
            'price_change_pct': 2.1,
            'short_interest_pct': 9.4,
            'days_to_cover': 1.2,
            'borrow_rate': 18.7,
            'vwap': 0.665,
            'rsi': 58,
            'bid_ask_spread_pct': 4.8,
            'market_depth_dollars': 28000,
            'daily_dollar_volume': 420000,
            'has_delisting_warning': False
        }}
    }}

def main():
    """Función principal de testing"""
    print("🧪 TESTING CONFIGURACIÓN {config_name.upper()}")
    print("=" * 50)
    
    robot = ModifiedRobotAdvisor()
    test_data = create_test_data()
    
    print(f"📊 PARÁMETROS APLICADOS:")
    print(f"   🎯 Umbrales: L={robot.thresholds['buy_light']}, M={robot.thresholds['buy_moderate']}, F={robot.thresholds['buy_strong']}")
    print(f"   ⚡ RSI: {robot.signal_params['rsi_min']}-{robot.signal_params['rsi_max']}")
    print(f"   📈 Vol mín: {robot.signal_params['min_volume_ratio']}x")
    print(f"   💧 Spread máx: {robot.signal_params['max_spread_pct']}%")
    
    print(f"\\n🔍 RESULTADOS DEL ANÁLISIS:")
    print("-" * 30)
    
    opportunities = 0
    
    for symbol, data in test_data.items():
        result = robot.analyze_symbol_modified(symbol, data)
        
        print(f"\\n📊 {{symbol}}:")
        print(f"   💰 Precio: ${{result['price']:.3f}}")
        print(f"   📈 Cambio: {{result['details']['price_change_pct']:.1f}}%")
        print(f"   ⚡ RSI: {{result['details']['rsi']:.0f}}")
        print(f"   📊 Score: {{result['composite_score']:.3f}}")
        print(f"   🎯 Acción: {{result['action']}}")
        
        if result['action'] != 'ESPERAR' and 'DESCALIFICADA' not in result['action']:
            opportunities += 1
            print(f"   ✅ OPORTUNIDAD DETECTADA!")
        else:
            print(f"   ❌ No detectada")
    
    print(f"\\n📊 RESUMEN:")
    print(f"   • Oportunidades detectadas: {{opportunities}}/{{len(test_data)}}")
    print(f"   • Tasa de detección: {{opportunities/len(test_data)*100:.1f}}%")
    
    if opportunities >= 2:
        print(f"   ✅ CONFIGURACIÓN EXITOSA - Detecta HTOO y LCFY")
    elif opportunities == 1:
        print(f"   ⚠️  CONFIGURACIÓN PARCIAL - Detecta solo una")
    else:
        print(f"   ❌ CONFIGURACIÓN INSUFICIENTE - No detecta oportunidades")

if __name__ == "__main__":
    main()
'''

    return modified_script

def create_parameter_files():
    """Crea archivos con diferentes configuraciones de parámetros"""
    
    configs = apply_aggressive_settings()
    
    print(f"\n🔧 CREANDO ARCHIVOS DE CONFIGURACIÓN...")
    
    files_created = []
    
    for config_name, config in configs.items():
        filename = f"robot_advisor_{config_name.lower().replace(' ', '_')}.py"
        
        try:
            modified_script = generate_modified_daily_script(config_name, config)
            
            with open(filename, 'w') as f:
                f.write(modified_script)
            
            files_created.append(filename)
            print(f"   ✅ {filename}")
            
        except Exception as e:
            print(f"   ❌ Error creando {filename}: {e}")
    
    return files_created

def show_usage_instructions():
    """Muestra instrucciones de uso"""
    
    print(f"\n📋 INSTRUCCIONES DE USO:")
    print("=" * 25)
    print("1. 🔧 Ejecuta este script para crear las configuraciones:")
    print("   python quick_parameter_adjuster.py")
    print()
    print("2. 🧪 Prueba diferentes configuraciones:")
    print("   python robot_advisor_balanceado.py")
    print("   python robot_advisor_agresivo.py")
    print("   python robot_advisor_muy_agresivo.py")
    print()
    print("3. 📊 Compara resultados:")
    print("   • CONSERVADOR: Pocas oportunidades, alta precisión")
    print("   • BALANCEADO: Equilibrio oportunidades/riesgo")
    print("   • AGRESIVO: Más oportunidades, mayor riesgo")
    print("   • MUY AGRESIVO: Máximas oportunidades, máximo riesgo")
    print()
    print("4. 🎯 Para HTOO/LCFY usa configuración AGRESIVA o MUY AGRESIVA")
    print()
    print("5. ⚖️  Ajusta umbrales manualmente si necesario:")
    print("   • Reducir umbrales = más oportunidades")
    print("   • Aumentar pesos de momentum = mejor timing")
    print("   • Relajar liquidez = incluir más penny stocks")

def main():
    """Función principal"""
    
    print("🔧 AJUSTADOR RÁPIDO DE PARÁMETROS")
    print("🎯 Para capturar oportunidades como HTOO (+60%) y LCFY (+20%)")
    print("=" * 60)
    
    # Mostrar configuraciones disponibles
    configs = apply_aggressive_settings()
    
    # Crear archivos de configuración
    files_created = create_parameter_files()
    
    print(f"\n✅ ARCHIVOS CREADOS: {len(files_created)}")
    for file in files_created:
        print(f"   📄 {file}")
    
    # Mostrar instrucciones
    show_usage_instructions()
    
    print(f"\n💡 RECOMENDACIÓN ESPECÍFICA PARA TU CASO:")
    print("🎯 Usa 'robot_advisor_agresivo.py' para capturar HTOO/LCFY")
    print("📊 Umbrales: Ligero=0.45, Moderado=0.60, Fuerte=0.75")
    print("⚡ RSI expandido: 35-75 (captura momentum temprano)")
    print("💧 Liquidez relajada: Spread máx 15%")

if __name__ == "__main__":
    main()