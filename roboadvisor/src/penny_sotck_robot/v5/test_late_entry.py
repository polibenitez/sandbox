#!/usr/bin/env python3
"""
Test script para verificar detección de entrada tardía (día 3)
"""

import sys
import os

# Agregar utils al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from penny_stock_advisor_v5 import PennyStockAdvisorV5
from integration_v5_trading_manager import MarketContextAnalyzerV5

def test_late_entry_asst():
    """Test específico para ASST"""
    print("\n" + "="*70)
    print("TEST: Detección de Entrada Tardía (Día 3) - ASST")
    print("="*70)

    # Inicializar sistema
    advisor = PennyStockAdvisorV5(config_preset="balanced", enable_logging=True)
    market_context = MarketContextAnalyzerV5()

    # Obtener contexto de mercado
    print("\n🌍 Obteniendo contexto de mercado...")
    context = market_context.get_market_context()
    print(f"   • SPY: {context['spy_trend']}")
    print(f"   • VIX: {context['vix']:.1f}")

    # Analizar ASST
    print("\n📊 Analizando ASST...")
    try:
        market_data, historical_data = advisor.get_enhanced_market_data("ASST")

        if market_data is None:
            print("   ❌ No se pudieron obtener datos de ASST")
            return

        print(f"   • Precio actual: ${market_data['price']:.3f}")
        print(f"   • Volumen: {market_data['volume']:,}")
        print(f"   • Volumen promedio: {market_data['avg_volume_20d']:,}")
        print(f"   • RSI: {market_data['rsi']:.0f}")

        # Ejecutar análisis completo
        print("\n🔬 Ejecutando análisis V5...")
        analysis = advisor.analyze_symbol_v5("ASST", market_data, historical_data, context)

        # Mostrar resultados
        decision = analysis['trading_decision']
        phase2 = analysis['phase2_trigger']
        explosion = phase2['explosion_info']

        print("\n" + "="*70)
        print("RESULTADOS DEL ANÁLISIS")
        print("="*70)

        print(f"\n📈 DECISIÓN: {decision['action']}")
        print(f"   • Score final: {decision.get('score', 0):.1f}/100")
        if 'raw_score' in decision:
            print(f"   • Score bruto: {decision['raw_score']:.1f}/100")
            print(f"   • Penalizaciones: {decision.get('penalty_applied', 0):.0f} pts")
        if 'reason' in decision:
            print(f"   • Razón: {decision['reason']}")

        print(f"\n⏰ EXPLOSIÓN:")
        print(f"   • Día: {explosion['explosion_day']}")
        print(f"   • Status: {explosion['status']}")
        print(f"   • Early enough: {explosion['is_early_enough']}")
        print(f"   • Late entry qualified: {explosion.get('late_entry_qualified', False)}")
        print(f"   • Penalización: {explosion['penalty_points']} pts")

        if decision.get('is_late_entry', False):
            print(f"\n⚠️  ENTRADA TARDÍA DETECTADA:")
            print(f"   • Día de explosión: {decision.get('explosion_day', 'N/A')}")
            print(f"   • Posición recomendada: {decision.get('position_size_pct', 0):.1f}% (reducida)")
            if 'stop_loss' in decision:
                print(f"   • Stop loss: ${decision['stop_loss']:.3f}")

        if 'warnings' in decision:
            print(f"\n⚠️  ADVERTENCIAS:")
            for warning in decision['warnings']:
                print(f"   • {warning}")

        print(f"\n🎯 SEÑALES CLAVE:")
        for signal in decision.get('key_signals', [])[:5]:
            print(f"   • {signal}")

        print("\n" + "="*70)

        # Mostrar configuración de día 3
        print("\n⚙️  CONFIGURACIÓN DÍA 3:")
        print(f"   • Permitir día 3: {advisor.trigger_params.get('allow_late_entry_day3', False)}")
        print(f"   • Volumen mínimo día 3: {advisor.trigger_params.get('day3_min_volume_spike', 0)}x")
        print(f"   • RSI máximo día 3: {advisor.trigger_params.get('day3_max_rsi', 0)}")
        print(f"   • Cambio precio máx día 3: {advisor.trigger_params.get('day3_max_price_change_3d', 0)}%")
        print(f"   • Penalización reducida: {advisor.trigger_params.get('day3_penalty_reduced', 0)} pts")

        print("\n" + "="*70)

    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_late_entry_asst()
