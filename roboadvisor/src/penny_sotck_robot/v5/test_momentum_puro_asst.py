#!/usr/bin/env python3
"""
Test script para verificar detección de ASST con Momentum Puro
"""

import sys
import os

# Agregar utils al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from penny_stock_advisor_v5 import PennyStockAdvisorV5
from integration_v5_trading_manager import MarketContextAnalyzerV5

def test_momentum_puro_asst():
    """Test específico para ASST con Momentum Puro"""
    print("\n" + "="*70)
    print("TEST: Detección de ASST con MOMENTUM PURO")
    print("="*70)

    # Inicializar sistema
    advisor = PennyStockAdvisorV5(config_preset="balanced", enable_logging=True)
    market_context = MarketContextAnalyzerV5()

    # Verificar configuración de Momentum Puro
    print("\n⚙️  CONFIGURACIÓN MOMENTUM PURO:")
    print(f"   • Habilitado: {advisor.momentum_puro_params.get('enabled', False)}")
    print(f"   • Volumen mínimo: {advisor.momentum_puro_params.get('min_volume_spike', 0)}x")
    print(f"   • Día máximo: {advisor.momentum_puro_params.get('max_explosion_day', 0)}")
    print(f"   • Cambio precio mín: {advisor.momentum_puro_params.get('min_price_change_3d', 0)}%")
    print(f"   • Cambio precio máx: {advisor.momentum_puro_params.get('max_price_change_3d', 0)}%")
    print(f"   • RSI rango: {advisor.momentum_puro_params.get('rsi_min_bounce', 0)}-{advisor.momentum_puro_params.get('rsi_max_momentum', 0)}")
    print(f"   • Threshold: {advisor.momentum_puro_params.get('score_threshold', 0)}")

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
        print(f"   • Ratio volumen: {market_data['volume']/market_data['avg_volume_20d']:.1f}x")
        print(f"   • RSI: {market_data['rsi']:.0f}")

        # Ejecutar análisis completo
        print("\n🔬 Ejecutando análisis V5 con ambas filosofías...")
        analysis = advisor.analyze_symbol_v5("ASST", market_data, historical_data, context)

        # Mostrar resultados
        print("\n" + "="*70)
        print("RESULTADOS DEL ANÁLISIS")
        print("="*70)

        opportunity_type = analysis.get('opportunity_type', 'Desconocido')
        print(f"\n🎯 TIPO DE OPORTUNIDAD: {opportunity_type.upper()}")

        decision = analysis['trading_decision']
        print(f"\n📈 DECISIÓN: {decision['action']}")
        print(f"   • Score final: {decision.get('score', 0):.1f}/100")

        if opportunity_type == "Momentum Puro":
            print(f"\n✅ ASST DETECTADO COMO MOMENTUM PURO!")

            if 'momentum_puro_stats' in decision:
                stats = decision['momentum_puro_stats']
                print(f"\n📊 ESTADÍSTICAS MOMENTUM PURO:")
                print(f"   • Volumen: {stats['volume_ratio']:.1f}x promedio")
                print(f"   • Cambio 3d: {stats['price_change_3d']:.1f}%")
                print(f"   • Breakout desde mínimo: {stats['breakout_pct']:.1%}")
                print(f"   • RSI: {stats['rsi']:.0f}")

            print(f"\n💰 PLAN DE TRADING:")
            print(f"   • Precio entrada: ${decision['current_price']:.3f}")
            print(f"   • Posición: {decision['position_size_pct']:.2f}% del capital")
            print(f"   • Stop loss: ${decision['stop_loss']:.3f}")
            print(f"   • TP1: ${decision['take_profit_1']:.3f}")
            print(f"   • TP2: ${decision['take_profit_2']:.3f}")
            print(f"   • Max holding: {decision.get('max_holding_days', 0)} días")

            print(f"\n⚠️  ADVERTENCIAS:")
            for warning in decision.get('warnings', []):
                print(f"   • {warning}")

            print(f"\n🎯 SEÑALES CLAVE:")
            for signal in decision.get('key_signals', []):
                print(f"   • {signal}")

        else:
            print(f"\n❌ ASST NO detectado como Momentum Puro")
            print(f"   Tipo: {opportunity_type}")

            # Mostrar info de momentum_puro si existe
            if analysis.get('momentum_puro'):
                mp = analysis['momentum_puro']
                print(f"\n   Evaluación Momentum Puro:")
                print(f"   • Califica: {mp.get('is_momentum_puro', False)}")
                print(f"   • Score: {mp.get('score', 0):.0f}/{mp.get('threshold', 50)}")
                print(f"   • Razón: {mp.get('reason', 'N/A')}")

                if 'signals' in mp:
                    print(f"   • Señales:")
                    for sig in mp['signals'][:5]:
                        print(f"      - {sig}")

        print("\n" + "="*70)

    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_momentum_puro_asst()
