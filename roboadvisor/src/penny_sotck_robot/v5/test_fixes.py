#!/usr/bin/env python3
"""
Test para verificar que los fixes funcionan
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from penny_stock_advisor_v5 import PennyStockAdvisorV5
from integration_v5_trading_manager import MarketContextAnalyzerV5

print("="*70)
print("TEST DE CORRECCIONES")
print("="*70)

# Inicializar sistema
print("\n🔧 Inicializando sistema...")
advisor = PennyStockAdvisorV5(config_preset="balanced", enable_logging=True)
market_context = MarketContextAnalyzerV5()

print("\n" + "="*70)
print("1️⃣ VERIFICACIÓN MODELO ML")
print("="*70)

print(f"   • Modelo ML cargado: {advisor.ml_predictor.model is not None}")
print(f"   • is_trained: {advisor.ml_predictor.is_trained}")

# Test de predicción
test_features = {
    'bb_width': 0.07,
    'adx': 19.0,
    'vol_ratio': 3.0,
    'rsi': 58,
    'macd_diff': 0.004,
    'atr_ratio': 0.015,
    'short_float': 0.16,
    'compression_days': 8,
    'volume_dry': 1,
    'price_range_pct': 6.0
}

prediction = advisor.ml_predictor.predict(test_features)
print(f"\n   Test de predicción:")
print(f"   • Predicción: {prediction['prediction']}")
print(f"   • Probabilidad: {prediction['probability']:.1%}")
print(f"   • Confianza: {prediction['confidence']}")
print(f"   • Modelo disponible: {prediction['model_available']}")

if prediction['model_available']:
    print(f"\n   ✅ MODELO ML FUNCIONANDO CORRECTAMENTE")
else:
    print(f"\n   ❌ MODELO ML NO DISPONIBLE")

print("\n" + "="*70)
print("2️⃣ VERIFICACIÓN REDDIT SENTIMENT")
print("="*70)

print(f"   • AlternativeDataProvider use_api: {advisor.alt_data_provider.use_api}")

# Test Reddit sentiment para ASST
reddit_asst = advisor.alt_data_provider.get_reddit_sentiment('ASST')
print(f"\n   Reddit sentiment para ASST:")
print(f"   • Mentions: {reddit_asst['mentions']}")
print(f"   • Sentiment: {reddit_asst['sentiment']}")
print(f"   • Sentiment score: {reddit_asst['sentiment_score']:.2f}")
print(f"   • Trending: {reddit_asst['trending']}")
print(f"   • Source: {reddit_asst['source']}")

if reddit_asst['source'] == 'local':
    print(f"\n   ✅ REDDIT USANDO DATOS LOCALES")
elif reddit_asst['source'] == 'default':
    print(f"\n   ⚠️ REDDIT USANDO DEFAULTS (no se encontraron datos locales)")
else:
    print(f"\n   ⚠️ REDDIT SOURCE: {reddit_asst['source']}")

# Test short borrow rate
short_asst = advisor.alt_data_provider.get_short_borrow_rate('ASST')
print(f"\n   Short borrow rate para ASST:")
print(f"   • Borrow rate: {short_asst['borrow_rate_pct']:.1f}%")
print(f"   • Availability: {short_asst['availability']}")
print(f"   • Squeeze risk: {short_asst['short_squeeze_risk']}")
print(f"   • Source: {short_asst['source']}")

if short_asst['source'] == 'local':
    print(f"\n   ✅ SHORT RATE USANDO DATOS LOCALES")

# Test combined
combined = advisor.alt_data_provider.get_combined_alternative_data('ASST')
print(f"\n   Datos combinados para ASST:")
print(f"   • Combined score: {combined['combined_score']:.1f}/100")

print("\n" + "="*70)
print("3️⃣ TEST COMPLETO CON ASST")
print("="*70)

try:
    market_data, historical_data = advisor.get_enhanced_market_data("ASST")

    if market_data:
        context = market_context.get_market_context()
        analysis = advisor.analyze_symbol_v5("ASST", market_data, historical_data, context)

        print(f"\n   Análisis de ASST:")
        print(f"   • Opportunity type: {analysis.get('opportunity_type', 'N/A')}")
        print(f"   • Final score: {analysis['final_score']:.1f}/100")
        print(f"   • ML adjustment: {analysis['ml_adjustment']['adjustment']:.0f} pts")
        print(f"   • ML probability: {analysis['ml_adjustment']['probability']:.1%}")

        if analysis['ml_adjustment']['model_available']:
            print(f"\n   ✅ ML ADJUSTMENT APLICADO")

        # Verificar datos alternativos en fase 3
        phase3 = analysis['phase3_context']
        print(f"\n   Fase 3 (Context + Alt Data):")
        print(f"   • Score: {phase3['score']:.1f}/100")
        for signal in phase3['signals']:
            print(f"      • {signal}")

        print(f"\n   ✅ ANÁLISIS COMPLETO EXITOSO")
    else:
        print(f"\n   ❌ No se pudieron obtener datos de ASST")

except Exception as e:
    print(f"\n   ❌ Error en análisis: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("RESUMEN")
print("="*70)
print(f"\n✅ Modelo ML: {'FUNCIONANDO' if advisor.ml_predictor.is_trained else 'NO FUNCIONA'}")
print(f"✅ Reddit: {'USANDO DATOS LOCALES' if reddit_asst['source'] == 'local' else 'USANDO DEFAULTS'}")
print(f"✅ Short Rates: {'USANDO DATOS LOCALES' if short_asst['source'] == 'local' else 'USANDO DEFAULTS'}")
print("\n" + "="*70)
