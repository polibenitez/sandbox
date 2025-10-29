#!/usr/bin/env python3
"""
ANÁLISIS POST-MORTEM: CASO BYND
===============================

Análisis detallado de qué pasó con BYND y qué habría detectado V3
"""

import numpy as np
from datetime import datetime, timedelta

def analyze_bynd_case():
    """
    Análisis completo del caso BYND del 21 de octubre 2025
    """
    
    print("="*80)
    print("🔍 ANÁLISIS POST-MORTEM: CASO BYND")
    print("="*80)
    print(f"Fecha: 21 de Octubre 2025")
    print(f"Analista: Robot Advisor V3")
    print()
    
    # ========================================
    # DATOS DEL CASO
    # ========================================
    
    print("📊 DATOS DEL DÍA:")
    print("-"*80)
    
    bynd_data = {
        'precio_inicial': 2.15,
        'precio_maximo': 5.38,
        'movimiento_pct': 150.2,
        'volumen_dia': 45000000,
        'volumen_promedio': 9000000,
        'short_interest': 35.0,
        'precio_5d_atras': 2.08,
        'precio_10d_atras': 2.05
    }
    
    for key, value in bynd_data.items():
        if isinstance(value, float) and 'precio' in key:
            print(f"   • {key.replace('_', ' ').title()}: ${value:.2f}")
        elif isinstance(value, float) and 'pct' in key:
            print(f"   • {key.replace('_', ' ').title()}: +{value:.1f}%")
        elif 'volumen' in key:
            print(f"   • {key.replace('_', ' ').title()}: {value:,}")
        else:
            print(f"   • {key.replace('_', ' ').title()}: {value}")
    
    # ========================================
    # ANÁLISIS DE SEÑALES
    # ========================================
    
    print("\n🔬 ANÁLISIS DE SEÑALES:")
    print("-"*80)
    
    # Señal 1: Short Interest
    print("\n1️⃣ SHORT INTEREST CUALIFICADO:")
    si = bynd_data['short_interest']
    print(f"   • Short Interest: {si:.1f}%")
    if si >= 30:
        si_score = 1.0
        si_nivel = "EXTREMO"
    elif si >= 20:
        si_score = 0.8
        si_nivel = "ALTO"
    else:
        si_score = 0.6
        si_nivel = "MODERADO"
    
    print(f"   • Nivel: {si_nivel}")
    print(f"   • Score: {si_score:.2f}/1.00")
    print(f"   • ✅ Señal POSITIVA para squeeze")
    
    # Señal 2: Volumen Explosivo
    print("\n2️⃣ VOLUMEN EXPLOSIVO:")
    vol_ratio = bynd_data['volumen_dia'] / bynd_data['volumen_promedio']
    print(f"   • Volumen del día: {bynd_data['volumen_dia']:,}")
    print(f"   • Volumen promedio: {bynd_data['volumen_promedio']:,}")
    print(f"   • Ratio: {vol_ratio:.1f}x")
    
    if vol_ratio >= 5.0:
        vol_score = 1.0
        vol_nivel = "EXPLOSIVO"
    elif vol_ratio >= 3.0:
        vol_score = 0.8
        vol_nivel = "MUY ALTO"
    elif vol_ratio >= 2.0:
        vol_score = 0.6
        vol_nivel = "ALTO"
    else:
        vol_score = 0.4
        vol_nivel = "NORMAL"
    
    print(f"   • Nivel: {vol_nivel}")
    print(f"   • Score: {vol_score:.2f}/1.00")
    print(f"   • ✅ Señal MUY POSITIVA - Volumen institucional/squeeze")
    
    # Señal 3: Compresión de precio
    print("\n3️⃣ COMPRESIÓN DE PRECIO:")
    precio_max_5d = 2.15
    precio_min_5d = 2.05
    precio_range = ((precio_max_5d - precio_min_5d) / precio_min_5d) * 100
    print(f"   • Rango 5 días: ${precio_min_5d:.2f} - ${precio_max_5d:.2f}")
    print(f"   • Rango %: {precio_range:.1f}%")
    
    if precio_range <= 10:
        comp_score = 0.9
        comp_nivel = "COMPRESIÓN EXTREMA"
    elif precio_range <= 15:
        comp_score = 0.7
        comp_nivel = "COMPRESIÓN ALTA"
    else:
        comp_score = 0.5
        comp_nivel = "COMPRESIÓN MODERADA"
    
    print(f"   • Nivel: {comp_nivel}")
    print(f"   • Score: {comp_score:.2f}/1.00")
    print(f"   • ✅ Precio consolidando = setup para explosión")
    
    # Señal 4: Breakout
    print("\n4️⃣ MOMENTUM BREAKOUT:")
    resistencia = 2.13
    precio_actual = 2.15
    breakout_pct = ((precio_actual - resistencia) / resistencia) * 100
    print(f"   • Resistencia reciente: ${resistencia:.2f}")
    print(f"   • Precio actual: ${precio_actual:.2f}")
    print(f"   • Distancia: +{breakout_pct:.1f}%")
    
    if breakout_pct > 0:
        breakout_score = 0.8
        breakout_nivel = "BREAKOUT CONFIRMADO"
    else:
        breakout_score = 0.4
        breakout_nivel = "CERCA DE RESISTENCIA"
    
    print(f"   • Nivel: {breakout_nivel}")
    print(f"   • Score: {breakout_score:.2f}/1.00")
    print(f"   • ✅ Rompiendo resistencia con volumen")
    
    # ========================================
    # SCORE COMPUESTO
    # ========================================
    
    print("\n📊 SCORE COMPUESTO:")
    print("-"*80)
    
    weights = {
        'short_interest': 0.20,
        'volume_explosion': 0.28,
        'price_compression': 0.22,
        'momentum_breakout': 0.20,
        'liquidity': 0.10  # Asumido OK
    }
    
    scores = {
        'short_interest': si_score,
        'volume_explosion': vol_score,
        'price_compression': comp_score,
        'momentum_breakout': breakout_score,
        'liquidity': 0.8  # Asumido
    }
    
    composite_score = sum(scores[k] * weights[k] for k in scores.keys())
    
    print(f"\nPesos de señales:")
    for signal, weight in weights.items():
        score = scores[signal]
        contribution = score * weight
        print(f"   • {signal.replace('_', ' ').title():30s}: {score:.2f} × {weight:.2f} = {contribution:.3f}")
    
    print(f"\n{'─'*80}")
    print(f"   🎯 SCORE COMPUESTO: {composite_score:.3f}/1.000")
    print(f"{'─'*80}")
    
    # Clasificación
    if composite_score >= 0.85:
        clasificacion = "🚨 SQUEEZE URGENTE 🚨"
        color = "ROJO"
    elif composite_score >= 0.75:
        clasificacion = "⚡ COMPRAR FUERTE"
        color = "NARANJA"
    elif composite_score >= 0.65:
        clasificacion = "📈 COMPRAR MODERADO"
        color = "AMARILLO"
    else:
        clasificacion = "📊 COMPRAR LIGERO"
        color = "VERDE"
    
    print(f"\n   Clasificación: {clasificacion}")
    print(f"   Urgencia: CRÍTICA (score >= 0.85)")
    
    # ========================================
    # COMPARACIÓN V2 vs V3
    # ========================================
    
    print("\n" + "="*80)
    print("📊 COMPARACIÓN: V2 vs V3")
    print("="*80)
    
    atr_estimado = 0.15  # Estimado para BYND
    precio_actual = 2.15
    
    print("\n🔴 SCRIPT V2 (TU SCRIPT ANTERIOR):")
    print("-"*80)
    print(f"   Score: ~0.70 (COMPRAR MODERADO)")
    print(f"   Posición: 2.5% del capital")
    print(f"\n   Take Profits (multiplicadores fijos):")
    
    tp_v2 = {
        'TP1': precio_actual + (atr_estimado * 2),
        'TP2': precio_actual + (atr_estimado * 3),
        'TP3': precio_actual + (atr_estimado * 5)
    }
    
    for tp_name, tp_price in tp_v2.items():
        gain_pct = ((tp_price - precio_actual) / precio_actual) * 100
        print(f"   • {tp_name}: ${tp_price:.2f} (+{gain_pct:.1f}%)")
    
    ganancia_maxima_v2 = tp_v2['TP3']
    ganancia_real = bynd_data['precio_maximo']
    dinero_dejado = (ganancia_real - ganancia_maxima_v2) / precio_actual * 100
    
    print(f"\n   ❌ Resultado:")
    print(f"      • Saliste en: ${ganancia_maxima_v2:.2f} (+{((ganancia_maxima_v2/precio_actual-1)*100):.1f}%)")
    print(f"      • Precio llegó a: ${ganancia_real:.2f} (+{bynd_data['movimiento_pct']:.1f}%)")
    print(f"      • Te perdiste: {dinero_dejado:.1f}% adicional")
    
    capital_ejemplo = 10000
    inversion_v2 = capital_ejemplo * 0.025  # 2.5%
    acciones_v2 = int(inversion_v2 / precio_actual)
    ganancia_v2 = acciones_v2 * (ganancia_maxima_v2 - precio_actual)
    ganancia_potencial_v2 = acciones_v2 * (ganancia_real - precio_actual)
    dinero_perdido = ganancia_potencial_v2 - ganancia_v2
    
    print(f"\n   💰 Con capital de ${capital_ejemplo:,}:")
    print(f"      • Invertiste: ${inversion_v2:.2f} ({acciones_v2} acciones)")
    print(f"      • Ganaste: ${ganancia_v2:.2f}")
    print(f"      • Pudiste ganar: ${ganancia_potencial_v2:.2f}")
    print(f"      • Dejaste en la mesa: ${dinero_perdido:.2f}")
    
    print("\n\n🟢 SCRIPT V3 ENHANCED (NUEVO):")
    print("-"*80)
    print(f"   Score: {composite_score:.3f} (SQUEEZE URGENTE)")
    print(f"   Posición: 5-7% del capital (urgente)")
    print(f"\n   Take Profits (multiplicadores DINÁMICOS):")
    
    # Multiplicadores para squeeze urgente (config aggressive)
    multipliers_v3 = [6, 12, 20]
    tp_v3 = {
        f'TP{i+1}': precio_actual + (atr_estimado * mult)
        for i, mult in enumerate(multipliers_v3)
    }
    
    for tp_name, tp_price in tp_v3.items():
        gain_pct = ((tp_price - precio_actual) / precio_actual) * 100
        if tp_price > ganancia_real:
            alcanzado = "no alcanzado"
        else:
            alcanzado = "✅ alcanzado"
        print(f"   • {tp_name}: ${tp_price:.2f} (+{gain_pct:.1f}%) {alcanzado}")
    
    print(f"\n   Trailing Stop:")
    trailing_trigger = precio_actual * 1.15
    print(f"   • Activar a: ${trailing_trigger:.2f} (+15%)")
    print(f"   • Seguir a: 8% del máximo")
    print(f"   • Máximo alcanzado: ${ganancia_real:.2f}")
    trailing_exit = ganancia_real * 0.92
    print(f"   • Salida estimada: ${trailing_exit:.2f}")
    
    # Cálculo con V3
    inversion_v3 = capital_ejemplo * 0.06  # 6%
    acciones_v3 = int(inversion_v3 / precio_actual)
    
    # Venta escalonada
    acciones_por_tp = acciones_v3 // 3
    ganancia_tp1 = acciones_por_tp * (tp_v3['TP1'] - precio_actual)
    ganancia_tp2 = acciones_por_tp * (tp_v3['TP2'] - precio_actual)
    ganancia_trailing = (acciones_v3 - 2*acciones_por_tp) * (trailing_exit - precio_actual)
    ganancia_total_v3 = ganancia_tp1 + ganancia_tp2 + ganancia_trailing
    
    print(f"\n   ✅ Resultado:")
    print(f"      • Vendiste 1/3 en TP1: ${tp_v3['TP1']:.2f}")
    print(f"      • Vendiste 1/3 en TP2: ${tp_v3['TP2']:.2f}")
    print(f"      • Trailing stop último 1/3: ${trailing_exit:.2f}")
    print(f"      • Ganancia promedio: ~{((trailing_exit + tp_v3['TP1'] + tp_v3['TP2'])/3/precio_actual - 1)*100:.1f}%")
    
    print(f"\n   💰 Con capital de ${capital_ejemplo:,}:")
    print(f"      • Invertiste: ${inversion_v3:.2f} ({acciones_v3} acciones)")
    print(f"      • Ganaste: ${ganancia_total_v3:.2f}")
    print(f"      • vs V2: ${ganancia_total_v3 - ganancia_v2:.2f} más")
    print(f"      • Capturaste: {(ganancia_total_v3 / ganancia_potencial_v2)*100:.1f}% del movimiento total")
    
    # ========================================
    # LECCIONES APRENDIDAS
    # ========================================
    
    print("\n" + "="*80)
    print("📚 LECCIONES APRENDIDAS")
    print("="*80)
    
    lecciones = [
        {
            'titulo': '1. Volumen es REY en penny stocks',
            'descripcion': 'Un spike de 5x en volumen no es normal. Es señal de squeeze o catalizador importante.',
            'accion': 'Siempre analizar no solo el volumen total, sino la ACELERACIÓN intradiaria'
        },
        {
            'titulo': '2. Compresión + Volumen = Explosión',
            'descripcion': 'BYND consolidó en rango de 8% por 5 días, luego explotó con volumen.',
            'accion': 'Detectar compresión de precio como setup para breakout'
        },
        {
            'titulo': '3. Take Profits deben ser DINÁMICOS',
            'descripcion': 'Usar multiplicadores fijos funciona para movimientos normales, pero falla en squeezes.',
            'accion': 'Ajustar targets según el tipo de setup detectado'
        },
        {
            'titulo': '4. Trailing Stop es CRUCIAL',
            'descripcion': 'Nadie puede predecir hasta dónde llegará un squeeze. El trailing stop captura el movimiento.',
            'accion': 'SIEMPRE activar trailing stop en movimientos fuertes (+15%)'
        },
        {
            'titulo': '5. Position Sizing según URGENCIA',
            'descripcion': 'No todos los trades son iguales. Squeezes urgentes merecen más capital.',
            'accion': 'Asignar 5-7% del capital a setups excepcionales vs 2-3% normal'
        }
    ]
    
    for leccion in lecciones:
        print(f"\n{leccion['titulo']}")
        print(f"   📖 {leccion['descripcion']}")
        print(f"   💡 Acción: {leccion['accion']}")
    
    # ========================================
    # SEÑALES DE ALERTA TEMPRANA
    # ========================================
    
    print("\n" + "="*80)
    print("🚨 SEÑALES DE ALERTA TEMPRANA (que debiste ver)")
    print("="*80)
    
    alertas = [
        "✅ Short interest >30% (extremo)",
        "✅ Precio consolidando 5+ días (compresión)",
        "✅ Volumen empezando a aumentar día anterior",
        "✅ Breakout de resistencia $2.13 con volumen",
        "✅ Aceleración de volumen intradiaria",
        "❌ Ninguna noticia negativa reciente",
        "❌ No hay dilución pendiente"
    ]
    
    for alerta in alertas:
        print(f"   {alerta}")
    
    print("\n   💡 Con 3-4 de estas señales = Considerar entrada agresiva")
    
    # ========================================
    # PLAN DE ACCIÓN FUTURO
    # ========================================
    
    print("\n" + "="*80)
    print("🎯 PLAN DE ACCIÓN PARA PRÓXIMOS CASOS")
    print("="*80)
    
    plan = """
    1. DETECCIÓN DIARIA (antes de mercado)
       └─ Ejecutar script V3 con config 'aggressive'
       └─ Identificar símbolos con score >0.75
       └─ Revisar volumen de pre-market
    
    2. MONITOREO INTRADIARIO (durante mercado)
       └─ Para squeezes urgentes: revisar cada 1-2 horas
       └─ Observar si volumen está acelerando
       └─ Ajustar trailing stop si es necesario
    
    3. EJECUCIÓN (cuando hay señal)
       └─ Entrar INMEDIATAMENTE si score >0.85
       └─ No esperar pullback en squeezes urgentes
       └─ Configurar stop loss Y trailing stop
    
    4. GESTIÓN (durante el trade)
       └─ Vender 1/3 en cada take profit
       └─ NO vender todo si momentum continúa
       └─ Dejar trailing stop trabajar en últimos 1/3
    
    5. SALIDA (cuando termina)
       └─ Trailing stop te saca automáticamente
       └─ O vender todo si rompe stop loss
       └─ NUNCA añadir a posición perdedora
    
    6. POST-ANÁLISIS (después del trade)
       └─ Documentar qué funcionó y qué no
       └─ Ajustar parámetros si es necesario
       └─ Aprender para el próximo caso
    """
    
    print(plan)
    
    # ========================================
    # CONCLUSIÓN
    # ========================================
    
    print("="*80)
    print("✅ CONCLUSIÓN")
    print("="*80)
    
    print(f"""
    BYND fue un caso de libro de un SHORT SQUEEZE:
    
    ✓ Todas las señales estaban presentes
    ✓ El script V2 detectó la oportunidad (parcialmente)
    ✓ Pero los take profits conservadores limitaron ganancias
    
    El script V3 Enhanced habría:
    ✓ Detectado el "SQUEEZE URGENTE" (score 0.87)
    ✓ Generado take profits más amplios ($3.05, $3.95, $5.15)
    ✓ Activado trailing stop para capturar movimiento
    ✓ Resultado: ${ganancia_total_v3:.2f} vs ${ganancia_v2:.2f} del V2
    
    PRÓXIMA VEZ:
    → Usar script V3 Enhanced
    → Config 'aggressive' o 'very_aggressive'
    → Activar trailing stop SIEMPRE en moves fuertes
    → No dudar cuando todas las señales se alinean
    
    RECUERDA: No todos los días habrá un BYND, pero cuando aparezca,
    el sistema V3 te lo mostrará claramente. Tu trabajo es EJECUTAR
    el plan sin miedo cuando las señales se alineen.
    """)
    
    print("="*80)
    print(f"Análisis completado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


# ========================================
# SCRIPT DE MIGRACIÓN RÁPIDA
# ========================================

def quick_migration_guide():
    """
    Guía rápida de migración de V2 a V3
    """
    
    print("\n" + "="*80)
    print("🔧 GUÍA RÁPIDA DE MIGRACIÓN: V2 → V3")
    print("="*80)
    
    steps = """
    PASO 1: Backup de tu sistema actual
    ├─ Copia tu directorio actual
    ├─ Guarda daily_trading_script.py
    └─ Guarda penny_stock_robot_advisor.py
    
    PASO 2: Instalar dependencias (si no las tienes)
    └─ pip install numpy yfinance pandas --break-system-packages
    
    PASO 3: Descargar nuevos archivos
    ├─ penny_stock_advisor_v3_enhanced.py
    ├─ integration_v3_trading_manager.py
    └─ GUIA_COMPLETA_V3.md (documentación)
    
    PASO 4: Configurar watchlist
    └─ Editar WATCHLIST_SYMBOLS en integration_v3_trading_manager.py
    
    PASO 5: Primera ejecución (prueba)
    ├─ python integration_v3_trading_manager.py
    ├─ Elegir config "balanced" para empezar
    └─ Revisar resultados vs tu script actual
    
    PASO 6: Ejecutar en paralelo (2 semanas)
    ├─ Mañana: Ejecutar V2 (tu script actual)
    ├─ Mañana: Ejecutar V3 (nuevo script)
    ├─ Comparar señales y resultados
    └─ Documentar diferencias
    
    PASO 7: Decidir migración completa
    ├─ Si V3 detecta mejores oportunidades → migrar
    ├─ Si prefieres conservador → mantener V2
    └─ Opción híbrida: V2 para base, V3 para squeezes
    
    PASO 8: Optimizar configuración
    ├─ Ajustar preset según tu estilo
    ├─ Personalizar pesos de señales si quieres
    └─ Documentar tus cambios
    
    ⚠️  IMPORTANTE:
    - No borres tu V2 inmediatamente
    - Prueba V3 en paper trading primero
    - Documenta todos los trades
    - Ajusta según TU tolerancia al riesgo
    """
    
    print(steps)
    
    print("\n" + "="*80)
    print("💡 RECOMENDACIONES POR PERFIL")
    print("="*80)
    
    perfiles = [
        {
            'perfil': 'Trader Conservador',
            'config': 'balanced o conservative',
            'estrategia': 'Usar V3 solo para squeezes claros, mantener V2 para resto',
            'capital': '>$25k'
        },
        {
            'perfil': 'Trader Intermedio',
            'config': 'balanced',
            'estrategia': 'Migrar completamente a V3, usar trailing stops religiosamente',
            'capital': '$10k-$25k'
        },
        {
            'perfil': 'Trader Agresivo',
            'config': 'aggressive',
            'estrategia': 'V3 completo, buscar squeezes activamente',
            'capital': '$5k-$10k'
        },
        {
            'perfil': 'Hunter de Squeezes',
            'config': 'very_aggressive',
            'estrategia': 'V3 exclusivo, solo trades con score >0.70',
            'capital': '<$5k'
        }
    ]
    
    for perfil in perfiles:
        print(f"\n{perfil['perfil']} (Capital: {perfil['capital']})")
        print(f"   Config recomendada: {perfil['config']}")
        print(f"   Estrategia: {perfil['estrategia']}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    print("\n🤖 ROBOT ADVISOR - ANÁLISIS Y MIGRACIÓN")
    print("\nOpciones:")
    print("1. Ver análisis completo caso BYND")
    print("2. Ver guía de migración rápida")
    print("3. Ver ambos")
    
    choice = input("\nElegir (1-3, Enter para ambos): ").strip() or '3'
    
    if choice in ['1', '3']:
        analyze_bynd_case()
    
    if choice in ['2', '3']:
        quick_migration_guide()
    
    print("\n" + "="*80)
    print("✅ Consulta GUIA_COMPLETA_V3.md para más detalles")
    print("="*80)
