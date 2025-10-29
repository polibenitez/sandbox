#!/usr/bin/env python3
"""
Análisis retrospectivo de ASST el 26 de octubre 2025
Para determinar si la nueva lógica de día 3 lo habría detectado
"""

import sys
import os
import yfinance as yf
from datetime import datetime, timedelta

# Agregar utils al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from penny_stock_advisor_v5 import PennyStockAdvisorV5
from integration_v5_trading_manager import MarketContextAnalyzerV5

def analyze_asst_october26():
    """Analiza ASST con datos del 26 de octubre (retrospectivo)"""
    print("\n" + "="*70)
    print("ANÁLISIS RETROSPECTIVO: ASST - 26 OCTUBRE 2025")
    print("="*70)
    print("\n🔍 Obteniendo datos históricos de ASST hasta el 26 de octubre...")

    try:
        # Obtener datos hasta el 26 de octubre
        ticker = yf.Ticker("ASST")

        # Datos desde principios de octubre hasta el 26
        hist = ticker.history(start="2025-10-01", end="2025-10-27")

        if len(hist) == 0:
            print("❌ No se pudieron obtener datos históricos")
            return

        print(f"✅ Datos obtenidos: {len(hist)} días")
        print(f"\n📊 ÚLTIMOS 10 DÍAS DE TRADING:")
        print("-" * 70)
        print(f"{'Fecha':<12} {'Cierre':>10} {'Volumen':>15} {'Alto':>10} {'Bajo':>10}")
        print("-" * 70)

        for idx in range(max(0, len(hist) - 10), len(hist)):
            date = hist.index[idx].strftime('%Y-%m-%d')
            close = hist['Close'].iloc[idx]
            volume = hist['Volume'].iloc[idx]
            high = hist['High'].iloc[idx]
            low = hist['Low'].iloc[idx]
            print(f"{date:<12} ${close:>9.3f} {volume:>15,} ${high:>9.3f} ${low:>9.3f}")

        # Analizar datos del 25 de octubre (viernes antes del fin de semana)
        # El 26 fue sábado, así que los últimos datos disponibles serían del 25
        print(f"\n" + "="*70)
        print("DATOS DEL 25 DE OCTUBRE 2025 (VIERNES)")
        print("="*70)

        # Buscar el 25 de octubre
        oct25_data = None
        oct25_idx = None

        for idx in range(len(hist)):
            if hist.index[idx].strftime('%Y-%m-%d') == '2025-10-25':
                oct25_data = hist.iloc[idx]
                oct25_idx = idx
                break

        if oct25_data is None:
            print("⚠️ No hay datos del 25 de octubre (mercado cerrado o data no disponible)")
            # Usar el último día disponible
            oct25_idx = len(hist) - 1
            oct25_data = hist.iloc[oct25_idx]
            oct25_date = hist.index[oct25_idx].strftime('%Y-%m-%d')
            print(f"📅 Usando último día disponible: {oct25_date}")

        print(f"\n📈 Precio cierre: ${oct25_data['Close']:.3f}")
        print(f"📊 Volumen: {oct25_data['Volume']:,}")
        print(f"📉 Rango: ${oct25_data['Low']:.3f} - ${oct25_data['High']:.3f}")

        # Calcular métricas clave
        print(f"\n" + "="*70)
        print("ANÁLISIS TÉCNICO AL 25 DE OCTUBRE")
        print("="*70)

        # Volumen promedio 20 días
        if oct25_idx >= 20:
            avg_volume_20d = hist['Volume'].iloc[oct25_idx-19:oct25_idx+1].mean()
            volume_ratio = oct25_data['Volume'] / avg_volume_20d
            print(f"\n📊 VOLUMEN:")
            print(f"   • Volumen 25 oct: {oct25_data['Volume']:,}")
            print(f"   • Promedio 20d: {avg_volume_20d:,.0f}")
            print(f"   • Ratio: {volume_ratio:.2f}x")
        else:
            avg_volume_20d = hist['Volume'].mean()
            volume_ratio = oct25_data['Volume'] / avg_volume_20d
            print(f"   • Datos insuficientes para 20d, usando promedio total")

        # Calcular días consecutivos de volumen alto
        consecutive_high_volume = 0
        spike_threshold = 2.5

        for i in range(oct25_idx, -1, -1):
            vol_ratio = hist['Volume'].iloc[i] / avg_volume_20d if avg_volume_20d > 0 else 0
            if vol_ratio >= spike_threshold:
                consecutive_high_volume += 1
            else:
                break

        print(f"\n⏰ DÍAS DE EXPLOSIÓN:")
        print(f"   • Días consecutivos vol > {spike_threshold}x: {consecutive_high_volume}")
        print(f"   • Status: ", end="")

        if consecutive_high_volume == 1:
            print("DÍA 1 - PERFECTO")
        elif consecutive_high_volume == 2:
            print("DÍA 2 - BUENO")
        elif consecutive_high_volume == 3:
            print("DÍA 3 - TARDÍO (evaluar condiciones)")
        else:
            print(f"DÍA {consecutive_high_volume} - MUY TARDÍO")

        # Cambio de precio en 3 días
        if oct25_idx >= 3:
            price_3d_ago = hist['Close'].iloc[oct25_idx - 3]
            price_current = oct25_data['Close']
            price_change_3d = ((price_current - price_3d_ago) / price_3d_ago * 100)

            print(f"\n💰 PRECIO:")
            print(f"   • Hace 3 días: ${price_3d_ago:.3f}")
            print(f"   • 25 octubre: ${price_current:.3f}")
            print(f"   • Cambio 3d: {price_change_3d:+.1f}%")

        # Compresión previa (5 días antes del spike)
        if oct25_idx >= 5:
            # Ver si hubo compresión en días previos
            lookback = 5
            if consecutive_high_volume > 0:
                # Ver compresión antes del primer día de spike
                compression_start = oct25_idx - consecutive_high_volume - lookback
                compression_end = oct25_idx - consecutive_high_volume

                if compression_start >= 0:
                    compression_period = hist['Close'].iloc[compression_start:compression_end+1]
                    price_range_pct = ((compression_period.max() - compression_period.min()) / compression_period.min() * 100)

                    print(f"\n🔄 COMPRESIÓN PREVIA:")
                    print(f"   • Período analizado: {lookback} días antes del spike")
                    print(f"   • Rango de precio: {price_range_pct:.1f}%")
                    print(f"   • ¿Comprimido? (<8%): {'✅ SÍ' if price_range_pct <= 8 else '❌ NO'}")

        # RSI simple (14 períodos)
        if oct25_idx >= 14:
            closes = hist['Close'].iloc[:oct25_idx+1]
            deltas = closes.diff()
            gain = deltas.where(deltas > 0, 0).rolling(window=14).mean()
            loss = (-deltas.where(deltas < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            rsi_value = rsi.iloc[-1]

            print(f"\n📊 RSI (14):")
            print(f"   • Valor: {rsi_value:.0f}")
            print(f"   • Status: ", end="")
            if rsi_value < 30:
                print("Sobrevendido")
            elif rsi_value > 70:
                print("Sobrecomprado")
            else:
                print("Neutral")

        # Evaluación con nueva lógica día 3
        print(f"\n" + "="*70)
        print("EVALUACIÓN CON NUEVA LÓGICA DÍA 3")
        print("="*70)

        if consecutive_high_volume == 3:
            print(f"\n✅ Día 3 detectado - Evaluando condiciones estrictas:")
            print(f"\n   1️⃣ Volumen ≥ 3.5x promedio:")
            if volume_ratio >= 3.5:
                print(f"      ✅ CUMPLE: {volume_ratio:.2f}x ≥ 3.5x")
            else:
                print(f"      ❌ NO CUMPLE: {volume_ratio:.2f}x < 3.5x")

            print(f"\n   2️⃣ RSI ≤ 75:")
            if oct25_idx >= 14:
                if rsi_value <= 75:
                    print(f"      ✅ CUMPLE: {rsi_value:.0f} ≤ 75")
                else:
                    print(f"      ❌ NO CUMPLE: {rsi_value:.0f} > 75")
            else:
                print(f"      ⚠️ Datos insuficientes para RSI")

            print(f"\n   3️⃣ Cambio precio ≤ 60% en 3d:")
            if oct25_idx >= 3:
                if price_change_3d <= 60:
                    print(f"      ✅ CUMPLE: {price_change_3d:.1f}% ≤ 60%")
                else:
                    print(f"      ❌ NO CUMPLE: {price_change_3d:.1f}% > 60%")

            # Conclusión
            print(f"\n" + "-"*70)
            conditions_met = 0
            if volume_ratio >= 3.5:
                conditions_met += 1
            if oct25_idx >= 14 and rsi_value <= 75:
                conditions_met += 1
            if oct25_idx >= 3 and price_change_3d <= 60:
                conditions_met += 1

            if conditions_met >= 3:
                print(f"✅ CALIFICA PARA ENTRADA TARDÍA DÍA 3")
                print(f"   • Todas las condiciones cumplidas ({conditions_met}/3)")
                print(f"   • Penalización: -15 pts (vs -30 pts)")
                print(f"   • Posición reducida: 2% (vs 3%)")
            else:
                print(f"❌ NO CALIFICA PARA ENTRADA TARDÍA")
                print(f"   • Condiciones cumplidas: {conditions_met}/3")
                print(f"   • Penalización: -30 pts")
        elif consecutive_high_volume < 3:
            print(f"\n✅ Día {consecutive_high_volume} - NO necesita evaluación especial")
            print(f"   • Penalización: 0 pts")
        else:
            print(f"\n❌ Día {consecutive_high_volume} - MUY TARDE")
            print(f"   • Penalización: -30 pts")

        print(f"\n" + "="*70)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_asst_october26()
