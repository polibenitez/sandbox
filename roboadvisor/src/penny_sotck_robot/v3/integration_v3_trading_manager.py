#!/usr/bin/env python3
"""
INTEGRACIÓN V3 ENHANCED CON SISTEMA EXISTENTE
============================================

Este script actualiza tu sistema para usar las nuevas capacidades V3:
- Análisis de volumen intradiario
- Detección de squeezes urgentes
- Take profits dinámicos

Se integra con tu daily_trading_script.py existente
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import json
from penny_stock_advisor_v3_enhanced import PennyStockAdvisorV3Enhanced

# Tu watchlist actual
WATCHLIST_SYMBOLS = [
    "OPEN", "CHPT", "LCFY", "SIRI", "XAIR",
    "HTOO", "CTMX", "CLOV", "ALBT", "ADIL",
    "BYND", "AKBA", "OPAD", "AIRE", "YYAI",
    "RANI", "WOK", "AREB", "BENF", "CJET", "SBEV", "ISRG", "VTYX"
]


class VolumeAnalyzer:
    """
    Analizador especializado de volumen intradiario
    
    Analogía: Como un sismógrafo que no solo detecta el terremoto,
    sino que predice su llegada midiendo las ondas precursoras.
    """
    
    def __init__(self):
        self.history = {}
    
    def get_intraday_volume_profile(self, symbol, interval="5m", period="1d"):
        """
        Obtiene perfil de volumen intradiario para detectar aceleraciones
        
        Returns:
            dict con análisis de volumen intradiario
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Obtener datos intradiarios
            intraday = ticker.history(interval=interval, period=period)
            
            if len(intraday) == 0:
                return None
            
            volumes = intraday['Volume'].values
            times = intraday.index
            
            # Analizar aceleración
            if len(volumes) >= 10:
                # Volumen primera mitad vs segunda mitad del día
                mid_point = len(volumes) // 2
                first_half_avg = np.mean(volumes[:mid_point])
                second_half_avg = np.mean(volumes[mid_point:])
                
                acceleration = second_half_avg / first_half_avg if first_half_avg > 0 else 1.0
                
                # Detectar spikes de volumen
                volume_mean = np.mean(volumes)
                volume_std = np.std(volumes)
                spikes = []
                
                for i, vol in enumerate(volumes):
                    if vol > volume_mean + 2*volume_std:
                        spikes.append({
                            'time': times[i],
                            'volume': int(vol),
                            'multiplier': vol / volume_mean
                        })
                
                # Momentum actual (últimos 30 min vs promedio)
                recent_volume = np.mean(volumes[-6:]) if len(volumes) >= 6 else volumes[-1]
                momentum = recent_volume / volume_mean if volume_mean > 0 else 1.0
                
                return {
                    'symbol': symbol,
                    'total_volume': int(np.sum(volumes)),
                    'acceleration': acceleration,
                    'momentum_current': momentum,
                    'spikes_detected': len(spikes),
                    'spike_details': spikes[:5],  # Top 5
                    'is_accelerating': acceleration > 1.5,
                    'is_explosive': acceleration > 2.0 and momentum > 2.0,
                    'analysis_time': datetime.now(),
                    'data_points': len(volumes)
                }
            
            return None
            
        except Exception as e:
            print(f"⚠️  Error analizando volumen intradiario de {symbol}: {e}")
            return None
    
    def analyze_volume_patterns(self, symbol, days=5):
        """
        Analiza patrones de volumen en múltiples días
        Detecta si está en fase de acumulación o distribución
        """
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=f"{days}d", interval="1h")
            
            if len(hist) < 10:
                return None
            
            volumes = hist['Volume'].values
            
            # Calcular tendencia de volumen
            x = np.arange(len(volumes))
            z = np.polyfit(x, volumes, 1)
            trend_slope = z[0]
            
            # Detectar patrón
            if trend_slope > 0:
                pattern = "ACUMULACIÓN" if trend_slope > np.mean(volumes) * 0.1 else "ALCISTA"
                score = min(1.0, trend_slope / (np.mean(volumes) * 0.2))
            else:
                pattern = "DISTRIBUCIÓN"
                score = 0.3
            
            return {
                'pattern': pattern,
                'trend_slope': trend_slope,
                'score': score,
                'avg_volume': int(np.mean(volumes)),
                'max_volume': int(np.max(volumes))
            }
            
        except Exception as e:
            return None


class TradingManagerV3:
    """
    Gestor principal actualizado para usar Robot Advisor V3
    """
    
    def __init__(self, config_preset="balanced"):
        """
        config_preset opciones: 
        - "conservative": Menos señales, más precisión
        - "balanced": Equilibrio óptimo (RECOMENDADO)
        - "aggressive": Más señales, detecta squeezes early
        - "very_aggressive": Máxima sensibilidad
        """
        self.robot = PennyStockAdvisorV3Enhanced(config_preset=config_preset)
        self.volume_analyzer = VolumeAnalyzer()
        self.watchlist = WATCHLIST_SYMBOLS
        self.robot.update_watchlist(self.watchlist)
        
        print(f"\n🤖 Trading Manager V3 inicializado")
        print(f"📊 Configuración: {config_preset.upper()}")
        print(f"🎯 Watchlist: {len(self.watchlist)} símbolos")
    
    def run_full_analysis(self):
        """
        Ejecuta análisis completo con volumen intradiario
        """
        print(f"\n{'='*70}")
        print(f"🚀 ANÁLISIS COMPLETO - VERSIÓN V3 ENHANCED")
        print(f"{'='*70}")
        print(f"📅 {datetime.now().strftime('%A, %d de %B %Y - %H:%M')}")
        print(f"📊 Analizando {len(self.watchlist)} símbolos")
        print()
        
        results = []
        volume_insights = []
        
        for symbol in self.watchlist:
            print(f"\n🔍 Analizando {symbol}...")
            
            # 1. Obtener datos de mercado
            market_data, historical_data = self.robot.get_enhanced_market_data(symbol)
            
            if market_data is None:
                print(f"  ⚠️  Sin datos disponibles")
                continue
            
            # 2. Análisis de volumen intradiario (NUEVO)
            intraday_volume = self.volume_analyzer.get_intraday_volume_profile(symbol)
            
            if intraday_volume:
                print(f"  📊 Volumen intradiario:")
                print(f"     • Aceleración: {intraday_volume['acceleration']:.2f}x")
                print(f"     • Momentum actual: {intraday_volume['momentum_current']:.2f}x")
                print(f"     • Spikes detectados: {intraday_volume['spikes_detected']}")
                
                if intraday_volume['is_explosive']:
                    print(f"     🚨 *** VOLUMEN EXPLOSIVO DETECTADO ***")
                
                volume_insights.append(intraday_volume)
                
                # Actualizar market_data con insights de volumen intradiario
                if intraday_volume['is_explosive']:
                    market_data['volume'] = int(market_data['volume'] * intraday_volume['momentum_current'])
            
            # 3. Análisis con Robot V3
            analysis = self.robot.analyze_symbol_enhanced(symbol, market_data, historical_data)
            results.append(analysis)
        
        # Ordenar por score
        results.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # Generar reporte
        print(f"\n{'='*70}")
        print("📊 RESUMEN DE ANÁLISIS")
        print(f"{'='*70}")
        print(f"   • Símbolos analizados: {len(results)}")
        print(f"   • Con volumen explosivo: {sum(1 for v in volume_insights if v.get('is_explosive', False))}")
        
        buy_signals = self.robot.generate_trading_report_v3(results)
        
        # Guardar resultados
        self.save_results(results, volume_insights, buy_signals)
        
        return results, buy_signals
    
    def save_results(self, results, volume_insights, buy_signals):
        """Guarda resultados del análisis"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        filename = f"trading_results_v3_{timestamp}.json"
        
        output = {
            'timestamp': datetime.now().isoformat(),
            'config_preset': self.robot.config_preset,
            'total_analyzed': len(results),
            'buy_signals_count': len(buy_signals),
            'volume_insights': volume_insights,
            'buy_signals': [
                {
                    'symbol': r['symbol'],
                    'score': r['composite_score'],
                    'action': r['trading_action']['action'],
                    'urgency': r['trading_action']['urgency'],
                    'is_urgent_squeeze': r['is_urgent_squeeze'],
                    'price': r['trading_action']['current_price'],
                    'stop_loss': r['trading_action']['stop_loss'],
                    'take_profits': r['trading_action']['take_profit_levels'],
                    'position_size': r['trading_action']['position_size_pct']
                }
                for r in buy_signals
            ],
            'all_results': [
                {
                    'symbol': r['symbol'],
                    'score': r['composite_score'],
                    'action': r['trading_action']['action']
                }
                for r in results
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"\n💾 Resultados guardados en {filename}")
    
    def generate_broker_instructions(self, buy_signals, capital=10000):
        """
        Genera instrucciones específicas para el broker
        Formato actualizado con take profits dinámicos
        """
        if not buy_signals:
            return
        
        print(f"\n{'='*70}")
        print("🔧 INSTRUCCIONES PARA EL BROKER")
        print(f"{'='*70}")
        print(f"💰 Capital: ${capital:,.2f}")
        print()
        
        for i, signal in enumerate(buy_signals, 1):
            ta = signal['trading_action']
            
            print(f"\n{'─'*70}")
            print(f"OPERACIÓN #{i}: {ta['symbol']}")
            
            if ta['is_urgent_squeeze']:
                print("🚨 *** SQUEEZE URGENTE - PRIORIDAD MÁXIMA ***")
            
            print(f"{'─'*70}")
            
            # Cálculo de acciones
            position_value = capital * (ta['position_size_pct'] / 100)
            shares = int(position_value / ta['current_price'])
            
            print(f"\n1️⃣ ORDEN DE COMPRA:")
            print(f"   BUY {shares} shares {ta['symbol']} @ ${ta['current_price']:.3f}")
            print(f"   Valor de posición: ${position_value:.2f} ({ta['position_size_pct']:.1f}% capital)")
            
            print(f"\n2️⃣ STOP LOSS (configurar INMEDIATAMENTE):")
            print(f"   STOP {ta['symbol']} @ ${ta['stop_loss']:.3f}")
            print(f"   Riesgo: ${(ta['current_price'] - ta['stop_loss']) * shares:.2f}")
            
            print(f"\n3️⃣ TRAILING STOP (configurar después de compra):")
            ts = ta['trailing_stop']
            trigger_price = ta['current_price'] * (1 + ts['trigger_gain_pct'])
            print(f"   ACTIVAR cuando precio alcance: ${trigger_price:.3f} (+{ts['trigger_gain_pct']:.1%})")
            print(f"   DISTANCIA: {ts['trail_distance_pct']:.1%} desde máximo")
            print(f"   Ejemplo: Si sube a ${trigger_price:.3f}, trailing stop en ${trigger_price * (1-ts['trail_distance_pct']):.3f}")
            
            print(f"\n4️⃣ TAKE PROFITS ESCALONADOS:")
            tp_levels = ta['take_profit_levels']
            shares_per_level = shares // len(tp_levels)
            
            for j, tp_price in enumerate(tp_levels, 1):
                if j < len(tp_levels):
                    tp_shares = shares_per_level
                else:
                    tp_shares = shares - (shares_per_level * (j-1))
                
                gain_pct = (tp_price - ta['current_price']) / ta['current_price']
                gain_dollars = (tp_price - ta['current_price']) * tp_shares
                
                print(f"   TP{j}: SELL {tp_shares} {ta['symbol']} @ ${tp_price:.3f}")
                print(f"        Ganancia: +{gain_pct:.1%} = ${gain_dollars:.2f}")
            
            print(f"\n5️⃣ PARÁMETROS DE GESTIÓN:")
            print(f"   • Max holding: {ta['max_holding_days']} días")
            print(f"   • Risk/Reward: 1:{ta['risk_reward_ratio']:.1f}")
            print(f"   • Monitoreo: Diario (mínimo)")
            
            if ta['is_urgent_squeeze']:
                print(f"\n🚨 ALERTA ESPECIAL:")
                print(f"   • Setup similar a squeezes históricos grandes")
                print(f"   • Monitorear CADA HORA durante mercado")
                print(f"   • Considerar aumentar posición si confirma squeeze")
                print(f"   • No vender todo en primer TP si momentum continúa")
        
        print(f"\n{'='*70}")
        print("⚠️  RECORDATORIOS CRÍTICOS:")
        print("   1. NUNCA operar sin stop loss configurado")
        print("   2. ACTIVAR trailing stop cuando alcance el trigger")
        print("   3. VENDER por tramos, no todo de una vez")
        print("   4. NO añadir a posiciones perdedoras")
        print("   5. Si el precio hace gap up >10%, evaluar antes de comprar")
        print("   6. Para SQUEEZE URGENTE: monitoreo intensivo requerido")
        print(f"{'='*70}\n")


def main():
    """
    Función principal - ejecuta análisis V3
    """
    print("\n🚀 INICIANDO ROBOT ADVISOR V3 ENHANCED")
    print("="*70)
    
    # Preguntar configuración
    print("\n🔧 SELECCIONAR CONFIGURACIÓN:")
    print("   1. Conservative - Máxima precisión, menos señales")
    print("   2. Balanced - Equilibrio óptimo (RECOMENDADO)")
    print("   3. Aggressive - Más señales, detecta squeezes early")
    print("   4. Very Aggressive - Máxima sensibilidad")
    
    config_options = {
        '1': 'conservative',
        '2': 'balanced',
        '3': 'aggressive',
        '4': 'very_aggressive'
    }
    
    choice = input("\nElegir (1-4, Enter para balanced): ").strip() or '2'
    config = config_options.get(choice, 'balanced')
    
    print(f"\n✅ Configuración seleccionada: {config.upper()}")
    
    # Crear manager y ejecutar
    manager = TradingManagerV3(config_preset=config)
    
    print("\n🔄 Ejecutando análisis completo...")
    results, buy_signals = manager.run_full_analysis()
    
    # Generar instrucciones para broker
    if buy_signals:
        print("\n📋 Generando instrucciones para broker...")
        manager.generate_broker_instructions(buy_signals, capital=10000)
    
    print("\n✅ ANÁLISIS COMPLETADO")
    print("="*70)
    print(f"📊 Total analizado: {len(results)} símbolos")
    print(f"🎯 Oportunidades: {len(buy_signals)}")
    print(f"⏰ Próximo análisis: Mañana a la misma hora")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Análisis interrumpido por usuario")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
