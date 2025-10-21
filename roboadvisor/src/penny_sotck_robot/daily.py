#!/usr/bin/env python3
"""
ROBOT ADVISOR - SCRIPT DIARIO DE EJECUCIÓN
==========================================

Este es tu script principal para ejecutar cada día de trading.
Combina todas las funcionalidades en un flujo automatizado.

Uso:
    python daily_trading_script.py

Configuración:
    1. Actualiza WATCHLIST_SYMBOLS con tus acciones
    2. Configura API_KEYS si tienes APIs premium
    3. Ejecuta diariamente antes del mercado
"""

import sys
import os
import numpy as np
from datetime import datetime, timedelta
import json
import yfinance as yf
#import talib
import random
import traceback
from penny_stock_robot_advisor import PennyStockRobotAdvisor


# Configuración principal
WATCHLIST_SYMBOLS = [
    "OPEN",
    "CHPT",
    "LCFY",
    "SIRI",
    "XAIR",
    "HTOO",
    "CTMX",
    "CLOV",
    "ALBT", 
    "ADIL", 
    "bynd"
]

API_KEYS = {
    "reddit": None,  # API Reddit para sentimiento
    "twitter": None,  # API Twitter/X para trends
    "finra": None,  # API FINRA para short interest
    "alpha_vantage": None,  # API Alpha Vantage para datos fundamentales
}


class DailyTradingManager:
    """Gestor principal para ejecución diaria del robot advisor"""

    def __init__(self):
        self.setup_environment()
        # self.load_components()
        # config_preset (str): "conservative", "balanced", "aggressive", "very_aggressive"
        self.robot = PennyStockRobotAdvisor(config_preset="balanced")

    def setup_environment(self):
        """Configura el entorno de ejecución"""
        print("🚀 INICIANDO ROBOT ADVISOR")
        print("=" * 50)
        print(f"📅 Fecha: {datetime.now().strftime('%A, %d de %B %Y - %H:%M')}")
        print(f"🎯 Watchlist: {len(WATCHLIST_SYMBOLS)} símbolos")
        print("-" * 50)

    def create_fallback_data(self, symbol):
        """Crea datos de fallback seguros para evitar errores"""

        price = random.uniform(0.1, 2.0)
        return {
            "price": price,
            "volume": random.randint(500000, 20000000),
            "avg_volume_20d": random.randint(300000, 10000000),
            "price_change_pct": random.uniform(-10, 15),
            "short_interest_pct": random.uniform(5, 40),
            "days_to_cover": random.uniform(0.5, 5.0),
            "borrow_rate": random.uniform(10, 100),
            "vwap": price * random.uniform(0.95, 1.05),
            "rsi": random.uniform(25, 75),
            "bid_ask_spread_pct": random.uniform(1, 15),
            "market_depth_dollars": random.randint(5000, 50000),
            "daily_dollar_volume": price * random.randint(100000, 15000000),
            "atr_14": price * random.uniform(0.03, 0.08),
            "has_delisting_warning": price < 1.0,
            "last_updated": datetime.now().isoformat(),
        }

    def get_market_data(self):
        """Obtiene datos de mercado - integración con APIs reales"""
        print("\n🔍 OBTENIENDO DATOS DE MERCADO...")
        market_data = {}

        # Intenta usar yfinance si está disponible
        try:
            print("📊 Usando Yahoo Finance para datos básicos...")

            for symbol in WATCHLIST_SYMBOLS:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="1mo")
                    info = ticker.info

                    if len(hist) > 0:
                        current_price = hist["Close"].iloc[-1]
                        current_volume = hist["Volume"].iloc[-1]
                        avg_volume = hist["Volume"].mean()
                        max_price = hist["High"].max()
                        drawdown = ((max_price - current_price) / max_price) * 100

                        market_data[symbol] = {
                            "price": float(current_price),
                            "volume": int(current_volume),
                            "avg_volume_20d": int(avg_volume),
                            "short_interest_pct": info.get("shortPercentOfFloat", 0.1)
                            * 100,
                            "has_delisting_warning": current_price < 1.0,
                            "reddit_mentions": self.get_social_mentions(symbol),
                            "twitter_trend": False,  # Requiere API
                            "influencer_mention": False,  # Análisis manual
                            "sector": info.get("sector", "Unknown"),
                            "drawdown_pct": drawdown,
                            "sector_momentum": 0.0,
                        }
                        print(f"  ✅ {symbol}: ${current_price:.3f}")

                except Exception as e:
                    print(f"  ❌ {symbol}: Error - {e}")

        except ImportError:
            print("⚠️  yfinance no disponible, usando datos simulados...")
            # Datos simulados para demo
            for symbol in WATCHLIST_SYMBOLS:
                market_data[symbol] = {
                    "price": random.uniform(0.1, 2.0),
                    "volume": random.randint(1000000, 50000000),
                    "avg_volume_20d": random.randint(500000, 10000000),
                    "short_interest_pct": random.uniform(5, 40),
                    "has_delisting_warning": random.choice([True, False]),
                    "reddit_mentions": random.randint(0, 200),
                    "twitter_trend": random.choice([True, False]),
                    "influencer_mention": random.choice([True, False]),
                    "sector": random.choice(
                        ["Technology", "Healthcare", "Energy", "Automotive"]
                    ),
                    "drawdown_pct": random.uniform(70, 98),
                    "sector_momentum": random.uniform(-0.3, 0.3),
                }

        print(f"✅ Datos obtenidos para {len(market_data)} símbolos")
        return market_data

    def get_social_mentions(self, symbol):
        """Obtiene menciones en redes sociales (simplificado)"""
        # En producción: integrar con APIs de Reddit, Twitter, FINRA, etc.

        return random.randint(0, 200)

    
    def run_analysis(self, market_data):
        """
        ACTUALIZADO: Ejecuta el análisis usando la nueva clase PennyStockRobotAdvisor
        """
        print("\n🤖 EJECUTANDO ANÁLISIS...")

        # Configurar watchlist
        self.robot.update_watchlist(WATCHLIST_SYMBOLS)
        
        # Usar el método generate_daily_report de la nueva clase
        daily_report = self.robot.generate_daily_report(market_data, show_details=False)
        
        # Convertir el formato del nuevo reporte al formato esperado por generate_trading_report
        results = []
        for analysis in daily_report['all_results']:
            # Adaptar formato para compatibilidad
            adapted_result = {
                'symbol': analysis['symbol'],
                'composite_score': analysis['composite_score'],
                'trading_action': {
                    'symbol': analysis['trading_action']['symbol'],
                    'action': analysis['trading_action']['action'],
                    'urgency': analysis['trading_action'].get('urgency', 'BAJA'),
                    'composite_score': analysis['trading_action']['composite_score'],
                    'current_price': analysis['trading_action'].get('current_price', 0),  # ESTO FALTABA
                    'stop_loss': analysis['trading_action'].get('stop_loss', 0),
                    'stop_method': analysis['trading_action'].get('stop_method', 'Fixed'),
                    'stop_distance_pct': analysis['trading_action'].get('stop_distance_pct', 0.15),
                    'take_profit_levels': analysis['trading_action'].get('take_profit_levels', []),
                    'position_size_pct': analysis['trading_action'].get('position_size_pct', 0),
                    'max_holding_days': analysis['trading_action'].get('max_holding_days', 10),
                    'risk_reward_ratio': analysis['trading_action'].get('risk_reward_ratio', 0),
                    'config_preset': analysis['trading_action'].get('config_preset', 'aggressive'),
                    'reason': analysis['trading_action'].get('reason', '')
                },
                'signals': analysis.get('signals', []),
                'analysis_timestamp': analysis.get('analysis_timestamp', datetime.now())
            }
            results.append(adapted_result)
        
        # Mostrar estadísticas del análisis
        stats = daily_report['statistics']
        print(f"📊 Estadísticas:")
        print(f"   • Símbolos analizados: {stats['analyzed']}")
        print(f"   • Pasaron filtros: {stats['passed_liquidity']}")
        print(f"   • Señales de compra: {stats['buy_signals']}")
        print(f"   • Tasa de rechazo: {stats['rejection_rate']:.1f}%")
        
        return results

    def generate_trading_report(self, results):
        """
        ACTUALIZADO: Genera reporte de trading compatible con la nueva estructura
        """
        print("\n" + "="*60)
        print("📋 REPORTE DIARIO DE TRADING")
        print("="*60)
        
        # Filtrar oportunidades de compra
        buy_signals = []
        for r in results:
            action = r['trading_action']['action']
            if action not in ['ESPERAR', 'DESCALIFICADA - LIQUIDEZ INSUFICIENTE']:
                buy_signals.append(r)
        
        if not buy_signals:
            print("\n⏸️  SIN OPORTUNIDADES HOY")
            print("   ✅ Algoritmo selectivo funcionando correctamente")
            print("   ✅ Mejor esperar mejores oportunidades")
            print("   💡 Considera usar configuración más agresiva si necesitas más trades")
        else:
            print(f"\n🚨 {len(buy_signals)} OPORTUNIDADES DETECTADAS:")
            print("-" * 50)
            
            for i, result in enumerate(buy_signals[:5], 1):  # Top 5
                ta = result['trading_action']
                
                # Validar que tenemos todos los campos necesarios
                current_price = ta.get('current_price', 0)
                if current_price == 0:
                    print(f"⚠️  {ta['symbol']}: Precio no disponible, saltando...")
                    continue
                    
                print(f"\n{i}. 📈 {ta['symbol']} - {ta['action']} ({ta.get('urgency', 'MEDIA')})")
                print(f"   💰 Precio: ${current_price:.3f}")
                print(f"   📊 Score: {ta['composite_score']:.3f}/1.000")
                print(f"   🔧 Config: {ta.get('config_preset', 'N/A').upper()}")
                print(f"   💸 Posición: {ta.get('position_size_pct', 0):.1f}% del capital")
                
                # Stop loss con método
                stop_loss = ta.get('stop_loss', current_price * 0.85)
                stop_method = ta.get('stop_method', 'Fixed 15%')
                stop_distance = ta.get('stop_distance_pct', 0.15)
                print(f"   🛑 Stop Loss: ${stop_loss:.3f} ({stop_method}) = -{stop_distance:.1%}")
                
                # Take profit levels
                tp_levels = ta.get('take_profit_levels', [current_price * 1.25, current_price * 1.5, current_price * 2.0])
                if len(tp_levels) >= 3:
                    print(f"   🎯 Take Profit: ${tp_levels[0]:.3f} / ${tp_levels[1]:.3f} / ${tp_levels[2]:.3f}")
                
                # Risk/reward ratio
                rr_ratio = ta.get('risk_reward_ratio', 0)
                if rr_ratio > 0:
                    print(f"   📊 Risk/Reward: 1:{rr_ratio:.1f}")
                
                # Máximo holding
                max_days = ta.get('max_holding_days', 10)
                print(f"   ⏰ Máximo: {max_days} días")
                
                # Señales clave (si están disponibles)
                if 'signals' in result and result['signals']:
                    key_signals = [s for s in result['signals'] if s.get('score', 0) >= 0.6]
                    if key_signals:
                        signals_desc = ', '.join([f"{s['signal']}({s['score']:.2f})" for s in key_signals])
                        print(f"   ⭐ Señales clave: {signals_desc}")
        
        # Mostrar descalificados por liquidez
        liquidity_fails = [r for r in results if 'LIQUIDEZ' in r['trading_action']['action']]
        if liquidity_fails:
            print(f"\n❌ DESCALIFICADOS POR LIQUIDEZ ({len(liquidity_fails)}):")
            for result in liquidity_fails[:3]:  # Solo top 3
                ta = result['trading_action']
                reason = ta.get('reason', 'Liquidez insuficiente')
                print(f"   • {ta['symbol']}: {reason}")
        
        # Mostrar con score insuficiente
        low_scores = [r for r in results if r['trading_action']['action'] == 'ESPERAR']
        if low_scores:
            print(f"\n⏸️  SCORE INSUFICIENTE ({len(low_scores)}):")
            # Obtener umbral mínimo (si está disponible)
            min_threshold = 0.45  # Default para configuración agresiva
            if hasattr(self, 'robot') and hasattr(self.robot, 'thresholds'):
                min_threshold = self.robot.thresholds.get('buy_light', 0.45)
            
            for result in low_scores[:3]:  # Solo top 3
                ta = result['trading_action']
                score = ta.get('composite_score', 0)
                print(f"   • {ta['symbol']}: Score {score:.3f} < {min_threshold}")
        
        # Acciones para broker
        print("\n" + "="*60)
        print("🔧 ACCIONES ESPECÍFICAS PARA BROKER:")
        print("="*60)
        
        if buy_signals:
            print("\n1️⃣ ÓRDENES DE COMPRA:")
            for result in buy_signals:
                ta = result['trading_action']
                current_price = ta.get('current_price', 0)
                position_pct = ta.get('position_size_pct', 2.0)
                
                if current_price > 0:
                    # Asumir capital de $10,000 para cálculo de acciones
                    capital = 10000
                    position_value = capital * (position_pct / 100)
                    shares = int(position_value / current_price)
                    print(f"   • BUY {shares} shares {ta['symbol']} @ ${current_price:.3f}")
            
            print("\n2️⃣ STOP LOSSES (configurar inmediatamente):")
            for result in buy_signals:
                ta = result['trading_action']
                stop_price = ta.get('stop_loss', 0)
                if stop_price > 0:
                    print(f"   • STOP {ta['symbol']} @ ${stop_price:.3f}")
            
            print("\n3️⃣ TAKE PROFITS (órdenes escalonadas):")
            for result in buy_signals:
                ta = result['trading_action']
                current_price = ta.get('current_price', 0)
                position_pct = ta.get('position_size_pct', 2.0)
                tp_levels = ta.get('take_profit_levels', [])
                
                if current_price > 0 and tp_levels:
                    capital = 10000
                    position_value = capital * (position_pct / 100)
                    total_shares = int(position_value / current_price)
                    shares_per_level = total_shares // len(tp_levels)
                    
                    for i, tp_price in enumerate(tp_levels):
                        shares = shares_per_level if i < len(tp_levels) - 1 else total_shares - (shares_per_level * i)
                        print(f"   • SELL {shares} {ta['symbol']} @ ${tp_price:.3f}")
        else:
            print("   🚫 No hay órdenes para hoy")
            print("   💡 Considera ajustar configuración si necesitas más trades")
        
        # Recordatorios de riesgo
        print("\n⚠️  RECORDATORIOS DE RIESGO:")
        if hasattr(self, 'robot'):
            config = self.robot.config_preset.upper()
            print(f"   • Configuración: {config}")
            if config in ['AGGRESSIVE', 'VERY_AGGRESSIVE']:
                print("   • ⚠️ CONFIGURACIÓN AGRESIVA - Mayor riesgo")
                print("   • ⚠️ Monitorea posiciones más frecuentemente")
        
        print("   • Stop-loss OBLIGATORIO para todas las posiciones")
        print("   • Vender por TRAMOS en las subidas")
        print("   • NO perseguir precios después de gaps")
        print("   • Revisar posiciones DIARIAMENTE")
        print("="*60)
        
        return buy_signals

    # FUNCIÓN AUXILIAR PARA VALIDAR DATOS
    def validate_market_data(self, market_data):
        """Valida que los datos de mercado tengan los campos necesarios"""
        required_fields = [
            'price', 'volume', 'avg_volume_20d', 'price_change_pct',
            'short_interest_pct', 'days_to_cover', 'borrow_rate',
            'vwap', 'rsi', 'bid_ask_spread_pct', 'market_depth_dollars',
            'daily_dollar_volume', 'atr_14', 'has_delisting_warning'
        ]
        
        validated_data = {}
        missing_fields = []
        
        for symbol, data in market_data.items():
            validated_data[symbol] = {}
            symbol_missing = []
            
            for field in required_fields:
                if field in data and data[field] is not None:
                    validated_data[symbol][field] = data[field]
                else:
                    symbol_missing.append(field)
                    # Valores por defecto
                    defaults = {
                        'price': 1.0, 'volume': 1000000, 'avg_volume_20d': 500000,
                        'price_change_pct': 0, 'short_interest_pct': 10,
                        'days_to_cover': 1.5, 'borrow_rate': 25, 'vwap': 1.0,
                        'rsi': 50, 'bid_ask_spread_pct': 8, 'market_depth_dollars': 10000,
                        'daily_dollar_volume': 500000, 'atr_14': 0.05,
                        'has_delisting_warning': False
                    }
                    validated_data[symbol][field] = defaults.get(field, 0)
            
            if symbol_missing:
                missing_fields.append((symbol, symbol_missing))
        
        if missing_fields:
            print("⚠️  Campos faltantes en datos de mercado (usando valores por defecto):")
            for symbol, fields in missing_fields[:3]:  # Solo mostrar primeros 3
                print(f"   • {symbol}: {', '.join(fields[:5])}")  # Solo primeros 5 campos
        
        return validated_data

    # FUNCIÓN PRINCIPAL ACTUALIZADA PARA EL DAILY_TRADING_MANAGER
    def run_daily_analysis_updated(self):
        """Versión actualizada que usa la nueva clase"""
        try:
            # 1. Obtener datos
            market_data = self.get_market_data()
            
            # 2. Validar datos
            validated_data = self.validate_market_data(market_data)
            
            # 3. Ejecutar análisis con nueva clase
            results = self.run_analysis(validated_data)
            
            # 4. Generar reporte
            opportunities = self.generate_trading_report(results)
            
            # 5. Guardar resultados
            self.save_results(results)
            
            print(f"\n✅ Análisis completado: {len(opportunities)} oportunidades")
            return results
            
        except Exception as e:
            print(f"❌ Error en análisis diario: {e}")

            traceback.print_exc()
            return []

    def save_results(self, results):
        """Guarda resultados para seguimiento histórico"""
        filename = f"trading_results_{datetime.now().strftime('%Y%m%d')}.json"

        output = {
            "date": datetime.now().isoformat(),
            "symbols_analyzed": len(WATCHLIST_SYMBOLS),
            "opportunities_found": len(
                [r for r in results if r["trading_action"]["action"] != "ESPERAR"]
            ),
            "results": results,
        }

        with open(filename, "w") as f:
            json.dump(output, f, indent=2, default=str)

        print(f"💾 Resultados guardados en {filename}")


def main():
    """Función principal"""
    try:
        print("🚀 INICIANDO ROBOT ADVISOR MEJORADO")
        print("📊 Nuevas métricas: Short Interest Cualificado + Momentum + Liquidez")
        print()

        # Crear manager y ejecutar
        manager = DailyTradingManager()
        results = manager.run_daily_analysis_updated()

        # Mensaje final
        print(f"\n🎯 ROBOT ADVISOR MEJORADO - Análisis completado")
        print(f"📈 Criterios más exigentes aplicados (umbrales: 0.55, 0.70, 0.85)")
        print(f"🛡️ Filtros de liquidez y momentum activos")
        print(f"📧 Revisa las órdenes sugeridas antes de ejecutar")
        print(f"⏰ Próximo análisis: mañana a la misma hora")

        if results:
            buy_count = sum(
                1 for r in results if r["trading_action"]["action"] != "ESPERAR"
            )
            print(f"📊 Oportunidades encontradas: {buy_count}/{len(results)}")

    except KeyboardInterrupt:
        print("\n\n⏹️  Análisis interrumpido por usuario")
    except Exception as e:
        print(f"\n❌ Error crítico: {e}")
        print("💡 Verifica que tengas instalado: pip install numpy yfinance")
        sys.exit(1)


if __name__ == "__main__":
    # Configuración inicial
    print("🔧 ROBOT ADVISOR MEJORADO - Configuración:")
    print(f"   • Símbolos watchlist: {len(WATCHLIST_SYMBOLS)}")
    print(f"   • APIs configuradas: {sum(1 for k, v in API_KEYS.items() if v)}")
    print(f"   • Nuevas métricas: Short Interest Cualificado, Momentum, Liquidez")
    print(f"   • Umbrales: 0.55 (Ligero), 0.70 (Moderado), 0.85 (Fuerte)")
    print()

    # Ejecutar análisis
    main()
