#!/usr/bin/env python3
"""
INTEGRATION V4 - TRADING MANAGER CON CONTEXTO DE MERCADO Y SALIDAS MEJORADAS
==============================================================================

FASE B: Contexto de mercado (SPY/QQQ/VIX)
FASE C: Gestión de salidas mejorada

Se integra con penny_stock_advisor_v4.py
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import json
from typing import Dict, List, Optional, Tuple
from penny_stock_advisor_v4 import PennyStockAdvisorV4

# Watchlist actual
WATCHLIST_SYMBOLS = [
    "OPEN", "CHPT", "LCFY", "SIRI", "XAIR",
    "HTOO", "CTMX", "CLOV", "ALBT", "ADIL",
    "BYND", "AKBA", "OPAD", "AIRE", "YYAI",
    "RANI", "WOK", "AREB", "BENF", "CJET", "SBEV", "ISRG", "VTYX", 
    "RGC", "RVPH", "ONDS", "ADTX", "CLSK", "BITF", "IREN", "WGRX", "ADAG", "QLGN",
    "VIVK", "ASNS", "DFLI", "DVLT", "ASST"
]
WATCHLIST_SYMBOLS = set(WATCHLIST_SYMBOLS)


# ========================================================================
# FASE B: CONTEXTO DE MERCADO
# ========================================================================

class MarketContextAnalyzer:
    """
    Analizador de contexto de mercado general

    No puedes luchar contra el mercado. Si SPY/QQQ están bajistas,
    incluso el mejor penny stock puede caer.
    """

    def __init__(self):
        self.spy_cache = None
        self.qqq_cache = None
        self.vix_cache = None
        self.cache_time = None

    def get_market_context(self) -> Dict:
        """
        🌍 FASE B.1: Obtiene contexto completo de mercado

        Returns:
            dict: {
                'spy_trend': 'bullish' | 'neutral' | 'bearish',
                'qqq_trend': 'bullish' | 'neutral' | 'bearish',
                'vix': float,
                'market_favorable': bool,
                'sector_sentiment': str,
                'timestamp': datetime
            }
        """
        # Actualizar cache si es necesario (cada 30 min)
        if self._needs_cache_update():
            self._update_market_cache()

        spy_trend = self._analyze_spy_trend()
        qqq_trend = self._analyze_qqq_trend()
        vix_level = self.get_vix_level()

        # Contexto general favorable si:
        # - SPY no está bajista
        # - VIX < 25
        market_favorable = (spy_trend != 'bearish') and (vix_level < 25)

        return {
            'spy_trend': spy_trend,
            'qqq_trend': qqq_trend,
            'vix': vix_level,
            'market_favorable': market_favorable,
            'sector_sentiment': 'neutral',  # Simplificado por ahora
            'timestamp': datetime.now()
        }

    def get_vix_level(self) -> float:
        """
        🌍 FASE B.2: Obtiene nivel de VIX

        VIX < 15: Complacencia
        VIX 15-20: Normal
        VIX 20-25: Nerviosismo
        VIX > 25: Pánico (evitar penny stocks)
        """
        if self.vix_cache is not None:
            return self.vix_cache

        try:
            vix = yf.Ticker("^VIX")
            hist = vix.history(period="5d")

            if len(hist) > 0:
                vix_value = hist['Close'].iloc[-1]
                self.vix_cache = float(vix_value)
                return float(vix_value)
            else:
                return 15.0  # Default

        except Exception as e:
            print(f"⚠️  Error obteniendo VIX: {e}")
            return 15.0

    def get_sector_sentiment(self, symbol: str) -> str:
        """
        🌍 FASE B.3: Obtiene sentimiento del sector

        Returns:
            'bullish' | 'neutral' | 'panic'
        """
        # Simplificado por ahora - podría expandirse con análisis sectorial real
        return 'neutral'

    def _needs_cache_update(self) -> bool:
        """Verifica si necesitamos actualizar el cache"""
        if self.cache_time is None:
            return True

        elapsed = datetime.now() - self.cache_time
        return elapsed.total_seconds() > 1800  # 30 minutos

    def _update_market_cache(self):
        """Actualiza cache de datos de mercado"""
        try:
            # SPY
            spy = yf.Ticker("SPY")
            spy_hist = spy.history(period="10d")
            self.spy_cache = spy_hist if len(spy_hist) > 0 else None

            # QQQ
            qqq = yf.Ticker("QQQ")
            qqq_hist = qqq.history(period="10d")
            self.qqq_cache = qqq_hist if len(qqq_hist) > 0 else None

            # VIX (actualizar directamente)
            self.get_vix_level()

            self.cache_time = datetime.now()
            print(f"✅ Cache de mercado actualizado: {self.cache_time.strftime('%H:%M:%S')}")

        except Exception as e:
            print(f"⚠️  Error actualizando cache de mercado: {e}")

    def _analyze_spy_trend(self) -> str:
        """Analiza tendencia de SPY en últimos 5 días"""
        if self.spy_cache is None or len(self.spy_cache) < 5:
            return 'neutral'

        prices = self.spy_cache['Close'].values[-5:]
        sma_5 = np.mean(prices)
        current = prices[-1]

        # Cambio porcentual en 5 días
        change_5d = ((current - prices[0]) / prices[0] * 100) if prices[0] > 0 else 0

        if change_5d > 2:
            return 'bullish'
        elif change_5d < -2:
            return 'bearish'
        else:
            return 'neutral'

    def _analyze_qqq_trend(self) -> str:
        """Analiza tendencia de QQQ en últimos 5 días"""
        if self.qqq_cache is None or len(self.qqq_cache) < 5:
            return 'neutral'

        prices = self.qqq_cache['Close'].values[-5:]
        change_5d = ((prices[-1] - prices[0]) / prices[0] * 100) if prices[0] > 0 else 0

        if change_5d > 2:
            return 'bullish'
        elif change_5d < -2:
            return 'bearish'
        else:
            return 'neutral'


# ========================================================================
# FASE C: GESTIÓN DE SALIDAS MEJORADA
# ========================================================================

class ExitManager:
    """
    Gestor de salidas mejorado con divergencias y patrones de distribución
    """

    def __init__(self):
        self.positions = {}  # Tracking de posiciones

    def detect_bearish_divergence(self, price_history: np.ndarray,
                                   rsi_history: np.ndarray,
                                   lookback: int = 5) -> Dict:
        """
        🔴 FASE C.1: Detecta divergencia bajista en RSI

        Divergencia bajista:
        - Precio hace higher high
        - RSI hace lower high
        - Señal de distribución inminente

        Returns:
            dict: {
                'has_divergence': bool,
                'strength': 'weak' | 'moderate' | 'strong',
                'days_confirmed': int,
                'should_exit': bool
            }
        """
        if price_history is None or rsi_history is None:
            return {'has_divergence': False, 'should_exit': False}

        if len(price_history) < lookback or len(rsi_history) < lookback:
            return {'has_divergence': False, 'should_exit': False}

        # Últimos precios y RSI
        recent_prices = price_history[-lookback:]
        recent_rsi = rsi_history[-lookback:]

        # Encontrar picos locales
        price_peaks = self._find_peaks(recent_prices)
        rsi_peaks = self._find_peaks(recent_rsi)

        if len(price_peaks) < 2 or len(rsi_peaks) < 2:
            return {'has_divergence': False, 'should_exit': False}

        # Verificar divergencia: precio sube, RSI baja
        price_higher = recent_prices[price_peaks[-1]] > recent_prices[price_peaks[-2]]
        rsi_lower = recent_rsi[rsi_peaks[-1]] < recent_rsi[rsi_peaks[-2]]

        has_divergence = price_higher and rsi_lower

        if has_divergence:
            # Calcular fuerza de la divergencia
            price_change = ((recent_prices[price_peaks[-1]] - recent_prices[price_peaks[-2]]) /
                          recent_prices[price_peaks[-2]] * 100)
            rsi_change = recent_rsi[rsi_peaks[-1]] - recent_rsi[rsi_peaks[-2]]

            if price_change > 10 and rsi_change < -10:
                strength = 'strong'
                should_exit = True
            elif price_change > 5 and rsi_change < -5:
                strength = 'moderate'
                should_exit = True
            else:
                strength = 'weak'
                should_exit = False

            return {
                'has_divergence': True,
                'strength': strength,
                'price_change_pct': price_change,
                'rsi_change': rsi_change,
                'should_exit': should_exit,
                'reason': f'Divergencia bajista {strength}: precio +{price_change:.1f}%, RSI {rsi_change:.1f}'
            }

        return {'has_divergence': False, 'should_exit': False}

    def calculate_dynamic_trailing_stop(self, entry_price: float,
                                        current_price: float,
                                        highest_price: float,
                                        days_held: int,
                                        volatility: float) -> Dict:
        """
        🔴 FASE C.2: Calcula trailing stop dinámico

        El trailing stop se ajusta según:
        - Ganancia actual
        - Volatilidad del activo
        - Días en posición

        Returns:
            dict: {
                'trailing_stop_price': float,
                'distance_pct': float,
                'should_activate': bool,
                'reason': str
            }
        """
        current_gain_pct = ((current_price - entry_price) / entry_price) if entry_price > 0 else 0

        # ¿Activar trailing stop?
        should_activate = current_gain_pct >= 0.15  # Activar a +15%

        if not should_activate:
            return {
                'trailing_stop_price': None,
                'distance_pct': 0,
                'should_activate': False,
                'reason': f'Ganancia {current_gain_pct:.1%} < 15%, trailing stop no activado'
            }

        # Calcular distancia del trailing stop (dinámico según ganancia)
        if current_gain_pct >= 0.50:  # +50%
            distance_pct = 0.12  # 12% de margen
        elif current_gain_pct >= 0.30:  # +30%
            distance_pct = 0.10  # 10% de margen
        else:  # +15% to +30%
            distance_pct = 0.08  # 8% de margen

        # Ajustar por volatilidad
        if volatility > 0.10:  # Alta volatilidad
            distance_pct += 0.02

        trailing_stop_price = highest_price * (1 - distance_pct)

        return {
            'trailing_stop_price': trailing_stop_price,
            'distance_pct': distance_pct,
            'should_activate': True,
            'highest_price': highest_price,
            'current_gain_pct': current_gain_pct,
            'reason': f'Trailing stop a {distance_pct:.1%} desde máximo (${highest_price:.2f})'
        }

    def calculate_partial_exit_levels(self, entry_price: float,
                                      current_price: float,
                                      position_size: int,
                                      atr: float) -> Dict:
        """
        🔴 FASE C.3: Calcula niveles de salida parcial escalonada

        Salidas escalonadas:
        - TP1 (30%): +12-15%
        - TP2 (30%): +25-30%
        - TP3 (40%): Trailing stop o señal técnica

        Returns:
            dict: {
                'tp1': {'price': float, 'shares': int, 'gain_pct': float},
                'tp2': {'price': float, 'shares': int, 'gain_pct': float},
                'remaining': {'shares': int, 'strategy': str}
            }
        """
        # TP1: +15%
        tp1_price = entry_price * 1.15
        tp1_shares = int(position_size * 0.30)

        # TP2: +30%
        tp2_price = entry_price * 1.30
        tp2_shares = int(position_size * 0.30)

        # Restante: 40%
        remaining_shares = position_size - tp1_shares - tp2_shares

        return {
            'tp1': {
                'price': tp1_price,
                'shares': tp1_shares,
                'gain_pct': 15,
                'allocation': '30%',
                'status': 'pending' if current_price < tp1_price else 'ready'
            },
            'tp2': {
                'price': tp2_price,
                'shares': tp2_shares,
                'gain_pct': 30,
                'allocation': '30%',
                'status': 'pending' if current_price < tp2_price else 'ready'
            },
            'remaining': {
                'shares': remaining_shares,
                'allocation': '40%',
                'strategy': 'Trailing stop o divergencia bajista',
                'exit_signals': [
                    'Divergencia RSI bajista confirmada',
                    'Volumen de distribución (>2x avg + día rojo)',
                    'Cierre bajo EMA(20) y EMA(50)',
                    'Mercado general cae >2% intradía'
                ]
            }
        }

    def detect_distribution_pattern(self, price_history: np.ndarray,
                                    volume_history: np.ndarray,
                                    avg_volume: float) -> Dict:
        """
        🔴 FASE C.4: Detecta patrón de distribución

        Señales de distribución:
        - Volumen alto en días rojos
        - Precio estancado o bajando
        - Institucionales saliendo

        Returns:
            dict: {
                'is_distributing': bool,
                'strength': str,
                'should_exit_immediately': bool
            }
        """
        if price_history is None or volume_history is None:
            return {'is_distributing': False, 'should_exit_immediately': False}

        if len(price_history) < 3 or len(volume_history) < 3:
            return {'is_distributing': False, 'should_exit_immediately': False}

        # Últimos 3 días
        recent_prices = price_history[-3:]
        recent_volumes = volume_history[-3:]

        # Contar días rojos con volumen alto
        distribution_days = 0
        for i in range(len(recent_prices) - 1):
            price_down = recent_prices[i+1] < recent_prices[i]
            volume_high = recent_volumes[i+1] > (avg_volume * 2.0)

            if price_down and volume_high:
                distribution_days += 1

        is_distributing = distribution_days >= 2

        if is_distributing:
            strength = 'strong' if distribution_days == 3 else 'moderate'
            should_exit = distribution_days >= 2

            return {
                'is_distributing': True,
                'strength': strength,
                'distribution_days': distribution_days,
                'should_exit_immediately': should_exit,
                'reason': f'{distribution_days} días de distribución (volumen alto + precio bajo)'
            }

        return {'is_distributing': False, 'should_exit_immediately': False}

    def _find_peaks(self, data: np.ndarray) -> List[int]:
        """Encuentra índices de picos locales"""
        peaks = []
        for i in range(1, len(data) - 1):
            if data[i] > data[i-1] and data[i] > data[i+1]:
                peaks.append(i)
        return peaks


# ========================================================================
# TRADING MANAGER V4 COMPLETO
# ========================================================================

class TradingManagerV4:
    """
    Gestor principal V4 con todas las fases implementadas
    """

    def __init__(self, config_preset="balanced"):
        """
        Args:
            config_preset: "conservative" | "balanced" | "aggressive"
        """
        self.robot = PennyStockAdvisorV4(config_preset=config_preset)
        self.market_context = MarketContextAnalyzer()
        self.exit_manager = ExitManager()
        self.watchlist = WATCHLIST_SYMBOLS
        self.robot.update_watchlist(self.watchlist)

        print(f"\n🤖 Trading Manager V4 - Paradigm Shift Edition")
        print(f"📊 Configuración: {config_preset.upper()}")
        print(f"🎯 Watchlist: {len(self.watchlist)} símbolos")
        print(f"✅ Módulos: Scoring 3 capas + Contexto + Salidas mejoradas")

    def run_full_analysis(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Ejecuta análisis completo V4

        Returns:
            (results, buy_signals)
        """
        print(f"\n{'='*70}")
        print(f"🚀 ANÁLISIS COMPLETO V4 - PARADIGM SHIFT")
        print(f"{'='*70}")
        print(f"📅 {datetime.now().strftime('%A, %d de %B %Y - %H:%M')}")
        print()

        # 1. Obtener contexto de mercado (FASE B)
        print("🌍 Analizando contexto de mercado...")
        market_context = self.market_context.get_market_context()

        print(f"   • SPY: {market_context['spy_trend'].upper()}")
        print(f"   • QQQ: {market_context['qqq_trend'].upper()}")
        print(f"   • VIX: {market_context['vix']:.1f}")
        print(f"   • Favorable: {'✅ SÍ' if market_context['market_favorable'] else '❌ NO'}")

        if not market_context['market_favorable']:
            print(f"\n⚠️  ADVERTENCIA: Mercado desfavorable - penny stocks de alto riesgo")

        # 2. Analizar símbolos
        results = []

        print(f"\n📊 Analizando {len(self.watchlist)} símbolos con sistema V4...\n")

        for symbol in self.watchlist:
            try:
                # Obtener datos
                market_data, historical_data = self.robot.get_enhanced_market_data(symbol)

                if market_data is None:
                    print(f"  ⚠️  {symbol}: Sin datos disponibles")
                    continue

                # Análisis V4
                analysis = self.robot.analyze_symbol_v4(
                    symbol, market_data, historical_data, market_context
                )

                results.append(analysis)

                # Mostrar resumen
                decision = analysis['trading_decision']
                score = analysis['final_score']
                action = decision['action']

                print(f"  {symbol}: Score {score:.0f}/100 → {action}")

            except Exception as e:
                print(f"  ❌ {symbol}: Error - {e}")

        # Ordenar por score
        results.sort(key=lambda x: x['final_score'], reverse=True)

        # 3. Generar reporte
        buy_signals = self.generate_trading_report_v4(results)

        # 4. Guardar resultados
        self.save_results(results, buy_signals, market_context)

        return results, buy_signals

    def generate_trading_report_v4(self, results: List[Dict]) -> List[Dict]:
        """Genera reporte de trading V4"""
        print(f"\n{'='*70}")
        print("🎯 OPORTUNIDADES DE TRADING - V4 PARADIGM SHIFT")
        print(f"{'='*70}")

        # Filtrar señales de compra
        buy_signals = [r for r in results
                      if r['trading_decision']['action'] in ['COMPRA FUERTE', 'COMPRA MODERADA']]

        watchlist_signals = [r for r in results
                            if r['trading_decision']['action'] == 'WATCHLIST']

        if not buy_signals:
            print("\n⏸️  No hay oportunidades de COMPRA hoy")
            print("   El sistema V4 es muy selectivo - esperando setups perfectos")

            if watchlist_signals:
                print(f"\n👀 WATCHLIST ({len(watchlist_signals)} símbolos):")
                for r in watchlist_signals[:5]:
                    print(f"   • {r['symbol']}: {r['trading_decision']['reason']}")

            return []

        print(f"\n✅ {len(buy_signals)} OPORTUNIDADES DETECTADAS\n")

        for i, result in enumerate(buy_signals, 1):
            decision = result['trading_decision']
            symbol = result['symbol']

            print(f"\n{'─'*70}")
            print(f"{i}. 📈 {symbol} - {decision['action']}")
            print(f"{'─'*70}")

            print(f"💯 SCORING:")
            print(f"   • Score final: {decision['score']:.0f}/100")
            print(f"   • Score bruto: {decision['raw_score']:.0f}/100")
            print(f"   • Penalizaciones: {decision['penalty_applied']:.0f} puntos")

            print(f"\n📊 ANÁLISIS POR FASES:")
            print(f"   🔵 Fase 1 (Setup): {result['phase1_setup']['score']:.0f}/{result['phase1_setup']['max_score']:.0f}")
            for signal in result['phase1_setup']['signals'][:2]:
                print(f"      • {signal}")

            print(f"   🟡 Fase 2 (Trigger): {result['phase2_trigger']['score']:.0f}/{result['phase2_trigger']['max_score']:.0f}")
            explosion = result['phase2_trigger']['explosion_info']
            print(f"      • {explosion['status']}")

            print(f"   🌍 Fase 3 (Contexto): {result['phase3_context']['score']:.0f}/{result['phase3_context']['max_score']:.0f}")

            print(f"\n💰 PLAN DE TRADING:")
            print(f"   • Precio entrada: ${decision['current_price']:.3f}")
            print(f"   • Posición: {decision['position_size_pct']:.0f}% del capital")
            print(f"   • Stop loss: ${decision['stop_loss']:.3f} (-{((decision['current_price']-decision['stop_loss'])/decision['current_price']*100):.1f}%)")

            print(f"\n🎯 TAKE PROFITS ESCALONADOS:")
            print(f"   • TP1 ({decision['tp1_allocation']:.0f}%): ${decision['take_profit_1']:.3f} (+{decision.get('exit_params', {}).get('tp1_pct', 15)*100:.0f}%)")
            print(f"   • TP2 ({decision['tp2_allocation']:.0f}%): ${decision['take_profit_2']:.3f} (+{decision.get('exit_params', {}).get('tp2_pct', 30)*100:.0f}%)")
            print(f"   • Restante (40%): Trailing stop o señal técnica")

            print(f"\n📈 TRAILING STOP:")
            print(f"   • Activar a: ${decision['trailing_stop_trigger']:.3f} (+15%)")

            if decision['warnings']:
                print(f"\n⚠️  ADVERTENCIAS:")
                for warning in decision['warnings']:
                    print(f"   • {warning}")

        print(f"\n{'='*70}")
        print("💡 FILOSOFÍA V4:")
        print("   ✅ Compramos el RESORTE COMPRIMIDO, no el resorte liberado")
        print("   ✅ Entramos DÍA 1-2 del movimiento, NO día 3+")
        print("   ✅ Penalizaciones severas evitan entradas tardías")
        print("   ✅ Contexto de mercado filtrado (SPY/VIX)")
        print(f"{'='*70}\n")

        return buy_signals

    def save_results(self, results: List[Dict], buy_signals: List[Dict],
                    market_context: Dict):
        """Guarda resultados del análisis"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        filename = f"trading_results_v4_{timestamp}.json"

        output = {
            'version': 'V4 - Paradigm Shift',
            'timestamp': datetime.now().isoformat(),
            'config_preset': self.robot.config_preset,
            'market_context': {
                'spy_trend': market_context['spy_trend'],
                'vix': market_context['vix'],
                'favorable': market_context['market_favorable']
            },
            'total_analyzed': len(results),
            'buy_signals_count': len(buy_signals),
            'buy_signals': [
                {
                    'symbol': r['symbol'],
                    'final_score': r['final_score'],
                    'raw_score': r['raw_score'],
                    'action': r['trading_decision']['action'],
                    'phase1_score': r['phase1_setup']['score'],
                    'phase2_score': r['phase2_trigger']['score'],
                    'phase3_score': r['phase3_context']['score'],
                    'penalties': r['penalties']['total_penalty'],
                    'explosion_day': r['phase2_trigger']['explosion_info']['explosion_day'],
                    'entry_price': r['trading_decision']['current_price'],
                    'stop_loss': r['trading_decision']['stop_loss'],
                    'tp1': r['trading_decision']['take_profit_1'],
                    'tp2': r['trading_decision']['take_profit_2']
                }
                for r in buy_signals
            ],
            'all_results': [
                {
                    'symbol': r['symbol'],
                    'final_score': r['final_score'],
                    'action': r['trading_decision']['action']
                }
                for r in results
            ]
        }

        with open(filename, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        print(f"💾 Resultados guardados en {filename}")

    def compare_v3_vs_v4(self, symbol="BYND"):
        """
        Compara cómo V3 vs V4 habría manejado BYND
        """
        print(f"\n{'='*70}")
        print(f"📊 COMPARACIÓN V3 vs V4 - Caso {symbol}")
        print(f"{'='*70}")

        print("\n🔍 ESCENARIO:")
        print(f"   • {symbol} @ $2.15 con volumen explosivo")
        print("   • Día 2-3 del movimiento alcista")
        print("   • Resultado real: Subió 150% a ~$5.38")

        print("\n❌ SISTEMA V3 (ANTERIOR):")
        print("   • Acción: COMPRAR")
        print("   • Razón: Vio volumen alto + breakout")
        print("   • Problema: NO sabía que era día 2-3")
        print("   • Resultado: Entró tarde, salió temprano")

        print("\n✅ SISTEMA V4 (NUEVO):")
        print("   • Acción: RECHAZAR o COMPRA con precaución")
        print("   • Razón: Detecta que es DÍA 2-3 de explosión")
        print("   • Penalización: -30 puntos por 'late to party'")
        print("   • Score: Baja de 75 a 45 → NO COMPRA")
        print("   • Resultado: Evita entrada tardía")

        print("\n💡 DIFERENCIA CLAVE:")
        print("   V3 → Compra la explosión (reactivo)")
        print("   V4 → Anticipa la compresión (proactivo)")

        print(f"{'='*70}\n")


def main():
    """Función principal"""
    print("\n🚀 PENNY STOCK ROBOT V4 - PARADIGM SHIFT EDITION")
    print("="*70)
    print("Ya no compramos cohetes en vuelo.")
    print("Ahora los encontramos en la plataforma de lanzamiento.")
    print("="*70)

    # Crear manager
    manager = TradingManagerV4(config_preset="balanced")

    # Mostrar comparación
    manager.compare_v3_vs_v4()

    # Ejecutar análisis
    print("\n🔄 Ejecutando análisis completo V4...")
    results, buy_signals = manager.run_full_analysis()

    print("\n✅ ANÁLISIS COMPLETADO")
    print("="*70)
    print(f"📊 Total analizado: {len(results)} símbolos")
    print(f"🎯 Oportunidades: {len(buy_signals)}")
    print(f"📈 Filosofía V4: Anticipar, no reaccionar")
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
