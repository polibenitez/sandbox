#!/usr/bin/env python3
"""
INTEGRATION V5 - TRADING MANAGER EVOLUTION
===========================================

Sistema completo V5 con:
✅ Penny Stock Advisor V5 (ML + Alt Data)
✅ Market Context Analyzer mejorado
✅ Exit Manager con divergencias RSI/MACD
✅ Backtesting paralelo
✅ Optimización dinámica de thresholds
✅ Sistema de caché completo

Orquestador principal del sistema V5
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import json
from typing import Dict, List, Optional, Tuple

# Agregar utils al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

# Imports V5
from penny_stock_advisor_v5 import PennyStockAdvisorV5
from logging_config_v5 import get_logger
from backtester_v5 import BacktesterV5
from divergence_detector_v5 import DivergenceDetector

# Logger
logger = get_logger('trading_manager_v5')

# Watchlist actual
WATCHLIST_SYMBOLS = [
    "OPEN", "CHPT", "LCFY", "SIRI", "XAIR",
    "HTOO", "CTMX", "CLOV", "ALBT", "ADIL",
    "BYND", "AKBA", "OPAD", "AIRE", "YYAI",
    "RANI", "WOK", "AREB", "BENF", "CJET", "SBEV", "ISRG", "VTYX", 
    "RGC", "RVPH", "ONDS", "ADTX", "CLSK", "BITF", "IREN", "WGRX", "ADAG", "QLGN",
    "VIVK", "ASNS", "DFLI", "DVLT", "ASST", "PROP", "DGXX", "BKYI", "SLGB",
    "KULR", "FGI", "PLUG", "REKR", "SLDP", "SLNH", "NUAI", "HIPO", "UAMY", "OSCR", "ZETA", 
    "NXDR", "UTZ", "ATON", "BTOG", "BINI", "UP", "KSS", "SCWO", "STM"
]
WATCHLIST_SYMBOLS = set(WATCHLIST_SYMBOLS)


# ========================================================================
# MARKET CONTEXT ANALYZER V5
# ========================================================================

class MarketContextAnalyzerV5:
    """
    Analizador de contexto de mercado V5

    Mejoras:
    - Caché optimizado
    - Análisis más profundo
    - Integración con alternative data
    """

    def __init__(self):
        self.spy_cache = None
        self.qqq_cache = None
        self.vix_cache = None
        self.cache_time = None
        logger.info("Market Context Analyzer V5 inicializado")

    def get_market_context(self) -> Dict:
        """
        Obtiene contexto completo de mercado

        Returns:
            dict con spy_trend, qqq_trend, vix, market_favorable
        """
        if self._needs_cache_update():
            self._update_market_cache()

        spy_trend = self._analyze_spy_trend()
        qqq_trend = self._analyze_qqq_trend()
        vix_level = self.get_vix_level()

        market_favorable = (spy_trend != 'bearish') and (vix_level < 25)

        return {
            'spy_trend': spy_trend,
            'qqq_trend': qqq_trend,
            'vix': vix_level,
            'market_favorable': market_favorable,
            'sector_sentiment': 'neutral',
            'timestamp': datetime.now()
        }

    def get_vix_level(self) -> float:
        """Obtiene nivel de VIX actual"""
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
                return 15.0

        except Exception as e:
            logger.warning(f"Error obteniendo VIX: {e}")
            return 15.0

    def _needs_cache_update(self) -> bool:
        """Verifica si el caché necesita actualización"""
        if self.cache_time is None:
            return True

        elapsed = datetime.now() - self.cache_time
        return elapsed.total_seconds() > 1800  # 30 minutos

    def _update_market_cache(self):
        """Actualiza caché de datos de mercado"""
        try:
            spy = yf.Ticker("SPY")
            spy_hist = spy.history(period="10d")
            self.spy_cache = spy_hist if len(spy_hist) > 0 else None

            qqq = yf.Ticker("QQQ")
            qqq_hist = qqq.history(period="10d")
            self.qqq_cache = qqq_hist if len(qqq_hist) > 0 else None

            self.get_vix_level()  # Actualiza VIX

            self.cache_time = datetime.now()
            logger.info(f"Cache de mercado actualizado: {self.cache_time.strftime('%H:%M:%S')}")

        except Exception as e:
            logger.error(f"Error actualizando cache de mercado: {e}")

    def _analyze_spy_trend(self) -> str:
        """Analiza tendencia de SPY"""
        if self.spy_cache is None or len(self.spy_cache) < 5:
            return 'neutral'

        prices = self.spy_cache['Close'].values[-5:]
        current = prices[-1]
        change_5d = ((current - prices[0]) / prices[0] * 100) if prices[0] > 0 else 0

        if change_5d > 2:
            return 'bullish'
        elif change_5d < -2:
            return 'bearish'
        else:
            return 'neutral'

    def _analyze_qqq_trend(self) -> str:
        """Analiza tendencia de QQQ"""
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
# EXIT MANAGER V5 CON DIVERGENCIAS
# ========================================================================

class ExitManagerV5:
    """
    Gestor de salidas V5 con detección de divergencias

    NUEVO:
    - Integración con DivergenceDetector
    - Trailing stop dinámico mejorado
    - Detección de distribución mejorada
    """

    def __init__(self):
        self.positions = {}
        self.divergence_detector = DivergenceDetector(lookback_window=10)
        logger.info("Exit Manager V5 inicializado")

    def should_exit_position(self, position: Dict, current_data: Dict,
                            price_history: np.ndarray,
                            rsi_history: np.ndarray,
                            macd_history: np.ndarray) -> Tuple[bool, str]:
        """
        Determina si se debe salir de una posición

        NUEVO V5: Chequea divergencias RSI/MACD

        Returns:
            (should_exit, reason)
        """
        current_price = current_data.get('price', 0)
        entry_price = position.get('entry_price', current_price)

        # 1. Stop loss
        stop_loss = position.get('stop_loss', entry_price * 0.92)
        if current_price <= stop_loss:
            return True, 'Stop Loss'

        # 2. Take Profit 1
        tp1 = position.get('tp1', entry_price * 1.15)
        if current_price >= tp1:
            return True, 'Take Profit 1'

        # 3. Holding time máximo
        entry_date = position.get('entry_date', datetime.now())
        days_held = (datetime.now() - entry_date).days
        if days_held >= position.get('max_holding_days', 7):
            return True, 'Max Holding Time'

        # 4. NUEVO V5: Divergencias críticas
        if price_history is not None and rsi_history is not None and macd_history is not None:
            try:
                divergences = self.divergence_detector.detect_all_divergences(
                    price_history, rsi_history, macd_history
                )

                if divergences['critical_exit_signal']:
                    return True, 'Critical Divergence (RSI + MACD)'

                if divergences['strong_exit_signal']:
                    # Si ya tenemos ganancia, salir
                    gain_pct = ((current_price - entry_price) / entry_price)
                    if gain_pct > 0.10:  # +10%
                        return True, 'Divergence + Profit Secured'

            except Exception as e:
                logger.debug(f"Error detectando divergencias: {e}")

        # 5. Trailing stop
        highest_price = position.get('highest_price', current_price)
        trailing_trigger = position.get('trailing_stop_trigger', entry_price * 1.15)

        if highest_price >= trailing_trigger:
            trailing_distance = 0.08  # 8%
            trailing_stop = highest_price * (1 - trailing_distance)

            if current_price <= trailing_stop:
                return True, 'Trailing Stop'

        return False, 'Hold'

    def calculate_dynamic_trailing_stop(self, entry_price: float,
                                        current_price: float,
                                        highest_price: float,
                                        days_held: int,
                                        volatility: float) -> Dict:
        """Calcula trailing stop dinámico (V4 logic mantenida)"""
        current_gain_pct = ((current_price - entry_price) / entry_price) if entry_price > 0 else 0

        should_activate = current_gain_pct >= 0.15  # +15%

        if not should_activate:
            return {
                'trailing_stop_price': None,
                'distance_pct': 0,
                'should_activate': False,
                'reason': f'Ganancia {current_gain_pct:.1%} < 15%'
            }

        # Distancia dinámica según ganancia
        if current_gain_pct >= 0.50:
            distance_pct = 0.12
        elif current_gain_pct >= 0.30:
            distance_pct = 0.10
        else:
            distance_pct = 0.08

        # Ajustar por volatilidad
        if volatility > 0.10:
            distance_pct += 0.02

        trailing_stop_price = highest_price * (1 - distance_pct)

        return {
            'trailing_stop_price': trailing_stop_price,
            'distance_pct': distance_pct,
            'should_activate': True,
            'highest_price': highest_price,
            'current_gain_pct': current_gain_pct,
            'reason': f'Trailing stop a {distance_pct:.1%} desde máximo'
        }

    def calculate_partial_exit_levels(self, entry_price: float,
                                      current_price: float,
                                      position_size: int,
                                      atr: float) -> Dict:
        """Calcula niveles de salida parcial (V4 logic)"""
        tp1_price = entry_price * 1.15
        tp1_shares = int(position_size * 0.30)

        tp2_price = entry_price * 1.30
        tp2_shares = int(position_size * 0.30)

        remaining_shares = position_size - tp1_shares - tp2_shares

        return {
            'tp1': {
                'price': tp1_price,
                'shares': tp1_shares,
                'gain_pct': 15,
                'allocation': '30%',
                'status': 'ready' if current_price >= tp1_price else 'pending'
            },
            'tp2': {
                'price': tp2_price,
                'shares': tp2_shares,
                'gain_pct': 30,
                'allocation': '30%',
                'status': 'ready' if current_price >= tp2_price else 'pending'
            },
            'remaining': {
                'shares': remaining_shares,
                'allocation': '40%',
                'strategy': 'Trailing stop o divergencia',
                'exit_signals': [
                    'Divergencia RSI/MACD crítica',
                    'Trailing stop activado',
                    'Max holding time alcanzado'
                ]
            }
        }


# ========================================================================
# TRADING MANAGER V5 COMPLETO
# ========================================================================

class TradingManagerV5:
    """
    Gestor principal V5 - Sistema completo integrado
    """

    def __init__(self, config_preset="balanced", enable_backtesting=False):
        """
        Args:
            config_preset: "conservative" | "balanced" | "aggressive"
            enable_backtesting: Si True, inicializa backtester
        """
        logger.info("="*70)
        logger.info("TRADING MANAGER V5 - EVOLUTION EDITION")
        logger.info("="*70)

        # Componentes principales
        self.robot = PennyStockAdvisorV5(config_preset=config_preset)
        self.market_context = MarketContextAnalyzerV5()
        self.exit_manager = ExitManagerV5()
        self.watchlist = WATCHLIST_SYMBOLS
        self.robot.update_watchlist(self.watchlist)

        # Backtester (opcional)
        self.backtester = BacktesterV5() if enable_backtesting else None

        logger.info(f"Configuración: {config_preset.upper()}")
        logger.info(f"Watchlist: {len(self.watchlist)} símbolos")
        logger.info(f"Backtesting: {'Enabled' if enable_backtesting else 'Disabled'}")
        logger.info(f"Módulos: Advisor V5 + Context + Exit Manager + Optimizer")
        logger.info("="*70)

    def run_full_analysis(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Ejecuta análisis completo V5

        Returns:
            (results, buy_signals)
        """
        logger.info("="*70)
        logger.info("ANÁLISIS COMPLETO V5 - EVOLUTION")
        logger.info("="*70)
        logger.info(f"Fecha: {datetime.now().strftime('%A, %d de %B %Y - %H:%M')}")
        logger.info("")

        # 1. Contexto de mercado
        logger.info("Analizando contexto de mercado...")
        market_context = self.market_context.get_market_context()

        logger.info(f"   • SPY: {market_context['spy_trend'].upper()}")
        logger.info(f"   • QQQ: {market_context['qqq_trend'].upper()}")
        logger.info(f"   • VIX: {market_context['vix']:.1f}")
        logger.info(f"   • Favorable: {'✅ SÍ' if market_context['market_favorable'] else '❌ NO'}")

        if not market_context['market_favorable']:
            logger.warning("ADVERTENCIA: Mercado desfavorable - alto riesgo en penny stocks")

        # 2. Obtener thresholds dinámicos
        dynamic_thresholds = self.robot.optimizer.get_current_thresholds()
        logger.info(f"\nThresholds dinámicos:")
        logger.info(f"   • Buy Strong: {dynamic_thresholds['buy_strong']}")
        logger.info(f"   • Buy Moderate: {dynamic_thresholds['buy_moderate']}")
        logger.info(f"   • Watchlist: {dynamic_thresholds['watchlist']}")

        # 3. Analizar símbolos
        results = []
        logger.info(f"\nAnalizando {len(self.watchlist)} símbolos con sistema V5...")

        for symbol in self.watchlist:
            try:
                # Obtener datos (usa caché automáticamente)
                market_data, historical_data = self.robot.get_enhanced_market_data(symbol)

                if market_data is None:
                    logger.debug(f"  ⚠️  {symbol}: Sin datos disponibles")
                    continue

                # Análisis V5 completo
                analysis = self.robot.analyze_symbol_v5(
                    symbol, market_data, historical_data, market_context
                )

                results.append(analysis)

                # Log resumido
                decision = analysis['trading_decision']
                score = analysis['final_score']
                action = decision['action']
                ml_prob = analysis['ml_adjustment']['probability']

                logger.info(f"  {symbol}: Score {score:.0f}/100 (ML: {ml_prob:.1%}) → {action}")

            except Exception as e:
                logger.error(f"  ❌ {symbol}: Error - {e}")

        # Ordenar por score
        results.sort(key=lambda x: x['final_score'], reverse=True)

        # 4. Generar reporte
        buy_signals = self.generate_trading_report_v5(results, market_context)

        # 5. Guardar resultados
        self.save_results(results, buy_signals, market_context)

        return results, buy_signals

    def generate_trading_report_v5(self, results: List[Dict],
                                   market_context: Dict) -> List[Dict]:
        """Genera reporte de trading V5"""
        logger.info("="*70)
        logger.info("OPORTUNIDADES DE TRADING - V5 EVOLUTION")
        logger.info("="*70)

        # Filtrar señales de compra
        buy_signals = [r for r in results
                      if r['trading_decision']['action'] in ['COMPRA FUERTE', 'COMPRA MODERADA']]

        watchlist_signals = [r for r in results
                            if r['trading_decision']['action'] == 'WATCHLIST']

        if not buy_signals:
            logger.info("\n⏸️  No hay oportunidades de COMPRA hoy")
            logger.info("   El sistema V5 es altamente selectivo")

            if watchlist_signals:
                logger.info(f"\n👀 WATCHLIST ({len(watchlist_signals)} símbolos):")
                for r in watchlist_signals[:5]:
                    logger.info(f"   • {r['symbol']}: {r['trading_decision']['reason']}")

            return []

        logger.info(f"\n✅ {len(buy_signals)} OPORTUNIDADES DETECTADAS\n")

        for i, result in enumerate(buy_signals, 1):
            decision = result['trading_decision']
            symbol = result['symbol']
            ml_adj = result['ml_adjustment']

            logger.info("─"*70)

            # Determinar tipo de oportunidad
            opportunity_type = result.get('opportunity_type', 'Resorte Comprimido')

            if opportunity_type == "Momentum Puro":
                logger.info(f"{i}. 🚀 {symbol} - {decision['action']} [MOMENTUM PURO]")
            elif decision.get('is_late_entry', False):
                logger.info(f"{i}. 📈 {symbol} - {decision['action']} ⚠️ ENTRADA TARDÍA DÍA 3")
            else:
                logger.info(f"{i}. 📈 {symbol} - {decision['action']} [RESORTE COMPRIMIDO]")
            logger.info("─"*70)

            logger.info(f"💯 SCORING:")
            logger.info(f"   • Tipo: {opportunity_type.upper()}")
            logger.info(f"   • Score final: {decision['score']:.0f}/100")
            if 'raw_score' in decision:
                logger.info(f"   • Score bruto: {decision['raw_score']:.0f}/100")
            if 'penalty_applied' in decision:
                logger.info(f"   • Penalizaciones: {decision['penalty_applied']:.0f} pts")
            logger.info(f"   • Día de explosión: {decision.get('explosion_day', 'N/A')}")

            # Información específica de Momentum Puro
            if opportunity_type == "Momentum Puro" and 'momentum_puro_stats' in decision:
                stats = decision['momentum_puro_stats']
                logger.info(f"\n🚀 MOMENTUM PURO STATS:")
                logger.info(f"   • Volumen: {stats['volume_ratio']:.1f}x promedio")
                logger.info(f"   • Cambio 3d: {stats['price_change_3d']:.1f}%")
                logger.info(f"   • Breakout desde mínimo: {stats['breakout_pct']:.1%}")
                logger.info(f"   • RSI: {stats['rsi']:.0f}")
            else:
                logger.info(f"\n🧠 MACHINE LEARNING:")
                logger.info(f"   • Probabilidad breakout: {ml_adj['probability']:.1%}")
                logger.info(f"   • Confianza: {ml_adj['confidence'].upper()}")
                logger.info(f"   • Modelo: {'Disponible' if ml_adj['model_available'] else 'No disponible'}")

            # Mostrar fases solo para Resorte Comprimido
            if opportunity_type != "Momentum Puro":
                logger.info(f"\n📊 ANÁLISIS POR FASES:")
                logger.info(f"   🔵 Fase 1 (Setup): {result['phase1_setup']['score']:.0f}/100")
                for signal in result['phase1_setup']['signals'][:2]:
                    logger.info(f"      • {signal}")

                logger.info(f"   🟡 Fase 2 (Trigger): {result['phase2_trigger']['score']:.0f}/100")
                explosion = result['phase2_trigger']['explosion_info']
                logger.info(f"      • {explosion['status']}")

                logger.info(f"   🌍 Fase 3 (Context+Alt): {result['phase3_context']['score']:.0f}/100")
                for signal in result['phase3_context']['signals'][:2]:
                    logger.info(f"      • {signal}")

            logger.info(f"\n💰 PLAN DE TRADING:")
            logger.info(f"   • Precio entrada: ${decision['current_price']:.3f}")
            logger.info(f"   • Posición: {decision['position_size_pct']:.0f}% del capital")
            logger.info(f"   • Stop loss: ${decision['stop_loss']:.3f} " +
                       f"(-{((decision['current_price']-decision['stop_loss'])/decision['current_price']*100):.1f}%)")

            logger.info(f"\n🎯 TAKE PROFITS:")
            logger.info(f"   • TP1 ({decision['tp1_allocation']:.0f}%): ${decision['take_profit_1']:.3f} (+15%)")
            logger.info(f"   • TP2 ({decision['tp2_allocation']:.0f}%): ${decision['take_profit_2']:.3f} (+30%)")
            logger.info(f"   • Restante (40%): Trailing stop o divergencia")

            if decision['warnings']:
                logger.info(f"\n⚠️  ADVERTENCIAS:")
                for warning in decision['warnings']:
                    logger.info(f"   • {warning}")

        logger.info("="*70)
        logger.info("💡 FILOSOFÍAS DE TRADING V5:")
        logger.info("")
        logger.info("   📊 RESORTE COMPRIMIDO (Tradicional):")
        logger.info("      • Requiere compresión previa (rango ≤ 8%)")
        logger.info("      • Volumen explosivo ≥ 2.5x")
        logger.info("      • ML + Alternative Data")
        logger.info("      • Posición estándar: 2-3%")
        logger.info("")
        logger.info("   🚀 MOMENTUM PURO (Nueva):")
        logger.info("      • Sin compresión requerida")
        logger.info("      • Volumen extremo ≥ 4x")
        logger.info("      • Rebote fuerte desde mínimos")
        logger.info("      • Posición reducida: 1-1.5% (Mayor riesgo)")
        logger.info("")
        logger.info("   ✅ ML predice probabilidad de éxito")
        logger.info("   ✅ Alternative data captura sentiment")
        logger.info("   ✅ Thresholds auto-ajustados")
        logger.info("   ✅ Divergencias para salidas óptimas")
        logger.info("   ✅ Sistema completo de gestión de riesgo")
        logger.info("="*70)

        return buy_signals

    def save_results(self, results: List[Dict], buy_signals: List[Dict],
                    market_context: Dict):
        """Guarda resultados del análisis"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        filename = f"trading_results_v5_{timestamp}.json"

        output = {
            'version': 'V5 - Evolution Edition',
            'timestamp': datetime.now().isoformat(),
            'config_preset': self.robot.config_preset,
            'market_context': {
                'spy_trend': market_context['spy_trend'],
                'qqq_trend': market_context['qqq_trend'],
                'vix': market_context['vix'],
                'favorable': market_context['market_favorable']
            },
            'optimizer_thresholds': self.robot.optimizer.get_current_thresholds(),
            'total_analyzed': len(results),
            'buy_signals_count': len(buy_signals),
            'buy_signals': [
                {
                    'symbol': r['symbol'],
                    'opportunity_type': r.get('opportunity_type', 'Resorte Comprimido'),
                    'final_score': r['final_score'],
                    'raw_score': r.get('raw_score', r['final_score']),
                    'ml_probability': r.get('ml_adjustment', {}).get('probability', 0.5),
                    'ml_confidence': r.get('ml_adjustment', {}).get('confidence', 'N/A'),
                    'action': r['trading_decision']['action'],
                    'phase1_score': r.get('phase1_setup', {}).get('score', 0),
                    'phase2_score': r.get('phase2_trigger', {}).get('score', 0),
                    'phase3_score': r.get('phase3_context', {}).get('score', 0),
                    'penalties': r.get('penalties', {}).get('total_penalty', 0),
                    'explosion_day': r['trading_decision'].get('explosion_day', 'N/A'),
                    'entry_price': r['trading_decision']['current_price'],
                    'stop_loss': r['trading_decision']['stop_loss'],
                    'tp1': r['trading_decision']['take_profit_1'],
                    'tp2': r['trading_decision']['take_profit_2'],
                    'position_size_pct': r['trading_decision'].get('position_size_pct', 0)
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

        logger.info(f"\n💾 Resultados guardados en {filename}")

    def run_backtest(self, symbols: List[str], start_date: str, end_date: str) -> Dict:
        """
        Ejecuta backtest V5

        Args:
            symbols: Lista de símbolos
            start_date: Fecha inicio (YYYY-MM-DD)
            end_date: Fecha fin (YYYY-MM-DD)

        Returns:
            Dict con resultados del backtest
        """
        if not self.backtester:
            logger.warning("Backtester no inicializado. Usa enable_backtesting=True")
            return {}

        logger.info("="*70)
        logger.info("BACKTESTING V5 - PARALLEL EXECUTION")
        logger.info("="*70)

        results = self.backtester.run_backtest(
            symbols=symbols,
            advisor=self.robot,
            start_date=start_date,
            end_date=end_date
        )

        # Generar reporte
        report = self.backtester.generate_report(results)
        logger.info(report)

        # Guardar resultados
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        filename = f"backtest_results_v5_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"\n💾 Backtest guardado en {filename}")

        return results


# ========================================================================
# MAIN
# ========================================================================

def main():
    """Función principal"""
    logger.info("\n🚀 PENNY STOCK ROBOT V5 - EVOLUTION EDITION")
    logger.info("="*70)
    logger.info("Sistema completo con ML, Alternative Data y Optimización")
    logger.info("="*70)

    # Crear manager
    manager = TradingManagerV5(config_preset="balanced", enable_backtesting=False)

    # Ejecutar análisis
    logger.info("\n🔄 Ejecutando análisis completo V5...")
    results, buy_signals = manager.run_full_analysis()

    logger.info("\n✅ ANÁLISIS COMPLETADO")
    logger.info("="*70)
    logger.info(f"📊 Total analizado: {len(results)} símbolos")
    logger.info(f"🎯 Oportunidades: {len(buy_signals)}")
    logger.info(f"📈 Sistema: ML + Alt Data + Optimizer + Cache")
    logger.info("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n⏹️  Análisis interrumpido por usuario")
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
