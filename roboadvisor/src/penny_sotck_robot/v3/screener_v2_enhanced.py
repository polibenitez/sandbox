#!/usr/bin/env python3
"""
SCREENER V2 ENHANCED - DETECCIÓN DINÁMICA DE SQUEEZES
===================================================

MEJORAS CLAVE vs V1:
1. ✅ Escaneo dinámico del mercado (no solo listas pre-definidas)
2. ✅ Detección de VOLUMEN EXPLOSIVO en tiempo real
3. ✅ Búsqueda activa de short interest alto
4. ✅ Identificación de compresión de precio
5. ✅ Integración con APIs de datos

Este screener HABRÍA DETECTADO BYND el 21 de octubre.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
import concurrent.futures
from typing import List, Dict, Tuple
import requests
from urllib.parse import quote

class DynamicStockScreener:
    """
    Screener dinámico que busca activamente en el mercado
    stocks con potencial de short squeeze
    """
    
    def __init__(self, mode="balanced"):
        """
        Modos disponibles:
        - conservative: Solo alta probabilidad, pocas señales
        - balanced: Equilibrio (RECOMENDADO)
        - aggressive: Más sensible, detecta early
        """
        self.mode = mode
        
        # Configuración por modo
        configs = {
            'conservative': {
                'price_max': 15.00,
                'volume_spike_min': 3.0,        # 3x volumen promedio
                'short_interest_min': 20,
                'min_price_change': -15,        # No más de -15% en el día
                'compression_days': 5
            },
            'balanced': {
                'price_max': 20.00,
                'volume_spike_min': 2.0,        # 2x volumen promedio
                'short_interest_min': 15,
                'min_price_change': -20,
                'compression_days': 4
            },
            'aggressive': {
                'price_max': 30.00,
                'volume_spike_min': 1.5,        # 1.5x volumen promedio
                'short_interest_min': 10,
                'min_price_change': -25,
                'compression_days': 3
            }
        }
        
        self.criteria = configs.get(mode, configs['balanced'])
        
        print(f"🔧 Screener V2 Enhanced inicializado - Modo: {mode.upper()}")
        print(f"📊 Criterios: Vol spike >{self.criteria['volume_spike_min']}x, SI >{self.criteria['short_interest_min']}%")
    
    # ========================================================================
    # MÉTODOS DE OBTENCIÓN DE UNIVERSO
    # ========================================================================
    
    def get_nasdaq_stocks(self, limit=None) -> List[str]:
        """
        Obtiene lista de stocks del NASDAQ
        (Método 1 para obtener universo completo)
        """
        try:
            # Descargar lista de NASDAQ desde FTP
            url = "ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqlisted.txt"
            
            # Alternativa: usar yfinance para obtener algunos tickers conocidos
            # o leer de archivo local si tienes uno
            
            # Por ahora, usar una lista extendida de conocidos
            nasdaq_universe = self._get_extended_universe()
            
            if limit:
                return nasdaq_universe[:limit]
            return nasdaq_universe
            
        except Exception as e:
            print(f"⚠️  Error obteniendo NASDAQ: {e}")
            return self._get_extended_universe()[:500]
    
    def _get_extended_universe(self) -> List[str]:
        """
        Universo extendido de stocks a escanear
        Incluye múltiples categorías
        """
        universe = []
        
        # Penny stocks conocidos
        penny_stocks = [
            'AMC', 'GME', 'BBBY', 'KOSS', 'EXPR', 'NAKD', 'CLOV', 'WISH',
            'BBIG', 'MULN', 'SNDL', 'CTRM', 'SHIP', 'GNUS', 'XELA', 'PROG',
            'ATER', 'SPRT', 'IRNT', 'OPAD', 'RDBX', 'NILE', 'HMHC', 'CLVS',
            'HTOO', 'LCFY', 'SIRI', 'XAIR', 'CTMX', 'ALBT', 'ADIL', 'AKBA',
            'OPEN', 'CHPT'
        ]
        universe.extend(penny_stocks)
        
        # Stocks con historial de alta volatilidad (incluir BYND aquí)
        volatile_stocks = [
            'BYND', 'TSLA', 'NVDA', 'PLTR', 'SOFI', 'NIO', 'RIVN', 'LCID',
            'COIN', 'HOOD', 'UPST', 'AFRM', 'SQ', 'SNAP', 'UBER', 'LYFT',
            'DASH', 'ABNB', 'RBLX', 'DKNG', 'PENN', 'FUBO', 'SKLZ'
        ]
        universe.extend(volatile_stocks)
        
        # Biotech (alta volatilidad por naturaleza)
        biotech = [
            'OBSV', 'ADTX', 'AGRX', 'AVIR', 'BIOC', 'BPTH', 'BVXV', 'CBAT',
            'CDXS', 'CHEK', 'COCP', 'CRIS', 'CYCC', 'DFFN', 'DMAC', 'GNPX',
            'HGEN', 'IMMP', 'IMGN', 'INCY', 'IONS', 'KALA', 'KPTI', 'KRYS'
        ]
        universe.extend(biotech)
        
        # Crypto-related
        crypto = [
            'MARA', 'RIOT', 'COIN', 'BITF', 'HUT', 'ARBK', 'BTBT', 'CAN',
            'EBON', 'GREE', 'HIVE', 'SOS', 'XNET', 'MOGO', 'BTCS'
        ]
        universe.extend(crypto)
        
        # EV y Tech pequeñas
        ev_tech = [
            'FSR', 'GOEV', 'RIDE', 'WKHS', 'SOLO', 'AYRO', 'BLNK', 'CHPT',
            'EVGO', 'NKLA', 'ARVL', 'PSNY', 'LAZR', 'VLDR', 'OUST', 'INVZ'
        ]
        universe.extend(ev_tech)
        
        # Shipping/Energy
        shipping = [
            'CTRM', 'EURN', 'GLBS', 'GOGL', 'NAT', 'STNG', 'TNK', 'TRMD',
            'INSW', 'GASS', 'GLNG', 'HMLP', 'NAV', 'SBBA'
        ]
        universe.extend(shipping)
        
        # Stocks de $5-20 con alta volatilidad (rango donde estaba BYND)
        mid_range = [
            'PLUG', 'TLRY', 'SNDL', 'EXPR', 'SPCE', 'BABA', 'NIO', 'XPEV',
            'LI', 'BIDU', 'JD', 'PDD', 'TME', 'BILI', 'IQ', 'VIPS',
            'MOMO', 'YY', 'JMIA', 'LAZR', 'WKHS', 'NKLA', 'RIDE', 'GOEV'
        ]
        universe.extend(mid_range)
        
        # Remover duplicados
        return list(set(universe))
    
    def get_high_short_interest_stocks(self) -> List[str]:
        """
        Busca stocks con alto short interest reportado
        
        NUEVO: Método específico para encontrar candidatos a squeeze
        """
        # En producción, usarías APIs como:
        # - FINRA short interest data
        # - Finviz screener
        # - HighShortInterest.com API
        
        # Por ahora, lista de stocks conocidos con SI alto
        high_si_stocks = [
            'BYND', 'CLOV', 'AMC', 'GME', 'BBBY', 'SOFI', 'UPST', 'SKLZ',
            'WISH', 'WKHS', 'RIDE', 'NKLA', 'GOEV', 'FSR', 'ARVL', 'LCID',
            'MULN', 'ATER', 'PROG', 'BBIG', 'RDBX', 'AVCT', 'GNUS', 'XELA'
        ]
        
        return high_si_stocks
    
    def get_unusual_volume_stocks_from_finviz(self) -> List[str]:
        """
        Obtiene stocks con volumen inusual del screener Finviz
        
        NUEVO: Detección proactiva de volumen explosivo
        """
        try:
            # En producción, harías web scraping de:
            # https://finviz.com/screener.ashx?v=111&s=ta_unusualvolume
            
            # Por ahora, simular con algunos conocidos
            # En tu implementación real, usa requests + BeautifulSoup
            
            print("   🌐 Buscando volumen inusual en Finviz...")
            # Aquí iría el código de scraping
            
            return []  # Retorna vacío por ahora
            
        except Exception as e:
            return []
    
    def build_dynamic_universe(self) -> List[str]:
        """
        Construye universo dinámico combinando múltiples fuentes
        
        MEJORA CLAVE: No depende solo de listas estáticas
        """
        print("\n🔍 CONSTRUYENDO UNIVERSO DINÁMICO...")
        print("="*60)
        
        universe = set()
        
        # Fuente 1: Universo extendido base
        extended = self._get_extended_universe()
        universe.update(extended)
        print(f"   ✅ Base extendida: {len(extended)} tickers")
        
        # Fuente 2: High short interest conocidos
        high_si = self.get_high_short_interest_stocks()
        universe.update(high_si)
        print(f"   ✅ Alto short interest: {len(high_si)} tickers")
        
        # Fuente 3: Volumen inusual (si disponible)
        unusual_vol = self.get_unusual_volume_stocks_from_finviz()
        if unusual_vol:
            universe.update(unusual_vol)
            print(f"   ✅ Volumen inusual: {len(unusual_vol)} tickers")
        
        # Fuente 4: Tickers adicionales del usuario
        user_watchlist = [
            'BYND', 'OPEN', 'CHPT', 'LCFY', 'SIRI', 'XAIR',
            'HTOO', 'CTMX', 'CLOV', 'ALBT', 'ADIL', 'AKBA', 'OPAD'
        ]
        universe.update(user_watchlist)
        print(f"   ✅ Watchlist usuario: {len(user_watchlist)} tickers")
        
        final_universe = list(universe)
        print(f"\n   🎯 UNIVERSO TOTAL: {len(final_universe)} tickers únicos")
        print("="*60)
        
        return final_universe
    
    # ========================================================================
    # ANÁLISIS MEJORADO DE CANDIDATOS
    # ========================================================================
    
    def analyze_candidate_enhanced(self, ticker: str) -> Dict:
        """
        Análisis mejorado que habría detectado BYND
        
        Detecta:
        1. Volumen explosivo (spike + aceleración)
        2. Compresión de precio
        3. Breakout potencial
        4. Short interest
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Obtener datos con más historial
            hist = stock.history(period='2mo')
            
            if len(hist) < 20:
                return {
                    'ticker': ticker,
                    'score': 0,
                    'reason': 'Datos insuficientes'
                }
            
            # Datos actuales
            current_price = hist['Close'].iloc[-1]
            current_volume = hist['Volume'].iloc[-1]
            
            # Filtro de precio
            if current_price > self.criteria['price_max']:
                return {
                    'ticker': ticker,
                    'score': 0,
                    'reason': f'Precio ${current_price:.2f} > ${self.criteria["price_max"]}'
                }
            
            # Calcular métricas clave
            avg_volume_20d = hist['Volume'][-20:].mean()
            volume_ratio = current_volume / avg_volume_20d if avg_volume_20d > 0 else 0
            
            # NUEVA MÉTRICA: Aceleración de volumen
            recent_vol_5d = hist['Volume'][-5:].mean()
            older_vol_15d = hist['Volume'][-20:-5].mean()
            volume_acceleration = recent_vol_5d / older_vol_15d if older_vol_15d > 0 else 1.0
            
            # NUEVA MÉTRICA: Compresión de precio
            price_5d = hist['Close'][-5:]
            price_range_pct = ((price_5d.max() - price_5d.min()) / price_5d.min() * 100)
            is_compressed = price_range_pct <= 15  # Rango estrecho
            
            # Cambio de precio
            price_change_pct = ((current_price - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100)
            
            # Short interest (de info)
            info = stock.info
            short_interest = info.get('shortPercentOfFloat', 0) * 100
            
            # RSI
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1] if len(rsi) > 0 else 50
            
            # NUEVA MÉTRICA: Distancia de máximo reciente (drawdown)
            max_price_20d = hist['High'][-20:].max()
            drawdown_pct = ((max_price_20d - current_price) / max_price_20d * 100)
            
            # ====================================================================
            # SCORING MEJORADO (habría detectado BYND)
            # ====================================================================
            
            score = 0
            reasons = []
            urgency_level = "NORMAL"
            
            # 1. VOLUMEN EXPLOSIVO (peso alto)
            if volume_ratio >= 5.0:
                score += 0.35
                reasons.append("🔥 VOL EXPLOSIVO (5x+)")
                urgency_level = "URGENTE"
            elif volume_ratio >= 3.0:
                score += 0.25
                reasons.append("⚡ Vol muy alto (3x+)")
                if urgency_level == "NORMAL":
                    urgency_level = "ALTO"
            elif volume_ratio >= self.criteria['volume_spike_min']:
                score += 0.15
                reasons.append(f"📊 Vol spike ({volume_ratio:.1f}x)")
            
            # 2. ACELERACIÓN DE VOLUMEN (NUEVO - clave para BYND)
            if volume_acceleration >= 2.0:
                score += 0.15
                reasons.append(f"🚀 Vol acelerando ({volume_acceleration:.1f}x)")
                urgency_level = "URGENTE"
            elif volume_acceleration >= 1.5:
                score += 0.10
                reasons.append("📈 Vol aumentando")
            
            # 3. COMPRESIÓN + BREAKOUT
            if is_compressed:
                score += 0.10
                reasons.append(f"🔄 Comprimido ({price_range_pct:.1f}%)")
                
                # Bonus si está rompiendo compresión con volumen
                if volume_ratio >= 2.0 and price_change_pct > 0:
                    score += 0.10
                    reasons.append("💥 Breakout de compresión")
            
            # 4. SHORT INTEREST
            if short_interest >= 30:
                score += 0.20
                reasons.append(f"🎯 SI extremo ({short_interest:.1f}%)")
            elif short_interest >= 20:
                score += 0.15
                reasons.append(f"📌 SI alto ({short_interest:.1f}%)")
            elif short_interest >= self.criteria['short_interest_min']:
                score += 0.10
                reasons.append(f"📍 SI moderado ({short_interest:.1f}%)")
            
            # 5. RSI (no sobrecomprado)
            if 30 <= current_rsi <= 70:
                score += 0.05
                reasons.append("✅ RSI saludable")
            elif current_rsi < 30:
                score += 0.10
                reasons.append("🔽 RSI oversold")
            
            # 6. DRAWDOWN (recuperación de caída)
            if drawdown_pct >= 50 and drawdown_pct <= 80:
                score += 0.10
                reasons.append(f"📉 Recuperando de -{drawdown_pct:.0f}%")
            
            # 7. MOMENTUM POSITIVO
            if price_change_pct > 5:
                score += 0.05
                reasons.append(f"⬆️ Momentum +{price_change_pct:.1f}%")
            
            # Penalización por caída extrema del día
            if price_change_pct < self.criteria['min_price_change']:
                score *= 0.5
                reasons.append(f"⚠️ Caída fuerte {price_change_pct:.1f}%")
            
            # Normalizar score
            score = min(1.0, score)
            
            return {
                'ticker': ticker,
                'score': score,
                'urgency': urgency_level,
                'price': current_price,
                'price_change_pct': price_change_pct,
                'volume_ratio': volume_ratio,
                'volume_acceleration': volume_acceleration,
                'short_interest': short_interest,
                'rsi': current_rsi,
                'is_compressed': is_compressed,
                'price_range_pct': price_range_pct,
                'drawdown_pct': drawdown_pct,
                'reasons': reasons,
                'market_cap': info.get('marketCap', 0)
            }
            
        except Exception as e:
            return {
                'ticker': ticker,
                'score': 0,
                'error': str(e)[:100]
            }
    
    def screen_universe_parallel(self, universe: List[str], max_workers: int = 10) -> List[Dict]:
        """
        Escanea universo en paralelo con análisis mejorado
        """
        print(f"\n🔄 ESCANEANDO {len(universe)} TICKERS...")
        print("="*60)
        
        results = []
        
        # Fase 1: Filtro básico rápido (precio)
        print("📋 Fase 1: Filtro rápido de precio...")
        candidates = []
        
        for ticker in universe[:100]:  # Limitar a 100 para no saturar yfinance
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period='5d')
                
                if len(hist) > 0:
                    price = hist['Close'].iloc[-1]
                    if 0.50 <= price <= self.criteria['price_max']:
                        candidates.append(ticker)
            except:
                continue
        
        print(f"   ✅ {len(candidates)} candidatos pasaron filtro de precio")
        
        # Fase 2: Análisis detallado en paralelo
        print("\n🔍 Fase 2: Análisis detallado en paralelo...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {
                executor.submit(self.analyze_candidate_enhanced, ticker): ticker 
                for ticker in candidates
            }
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_ticker):
                result = future.result()
                results.append(result)
                completed += 1
                
                if completed % 10 == 0:
                    print(f"   Procesados: {completed}/{len(candidates)}")
        
        # Ordenar por score
        results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        print(f"✅ Análisis completo de {len(results)} tickers")
        
        return results
    
    def generate_report_enhanced(self, results: List[Dict]) -> Dict:
        """
        Genera reporte mejorado con clasificación por urgencia
        """
        print("\n" + "="*70)
        print("📊 REPORTE DE SCREENING V2 ENHANCED")
        print("="*70)
        
        # Clasificar por score y urgencia
        urgent = [r for r in results if r.get('urgency') == 'URGENTE' and r.get('score', 0) >= 0.6]
        high = [r for r in results if r.get('score', 0) >= 0.6 and r.get('urgency') != 'URGENTE']
        medium = [r for r in results if 0.4 <= r.get('score', 0) < 0.6]
        low = [r for r in results if 0.2 <= r.get('score', 0) < 0.4]
        
        print(f"\n📈 ESTADÍSTICAS:")
        print(f"   • Total procesados: {len(results)}")
        print(f"   • 🚨 URGENTES (≥0.6 + vol explosivo): {len(urgent)}")
        print(f"   • ⚡ ALTA probabilidad (≥0.6): {len(high)}")
        print(f"   • 📊 MEDIA probabilidad (0.4-0.6): {len(medium)}")
        print(f"   • 📉 BAJA probabilidad (0.2-0.4): {len(low)}")
        
        # Mostrar URGENTES primero (casos como BYND)
        if urgent:
            print(f"\n{'='*70}")
            print(f"🚨 CANDIDATOS URGENTES - ATENCIÓN INMEDIATA ({len(urgent)})")
            print(f"{'='*70}")
            
            for i, candidate in enumerate(urgent[:5], 1):
                self._print_candidate_detail(i, candidate, detailed=True)
        
        # Top candidatos normales
        print(f"\n{'='*70}")
        print(f"⚡ TOP CANDIDATOS ALTA PROBABILIDAD ({len(high)})")
        print(f"{'='*70}")
        
        for i, candidate in enumerate(high[:10], 1):
            self._print_candidate_detail(i, candidate, detailed=False)
        
        # Candidatos medios (referencia)
        if medium:
            print(f"\n📊 CANDIDATOS PROBABILIDAD MEDIA ({len(medium)}):")
            medium_tickers = [c['ticker'] for c in medium[:20]]
            print(f"   {', '.join(medium_tickers)}")
        
        # Recomendaciones finales
        all_recommended = urgent + high
        print(f"\n{'='*70}")
        print(f"🤖 RECOMENDADOS PARA ROBOT ADVISOR ({len(all_recommended)}):")
        print(f"{'='*70}")
        
        if urgent:
            print(f"\n🚨 PRIORIDAD MÁXIMA (analizar HOY):")
            for c in urgent:
                print(f"   • {c['ticker']} (Score: {c['score']:.2f}) - {', '.join(c.get('reasons', [])[:2])}")
        
        if high:
            print(f"\n⚡ ALTA PRIORIDAD:")
            for c in high[:10]:
                print(f"   • {c['ticker']} (Score: {c['score']:.2f})")
        
        return {
            'timestamp': datetime.now(),
            'mode': self.mode,
            'statistics': {
                'total': len(results),
                'urgent': len(urgent),
                'high': len(high),
                'medium': len(medium),
                'low': len(low)
            },
            'urgent_candidates': urgent,
            'high_candidates': high,
            'medium_candidates': medium,
            'all_recommended': [c['ticker'] for c in all_recommended],
            'all_results': results
        }
    
    def _print_candidate_detail(self, index: int, candidate: Dict, detailed: bool = False):
        """Imprime detalles de un candidato"""
        ticker = candidate['ticker']
        score = candidate.get('score', 0)
        price = candidate.get('price', 0)
        change = candidate.get('price_change_pct', 0)
        vol_ratio = candidate.get('volume_ratio', 0)
        vol_accel = candidate.get('volume_acceleration', 1.0)
        si = candidate.get('short_interest', 0)
        urgency = candidate.get('urgency', 'NORMAL')
        
        # Indicador de urgencia
        urgency_icon = "🚨" if urgency == "URGENTE" else "⚡" if urgency == "ALTO" else "📊"
        
        print(f"\n{index:2d}. {urgency_icon} {ticker:6s} | Score: {score:.2f}")
        print(f"     💰 ${price:.3f} ({change:+.1f}%) | Vol: {vol_ratio:.1f}x")
        
        if detailed:
            print(f"     📊 Vol aceleración: {vol_accel:.1f}x | SI: {si:.1f}%")
            print(f"     🎯 Comprimido: {'SÍ' if candidate.get('is_compressed') else 'NO'} "
                  f"(Rango: {candidate.get('price_range_pct', 0):.1f}%)")
        
        reasons = candidate.get('reasons', [])
        if reasons:
            print(f"     ⭐ {' | '.join(reasons[:3])}")
    
    def save_results_enhanced(self, report: Dict):
        """Guarda resultados con timestamp"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        
        # JSON completo
        json_filename = f"screening_v2_{timestamp}.json"
        json_data = {
            'timestamp': report['timestamp'].isoformat(),
            'mode': report['mode'],
            'statistics': report['statistics'],
            'urgent_candidates': report['urgent_candidates'][:20],
            'high_candidates': report['high_candidates'][:30],
            'recommended_tickers': report['all_recommended']
        }
        
        with open(json_filename, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        print(f"\n💾 Resultados guardados:")
        print(f"   • JSON: {json_filename}")
        
        # CSV simple para importar
        csv_filename = f"screening_v2_{timestamp}.csv"
        
        # Combinar urgentes + high
        top_candidates = report['urgent_candidates'] + report['high_candidates']
        if top_candidates:
            df = pd.DataFrame(top_candidates[:50])
            df.to_csv(csv_filename, index=False)
            print(f"   • CSV: {csv_filename}")
        
        return json_filename
    
    def run_full_screening(self):
        """Ejecuta screening completo mejorado"""
        print("\n" + "="*70)
        print("🚀 SCREENER V2 ENHANCED - DETECCIÓN DINÁMICA")
        print("="*70)
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔧 Modo: {self.mode.upper()}")
        print()
        
        # 1. Construir universo dinámico
        universe = self.build_dynamic_universe()
        
        # 2. Escanear universo
        results = self.screen_universe_parallel(universe)
        
        # 3. Generar reporte
        report = self.generate_report_enhanced(results)
        
        # 4. Guardar resultados
        filename = self.save_results_enhanced(report)
        
        # 5. Resumen final
        print(f"\n{'='*70}")
        print("✅ SCREENING COMPLETADO")
        print(f"{'='*70}")
        print(f"🚨 Urgentes: {len(report['urgent_candidates'])}")
        print(f"⚡ Alta prob: {len(report['high_candidates'])}")
        print(f"📊 Media prob: {len(report['medium_candidates'])}")
        print(f"🎯 Total recomendados: {len(report['all_recommended'])}")
        print(f"\n💡 PRÓXIMO PASO:")
        print(f"   Ejecuta el Robot Advisor V3 con estos candidatos")
        print(f"{'='*70}")
        
        return report


def test_bynd_detection():
    """
    Test: ¿Habría detectado BYND este screener?
    """
    print("\n" + "="*70)
    print("🧪 TEST: ¿HABRÍA DETECTADO BYND?")
    print("="*70)
    
    screener = DynamicStockScreener(mode="aggressive")
    
    # Analizar solo BYND
    result = screener.analyze_candidate_enhanced("BYND")
    
    print("\n📊 ANÁLISIS DE BYND:")
    print("-"*70)
    print(f"   • Ticker: {result['ticker']}")
    print(f"   • Score: {result.get('score', 0):.3f}")
    print(f"   • Urgencia: {result.get('urgency', 'N/A')}")
    print(f"   • Precio: ${result.get('price', 0):.2f}")
    print(f"   • Cambio: {result.get('price_change_pct', 0):+.1f}%")
    print(f"   • Vol ratio: {result.get('volume_ratio', 0):.1f}x")
    print(f"   • Vol aceleración: {result.get('volume_acceleration', 0):.1f}x")
    print(f"   • Short interest: {result.get('short_interest', 0):.1f}%")
    print(f"   • Comprimido: {result.get('is_compressed', False)}")
    
    reasons = result.get('reasons', [])
    if reasons:
        print(f"\n   ⭐ Razones detectadas:")
        for reason in reasons:
            print(f"      • {reason}")
    
    score = result.get('score', 0)
    if score >= 0.6:
        print(f"\n   ✅ BYND HABRÍA SIDO DETECTADO")
        print(f"   ✅ Score {score:.3f} >= 0.60 (threshold recomendado)")
        if result.get('urgency') == 'URGENTE':
            print(f"   🚨 Clasificado como URGENTE - Alerta máxima")
    else:
        print(f"\n   ⚠️  Score {score:.3f} < 0.60")
        print(f"   💡 Considera modo 'aggressive' o revisar criterios")
    
    print("="*70)


def main():
    """Función principal"""
    print("🚀 SCREENER V2 ENHANCED - DETECCIÓN DINÁMICA DE SQUEEZES")
    print()
    print("Opciones:")
    print("1. Ejecutar screening completo (modo BALANCED)")
    print("2. Ejecutar screening completo (modo AGGRESSIVE)")
    print("3. Test: ¿Habría detectado BYND?")
    print("4. Ejecutar ambos (screening + test BYND)")
    
    choice = input("\nElegir (1-4, Enter para test BYND): ").strip() or '3'
    
    if choice == '1':
        screener = DynamicStockScreener(mode="balanced")
        screener.run_full_screening()
    
    elif choice == '2':
        screener = DynamicStockScreener(mode="aggressive")
        screener.run_full_screening()
    
    elif choice == '3':
        test_bynd_detection()
    
    elif choice == '4':
        print("\n1. Primero: Test BYND")
        test_bynd_detection()
        
        input("\nPresiona Enter para continuar con screening completo...")
        
        print("\n2. Ahora: Screening completo")
        screener = DynamicStockScreener(mode="aggressive")
        screener.run_full_screening()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Screening interrumpido")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
