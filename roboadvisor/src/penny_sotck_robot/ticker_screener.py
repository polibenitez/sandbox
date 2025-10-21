#!/usr/bin/env python3
"""
TICKER SCREENER - BUSCADOR DE CANDIDATOS
=======================================

Script independiente para encontrar penny stocks candidatos
para el Robot Advisor basado en criterios pre-filtrados.

Busca stocks con potencial de short squeeze antes del análisis completo.

Uso: python ticker_screener.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
import time
import concurrent.futures
from typing import List, Dict, Tuple

class TickerScreener:
    """
    Screener para encontrar penny stocks candidatos
    """
    
    def __init__(self):
        self.screening_criteria = {
            'price_range': {'min': 0.10, 'max': 10.00},  # Penny stocks
            'volume_min': 1000000,                        # Mínimo 1M volumen
            'market_cap_max': 2000000000,                 # Máximo $2B market cap
            'short_interest_min': 10,                     # Mínimo 10% SI
            'price_change_range': {'min': -20, 'max': 50} # No gaps extremos
        }
        
        self.data_sources = {
            'finviz': True,      # Screener web
            'yahoo': True,       # Yahoo Finance
            'reddit': True,      # Reddit mentions
            'custom_lists': True # Listas personalizadas
        }
        
    def get_penny_stock_universe(self) -> List[str]:
        """
        Obtiene universo inicial de penny stocks desde múltiples fuentes
        """
        print("🔍 OBTENIENDO UNIVERSO DE PENNY STOCKS...")
        all_tickers = set()
        
        # Fuente 1: Lista manual de penny stocks conocidos
        manual_tickers = self.get_manual_penny_stocks()
        all_tickers.update(manual_tickers)
        print(f"   📋 Manual: {len(manual_tickers)} tickers")
        
        # Fuente 2: Screener de Finviz (si disponible)
        finviz_tickers = self.get_finviz_penny_stocks()
        all_tickers.update(finviz_tickers)
        print(f"   🌐 Finviz: {len(finviz_tickers)} tickers")
        
        # Fuente 3: Reddit mentions recientes
        reddit_tickers = self.get_reddit_mentioned_stocks()
        all_tickers.update(reddit_tickers)
        print(f"   🗨️  Reddit: {len(reddit_tickers)} tickers")
        
        # Fuente 4: Lista de stocks con delisting warnings
        delisting_tickers = self.get_delisting_candidates()
        all_tickers.update(delisting_tickers)
        print(f"   ⚠️  Delisting: {len(delisting_tickers)} tickers")
        
        final_list = list(all_tickers)
        print(f"✅ Total universo: {len(final_list)} tickers únicos")
        
        return final_list
    
    def get_manual_penny_stocks(self) -> List[str]:
        """Lista manual de penny stocks conocidos por alta volatilidad"""
        return [
            # Meme stocks clásicos
            'AMC', 'GME', 'BBBY', 'KOSS', 'EXPR', 'NAKD', 'CLOV', 'WISH',
            
            # Penny stocks con historial de squeezes
            'BBIG', 'MULN', 'SNDL', 'CTRM', 'SHIP', 'GNUS', 'XELA', 'PROG',
            'ATER', 'SPRT', 'IRNT', 'OPAD', 'RDBX', 'NILE', 'HMHC', 'CLVS',
            
            # Biotech penny stocks (alta volatilidad)
            'OBSV', 'ADTX', 'AGRX', 'AVIR', 'BIOC', 'BPTH', 'BVXV', 'CBAT',
            'CDXS', 'CHEK', 'COCP', 'CRIS', 'CTMX', 'CYCC', 'DFFN', 'DMAC',
            
            # Crypto/Tech penny stocks
            'EBON', 'EQOS', 'GREE', 'HIVE', 'MARA', 'RIOT', 'SOS', 'XNET',
            
            # Shipping/Energy penny stocks
            'CTRM', 'EURN', 'GLBS', 'GOGL', 'NAT', 'STNG', 'TNK', 'TRMD',
            
            # Añadir los que has mencionado
            'HTOO', 'LCFY', 'SIRI', 'XAIR'
        ]
    
    def get_finviz_penny_stocks(self) -> List[str]:
        """
        Obtiene penny stocks desde Finviz screener
        (Simulado - en producción usarías web scraping o API)
        """
        # En producción, aquí harías scraping de:
        # https://finviz.com/screener.ashx?v=111&f=cap_micro,sh_price_u10,sh_short_o10
        
        # Por ahora, simulamos con lista conocida
        simulated_finviz = [
            'AAOI', 'ABVC', 'ACST', 'ADIL', 'ADOM', 'ADTX', 'AEHR', 'AETI',
            'AGBA', 'AGRX', 'AIHS', 'AINV', 'AISP', 'AKBA', 'ALBT', 'ALCO',
            'ALGS', 'ALIM', 'ALLK', 'ALPP', 'ALVO', 'AMST', 'ANEB', 'ANIX',
            'ANTE', 'ANY', 'AOUT', 'APDN', 'APLM', 'APLS', 'APPH', 'APPN',
            'APVO', 'ARDX', 'AREC', 'ARQT', 'ARTL', 'ARTW', 'ARVL', 'ASLN',
            'ASRT', 'ASTC', 'ASUR', 'ATAK', 'ATCX', 'ATER', 'ATGL', 'ATHM',
            'ATLC', 'ATNX', 'ATOM', 'ATOS', 'ATXI', 'AUPH', 'AVGR', 'AVIR',
            'AVTE', 'AXDX', 'AXGN', 'AYTU', 'AZPN', 'BABA', 'BBCP', 'BBIG'
        ]
        
        return simulated_finviz[:30]  # Limitar para demo
    
    def get_reddit_mentioned_stocks(self) -> List[str]:
        """
        Obtiene stocks mencionados en Reddit (WSB, pennystocks, etc.)
        (Simulado - en producción usarías Reddit API)
        """
        # En producción, usarías praw (Python Reddit API Wrapper)
        # para buscar en r/wallstreetbets, r/pennystocks, etc.
        
        simulated_reddit = [
            'BBBY', 'MULN', 'SNDL', 'CTRM', 'SHIP', 'ATER', 'PROG', 'GNUS',
            'XELA', 'NILE', 'HMHC', 'RDBX', 'AVCT', 'TOPS', 'BIOC', 'CHEK'
        ]
        
        return simulated_reddit
    
    def get_delisting_candidates(self) -> List[str]:
        """
        Obtiene stocks con warnings de delisting o reverse split
        """
        # Lista de stocks que han tenido warnings recientes
        delisting_candidates = [
            'MULN', 'GNUS', 'XELA', 'NILE', 'HMHC', 'TOPS', 'SHIP', 'CTRM',
            'NAKD', 'CLVS', 'PROG', 'ATER', 'BBIG', 'RDBX', 'AVCT'
        ]
        
        return delisting_candidates
    
    def quick_filter_ticker(self, ticker: str) -> Tuple[bool, Dict]:
        """
        Filtro rápido inicial para descartar tickers obviamente malos
        
        Returns:
            (bool, dict): (pasa_filtro, datos_básicos)
        """
        try:
            # Obtener datos básicos
            stock = yf.Ticker(ticker)
            hist = stock.history(period='5d')
            info = stock.info
            
            if len(hist) < 2:
                return False, {'reason': 'Sin datos históricos'}
            
            current_price = hist['Close'].iloc[-1]
            volume = hist['Volume'].iloc[-1]
            
            # Filtros básicos
            basic_data = {
                'price': current_price,
                'volume': volume,
                'market_cap': info.get('marketCap', 0)
            }
            
            # Filtro 1: Precio en rango penny stock
            if not (self.screening_criteria['price_range']['min'] <= current_price <= self.screening_criteria['price_range']['max']):
                return False, {**basic_data, 'reason': f'Precio ${current_price:.3f} fuera de rango'}
            
            # Filtro 2: Volumen mínimo
            if volume < self.screening_criteria['volume_min']:
                return False, {**basic_data, 'reason': f'Volumen {volume:,} muy bajo'}
            
            # Filtro 3: Market cap máximo
            market_cap = info.get('marketCap', 0)
            if market_cap > self.screening_criteria['market_cap_max']:
                return False, {**basic_data, 'reason': f'Market cap ${market_cap:,} muy alto'}
            
            return True, basic_data
            
        except Exception as e:
            return False, {'reason': f'Error: {str(e)[:50]}'}
    
    def detailed_screening(self, ticker: str) -> Dict:
        """
        Análisis detallado de un ticker candidato
        """
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='1mo')
            info = stock.info
            
            if len(hist) < 20:
                return {'ticker': ticker, 'score': 0, 'reason': 'Datos insuficientes'}
            
            # Datos actuales
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2]
            current_volume = hist['Volume'].iloc[-1]
            avg_volume = hist['Volume'].tail(20).mean()
            
            # Cálculos técnicos
            price_change_pct = ((current_price - prev_close) / prev_close) * 100
            volume_ratio = current_volume / max(avg_volume, 1)
            
            # Short interest (simulado - en producción desde FINRA)
            short_interest = info.get('shortPercentOfFloat', 0.05) * 100
            
            # RSI aproximado
            closes = hist['Close'].values
            delta = np.diff(closes)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else np.mean(gain)
            avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else np.mean(loss)
            rs = avg_gain / max(avg_loss, 0.001)
            rsi = 100 - (100 / (1 + rs))
            
            # Scoring preliminar
            score = 0
            reasons = []
            
            # Criterio 1: Short Interest
            if short_interest >= 20:
                score += 0.3
                reasons.append(f"SI_{short_interest:.1f}%")
            elif short_interest >= 10:
                score += 0.2
                reasons.append(f"SI_{short_interest:.1f}%_ok")
            
            # Criterio 2: Volumen anómalo
            if volume_ratio >= 5:
                score += 0.3
                reasons.append(f"Vol_{volume_ratio:.1f}x")
            elif volume_ratio >= 2:
                score += 0.2
                reasons.append(f"Vol_{volume_ratio:.1f}x_ok")
            
            # Criterio 3: Precio y momentum
            if 0 < price_change_pct <= 20:
                score += 0.2
                reasons.append(f"Up_{price_change_pct:.1f}%")
            elif price_change_pct > 20:
                score += 0.1  # Posible gap up ya tardío
                reasons.append(f"Gap_{price_change_pct:.1f}%")
            
            # Criterio 4: RSI
            if 40 <= rsi <= 70:
                score += 0.1
                reasons.append(f"RSI_{rsi:.0f}")
            
            # Criterio 5: Precio bajo (penny stock)
            if current_price < 1:
                score += 0.1
                reasons.append("Penny")
            
            return {
                'ticker': ticker,
                'score': score,
                'price': current_price,
                'price_change_pct': price_change_pct,
                'volume_ratio': volume_ratio,
                'short_interest': short_interest,
                'rsi': rsi,
                'market_cap': info.get('marketCap', 0),
                'reasons': reasons,
                'data_quality': 'good' if len(hist) >= 20 else 'limited'
            }
            
        except Exception as e:
            return {
                'ticker': ticker,
                'score': 0,
                'error': str(e)[:100],
                'data_quality': 'error'
            }
    
    def screen_batch(self, tickers: List[str], max_workers: int = 10) -> List[Dict]:
        """
        Procesa lote de tickers en paralelo
        """
        print(f"🔄 Procesando {len(tickers)} tickers en paralelo...")
        
        results = []
        passed_initial = []
        
        # Fase 1: Filtro rápido inicial
        print("   📋 Fase 1: Filtro básico...")
        for ticker in tickers:
            passes, data = self.quick_filter_ticker(ticker)
            if passes:
                passed_initial.append(ticker)
            else:
                results.append({
                    'ticker': ticker,
                    'score': 0,
                    'filtered_out': True,
                    'reason': data.get('reason', 'Filtro básico')
                })
        
        print(f"   ✅ {len(passed_initial)}/{len(tickers)} pasaron filtro básico")
        
        # Fase 2: Análisis detallado en paralelo
        if passed_initial:
            print("   🔍 Fase 2: Análisis detallado...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_ticker = {
                    executor.submit(self.detailed_screening, ticker): ticker 
                    for ticker in passed_initial
                }
                
                for future in concurrent.futures.as_completed(future_to_ticker):
                    result = future.result()
                    results.append(result)
        
        # Ordenar por score
        results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return results
    
    def generate_screening_report(self, results: List[Dict]) -> Dict:
        """
        Genera reporte completo del screening
        """
        print("\n📊 REPORTE DE SCREENING")
        print("=" * 50)
        
        # Estadísticas
        total = len(results)
        high_score = len([r for r in results if r.get('score', 0) >= 0.7])
        medium_score = len([r for r in results if 0.4 <= r.get('score', 0) < 0.7])
        low_score = len([r for r in results if 0.1 <= r.get('score', 0) < 0.4])
        filtered_out = len([r for r in results if r.get('score', 0) == 0])
        
        print(f"📈 ESTADÍSTICAS:")
        print(f"   • Total procesados: {total}")
        print(f"   • Score alto (≥0.7): {high_score}")
        print(f"   • Score medio (0.4-0.7): {medium_score}")
        print(f"   • Score bajo (0.1-0.4): {low_score}")
        print(f"   • Filtrados: {filtered_out}")
        
        # Top candidatos
        top_candidates = [r for r in results if r.get('score', 0) >= 0.4][:10]
        
        print(f"\n🎯 TOP {len(top_candidates)} CANDIDATOS:")
        print("-" * 30)
        
        for i, candidate in enumerate(top_candidates, 1):
            ticker = candidate['ticker']
            score = candidate.get('score', 0)
            price = candidate.get('price', 0)
            change = candidate.get('price_change_pct', 0)
            vol_ratio = candidate.get('volume_ratio', 0)
            si = candidate.get('short_interest', 0)
            reasons = ', '.join(candidate.get('reasons', []))
            
            print(f"{i:2d}. {ticker:6s} Score: {score:.2f} | ${price:.3f} ({change:+5.1f}%) | Vol: {vol_ratio:.1f}x | SI: {si:.1f}%")
            if reasons:
                print(f"      Razones: {reasons}")
        
        # Candidatos recomendados para el robot advisor
        recommended = [r['ticker'] for r in results if r.get('score', 0) >= 0.5]
        
        print(f"\n🤖 RECOMENDADOS PARA ROBOT ADVISOR ({len(recommended)}):")
        print(f"   {', '.join(recommended[:20])}")  # Máximo 20
        
        return {
            'timestamp': datetime.now(),
            'statistics': {
                'total': total,
                'high_score': high_score,
                'medium_score': medium_score,
                'low_score': low_score,
                'filtered_out': filtered_out
            },
            'top_candidates': top_candidates,
            'recommended_tickers': recommended,
            'all_results': results
        }
    
    def save_results(self, report: Dict, filename: str = None):
        """Guarda resultados del screening"""
        if filename is None:
            filename = f"screening_results_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        
        # Preparar datos para JSON
        json_data = {
            'timestamp': report['timestamp'].isoformat(),
            'statistics': report['statistics'],
            'recommended_tickers': report['recommended_tickers'],
            'top_candidates': report['top_candidates'][:20],  # Solo top 20
            'screening_criteria': self.screening_criteria
        }
        
        with open(filename, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        print(f"💾 Resultados guardados en: {filename}")
        
        # También crear CSV simple para fácil importación
        csv_filename = filename.replace('.json', '.csv')
        candidates_df = pd.DataFrame(report['top_candidates'][:50])  # Top 50
        if not candidates_df.empty:
            candidates_df.to_csv(csv_filename, index=False)
            print(f"💾 CSV guardado en: {csv_filename}")
        
        return filename
    
    def run_full_screening(self):
        """Ejecuta screening completo"""
        print("🚀 INICIANDO SCREENING COMPLETO")
        print("=" * 40)
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 1. Obtener universo
        universe = self.get_penny_stock_universe()
        
        # 2. Procesar en lotes
        results = self.screen_batch(universe)
        
        # 3. Generar reporte
        report = self.generate_screening_report(results)
        
        # 4. Guardar resultados
        filename = self.save_results(report)
        
        print(f"\n✅ SCREENING COMPLETADO")
        print(f"🎯 {len(report['recommended_tickers'])} tickers recomendados")
        print(f"📄 Resultados en: {filename}")
        
        return report

def main():
    """Función principal"""
    try:
        # Verificar dependencias
        try:
            import yfinance
            print("✅ yfinance disponible")
        except ImportError:
            print("❌ yfinance requerido: pip install yfinance")
            return
        
        # Crear y ejecutar screener
        screener = TickerScreener()
        report = screener.run_full_screening()
        
        # Mostrar próximos pasos
        print(f"\n💡 PRÓXIMOS PASOS:")
        print(f"1. Revisa los candidatos recomendados")
        print(f"2. Añade los mejores a tu watchlist del robot advisor")
        print(f"3. Ejecuta el robot advisor con la nueva lista")
        print(f"4. Programa este screener para ejecutar semanalmente")
        
        return report
        
    except KeyboardInterrupt:
        print("\n⏹️ Screening interrumpido por usuario")
    except Exception as e:
        print(f"❌ Error durante screening: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()