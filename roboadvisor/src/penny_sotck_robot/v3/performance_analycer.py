#!/usr/bin/env python3
"""
PERFORMANCE ANALYZER - ANÁLISIS DE RECOMENDACIONES
==================================================

Analiza qué tan buenas fueron las recomendaciones del Robot Advisor.
Lee archivos trading_results_v3_*.json y compara vs precios actuales/históricos.

Uso:
    python performance_analyzer.py
"""

import json
import os
import glob
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
from typing import List, Dict, Tuple
import pandas as pd


class PerformanceAnalyzer:
    """
    Analiza performance de recomendaciones del Robot Advisor
    """
    
    def __init__(self):
        self.results_files = []
        self.analyses = []
        
        print("📊 Performance Analyzer inicializado")
    
    # ========================================================================
    # BÚSQUEDA Y CARGA DE ARCHIVOS
    # ========================================================================
    
    def find_results_files(self, directory: str = '.') -> List[str]:
        """
        Busca archivos de resultados en el directorio
        """
        pattern = os.path.join(directory, 'trading_results_v3_*.json')
        files = glob.glob(pattern)
        
        # Ordenar por fecha (más reciente primero)
        files.sort(reverse=True)
        
        self.results_files = files
        print(f"📁 Encontrados {len(files)} archivos de resultados")
        
        return files
    
    def load_results_file(self, filename: str) -> Dict:
        """
        Carga un archivo de resultados
        """
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            return data
        
        except Exception as e:
            print(f"❌ Error cargando {filename}: {e}")
            return None
    
    # ========================================================================
    # ANÁLISIS DE PRECIO HISTÓRICO
    # ========================================================================
    
    def get_historical_performance(self, symbol: str, entry_date: datetime, 
                                   entry_price: float, stop_loss: float,
                                   take_profits: List[float],
                                   days_to_analyze: int = 30) -> Dict:
        """
        Obtiene performance histórico de un símbolo desde fecha de recomendación
        
        Analiza:
        - ¿Se alcanzaron los TPs?
        - ¿Se disparó el stop?
        - Precio máximo alcanzado
        - Precio actual
        - Días hasta máximo
        """
        try:
            stock = yf.Ticker(symbol)
            
            # Obtener historial desde fecha de entrada
            start_date_dt = entry_date.date()
            # Para 'end', usamos la fecha de mañana para asegurar 
            # que se incluyan los datos de hoy (yfinance es exclusivo en 'end').
            end_date_dt = (datetime.now() + timedelta(days=1)).date()
            hist = stock.history(start=start_date_dt, end=end_date_dt)
            
            if len(hist) == 0:
                return {
                    'symbol': symbol,
                    'status': 'no_data',
                    'error': 'Sin datos históricos'
                }
            
            # Precio actual
            current_price = hist['Close'].iloc[-1]
            current_gain_pct = ((current_price - entry_price) / entry_price) * 100
            
            # Máximo alcanzado desde entrada
            max_price = hist['High'].max()
            max_gain_pct = ((max_price - entry_price) / entry_price) * 100
            days_to_max = (hist['High'].idxmax() - hist.index[0]).days if len(hist) > 0 else 0
            
            # Mínimo alcanzado
            min_price = hist['Low'].min()
            max_drawdown_pct = ((min_price - entry_price) / entry_price) * 100
            
            # ¿Se disparó stop loss?
            stop_hit = min_price <= stop_loss
            if stop_hit:
                # Encontrar cuándo
                stop_hit_date = None
                for date, row in hist.iterrows():
                    if row['Low'] <= stop_loss:
                        stop_hit_date = date
                        break
                days_to_stop = (stop_hit_date - hist.index[0]).days if stop_hit_date else 0
            else:
                days_to_stop = None
            
            # ¿Se alcanzaron TPs?
            tps_reached = []
            for i, tp_price in enumerate(take_profits, 1):
                tp_reached = any(hist['High'] >= tp_price)
                if tp_reached:
                    # Encontrar cuándo
                    tp_date = None
                    for date, row in hist.iterrows():
                        if row['High'] >= tp_price:
                            tp_date = date
                            break
                    days_to_tp = (tp_date - hist.index[0]).days if tp_date else 0
                    
                    tps_reached.append({
                        'level': i,
                        'price': tp_price,
                        'reached': True,
                        'date': tp_date.strftime('%Y-%m-%d') if tp_date else None,
                        'days': days_to_tp
                    })
                else:
                    tps_reached.append({
                        'level': i,
                        'price': tp_price,
                        'reached': False
                    })
            
            # Calcular ganancia realizable (simulando estrategia)
            if stop_hit:
                # Si se disparó stop, pérdida
                realized_gain_pct = ((stop_loss - entry_price) / entry_price) * 100
                exit_reason = 'STOP_LOSS'
            else:
                # Simular venta escalonada en TPs alcanzados
                total_shares = 100  # Normalizado
                shares_per_tp = total_shares // len(take_profits)
                remaining_shares = total_shares
                total_value = 0
                
                for tp_data in tps_reached:
                    if tp_data['reached']:
                        shares_to_sell = shares_per_tp
                        total_value += shares_to_sell * tp_data['price']
                        remaining_shares -= shares_to_sell
                
                # Acciones restantes al precio actual
                if remaining_shares > 0:
                    total_value += remaining_shares * current_price
                
                realized_gain_pct = ((total_value / total_shares - entry_price) / entry_price) * 100
                
                tps_reached_count = sum(1 for tp in tps_reached if tp['reached'])
                if tps_reached_count == len(take_profits):
                    exit_reason = 'ALL_TPS_HIT'
                elif tps_reached_count > 0:
                    exit_reason = f'{tps_reached_count}_TPS_HIT'
                else:
                    exit_reason = 'HOLDING'
            
            return {
                'symbol': symbol,
                'status': 'success',
                'entry_price': entry_price,
                'entry_date': entry_date.strftime('%Y-%m-%d'),
                'current_price': float(current_price),
                'current_gain_pct': current_gain_pct,
                'max_price': float(max_price),
                'max_gain_pct': max_gain_pct,
                'days_to_max': days_to_max,
                'min_price': float(min_price),
                'max_drawdown_pct': max_drawdown_pct,
                'stop_hit': stop_hit,
                'days_to_stop': days_to_stop,
                'tps_reached': tps_reached,
                'realized_gain_pct': realized_gain_pct,
                'exit_reason': exit_reason,
                'days_analyzed': len(hist)
            }
            
        except Exception as e:
            return {
                'symbol': symbol,
                'status': 'error',
                'error': str(e)
            }
    
    # ========================================================================
    # ANÁLISIS POR ARCHIVO
    # ========================================================================
    
    def analyze_file(self, filename: str) -> Dict:
        """
        Analiza un archivo de resultados completo
        """
        print(f"\n{'='*70}")
        print(f"📄 Analizando: {os.path.basename(filename)}")
        print(f"{'='*70}")
        
        # Cargar datos
        data = self.load_results_file(filename)
        
        if data is None:
            return None
        
        # Extraer información
        file_timestamp = data.get('timestamp', '')
        if isinstance(file_timestamp, str):
            try:
                file_date = datetime.fromisoformat(file_timestamp.replace('Z', '+00:00'))
            except:
                file_date = datetime.now() - timedelta(days=7)  # Default
        else:
            file_date = datetime.now() - timedelta(days=7)
        
        config_preset = data.get('config_preset', 'unknown')
        buy_signals = data.get('buy_signals', [])
        
        print(f"📅 Fecha: {file_date.strftime('%Y-%m-%d %H:%M')}")
        print(f"⚙️  Config: {config_preset.upper()}")
        print(f"📊 Señales de compra: {len(buy_signals)}")
        
        if not buy_signals:
            print("⚠️  No hay señales de compra en este archivo")
            return {
                'filename': filename,
                'file_date': file_date,
                'config': config_preset,
                'signals_count': 0,
                'analyses': []
            }
        
        # Analizar cada señal
        signal_analyses = []
        
        print(f"\n🔍 Analizando cada señal...")
        
        for i, signal in enumerate(buy_signals, 1):
            symbol = signal['symbol']
            entry_price = signal['price']
            stop_loss = signal['stop_loss']
            take_profits = signal['take_profits']
            is_urgent = signal.get('is_urgent_squeeze', False)
            
            print(f"\n{i}. {symbol} (Entrada: ${entry_price:.3f})")
            
            # Análisis histórico
            analysis = self.get_historical_performance(
                symbol, file_date, entry_price, 
                stop_loss, take_profits
            )
            
            # Añadir info adicional
            analysis['signal_data'] = {
                'urgency': signal.get('urgency', 'NORMAL'),
                'is_urgent_squeeze': is_urgent,
                'position_size_pct': signal.get('position_size', 2.0)
            }
            
            signal_analyses.append(analysis)
            
            # Mostrar resultado
            if analysis['status'] == 'success':
                current_gain = analysis['current_gain_pct']
                max_gain = analysis['max_gain_pct']
                realized_gain = analysis['realized_gain_pct']
                
                # Emoji según resultado
                if realized_gain >= 20:
                    emoji = "🟢"
                elif realized_gain >= 0:
                    emoji = "🟡"
                else:
                    emoji = "🔴"
                
                print(f"   {emoji} Actual: {current_gain:+.1f}%")
                print(f"   📈 Máximo: {max_gain:+.1f}% (día {analysis['days_to_max']})")
                print(f"   💰 Realizado: {realized_gain:+.1f}% ({analysis['exit_reason']})")
                
                if analysis['stop_hit']:
                    print(f"   🛑 Stop hit (día {analysis['days_to_stop']})")
                
                tps_hit = sum(1 for tp in analysis['tps_reached'] if tp['reached'])
                print(f"   🎯 TPs: {tps_hit}/{len(take_profits)} alcanzados")
            
            else:
                print(f"   ❌ {analysis.get('error', 'Error desconocido')}")
        
        return {
            'filename': filename,
            'file_date': file_date,
            'config': config_preset,
            'signals_count': len(buy_signals),
            'analyses': signal_analyses
        }
    
    # ========================================================================
    # ESTADÍSTICAS Y REPORTES
    # ========================================================================
    
    def generate_file_statistics(self, file_analysis: Dict) -> Dict:
        """
        Genera estadísticas de un archivo
        """
        if not file_analysis or file_analysis['signals_count'] == 0:
            return None
        
        analyses = file_analysis['analyses']
        successful = [a for a in analyses if a['status'] == 'success']
        
        if not successful:
            return None
        
        # Estadísticas básicas
        current_gains = [a['current_gain_pct'] for a in successful]
        realized_gains = [a['realized_gain_pct'] for a in successful]
        max_gains = [a['max_gain_pct'] for a in successful]
        
        # Win rate (basado en realized gains)
        winners = [g for g in realized_gains if g > 0]
        losers = [g for g in realized_gains if g <= 0]
        win_rate = (len(winners) / len(realized_gains)) * 100 if realized_gains else 0
        
        # Stops disparados
        stops_hit = sum(1 for a in successful if a.get('stop_hit', False))
        
        # TPs alcanzados
        all_tps_reached = []
        for a in successful:
            tps = a.get('tps_reached', [])
            for tp in tps:
                if tp['reached']:
                    all_tps_reached.append(tp['level'])
        
        # Urgentes vs normales
        urgent_signals = [a for a in successful if a['signal_data'].get('is_urgent_squeeze', False)]
        normal_signals = [a for a in successful if not a['signal_data'].get('is_urgent_squeeze', False)]
        
        urgent_avg = np.mean([a['realized_gain_pct'] for a in urgent_signals]) if urgent_signals else 0
        normal_avg = np.mean([a['realized_gain_pct'] for a in normal_signals]) if normal_signals else 0
        
        stats = {
            'total_signals': len(analyses),
            'successful_analyses': len(successful),
            'win_rate': win_rate,
            'avg_current_gain': np.mean(current_gains),
            'avg_realized_gain': np.mean(realized_gains),
            'avg_max_gain': np.mean(max_gains),
            'best_trade': max(realized_gains),
            'worst_trade': min(realized_gains),
            'stops_hit': stops_hit,
            'stops_hit_pct': (stops_hit / len(successful)) * 100,
            'avg_days_to_max': np.mean([a['days_to_max'] for a in successful]),
            'urgent_signals': len(urgent_signals),
            'urgent_avg_gain': urgent_avg,
            'normal_avg_gain': normal_avg,
            'tps_distribution': {
                'tp1': all_tps_reached.count(1),
                'tp2': all_tps_reached.count(2),
                'tp3': all_tps_reached.count(3)
            }
        }
        
        return stats
    
    def print_file_statistics(self, file_analysis: Dict, stats: Dict):
        """
        Imprime estadísticas de un archivo
        """
        print(f"\n{'='*70}")
        print(f"📊 ESTADÍSTICAS DEL ARCHIVO")
        print(f"{'='*70}")
        
        print(f"\n📈 PERFORMANCE:")
        print(f"   • Win Rate: {stats['win_rate']:.1f}%")
        print(f"   • Ganancia promedio (realizada): {stats['avg_realized_gain']:+.1f}%")
        print(f"   • Ganancia promedio (actual): {stats['avg_current_gain']:+.1f}%")
        print(f"   • Máximo promedio alcanzado: {stats['avg_max_gain']:+.1f}%")
        print(f"   • Mejor trade: {stats['best_trade']:+.1f}%")
        print(f"   • Peor trade: {stats['worst_trade']:+.1f}%")
        
        print(f"\n⏱️  TIMING:")
        print(f"   • Días promedio hasta máximo: {stats['avg_days_to_max']:.1f}")
        
        print(f"\n🎯 TAKE PROFITS:")
        tps = stats['tps_distribution']
        total_tps = sum(tps.values())
        print(f"   • TP1 alcanzado: {tps['tp1']} veces")
        print(f"   • TP2 alcanzado: {tps['tp2']} veces")
        print(f"   • TP3 alcanzado: {tps['tp3']} veces")
        
        print(f"\n🛑 STOPS:")
        print(f"   • Stops disparados: {stats['stops_hit']} ({stats['stops_hit_pct']:.1f}%)")
        
        if stats['urgent_signals'] > 0:
            print(f"\n🚨 SQUEEZES URGENTES vs NORMALES:")
            print(f"   • Urgentes: {stats['urgent_signals']} señales, {stats['urgent_avg_gain']:+.1f}% avg")
            print(f"   • Normales: {stats['successful_analyses'] - stats['urgent_signals']} señales, {stats['normal_avg_gain']:+.1f}% avg")
    
    def generate_summary_report(self, all_analyses: List[Dict]) -> Dict:
        """
        Genera reporte resumen de todos los archivos
        """
        print(f"\n{'='*70}")
        print(f"📊 REPORTE RESUMEN - TODOS LOS ARCHIVOS")
        print(f"{'='*70}")
        
        # Filtrar archivos válidos
        valid_analyses = [a for a in all_analyses if a and a['signals_count'] > 0]
        
        if not valid_analyses:
            print("⚠️  No hay datos para analizar")
            return None
        
        print(f"\n📁 Archivos analizados: {len(valid_analyses)}")
        
        # Estadísticas de cada archivo
        all_stats = []
        for analysis in valid_analyses:
            stats = self.generate_file_statistics(analysis)
            if stats:
                all_stats.append({
                    'filename': os.path.basename(analysis['filename']),
                    'date': analysis['file_date'],
                    'config': analysis['config'],
                    **stats
                })
        
        if not all_stats:
            print("⚠️  No se pudieron calcular estadísticas")
            return None
        
        # Ordenar por ganancia promedio
        all_stats.sort(key=lambda x: x['avg_realized_gain'], reverse=True)
        
        # Estadísticas agregadas
        total_signals = sum(s['total_signals'] for s in all_stats)
        overall_win_rate = np.mean([s['win_rate'] for s in all_stats])
        overall_avg_gain = np.mean([s['avg_realized_gain'] for s in all_stats])
        
        print(f"\n📊 ESTADÍSTICAS GLOBALES:")
        print(f"   • Total señales analizadas: {total_signals}")
        print(f"   • Win rate promedio: {overall_win_rate:.1f}%")
        print(f"   • Ganancia promedio: {overall_avg_gain:+.1f}%")
        
        # Top 5 mejores días
        print(f"\n🏆 TOP 5 MEJORES DÍAS:")
        for i, stats in enumerate(all_stats[:5], 1):
            date_str = stats['date'].strftime('%Y-%m-%d')
            print(f"   {i}. {date_str} ({stats['config']})")
            print(f"      • {stats['total_signals']} señales | Win: {stats['win_rate']:.0f}% | Avg: {stats['avg_realized_gain']:+.1f}%")
        
        # Bottom 5 peores días
        if len(all_stats) > 5:
            print(f"\n📉 DÍAS CON PEOR PERFORMANCE:")
            for i, stats in enumerate(all_stats[-5:][::-1], 1):
                date_str = stats['date'].strftime('%Y-%m-%d')
                print(f"   {i}. {date_str} ({stats['config']})")
                print(f"      • {stats['total_signals']} señales | Win: {stats['win_rate']:.0f}% | Avg: {stats['avg_realized_gain']:+.1f}%")
        
        # Comparación de configuraciones
        configs = {}
        for stats in all_stats:
            config = stats['config']
            if config not in configs:
                configs[config] = []
            configs[config].append(stats['avg_realized_gain'])
        
        if len(configs) > 1:
            print(f"\n⚙️  COMPARACIÓN DE CONFIGURACIONES:")
            for config, gains in configs.items():
                avg = np.mean(gains)
                print(f"   • {config.upper()}: {avg:+.1f}% promedio ({len(gains)} archivos)")
        
        return {
            'total_files': len(valid_analyses),
            'total_signals': total_signals,
            'overall_win_rate': overall_win_rate,
            'overall_avg_gain': overall_avg_gain,
            'all_stats': all_stats
        }
    
    def save_results_to_csv(self, all_analyses: List[Dict], filename: str = 'performance_analysis.csv'):
        """
        Guarda resultados en CSV para análisis en Excel
        """
        rows = []
        
        for file_analysis in all_analyses:
            if not file_analysis or file_analysis['signals_count'] == 0:
                continue
            
            file_date = file_analysis['file_date'].strftime('%Y-%m-%d')
            config = file_analysis['config']
            
            for analysis in file_analysis['analyses']:
                if analysis['status'] != 'success':
                    continue
                
                row = {
                    'file_date': file_date,
                    'config': config,
                    'symbol': analysis['symbol'],
                    'entry_price': analysis['entry_price'],
                    'current_price': analysis['current_price'],
                    'current_gain_pct': analysis['current_gain_pct'],
                    'max_price': analysis['max_price'],
                    'max_gain_pct': analysis['max_gain_pct'],
                    'realized_gain_pct': analysis['realized_gain_pct'],
                    'days_to_max': analysis['days_to_max'],
                    'stop_hit': analysis['stop_hit'],
                    'exit_reason': analysis['exit_reason'],
                    'is_urgent_squeeze': analysis['signal_data']['is_urgent_squeeze'],
                    'tp1_reached': analysis['tps_reached'][0]['reached'] if len(analysis['tps_reached']) > 0 else False,
                    'tp2_reached': analysis['tps_reached'][1]['reached'] if len(analysis['tps_reached']) > 1 else False,
                    'tp3_reached': analysis['tps_reached'][2]['reached'] if len(analysis['tps_reached']) > 2 else False,
                }
                
                rows.append(row)
        
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(filename, index=False)
            print(f"\n💾 Resultados guardados en: {filename}")
            print(f"   {len(rows)} trades analizados")
        else:
            print("\n⚠️  No hay datos para guardar")
    
    # ========================================================================
    # FLUJO PRINCIPAL
    # ========================================================================
    
    def run_full_analysis(self, directory: str = '.'):
        """
        Ejecuta análisis completo de todos los archivos
        """
        print("\n" + "="*70)
        print("🚀 PERFORMANCE ANALYZER - ANÁLISIS COMPLETO")
        print("="*70)
        
        # Buscar archivos
        files = self.find_results_files(directory)
        
        if not files:
            print(f"\n❌ No se encontraron archivos trading_results_v3_*.json en {directory}")
            return
        
        print(f"\n📋 Se analizarán {len(files)} archivos\n")
        
        input("Presiona Enter para continuar...")
        
        # Analizar cada archivo
        all_analyses = []
        
        for i, filename in enumerate(files, 1):
            print(f"\n{'#'*70}")
            print(f"# ARCHIVO {i}/{len(files)}")
            print(f"{'#'*70}")
            
            analysis = self.analyze_file(filename)
            
            if analysis and analysis['signals_count'] > 0:
                # Generar y mostrar estadísticas
                stats = self.generate_file_statistics(analysis)
                if stats:
                    self.print_file_statistics(analysis, stats)
            
            all_analyses.append(analysis)
            
            # Pausa entre archivos (excepto el último)
            if i < len(files):
                print("\n" + "-"*70)
                input("Presiona Enter para siguiente archivo...")
        
        # Reporte resumen
        summary = self.generate_summary_report(all_analyses)
        
        # Guardar CSV
        self.save_results_to_csv(all_analyses)
        
        print(f"\n{'='*70}")
        print("✅ ANÁLISIS COMPLETADO")
        print(f"{'='*70}")
        
        return {
            'all_analyses': all_analyses,
            'summary': summary
        }


# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def analyze_specific_file(filename: str):
    """
    Analiza un archivo específico
    """
    analyzer = PerformanceAnalyzer()
    
    if not os.path.exists(filename):
        print(f"❌ Archivo no encontrado: {filename}")
        return
    
    analysis = analyzer.analyze_file(filename)
    
    if analysis and analysis['signals_count'] > 0:
        stats = analyzer.generate_file_statistics(analysis)
        if stats:
            analyzer.print_file_statistics(analysis, stats)


def compare_configs():
    """
    Compara performance de diferentes configuraciones
    """
    print("\n" + "="*70)
    print("⚙️  COMPARACIÓN DE CONFIGURACIONES")
    print("="*70)
    
    analyzer = PerformanceAnalyzer()
    files = analyzer.find_results_files()
    
    if not files:
        print("❌ No se encontraron archivos")
        return
    
    # Agrupar por configuración
    configs = {}
    
    for filename in files:
        data = analyzer.load_results_file(filename)
        if not data:
            continue
        
        config = data.get('config_preset', 'unknown')
        
        if config not in configs:
            configs[config] = []
        
        configs[config].append(filename)
    
    print(f"\n📊 Configuraciones encontradas:")
    for config, files_list in configs.items():
        print(f"   • {config.upper()}: {len(files_list)} archivos")
    
    print("\n🔬 Analizando cada configuración...\n")
    
    results_by_config = {}
    
    for config, files_list in configs.items():
        print(f"\n{'='*70}")
        print(f"⚙️  CONFIGURACIÓN: {config.upper()}")
        print(f"{'='*70}")
        
        all_gains = []
        all_win_rates = []
        
        for filename in files_list:
            analysis = analyzer.analyze_file(filename)
            
            if analysis and analysis['signals_count'] > 0:
                stats = analyzer.generate_file_statistics(analysis)
                if stats:
                    all_gains.append(stats['avg_realized_gain'])
                    all_win_rates.append(stats['win_rate'])
        
        if all_gains:
            results_by_config[config] = {
                'avg_gain': np.mean(all_gains),
                'avg_win_rate': np.mean(all_win_rates),
                'best_day': max(all_gains),
                'worst_day': min(all_gains),
                'files_count': len(files_list)
            }
            
            print(f"\n   📊 Resultados:")
            print(f"      • Ganancia promedio: {results_by_config[config]['avg_gain']:+.1f}%")
            print(f"      • Win rate promedio: {results_by_config[config]['avg_win_rate']:.1f}%")
            print(f"      • Mejor día: {results_by_config[config]['best_day']:+.1f}%")
            print(f"      • Peor día: {results_by_config[config]['worst_day']:+.1f}%")
    
    # Ranking
    if len(results_by_config) > 1:
        print(f"\n{'='*70}")
        print(f"🏆 RANKING DE CONFIGURACIONES")
        print(f"{'='*70}")
        
        sorted_configs = sorted(results_by_config.items(), 
                              key=lambda x: x[1]['avg_gain'], 
                              reverse=True)
        
        for i, (config, results) in enumerate(sorted_configs, 1):
            print(f"\n{i}. {config.upper()}")
            print(f"   Ganancia: {results['avg_gain']:+.1f}% | Win rate: {results['avg_win_rate']:.1f}%")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Función principal"""
    print("\n" + "="*70)
    print("📊 PERFORMANCE ANALYZER")
    print("="*70)
    
    print("\n¿Qué quieres hacer?")
    print("1. Analizar todos los archivos (reporte completo)")
    print("2. Analizar archivo específico")
    print("3. Comparar configuraciones (balanced vs aggressive)")
    print("4. Salir")
    
    choice = input("\nElegir (1-4): ").strip()
    
    if choice == '1':
        analyzer = PerformanceAnalyzer()
        analyzer.run_full_analysis()
    
    elif choice == '2':
        filename = input("Nombre del archivo (ej: trading_results_v3_20251021_1500.json): ").strip()
        analyze_specific_file(filename)
    
    elif choice == '3':
        compare_configs()
    
    elif choice == '4':
        print("\n👋 ¡Hasta luego!")
        return
    
    else:
        print("\n⚠️ Opción inválida")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Análisis interrumpido")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()