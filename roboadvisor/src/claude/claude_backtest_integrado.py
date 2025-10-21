"""
Sistema de Backtesting Integrado - Algoritmo Deductivo vs S&P 500
=================================================================

Combina la mecánica probada del sistema simple con el algoritmo deductivo completo.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import os

warnings.filterwarnings('ignore')

# Importar el sistema deductivo
try:
    from claude_v3 import DeductiveTradingSystem
except ImportError:
    print("ERROR: No se puede importar DeductiveTradingSystem desde claude_v3.py")
    exit(1)


class IntegratedBacktester:
    """
    Backtester que integra el sistema deductivo con mecánica probada.
    """
    
    def __init__(self, initial_capital=100000, max_positions=10, 
                 transaction_cost=0.001, use_deductive=True):
        """
        Args:
            initial_capital: Capital inicial
            max_positions: Número máximo de posiciones
            transaction_cost: Costo por transacción
            use_deductive: Si usar el sistema deductivo o estrategia simple
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.max_positions = max_positions
        self.transaction_cost = transaction_cost
        self.use_deductive = use_deductive
        
        # Sistema deductivo
        self.trading_system = DeductiveTradingSystem(
            lookback_days=30,
            prediction_days=5,
            threshold_return=0.01  # 1% más realista
        )
        
        # Historial
        self.portfolio_history = []
        self.trades = []
        self.daily_signals_history = []
        
    def get_sp500_tickers(self, limit=50):
        """Obtiene tickers del S&P 500."""
        try:
            tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            sp500_table = tables[0]
            tickers = sp500_table['Symbol'].tolist()
            tickers = [ticker.replace('.', '-') for ticker in tickers]
            return tickers[:limit]
        except:
            # Lista de respaldo
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
                   'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS',
                   'BAC', 'XOM', 'CVX', 'KO', 'PEP'][:limit]
    
    def download_all_data(self, tickers, start_date, end_date):
        """Descarga datos históricos de todos los tickers."""
        print(f"\nDescargando datos para {len(tickers)} tickers...")
        data = {}
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {}
            for ticker in tickers:
                future = executor.submit(self._download_ticker_data, ticker, start_date, end_date)
                futures[future] = ticker
            
            for future in tqdm(as_completed(futures), total=len(tickers), desc="Descargando"):
                ticker = futures[future]
                try:
                    df = future.result()
                    if df is not None and not df.empty:
                        data[ticker] = df
                except:
                    continue
        
        print(f"Datos descargados exitosamente para {len(data)} tickers")
        return data
    
    def _download_ticker_data(self, ticker, start_date, end_date):
        """Descarga datos de un ticker individual."""
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if df.empty:
                return None
            return df
        except:
            return None
    
    def train_annual_models(self, year, tickers, data):
        """Entrena modelos deductivos anualmente."""
        if not self.use_deductive:
            return {}
            
        print(f"\n=== Entrenando modelos para {year} ===")
        
        # Período de entrenamiento: 2 años anteriores
        train_end = datetime(year - 1, 12, 31)
        train_start = train_end - timedelta(days=730)
        
        trained_models = {}
        successful = 0
        failed = 0
        
        for ticker in tqdm(tickers, desc=f"Entrenando modelos {year}"):
            if ticker not in data:
                continue
                
            try:
                # Usar el método del sistema deductivo
                df = self.trading_system.download_stock_data(ticker, train_start, train_end)
                
                if df is None or len(df) < 100:
                    failed += 1
                    continue
                
                # Entrenar modelo
                result = self.trading_system.train_model(ticker, df)
                
                if result is not None and ticker in self.trading_system.models:
                    trained_models[ticker] = {
                        'accuracy': result['accuracy'],
                        'features': result['feature_importance'].head(5).to_dict('records')
                    }
                    successful += 1
                else:
                    failed += 1
                    
            except Exception as e:
                failed += 1
                continue
        
        print(f"Modelos entrenados: {successful} exitosos, {failed} fallidos")
        
        # Guardar modelos para uso posterior
        if trained_models:
            with open(f'models_{year}.pkl', 'wb') as f:
                pickle.dump({
                    'models': self.trading_system.models,
                    'scalers': self.trading_system.scalers,
                    'features': self.trading_system.feature_names_by_ticker
                }, f)
        
        return trained_models
    
    def generate_signals_deductive(self, date, tickers, data):
        """Genera señales usando el sistema deductivo."""
        signals = {}
        
        for ticker in tickers:
            if ticker not in self.trading_system.models or ticker not in data:
                continue
                
            try:
                # Obtener datos recientes para el ticker
                ticker_data = data[ticker]
                if date not in ticker_data.index:
                    continue
                
                # Preparar datos para predicción
                lookback_start = date - timedelta(days=60)
                df = self.trading_system.download_stock_data(ticker, lookback_start, date)
                
                if df is None or len(df) < 30:
                    continue
                
                # Generar predicción
                prediction = self.trading_system.predict_signal(ticker, df)
                
                if prediction:
                    current_price = float(ticker_data.loc[date, 'Close'])
                    signals[ticker] = {
                        'signal': prediction['signal'],
                        'confidence': prediction['confidence'],
                        'price': current_price,
                        'prediction_value': prediction['prediction_value']
                    }
                    
            except Exception as e:
                continue
        
        return signals
    
    def generate_signals_simple(self, date, tickers, data):
        """Genera señales usando estrategia simple de media móvil."""
        signals = {}
        
        for ticker in tickers:
            if ticker not in data or date not in data[ticker].index:
                continue
                
            try:
                # Obtener histórico hasta la fecha
                hist = data[ticker][data[ticker].index <= date].tail(50)
                
                if len(hist) < 20:
                    continue
                
                current_price = float(hist['Close'].iloc[-1])
                ma20 = float(hist['Close'].rolling(20).mean().iloc[-1])
                
                if pd.isna(current_price) or pd.isna(ma20):
                    continue
                
                # Generar señal
                if current_price < ma20 * 0.95:
                    signal = 'COMPRAR'
                    confidence = min((ma20 - current_price) / ma20, 0.1) / 0.1
                elif current_price > ma20 * 1.05:
                    signal = 'VENDER'
                    confidence = min((current_price - ma20) / ma20, 0.1) / 0.1
                else:
                    continue
                
                signals[ticker] = {
                    'signal': signal,
                    'confidence': confidence,
                    'price': current_price,
                    'ma20': ma20
                }
                
            except:
                continue
        
        return signals
    
    def execute_trades(self, date, signals, data):
        """Ejecuta las operaciones basadas en las señales."""
        executed_trades = []
        
        # PASO 1: Vender posiciones con señal de venta
        for ticker in list(self.positions.keys()):
            if ticker in signals and signals[ticker]['signal'] == 'VENDER':
                price = float(signals[ticker]['price'])
                shares = self.positions[ticker]['shares']
                value = shares * price * (1 - self.transaction_cost)
                
                self.cash += value
                
                # Calcular ganancia/pérdida
                cost_basis = self.positions[ticker]['cost_basis']
                profit = value - cost_basis
                profit_pct = (profit / cost_basis) * 100
                
                executed_trades.append({
                    'date': date,
                    'ticker': ticker,
                    'action': 'SELL',
                    'shares': shares,
                    'price': price,
                    'value': value,
                    'profit': profit,
                    'profit_pct': profit_pct
                })
                
                del self.positions[ticker]
        
        # PASO 2: Comprar nuevas posiciones
        if len(self.positions) < self.max_positions:
            # Filtrar señales de compra
            buy_signals = [(t, s) for t, s in signals.items() 
                          if s['signal'] == 'COMPRAR' and t not in self.positions]
            
            # Ordenar por confianza
            buy_signals.sort(key=lambda x: x[1]['confidence'], reverse=True)
            
            # Calcular tamaño de posición
            available_slots = self.max_positions - len(self.positions)
            if available_slots > 0 and self.cash > 1000:  # Mínimo $1000 para operar
                position_size = min(self.cash / available_slots, self.cash * 0.95)
                
                for ticker, signal in buy_signals[:available_slots]:
                    price = float(signal['price'])
                    cost = position_size * (1 + self.transaction_cost)
                    
                    if cost > self.cash:
                        continue
                    
                    shares = int(position_size / price)
                    if shares == 0:
                        continue
                    
                    actual_cost = shares * price * (1 + self.transaction_cost)
                    
                    if actual_cost <= self.cash:
                        self.cash -= actual_cost
                        
                        self.positions[ticker] = {
                            'shares': shares,
                            'cost_basis': actual_cost,
                            'entry_price': price,
                            'entry_date': date
                        }
                        
                        executed_trades.append({
                            'date': date,
                            'ticker': ticker,
                            'action': 'BUY',
                            'shares': shares,
                            'price': price,
                            'value': actual_cost,
                            'confidence': signal['confidence']
                        })
        
        return executed_trades
    
    def calculate_portfolio_value(self, date, data):
        """Calcula el valor total del portafolio."""
        value = self.cash
        
        for ticker, position in self.positions.items():
            if ticker in data and date in data[ticker].index:
                current_price = float(data[ticker].loc[date, 'Close'])
                value += position['shares'] * current_price
        
        return value
    
    def run_backtest(self, start_date, end_date):
        """Ejecuta el backtest completo."""
        print("\n=== INICIANDO BACKTEST INTEGRADO ===")
        print(f"Período: {start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')}")
        print(f"Capital inicial: ${self.initial_capital:,.2f}")
        print(f"Estrategia: {'Deductiva' if self.use_deductive else 'Media Móvil Simple'}")
        
        # Obtener tickers
        tickers = self.get_sp500_tickers(limit=30)  # Empezar con 30 para pruebas
        
        # Descargar datos
        # Necesitamos datos extra para entrenar modelos
        data_start = start_date - timedelta(days=800) if self.use_deductive else start_date - timedelta(days=50)
        all_data = self.download_all_data(tickers, data_start, end_date)
        
        # Descargar S&P 500 para comparación
        print("\nDescargando S&P 500...")
        sp500 = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
        
        # Obtener fechas de trading
        trading_dates = []
        for ticker_data in all_data.values():
            dates_in_range = ticker_data.index[(ticker_data.index >= start_date) & 
                                              (ticker_data.index <= end_date)]
            trading_dates.extend(dates_in_range.tolist())
        trading_dates = sorted(list(set(trading_dates)))
        
        print(f"Días de trading: {len(trading_dates)}")
        
        # Variables para el backtest
        current_year = None
        trained_models = None
        sp500_initial = float(sp500['Close'].iloc[0]) if not sp500.empty else 100
        
        # Loop principal
        for i, date in enumerate(tqdm(trading_dates, desc="Simulando")):
            # Entrenar modelos anualmente (solo para sistema deductivo)
            if self.use_deductive and date.year != current_year:
                current_year = date.year
                trained_models = self.train_annual_models(current_year, tickers, all_data)
            
            # Generar señales
            if self.use_deductive and trained_models:
                signals = self.generate_signals_deductive(date, tickers, all_data)
            else:
                signals = self.generate_signals_simple(date, tickers, all_data)
            
            # Debug mensual
            if date.day <= 3 and signals:
                buy_count = sum(1 for s in signals.values() if s['signal'] == 'COMPRAR')
                sell_count = sum(1 for s in signals.values() if s['signal'] == 'VENDER')
                print(f"\n{date.strftime('%Y-%m')}: {buy_count} compras, {sell_count} ventas")
            
            # Ejecutar operaciones
            trades = self.execute_trades(date, signals, all_data)
            self.trades.extend(trades)
            
            # Guardar historial de señales
            if signals:
                self.daily_signals_history.append({
                    'date': date,
                    'n_buy_signals': sum(1 for s in signals.values() if s['signal'] == 'COMPRAR'),
                    'n_sell_signals': sum(1 for s in signals.values() if s['signal'] == 'VENDER'),
                    'total_signals': len(signals)
                })
            
            # Calcular y guardar valor del portafolio
            portfolio_value = self.calculate_portfolio_value(date, all_data)
            
            # Valor del S&P 500
            sp500_value = float(sp500.loc[date, 'Close']) if date in sp500.index else sp500_initial
            
            self.portfolio_history.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': self.cash,
                'n_positions': len(self.positions),
                'portfolio_return': ((portfolio_value / self.initial_capital) - 1) * 100,
                'sp500_value': sp500_value,
                'sp500_return': ((sp500_value / sp500_initial) - 1) * 100
            })
        
        # Convertir a DataFrames
        self.portfolio_df = pd.DataFrame(self.portfolio_history)
        self.trades_df = pd.DataFrame(self.trades)
        self.signals_df = pd.DataFrame(self.daily_signals_history)
        
        return self.portfolio_df, self.trades_df
    
    def analyze_results(self):
        """Analiza y muestra los resultados del backtest."""
        if self.portfolio_df.empty:
            print("No hay resultados para analizar")
            return
        
        # Métricas básicas
        initial_value = self.initial_capital
        final_value = self.portfolio_df['portfolio_value'].iloc[-1]
        total_return = self.portfolio_df['portfolio_return'].iloc[-1]
        sp500_return = self.portfolio_df['sp500_return'].iloc[-1]
        alpha = total_return - sp500_return
        
        # Calcular métricas adicionales
        returns = self.portfolio_df['portfolio_value'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100
        
        # Sharpe ratio
        risk_free_rate = 0.02
        excess_returns = returns.mean() * 252 - risk_free_rate
        sharpe_ratio = excess_returns / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        print("\n=== RESULTADOS DEL BACKTEST ===")
        print(f"\nCapital inicial: ${initial_value:,.2f}")
        print(f"Valor final: ${final_value:,.2f}")
        print(f"\nRentabilidad:")
        print(f"  Portafolio: {total_return:.2f}%")
        print(f"  S&P 500: {sp500_return:.2f}%")
        print(f"  Alpha: {alpha:.2f}%")
        print(f"\nRiesgo:")
        print(f"  Volatilidad anual: {volatility:.2f}%")
        print(f"  Máximo drawdown: {max_drawdown:.2f}%")
        print(f"  Sharpe ratio: {sharpe_ratio:.2f}")
        
        if not self.trades_df.empty:
            print(f"\nActividad de trading:")
            print(f"  Total operaciones: {len(self.trades_df)}")
            print(f"  Compras: {len(self.trades_df[self.trades_df['action'] == 'BUY'])}")
            print(f"  Ventas: {len(self.trades_df[self.trades_df['action'] == 'SELL'])}")
            
            # Análisis de trades ganadores
            sells = self.trades_df[self.trades_df['action'] == 'SELL']
            if not sells.empty:
                winners = sells[sells['profit'] > 0]
                win_rate = len(winners) / len(sells) * 100
                avg_win = winners['profit_pct'].mean() if not winners.empty else 0
                avg_loss = sells[sells['profit'] < 0]['profit_pct'].mean() if len(sells[sells['profit'] < 0]) > 0 else 0
                
                print(f"\nAnálisis de operaciones:")
                print(f"  Tasa de acierto: {win_rate:.1f}%")
                print(f"  Ganancia promedio: {avg_win:.1f}%")
                print(f"  Pérdida promedio: {avg_loss:.1f}%")
        
        # Información sobre señales
        if not self.signals_df.empty:
            print(f"\nGeneración de señales:")
            print(f"  Promedio diario de señales de compra: {self.signals_df['n_buy_signals'].mean():.1f}")
            print(f"  Promedio diario de señales de venta: {self.signals_df['n_sell_signals'].mean():.1f}")
    
    def plot_results(self):
        """Genera visualizaciones de los resultados."""
        fig, axes = plt.subplots(4, 1, figsize=(12, 14))
        
        # 1. Rentabilidad acumulada
        ax1 = axes[0]
        ax1.plot(self.portfolio_df['date'], self.portfolio_df['portfolio_return'], 
                label='Portafolio', linewidth=2, color='green')
        ax1.plot(self.portfolio_df['date'], self.portfolio_df['sp500_return'], 
                label='S&P 500', linewidth=2, color='blue', alpha=0.7)
        ax1.set_title('Rentabilidad Acumulada: Portafolio vs S&P 500', fontsize=14)
        ax1.set_ylabel('Rentabilidad (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Número de posiciones
        ax2 = axes[1]
        ax2.fill_between(self.portfolio_df['date'], 0, self.portfolio_df['n_positions'], 
                        alpha=0.5, color='orange')
        ax2.plot(self.portfolio_df['date'], self.portfolio_df['n_positions'], 
                color='darkorange', linewidth=2)
        ax2.set_title('Número de Posiciones en Cartera', fontsize=14)
        ax2.set_ylabel('Posiciones')
        ax2.set_ylim(0, self.max_positions + 1)
        ax2.grid(True, alpha=0.3)
        
        # 3. Cash vs Invertido
        ax3 = axes[2]
        invested = self.portfolio_df['portfolio_value'] - self.portfolio_df['cash']
        ax3.stackplot(self.portfolio_df['date'], 
                     self.portfolio_df['cash'], invested,
                     labels=['Cash', 'Invertido'],
                     colors=['lightblue', 'darkblue'],
                     alpha=0.7)
        ax3.set_title('Distribución del Capital', fontsize=14)
        ax3.set_ylabel('Valor ($)')
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # 4. Señales diarias
        if not self.signals_df.empty:
            ax4 = axes[3]
            ax4.bar(self.signals_df['date'], self.signals_df['n_buy_signals'], 
                   label='Compra', color='green', alpha=0.7)
            ax4.bar(self.signals_df['date'], -self.signals_df['n_sell_signals'], 
                   label='Venta', color='red', alpha=0.7)
            ax4.set_title('Señales de Trading Diarias', fontsize=14)
            ax4.set_ylabel('Número de Señales')
            ax4.set_xlabel('Fecha')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('integrated_backtest_results.png', dpi=300)
        plt.show()


def main():
    """Función principal para ejecutar el backtest integrado."""
    
    # Configuración
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    # Primero probar con estrategia simple para verificar
    print("\n=== FASE 1: Probando con estrategia simple ===")
    simple_backtester = IntegratedBacktester(
        initial_capital=100000,
        max_positions=10,
        transaction_cost=0.001,
        use_deductive=False  # Estrategia simple
    )
    
    portfolio_df_simple, trades_df_simple = simple_backtester.run_backtest(start_date, end_date)
    simple_backtester.analyze_results()
    
    # Luego probar con sistema deductivo
    print("\n\n=== FASE 2: Probando con sistema deductivo ===")
    deductive_backtester = IntegratedBacktester(
        initial_capital=100000,
        max_positions=10,
        transaction_cost=0.001,
        use_deductive=True  # Sistema deductivo
    )
    
    portfolio_df_deductive, trades_df_deductive = deductive_backtester.run_backtest(start_date, end_date)
    deductive_backtester.analyze_results()
    
    # Comparar resultados
    print("\n\n=== COMPARACIÓN DE ESTRATEGIAS ===")
    print(f"Retorno Estrategia Simple: {portfolio_df_simple['portfolio_return'].iloc[-1]:.2f}%")
    print(f"Retorno Sistema Deductivo: {portfolio_df_deductive['portfolio_return'].iloc[-1]:.2f}%")
    print(f"Retorno S&P 500: {portfolio_df_simple['sp500_return'].iloc[-1]:.2f}%")
    
    # Guardar resultados
    portfolio_df_simple.to_csv('backtest_simple_portfolio.csv', index=False)
    portfolio_df_deductive.to_csv('backtest_deductive_portfolio.csv', index=False)
    trades_df_deductive.to_csv('backtest_deductive_trades.csv', index=False)
    
    # Visualizar resultados del sistema deductivo
    deductive_backtester.plot_results()
    
    return simple_backtester, deductive_backtester


if __name__ == "__main__":
    simple_bt, deductive_bt = main()