"""
Sistema de Backtesting Completo - Algoritmo Deductivo vs S&P 500
================================================================

Este sistema evalúa si el algoritmo deductivo de valoración habría superado
al S&P 500 mediante una simulación histórica con las siguientes reglas:
- Cartera de 0-10 acciones
- Rebalanceo diario basado en señales deductivas
- Entrenamiento anual de modelos
- Comparación con benchmark S&P 500
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

warnings.filterwarnings('ignore')

# Importar el sistema de trading deductivo
# Asegúrate de que el archivo claude_v3.py esté en el mismo directorio
try:
    from claude_v3 import DeductiveTradingSystem
except ImportError:
    print("ERROR: No se puede importar DeductiveTradingSystem desde claude_v3.py")
    print("Asegúrate de que el archivo claude_v3.py esté en el mismo directorio")
    exit(1)


class DeductiveBacktester:
    """
    Sistema de backtesting que simula una cartera gestionada por el algoritmo deductivo.
    """
    
    def __init__(self, initial_capital=100000, max_positions=10, 
                 transaction_cost=0.001, annual_retrain=True):
        """
        Args:
            initial_capital (float): Capital inicial en USD
            max_positions (int): Número máximo de posiciones en cartera
            transaction_cost (float): Costo de transacción (0.1% default)
            annual_retrain (bool): Si reentrenar modelos anualmente
        """
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.transaction_cost = transaction_cost
        self.annual_retrain = annual_retrain
        
        # Estado de la cartera
        self.cash = initial_capital
        self.positions = {}  # {ticker: {'shares': n, 'avg_price': p}}
        self.portfolio_value = []
        self.trades_history = []
        
        # Modelos y predicciones
        self.trading_system = DeductiveTradingSystem(
            lookback_days=30,
            prediction_days=5,
            threshold_return=0.02
        )
        self.trained_models = {}
        self.current_predictions = {}
        
    def get_sp500_tickers(self):
        """
        Obtiene la lista actual de tickers del S&P 500.
        """
        try:
            # Intenta obtener la lista de Wikipedia
            tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            sp500_table = tables[0]
            tickers = sp500_table['Symbol'].tolist()
            
            # Limpiar símbolos
            tickers = [ticker.replace('.', '-') for ticker in tickers]
            
            return tickers[:50]  # Limitamos a 50 para pruebas rápidas
            
        except Exception as e:
            print(f"Error obteniendo lista S&P 500: {e}")
            # Lista de respaldo con las principales empresas
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
                   'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS',
                   'BAC', 'XOM', 'CVX', 'KO', 'PEP', 'VZ', 'CSCO', 'INTC']
    
    def download_historical_data(self, tickers, start_date, end_date):
        """
        Descarga datos históricos para múltiples tickers en paralelo.
        """
        all_data = {}
        failed_tickers = []
        
        def download_ticker(ticker):
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date, end=end_date)
                if not hist.empty:
                    return ticker, hist
                else:
                    return ticker, None
            except:
                return ticker, None
        
        # Descarga paralela
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(download_ticker, ticker): ticker 
                      for ticker in tickers}
            
            for future in tqdm(as_completed(futures), total=len(tickers), 
                             desc="Descargando datos históricos"):
                ticker, data = future.result()
                if data is not None:
                    all_data[ticker] = data
                else:
                    failed_tickers.append(ticker)
        
        if failed_tickers:
            print(f"No se pudieron descargar {len(failed_tickers)} tickers")
            
        return all_data
    
    def train_annual_models(self, year, tickers, historical_data):
        """
        Entrena modelos para todos los tickers usando datos hasta el año especificado.
        """
        print(f"\nEntrenando modelos para el año {year}...")
        
        # Usar datos de los 2 años anteriores para entrenamiento
        train_end = datetime(year - 1, 12, 31)
        train_start = train_end - timedelta(days=730)
        
        trained_models = {}
        model_scores = []
        failed_tickers = []
        
        for ticker in tqdm(tickers, desc=f"Entrenando modelos {year}"):
            if ticker not in historical_data:
                continue
            
            try:
                # Preparar datos para el sistema de trading
                df = self.trading_system.download_stock_data(
                    ticker, train_start, train_end
                )
                
                if df is None or len(df) < 100:
                    failed_tickers.append(ticker)
                    continue
                
                # Entrenar modelo
                result = self.trading_system.train_model(ticker, df)
                
                if result is not None and ticker in self.trading_system.models:
                    trained_models[ticker] = {
                        'model': self.trading_system.models[ticker],
                        'scaler': self.trading_system.scalers[ticker],
                        'features': self.trading_system.feature_names_by_ticker[ticker],
                        'accuracy': result['accuracy']
                    }
                    model_scores.append(result['accuracy'])
                else:
                    failed_tickers.append(ticker)
                    
            except Exception as e:
                failed_tickers.append(ticker)
                continue
        
        print(f"Modelos entrenados exitosamente: {len(trained_models)}")
        print(f"Modelos fallidos: {len(failed_tickers)}")
        if model_scores:
            print(f"Precisión promedio: {np.mean(model_scores):.2%}")
        else:
            print("ADVERTENCIA: No se pudo entrenar ningún modelo")
        
        return trained_models
    
    def generate_daily_signals(self, date, tickers, historical_data, trained_models):
        """
        Genera señales de compra/venta para todos los tickers en una fecha específica.
        """
        signals = {}
        
        for ticker in tickers:
            if ticker not in trained_models or ticker not in historical_data:
                continue
            
            try:
                # Obtener datos hasta la fecha actual
                ticker_data = historical_data[ticker]
                current_data = ticker_data[ticker_data.index <= date]
                
                if len(current_data) < 50:  # Necesitamos suficiente historia
                    continue
                
                # Restaurar modelo entrenado
                self.trading_system.models[ticker] = trained_models[ticker]['model']
                self.trading_system.scalers[ticker] = trained_models[ticker]['scaler']
                self.trading_system.feature_names_by_ticker[ticker] = trained_models[ticker]['features']
                
                # Generar predicción usando el método existente
                # Necesitamos simular la estructura de datos que espera el sistema
                df = self.trading_system.download_stock_data(
                    ticker, 
                    date - timedelta(days=60),  # 60 días de historia
                    date
                )
                
                if df is None or len(df) < 30:
                    continue
                
                # Obtener predicción
                prediction = self.trading_system.predict_signal(ticker, df)
                
                if prediction is not None:
                    # Agregar precio actual
                    current_price = current_data['Close'].iloc[-1]
                    prediction['price'] = current_price
                    
                    # El potencial se basa en la confianza y el threshold de retorno
                    prediction['potential'] = prediction['confidence'] * self.trading_system.threshold_return
                    
                    signals[ticker] = prediction
                    
            except Exception as e:
                # Silenciosamente continuar si hay error con un ticker específico
                continue
        
        return signals
    
    def rebalance_portfolio(self, date, signals, historical_data):
        """
        Rebalancea la cartera basándose en las señales del día.
        """
        trades = []
        
        # 1. VENDER: Revisar posiciones actuales
        positions_to_sell = []
        for ticker, position in self.positions.items():
            if ticker not in signals or signals[ticker]['signal'] != 'COMPRAR':
                positions_to_sell.append(ticker)
        
        # Ejecutar ventas
        for ticker in positions_to_sell:
            if ticker in historical_data and date in historical_data[ticker].index:
                price = historical_data[ticker].loc[date, 'Close']
                shares = self.positions[ticker]['shares']
                value = shares * price * (1 - self.transaction_cost)
                
                self.cash += value
                avg_price = self.positions[ticker]['avg_price']
                profit = (price - avg_price) * shares
                
                trades.append({
                    'date': date,
                    'ticker': ticker,
                    'action': 'SELL',
                    'shares': shares,
                    'price': price,
                    'value': value,
                    'profit': profit
                })
                
                del self.positions[ticker]
        
        # 2. COMPRAR: Identificar oportunidades
        current_positions = len(self.positions)
        available_slots = self.max_positions - current_positions
        
        if available_slots > 0:
            # Filtrar señales de compra no en cartera
            buy_candidates = []
            for ticker, signal in signals.items():
                if (signal['signal'] == 'COMPRAR' and 
                    ticker not in self.positions and
                    ticker in historical_data and 
                    date in historical_data[ticker].index):
                    buy_candidates.append({
                        'ticker': ticker,
                        'potential': signal['potential'],
                        'confidence': signal['confidence']
                    })
            
            # Ordenar por potencial de revalorización
            buy_candidates = sorted(buy_candidates, 
                                  key=lambda x: x['potential'], 
                                  reverse=True)
            
            # Comprar las mejores oportunidades
            position_size = self.cash / available_slots if available_slots > 0 else 0
            
            for candidate in buy_candidates[:available_slots]:
                ticker = candidate['ticker']
                if self.cash > position_size * 1.1:  # Margen de seguridad
                    price = historical_data[ticker].loc[date, 'Close']
                    shares = int(position_size / price)
                    
                    if shares > 0:
                        cost = shares * price * (1 + self.transaction_cost)
                        
                        if cost <= self.cash:
                            self.cash -= cost
                            
                            self.positions[ticker] = {
                                'shares': shares,
                                'avg_price': price * (1 + self.transaction_cost)
                            }
                            
                            trades.append({
                                'date': date,
                                'ticker': ticker,
                                'action': 'BUY',
                                'shares': shares,
                                'price': price,
                                'value': cost,
                                'profit': 0
                            })
        
        return trades
    
    def diagnose_signals(self, date, tickers, historical_data, trained_models):
        """
        Función de diagnóstico para entender por qué no hay señales de compra.
        """
        print(f"\n=== DIAGNÓSTICO para {date.strftime('%Y-%m-%d')} ===")
        
        if not trained_models:
            print("ERROR: No hay modelos entrenados")
            return
            
        print(f"Modelos disponibles: {len(trained_models)}")
        
        # Probar con los primeros 5 tickers
        test_tickers = list(trained_models.keys())[:5]
        
        for ticker in test_tickers:
            print(f"\nProbando {ticker}:")
            
            try:
                # Intentar generar señal
                df = self.trading_system.download_stock_data(
                    ticker, 
                    date - timedelta(days=60),
                    date
                )
                
                if df is None:
                    print(f"  - No se pudieron descargar datos")
                    continue
                    
                print(f"  - Datos descargados: {len(df)} filas")
                
                # Restaurar modelo
                self.trading_system.models[ticker] = trained_models[ticker]['model']
                self.trading_system.scalers[ticker] = trained_models[ticker]['scaler']
                self.trading_system.feature_names_by_ticker[ticker] = trained_models[ticker]['features']
                
                # Generar predicción
                prediction = self.trading_system.predict_signal(ticker, df)
                
                if prediction:
                    print(f"  - Señal: {prediction['signal']}")
                    print(f"  - Confianza: {prediction['confidence']:.2%}")
                    print(f"  - Valor predicción: {prediction['prediction_value']:.4f}")
                else:
                    print(f"  - No se pudo generar predicción")
                    
            except Exception as e:
                print(f"  - Error: {str(e)}")
        
        print("=" * 50)
    
    def calculate_portfolio_value(self, date, historical_data):
        """
        Calcula el valor total de la cartera en una fecha específica.
        """
        portfolio_value = self.cash
        
        for ticker, position in self.positions.items():
            if ticker in historical_data and date in historical_data[ticker].index:
                price = historical_data[ticker].loc[date, 'Close']
                portfolio_value += position['shares'] * price
        
        return portfolio_value
    
    def run_backtest(self, start_date, end_date):
        """
        Ejecuta el backtest completo.
        """
        print("=== INICIANDO BACKTEST DEDUCTIVO ===")
        print(f"Período: {start_date} a {end_date}")
        print(f"Capital inicial: ${self.initial_capital:,.2f}")
        print(f"Máximo de posiciones: {self.max_positions}")
        
        # Obtener tickers del S&P 500
        sp500_tickers = self.get_sp500_tickers()
        print(f"Universo de inversión: {len(sp500_tickers)} acciones")
        
        # Descargar datos históricos
        print("\nDescargando datos históricos...")
        # Necesitamos datos desde 2 años antes del inicio para entrenar
        data_start = start_date - timedelta(days=730)
        historical_data = self.download_historical_data(
            sp500_tickers, data_start, end_date
        )
        
        # Descargar benchmark S&P 500
        print("Descargando datos del S&P 500...")
        sp500 = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
        
        # Verificar que los datos se descargaron correctamente
        if sp500.empty:
            raise ValueError("No se pudieron descargar datos del S&P 500")
        
        # Si yfinance devuelve MultiIndex, aplanar las columnas
        if isinstance(sp500.columns, pd.MultiIndex):
            sp500.columns = sp500.columns.get_level_values(0)
        
        # Generar fechas de trading
        trading_dates = pd.date_range(start=start_date, end=end_date, freq='B')
        trading_dates = [d for d in trading_dates if d in sp500.index]
        
        # Variables para tracking
        portfolio_values = []
        sp500_values = []
        all_trades = []
        
        # Valor inicial del S&P 500
        sp500_initial = sp500['Close'].iloc[0]
        
        # Loop principal del backtest
        current_year = None
        trained_models = None
        first_diagnostic = True
        
        for i, date in enumerate(tqdm(trading_dates, desc="Simulando días de trading")):
            # Reentrenar modelos anualmente
            if self.annual_retrain and date.year != current_year:
                current_year = date.year
                trained_models = self.train_annual_models(
                    current_year, sp500_tickers, historical_data
                )
                
                # Diagnóstico en el primer día después del entrenamiento
                if first_diagnostic and trained_models:
                    self.diagnose_signals(date, sp500_tickers, historical_data, trained_models)
                    first_diagnostic = False
            
            # Generar señales del día
            if trained_models:
                signals = self.generate_daily_signals(
                    date, sp500_tickers, historical_data, trained_models
                )
                
                # Debug: imprimir información sobre las señales
                if date.day == 1:  # Solo el primer día de cada mes
                    buy_signals = sum(1 for s in signals.values() if s['signal'] == 'COMPRAR')
                    sell_signals = sum(1 for s in signals.values() if s['signal'] == 'VENDER')
                    print(f"\n{date.strftime('%Y-%m')}: {buy_signals} compras, {sell_signals} ventas de {len(signals)} señales totales")
                
                # Rebalancear cartera
                trades = self.rebalance_portfolio(date, signals, historical_data)
                all_trades.extend(trades)
            
            # Calcular valor de la cartera
            portfolio_value = self.calculate_portfolio_value(date, historical_data)
            portfolio_values.append({
                'date': date,
                'value': portfolio_value,
                'return': (portfolio_value / self.initial_capital - 1) * 100
            })
            
            # Valor del S&P 500
            if date in sp500.index:
                sp500_value = sp500.loc[date, 'Close']
                sp500_values.append({
                    'date': date,
                    'value': sp500_value,
                    'return': (sp500_value / sp500_initial - 1) * 100
                })
        
        # Crear DataFrames de resultados
        self.portfolio_df = pd.DataFrame(portfolio_values)
        self.sp500_df = pd.DataFrame(sp500_values)
        self.trades_df = pd.DataFrame(all_trades)
        
        # Calcular métricas
        self.calculate_metrics()
        
        return self.portfolio_df, self.sp500_df, self.trades_df
    
    def calculate_metrics(self):
        """
        Calcula métricas de rendimiento del backtest.
        """
        # Verificar que tenemos datos
        if self.portfolio_df.empty or self.sp500_df.empty:
            print("No hay suficientes datos para calcular métricas")
            return
            
        # Rentabilidad total
        total_return = (self.portfolio_df['value'].iloc[-1] / self.initial_capital - 1) * 100
        sp500_return = self.sp500_df['return'].iloc[-1]
        
        # Rentabilidad anualizada
        years = (self.portfolio_df['date'].iloc[-1] - self.portfolio_df['date'].iloc[0]).days / 365
        
        if years > 0:
            annual_return = (((self.portfolio_df['value'].iloc[-1] / self.initial_capital) ** (1/years)) - 1) * 100
            sp500_annual = (((self.sp500_df['value'].iloc[-1] / self.sp500_df['value'].iloc[0]) ** (1/years)) - 1) * 100
        else:
            annual_return = total_return
            sp500_annual = sp500_return
        
        # Volatilidad
        if len(self.portfolio_df) > 1:
            portfolio_returns = self.portfolio_df['value'].pct_change().dropna()
            if len(portfolio_returns) > 0:
                volatility = portfolio_returns.std() * np.sqrt(252) * 100
            else:
                volatility = 0
        else:
            volatility = 0
            
        if len(self.sp500_df) > 1:
            sp500_returns = self.sp500_df['value'].pct_change().dropna()
            if len(sp500_returns) > 0:
                sp500_volatility = sp500_returns.std() * np.sqrt(252) * 100
            else:
                sp500_volatility = 0
        else:
            sp500_volatility = 0
        
        # Sharpe Ratio (asumiendo tasa libre de riesgo del 2%)
        risk_free_rate = 0.02
        if volatility > 0:
            sharpe_ratio = (annual_return/100 - risk_free_rate) / (volatility/100)
        else:
            sharpe_ratio = 0
            
        if sp500_volatility > 0:
            sp500_sharpe = (sp500_annual/100 - risk_free_rate) / (sp500_volatility/100)
        else:
            sp500_sharpe = 0
        
        # Máximo drawdown
        if len(portfolio_returns) > 0:
            cumulative = (1 + portfolio_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min() * 100
        else:
            max_drawdown = 0
        
        # Imprimir resultados
        print("\n=== RESULTADOS DEL BACKTEST ===")
        print(f"\nRentabilidad Total:")
        print(f"  Cartera Deductiva: {total_return:.2f}%")
        print(f"  S&P 500: {sp500_return:.2f}%")
        print(f"  Alpha: {total_return - sp500_return:.2f}%")
        
        print(f"\nRentabilidad Anualizada:")
        print(f"  Cartera Deductiva: {annual_return:.2f}%")
        print(f"  S&P 500: {sp500_annual:.2f}%")
        
        print(f"\nVolatilidad Anualizada:")
        print(f"  Cartera Deductiva: {volatility:.2f}%")
        print(f"  S&P 500: {sp500_volatility:.2f}%")
        
        print(f"\nSharpe Ratio:")
        print(f"  Cartera Deductiva: {sharpe_ratio:.2f}")
        print(f"  S&P 500: {sp500_sharpe:.2f}")
        
        print(f"\nMáximo Drawdown:")
        print(f"  Cartera Deductiva: {max_drawdown:.2f}%")
        
        print(f"\nOperaciones:")
        if not self.trades_df.empty:
            print(f"  Total: {len(self.trades_df)}")
            print(f"  Promedio mensual: {len(self.trades_df) / (years * 12):.1f}")
        else:
            print(f"  Total: 0")
        
        self.metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'alpha': total_return - sp500_return
        }
    
    def plot_results(self):
        """
        Genera visualizaciones de los resultados del backtest.
        """
        # Verificar que tenemos datos para graficar
        if self.portfolio_df.empty or self.sp500_df.empty:
            print("No hay suficientes datos para generar gráficos")
            return
            
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # 1. Evolución del valor de la cartera
        ax1 = axes[0]
        ax1.plot(self.portfolio_df['date'], self.portfolio_df['return'], 
                label='Cartera Deductiva', linewidth=2, color='green')
        ax1.plot(self.sp500_df['date'], self.sp500_df['return'], 
                label='S&P 500', linewidth=2, color='blue', alpha=0.7)
        ax1.set_title('Rentabilidad Acumulada: Algoritmo Deductivo vs S&P 500', fontsize=14)
        ax1.set_xlabel('Fecha')
        ax1.set_ylabel('Rentabilidad (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Número de posiciones en cartera
        ax2 = axes[1]
        positions_over_time = []
        
        if not self.trades_df.empty:
            for date in self.portfolio_df['date']:
                # Contar posiciones activas en cada fecha
                buys = self.trades_df[
                    (self.trades_df['date'] <= date) & 
                    (self.trades_df['action'] == 'BUY')
                ].groupby('ticker')['shares'].sum()
                
                sells = self.trades_df[
                    (self.trades_df['date'] <= date) & 
                    (self.trades_df['action'] == 'SELL')
                ].groupby('ticker')['shares'].sum()
                
                net_positions = buys.subtract(sells, fill_value=0)
                active_positions = len(net_positions[net_positions > 0])
                positions_over_time.append(active_positions)
        else:
            positions_over_time = [0] * len(self.portfolio_df)
        
        ax2.fill_between(self.portfolio_df['date'], 0, positions_over_time, 
                        alpha=0.5, color='orange')
        ax2.plot(self.portfolio_df['date'], positions_over_time, 
                color='darkorange', linewidth=2)
        ax2.set_title('Número de Posiciones en Cartera', fontsize=14)
        ax2.set_xlabel('Fecha')
        ax2.set_ylabel('Número de Acciones')
        ax2.set_ylim(0, self.max_positions + 1)
        ax2.grid(True, alpha=0.3)
        
        # 3. Distribución de operaciones
        ax3 = axes[2]
        if not self.trades_df.empty:
            self.trades_df['month'] = pd.to_datetime(self.trades_df['date']).dt.to_period('M')
            monthly_trades = self.trades_df.groupby('month').size()
            
            # Convertir a fechas para el gráfico
            dates = monthly_trades.index.to_timestamp()
            ax3.bar(dates, monthly_trades.values, 
                   width=20, alpha=0.7, color='purple')
            ax3.set_title('Frecuencia de Operaciones por Mes', fontsize=14)
            ax3.set_xlabel('Fecha')
            ax3.set_ylabel('Número de Operaciones')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No hay operaciones para mostrar', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax3.transAxes)
        
        plt.tight_layout()
        plt.savefig('backtest_results_deductive.png', dpi=300)
        plt.show()


def main():
    """
    Función principal para ejecutar el backtest.
    """
    # Configuración del backtest
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    # Crear backtester con parámetros más agresivos para pruebas
    backtester = DeductiveBacktester(
        initial_capital=100000,
        max_positions=10,
        transaction_cost=0.001,
        annual_retrain=True
    )
    
    # Modificar temporalmente el threshold para ser menos restrictivo
    backtester.trading_system.threshold_return = 0.01  # 1% en lugar de 2%
    
    # Ejecutar backtest
    portfolio_df, sp500_df, trades_df = backtester.run_backtest(start_date, end_date)
    
    # Visualizar resultados
    backtester.plot_results()
    
    # Guardar resultados
    portfolio_df.to_csv('backtest_portfolio_values.csv', index=False)
    trades_df.to_csv('backtest_trades_history.csv', index=False)
    
    # Imprimir resumen de operaciones
    if not trades_df.empty:
        print("\n=== RESUMEN DE OPERACIONES ===")
        print(f"Total de operaciones: {len(trades_df)}")
        print(f"Compras: {len(trades_df[trades_df['action'] == 'BUY'])}")
        print(f"Ventas: {len(trades_df[trades_df['action'] == 'SELL'])}")
        print("\nPrimeras 5 operaciones:")
        print(trades_df.head())
    else:
        print("\nADVERTENCIA: No se realizaron operaciones durante el backtest")
    
    return backtester


if __name__ == "__main__":
    backtester = main()