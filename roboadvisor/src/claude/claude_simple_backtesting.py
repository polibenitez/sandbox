"""
Sistema de Backtesting Simplificado - Versión Debug
===================================================

Versión simplificada del backtester para identificar y resolver problemas.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Importar el sistema base
try:
    from claude_v3 import DeductiveTradingSystem
except ImportError:
    print("ERROR: No se puede importar DeductiveTradingSystem desde claude_v3.py")
    exit(1)


class SimpleBacktester:
    """
    Backtester simplificado con estrategia más directa.
    """
    
    def __init__(self, initial_capital=100000, max_positions=10):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.max_positions = max_positions
        self.portfolio_history = []
        self.trades = []
        
    def get_test_tickers(self):
        """Lista pequeña de tickers para pruebas."""
        additional_tickers = [
            "ASML",
            "APLD",
            "DOCU",
            "HIMS",
            "CRWV",
            "PLUN",
            "CORZ",
            "QTUM",
            "ONDS",
            "LAES",
            "NVTS",
            "SOFI",
            "SYTA",
            "MLGO",
            "REKR",
            "CGTX",
            "ABVE",
            "MRSN",
            "MVST",
            "EVTV",
            "DRO",
            "GLNG",
            "SLDP",
            "EOSE",
            "NRBT",
            "NBIS",
            "HOTH",
            "OPEN",
        ]

        tickets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 
                'JPM', 'JNJ', 'UNH', 'V', 'WMT']
        
        return tickets + additional_tickers
    
    def simple_strategy(self, ticker, data, lookback=20):
        """
        Estrategia simple: comprar si el precio está por debajo de la media móvil.
        """
        if len(data) < lookback:
            return None
            
        try:
            current_price = float(data['Close'].iloc[-1])
            ma20 = float(data['Close'].rolling(window=lookback).mean().iloc[-1])
            
            # Saltar si hay valores NaN
            if pd.isna(current_price) or pd.isna(ma20):
                return None
            
            # Comprar si el precio está 5% por debajo de la MA20
            if current_price < ma20 * 0.95:
                return 'BUY'
            # Vender si el precio está 5% por encima de la MA20
            elif current_price > ma20 * 1.05:
                return 'SELL'
            else:
                return 'HOLD'
        except Exception as e:
            return None
    
    def run_simple_backtest(self, start_date, end_date):
        """
        Ejecuta un backtest simple para verificar la mecánica básica.
        """
        print("=== BACKTEST SIMPLIFICADO ===")
        print(f"Capital inicial: ${self.initial_capital:,.2f}")
        
        # Obtener tickers
        tickers = self.get_test_tickers()
        print(f"Tickers de prueba: {tickers}")
        
        # Descargar datos
        print("\nDescargando datos...")
        data = {}
        for ticker in tqdm(tickers):
            try:
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not df.empty:
                    data[ticker] = df
            except:
                continue
        
        print(f"Datos descargados para {len(data)} tickers")
        
        # Obtener fechas de trading
        all_dates = []
        for df in data.values():
            all_dates.extend(df.index.tolist())
        trading_dates = sorted(list(set(all_dates)))
        
        print(f"Días de trading: {len(trading_dates)}")
        
        # Simular trading
        print("\nEjecutando simulación...")
        
        # Agregar debug para los primeros días
        debug_days = 0
        
        for date in tqdm(trading_dates, desc="Simulando"):
            daily_signals = {}
            
            # Generar señales para cada ticker
            for ticker in data:
                if date in data[ticker].index:
                    # Obtener datos hasta la fecha actual
                    hist_data = data[ticker][data[ticker].index <= date]
                    signal = self.simple_strategy(ticker, hist_data)
                    if signal and signal != 'HOLD':
                        # Asegurarse de obtener un valor escalar
                        price = float(data[ticker].loc[date, 'Close'])
                        daily_signals[ticker] = {
                            'signal': signal,
                            'price': price
                        }
            
            # Ejecutar operaciones
            # Primero vender
            for ticker in list(self.positions.keys()):
                if ticker in daily_signals and daily_signals[ticker]['signal'] == 'SELL':
                    price = float(daily_signals[ticker]['price'])  # Asegurar escalar
                    shares = self.positions[ticker]['shares']
                    self.cash += shares * price
                    
                    self.trades.append({
                        'date': date,
                        'ticker': ticker,
                        'action': 'SELL',
                        'shares': shares,
                        'price': price
                    })
                    
                    del self.positions[ticker]
            
            # Luego comprar
            if len(self.positions) < self.max_positions:
                buy_candidates = [(t, s) for t, s in daily_signals.items() 
                                if s['signal'] == 'BUY' and t not in self.positions]
                
                # Comprar hasta llenar cartera
                available_slots = self.max_positions - len(self.positions)
                if available_slots > 0 and self.cash > 0:
                    position_size = self.cash / available_slots
                    
                    for ticker, signal in buy_candidates[:available_slots]:
                        price = float(signal['price'])  # Asegurar escalar
                        shares = int(position_size / price)
                        
                        if shares > 0 and shares * price <= self.cash:
                            self.cash -= shares * price
                            self.positions[ticker] = {
                                'shares': shares,
                                'avg_price': price
                            }
                            
                            self.trades.append({
                                'date': date,
                                'ticker': ticker,
                                'action': 'BUY',
                                'shares': shares,
                                'price': price
                            })
            
            # Calcular valor de cartera
            portfolio_value = self.cash
            for ticker, position in self.positions.items():
                if ticker in data and date in data[ticker].index:
                    current_price = float(data[ticker].loc[date, 'Close'])  # Asegurar escalar
                    portfolio_value += position['shares'] * current_price
            
            self.portfolio_history.append({
                'date': date,
                'value': portfolio_value,
                'cash': self.cash,
                'n_positions': len(self.positions)
            })
        
        # Convertir a DataFrame
        self.portfolio_df = pd.DataFrame(self.portfolio_history)
        self.trades_df = pd.DataFrame(self.trades)
        
        return self.portfolio_df, self.trades_df
    
    def analyze_results(self):
        """
        Analiza y muestra los resultados del backtest.
        """
        if self.portfolio_df.empty:
            print("No hay resultados para analizar")
            return
            
        # Calcular retorno
        initial_value = self.portfolio_df['value'].iloc[0]
        final_value = self.portfolio_df['value'].iloc[-1]
        total_return = (final_value / initial_value - 1) * 100
        
        print("\n=== RESULTADOS ===")
        print(f"Valor inicial: ${initial_value:,.2f}")
        print(f"Valor final: ${final_value:,.2f}")
        print(f"Retorno total: {total_return:.2f}%")
        
        if not self.trades_df.empty:
            print(f"\nOperaciones totales: {len(self.trades_df)}")
            print(f"Compras: {len(self.trades_df[self.trades_df['action'] == 'BUY'])}")
            print(f"Ventas: {len(self.trades_df[self.trades_df['action'] == 'SELL'])}")
            
            print("\nPrimeras 10 operaciones:")
            print(self.trades_df.head(10))
        else:
            print("\nNo se realizaron operaciones")
        
        # Graficar
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(self.portfolio_df['date'], self.portfolio_df['value'])
        plt.title('Valor de la Cartera')
        plt.ylabel('Valor ($)')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.plot(self.portfolio_df['date'], self.portfolio_df['n_positions'])
        plt.title('Número de Posiciones')
        plt.ylabel('Posiciones')
        plt.xlabel('Fecha')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('simple_backtest_results.png')
        plt.show()


def test_deductive_system():
    """
    Prueba el sistema deductivo con un solo ticker.
    """
    print("\n=== PRUEBA DEL SISTEMA DEDUCTIVO ===")
    
    system = DeductiveTradingSystem(
        lookback_days=30,
        prediction_days=5,
        threshold_return=0.01  # 1% threshold
    )
    
    # Probar con Apple
    ticker = 'AAPL'
    start_date = datetime.now() - timedelta(days=730)
    end_date = datetime.now()
    
    print(f"\nDescargando datos de {ticker}...")
    df = system.download_stock_data(ticker, start_date, end_date)
    
    if df is None:
        print("Error descargando datos")
        return
        
    print(f"Datos descargados: {len(df)} filas")
    print(f"Columnas: {df.columns.tolist()}")
    
    # Entrenar modelo
    print("\nEntrenando modelo...")
    result = system.train_model(ticker, df)
    
    if result:
        print(f"Modelo entrenado con precisión: {result['accuracy']:.2%}")
        print("\nTop 5 características importantes:")
        print(result['feature_importance'].head())
        
        # Generar predicción
        print("\nGenerando predicción actual...")
        signal = system.predict_signal(ticker, df)
        
        if signal:
            print(f"Señal: {signal['signal']}")
            print(f"Confianza: {signal['confidence']:.2%}")
            print(f"Valor de predicción: {signal['prediction_value']:.4f}")
        else:
            print("No se pudo generar predicción")
    else:
        print("Error entrenando modelo")


def main():
    """
    Función principal para ejecutar pruebas.
    """
    # Primero probar el sistema deductivo
    test_deductive_system()
    
    # Luego ejecutar backtest simple
    print("\n" + "="*60 + "\n")
    
    backtester = SimpleBacktester()
    
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    portfolio_df, trades_df = backtester.run_simple_backtest(start_date, end_date)
    
    backtester.analyze_results()
    
    return backtester


if __name__ == "__main__":
    backtester = main()