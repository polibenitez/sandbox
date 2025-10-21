#!/usr/bin/env python3
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Importa tu clase del otro archivo
from penny_stock_robot_advisor import PennyStockRobotAdvisor

# --- CONFIGURACIÓN DEL BACKTEST ---
WATCHLIST_SYMBOLS = [
    "SNDL",
    "AMC",
    "GME",
    "OPEN",
    "CTRM",
    "SHIP",
    "NAKD",
    "GNUS",
    "OPEN",
    "chpt",
    "LCFY",
    "SIRI",
    "XAIR",
    "HTOO",
    "CTMX",
    "CLOV",
] # Ejemplo de watchlist
BENCHMARK_TICKER = 'SPY'  # S&P 500 ETF como benchmark
INITIAL_CAPITAL = 100000.0
START_DATE = '2024-05-01'
END_DATE = '2024-07-01'
ROBOT_CONFIG_PRESET = 'very_aggressive' # 'conservative', 'balanced', 'aggressive', 'very_aggressive'

class Backtester:
    """
    Motor de backtesting para simular la estrategia del PennyStockRobotAdvisor
    y comparar su rendimiento contra un benchmark.
    """
    def __init__(self, robot, watchlist, initial_capital, start_date, end_date):
        self.robot = robot
        self.watchlist = watchlist
        self.initial_capital = initial_capital
        self.start_date = start_date
        self.end_date = end_date
        
        self.cash = initial_capital
        self.positions = {} # { 'SYMBOL': {'shares': X, 'entry_price': Y, 'stop_loss': Z, 'take_profit': W, ...} }
        self.portfolio_history = [] # Para registrar el valor diario del portafolio
        
        print("🔧 Backtester inicializado.")
        print(f"💰 Capital Inicial: ${initial_capital:,.2f}")
        print(f"📅 Período: {start_date} a {end_date}")
        print(f"🤖 Configuración Robot: {self.robot.config_preset.upper()}")

    def _get_historical_data(self):
        """Descarga todos los datos históricos necesarios."""
        all_tickers = self.watchlist + [BENCHMARK_TICKER]
        print(f"\n📥 Descargando datos para: {', '.join(all_tickers)}...")
        data = yf.download(all_tickers, start=self.start_date, end=self.end_date, progress=False)
        self.historical_data = data.ffill().bfill() # Rellenar huecos
        print("✅ Datos descargados.")

    def run(self):
        """Ejecuta la simulación de backtesting día a día."""
        self._get_historical_data()
        
        trading_days = self.historical_data.index
        
        for day in trading_days:
            current_date_str = day.strftime('%Y-%m-%d')
            
            # 1. Actualizar valor del portafolio y registrarlo
            self._update_portfolio_value(day)
            
            # 2. Chequear stop-loss y take-profit para posiciones abiertas
            self._check_stops_and_profits(day)
            
            # 3. Preparar datos de mercado para el análisis del robot
            market_data_for_robot = self._prepare_market_data_for_day(day)
            
            # 4. Ejecutar el algoritmo para obtener señales de compra
            # Simulamos tu función 'run_analysis' llamando directamente al robot
            self.robot.update_watchlist(self.watchlist)
            daily_report = self.robot.generate_daily_report(market_data_for_robot, show_details=False)
            buy_signals = daily_report['opportunities']
            
            # 5. Procesar señales de compra
            self._process_buy_signals(buy_signals, day)

        print("\n🏁 Backtesting finalizado.")
        return self.analyze_performance()

    def _update_portfolio_value(self, day):
        """Calcula el valor total del portafolio (cash + posiciones) para el día."""
        total_value = self.cash
        for symbol, position in self.positions.items():
            current_price = self.historical_data['Close'][symbol].get(day, position['entry_price'])
            total_value += position['shares'] * current_price
        
        self.portfolio_history.append({'date': day, 'value': total_value})

    def _check_stops_and_profits(self, day):
        """Vende posiciones si alcanzan su stop-loss o take-profit."""
        symbols_to_sell = []
        for symbol, position in self.positions.items():
            low_price = self.historical_data['Low'][symbol].get(day)
            high_price = self.historical_data['High'][symbol].get(day)
            
            # Simplificación: si el SL se toca, se vende a ese precio.
            if low_price <= position['stop_loss']:
                sell_price = position['stop_loss']
                print(f"🔴 {day.date()} | STOP-LOSS: Vendiendo {symbol} a ${sell_price:.3f}")
                self.cash += position['shares'] * sell_price
                symbols_to_sell.append(symbol)
                
            # Simplificación: si el TP (nivel 1) se toca, se vende a ese precio.
            elif high_price >= position['take_profit'][0]:
                sell_price = position['take_profit'][0]
                print(f"🟢 {day.date()} | TAKE-PROFIT: Vendiendo {symbol} a ${sell_price:.3f}")
                self.cash += position['shares'] * sell_price
                symbols_to_sell.append(symbol)

        for symbol in symbols_to_sell:
            del self.positions[symbol]

    def _prepare_market_data_for_day(self, day):
        """Crea el diccionario de datos de mercado que el robot espera."""
        market_data_batch = {}
        for symbol in self.watchlist:
            # Simplificación: Usamos datos históricos para simular las métricas del robot.
            # En un sistema real, estas métricas serían más complejas (RSI, VWAP, etc.)
            # Aquí usamos proxies para que el backtest funcione.
            
            prev_day = day - timedelta(days=1)
            if symbol not in self.historical_data['Close'] or prev_day not in self.historical_data['Close'][symbol]:
                continue
            
            price = self.historical_data['Close'][symbol].get(day)
            volume = self.historical_data['Volume'][symbol].get(day)
            avg_volume_20d = self.historical_data['Volume'][symbol].rolling(window=20).mean().get(prev_day, volume)
            price_change_pct = (price / self.historical_data['Close'][symbol].get(prev_day, price) - 1) * 100
            
            # Valores Fijos/Simulados para el backtest (MEJORAR ESTO EN EL FUTURO)
            # Un backtest más preciso requeriría calcular estas métricas diariamente.
            market_data_batch[symbol] = {
                'price': price,
                'volume': volume,
                'avg_volume_20d': avg_volume_20d if avg_volume_20d > 0 else 1,
                'price_change_pct': price_change_pct,
                'vwap': self.historical_data['Close'][symbol].rolling(window=5).mean().get(day, price), # Proxy de VWAP
                'rsi': 50, # Valor neutral simulado
                'short_interest_pct': 25, # Valor simulado alto para provocar señales
                'days_to_cover': 3, # Valor simulado
                'borrow_rate': 80, # Valor simulado
                'has_delisting_warning': price < 1.0, # Lógica simple
                'bid_ask_spread_pct': 2.0, # Valor simulado bajo
                'market_depth_dollars': 50000, # Valor simulado alto
                'daily_dollar_volume': price * volume,
                'atr_14': (self.historical_data['High'][symbol] - self.historical_data['Low'][symbol]).rolling(window=14).mean().get(day, price * 0.1) # Proxy de ATR
            }
        return market_data_batch

    def _process_buy_signals(self, signals, day):
        """Ejecuta órdenes de compra basadas en las señales del robot."""
        for signal in signals:
            symbol = signal['trading_action']['symbol']
            if symbol in self.positions: # No comprar si ya tenemos una posición
                continue
                
            price = self.historical_data['Close'][symbol].get(day)
            if price is None:
                continue

            position_size_pct = signal['trading_action']['position_size_pct'] / 100
            amount_to_invest = self.initial_capital * position_size_pct # Basado en capital inicial, no en el actual
            
            if self.cash >= amount_to_invest:
                num_shares = amount_to_invest / price
                self.cash -= amount_to_invest
                self.positions[symbol] = {
                    'shares': num_shares,
                    'entry_price': price,
                    'stop_loss': signal['trading_action']['stop_loss'],
                    'take_profit': signal['trading_action']['take_profit_levels']
                }
                print(f"🔵 {day.date()} | COMPRA: {num_shares:.0f} acciones de {symbol} a ${price:.3f}")

    def analyze_performance(self):
        """Calcula y muestra las métricas de rendimiento y genera el gráfico."""
        portfolio_df = pd.DataFrame(self.portfolio_history).set_index('date')
        portfolio_df['daily_return'] = portfolio_df['value'].pct_change()
        
        # Benchmark
        benchmark_df = self.historical_data['Close'][[BENCHMARK_TICKER]].copy()
        benchmark_df.columns = ['value']
        benchmark_df['value'] = (benchmark_df['value'] / benchmark_df['value'].iloc[0]) * self.initial_capital
        benchmark_df['daily_return'] = benchmark_df['value'].pct_change()

        # Métricas
        days = len(portfolio_df)
        
        final_value_algo = portfolio_df['value'].iloc[-1]
        total_return_algo = (final_value_algo / self.initial_capital) - 1
        annual_return_algo = (1 + total_return_algo)**(252 / days) - 1
        annual_volatility_algo = portfolio_df['daily_return'].std() * np.sqrt(252)
        sharpe_ratio_algo = annual_return_algo / annual_volatility_algo if annual_volatility_algo > 0 else 0
        
        final_value_bench = benchmark_df['value'].iloc[-1]
        total_return_bench = (final_value_bench / self.initial_capital) - 1
        annual_return_bench = (1 + total_return_bench)**(252 / days) - 1
        
        print("\n--- RESULTADOS DEL BACKTEST ---")
        print(f"Rendimiento del Algoritmo: {total_return_algo:+.2%}")
        print(f"Rendimiento del S&P 500:   {total_return_bench:+.2%}")
        print("-" * 30)
        print(f"Valor Final Algoritmo: ${final_value_algo:,.2f}")
        print(f"Valor Final S&P 500:   ${final_value_bench:,.2f}")
        print(f"Ratio de Sharpe Algoritmo: {sharpe_ratio_algo:.2f}")

        # Gráfico
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(portfolio_df.index, portfolio_df['value'], label='Algoritmo Penny Stock', color='royalblue', linewidth=2)
        ax.plot(benchmark_df.index, benchmark_df['value'], label='S&P 500 (SPY)', color='gray', linestyle='--')
        
        ax.set_title(f'Rendimiento del Algoritmo vs. S&P 500 ({ROBOT_CONFIG_PRESET.capitalize()})', fontsize=16)
        ax.set_ylabel('Valor del Portafolio ($)', fontsize=12)
        ax.set_xlabel('Fecha', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True)
        plt.figtext(0.1, 0.01, f'Rendimiento Total: Algo {total_return_algo:+.2%} vs SPY {total_return_bench:+.2%} | Sharpe Ratio: {sharpe_ratio_algo:.2f}', ha="left", fontsize=10)
        plt.show()

        return portfolio_df

# --- PUNTO DE ENTRADA PRINCIPAL ---
if __name__ == "__main__":
    # 1. Crear una instancia del robot
    robot_advisor = PennyStockRobotAdvisor(config_preset=ROBOT_CONFIG_PRESET)

    # 2. Crear una instancia del backtester
    backtester = Backtester(
        robot=robot_advisor,
        watchlist=WATCHLIST_SYMBOLS,
        initial_capital=INITIAL_CAPITAL,
        start_date=START_DATE,
        end_date=END_DATE
    )

    # 3. Ejecutar el backtest
    results_df = backtester.run()