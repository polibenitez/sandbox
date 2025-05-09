# crypto_trading_app.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import ccxt  # Biblioteca para interactuar con exchanges de criptomonedas
import talib  # Biblioteca de indicadores técnicos
import logging

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("crypto_trader")

class CryptoTradingApp:
    def __init__(self, mode='test', symbol='BTC/USDT', timeframe='1h'):
        """
        Inicializar la aplicación de trading
        
        Args:
            mode (str): 'test' para backtest con datos históricos, 'real' para trading en vivo
            symbol (str): Par de trading (ej. 'BTC/USDT')
            timeframe (str): Intervalo de tiempo para los datos ('1m', '5m', '15m', '1h', '4h', '1d')
        """
        self.mode = mode
        self.symbol = symbol
        self.timeframe = timeframe
        self.data = None
        self.initial_balance = 1000.0  # Balance inicial en USDT para backtesting
        self.balance = self.initial_balance
        self.position = 0  # 0 = sin posición, 1 = long
        self.trades = []
        
        # Si estamos en modo real, inicializar la conexión con Binance
        if self.mode == 'real':
            try:
                # Cargar API keys desde variables de entorno por seguridad
                api_key = os.environ.get('BINANCE_API_KEY')
                api_secret = os.environ.get('BINANCE_API_SECRET')
                
                if not api_key or not api_secret:
                    raise ValueError("API keys no encontradas. Define BINANCE_API_KEY y BINANCE_API_SECRET como variables de entorno.")
                
                self.exchange = ccxt.binance({
                    'apiKey': api_key,
                    'secret': api_secret,
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'future'  # Para trading de futuros. Usa 'spot' para spot trading
                    }
                })
                logger.info(f"Conectado con éxito a Binance en modo REAL")
            except Exception as e:
                logger.error(f"Error al conectar con Binance: {e}")
                raise
        else:
            # En modo test, usamos ccxt solo para descargar datos históricos
            self.exchange = ccxt.binance({'enableRateLimit': True})
            logger.info(f"Modo TEST activado para {self.symbol}")
    
    def download_historical_data(self, days_back=100):
        """
        Descarga datos históricos para backtesting
        
        Args:
            days_back (int): Número de días hacia atrás para descargar
        
        Returns:
            pandas.DataFrame: DataFrame con los datos históricos
        """
        try:
            since = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, since)
            
            # Convertir a DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            self.data = df
            logger.info(f"Datos históricos descargados: {len(df)} registros")
            return df
        except Exception as e:
            logger.error(f"Error al descargar datos históricos: {e}")
            raise
    
    def apply_indicators(self):
        """
        Aplicar indicadores técnicos al dataset
        """
        if self.data is None:
            raise ValueError("Necesitas descargar datos históricos primero")
        
        # Convertir a numpy arrays para talib
        close = self.data['close'].values
        high = self.data['high'].values
        low = self.data['low'].values
        volume = self.data['volume'].values
        
        # Medias móviles
        self.data['sma_20'] = talib.SMA(close, timeperiod=20)
        self.data['sma_50'] = talib.SMA(close, timeperiod=50)
        self.data['sma_200'] = talib.SMA(close, timeperiod=200)
        
        # RSI - Índice de Fuerza Relativa
        self.data['rsi'] = talib.RSI(close, timeperiod=14)
        
        # MACD - Convergencia/Divergencia de Medias Móviles
        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        self.data['macd'] = macd
        self.data['macd_signal'] = macd_signal
        self.data['macd_hist'] = macd_hist
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        self.data['bb_upper'] = upper
        self.data['bb_middle'] = middle
        self.data['bb_lower'] = lower
        
        # ATR - Average True Range (para gestión de riesgo)
        self.data['atr'] = talib.ATR(high, low, close, timeperiod=14)
        
        # Ichimoku Cloud (opcional, más complejo)
        # Eliminar NaN iniciales 
        self.data = self.data.dropna()
        
        logger.info("Indicadores técnicos aplicados correctamente")
        return self.data
    
    def trading_algorithm(self, row, prev_row=None):
        """
        Implementa el algoritmo de trading
        
        Args:
            row: Fila actual de datos
            prev_row: Fila anterior de datos (para comparación)
            
        Returns:
            int: Señal (1 para compra, -1 para venta, 0 para mantener)
        """
        # Algoritmo básico inicial
        # Aquí implementaremos nuestra estrategia de trading
        
        # ============= INICIO DEL ALGORITMO DE TRADING =============
        signal = 0  # Por defecto, mantener posición
        
        # Estrategia 1: Cruce de medias móviles con confirmación de RSI y MACD
        if prev_row is not None:
            # Señal de compra
            if (row['sma_20'] > row['sma_50'] and prev_row['sma_20'] <= prev_row['sma_50'] and  # Cruce alcista de medias
                row['rsi'] > 40 and  # RSI muestra fuerza (pero no sobrecompra)
                row['macd'] > row['macd_signal'] and  # MACD por encima de su señal
                self.position == 0):  # No tenemos posición abierta
                signal = 1  # Comprar
                
            # Señal de venta
            elif (row['sma_20'] < row['sma_50'] and prev_row['sma_20'] >= prev_row['sma_50'] or  # Cruce bajista de medias
                  row['rsi'] > 70 or  # RSI en sobrecompra
                  (row['macd'] < row['macd_signal'] and prev_row['macd'] >= prev_row['macd_signal'])) and  # Cruce bajista de MACD
                 self.position == 1:  # Tenemos una posición abierta
                signal = -1  # Vender
        
        # Estrategia 2: Breakout de Bollinger Bands con volumen
        # Compra en soporte de banda inferior con alto volumen
        if (row['close'] <= row['bb_lower'] and 
            row['volume'] > self.data['volume'].rolling(20).mean().iloc[-1] * 1.5 and  # Volumen 50% mayor que la media
            row['rsi'] < 30 and  # RSI en sobreventa
            self.position == 0):
            signal = 1  # Comprar
            
        # Venta en resistencia de banda superior
        elif (row['close'] >= row['bb_upper'] and 
              row['rsi'] > 70 and  # RSI en sobrecompra
              self.position == 1):
            signal = -1  # Vender
            
        # Estrategia 3: Detección de tendencia con SMA de 200 períodos
        # Solo comprar en tendencia alcista (precio por encima de SMA 200)
        if signal == 1 and row['close'] < row['sma_200']:
            signal = 0  # Cancelar señal de compra en tendencia bajista
            
        # ============= FIN DEL ALGORITMO DE TRADING =============
        
        return signal
    
    def run_backtest(self):
        """
        Ejecuta el backtest en los datos históricos
        """
        if self.mode != 'test':
            raise ValueError("Esta función solo está disponible en modo test")
        
        if self.data is None or 'sma_20' not in self.data.columns:
            raise ValueError("Necesitas aplicar los indicadores primero")
        
        self.balance = self.initial_balance
        self.position = 0
        self.trades = []
        
        prev_row = None
        
        for idx, row in self.data.iterrows():
            if idx == self.data.index[0]:
                prev_row = row
                continue
                
            signal = self.trading_algorithm(row, prev_row)
            
            # Procesar señales
            if signal == 1 and self.position == 0:  # Comprar
                price = row['close']
                self.position = 1
                amount = self.balance * 0.95  # Usar 95% del balance (dejando algo para comisiones)
                quantity = amount / price
                self.balance -= amount
                
                self.trades.append({
                    'timestamp': idx,
                    'type': 'buy',
                    'price': price,
                    'quantity': quantity,
                    'amount': amount
                })
                logger.info(f"BACKTEST BUY: {idx} - Precio: {price:.2f}, Cantidad: {quantity:.6f}, Total: {amount:.2f}")
                
            elif signal == -1 and self.position == 1:  # Vender
                price = row['close']
                # Calcular cantidad a vender basada en la última compra
                quantity = self.trades[-1]['quantity']
                amount = quantity * price
                self.balance += amount
                self.position = 0
                
                self.trades.append({
                    'timestamp': idx,
                    'type': 'sell',
                    'price': price,
                    'quantity': quantity,
                    'amount': amount
                })
                logger.info(f"BACKTEST SELL: {idx} - Precio: {price:.2f}, Cantidad: {quantity:.6f}, Total: {amount:.2f}")
            
            prev_row = row
        
        # Cerrar posición abierta al final del backtest con el último precio
        if self.position == 1:
            last_price = self.data['close'].iloc[-1]
            quantity = self.trades[-1]['quantity']
            amount = quantity * last_price
            self.balance += amount
            
            self.trades.append({
                'timestamp': self.data.index[-1],
                'type': 'sell',
                'price': last_price,
                'quantity': quantity,
                'amount': amount,
                'note': 'Final position close'
            })
        
        # Calcular rendimiento
        self.calculate_performance()
        
    def calculate_performance(self):
        """
        Calcula y muestra el rendimiento del backtest
        """
        if not self.trades:
            logger.warning("No hay trades para calcular rendimiento")
            return
        
        # Convertir trades a DataFrame
        trades_df = pd.DataFrame(self.trades)
        
        # Rendimiento total
        total_return = (self.balance / self.initial_balance - 1) * 100
        
        # Número de operaciones
        num_trades = len([t for t in self.trades if t['type'] == 'buy'])
        
        # Operaciones ganadoras/perdedoras
        profit_trades = 0
        loss_trades = 0
        
        buy_trades = [t for t in self.trades if t['type'] == 'buy']
        sell_trades = [t for t in self.trades if t['type'] == 'sell']
        
        for i in range(min(len(buy_trades), len(sell_trades))):
            if sell_trades[i]['price'] > buy_trades[i]['price']:
                profit_trades += 1
            else:
                loss_trades += 1
        
        # Rendimiento máximo y mínimo
        if self.trades:
            balance_history = []
            current_balance = self.initial_balance
            
            for trade in self.trades:
                if trade['type'] == 'buy':
                    current_balance -= trade['amount']
                else:  # sell
                    current_balance += trade['amount']
                balance_history.append(current_balance)
            
            max_balance = max(balance_history)
            min_balance = min(balance_history)
            max_drawdown = ((max_balance - min_balance) / max_balance) * 100
        else:
            max_drawdown = 0
        
        # Mostrar resultados
        logger.info("\n" + "="*50)
        logger.info("RESULTADOS DEL BACKTEST")
        logger.info("="*50)
        logger.info(f"Balance inicial: {self.initial_balance:.2f} USDT")
        logger.info(f"Balance final: {self.balance:.2f} USDT")
        logger.info(f"Rendimiento total: {total_return:.2f}%")
        logger.info(f"Número de operaciones: {num_trades}")
        logger.info(f"Operaciones ganadoras: {profit_trades}")
        logger.info(f"Operaciones perdedoras: {loss_trades}")
        if num_trades > 0:
            logger.info(f"Porcentaje de éxito: {(profit_trades/num_trades*100):.2f}%")
        logger.info(f"Máximo drawdown: {max_drawdown:.2f}%")
        logger.info("="*50)
        
        return {
            'total_return': total_return,
            'num_trades': num_trades,
            'profit_trades': profit_trades,
            'loss_trades': loss_trades,
            'win_rate': (profit_trades/num_trades*100) if num_trades > 0 else 0,
            'max_drawdown': max_drawdown
        }
    
    def plot_results(self):
        """
        Genera gráficos con los resultados del backtest
        """
        if self.data is None or not self.trades:
            logger.warning("No hay datos o trades para graficar")
            return
        
        # Convertir trades a DataFrame
        buy_trades = [t for t in self.trades if t['type'] == 'buy']
        sell_trades = [t for t in self.trades if t['type'] == 'sell']
        
        buy_times = [t['timestamp'] for t in buy_trades]
        buy_prices = [t['price'] for t in buy_trades]
        sell_times = [t['timestamp'] for t in sell_trades]
        sell_prices = [t['price'] for t in sell_trades]
        
        # Crear figura
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Precio y señales
        plt.subplot(2, 1, 1)
        plt.plot(self.data.index, self.data['close'], label='Precio de cierre', alpha=0.7)
        plt.plot(self.data.index, self.data['sma_20'], label='SMA 20', alpha=0.6)
        plt.plot(self.data.index, self.data['sma_50'], label='SMA 50', alpha=0.6)
        
        # Marcar compras y ventas
        plt.scatter(buy_times, buy_prices, color='green', label='Compra', marker='^', s=100)
        plt.scatter(sell_times, sell_prices, color='red', label='Venta', marker='v', s=100)
        
        plt.title(f'Backtest de {self.symbol} - Precio y Señales')
        plt.legend()
        plt.grid(True)
        
        # Subplot 2: Indicadores
        plt.subplot(2, 1, 2)
        plt.plot(self.data.index, self.data['rsi'], label='RSI', color='purple', alpha=0.7)
        plt.axhline(y=70, color='r', linestyle='--')
        plt.axhline(y=30, color='g', linestyle='--')
        plt.title('RSI (14)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'backtest_{self.symbol.replace("/", "_")}_{datetime.now().strftime("%Y%m%d")}.png')
        plt.show()
    
    def execute_trade_real(self, signal):
        """
        Ejecuta una operación en tiempo real en Binance
        
        Args:
            signal (int): 1 para compra, -1 para venta
        """
        if self.mode != 'real':
            raise ValueError("Esta función solo está disponible en modo real")
        
        try:
            if signal == 1:  # Comprar
                # Obtener balance disponible
                balance = self.exchange.fetch_balance()
                usdt_balance = balance['USDT']['free']
                
                # Obtener precio actual
                ticker = self.exchange.fetch_ticker(self.symbol)
                current_price = ticker['last']
                
                # Calcular cantidad a comprar (95% del balance)
                amount_usdt = usdt_balance * 0.95
                quantity = amount_usdt / current_price
                
                # Redondear la cantidad según las reglas del exchange
                market_info = self.exchange.market(self.symbol)
                precision = market_info['precision']['amount']
                quantity = round(quantity, precision)
                
                # Ejecutar orden de compra
                order = self.exchange.create_market_buy_order(self.symbol, quantity)
                
                logger.info(f"ORDEN COMPRA EJECUTADA: {order}")
                return order
                
            elif signal == -1:  # Vender
                # Obtener posición actual
                positions = self.exchange.fetch_positions([self.symbol.split('/')[0]])
                position_size = None
                
                for position in positions:
                    if position['symbol'] == self.symbol.split('/')[0]:
                        position_size = position['contracts']
                        break
                
                if position_size and position_size > 0:
                    # Ejecutar orden de venta
                    order = self.exchange.create_market_sell_order(self.symbol, position_size)
                    logger.info(f"ORDEN VENTA EJECUTADA: {order}")
                    return order
                else:
                    logger.warning(f"No hay posición para vender en {self.symbol}")
            
            return None
        
        except Exception as e:
            logger.error(f"Error al ejecutar operación en tiempo real: {e}")
            raise
    
    def run_live_trading(self, interval_seconds=3600):
        """
        Ejecuta el trading en tiempo real
        
        Args:
            interval_seconds (int): Intervalo en segundos entre comprobaciones
        """
        if self.mode != 'real':
            raise ValueError("Esta función solo está disponible en modo real")
        
        logger.info(f"Iniciando trading en vivo para {self.symbol} con intervalo de {interval_seconds} segundos")
        
        try:
            while True:
                # Obtener datos recientes
                now = datetime.now()
                since = int((now - timedelta(days=3)).timestamp() * 1000)  # Últimos 3 días
                ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, since)
                
                # Convertir a DataFrame
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Aplicar indicadores
                self.data = df
                self.apply_indicators()
                
                # Obtener señal de trading
                current_row = self.data.iloc[-1]
                prev_row = self.data.iloc[-2] if len(self.data) > 1 else None
                
                signal = self.trading_algorithm(current_row, prev_row)
                
                # Ejecutar operación si hay señal
                if signal != 0:
                    result = self.execute_trade_real(signal)
                    if result:
                        logger.info(f"Operación ejecutada: {'COMPRA' if signal == 1 else 'VENTA'}")
                    else:
                        logger.info("No se ejecutó ninguna operación")
                else:
                    logger.info("No hay señal de trading en este momento")
                
                # Esperar hasta el próximo intervalo
                logger.info(f"Esperando {interval_seconds} segundos hasta la próxima comprobación...")
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            logger.info("Trading en vivo detenido por el usuario")
        except Exception as e:
            logger.error(f"Error en el trading en vivo: {e}")
            raise

# Función principal para ejecutar la aplicación
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Aplicación de Trading de Criptomonedas')
    parser.add_argument('--mode', type=str, default='test', choices=['test', 'real'],
                        help='Modo de operación: test o real')
    parser.add_argument('--symbol', type=str, default='BTC/USDT',
                        help='Símbolo de trading (ej. BTC/USDT)')
    parser.add_argument('--timeframe', type=str, default='1h',
                        help='Timeframe para los datos (1m, 5m, 15m, 1h, 4h, 1d)')
    parser.add_argument('--days', type=int, default=100,
                        help='Días de datos históricos para backtest')
    
    args = parser.parse_args()
    
    # Crear la aplicación
    app = CryptoTradingApp(mode=args.mode, symbol=args.symbol, timeframe=args.timeframe)
    
    if args.mode == 'test':
        # Modo de backtest
        logger.info("Iniciando backtest...")
        
        # Descargar datos históricos
        app.download_historical_data(days_back=args.days)
        
        # Aplicar indicadores técnicos
        app.apply_indicators()
        
        # Ejecutar backtest
        app.run_backtest()
        
        # Graficar resultados
        app.plot_results()
        
    else:
        # Modo de trading en vivo
        logger.info("Iniciando trading en vivo...")
        
        # Definir intervalo según el timeframe
        if args.timeframe == '1m':
            interval_seconds = 60
        elif args.timeframe == '5m':
            interval_seconds = 300
        elif args.timeframe == '15m':
            interval_seconds = 900
        elif args.timeframe == '1h':
            interval_seconds = 3600
        elif args.timeframe == '4h':
            interval_seconds = 14400
        else:  # 1d
            interval_seconds = 86400
        
        # Iniciar trading en vivo
        app.run_live_trading(interval_seconds=interval_seconds)

if __name__ == "__main__":
    main()