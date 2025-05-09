# usage_example.py
# Ejemplo de cómo usar el algoritmo avanzado de trading con nuestra aplicación

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from crypto_trading_app import CryptoTradingApp
from advanced_trading_algorithm import AdvancedTradingAlgorithm

def main():
    # Configurar la aplicación en modo test
    app = CryptoTradingApp(mode='test', symbol='BTC/USDT', timeframe='1h')
    
    # Descargar datos históricos
    print("Descargando datos históricos...")
    app.download_historical_data(days_back=200)
    
    # Aplicar indicadores técnicos
    print("Aplicando indicadores técnicos...")
    app.apply_indicators()
    
    # Descargar datos de timeframe superior para análisis multitimeframe
    app_daily = CryptoTradingApp(mode='test', symbol='BTC/USDT', timeframe='1d')
    app_daily.download_historical_data(days_back=200)
    app_daily.apply_indicators()
    
    # Inicializar algoritmo avanzado
    print("Inicializando algoritmo avanzado...")
    advanced_algo = AdvancedTradingAlgorithm(base_data=app.data)
    
    # Entrenar modelos de machine learning (opcional)
    print("Entrenando modelos de machine learning...")
    try:
        # Solo si tenemos suficientes datos
        if len(app.data) > 100:
            advanced_algo.train_random_forest()
            # El entrenamiento LSTM puede ser lento, descomentar si se necesita
            # advanced_algo.train_lstm()
    except Exception as e:
        print(f"Error al entrenar modelos: {e}")
    
    # Ejecutar backtesting con el algoritmo avanzado
    print("Ejecutando backtest con algoritmo avanzado...")
    
    # Almacenar resultados
    results = []
    signals = []
    
    # Balance inicial
    balance = 1000.0
    position = 0
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    
    # Backtesting
    for i in range(50, len(app.data)):
        # Obtener datos hasta el índice actual (simulando datos disponibles en ese momento)
        current_data = app.data.iloc[:i+1]
        higher_tf_data = app_daily.data
        
        # Aplicar algoritmo avanzado
        trading_decision = advanced_algo.advanced_trading_algorithm(
            current_data=current_data,
            higher_tf_data=higher_tf_data
        )
        
        # Extraer señal y detalles
        signal = trading_decision['signal']
        confidence = trading_decision['confidence']
        position_size = trading_decision.get('position_size', 0)
        risk_levels = trading_decision.get('risk_levels', None)
        
        current_price = current_data['close'].iloc[-1]
        current_time = current_data.index[-1]
        
        # Procesar señales
        if signal == 1 and position == 0:  # Comprar
            position = 1
            entry_price = current_price
            
            # Determinar stop loss y take profit
            if risk_levels:
                stop_loss = risk_levels['stop_loss']
                take_profit = risk_levels['take_profit']
            else:
                # Valores por defecto si no hay gestión de riesgo
                stop_loss = entry_price * 0.97  # 3% por debajo
                take_profit = entry_price * 1.06  # 6% por encima
            
            # Calcular cantidad a comprar
            amount = balance * 0.95  # Usar 95% del balance
            quantity = amount / current_price
            balance -= amount
            
            signals.append({
                'time': current_time,
                'price': current_price,
                'type': 'buy',
                'confidence': confidence,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            })
            
            print(f"COMPRA: {current_time} - Precio: {current_price:.2f}, Confianza: {confidence:.2f}")
            print(f"  Stop Loss: {stop_loss:.2f}, Take Profit: {take_profit:.2f}")
            
        elif signal == -1 and position == 1:  # Vender
            position = 0
            exit_price = current_price
            
            # Calcular ganancia/pérdida
            profit_pct = (exit_price / entry_price - 1) * 100
            
            # Cantidad original invertida
            original_amount = balance * 0.95
            
            # Calcular nuevo balance
            amount = original_amount * (1 + profit_pct/100)
            balance += amount
            
            signals.append({
                'time': current_time,
                'price': current_price,
                'type': 'sell',
                'confidence': confidence,
                'profit_pct': profit_pct
            })
            
            print(f"VENTA: {current_time} - Precio: {current_price:.2f}, Confianza: {confidence:.2f}")
            print(f"  Ganancia: {profit_pct:.2f}%")
            
        # Verificar stop loss y take profit si tenemos posición abierta
        elif position == 1:
            # Stop loss
            if current_price <= stop_loss:
                position = 0
                exit_price = current_price
                
                # Calcular ganancia/pérdida
                profit_pct = (exit_price / entry_price - 1) * 100
                
                # Cantidad original invertida
                original_amount = balance * 0.95
                
                # Calcular nuevo balance
                amount = original_amount * (1 + profit_pct/100)
                balance += amount
                
                signals.append({
                    'time': current_time,
                    'price': current_price,
                    'type': 'stop_loss',
                    'profit_pct': profit_pct
                })
                
                print(f"STOP LOSS: {current_time} - Precio: {current_price:.2f}")
                print(f"  Pérdida: {profit_pct:.2f}%")
                
            # Take profit
            elif current_price >= take_profit:
                position = 0
                exit_price = current_price
                
                # Calcular ganancia/pérdida
                profit_pct = (exit_price / entry_price - 1) * 100
                
                # Cantidad original invertida
                original_amount = balance * 0.95
                
                # Calcular nuevo balance
                amount = original_amount * (1 + profit_pct/100)
                balance += amount
                
                signals.append({
                    'time': current_time,
                    'price': current_price,
                    'type': 'take_profit',
                    'profit_pct': profit_pct
                })
                
                print(f"TAKE PROFIT: {current_time} - Precio: {current_price:.2f}")
                print(f"  Ganancia: {profit_pct:.2f}%")
        
        # Guardar estado actual
        results.append({
            'time': current_time,
            'price': current_price,
            'balance': balance,
            'position': position,
            'signal': signal,
            'confidence': confidence,
            'composite_score': trading_decision.get('composite_score', 0)
        })
    
    # Cerrar posición abierta al final del backtest
    if position == 1:
        last_price = app.data['close'].iloc[-1]
        last_time = app.data.index[-1]
        
        # Calcular ganancia/pérdida
        profit_pct = (last_price / entry_price - 1) * 100
        
        # Cantidad original invertida
        original_amount = balance * 0.95
        
        # Calcular nuevo balance
        amount = original_amount * (1 + profit_pct/100)
        balance += amount
        
        signals.append({
            'time': last_time,
            'price': last_price,
            'type': 'final_close',
            'profit_pct': profit_pct
        })
        
        print(f"CIERRE FINAL: {last_time} - Precio: {last_price:.2f}")
        print(f"  Ganancia: {profit_pct:.2f}%")
    
    # Convertir resultados a DataFrame
    results_df = pd.DataFrame(results)
    signals_df = pd.DataFrame(signals)
    
    # Calcular rendimiento
    total_return = (balance / 1000 - 1) * 100
    
    print("\n" + "="*50)
    print("RESULTADOS DEL BACKTEST (ALGORITMO AVANZADO)")
    print("="*50)
    print(f"Balance inicial: 1000.00 USDT")
    print(f"Balance final: {balance:.2f} USDT")
    print(f"Rendimiento total: {total_return:.2f}%")
    print(f"Número de operaciones: {len(signals_df[signals_df['type'] == 'buy'])}")
    
    # Calcular operaciones ganadoras/perdedoras
    if 'profit_pct' in signals_df.columns:
        winners = len(signals_df[signals_df['profit_pct'] > 0])
        losers = len(signals_df[signals_df['profit_pct'] <= 0])
        print(f"Operaciones ganadoras: {winners}")
        print(f"Operaciones perdedoras: {losers}")
        
        if winners + losers > 0:
            win_rate = winners / (winners + losers) * 100
            print(f"Porcentaje de éxito: {win_rate:.2f}%")
            
            # Rendimiento promedio por operación
            avg_win = signals_df[signals_df['profit_pct'] > 0]['profit_pct'].mean() if winners > 0 else 0
            avg_loss = signals_df[signals_df['profit_pct'] <= 0]['profit_pct'].mean() if losers > 0 else 0
            print(f"Ganancia promedio: {avg_win:.2f}%")
            print(f"Pérdida promedio: {avg_loss:.2f}%")
            
            # Relación riesgo/recompensa
            if avg_loss != 0:
                risk_reward = abs(avg_win / avg_loss)
                print(f"Relación riesgo/recompensa: {risk_reward:.2f}")
    
    print("="*50)
    
    # Visualizar resultados
    plt.figure(figsize=(16, 10))
    
    # Subplot 1: Precio y señales
    plt.subplot(2, 1, 1)
    plt.plot(app.data.index[50:], app.data['close'].iloc[50:], label='Precio de cierre', alpha=0.7)
    
    # Marcar señales
    if not signals_df.empty:
        # Compras
        buys = signals_df[signals_df['type'] == 'buy']
        if not buys.empty:
            plt.scatter(buys['time'], buys['price'], color='green', label='Compra', marker='^', s=100)
        
        # Ventas
        sells = signals_df[signals_df['type'] == 'sell']
        if not sells.empty:
            plt.scatter(sells['time'], sells['price'], color='red', label='Venta', marker='v', s=100)
        
        # Stop loss
        stops = signals_df[signals_df['type'] == 'stop_loss']
        if not stops.empty:
            plt.scatter(stops['time'], stops['price'], color='purple', label='Stop Loss', marker='x', s=100)
        
        # Take profit
        takes = signals_df[signals_df['type'] == 'take_profit']
        if not takes.empty:
            plt.scatter(takes['time'], takes['price'], color='blue', label='Take Profit', marker='o', s=100)
    
    plt.title(f'Backtest Algoritmo Avanzado - {app.symbol}')
    plt.ylabel('Precio')
    plt.legend()
    plt.grid(True)
    
    # Subplot 2: Balance
    plt.subplot(2, 1, 2)
    plt.plot(results_df['time'], results_df['balance'], label='Balance', color='orange')
    plt.title('Evolución del Balance')
    plt.ylabel('USDT')
    plt.grid(True)
    
    # Ajustar formato de fechas en eje x
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    plt.savefig(f'backtest_advanced_{app.symbol.replace("/", "_")}_{datetime.now().strftime("%Y%m%d")}.png')
    plt.show()
    
    # También mostrar gráfico de los componentes del score compuesto
    if 'composite_score' in results_df.columns:
        plt.figure(figsize=(16, 8))
        plt.plot(results_df['time'], results_df['composite_score'], label='Score Compuesto', color='blue')
        plt.axhline(y=0.2, color='green', linestyle='--', label='Umbral Compra')
        plt.axhline(y=-0.2, color='red', linestyle='--', label='Umbral Venta')
        plt.title('Evolución del Score Compuesto del Algoritmo')
        plt.ylabel('Score (-1 a 1)')
        plt.legend()
        plt.grid(True)
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        plt.savefig(f'composite_score_{app.symbol.replace("/", "_")}_{datetime.now().strftime("%Y%m%d")}.png')
        plt.show()
    
    # Guardar los resultados en un CSV para análisis posterior
    results_df.to_csv(f'backtest_results_{app.symbol.replace("/", "_")}_{datetime.now().strftime("%Y%m%d")}.csv')
    if not signals_df.empty:
        signals_df.to_csv(f'backtest_signals_{app.symbol.replace("/", "_")}_{datetime.now().strftime("%Y%m%d")}.csv')
    
    print("Backtest completado y resultados guardados.")

if __name__ == "__main__":
    main()