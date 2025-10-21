import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
import warnings
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

# Ignorar advertencias para una salida más limpia
warnings.filterwarnings('ignore')

# --- MÓDULO 1: OBTENCIÓN DE DATOS ---

def obtener_tickers_sp500():
    """
    Obtiene la lista de tickers del S&P 500 desde Wikipedia.
    """
    print("Obteniendo la lista de tickers del S&P 500...")
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tabla = pd.read_html(url)
        df_tickers = tabla[0]
        tickers = df_tickers['Symbol'].tolist()
        # Reemplazar tickers que yfinance no reconoce bien
        tickers = [t.replace('.', '-') for t in tickers]
        print(f"Se encontraron {len(tickers)} tickers.")
        return tickers
    except Exception as e:
        print(f"No se pudo obtener la lista de tickers: {e}")
        # Usar una lista estática como fallback
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'JPM', 'JNJ', 'V', 'XOM', 'WMT']

def preparar_datos_para_deduccion(df):
    """
    Prepara los datos para el modelo deductivo, aplicando transformaciones logarítmicas.
    """
    df_transformed = df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns
    
    for col in numeric_cols:
        df_transformed[f'log_{col}'] = np.log(df_transformed[col].apply(lambda x: 1e-9 if x <= 0 else x))
        
    return df_transformed

# --- MÓDULO 2: MOTOR DEL BACKTESTING ---

def entrenar_modelos_anuales(tickers, year, cache_financieros):
    """
    Entrena un modelo de regresión para cada ticker usando los dos años fiscales anteriores.
    """
    modelos = {}
    training_years = [year - 3, year - 2]
    
    print(f"\nEntrenando modelos para el año {year} usando datos de {training_years[0]}-{training_years[1]}...")
    
    for ticker in tqdm(tickers, desc=f"Entrenando modelos para {year}"):
        if ticker not in cache_financieros:
            continue
            
        df_hist = cache_financieros[ticker]
        
        training_data = df_hist[df_hist['Year'].isin(training_years)]
        
        if len(training_data) < 2:
            continue
        
        df_train_log = preparar_datos_para_deduccion(training_data.dropna())
        
        feature_cols = [col for col in df_train_log.columns if col.startswith('log_') and col != 'log_MarketCap']
        target_col = 'log_MarketCap'

        if not all(c in df_train_log.columns for c in feature_cols) or target_col not in df_train_log.columns:
            continue

        model = LinearRegression()
        model.fit(df_train_log[feature_cols], df_train_log[target_col])
        
        modelos[ticker] = {'model': model, 'features': feature_cols}
        
    return modelos

def ejecutar_backtest(start_date, end_date, tickers, umbral_decision=0.15):
    """
    Ejecuta la simulación de backtesting completa.
    """
    # 1. Descargar todos los datos de precios necesarios
    print("\nDescargando datos de precios históricos para las acciones...")
    
    raw_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
    
    if raw_data.empty:
        print("Error: La descarga de datos de acciones no devolvió resultados.")
        return pd.DataFrame(), pd.DataFrame()

    precios = raw_data['Close'] if isinstance(raw_data.columns, pd.MultiIndex) else raw_data[['Close']]

    precios = precios.dropna(axis=1, how='all')
    tickers = [t for t in tickers if t in precios.columns] 
    
    print("\nDescargando datos de precios históricos para el benchmark (S&P 500)...")
    try:
        benchmark_df = yf.download('^GSPC', start=start_date, end=end_date, auto_adjust=True)
        if benchmark_df.empty:
            raise ValueError("El DataFrame del benchmark está vacío.")
    except Exception as e:
        print(f"Error fatal: No se pudo descargar el benchmark ^GSPC. {e}")
        return pd.DataFrame(), pd.DataFrame()

    # 2. Descargar todos los datos financieros necesarios
    print("\nDescargando datos financieros históricos (puede tardar)...")
    cache_financieros = {}
    start_year_fin = pd.to_datetime(start_date).year - 3
    end_year_fin = pd.to_datetime(end_date).year - 1
    
    for ticker in tqdm(tickers, desc="Descargando datos financieros"):
        try:
            stock = yf.Ticker(ticker)
            financials = stock.financials
            balance_sheet = stock.balance_sheet
            shares = stock.info.get('sharesOutstanding')
            if not shares: continue

            data_anual = []
            for year in range(start_year_fin, end_year_fin + 1):
                fs_cols = [c for c in financials.columns if c.year == year]
                bs_cols = [c for c in balance_sheet.columns if c.year == year]
                if not fs_cols or not bs_cols: continue

                fs = financials[fs_cols[0]]
                bs = balance_sheet[bs_cols[0]]
                
                price_hist_series = precios.loc[precios.index.year == year, ticker]
                if price_hist_series.empty: continue
                price_hist = price_hist_series.iloc[-1]
                
                data_anual.append({
                    'Year': year, 'MarketCap': price_hist * shares,
                    'TotalRevenue': fs.get('Total Revenue'), 'EBITDA': fs.get('EBITDA'),
                    'NetIncome': fs.get('Net Income'), 'TotalAssets': bs.get('Total Assets'),
                    'TotalLiabilitiesNetMinorityInterest': bs.get('Total Liabilities Net Minority Interest')
                })
            cache_financieros[ticker] = pd.DataFrame(data_anual)
        except Exception:
            continue

    # 3. Inicialización de la cartera
    capital_inicial = 100000
    cartera = {'cash': capital_inicial, 'holdings': {}}
    historial_cartera = []
    modelos_por_ano = {}

    # 4. Bucle principal de simulación
    print("\nIniciando simulación de backtesting...")
    fechas_comunes = precios.index.intersection(benchmark_df.index)
    
    for fecha in tqdm(fechas_comunes, desc="Simulando día a día"):
        ano_actual = fecha.year
        
        if ano_actual not in modelos_por_ano:
            modelos_por_ano[ano_actual] = entrenar_modelos_anuales(tickers, ano_actual, cache_financieros)
        
        modelos_actuales = modelos_por_ano[ano_actual]
        
        # --- LÓGICA DE VENTA ---
        for ticker_en_cartera in list(cartera['holdings'].keys()):
            if ticker_en_cartera not in modelos_actuales or ticker_en_cartera not in cache_financieros:
                continue
            
            latest_financials_df = preparar_datos_para_deduccion(cache_financieros[ticker_en_cartera].iloc[-1:])
            modelo_info = modelos_actuales[ticker_en_cartera]
            
            if not all(c in latest_financials_df.columns for c in modelo_info['features']):
                continue

            pred_log_cap = modelo_info['model'].predict(latest_financials_df[modelo_info['features']])
            valor_deducido = np.exp(pred_log_cap[0])
            
            precio_actual = precios.loc[fecha, ticker_en_cartera]
            last_price_series = precios.loc[precios.index.year == ano_actual-1, ticker_en_cartera]
            if last_price_series.empty: continue
            
            shares = cache_financieros[ticker_en_cartera]['MarketCap'].iloc[-1] / last_price_series.iloc[-1]
            market_cap_actual = precio_actual * shares
            
            potencial = (valor_deducido - market_cap_actual) / market_cap_actual
            
            if potencial < umbral_decision:
                cantidad = cartera['holdings'][ticker_en_cartera]
                cartera['cash'] += cantidad * precio_actual
                del cartera['holdings'][ticker_en_cartera]

        # --- LÓGICA DE COMPRA ---
        posiciones_a_llenar = 10 - len(cartera['holdings'])
        if posiciones_a_llenar > 0 and cartera['cash'] > 0:
            posibles_compras = []
            for ticker in tickers:
                if ticker in cartera['holdings'] or ticker not in modelos_actuales or ticker not in cache_financieros:
                    continue
                
                latest_financials_df = preparar_datos_para_deduccion(cache_financieros[ticker].iloc[-1:])
                modelo_info = modelos_actuales[ticker]
                
                if not all(c in latest_financials_df.columns for c in modelo_info['features']):
                    continue

                pred_log_cap = modelo_info['model'].predict(latest_financials_df[modelo_info['features']])
                valor_deducido = np.exp(pred_log_cap[0])
                
                precio_actual = precios.loc[fecha, ticker]
                last_price_series = precios.loc[precios.index.year == ano_actual-1, ticker]
                if last_price_series.empty: continue
                
                shares = cache_financieros[ticker]['MarketCap'].iloc[-1] / last_price_series.iloc[-1]
                market_cap_actual = precio_actual * shares
                
                potencial = (valor_deducido - market_cap_actual) / market_cap_actual
                
                if potencial > umbral_decision:
                    posibles_compras.append({'ticker': ticker, 'potencial': potencial})
            
            mejores_compras = sorted(posibles_compras, key=lambda x: x['potencial'], reverse=True)
            inversion_por_accion = cartera['cash'] / posiciones_a_llenar if posiciones_a_llenar > 0 else 0
            
            for i in range(min(posiciones_a_llenar, len(mejores_compras))):
                ticker_a_comprar = mejores_compras[i]['ticker']
                precio_compra = precios.loc[fecha, ticker_a_comprar]
                if precio_compra <= 0: continue
                cantidad = inversion_por_accion / precio_compra
                
                cartera['holdings'][ticker_a_comprar] = cantidad
                cartera['cash'] -= inversion_por_accion
        
        valor_holdings = sum(cantidad * precios.loc[fecha, ticker] for ticker, cantidad in cartera['holdings'].items())
        valor_total = cartera['cash'] + valor_holdings
        historial_cartera.append({'Date': fecha, 'Portfolio_Value': valor_total})

    return pd.DataFrame(historial_cartera).set_index('Date'), benchmark_df


# --- MÓDULO 3: VISUALIZACIÓN Y RESULTADOS ---

def graficar_resultados(historial_cartera, benchmark, capital_inicial=100000):
    """
    Calcula la rentabilidad y grafica los resultados.
    """
    # --- CORRECCIÓN DE ERROR: Unir con una Serie para evitar conflictos de niveles ---
    # Extraer la serie 'Close' del benchmark para asegurar una unión de 1 nivel.
    if 'Close' not in benchmark.columns:
        print("Error crítico: La columna 'Close' no se encuentra en los datos del benchmark.")
        return
    
    benchmark_close_series = benchmark['Close']

    # Unir el historial de la cartera (DataFrame) con la serie de precios del benchmark.
    datos_comunes = historial_cartera.join(benchmark_close_series, how='inner')
    
    # Verificar que la columna del benchmark se unió correctamente antes de continuar
    if 'Close' not in datos_comunes.columns:
        print("Error crítico: No se pudo unir la columna 'Close' del benchmark.")
        print("Columnas disponibles:", datos_comunes.columns)
        return
        
    # Renombrar la columna para mayor claridad en los cálculos siguientes
    datos_comunes.rename(columns={'Close': 'Benchmark_Price'}, inplace=True)
    
    # Ahora, calcular los retornos usando el nuevo nombre de columna 'Benchmark_Price'
    datos_comunes['Portfolio_Return'] = (datos_comunes['Portfolio_Value'] / capital_inicial) - 1
    datos_comunes['Benchmark_Return'] = (datos_comunes['Benchmark_Price'] / datos_comunes['Benchmark_Price'].iloc[0]) - 1

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(15, 8))
    
    plt.plot(datos_comunes.index, datos_comunes['Portfolio_Return'] * 100, label='Rentabilidad del Algoritmo Deductivo', color='royalblue', linewidth=2)
    plt.plot(datos_comunes.index, datos_comunes['Benchmark_Return'] * 100, label='Rentabilidad del S&P 500 (^GSPC)', color='darkorange', linestyle='--', linewidth=2)
    
    plt.title('Backtest: Algoritmo Deductivo vs. S&P 500', fontsize=18)
    plt.xlabel('Fecha', fontsize=12)
    plt.ylabel('Rentabilidad Acumulada (%)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    rentabilidad_final_algo = datos_comunes['Portfolio_Return'].iloc[-1]
    rentabilidad_final_sp500 = datos_comunes['Benchmark_Return'].iloc[-1]
    
    print("\n" + "="*50)
    print("              RESULTADOS FINALES DEL BACKTEST")
    print("="*50)
    print(f"Rentabilidad final del Algoritmo: {rentabilidad_final_algo:.2%}")
    print(f"Rentabilidad final del S&P 500:   {rentabilidad_final_sp500:.2%}")
    
    if rentabilidad_final_algo > rentabilidad_final_sp500:
        print("\n¡El algoritmo ha batido al mercado en el período de prueba!")
    else:
        print("\nEl algoritmo no ha logrado superar al mercado en el período de prueba.")
    print("="*50)

    plt.show()

# --- EJECUCIÓN PRINCIPAL ---

if __name__ == '__main__':
    start_date_backtest = '2022-01-01'
    end_date_backtest = '2023-12-31'
    
    sp500_tickers = obtener_tickers_sp500()
    
    historial_cartera_df, benchmark_df = ejecutar_backtest(start_date_backtest, end_date_backtest, sp500_tickers)
    
    if not historial_cartera_df.empty and benchmark_df is not None and not benchmark_df.empty:
        graficar_resultados(historial_cartera_df, benchmark_df)
