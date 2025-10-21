import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import requests

# Ignorar advertencias comunes para una salida más limpia
warnings.filterwarnings('ignore')

# --- MÓDULO 1: ADQUISICIÓN Y PREPARACIÓN DE DATOS (Funciones Originales) ---

def obtener_datos_financieros(tickers, start_year, end_year):
    """
    Descarga datos financieros anuales y la capitalización de mercado para una lista de empresas.
    Utiliza yfinance como fuente de datos.
    
    Args:
        tickers (list): Lista de símbolos de las empresas (ej. ['AAPL', 'MSFT']).
        start_year (int): Año de inicio para la recolección de datos.
        end_year (int): Año de fin para la recolección de datos.
        
    Returns:
        pandas.DataFrame: Un DataFrame con los datos contables y de mercado de cada empresa por año.
    """
    all_data = []
    print(f"Obteniendo datos para {len(tickers)} empresas entre {start_year} y {end_year}...")
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            
            balance_sheet = stock.balance_sheet
            financials = stock.financials
            
            shares_outstanding = stock.info.get('sharesOutstanding')
            if not shares_outstanding:
                print(f"Advertencia: No se encontraron acciones en circulación para {ticker}. Se omite.")
                continue

            for year in range(start_year, end_year + 1):
                cols_balance = [col for col in balance_sheet.columns if col.year == year]
                cols_financials = [col for col in financials.columns if col.year == year]
                
                if not cols_balance or not cols_financials:
                    continue
                
                bs = balance_sheet[cols_balance[0]]
                fs = financials[cols_financials[0]]

                hist = stock.history(start=f"{year}-12-15", end=f"{year+1}-01-15")
                if hist.empty: continue
                
                market_cap = hist['Close'].iloc[0] * shares_outstanding
                if market_cap == 0: continue

                data_point = {
                    'Ticker': ticker,
                    'Year': year,
                    'MarketCap': market_cap,
                    'TotalRevenue': fs.get('Total Revenue'),
                    'EBITDA': fs.get('EBITDA'),
                    'NetIncome': fs.get('Net Income'),
                    'TotalAssets': bs.get('Total Assets'),
                    'TotalLiabilitiesNetMinorityInterest': bs.get('Total Liabilities Net Minority Interest'),
                }
                all_data.append(data_point)
        except Exception as e:
            print(f"Error al procesar {ticker}: {e}")
            
    df = pd.DataFrame(all_data).dropna()
    print("Datos obtenidos y limpiados con éxito.")
    return df

def preparar_datos_para_deduccion(df):
    """
    Prepara los datos para el modelo deductivo, aplicando transformaciones logarítmicas.
    """
    df_transformed = df.copy()
    # Asegurarse de que solo se procesan columnas numéricas que existen
    numeric_cols = df.select_dtypes(include=np.number).columns
    if 'Year' in numeric_cols:
        numeric_cols = numeric_cols.drop('Year')
    
    for col in numeric_cols:
        if col in df_transformed.columns:
            df_transformed[f'log_{col}'] = np.log(df_transformed[col].apply(lambda x: 1e-6 if x <= 0 else x))
        
    return df_transformed


# --- MÓDULO 2: NUEVA FUNCIONALIDAD PARA USO DIARIO ---

def generar_senales_diarias(tickers, umbral_decision=0.10):
    """
    Analiza una lista de tickers para generar señales de 'Comprar' o 'Vender'.
    """
    print("Iniciando análisis para generar señales de Compra/Vender...")
    resultados = []
    
    current_year = datetime.now().year
    # Usamos los dos últimos años fiscales completos para entrenar
    training_years = [current_year - 3, current_year - 2]

    for ticker in tickers:
        try:
            print(f"\nProcesando: {ticker}")
            stock = yf.Ticker(ticker)
            
            financials_hist = stock.financials
            balance_sheet_hist = stock.balance_sheet
            
            training_data = []
            for year in training_years:
                # --- CORRECCIÓN DEL ERROR ---
                # Antes buscaba una fecha fija 'YYYY-01-01'.
                # Ahora busca cualquier columna que corresponda al año fiscal correcto.
                fs_cols = [col for col in financials_hist.columns if col.year == year]
                bs_cols = [col for col in balance_sheet_hist.columns if col.year == year]

                if not fs_cols or not bs_cols:
                    continue # Si no hay datos para este año, se salta al siguiente.

                fs = financials_hist[fs_cols[0]]
                bs = balance_sheet_hist[bs_cols[0]]
                # --- FIN DE LA CORRECCIÓN ---
                
                hist_price_df = stock.history(start=f"{year}-12-31", end=f"{year+1}-01-10")
                if hist_price_df.empty: continue
                
                hist_price = hist_price_df['Close'].iloc[0]
                shares = stock.info.get('sharesOutstanding')
                if not shares: continue
                
                market_cap_hist = hist_price * shares
                
                training_data.append({
                    'MarketCap': market_cap_hist,
                    'TotalRevenue': fs.get('Total Revenue'), 'EBITDA': fs.get('EBITDA'),
                    'NetIncome': fs.get('Net Income'), 'TotalAssets': bs.get('Total Assets'),
                    'TotalLiabilitiesNetMinorityInterest': bs.get('Total Liabilities Net Minority Interest')
                })

            if len(training_data) < 2:
                print(f"  - No hay suficientes datos históricos ({len(training_data)}/2) para entrenar el modelo de {ticker}. Se omite.")
                continue

            df_train = pd.DataFrame(training_data).dropna()
            if len(df_train) < 2:
                 print(f"  - Datos insuficientes tras limpiar NAs para {ticker}. Se omite.")
                 continue
            df_train_log = preparar_datos_para_deduccion(df_train)

            fs_ttm = financials_hist[financials_hist.columns[0]]
            bs_ttm = balance_sheet_hist[balance_sheet_hist.columns[0]]
            current_market_cap = stock.info.get('marketCap')
            if not current_market_cap: continue

            current_data = {
                'TotalRevenue': fs_ttm.get('Total Revenue'), 'EBITDA': fs_ttm.get('EBITDA'),
                'NetIncome': fs_ttm.get('Net Income'), 'TotalAssets': bs_ttm.get('Total Assets'),
                'TotalLiabilitiesNetMinorityInterest': bs_ttm.get('Total Liabilities Net Minority Interest')
            }
            df_current = pd.DataFrame([current_data]).dropna()
            if df_current.empty:
                print(f"  - Faltan datos TTM para {ticker}. Se omite.")
                continue
            df_current_log = preparar_datos_para_deduccion(df_current)

            feature_cols = [col for col in df_train_log.columns if col.startswith('log_') and col != 'log_MarketCap']
            target_col = 'log_MarketCap'
            
            # Asegurarse de que las columnas actuales coinciden con las de entrenamiento
            missing_cols = set(feature_cols) - set(df_current_log.columns)
            if missing_cols:
                print(f"  - Faltan columnas TTM que estaban en el entrenamiento: {missing_cols}. Se omite.")
                continue
            df_current_log = df_current_log[feature_cols]

            model = LinearRegression()
            model.fit(df_train_log[feature_cols], df_train_log[target_col])
            
            predicted_log_cap = model.predict(df_current_log)
            deduced_market_cap = np.exp(predicted_log_cap[0])
            
            diferencia = (deduced_market_cap - current_market_cap) / current_market_cap
            
            if diferencia > umbral_decision:
                signal = 'Comprar'
            elif diferencia < -umbral_decision:
                signal = 'Vender'
            else:
                signal = 'Mantener'
            
            print(f"  - Capitalización Actual: ${current_market_cap:,.0f}")
            print(f"  - Capitalización Deducida: ${deduced_market_cap:,.0f}")
            print(f"  - Potencial: {diferencia:.2%}")
            print(f"  - Señal: {signal}")

            resultados.append({
                'Ticker': ticker,
                'Capitalización Actual': f"${current_market_cap:,.0f}",
                'Capitalización Deducida': f"${deduced_market_cap:,.0f}",
                'Potencial': f"{diferencia:.2%}",
                'Señal': signal
            })

        except Exception as e:
            print(f"  - Ocurrió un error procesando {ticker}: {e}")

    return pd.DataFrame(resultados)

def get_sp500_tickers():
    """
    Obtiene la lista de tickers del S&P 500 desde Wikipedia.
    """
    url = "https://www.nasdaq.com/market-activity/indexes/spx"
    headers = {"User-Agent": "Mozilla/5.0"}
    html = requests.get(url, headers=headers).text

    # Extraer las tablas de la página
    tables = pd.read_html(html)

    # Buscar la tabla que contiene los tickers
    tickers_table = None
    for table in tables:
        if "Symbol" in table.columns:
            tickers_table = table
            break

    if tickers_table is not None:
        tickers = tickers_table["Symbol"].tolist()
        print("Número de tickers:", len(tickers))
        print(tickers[:20])  # muestra los primeros 20
        return tickers
    else:
        print("No se encontró la tabla con tickers en Nasdaq.")
        return []


# --- MÓDULO 3: EJECUCIÓN PRINCIPAL ---

if __name__ == '__main__':
    # Lista de tickers para el análisis diario
    tickers_a_analizar = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'JPM', 'V', 'XOM']
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
    tickers_a_analizar.extend(additional_tickers)
    tickers_a_analizar.extend(get_sp500_tickers())
    # Eliminar duplicados
    tickers_a_analizar = list(set(tickers_a_analizar))
    # Generar las señales
    tabla_de_senales = generar_senales_diarias(tickers_a_analizar, umbral_decision=0.10)
    
    print("\n" + "="*80)
    print("                 RESULTADO FINAL - SEÑALES DE MERCADO")
    print("="*80)
    
    if not tabla_de_senales.empty:
        print("\nTabla de Decisiones:")
        print(tabla_de_senales.to_string())
    else:
        print("\nNo se pudieron generar señales con los datos disponibles.")

