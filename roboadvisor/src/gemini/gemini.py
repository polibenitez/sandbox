import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Ignorar advertencias comunes para una salida más limpia
warnings.filterwarnings('ignore')

# --- MÓDULO 1: ADQUISICIÓN Y PREPARACIÓN DE DATOS ---

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
            
            # Obtener estados contables anuales
            balance_sheet = stock.balance_sheet
            financials = stock.financials
            
            # Asegurarse de que tenemos datos de acciones en circulación
            shares_outstanding = stock.info.get('sharesOutstanding')
            if not shares_outstanding:
                print(f"Advertencia: No se encontraron acciones en circulación para {ticker}. Se omite.")
                continue

            for year in range(start_year, end_year + 1):
                # Buscar columnas que correspondan al año fiscal
                cols_balance = [col for col in balance_sheet.columns if col.year == year]
                cols_financials = [col for col in financials.columns if col.year == year]
                
                if not cols_balance or not cols_financials:
                    continue
                
                bs = balance_sheet[cols_balance[0]]
                fs = financials[cols_financials[0]]

                # Obtener la capitalización de mercado para el final del año fiscal
                # Se toma el precio de cierre del primer día de trading del año siguiente
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
            
    df = pd.DataFrame(all_data).dropna() # Eliminar filas con datos faltantes
    print("Datos obtenidos y limpiados con éxito.")
    return df

def preparar_datos_para_deduccion(df):
    """
    Prepara los datos para el modelo deductivo, aplicando transformaciones logarítmicas.
    Esto convierte las relaciones multiplicativas en aditivas, que la regresión lineal puede modelar.
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos brutos.
        
    Returns:
        pandas.DataFrame: DataFrame con las variables transformadas (log_).
    """
    df_transformed = df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns.drop('Year')
    
    for col in numeric_cols:
        # FUNDAMENTO CLAVE: Transformación Logarítmica
        # log(A*B) = log(A) + log(B). Esto permite a la regresión lineal (aditiva)
        # modelar relaciones multiplicativas (ej. Valoración por múltiplos).
        # Se manejan valores no positivos reemplazándolos por un número pequeño para evitar errores.
        df_transformed[f'log_{col}'] = np.log(df_transformed[col].apply(lambda x: 1e-6 if x <= 0 else x))
        
    return df_transformed

# --- MÓDULO 2: MÉTRICAS Y VALIDACIÓN ---

def calcular_coeficiente_ar(y_real, y_predicho):
    """
    Calcula el Coeficiente de Regresión de Arrays (AR) de David Ragel.
    Es análogo al coeficiente de correlación de rango de Spearman.
    
    Args:
        y_real (array): Valores reales.
        y_predicho (array): Valores predichos por el modelo.
        
    Returns:
        float: El coeficiente AR, que va de -1 a 1.
    """
    coef, _ = spearmanr(y_real, y_predicho)
    return coef

def ejecutar_walk_forward_testing(df, features, target):
    """
    Implementa la validación "Walk Forward Testing" para asegurar la robustez del modelo.
    Entrena en un período y prueba en el período inmediatamente posterior, avanzando en el tiempo.
    
    Args:
        df (pandas.DataFrame): El DataFrame completo y preparado.
        features (list): Lista de variables de entrada (X).
        target (str): La variable de salida (Y).
        
    Returns:
        tuple: (modelo final entrenado con todos los datos, DataFrame con los resultados de la validación).
    """
    resultados = []
    years = sorted(df['Year'].unique())
    
    if len(years) < 2:
        print("No hay suficientes años de datos para realizar el Walk Forward Testing.")
        return None, None

    print("\nIniciando Validación Walk-Forward...")
    for i in range(1, len(years)):
        train_years = years[:i]
        test_year = years[i]
        
        train_df = df[df['Year'].isin(train_years)]
        test_df = df[df['Year'] == test_year]

        if train_df.empty or test_df.empty:
            continue

        X_train = train_df[features]
        y_train = train_df[target]
        X_test = test_df[features]
        y_test = test_df[target]

        # PROCESO DE CÁLCULO EN "UN SOLO PASO"
        # La regresión lineal calcula los coeficientes de forma directa, no iterativa.
        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        
        # Revertimos la transformación logarítmica para evaluar en la escala original del precio
        y_test_orig = np.exp(y_test)
        y_pred_orig = np.exp(y_pred)

        r2 = r2_score(y_test_orig, y_pred_orig)
        ar_coef = calcular_coeficiente_ar(y_test_orig, y_pred_orig)
        
        print(f"  - Entrenando con {min(train_years)}-{max(train_years)}, Probando en {test_year}: R²={r2:.3f}, AR={ar_coef:.3f}")

        resultados.append({
            'Train_Years': f"{min(train_years)}-{max(train_years)}",
            'Test_Year': test_year,
            'R2_Score': r2,
            'AR_Coefficient': ar_coef
        })

    # Entrenamos un modelo final con todos los datos para deducir la relación general
    final_model = LinearRegression()
    final_model.fit(df[features], df[target])
    
    return final_model, pd.DataFrame(resultados)

# --- MÓDULO 3: EJECUCIÓN Y ANÁLISIS DEDUCTIVO ---

def main():
    """
    Flujo principal de la aplicación.
    """
    # PARÁMETROS
    # Usamos una lista de empresas de diferentes sectores para un modelo más robusto
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'XOM', 'PFE', 'V', 'NVDA', 'TSLA', 'WMT', 'JNJ']
    start_year = 2018
    end_year = 2023

    # 1. OBTENER Y PREPARAR DATOS
    raw_data = obtener_datos_financieros(tickers, start_year, end_year)
    if raw_data.empty:
        print("No se pudieron obtener datos suficientes. El programa terminará.")
        return
        
    datos_preparados = preparar_datos_para_deduccion(raw_data)

    # 2. DEFINIR VARIABLES PARA EL MODELO DEDUCTIVO (en su forma logarítmica)
    target_variable = 'log_MarketCap'
    feature_variables = [
        'log_TotalRevenue', 
        'log_EBITDA', 
        'log_NetIncome', 
        'log_TotalAssets', 
        'log_TotalLiabilitiesNetMinorityInterest'
    ]

    # 3. EJECUTAR VALIDACIÓN Y OBTENER EL MODELO FINAL
    modelo_final, resultados_validacion = ejecutar_walk_forward_testing(
        datos_preparados, feature_variables, target_variable
    )

    if modelo_final is None:
        return

    # 4. ANÁLISIS DE RESULTADOS: LA DEDUCCIÓN
    print("\n" + "="*70)
    print("              ANÁLISIS DEDUCTIVO DEL MODELO FINAL")
    print("="*70)

    # CAPACIDAD DEDUCTIVA: El modelo asigna coeficientes para deducir la "red topológica"
    coeficientes = pd.DataFrame({
        'Variable Financiera (log)': feature_variables,
        'Coeficiente Deductivo (β)': modelo_final.coef_
    }).round(4)
    
    print("\n>> DEDUCCIÓN DE LA ESTRUCTURA TOPOLÓGICA (COEFICIENTES):\n")
    
    formula_parts = [f"({row['Coeficiente Deductivo (β)']} * {row['Variable Financiera (log)']})" for _, row in coeficientes.iterrows()]
    formula = f"log(Precio) ≈ {modelo_final.intercept_:.4f} + " + " + ".join(formula_parts)
    
    print("La Ecuación Deductiva es:")
    print(formula)
    
    print("\nTabla de Coeficientes:")
    print(coeficientes.to_string())
    
    print("\nInterpretación Deductiva:")
    print("- El modelo ha 'deducido' la importancia relativa de cada variable para determinar el precio.")
    print("- Un coeficiente alto (ej. en 'log_EBITDA') indica una fuerte relación positiva.")
    print("- Coeficientes cercanos a cero sugieren que la variable es menos relevante o redundante una vez que las otras ya están en el modelo.")

    # 5. ROBUSTEZ DEL MODELO
    print("\n\n>> RESULTADOS DE LA VALIDACIÓN WALK FORWARD (ROBUSTEZ DEL MODELO):\n")
    if not resultados_validacion.empty:
        print(resultados_validacion.to_string())
        print("\nInterpretación de la Robustez:")
        print("- R² Score: Mide qué parte de la varianza del precio es explicada por el modelo en datos futuros.")
        print("- AR Coefficient: Mide si el modelo ordena correctamente las empresas (1 es perfecto).")
        print("- La consistencia de estas métricas a lo largo de los años indica que la deducción es robusta y no un simple sobreajuste.")

        # Visualización de la robustez
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax1 = plt.subplots(figsize=(14, 7))
        ax1.set_title('Robustez del Modelo a lo Largo del Tiempo (Validación Walk Forward)', fontsize=16, pad=20)
        
        ax1.plot(resultados_validacion['Test_Year'], resultados_validacion['R2_Score'], 'o-', color='dodgerblue', label='R² Score', linewidth=2, markersize=8)
        ax1.set_xlabel('Año de Prueba', fontsize=12)
        ax1.set_ylabel('R² Score', color='dodgerblue', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='dodgerblue')
        ax1.set_ylim(-0.2, 1)
        ax1.axhline(0, color='grey', linestyle='--', linewidth=0.8)

        ax2 = ax1.twinx()
        ax2.plot(resultados_validacion['Test_Year'], resultados_validacion['AR_Coefficient'], 's--', color='darkgreen', label='Coeficiente AR', linewidth=2, markersize=8)
        ax2.set_ylabel('Coeficiente AR (Spearman)', color='darkgreen', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='darkgreen')
        ax2.set_ylim(-0.2, 1)

        fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=2, fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.9])
        plt.show()

if __name__ == '__main__':
    main()
