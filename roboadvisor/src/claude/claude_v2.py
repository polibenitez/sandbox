"""
Sistema de Trading Deductivo basado en Valoración por Regresión Múltiple
Aplicación práctica para decisiones diarias de compra/venta

Este sistema descarga datos históricos de 2 años, entrena un modelo deductivo
y genera señales de compra/venta para una lista de tickers.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import concurrent.futures
from tqdm import tqdm

warnings.filterwarnings('ignore')


class DeductiveTradingSystem:
    """
    Sistema de trading que utiliza regresión múltiple deductiva
    para generar señales de compra/venta.
    """
    
    def __init__(self, lookback_days=30, prediction_days=5, 
                 apply_log_transform=True, threshold_return=0.02):
        """
        Args:
            lookback_days (int): Días de historia para calcular indicadores
            prediction_days (int): Días futuros para predecir
            apply_log_transform (bool): Si aplicar transformaciones logarítmicas
            threshold_return (float): Umbral de retorno para señal de compra (2% default)
        """
        self.lookback_days = lookback_days
        self.prediction_days = prediction_days
        self.apply_log_transform = apply_log_transform
        self.threshold_return = threshold_return
        self.models = {}  # Un modelo por ticker
        self.scalers = {}
        
    def download_stock_data(self, ticker, start_date, end_date):
        """
        Descarga datos históricos y calcula indicadores técnicos y fundamentales.
        """
        try:
            # Descargar datos
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)
            
            if hist.empty:
                print(f"No hay datos disponibles para {ticker}")
                return None
            
            # Obtener información fundamental (si está disponible)
            info = stock.info
            
            # Calcular indicadores técnicos
            hist['Returns'] = hist['Close'].pct_change()
            hist['Log_Returns'] = np.log(hist['Close'] / hist['Close'].shift(1))
            
            # Medias móviles
            hist['MA_5'] = hist['Close'].rolling(window=5).mean()
            hist['MA_20'] = hist['Close'].rolling(window=20).mean()
            hist['MA_50'] = hist['Close'].rolling(window=50).mean()
            
            # Ratios de medias móviles (relaciones multiplicativas)
            hist['MA_5_20_Ratio'] = hist['MA_5'] / hist['MA_20']
            hist['MA_20_50_Ratio'] = hist['MA_20'] / hist['MA_50']
            
            # Volatilidad
            hist['Volatility'] = hist['Returns'].rolling(window=20).std()
            
            # RSI
            hist['RSI'] = self._calculate_rsi(hist['Close'])
            
            # Volumen relativo
            hist['Volume_Ratio'] = hist['Volume'] / hist['Volume'].rolling(window=20).mean()
            
            # Momentum
            hist['Momentum_10'] = hist['Close'] / hist['Close'].shift(10)
            hist['Momentum_30'] = hist['Close'] / hist['Close'].shift(30)
            
            # Datos fundamentales (si están disponibles)
            if info:
                # Estos valores son estáticos, los expandimos para todo el DataFrame
                hist['PE_Ratio'] = info.get('trailingPE', np.nan)
                hist['PB_Ratio'] = info.get('priceToBook', np.nan)
                hist['Market_Cap'] = info.get('marketCap', np.nan)
                hist['Dividend_Yield'] = info.get('dividendYield', 0) if info.get('dividendYield') else 0
            
            # Variable objetivo: retorno futuro
            hist['Future_Return'] = hist['Close'].shift(-self.prediction_days) / hist['Close'] - 1
            hist['Signal'] = (hist['Future_Return'] > self.threshold_return).astype(int)
            
            # Eliminar filas con NaN
            hist = hist.dropna()
            
            return hist
            
        except Exception as e:
            print(f"Error descargando datos para {ticker}: {str(e)}")
            return None
    
    def _calculate_rsi(self, prices, period=14):
        """
        Calcula el RSI (Relative Strength Index).
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def prepare_features(self, df):
        """
        Prepara las características para el modelo.
        """
        # Características a usar en el modelo
        feature_cols = [
            'Returns', 'Log_Returns', 'MA_5_20_Ratio', 'MA_20_50_Ratio',
            'Volatility', 'RSI', 'Volume_Ratio', 'Momentum_10', 'Momentum_30'
        ]
        
        # Agregar características fundamentales si están disponibles
        if 'PE_Ratio' in df.columns and not df['PE_Ratio'].isna().all():
            feature_cols.extend(['PE_Ratio', 'PB_Ratio', 'Dividend_Yield'])
        
        # Filtrar solo columnas que existen y no son todas NaN
        available_features = [col for col in feature_cols if col in df.columns 
                            and not df[col].isna().all()]
        
        X = df[available_features].copy()
        
        # Reemplazar infinitos y NaN
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean())
        
        # Aplicar transformación logarítmica si es necesario
        if self.apply_log_transform:
            # Solo aplicar log a columnas positivas
            for col in X.columns:
                if (X[col] > 0).all():
                    X[f'log_{col}'] = np.log(X[col])
        
        return X
    
    def train_model(self, ticker, df):
        """
        Entrena un modelo deductivo para un ticker específico.
        """
        # Preparar características
        X = self.prepare_features(df)
        y = df['Signal']
        
        # Dividir datos (80% train, 20% test)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Normalizar datos
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Entrenar modelo de regresión
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # Evaluar modelo
        y_pred = model.predict(X_test_scaled)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        accuracy = accuracy_score(y_test, y_pred_binary)
        
        # Guardar modelo y scaler
        self.models[ticker] = model
        self.scalers[ticker] = scaler
        
        # Analizar coeficientes para entender la estructura deductiva
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'coefficient': model.coef_,
            'abs_coefficient': np.abs(model.coef_)
        }).sort_values('abs_coefficient', ascending=False)
        
        return {
            'ticker': ticker,
            'accuracy': accuracy,
            'feature_importance': feature_importance,
            'n_samples': len(df)
        }
    
    def predict_signal(self, ticker, df):
        """
        Genera señal de compra/venta para el día actual.
        """
        if ticker not in self.models:
            return None
        
        # Usar los últimos datos disponibles
        latest_data = df.iloc[-1:].copy()
        
        # Preparar características
        X = self.prepare_features(latest_data)
        
        # Normalizar
        X_scaled = self.scalers[ticker].transform(X)
        
        # Predecir
        prediction = self.models[ticker].predict(X_scaled)[0]
        
        # Generar señal
        signal = "COMPRAR" if prediction > 0.5 else "VENDER"
        confidence = abs(prediction - 0.5) * 2  # Convertir a escala 0-1
        
        return {
            'ticker': ticker,
            'signal': signal,
            'confidence': confidence,
            'prediction_value': prediction
        }
    
    def analyze_ticker_list(self, tickers, start_date=None, end_date=None):
        """
        Analiza una lista de tickers y genera señales de trading.
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=730)  # 2 años
        if end_date is None:
            end_date = datetime.now()
        
        results = []
        training_results = []
        
        print(f"Analizando {len(tickers)} tickers...")
        print("=" * 60)
        
        for ticker in tqdm(tickers, desc="Procesando tickers"):
            # Descargar datos
            df = self.download_stock_data(ticker, start_date, end_date)
            
            if df is None or len(df) < 100:
                print(f"Datos insuficientes para {ticker}")
                continue
            
            # Entrenar modelo
            train_result = self.train_model(ticker, df)
            training_results.append(train_result)
            
            # Generar predicción
            signal = self.predict_signal(ticker, df)
            
            if signal:
                results.append(signal)
                
        return pd.DataFrame(results), pd.DataFrame(training_results)


def main():
    """
    Ejemplo de uso del sistema de trading deductivo.
    """
    # Lista de tickers para analizar
    tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',  # Tech giants
        'JPM', 'BAC', 'GS',                       # Bancos
        'JNJ', 'PFE', 'UNH',                      # Healthcare
        'XOM', 'CVX',                             # Energía
        'WMT', 'PG', 'KO'                         # Consumo
    ]
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
    tickers.extend(additional_tickers)
    
    # Eliminar duplicados
    tickers = list(set(tickers))
    
    # Crear sistema de trading
    print("=== Sistema de Trading Deductivo ===\n")
    system = DeductiveTradingSystem(
        lookback_days=30,
        prediction_days=5,
        apply_log_transform=True,
        threshold_return=0.02  # 2% de retorno esperado
    )
    
    # Analizar tickers
    signals_df, training_df = system.analyze_ticker_list(tickers)
    
    # Mostrar resultados
    print("\n=== SEÑALES DE TRADING ===")
    print("=" * 60)
    
    # Ordenar por confianza
    signals_df = signals_df.sort_values('confidence', ascending=False)
    
    # Mostrar señales de COMPRA
    print("\n📈 SEÑALES DE COMPRA (ordenadas por confianza):")
    buy_signals = signals_df[signals_df['signal'] == 'COMPRAR']
    for _, row in buy_signals.iterrows():
        print(f"  {row['ticker']:6s} - Confianza: {row['confidence']:.1%}")
    
    # Mostrar señales de VENTA
    print("\n📉 SEÑALES DE VENTA (ordenadas por confianza):")
    sell_signals = signals_df[signals_df['signal'] == 'VENDER']
    for _, row in sell_signals.iterrows():
        print(f"  {row['ticker']:6s} - Confianza: {row['confidence']:.1%}")
    
    # Mostrar estadísticas del modelo
    print("\n=== ESTADÍSTICAS DEL MODELO ===")
    print("=" * 60)
    print(f"Precisión promedio: {training_df['accuracy'].mean():.1%}")
    print(f"Tickers analizados: {len(training_df)}")
    print(f"Señales de compra: {len(buy_signals)}")
    print(f"Señales de venta: {len(sell_signals)}")
    
    # Guardar resultados en CSV
    output_date = datetime.now().strftime("%Y%m%d")
    signals_df.to_csv(f'trading_signals_{output_date}.csv', index=False)
    print(f"\nResultados guardados en: trading_signals_{output_date}.csv")
    
    return signals_df, training_df


# Función adicional para análisis rápido de un solo ticker
def quick_analysis(ticker):
    """
    Análisis rápido de un ticker individual.
    """
    system = DeductiveTradingSystem()
    
    # Descargar datos
    start_date = datetime.now() - timedelta(days=730)
    df = system.download_stock_data(ticker, start_date, datetime.now())
    
    if df is None:
        print(f"No se pudieron obtener datos para {ticker}")
        return
    
    # Entrenar y predecir
    train_result = system.train_model(ticker, df)
    signal = system.predict_signal(ticker, df)
    
    print(f"\n=== Análisis de {ticker} ===")
    print(f"Señal: {signal['signal']}")
    print(f"Confianza: {signal['confidence']:.1%}")
    print(f"Precisión histórica: {train_result['accuracy']:.1%}")
    
    print("\nTop 5 indicadores más importantes:")
    for i, row in train_result['feature_importance'].head(5).iterrows():
        print(f"  {row['feature']:20s}: {row['coefficient']:8.4f}")


if __name__ == "__main__":
    # Ejecutar análisis principal
    signals, training_stats = main()
    
    # Ejemplo de análisis individual
    # quick_analysis('AAPL')