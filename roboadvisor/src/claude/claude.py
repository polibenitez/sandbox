"""
Sistema de Valoración Deductiva de Empresas mediante Regresión Múltiple
Basado en los principios de David Ragel Díaz-Jara

Este sistema implementa un algoritmo deductivo que determina el precio de una empresa
utilizando regresión múltiple con transformaciones logarítmicas y selección automática
de variables relevantes.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class DeductiveValuationModel:
    """
    Modelo de valoración deductiva que utiliza regresión múltiple
    para deducir el precio de una empresa a partir de sus estados contables.
    """
    
    def __init__(self, apply_log_transform=True, threshold_coefficient=0.01):
        """
        Inicializa el modelo de valoración deductiva.
        
        Args:
            apply_log_transform (bool): Si aplicar transformaciones logarítmicas
            threshold_coefficient (float): Umbral para considerar un coeficiente como relevante
        """
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.apply_log_transform = apply_log_transform
        self.threshold_coefficient = threshold_coefficient
        self.selected_features = []
        self.coefficients = None
        self.feature_names = []
        
    def prepare_data(self, df, target_col='price', exclude_cols=None):
        """
        Prepara los datos aplicando transformaciones necesarias.
        
        Args:
            df (pd.DataFrame): DataFrame con datos contables y precio
            target_col (str): Nombre de la columna objetivo (precio)
            exclude_cols (list): Columnas a excluir del análisis
            
        Returns:
            X (np.array): Variables independientes transformadas
            y (np.array): Variable objetivo transformada
            feature_names (list): Nombres de las características
        """
        if exclude_cols is None:
            exclude_cols = []
            
        # Separar características y objetivo
        feature_cols = [col for col in df.columns if col != target_col and col not in exclude_cols]
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Eliminar filas con valores cero o negativos si vamos a aplicar log
        if self.apply_log_transform:
            # Para X: reemplazar ceros con un valor pequeño para evitar log(0)
            X = X.replace(0, 1e-10)
            X = X[X > 0].dropna()
            y = y[X.index]
            
            # Aplicar transformación logarítmica
            X_log = np.log(X)
            y_log = np.log(y)
            
            # Crear características adicionales para relaciones multiplicativas/divisivas
            X_enhanced = self._create_interaction_features(X_log)
            
            return X_enhanced.values, y_log.values, X_enhanced.columns.tolist()
        else:
            return X.values, y.values, X.columns.tolist()
    
    def _create_interaction_features(self, X_log):
        """
        Crea características de interacción para capturar relaciones complejas.
        Como log(A*B) = log(A) + log(B) y log(A/B) = log(A) - log(B)
        """
        X_enhanced = X_log.copy()
        
        # Agregar algunas interacciones relevantes (sumas para multiplicaciones)
        for i, col1 in enumerate(X_log.columns):
            for j, col2 in enumerate(X_log.columns[i+1:], i+1):
                # Multiplicación en espacio original = suma en espacio log
                X_enhanced[f'{col1}_mult_{col2}'] = X_log[col1] + X_log[col2]
                
                # División en espacio original = resta en espacio log
                X_enhanced[f'{col1}_div_{col2}'] = X_log[col1] - X_log[col2]
        
        return X_enhanced
    
    def fit(self, X_train, y_train, feature_names):
        """
        Entrena el modelo y deduce las variables relevantes.
        """
        self.feature_names = feature_names
        
        # Normalizar datos
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Entrenar modelo
        self.model.fit(X_scaled, y_train)
        self.coefficients = self.model.coef_
        
        # Deducir variables relevantes (aquellas con coeficientes significativos)
        self._deduce_relevant_features()
        
        return self
    
    def _deduce_relevant_features(self):
        """
        Deduce qué variables son relevantes basándose en los coeficientes.
        Variables con coeficientes cercanos a cero se consideran irrelevantes.
        """
        self.selected_features = []
        
        for i, (coef, feature) in enumerate(zip(self.coefficients, self.feature_names)):
            if abs(coef) > self.threshold_coefficient:
                self.selected_features.append({
                    'feature': feature,
                    'coefficient': coef,
                    'importance': abs(coef)
                })
        
        # Ordenar por importancia
        self.selected_features = sorted(self.selected_features, 
                                      key=lambda x: x['importance'], 
                                      reverse=True)
    
    def predict(self, X_test):
        """
        Realiza predicciones con el modelo entrenado.
        """
        X_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_scaled)
    
    def get_deductive_structure(self):
        """
        Retorna la estructura deductiva del modelo.
        """
        return {
            'intercept': self.model.intercept_,
            'selected_features': self.selected_features,
            'total_features': len(self.feature_names),
            'relevant_features': len(self.selected_features)
        }
    
    def calculate_ar_coefficient(self, y_true, y_pred):
        """
        Calcula el Coeficiente de Regresión de Arrays (AR) de David Ragel.
        Similar al coeficiente de Spearman pero para regresión.
        """
        # Convertir a rankings
        rank_true = pd.Series(y_true).rank()
        rank_pred = pd.Series(y_pred).rank()
        
        # Calcular correlación de Spearman
        correlation = rank_true.corr(rank_pred, method='spearman')
        
        return correlation


class WalkForwardValidator:
    """
    Implementa Walk Forward Testing para validación robusta del modelo.
    """
    
    def __init__(self, n_splits=5, test_size=0.2):
        """
        Args:
            n_splits (int): Número de splits para validación
            test_size (float): Proporción de datos para test en cada split
        """
        self.n_splits = n_splits
        self.test_size = test_size
        
    def validate(self, model, X, y, feature_names):
        """
        Realiza Walk Forward Testing del modelo.
        """
        n_samples = len(X)
        test_samples = int(n_samples * self.test_size)
        train_samples = n_samples - test_samples * self.n_splits
        
        results = []
        
        for i in range(self.n_splits):
            # Definir índices de entrenamiento y prueba
            train_start = 0
            train_end = train_samples + i * test_samples
            test_start = train_end
            test_end = test_start + test_samples
            
            if test_end > n_samples:
                break
                
            # Dividir datos
            X_train = X[train_start:train_end]
            y_train = y[train_start:train_end]
            X_test = X[test_start:test_end]
            y_test = y[test_start:test_end]
            
            # Entrenar y evaluar
            model_copy = DeductiveValuationModel(
                apply_log_transform=model.apply_log_transform,
                threshold_coefficient=model.threshold_coefficient
            )
            model_copy.fit(X_train, y_train, feature_names)
            y_pred = model_copy.predict(X_test)
            
            # Si se aplicó transformación log, revertir para calcular métricas
            if model.apply_log_transform:
                y_test_original = np.exp(y_test)
                y_pred_original = np.exp(y_pred)
            else:
                y_test_original = y_test
                y_pred_original = y_pred
            
            # Calcular métricas
            r2 = r2_score(y_test_original, y_pred_original)
            rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
            ar_coef = model_copy.calculate_ar_coefficient(y_test_original, y_pred_original)
            
            results.append({
                'split': i + 1,
                'r2': r2,
                'rmse': rmse,
                'ar_coefficient': ar_coef,
                'n_features_selected': len(model_copy.selected_features)
            })
            
        return pd.DataFrame(results)


class FinancialDataLoader:
    """
    Carga y prepara datos financieros para el modelo.
    """
    
    @staticmethod
    def load_sample_data():
        """
        Genera datos de ejemplo para demostración.
        """
        np.random.seed(42)
        n_companies = 100
        n_periods = 10
        
        data = []
        
        for _ in range(n_companies * n_periods):
            # Generar datos contables sintéticos
            revenue = np.random.uniform(100, 1000)
            assets = np.random.uniform(500, 5000)
            liabilities = np.random.uniform(100, 3000)
            equity = assets - liabilities
            earnings = revenue * np.random.uniform(0.05, 0.15)
            
            # Precio basado en una combinación de factores (relación deductiva)
            # P = Earnings * PE_ratio + Assets * 0.1 - Liabilities * 0.05
            pe_ratio = np.random.uniform(10, 25)
            price = (earnings * pe_ratio + 
                    assets * 0.1 - 
                    liabilities * 0.05 + 
                    np.random.normal(0, 10))
            
            data.append({
                'revenue': revenue,
                'assets': assets,
                'liabilities': liabilities,
                'equity': equity,
                'earnings': earnings,
                'price': max(price, 1)  # Evitar precios negativos
            })
            
        return pd.DataFrame(data)
    
    @staticmethod
    def load_yahoo_finance_data(ticker, start_date, end_date):
        """
        Carga datos reales de Yahoo Finance.
        """
        stock = yf.Ticker(ticker)
        
        # Obtener datos históricos
        hist_data = stock.history(start=start_date, end=end_date)
        
        # Obtener información financiera
        info = stock.info
        financials = stock.quarterly_financials
        balance_sheet = stock.quarterly_balance_sheet
        
        # Preparar DataFrame combinado (simplificado para ejemplo)
        # En producción, se necesitaría un procesamiento más complejo
        
        return hist_data


# Ejemplo de uso
def main():
    """
    Función principal que demuestra el uso del sistema de valoración deductiva.
    """
    print("=== Sistema de Valoración Deductiva de Empresas ===\n")
    
    # 1. Cargar datos de ejemplo
    print("1. Cargando datos de ejemplo...")
    data = FinancialDataLoader.load_sample_data()
    print(f"   Datos cargados: {len(data)} registros\n")
    
    # 2. Preparar datos
    print("2. Preparando datos con transformaciones logarítmicas...")
    model = DeductiveValuationModel(apply_log_transform=True, threshold_coefficient=0.01)
    X, y, feature_names = model.prepare_data(data, target_col='price')
    print(f"   Características originales: 5")
    print(f"   Características después de interacciones: {len(feature_names)}\n")
    
    # 3. Entrenar modelo
    print("3. Entrenando modelo deductivo...")
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    model.fit(X_train, y_train, feature_names)
    
    # 4. Mostrar estructura deductiva
    print("4. Estructura deductiva descubierta:")
    structure = model.get_deductive_structure()
    print(f"   - Intercepto: {structure['intercept']:.4f}")
    print(f"   - Características totales: {structure['total_features']}")
    print(f"   - Características relevantes: {structure['relevant_features']}")
    print("\n   Top 5 características más importantes:")
    for i, feature in enumerate(structure['selected_features'][:5]):
        print(f"   {i+1}. {feature['feature']}: coef={feature['coefficient']:.4f}")
    
    # 5. Evaluación
    print("\n5. Evaluación del modelo:")
    y_pred = model.predict(X_test)
    
    # Revertir transformación log para métricas
    y_test_original = np.exp(y_test)
    y_pred_original = np.exp(y_pred)
    
    r2 = r2_score(y_test_original, y_pred_original)
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
    ar_coef = model.calculate_ar_coefficient(y_test_original, y_pred_original)
    
    print(f"   - R²: {r2:.4f}")
    print(f"   - RMSE: {rmse:.2f}")
    print(f"   - Coeficiente AR (David Ragel): {ar_coef:.4f}")
    
    # 6. Walk Forward Testing
    print("\n6. Walk Forward Testing para validación robusta:")
    validator = WalkForwardValidator(n_splits=3)
    validation_results = validator.validate(model, X, y, feature_names)
    
    print("\n   Resultados por split:")
    print(validation_results.to_string(index=False))
    print(f"\n   Promedio R²: {validation_results['r2'].mean():.4f}")
    print(f"   Promedio Coeficiente AR: {validation_results['ar_coefficient'].mean():.4f}")


if __name__ == "__main__":
    main()