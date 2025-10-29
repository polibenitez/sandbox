#!/usr/bin/env python3
"""
ML MODEL V5 - MACHINE LEARNING PARA PREDICCIÓN DE BREAKOUTS
============================================================

Features:
- RandomForestClassifier para predicción de explosiones
- Feature engineering automático
- Entrenamiento con histórico
- Probabilidad de ruptura real
"""

import logging
import numpy as np
import pandas as pd
import pickle
import os
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

logger = logging.getLogger('ml_model')


class BreakoutPredictor:
    """
    Modelo ML para predecir si un setup de compresión resultará en breakout real
    """

    def __init__(self, model_path: str = "models/breakout_model.pkl"):
        """
        Args:
            model_path: Ruta para guardar/cargar el modelo
        """
        self.model_path = model_path
        self.model = None
        self.feature_names = [
            'bb_width', 'adx', 'vol_ratio', 'rsi', 'macd_diff',
            'atr_ratio', 'short_float', 'compression_days',
            'volume_dry', 'price_range_pct'
        ]
        self.is_trained = False

        # Intentar cargar modelo existente
        self._load_model()

    def train(self, training_data: pd.DataFrame) -> Dict:
        """
        Entrena el modelo con datos históricos

        Args:
            training_data: DataFrame con columnas de features + 'exploded' (target)

        Returns:
            Dict con métricas de entrenamiento
        """
        logger.info(f"Entrenando modelo con {len(training_data)} samples")

        # Validar que tenemos las columnas necesarias
        required_cols = self.feature_names + ['exploded']
        missing_cols = [col for col in required_cols if col not in training_data.columns]

        if missing_cols:
            logger.error(f"Columnas faltantes en training_data: {missing_cols}")
            return {'error': f'Missing columns: {missing_cols}'}

        # Preparar features y target
        X = training_data[self.feature_names].values
        y = training_data['exploded'].values

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Entrenar RandomForest
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'  # Para manejar desbalance
        )

        logger.info("Entrenando RandomForestClassifier...")
        self.model.fit(X_train, y_train)

        # Evaluar
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        # Métricas
        accuracy = self.model.score(X_test, y_test)
        roc_auc = roc_auc_score(y_test, y_proba)

        # Feature importance
        feature_importance = dict(zip(
            self.feature_names,
            self.model.feature_importances_
        ))

        logger.info(f"Modelo entrenado - Accuracy: {accuracy:.2%}, ROC-AUC: {roc_auc:.3f}")

        # Guardar modelo
        self._save_model()

        self.is_trained = True

        return {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'feature_importance': feature_importance,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }

    def predict(self, features: Dict) -> Dict:
        """
        Predice probabilidad de breakout para un setup

        Args:
            features: Dict con los features del setup

        Returns:
            Dict con predicción y probabilidad
        """
        if not self.is_trained or self.model is None:
            logger.warning("Modelo no entrenado, usando probabilidad por defecto")
            return {
                'prediction': 1,  # Asumir que sí explotará
                'probability': 0.5,
                'confidence': 'low',
                'model_available': False
            }

        # Preparar features en el orden correcto
        try:
            X = np.array([[features.get(f, 0) for f in self.feature_names]])

            # Predecir
            prediction = self.model.predict(X)[0]
            probability = self.model.predict_proba(X)[0][1]  # Prob de clase 1 (exploded=1)

            # Interpretar confianza
            if probability >= 0.7:
                confidence = 'high'
            elif probability >= 0.5:
                confidence = 'medium'
            else:
                confidence = 'low'

            return {
                'prediction': int(prediction),
                'probability': float(probability),
                'confidence': confidence,
                'model_available': True
            }

        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            return {
                'prediction': 1,
                'probability': 0.5,
                'confidence': 'low',
                'model_available': False,
                'error': str(e)
            }

    def _save_model(self):
        """Guarda el modelo en disco"""
        try:
            model_dir = os.path.dirname(self.model_path)
            if model_dir and not os.path.exists(model_dir):
                os.makedirs(model_dir)

            with open(self.model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'feature_names': self.feature_names,
                    'is_trained': self.is_trained
                }, f)

            logger.info(f"Modelo guardado en {self.model_path}")
        except Exception as e:
            logger.error(f"Error guardando modelo: {e}")

    def _load_model(self):
        """Carga el modelo desde disco"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)

                self.model = data['model']
                self.feature_names = data['feature_names']
                self.is_trained = data.get('is_trained', True)

                logger.info(f"Modelo cargado desde {self.model_path}")
            else:
                logger.info("No hay modelo previo, se entrenará uno nuevo")
        except Exception as e:
            logger.warning(f"Error cargando modelo: {e}")
            self.model = None
            self.is_trained = False


def load_training_data_from_csv(filepath: str) -> Optional[pd.DataFrame]:
    """
    Carga datos de entrenamiento desde CSV

    Args:
        filepath: Ruta al archivo CSV

    Returns:
        DataFrame con los datos
    """
    try:
        df = pd.DataFrame(pd.read_csv(filepath))
        logger.info(f"Datos de entrenamiento cargados: {len(df)} filas desde {filepath}")
        return df
    except Exception as e:
        logger.error(f"Error cargando datos de entrenamiento: {e}")
        return None


if __name__ == "__main__":
    # Test del modelo ML
    from logging_config_v5 import setup_logging
    setup_logging(level="INFO")

    # Crear datos de ejemplo
    training_data = pd.read_csv("data/penny_stock_training.csv")
    logger.info(f"Datos de ejemplo creados: {len(training_data)} samples")

    # Expandir el dataset con datos sintéticos para tener más samples
    # (En producción, usarías datos reales)
    expanded_data = []
    for _ in range(20):
        for _, row in training_data.iterrows():
            new_row = row.copy()
            # Agregar ruido aleatorio pequeño
            for col in ['bb_width', 'adx', 'vol_ratio', 'rsi', 'macd_diff',
                       'atr_ratio', 'short_float', 'price_range_pct']:
                new_row[col] = new_row[col] * (1 + np.random.uniform(-0.1, 0.1))
            expanded_data.append(new_row)

    training_df = pd.DataFrame(expanded_data)
    logger.info(f"Dataset expandido: {len(training_df)} samples")

    # Entrenar modelo
    predictor = BreakoutPredictor()
    metrics = predictor.train(training_df)

    print("\n" + "="*70)
    print("RESULTADOS DEL ENTRENAMIENTO")
    print("="*70)
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
    print(f"\nFeature Importance:")
    for feature, importance in sorted(metrics['feature_importance'].items(),
                                     key=lambda x: x[1], reverse=True):
        print(f"  {feature:20s}: {importance:.3f}")

    # Test de predicción
    test_features = {
        'bb_width': 0.07,
        'adx': 19.0,
        'vol_ratio': 3.0,
        'rsi': 58,
        'macd_diff': 0.004,
        'atr_ratio': 0.015,
        'short_float': 0.16,
        'compression_days': 8,
        'volume_dry': 1,
        'price_range_pct': 6.0
    }

    prediction = predictor.predict(test_features)
    print(f"\n" + "="*70)
    print("TEST DE PREDICCIÓN")
    print("="*70)
    print(f"Predicción: {'EXPLODE' if prediction['prediction'] == 1 else 'NO EXPLODE'}")
    print(f"Probabilidad: {prediction['probability']:.2%}")
    print(f"Confianza: {prediction['confidence'].upper()}")
