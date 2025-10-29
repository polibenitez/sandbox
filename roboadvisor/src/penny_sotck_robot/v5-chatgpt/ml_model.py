# ============================================================
#  MODULE: SUPERVISED LEARNING FOR PENNY STOCK ADVISOR V5.1
# ============================================================
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import logging
import os

class BreakoutModelTrainer:
    """
    Clase encargada de entrenar y evaluar el modelo supervisado
    que predice la probabilidad de ruptura (explosión) en penny stocks.
    """

    def __init__(self, model_path="models/breakout_rf.pkl"):
        self.model_path = model_path
        self.model = None

    def load_data(self, csv_path: str):
        """Carga dataset histórico con features técnicos."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"No se encontró el dataset: {csv_path}")
        df = pd.read_csv(csv_path)

        # Espera columnas como:
        # ['symbol','date','bb_width','adx','vol_ratio','rsi','macd_diff','atr_ratio','short_float','exploded']
        expected = ['bb_width','adx','vol_ratio','rsi','macd_diff','atr_ratio','short_float','exploded']
        missing = [c for c in expected if c not in df.columns]
        if missing:
            raise ValueError(f"Faltan columnas: {missing}")

        df = df.dropna()
        return df

    def train(self, df: pd.DataFrame):
        """Entrena el modelo RandomForestClassifier."""
        X = df.drop(columns=["exploded"])
        y = df["exploded"].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_split=4,
            class_weight="balanced_subsample",
            random_state=42
        )

        self.model.fit(X_train, y_train)

        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        logging.info(f"✅ Entrenamiento completado — Accuracy: {acc:.3f}")
        logging.info(f"\n{classification_report(y_test, preds)}")

        # Persistir modelo
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        logging.info(f"💾 Modelo guardado en {self.model_path}")
        return self.model

    def predict_proba(self, features: dict):
        """Predice probabilidad de ruptura."""
        if self.model is None:
            if not os.path.exists(self.model_path):
                raise RuntimeError("Modelo no entrenado. Ejecuta .train() primero.")
            self.model = joblib.load(self.model_path)

        X = np.array([[features[k] for k in self.model.feature_names_in_]])
        return float(self.model.predict_proba(X)[0, 1])

    @staticmethod
    def example_usage():
        """Ejemplo rápido para probar el entrenamiento."""
        trainer = BreakoutModelTrainer()
        df = trainer.load_data("data/penny_stock_training.csv")
        model = trainer.train(df)
        sample = df.drop(columns=["exploded"]).iloc[0].to_dict()
        prob = trainer.predict_proba(sample)
        print(f"Probabilidad de ruptura (ejemplo): {prob:.2%}")
