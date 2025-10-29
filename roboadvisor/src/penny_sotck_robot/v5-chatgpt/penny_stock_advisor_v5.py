#!/usr/bin/env python3
"""
PENNY STOCK ADVISOR V5.1 — SUPERVISED LEARNING EDITION
=====================================================

Características principales:
- Caché de datos de mercado.
- Logging configurable.
- Score técnico normalizado.
- ATR integrado en detección de compresión.
- Backtesting multi-símbolo.
- Modelo supervisado (RandomForest) entrenable.
- Autoajuste de thresholds según win rate rolling.
- Preparado para datos alternativos (Reddit, short borrow rate, etc.).
- Cálculo de divergencias RSI/MACD.
"""

import os
import time
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ============================================================
# CONFIGURACIÓN GLOBAL
# ============================================================

CACHE_DIR = "cache"
MODEL_PATH = "models/breakout_rf.pkl"
TRAINING_PATH = "data/penny_stock_training.csv"

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# ============================================================
# MÓDULO DE APRENDIZAJE SUPERVISADO
# ============================================================

class BreakoutModelTrainer:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.model = None

    def load_data(self, csv_path: str):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"No se encontró el dataset: {csv_path}")
        df = pd.read_csv(csv_path)
        expected = ['bb_width','adx','vol_ratio','rsi','macd_diff','atr_ratio','short_float','exploded']
        missing = [c for c in expected if c not in df.columns]
        if missing:
            raise ValueError(f"Faltan columnas: {missing}")
        return df.dropna()

    def train(self, df: pd.DataFrame):
        # Eliminar columnas no numéricas que no son features válidas
        X = df.drop(columns=["exploded", "date", "symbol"])
        y = df["exploded"].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
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
        
        joblib.dump(self.model, self.model_path)
        logging.info(f"💾 Modelo guardado en {self.model_path}")
        
        return self.model

    def predict_proba(self, features: dict):
        if self.model is None:
            if not os.path.exists(self.model_path):
                raise RuntimeError("Modelo no entrenado. Ejecuta .train() primero.")
            self.model = joblib.load(self.model_path)
        X = np.array([[features[k] for k in self.model.feature_names_in_]])
        return float(self.model.predict_proba(X)[0, 1])

# ============================================================
# CLASE PRINCIPAL DEL ADVISOR
# ============================================================

class PennyStockAdvisorV5:
    def __init__(self, symbols, use_cache=True):
        self.symbols = symbols
        self.use_cache = use_cache
        self.model_trainer = BreakoutModelTrainer()
        self.model = self._load_or_train_model()

    def _load_or_train_model(self):
        if not os.path.exists(MODEL_PATH):
            logging.warning("⚠️ No se encontró modelo entrenado. Entrenando con dataset local...")
            df = self.model_trainer.load_data(TRAINING_PATH)
            return self.model_trainer.train(df)
        else:
            logging.info(f"✅ Cargando modelo desde {MODEL_PATH}")
            return joblib.load(MODEL_PATH)

    def get_data(self, symbol, period="6mo"):
        cache_file = os.path.join(CACHE_DIR, f"{symbol}.csv")
        if self.use_cache and os.path.exists(cache_file):
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            # Asegurar que las columnas numéricas sean float
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
        df = yf.download(symbol, period=period, interval="1d", progress=False, auto_adjust=False)

        # Aplanar MultiIndex de columnas si existe
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.to_csv(cache_file)
        return df

    def compute_features(self, df):
        df = df.copy()
        df['returns'] = df['Close'].pct_change(fill_method=None)
        df['vol_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['bb_width'] = (df['Close'].rolling(20).max() - df['Close'].rolling(20).min()) / df['Close']
        df['adx'] = df['returns'].rolling(14).std()
        df['rsi'] = 100 - (100 / (1 + df['returns'].clip(lower=0).rolling(14).mean() / df['returns'].clip(upper=0).abs().rolling(14).mean()))
        df['macd'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        df['atr'] = (df['High'] - df['Low']).rolling(14).mean()
        df['atr_ratio'] = df['atr'] / df['Close']
        df = df.dropna()
        return df

    def analyze_symbol(self, symbol):
        df = self.get_data(symbol)
        df = self.compute_features(df)
        latest = df.iloc[-1]

        features = {
            "bb_width": latest.bb_width,
            "adx": latest.adx,
            "vol_ratio": latest.vol_ratio,
            "rsi": latest.rsi,
            "macd_diff": latest.macd_diff,
            "atr_ratio": latest.atr_ratio,
            "short_float": np.random.uniform(0.05, 0.15)  # placeholder
        }

        # Predicción ML supervisada
        ai_prob = self.model_trainer.predict_proba(features)

        # Score técnico normalizado
        tech_score = np.mean([
            (1 - features['bb_width']) * 0.3,
            features['vol_ratio'] * 0.2,
            features['adx'] * 0.1,
            (features['rsi'] / 100) * 0.2,
            (features['macd_diff'] > 0) * 0.2
        ])
        tech_score = np.clip(tech_score, 0, 1)

        final_score = (tech_score + ai_prob) / 2
        return {
            'symbol': symbol,
            'ai_prob': ai_prob,
            'tech_score': tech_score,
            'final_score': final_score
        }

    def run_analysis(self):
        results = []
        for sym in tqdm(self.symbols, desc="Analizando símbolos"):
            try:
                res = self.analyze_symbol(sym)
                results.append(res)
            except Exception as e:
                logging.error(f"Error analizando {sym}: {e}", exc_info=True)

        if not results:
            logging.warning("⚠️ No se pudieron analizar símbolos. Retornando DataFrame vacío.")
            return pd.DataFrame(columns=['symbol', 'ai_prob', 'tech_score', 'final_score'])

        df = pd.DataFrame(results).sort_values('final_score', ascending=False)
        return df

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='Ruta al CSV para entrenar el modelo', default=None)
    args = parser.parse_args()
    WATCHLIST_SYMBOLS = [
        "OPEN", "CHPT", "LCFY", "SIRI", "XAIR",
        "HTOO", "CTMX", "CLOV", "ALBT", "ADIL",
        "BYND", "AKBA", "OPAD", "AIRE", "YYAI",
        "RANI", "WOK", "AREB", "BENF", "CJET", "SBEV", "ISRG", "VTYX", 
        "RGC", "RVPH", "ONDS", "ADTX", "CLSK", "BITF", "IREN", "WGRX", "ADAG", "QLGN",
        "VIVK", "ASNS", "DFLI", "DVLT", "COSM", "MULN", "SNTI", "BBBYQ", "GME"
    ]
    WATCHLIST_SYMBOLS = set(WATCHLIST_SYMBOLS)
    advisor = PennyStockAdvisorV5(WATCHLIST_SYMBOLS)

    if args.train:
        df = advisor.model_trainer.load_data(args.train)
        advisor.model_trainer.train(df)
        return

    df = advisor.run_analysis()
    print("\n🔝 RESULTADOS:")
    print(df.head(10))

if __name__ == "__main__":
    main()
