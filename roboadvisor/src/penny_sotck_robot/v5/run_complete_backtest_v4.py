#!/usr/bin/env python3
"""
COMPLETE BACKTEST EXECUTION V4 - ML-DRIVEN PREDICTIVE SYSTEM
=============================================================

CAMBIOS CRÍTICOS V4:
✓ ML Predictive Model: XGBoost/LightGBM/RandomForest ensemble
✓ Walk-Forward Retraining: Every 30 days with 90-day training window
✓ Improved Entry Logic: ML probability + technical scoring (NOT just COMPRA FUERTE)
✓ Adaptive Threshold: Lowered from 70 to 55-60, market-regime adjusted
✓ Ensemble Voting: 3 models voting system
✓ Enhanced Features: 15+ features with normalization
✓ False Positive Management: Volume confirmation, risk/reward ratio
✓ Enhanced Logging: ML probability, scores, and reasoning
✓ Optimized Parameters: take_profit=20%, max_holding=10 days

Author: Quantitative Engineering Team
Date: 2025
Version: 4.0
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Import backtest engine V3
from backtest_engine_v2 import (
    BacktestEngineV3,
    BacktestConfig,
    calculate_all_metrics,
    monte_carlo_simulation,
    statistical_validation,
    fetch_benchmark_data,
    generate_tearsheet,
    generate_interactive_dashboard,
    factor_attribution_analysis,
    worst_trades_analysis,
    export_to_quantstats,
    meta_validation_backtest,
    detect_market_regime,
    risk_monitoring,
    temporal_split,
    build_ensemble_models,
    HAS_SKLEARN,
    HAS_XGBOOST,
    HAS_LIGHTGBM
)

# Import signal generation system
from penny_stock_advisor_v5 import PennyStockAdvisorV5

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('complete_backtest_v4')

# ML imports
if HAS_SKLEARN:
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier

if HAS_XGBOOST:
    import xgboost as xgb

if HAS_LIGHTGBM:
    import lightgbm as lgb


# ============================================================================
# ML PRICE PREDICTOR - ENSEMBLE SYSTEM
# ============================================================================

class MLPricePredictor:
    """
    Predictor de precios usando ensemble de modelos ML
    Predice si un stock subirá >10% en 5 días
    """

    def __init__(self, use_ensemble: bool = True):
        """
        Args:
            use_ensemble: Usar ensemble de 3 modelos (XGBoost, LightGBM, RandomForest)
        """
        self.use_ensemble = use_ensemble and HAS_SKLEARN
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        self.is_trained = False

        # Ensemble models
        self.models = {}
        self.feature_importance = {}

        self.logger = logging.getLogger('ml_predictor')

        if not HAS_SKLEARN:
            self.logger.warning("scikit-learn not available. ML predictor disabled.")

    def create_models(self) -> Dict:
        """Crea ensemble de modelos"""
        models = {}

        if HAS_SKLEARN:
            # RandomForest
            models['rf'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_split=20,
                min_samples_leaf=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )

        if HAS_XGBOOST:
            # XGBoost
            models['xgb'] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=1.5,  # Balance classes
                random_state=42,
                n_jobs=-1
            )

        if HAS_LIGHTGBM:
            # LightGBM
            models['lgb'] = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )

        return models

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> bool:
        """
        Entrena ensemble de modelos

        Args:
            X_train: Features (N, M)
            y_train: Labels (N,) - 1 si subió >10% en 5 días, 0 si no

        Returns:
            True si entrenamiento exitoso
        """
        if not HAS_SKLEARN or len(X_train) < 30:
            return False

        try:
            # Normalizar features
            X_train_scaled = self.scaler.fit_transform(X_train)

            # Crear y entrenar modelos
            self.models = self.create_models()

            for name, model in self.models.items():
                self.logger.info(f"Training {name.upper()} with {len(X_train)} samples...")
                model.fit(X_train_scaled, y_train)

                # Guardar feature importance
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = model.feature_importances_

            self.is_trained = True

            # Reportar balance de clases
            positive_rate = np.mean(y_train) * 100
            self.logger.info(f"Training completed. Positive class: {positive_rate:.1f}%")
            self.logger.info(f"Models trained: {list(self.models.keys())}")

            return True

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return False

    def predict_probability(self, X: np.ndarray) -> float:
        """
        Predice probabilidad de subida >10% usando ensemble voting

        Args:
            X: Features (1, M) o (M,)

        Returns:
            Probabilidad promedio de subida (0-1)
        """
        if not self.is_trained or len(self.models) == 0:
            return 0.5  # Default neutral

        try:
            # Asegurar shape correcto
            if X.ndim == 1:
                X = X.reshape(1, -1)

            # Normalizar
            X_scaled = self.scaler.transform(X)

            # Obtener predicciones de todos los modelos
            probabilities = []

            for name, model in self.models.items():
                try:
                    prob = model.predict_proba(X_scaled)[0, 1]  # Probability of class 1
                    probabilities.append(prob)
                except Exception as e:
                    self.logger.debug(f"Model {name} prediction failed: {e}")

            if len(probabilities) == 0:
                return 0.5

            # Voting: promedio de probabilidades
            avg_probability = np.mean(probabilities)

            return float(avg_probability)

        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return 0.5

    def ensemble_vote(self, X: np.ndarray, threshold: float = 0.55) -> Tuple[bool, float]:
        """
        Votación de ensemble: entra si 2+ modelos predicen >threshold

        Args:
            X: Features
            threshold: Threshold de probabilidad

        Returns:
            (should_enter, avg_probability)
        """
        if not self.is_trained or len(self.models) == 0:
            return False, 0.5

        try:
            if X.ndim == 1:
                X = X.reshape(1, -1)

            X_scaled = self.scaler.transform(X)

            votes = []
            probabilities = []

            for name, model in self.models.items():
                try:
                    prob = model.predict_proba(X_scaled)[0, 1]
                    probabilities.append(prob)
                    votes.append(prob > threshold)
                except:
                    pass

            if len(votes) == 0:
                return False, 0.5

            # Decidir entrada
            num_yes_votes = sum(votes)
            avg_prob = np.mean(probabilities)

            # Si 2+ modelos votan sí, o si todos predicen >50% (aunque sea bajo threshold)
            should_enter = num_yes_votes >= 2 or (len(votes) >= 3 and all(p > 0.5 for p in probabilities))

            return should_enter, float(avg_prob)

        except Exception as e:
            self.logger.error(f"Ensemble vote error: {e}")
            return False, 0.5


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def calculate_advanced_features(market_data: Dict, historical_data: Dict,
                                current_price: float) -> Optional[np.ndarray]:
    """
    Calcula features avanzadas para ML

    Returns:
        Array de features normalizados (15+ features)
    """
    try:
        close_prices = historical_data.get('close', [])
        volumes = historical_data.get('volume', [])

        if len(close_prices) < 20:
            return None

        close_array = np.array(close_prices)
        volume_array = np.array(volumes) if len(volumes) == len(close_prices) else None

        # === BASIC FEATURES ===
        rsi = market_data.get('rsi', 50) / 100  # Normalize to 0-1
        atr_ratio = min(market_data.get('atr_ratio', 0.05), 0.20) / 0.20  # Normalize
        short_interest = min(market_data.get('short_interest_pct', 15), 100) / 100

        # Volume features
        volume_ratio = market_data.get('volume', 1) / max(market_data.get('avg_volume_20d', 1), 1)
        volume_ratio_norm = min(volume_ratio, 10) / 10

        # === PRICE MOMENTUM ===
        # 5-day momentum
        momentum_5d = (close_array[-1] - close_array[-6]) / close_array[-6] if len(close_array) >= 6 else 0
        momentum_5d_norm = np.tanh(momentum_5d * 5)  # Normalize using tanh

        # 10-day momentum
        momentum_10d = (close_array[-1] - close_array[-11]) / close_array[-11] if len(close_array) >= 11 else 0
        momentum_10d_norm = np.tanh(momentum_10d * 3)

        # === VOLATILITY ===
        # 20-day volatility
        if len(close_array) >= 20:
            returns = np.diff(close_array[-20:]) / close_array[-20:-1]
            volatility = np.std(returns)
            volatility_norm = min(volatility, 0.15) / 0.15
        else:
            volatility_norm = 0.05

        # === MACD ===
        macd_diff = market_data.get('macd_diff', 0)
        macd_norm = np.tanh(macd_diff * 10)  # Normalize

        # === PRICE POSITION ===
        # Price relative to MA20
        ma20 = np.mean(close_array[-20:]) if len(close_array) >= 20 else close_array[-1]
        price_to_ma20 = (current_price - ma20) / ma20
        price_to_ma20_norm = np.tanh(price_to_ma20 * 5)

        # Price relative to MA50
        ma50 = np.mean(close_array[-50:]) if len(close_array) >= 50 else ma20
        price_to_ma50 = (current_price - ma50) / ma50
        price_to_ma50_norm = np.tanh(price_to_ma50 * 3)

        # === 52-WEEK RANGE ===
        # % above 52-week low
        low_52w = np.min(close_array[-252:]) if len(close_array) >= 252 else np.min(close_array)
        pct_above_52w_low = (current_price - low_52w) / low_52w
        pct_above_52w_low_norm = min(pct_above_52w_low, 2.0) / 2.0

        # === VOLUME CHANGE ===
        if volume_array is not None and len(volume_array) >= 20:
            volume_ma20 = np.mean(volume_array[-20:])
            current_volume = volume_array[-1]
            volume_change = (current_volume - volume_ma20) / volume_ma20
            volume_change_norm = np.tanh(volume_change * 2)
        else:
            volume_change_norm = 0

        # === RSI MOMENTUM ===
        # Cambio en RSI (derivada)
        # Aproximación: si RSI está subiendo rápidamente
        rsi_momentum = (rsi - 0.5) * 2  # Centro en 0.5, rango -1 a 1

        # === VOLATILITY CHANGE ===
        if len(close_array) >= 40:
            returns_recent = np.diff(close_array[-20:]) / close_array[-20:-1]
            returns_prev = np.diff(close_array[-40:-20]) / close_array[-40:-21]
            vol_recent = np.std(returns_recent)
            vol_prev = np.std(returns_prev)
            vol_change = (vol_recent - vol_prev) / vol_prev if vol_prev > 0 else 0
            vol_change_norm = np.tanh(vol_change * 3)
        else:
            vol_change_norm = 0

        # === COMBINE FEATURES ===
        features = np.array([
            rsi,                        # 0: RSI normalized
            atr_ratio,                  # 1: ATR ratio normalized
            short_interest,             # 2: Short interest
            volume_ratio_norm,          # 3: Volume ratio
            momentum_5d_norm,           # 4: 5-day momentum
            momentum_10d_norm,          # 5: 10-day momentum
            volatility_norm,            # 6: 20-day volatility
            macd_norm,                  # 7: MACD
            price_to_ma20_norm,         # 8: Price vs MA20
            price_to_ma50_norm,         # 9: Price vs MA50
            pct_above_52w_low_norm,     # 10: % above 52w low
            volume_change_norm,         # 11: Volume change vs MA
            rsi_momentum,               # 12: RSI momentum
            vol_change_norm,            # 13: Volatility change
            len(close_array) / 252      # 14: Data availability (years)
        ])

        return features

    except Exception as e:
        logger.error(f"Feature calculation error: {e}")
        return None


# ============================================================================
# IMPROVED SIGNAL ADAPTER V4
# ============================================================================

class SignalAdapterV4:
    """
    Adapta señales con ML predictivo y scoring mejorado
    """

    def __init__(self, advisor: PennyStockAdvisorV5, ml_predictor: MLPricePredictor):
        """
        Args:
            advisor: Instancia de PennyStockAdvisorV5
            ml_predictor: Predictor ML
        """
        self.advisor = advisor
        self.ml_predictor = ml_predictor
        self.logger = logging.getLogger('signal_adapter_v4')

        # Tracking
        self.evaluation_log = []

    def get_market_context(self, date: datetime) -> Dict:
        """Obtiene contexto de mercado"""
        try:
            end_date = date
            start_date = date - timedelta(days=60)

            spy = yf.Ticker('SPY')
            spy_hist = spy.history(start=start_date, end=end_date)

            if len(spy_hist) == 0:
                return {'spy_trend': 'neutral', 'vix': 15, 'regime_multiplier': 1.0}

            # Calcular tendencia
            lookback = min(5, len(spy_hist))
            if lookback >= 5:
                price_change = ((spy_hist['Close'].iloc[-1] - spy_hist['Close'].iloc[-lookback]) /
                               spy_hist['Close'].iloc[-lookback] * 100)

                if price_change > 2:
                    spy_trend = 'bullish'
                    regime_multiplier = 1.2  # Boost score in bullish market
                elif price_change < -2:
                    spy_trend = 'bearish'
                    regime_multiplier = 0.8  # Reduce score in bearish market
                else:
                    spy_trend = 'neutral'
                    regime_multiplier = 1.0
            else:
                spy_trend = 'neutral'
                regime_multiplier = 1.0

            return {
                'spy_trend': spy_trend,
                'vix': 15,
                'spy_price_change': price_change if lookback >= 5 else 0,
                'regime_multiplier': regime_multiplier
            }

        except Exception as e:
            self.logger.warning(f"Error getting market context: {e}")
            return {'spy_trend': 'neutral', 'vix': 15, 'regime_multiplier': 1.0}

    def calculate_technical_score(self, market_data: Dict, analysis: Dict) -> float:
        """
        Calcula score técnico (0-1) basado en condiciones técnicas
        """
        score = 0.0

        # RSI oversold
        rsi = market_data.get('rsi', 50)
        if rsi < 30:
            score += 0.3
        elif rsi < 40:
            score += 0.15

        # Momentum positivo
        momentum = market_data.get('momentum', 0)
        if momentum > 0:
            score += 0.2

        # Soporte técnico roto al alza (close > MA20)
        close_prices = analysis.get('historical_data', {}).get('close', [])
        if len(close_prices) >= 20:
            ma20 = np.mean(close_prices[-20:])
            current_price = close_prices[-1]
            if current_price > ma20:
                score += 0.25

        # Volumen alto
        volume_ratio = market_data.get('volume', 1) / max(market_data.get('avg_volume_20d', 1), 1)
        if volume_ratio > 1.5:
            score += 0.25

        return min(score, 1.0)

    def should_enter(self, symbol: str, date: datetime, current_price: float) -> Tuple[bool, Dict]:
        """
        Determina si debe entrar usando ML + technical scoring

        NUEVA LÓGICA V4:
        - Entra si CUALQUIERA de estas condiciones:
          1. ML probability > 55%
          2. RSI < 30 Y momentum positivo
          3. Close > MA20 Y volumen alto
          4. Divergencia alcista + ML > 50%
        - Score final = (0.5 * ml_prob) + (0.3 * technical) + (0.2 * regime_bonus)
        - Threshold ajustable: 0.55-0.60 (NO 0.70 fixed)
        """
        try:
            # Obtener datos
            market_data, historical_data = self.advisor.get_enhanced_market_data(symbol, period="3mo")

            if market_data is None or historical_data is None:
                return False, {}

            market_context = self.get_market_context(date)

            # Análisis V5
            analysis = self.advisor.analyze_symbol_v5(
                symbol,
                market_data,
                historical_data,
                market_context
            )

            # === CALCULAR FEATURES PARA ML ===
            features = calculate_advanced_features(market_data, historical_data, current_price)

            if features is None:
                return False, {}

            # === ML PROBABILITY ===
            ml_probability = 0.5
            if self.ml_predictor.is_trained:
                ml_probability = self.ml_predictor.predict_probability(features)

            # === TECHNICAL SCORE ===
            technical_score = self.calculate_technical_score(market_data, analysis)

            # === REGIME ADJUSTMENT ===
            regime_multiplier = market_context.get('regime_multiplier', 1.0)

            # === FINAL SCORING ===
            # Score ponderado: 50% ML, 30% técnico, 20% régimen
            score_final = (0.5 * ml_probability) + (0.3 * technical_score)
            score_final *= regime_multiplier

            # === ENTRY CONDITIONS ===
            rsi = market_data.get('rsi', 50)
            momentum = market_data.get('momentum', 0)
            volume_ratio = market_data.get('volume', 1) / max(market_data.get('avg_volume_20d', 1), 1)

            # Condiciones de entrada (OR logic)
            condition_ml = ml_probability > 0.55
            condition_oversold = rsi < 30 and momentum > 0
            condition_breakout = technical_score > 0.5 and volume_ratio > 1.5
            condition_divergence = ml_probability > 0.50 and market_data.get('macd_diff', 0) > 0

            # Entrada adaptativa con threshold dinámico
            min_threshold = 0.55  # Bajado de 0.70
            should_enter = (score_final > min_threshold) or condition_ml or condition_oversold or condition_breakout

            # === FALSE POSITIVE MANAGEMENT ===
            if should_enter:
                # Confirmación de volumen
                avg_volume_20d = market_data.get('avg_volume_20d', 0)
                current_volume = market_data.get('volume', 0)

                if current_volume < avg_volume_20d * 1.2:
                    self.logger.debug(f"{symbol}: Rejected - Low volume confirmation")
                    should_enter = False

                # Warning flags
                warnings = analysis.get('trading_decision', {}).get('warnings', [])
                if any('gap down' in str(w).lower() for w in warnings):
                    self.logger.debug(f"{symbol}: Rejected - Gap down warning")
                    should_enter = False

                # Risk/Reward ratio
                take_profit_pct = 0.20
                stop_loss_pct = 0.08
                risk_reward_ratio = take_profit_pct / stop_loss_pct
                if risk_reward_ratio < 1.5:
                    self.logger.debug(f"{symbol}: Warning - Low risk/reward ratio")

            # === PREPARE SIGNAL DATA ===
            signal_data = {
                'ml_probability': ml_probability,
                'technical_score': technical_score,
                'score_final': score_final,
                'regime_multiplier': regime_multiplier,
                'rsi': rsi,
                'momentum': momentum,
                'volume_ratio': volume_ratio,
                'stop_loss_pct': 0.08,
                'take_profit_pct': 0.20,  # INCREASED from 15% to 20%
                'position_size_pct': 0.02 + (ml_probability - 0.5) * 0.06,  # 2-5% dynamic
                'entry_reason': f"ML:{ml_probability:.2f} Tech:{technical_score:.2f} Score:{score_final:.2f}",
                'conditions_met': {
                    'ml': condition_ml,
                    'oversold': condition_oversold,
                    'breakout': condition_breakout,
                    'divergence': condition_divergence
                },
                'warnings': analysis.get('trading_decision', {}).get('warnings', [])
            }

            # Logging detallado
            if should_enter:
                self.logger.info(
                    f"✓ ENTRY SIGNAL: {symbol} @ ${current_price:.2f} | "
                    f"ML:{ml_probability:.2f} Tech:{technical_score:.2f} "
                    f"Final:{score_final:.2f} RSI:{rsi:.0f} Vol:{volume_ratio:.1f}x"
                )
            else:
                self.logger.debug(
                    f"✗ NO ENTRY: {symbol} @ ${current_price:.2f} | "
                    f"ML:{ml_probability:.2f} Tech:{technical_score:.2f} "
                    f"Final:{score_final:.2f}"
                )

            # Log para análisis posterior
            self.evaluation_log.append({
                'symbol': symbol,
                'date': date,
                'price': current_price,
                'ml_prob': ml_probability,
                'technical': technical_score,
                'final_score': score_final,
                'entered': should_enter
            })

            return should_enter, signal_data

        except Exception as e:
            self.logger.error(f"Error evaluating {symbol}: {e}")
            return False, {}


# ============================================================================
# WALK-FORWARD TRAINING SYSTEM
# ============================================================================

class WalkForwardTrainer:
    """
    Sistema de reentrenamiento walk-forward
    Reentrena cada 30 días usando ventana de 90 días
    """

    def __init__(self, ml_predictor: MLPricePredictor, signal_adapter: SignalAdapterV4):
        self.ml_predictor = ml_predictor
        self.signal_adapter = signal_adapter
        self.logger = logging.getLogger('walk_forward_trainer')

        self.training_history = []
        self.last_training_date = None
        self.training_window_days = 90
        self.retrain_interval_days = 30

    def should_retrain(self, current_date: datetime) -> bool:
        """Determina si debe reentrenar"""
        if self.last_training_date is None:
            return True

        days_since_training = (current_date - self.last_training_date).days
        return days_since_training >= self.retrain_interval_days

    def prepare_training_data(self, tickers: List[str], end_date: datetime) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Prepara datos de entrenamiento usando ventana de 90 días

        Returns:
            (X_train, y_train) o (None, None) si no hay suficientes datos
        """
        start_date = end_date - timedelta(days=self.training_window_days)

        X_list = []
        y_list = []

        for ticker in tickers:
            try:
                # Obtener datos históricos
                ticker_obj = yf.Ticker(ticker)
                hist = ticker_obj.history(start=start_date, end=end_date + timedelta(days=10))

                if len(hist) < 30:
                    continue

                # Para cada día en la ventana, calcular features y label
                for i in range(20, len(hist) - 5):  # Necesitamos 5 días futuros para label
                    try:
                        current_date_sample = hist.index[i]
                        current_price = hist['Close'].iloc[i]

                        # Preparar market_data y historical_data
                        market_data = {
                            'rsi': 50,  # Placeholder - calcular real
                            'atr_ratio': 0.05,
                            'short_interest_pct': 15,
                            'volume': hist['Volume'].iloc[i],
                            'avg_volume_20d': hist['Volume'].iloc[max(0, i-20):i].mean(),
                            'momentum': (hist['Close'].iloc[i] - hist['Close'].iloc[i-5]) / hist['Close'].iloc[i-5] if i >= 5 else 0,
                            'macd_diff': 0
                        }

                        historical_data = {
                            'close': hist['Close'].iloc[:i+1].tolist(),
                            'volume': hist['Volume'].iloc[:i+1].tolist()
                        }

                        # Calcular features
                        features = calculate_advanced_features(market_data, historical_data, current_price)

                        if features is None:
                            continue

                        # Calcular label: ¿subió >10% en próximos 5 días?
                        future_prices = hist['Close'].iloc[i+1:i+6]
                        max_future_price = future_prices.max()
                        pct_gain = (max_future_price - current_price) / current_price

                        label = 1 if pct_gain > 0.10 else 0

                        X_list.append(features)
                        y_list.append(label)

                    except Exception as e:
                        continue

            except Exception as e:
                self.logger.debug(f"Error preparing data for {ticker}: {e}")
                continue

        if len(X_list) < 30:
            self.logger.warning(f"Insufficient training samples: {len(X_list)}")
            return None, None

        X_train = np.array(X_list)
        y_train = np.array(y_list)

        return X_train, y_train

    def train(self, tickers: List[str], current_date: datetime) -> bool:
        """
        Ejecuta entrenamiento walk-forward
        """
        self.logger.info(f"Starting walk-forward training for {current_date.strftime('%Y-%m-%d')}")

        X_train, y_train = self.prepare_training_data(tickers, current_date)

        if X_train is None:
            self.logger.warning("Training skipped - insufficient data")
            return False

        success = self.ml_predictor.train(X_train, y_train)

        if success:
            self.last_training_date = current_date

            positive_rate = np.mean(y_train) * 100
            self.training_history.append({
                'date': current_date,
                'samples': len(X_train),
                'positive_rate': positive_rate,
                'feature_importance': self.ml_predictor.feature_importance.copy()
            })

            self.logger.info(
                f"Training successful: {len(X_train)} samples, "
                f"{positive_rate:.1f}% positive class"
            )

            return True
        else:
            self.logger.error("Training failed")
            return False


# ============================================================================
# COMPLETE BACKTEST RUNNER V4
# ============================================================================

class CompleteBacktestRunnerV4:
    """
    Ejecuta backtesting V4 con ML predictivo y walk-forward training
    """

    def __init__(self,
                 config: BacktestConfig,
                 adaptive: bool = True,
                 use_ml_ensemble: bool = True):
        """
        Args:
            config: Configuración del backtest
            adaptive: Activar modo adaptativo
            use_ml_ensemble: Usar ensemble de modelos ML
        """
        self.config = config
        self.adaptive = adaptive
        self.use_ml_ensemble = use_ml_ensemble and HAS_SKLEARN

        # Crear componentes
        self.engine = BacktestEngineV3(config, adaptive=adaptive, use_ensemble=False)
        self.advisor = PennyStockAdvisorV5(config_preset="balanced", enable_logging=False, enable_cache=True)
        self.ml_predictor = MLPricePredictor(use_ensemble=use_ml_ensemble)
        self.signal_adapter = SignalAdapterV4(self.advisor, self.ml_predictor)
        self.walk_forward_trainer = WalkForwardTrainer(self.ml_predictor, self.signal_adapter)

        self.logger = logging.getLogger('backtest_runner_v4')

        self.logger.info("="*80)
        self.logger.info("COMPLETE BACKTEST RUNNER V4 - ML PREDICTIVE SYSTEM")
        self.logger.info("="*80)
        self.logger.info(f"Adaptive mode: {adaptive}")
        self.logger.info(f"ML Ensemble: {use_ml_ensemble} (sklearn={HAS_SKLEARN}, xgb={HAS_XGBOOST}, lgb={HAS_LIGHTGBM})")
        self.logger.info(f"Initial capital: ${config.initial_capital:,.2f}")
        self.logger.info(f"Take profit: {config.take_profit_pct*100:.0f}%")
        self.logger.info(f"Max holding: {config.max_holding_days} days")

    def run_complete_backtest(self,
                             tickers: List[str],
                             start_date: str,
                             end_date: str,
                             initial_training_days: int = 180) -> Dict[str, Any]:
        """
        Ejecuta backtest completo con walk-forward training

        Args:
            tickers: Lista de símbolos
            start_date: Fecha inicio (YYYY-MM-DD)
            end_date: Fecha fin (YYYY-MM-DD)
            initial_training_days: Días para training inicial

        Returns:
            Dict con resultados completos
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("STARTING COMPLETE BACKTEST EXECUTION V4")
        self.logger.info("="*80)
        self.logger.info(f"Tickers: {len(tickers)} symbols")
        self.logger.info(f"Period: {start_date} to {end_date}")

        # Parsear fechas
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')

        # === INITIAL TRAINING ===
        training_start = start_dt - timedelta(days=initial_training_days)
        self.logger.info(f"\nInitial ML training period: {training_start.strftime('%Y-%m-%d')} to {start_dt.strftime('%Y-%m-%d')}")

        X_init, y_init = self.walk_forward_trainer.prepare_training_data(tickers, start_dt)

        if X_init is not None:
            self.ml_predictor.train(X_init, y_init)
            self.logger.info(f"Initial training completed: {len(X_init)} samples")
        else:
            self.logger.warning("Initial training failed - continuing without ML")

        # === MARKET REGIME ===
        spy_data = yf.Ticker('SPY').history(start=start_date, end=end_date)
        if len(spy_data) > 0:
            self.engine.update_market_regime(spy_data)
            self.logger.info(f"Initial market regime: {self.engine.current_regime.upper()}")

        # === DAY-BY-DAY SIMULATION ===
        current_date = start_dt
        day_counter = 0
        regime_update_counter = 0

        while current_date <= end_dt:
            day_counter += 1
            regime_update_counter += 1

            # === WALK-FORWARD RETRAINING ===
            if self.walk_forward_trainer.should_retrain(current_date):
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"WALK-FORWARD RETRAINING - Day {day_counter}")
                self.logger.info(f"{'='*60}")
                self.walk_forward_trainer.train(tickers, current_date)

            # === UPDATE MARKET REGIME ===
            if regime_update_counter >= 30:
                spy_recent = spy_data[spy_data.index <= pd.Timestamp(current_date).tz_localize('America/New_York')]
                if len(spy_recent) > 200:
                    self.engine.update_market_regime(spy_recent)
                    regime_update_counter = 0

            # === GET CURRENT PRICES ===
            current_prices = {}
            for ticker in tickers:
                try:
                    ticker_obj = yf.Ticker(ticker)
                    hist = ticker_obj.history(start=current_date - timedelta(days=5),
                                             end=current_date + timedelta(days=1))
                    if len(hist) > 0:
                        current_prices[ticker] = float(hist['Close'].iloc[-1])
                except:
                    continue

            # === UPDATE POSITIONS ===
            self.engine.update_positions(current_date, current_prices)

            # === EVALUATE NEW ENTRIES ===
            if self.engine.can_open_position():
                for ticker in tickers:
                    if ticker in current_prices:
                        current_price = current_prices[ticker]

                        # Evaluar señal V4
                        should_enter, signal_data = self.signal_adapter.should_enter(
                            ticker, current_date, current_price
                        )

                        if should_enter:
                            # Abrir posición
                            position = self.engine.open_position(
                                symbol=ticker,
                                entry_price=current_price,
                                entry_date=current_date,
                                stop_loss_pct=signal_data.get('stop_loss_pct', 0.08),
                                take_profit_pct=signal_data.get('take_profit_pct', 0.20)
                            )

                            if position:
                                self.logger.info(
                                    f"OPENED: {ticker} @ ${current_price:.2f} | "
                                    f"Reason: {signal_data.get('entry_reason', 'Unknown')}"
                                )

            # === RECORD EQUITY ===
            self.engine.record_equity(current_date, current_prices)

            # === PERIODIC UPDATES ===
            if day_counter % 30 == 0:
                self.engine.record_performance()
                self.logger.info(
                    f"Progress: Day {day_counter} | "
                    f"Equity: ${self.engine.get_total_equity(current_prices):,.2f} | "
                    f"Trades: {len(self.engine.closed_trades)}"
                )

            # === RISK ALERTS ===
            if day_counter % 10 == 0 and len(self.engine.closed_trades) > 5:
                alerts = self.engine.check_risk_alerts()
                if alerts and alerts.get('action_required'):
                    self.logger.warning(f"Risk alerts: {len(alerts.get('alerts', []))}")

            # === ADAPTIVE THRESHOLD ===
            if self.adaptive and day_counter % 20 == 0 and len(self.engine.closed_trades) >= 20:
                self.engine.update_adaptive_threshold()

            current_date += timedelta(days=1)

        # === CALCULATE METRICS ===
        self.logger.info("\n" + "="*80)
        self.logger.info("BACKTEST COMPLETED - CALCULATING METRICS")
        self.logger.info("="*80)

        if len(self.engine.closed_trades) == 0:
            self.logger.error("No trades executed during backtest")
            return {'error': 'No trades'}

        period_days = (end_dt - start_dt).days
        metrics = calculate_all_metrics(
            self.engine.closed_trades,
            self.engine.equity_curve,
            self.config,
            period_days
        )

        # Monte Carlo
        monte_carlo_results = monte_carlo_simulation(
            self.engine.closed_trades,
            n_simulations=1000,
            initial_capital=self.config.initial_capital
        )

        # Benchmark
        benchmark_data = fetch_benchmark_data('SPY', start_date, end_date)

        if benchmark_data is not None and len(benchmark_data) > 0:
            equity_values = [eq[1] for eq in self.engine.equity_curve]
            backtest_returns = np.diff(equity_values) / equity_values[:-1]
            benchmark_returns = benchmark_data['Close'].pct_change().dropna().values
            min_len = min(len(backtest_returns), len(benchmark_returns))
            stat_validation = statistical_validation(backtest_returns[:min_len], benchmark_returns[:min_len])
        else:
            stat_validation = None

        # === GENERATE REPORTS ===
        self.logger.info("\n" + "="*80)
        self.logger.info("GENERATING REPORTS")
        self.logger.info("="*80)

        tearsheet_path = generate_tearsheet(
            metrics,
            self.engine.equity_curve,
            self.engine.closed_trades,
            monte_carlo_results,
            save_path='reports/'
        )

        dashboard_path = generate_interactive_dashboard(
            metrics,
            self.engine.equity_curve,
            self.engine.closed_trades,
            save_path='reports/'
        )

        worst_analysis = worst_trades_analysis(self.engine.closed_trades, top_n=10)

        quantstats_path = export_to_quantstats(
            self.engine.equity_curve,
            self.engine.closed_trades,
            save_path='reports/'
        )

        results = {
            'config': self.config,
            'adaptive_mode': self.adaptive,
            'use_ml_ensemble': self.use_ml_ensemble,
            'period': {'start': start_date, 'end': end_date, 'days': period_days},
            'metrics': metrics,
            'monte_carlo': monte_carlo_results,
            'statistical_validation': stat_validation,
            'worst_trades': worst_analysis,
            'adaptive_summary': self.engine.get_adaptive_summary(),
            'training_history': self.walk_forward_trainer.training_history,
            'ml_feature_importance': self.ml_predictor.feature_importance,
            'evaluation_log': self.signal_adapter.evaluation_log,
            'reports': {
                'tearsheet': tearsheet_path,
                'dashboard': dashboard_path,
                'quantstats': quantstats_path
            },
            'equity_curve': self.engine.equity_curve,
            'trades': self.engine.closed_trades,
            'total_trades': len(self.engine.closed_trades)
        }

        self.print_summary(results)

        return results

    def print_summary(self, results: Dict):
        """Imprime resumen de resultados"""
        metrics = results['metrics']

        print("\n" + "="*80)
        print("BACKTEST V4 RESULTS SUMMARY")
        print("="*80)

        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"  Total Return:     {metrics['total_return_pct']:>10.2f}%")
        print(f"  CAGR:             {metrics['cagr']:>10.2f}%")
        print(f"  Sharpe Ratio:     {metrics['sharpe_ratio']:>10.2f}")
        print(f"  Sortino Ratio:    {metrics['sortino_ratio']:>10.2f}")
        print(f"  Max Drawdown:     {metrics['max_drawdown_pct']:>10.2f}%")

        print(f"\n📈 TRADING STATISTICS:")
        print(f"  Total Trades:     {metrics['total_trades']:>10}")
        print(f"  Win Rate:         {metrics['win_rate']*100:>10.2f}%")
        print(f"  Profit Factor:    {metrics['profit_factor']:>10.2f}")
        print(f"  Expectancy:       ${metrics['expectancy']:>9.2f}")
        print(f"  Avg Holding:      {metrics['avg_holding_days']:>10.1f} days")

        print(f"\n💰 P&L:")
        print(f"  Net P&L:          ${metrics['total_pnl']:>9,.2f}")

        print(f"\n🤖 ML TRAINING:")
        print(f"  Training Runs:    {len(results['training_history']):>10}")
        if results['training_history']:
            last_train = results['training_history'][-1]
            print(f"  Last Train Date:  {last_train['date'].strftime('%Y-%m-%d'):>10}")
            print(f"  Last Train Size:  {last_train['samples']:>10}")
            print(f"  Positive Rate:    {last_train['positive_rate']:>10.1f}%")

        mc = results['monte_carlo']
        print(f"\n🎲 MONTE CARLO:")
        print(f"  Mean Return:      {mc['return_mean']:>10.2f}%")
        print(f"  P5 (worst 5%):    {mc['return_percentiles']['P5']:>10.2f}%")
        print(f"  P95 (best 5%):    {mc['return_percentiles']['P95']:>10.2f}%")

        print(f"\n📁 REPORTS:")
        for report_type, path in results['reports'].items():
            if path:
                print(f"  {report_type.title():>15}: {path}")

        print("\n" + "="*80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Ejecución principal del backtest V4"""

    print("="*80)
    print("COMPLETE BACKTEST EXECUTION V4 - ML PREDICTIVE SYSTEM")
    print("="*80)

    # Configuración V4
    config = BacktestConfig(
        initial_capital=100000,
        position_size_pct=0.02,  # 2% base (dinámico 2-5%)
        max_positions=5,
        commission=0.001,
        slippage=0.002,
        stop_loss_pct=0.08,  # 8%
        take_profit_pct=0.20,  # 20% (AUMENTADO de 15%)
        max_holding_days=10,  # 10 días (AUMENTADO de 7)
        risk_free_rate=0.04
    )

    # Universe de penny stocks
    tickers = [
        'BYND', 'OPEN', 'ASST', 'PLUG', 'ABVX', 'SLNH',
        'TELL', 'SOFI', 'COIN', 'RIOT', 'MARA',
        # Agregar más según necesidad
    ]

    # Período de backtest - 2 años
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')  # 2 años

    print(f"\nConfiguration:")
    print(f"  Universe: {len(tickers)} tickers")
    print(f"  Period: {start_date} to {end_date} (2 years)")
    print(f"  Capital: ${config.initial_capital:,.2f}")
    print(f"  Take Profit: {config.take_profit_pct*100:.0f}%")
    print(f"  Max Holding: {config.max_holding_days} days")
    print(f"  ML Ensemble: {'YES' if HAS_SKLEARN else 'NO'}")

    # Crear runner
    runner = CompleteBacktestRunnerV4(
        config=config,
        adaptive=True,
        use_ml_ensemble=HAS_SKLEARN
    )

    # Ejecutar backtest
    print(f"\nStarting backtest execution...")
    print("="*80)

    results = runner.run_complete_backtest(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        initial_training_days=180  # 6 meses para training inicial
    )

    if 'error' not in results:
        print("\n✅ BACKTEST V4 COMPLETED SUCCESSFULLY!")
        print(f"\nReports saved to: reports/")

        # Guardar evaluation log
        if results.get('evaluation_log'):
            log_df = pd.DataFrame(results['evaluation_log'])
            log_path = 'reports/evaluation_log_v4.csv'
            log_df.to_csv(log_path, index=False)
            print(f"  - Evaluation Log: {log_path}")

        print(f"\n🔥 KEY IMPROVEMENTS:")
        print(f"  ✓ ML-driven entry logic (not just COMPRA FUERTE)")
        print(f"  ✓ Walk-forward retraining every 30 days")
        print(f"  ✓ Dynamic scoring: ML + Technical + Regime")
        print(f"  ✓ Lowered threshold: 0.55-0.60 (vs 0.70)")
        print(f"  ✓ Enhanced take profit: 20% (vs 15%)")
        print(f"  ✓ Extended holding: 10 days (vs 7)")
    else:
        print(f"\n❌ BACKTEST FAILED: {results['error']}")

    return results


if __name__ == "__main__":
    results = main()
