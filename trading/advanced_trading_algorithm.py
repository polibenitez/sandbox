# Mejora del algoritmo de trading con características avanzadas

import numpy as np
import pandas as pd
import talib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import requests
import json
import logging
import datetime
import os

# Configuración de logging
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = logging.INFO

# Crear directorio para logs si no existe
if not os.path.exists("logs"):
    os.makedirs("logs")

# Configurar logger
logger = logging.getLogger("advanced_trading_algorithm")
logger.setLevel(LOG_LEVEL)

# Handler para archivo
log_filename = (
    f'logs/advanced_trading_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
)
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(LOG_LEVEL)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT))

# Handler para consola
console_handler = logging.StreamHandler()
console_handler.setLevel(LOG_LEVEL)
console_handler.setFormatter(logging.Formatter(LOG_FORMAT))

# Añadir handlers al logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info("Inicializando módulo de algoritmo de trading avanzado")


class AdvancedTradingAlgorithm:
    def __init__(self, base_data=None):
        """
        Inicializar el algoritmo avanzado de trading

        Args:
            base_data: DataFrame con datos históricos ya procesados
        """
        self.data = base_data
        self.ml_model = None
        self.lstm_model = None
        self.scaler = StandardScaler()
        self.stop_loss_pct = 0.03  # 3% por defecto
        self.take_profit_pct = 0.06  # 6% por defecto

    def prepare_features(self, data=None):
        """
        Prepara características para los modelos de machine learning

        Args:
            data: DataFrame con datos (usa self.data si no se proporciona)

        Returns:
            DataFrame con características preparadas
        """
        df = data if data is not None else self.data

        if df is None:
            raise ValueError("No hay datos disponibles para procesar")

        # Asegurarse de que las características básicas estén presentes
        required_cols = [
            "sma_20",
            "sma_50",
            "sma_200",
            "rsi",
            "macd",
            "macd_signal",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "atr",
        ]

        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"La columna {col} no está presente en los datos")

        # Crear características adicionales
        features = pd.DataFrame(index=df.index)

        # 1. Indicadores relativos al precio
        features["close_sma20_ratio"] = df["close"] / df["sma_20"]
        features["close_sma50_ratio"] = df["close"] / df["sma_50"]
        features["close_sma200_ratio"] = df["close"] / df["sma_200"]

        # 2. Distancia a Bandas de Bollinger (normalizada)
        bb_width = df["bb_upper"] - df["bb_lower"]
        features["bb_position"] = (df["close"] - df["bb_lower"]) / bb_width

        # 3. Volatilidad relativa
        features["atr_close_ratio"] = df["atr"] / df["close"]

        # 4. Momentum y fuerza de tendencia
        features["macd_hist"] = df["macd"] - df["macd_signal"]
        features["rsi_momentum"] = df["rsi"].diff(3)

        # 5. Patrones de volumen
        features["volume_sma20"] = df["volume"] / df["volume"].rolling(20).mean()

        # 6. Indicadores adicionales
        # Stochastic Oscillator
        features["stoch_k"], features["stoch_d"] = talib.STOCH(
            df["high"].values,
            df["low"].values,
            df["close"].values,
            fastk_period=14,
            slowk_period=3,
            slowk_matype=0,
            slowd_period=3,
            slowd_matype=0,
        )

        # CMF - Chaikin Money Flow (Indicador de flujo de dinero)
        mfv = (
            ((df["close"] - df["low"]) - (df["high"] - df["close"]))
            / (df["high"] - df["low"])
            * df["volume"]
        )
        features["cmf"] = mfv.rolling(20).sum() / df["volume"].rolling(20).sum()

        # 7. Cruce de medias móviles (señales binarias)
        features["sma_20_above_50"] = (df["sma_20"] > df["sma_50"]).astype(int)
        features["sma_20_above_200"] = (df["sma_20"] > df["sma_200"]).astype(int)
        features["sma_50_above_200"] = (df["sma_50"] > df["sma_200"]).astype(int)

        # 8. Señales de RSI
        features["rsi_oversold"] = (df["rsi"] < 30).astype(int)
        features["rsi_overbought"] = (df["rsi"] > 70).astype(int)

        # 9. Tendencia del precio
        for period in [5, 10, 20]:
            # Tendencia calculada como la pendiente de la SMA
            features[f"trend_{period}"] = (
                df["close"]
                .rolling(period)
                .apply(
                    lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] / x[0],
                    raw=True,
                )
            )

        # 10. Características de retornos históricos
        for period in [1, 3, 5, 10]:
            features[f"return_{period}"] = df["close"].pct_change(period)

        # Llenar NaN
        features = features.fillna(0)

        return features

    def prepare_target(self, horizon=5):
        """
        Prepara la variable objetivo para machine learning

        Args:
            horizon: Número de períodos hacia adelante para calcular retornos

        Returns:
            Series con la variable objetivo (1 para subida, 0 para bajada)
        """
        if self.data is None:
            raise ValueError("No hay datos disponibles")

        # Calcular retorno futuro
        future_return = self.data["close"].shift(-horizon) / self.data["close"] - 1

        # Convertir a variable binaria (1 si sube, 0 si baja)
        target = (future_return > 0).astype(int)

        return target

    def train_random_forest(self, horizon=5, n_estimators=100):
        """
        Entrena un modelo Random Forest para predecir movimientos de precio

        Args:
            horizon: Horizonte de predicción (períodos)
            n_estimators: Número de árboles en el bosque

        Returns:
            Modelo Random Forest entrenado
        """
        # Preparar características y objetivo
        features = self.prepare_features()
        target = self.prepare_target(horizon=horizon)

        # Eliminar filas con NaN
        valid_idx = ~features.isnull().any(axis=1) & ~target.isnull()
        X = features[valid_idx].values
        y = target[valid_idx].values

        # Dividir en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        # Normalizar características
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        # Entrenar modelo
        self.ml_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=5,
            min_samples_split=10,
            random_state=42,
            class_weight="balanced",
        )

        self.ml_model.fit(X_train, y_train)

        # Evaluar modelo
        train_accuracy = self.ml_model.score(X_train, y_train)
        test_accuracy = self.ml_model.score(X_test, y_test)

        print(f"Random Forest - Precisión entrenamiento: {train_accuracy:.4f}")
        print(f"Random Forest - Precisión prueba: {test_accuracy:.4f}")

        # Importancia de características
        feature_importance = pd.DataFrame(
            {
                "feature": features.columns,
                "importance": self.ml_model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        print("Top 10 características más importantes:")
        print(feature_importance.head(10))

        return self.ml_model

    def train_lstm(self, sequence_length=10, epochs=50, batch_size=32):
        """
        Entrena un modelo LSTM para predecir movimientos de precio

        Args:
            sequence_length: Longitud de secuencia para LSTM
            epochs: Número de épocas de entrenamiento
            batch_size: Tamaño del lote

        Returns:
            Modelo LSTM entrenado
        """
        # Preparar características
        features = self.prepare_features()

        # Escalar características
        scaled_features = self.scaler.fit_transform(features.values)

        # Preparar secuencias y objetivo
        X, y = [], []
        for i in range(len(scaled_features) - sequence_length - 5):
            X.append(scaled_features[i : i + sequence_length])
            # Objetivo: retorno en 5 días (binario)
            future_return = (
                self.data["close"].iloc[i + sequence_length + 5]
                / self.data["close"].iloc[i + sequence_length]
                - 1
            )
            y.append(1 if future_return > 0 else 0)

        X = np.array(X)
        y = np.array(y)

        # Dividir en entrenamiento y prueba
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Construir modelo LSTM
        self.lstm_model = Sequential(
            [
                LSTM(
                    50,
                    return_sequences=True,
                    input_shape=(sequence_length, features.shape[1]),
                ),
                Dropout(0.2),
                LSTM(50),
                Dropout(0.2),
                Dense(1, activation="sigmoid"),
            ]
        )

        self.lstm_model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

        # Entrenar modelo
        history = self.lstm_model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1,
        )

        # Evaluar modelo
        _, test_accuracy = self.lstm_model.evaluate(X_test, y_test, verbose=0)
        print(f"LSTM - Precisión prueba: {test_accuracy:.4f}")

        return self.lstm_model

    def predict_random_forest(self, features):
        """
        Realiza predicción con Random Forest

        Args:
            features: Características para la predicción

        Returns:
            Probabilidad de movimiento alcista
        """
        if self.ml_model is None:
            raise ValueError("El modelo Random Forest no ha sido entrenado")

        # Normalizar características
        scaled_features = self.scaler.transform(features.values.reshape(1, -1))

        # Predecir probabilidad
        prob_up = self.ml_model.predict_proba(scaled_features)[0][1]

        return prob_up

    def predict_lstm(self, sequence):
        """
        Realiza predicción con LSTM

        Args:
            sequence: Secuencia de características para la predicción

        Returns:
            Probabilidad de movimiento alcista
        """
        if self.lstm_model is None:
            raise ValueError("El modelo LSTM no ha sido entrenado")

        # Normalizar secuencia
        scaled_sequence = self.scaler.transform(sequence)

        # Redimensionar para LSTM
        scaled_sequence = scaled_sequence.reshape(
            1, scaled_sequence.shape[0], scaled_sequence.shape[1]
        )

        # Predecir probabilidad
        prob_up = self.lstm_model.predict(scaled_sequence)[0][0]

        return float(prob_up)

    def get_sentiment_score(self, symbol):
        """
        Obtiene puntuación de sentimiento desde API externa

        Args:
            symbol: Símbolo de la criptomoneda (ej. 'BTC')

        Returns:
            Puntuación de sentimiento (-1 a 1)
        """
        # En un caso real, se conectaría a una API de sentimiento
        # Por ejemplo, utilizando la API de Crypto Fear & Greed Index
        try:
            # Simular llamada a API
            # En un caso real: response = requests.get(f"https://api.sentiment.com/{symbol}")

            # Simulación de respuesta
            sentiment_values = {
                "BTC": 0.35,
                "ETH": 0.28,
                "SOL": 0.42,
                "ADA": -0.15,
                "DOT": 0.05,
            }

            # Devolver valor predeterminado si el símbolo no está en la lista
            return sentiment_values.get(symbol, 0.0)

        except Exception as e:
            print(f"Error al obtener sentimiento: {e}")
            return 0.0  # Valor neutral por defecto

    def get_onchain_metrics(self, symbol):
        """
        Obtiene métricas on-chain para criptomonedas

        Args:
            symbol: Símbolo de la criptomoneda

        Returns:
            Dict con métricas on-chain
        """
        # En un caso real, se conectaría a una API de análisis on-chain
        # Por ejemplo, Glassnode, CryptoQuant, etc.
        try:
            # Simular llamada a API
            # En un caso real: response = requests.get(f"https://api.onchain.com/{symbol}")

            # Simulación de respuesta
            if symbol == "BTC":
                return {
                    "exchange_inflow": 3250.5,
                    "exchange_outflow": 4120.8,
                    "active_addresses": 1050000,
                    "hashrate": 289.5,  # EH/s
                    "mining_difficulty": 55.6,  # T
                    "sopr": 1.05,  # Spent Output Profit Ratio
                }
            elif symbol == "ETH":
                return {
                    "exchange_inflow": 12500.3,
                    "exchange_outflow": 11890.2,
                    "active_addresses": 650000,
                    "gas_price": 25.4,  # Gwei
                    "staking_rate": 0.32,
                    "defi_tvl": 15.6,  # Miles de millones USD
                }
            else:
                return {}

        except Exception as e:
            print(f"Error al obtener métricas on-chain: {e}")
            return {}

    def calculate_position_size(
        self, confidence, volatility, capital=1000.0, max_risk_pct=0.02
    ):
        """
        Calcula el tamaño de posición óptimo basado en confianza y volatility

        Args:
            confidence: Nivel de confianza en la señal (0-1)
            volatility: Volatilidad actual (ATR)
            capital: Capital disponible
            max_risk_pct: Máximo porcentaje de capital a arriesgar por operación

        Returns:
            Tamaño de posición recomendado
        """
        # Base de riesgo es el 2% del capital
        base_risk = capital * max_risk_pct

        # Ajustar según confianza (0.5 = neutral, 1.0 = máxima confianza)
        confidence_factor = (confidence - 0.5) * 2  # Convierte 0.5-1.0 a 0.0-1.0

        # Limitar entre 0.2 y 1.5 (20% a 150% del riesgo base)
        multiplier = max(0.2, min(1.5, 1.0 + confidence_factor))

        # Calcular cantidad arriesgada
        risk_amount = base_risk * multiplier

        # Calcular stop loss como múltiplo de ATR
        stop_loss_distance = volatility * 2.5

        # Calcular tamaño de posición
        position_size = risk_amount / stop_loss_distance

        return position_size

    def set_risk_management_levels(self, entry_price, signal, atr):
        """
        Establece niveles de stop loss y take profit

        Args:
            entry_price: Precio de entrada
            signal: Dirección (1 para largo, -1 para corto)
            atr: ATR actual

        Returns:
            Diccionario con niveles de stop loss y take profit
        """
        # Stop loss es 2.5 * ATR o 3% del precio, lo que sea menor
        atr_stop = atr * 2.5
        pct_stop = entry_price * self.stop_loss_pct
        stop_distance = min(atr_stop, pct_stop)

        # Take profit es 2 veces la distancia del stop loss
        tp_distance = stop_distance * 2

        # Calcular niveles según dirección
        if signal > 0:  # Posición larga
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + tp_distance
        else:  # Posición corta
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - tp_distance

        # Toma de ganancias escalonada
        take_profit_1 = (
            entry_price + (stop_distance * 1.5)
            if signal > 0
            else entry_price - (stop_distance * 1.5)
        )
        take_profit_2 = (
            entry_price + (stop_distance * 3.0)
            if signal > 0
            else entry_price - (stop_distance * 3.0)
        )

        return {
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "take_profit_levels": [
                {"price": take_profit_1, "percentage": 0.5},
                {"price": take_profit_2, "percentage": 0.5},
            ],
        }

    def analyze_higher_timeframe(self, data_higher_tf, lookback=20):
        """
        Analiza tendencia en timeframe superior

        Args:
            data_higher_tf: DataFrame con datos de timeframe superior
            lookback: Número de períodos a analizar

        Returns:
            Puntuación de tendencia (-1 a 1)
        """
        if data_higher_tf is None or len(data_higher_tf) < lookback:
            return 0

        # Obtener subconjunto de datos recientes
        recent_data = data_higher_tf.iloc[-lookback:]

        # Indicadores de tendencia
        sma_short = recent_data["close"].rolling(window=5).mean().iloc[-1]
        sma_long = recent_data["close"].rolling(window=20).mean().iloc[-1]

        # Comprobar si precio está sobre/debajo medias móviles
        price_above_short = recent_data["close"].iloc[-1] > sma_short
        price_above_long = recent_data["close"].iloc[-1] > sma_long

        # Pendiente de la media móvil larga
        sma_slope = (
            sma_long - recent_data["close"].rolling(window=20).mean().iloc[-5]
        ) / recent_data["close"].rolling(window=20).mean().iloc[-5]

        # Calcular puntuación de tendencia
        trend_score = 0

        # Factor 1: Posición relativa a medias móviles
        if price_above_short and price_above_long:
            trend_score += 0.5
        elif price_above_short and not price_above_long:
            trend_score += 0.25
        elif not price_above_short and not price_above_long:
            trend_score -= 0.5

        # Factor 2: Dirección de la media móvil
        trend_score += min(0.5, max(-0.5, sma_slope * 10))  # Escalar pendiente

        return trend_score

    def advanced_trading_algorithm(self, current_data, higher_tf_data=None):
        """
        Implementación del algoritmo avanzado de trading

        Args:
            current_data: DataFrame con datos del timeframe actual
            higher_tf_data: DataFrame con datos de timeframe superior

        Returns:
            Dict con señal, confianza y niveles de gestión de riesgo
        """
        if current_data is None or len(current_data) < 200:
            return {"signal": 0, "confidence": 0}

        # Obtener filas actuales y previas
        current_row = current_data.iloc[-1]
        prev_row = current_data.iloc[-2]

        # Preparar características para predicción
        features = self.prepare_features(current_data).iloc[-1]

        # 1. Analizar tendencia en timeframe superior (si está disponible)
        higher_tf_trend = 0
        if higher_tf_data is not None:
            higher_tf_trend = self.analyze_higher_timeframe(higher_tf_data)

        # 2. Obtener señales de estrategias base
        # Estrategia 1: Cruce de medias móviles
        signal_ma_cross = 0
        if (
            current_row["sma_20"] > current_row["sma_50"]
            and prev_row["sma_20"] <= prev_row["sma_50"]
        ):
            signal_ma_cross = 1
        elif (
            current_row["sma_20"] < current_row["sma_50"]
            and prev_row["sma_20"] >= prev_row["sma_50"]
        ):
            signal_ma_cross = -1

        # Estrategia 2: Breakout de Bollinger Bands
        signal_bb = 0
        if (
            current_row["close"] <= current_row["bb_lower"]
            and current_row["volume"]
            > current_data["volume"].rolling(20).mean().iloc[-1] * 1.5
            and current_row["rsi"] < 30
        ):
            signal_bb = 1
        elif (
            current_row["close"] >= current_row["bb_upper"] and current_row["rsi"] > 70
        ):
            signal_bb = -1

        # Estrategia 3: Divergencias RSI
        signal_rsi_div = 0
        # Divergencia alcista: precio hace mínimos más bajos pero RSI hace mínimos más altos
        if (
            current_data["close"].iloc[-3] > current_data["close"].iloc[-1]
            and current_data["rsi"].iloc[-3] < current_data["rsi"].iloc[-1]
            and current_data["close"].iloc[-1] < current_data["sma_50"].iloc[-1]
            and current_data["rsi"].iloc[-1] < 40
        ):
            signal_rsi_div = 1
            logger.debug("Señal: Divergencia alcista RSI")
        # Divergencia bajista: precio hace máximos más altos pero RSI hace máximos más bajos
        elif (
            current_data["close"].iloc[-3] < current_data["close"].iloc[-1]
            and current_data["rsi"].iloc[-3] > current_data["rsi"].iloc[-1]
            and current_data["close"].iloc[-1] > current_data["sma_50"].iloc[-1]
            and current_data["rsi"].iloc[-1] > 60
        ):
            signal_rsi_div = -1
            logger.debug("Señal: Divergencia bajista RSI")

        # 3. Obtener predicciones de modelos ML (si están entrenados)
        ml_prob = 0.5  # Neutral por defecto
        lstm_prob = 0.5  # Neutral por defecto

        if self.ml_model is not None:
            try:
                ml_prob = self.predict_random_forest(features)
                logger.debug(f"Predicción Random Forest: {ml_prob:.4f}")
            except Exception as e:
                logger.error(f"Error en predicción Random Forest: {e}")

        if self.lstm_model is not None:
            try:
                # Preparar secuencia para LSTM (últimos 10 períodos)
                sequence = self.prepare_features(current_data).iloc[-10:].values
                lstm_prob = self.predict_lstm(sequence)
                logger.debug(f"Predicción LSTM: {lstm_prob:.4f}")
            except Exception as e:
                logger.error(f"Error en predicción LSTM: {e}")

        # 4. Obtener datos externos (sentimiento y on-chain)
        sentiment = 0
        symbol = current_data.get(
            "symbol", "BTC"
        )  # Obtener símbolo o usar BTC por defecto
        if isinstance(symbol, pd.Series):
            symbol = symbol.iloc[0] if len(symbol) > 0 else "BTC"

        try:
            # Extraer el símbolo base (BTC de BTC/USDT)
            base_symbol = symbol.split("/")[0] if "/" in symbol else symbol

            sentiment = self.get_sentiment_score(base_symbol)
            on_chain = self.get_onchain_metrics(base_symbol)
            logger.debug(f"Sentimiento para {base_symbol}: {sentiment:.4f}")
            logger.debug(f"Métricas on-chain: {len(on_chain)} valores obtenidos")
        except Exception as e:
            logger.error(f"Error al obtener datos externos: {e}")
            on_chain = {}

        # 5. Calcular señal compuesta y confianza
        signal = 0
        confidence = 0.5  # Neutral por defecto

        # Pesos para cada componente
        weights = {
            "ma_cross": 0.15,
            "bb": 0.15,
            "rsi_div": 0.15,
            "ml": 0.20,
            "lstm": 0.15,
            "higher_tf": 0.10,
            "sentiment": 0.05,
            "on_chain": 0.05,
        }

        # Calcular score compuesto (-1 a 1)
        composite_score = (
            weights["ma_cross"] * signal_ma_cross
            + weights["bb"] * signal_bb
            + weights["rsi_div"] * signal_rsi_div
            + weights["ml"] * (ml_prob * 2 - 1)  # Convertir 0-1 a -1 a 1
            + weights["lstm"] * (lstm_prob * 2 - 1)
            + weights["higher_tf"] * higher_tf_trend
        )

        # Añadir sentimiento (si está disponible)
        composite_score += weights["sentiment"] * sentiment

        # Añadir factores on-chain (si están disponibles)
        if "sopr" in on_chain:  # Spent Output Profit Ratio para BTC
            # SOPR > 1 significa que en promedio las monedas se están vendiendo con beneficio
            # SOPR < 1 significa que se están vendiendo con pérdida
            sopr_score = (on_chain["sopr"] - 1) * 2  # Escalar
            composite_score += weights["on_chain"] * min(1, max(-1, sopr_score))
            logger.debug(f"Factor SOPR: {sopr_score:.4f}")
        elif "staking_rate" in on_chain:  # Tasa de staking para ETH
            # Mayor staking = más bullish
            staking_score = (
                on_chain["staking_rate"] - 0.25
            ) * 4  # Escalar (0.25-0.5 a 0-1)
            composite_score += weights["on_chain"] * min(1, max(-1, staking_score))
            logger.debug(f"Factor staking: {staking_score:.4f}")

        logger.info(f"Score compuesto calculado: {composite_score:.4f}")

        # Determinar señal basándose en score compuesto
        if composite_score > 0.2:
            signal = 1  # Comprar
            confidence = 0.5 + min(0.5, composite_score)  # Escalar a 0.5-1.0
            logger.info(f"SEÑAL DE COMPRA generada con confianza {confidence:.4f}")
        elif composite_score < -0.2:
            signal = -1  # Vender
            confidence = 0.5 + min(0.5, abs(composite_score))  # Escalar a 0.5-1.0
            logger.info(f"SEÑAL DE VENTA generada con confianza {confidence:.4f}")
        else:
            logger.info("No se genera señal (score compuesto dentro de zona neutral)")

        # 6. Verificar filtro de tendencia
        # Cancelar señales de compra en tendencia bajista fuerte
        if (
            signal == 1
            and higher_tf_trend < -0.5
            and current_row["close"] < current_row["sma_200"]
        ):
            logger.warning("Filtro de tendencia activado: tendencia bajista fuerte")
            # Mantener señal pero reducir confianza
            original_confidence = confidence
            confidence = max(0.5, confidence - 0.2)
            logger.debug(
                f"Confianza reducida: {original_confidence:.4f} -> {confidence:.4f}"
            )

            # Si la confianza cae demasiado, cancelar señal
            if confidence <= 0.55:
                logger.warning("Señal de compra cancelada por filtro de tendencia")
                signal = 0
                confidence = 0.5

        # 7. Calcular niveles de gestión de riesgo
        risk_levels = None
        position_size = 0

        if signal != 0:
            try:
                # Calcular tamaño de posición basado en confianza y volatilidad
                position_size = self.calculate_position_size(
                    confidence=confidence, volatility=current_row["atr"]
                )

                # Establecer stop loss y take profit
                risk_levels = self.set_risk_management_levels(
                    entry_price=current_row["close"],
                    signal=signal,
                    atr=current_row["atr"],
                )

                logger.info(f"Tamaño de posición: {position_size:.6f}")
                logger.info(
                    f"Stop Loss: {risk_levels['stop_loss']:.2f}, Take Profit: {risk_levels['take_profit']:.2f}"
                )
            except Exception as e:
                logger.error(f"Error al calcular gestión de riesgo: {e}")

        # Componentes individuales para análisis
        components = {
            "ma_cross": signal_ma_cross,
            "bb": signal_bb,
            "rsi_div": signal_rsi_div,
            "ml_prob": ml_prob,
            "lstm_prob": lstm_prob,
            "higher_tf_trend": higher_tf_trend,
            "sentiment": sentiment,
        }

        # Registro detallado de todos los componentes
        logger.debug(f"Componentes detallados: {components}")

        # Resultado final
        result = {
            "signal": signal,
            "confidence": confidence,
            "position_size": position_size,
            "risk_levels": risk_levels,
            "composite_score": composite_score,
            "components": components,
        }

        logger.info(
            f"Resultado final del algoritmo: señal={signal}, confianza={confidence:.4f}"
        )

        return result
