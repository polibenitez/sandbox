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
import time
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

# Configuración de logging avanzada
LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
)
LOG_LEVEL = logging.INFO

# Crear estructura de directorios para logs
log_dirs = ["logs", "logs/trades", "logs/performance", "logs/errors"]
for directory in log_dirs:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Configurar logger principal
logger = logging.getLogger("advanced_trading_algorithm")
logger.setLevel(LOG_LEVEL)
logger.propagate = False  # Evitar duplicación de logs

# Handler para archivo principal (rotación por tamaño)
main_log_file = f'logs/trading_{datetime.datetime.now().strftime("%Y%m%d")}.log'
main_handler = RotatingFileHandler(
    main_log_file,
    maxBytes=10 * 1024 * 1024,
    backupCount=5,  # 10MB por archivo, 5 archivos máximo
)
main_handler.setLevel(LOG_LEVEL)
main_handler.setFormatter(logging.Formatter(LOG_FORMAT))

# Handler específico para errores (rotación diaria)
error_log_file = "logs/errors/errors.log"
error_handler = TimedRotatingFileHandler(
    error_log_file,
    when="midnight",
    interval=1,
    backupCount=30,  # Rotación diaria, 30 días
)
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(logging.Formatter(LOG_FORMAT))

# Handler para operaciones de trading
trade_log_file = f'logs/trades/trades_{datetime.datetime.now().strftime("%Y%m%d")}.log'
trade_handler = RotatingFileHandler(
    trade_log_file, maxBytes=5 * 1024 * 1024, backupCount=10
)
trade_handler.setLevel(logging.INFO)
trade_handler.setFormatter(logging.Formatter(LOG_FORMAT))

# Handler para rendimiento
perf_log_file = "logs/performance/performance.log"
perf_handler = RotatingFileHandler(
    perf_log_file, maxBytes=5 * 1024 * 1024, backupCount=5
)
perf_handler.setLevel(logging.INFO)
perf_handler.setFormatter(logging.Formatter(LOG_FORMAT))

# Handler para consola
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))

# Añadir handlers al logger principal
logger.addHandler(main_handler)
logger.addHandler(error_handler)
logger.addHandler(console_handler)

# Loggers especializados
trade_logger = logging.getLogger("advanced_trading_algorithm.trades")
trade_logger.setLevel(logging.INFO)
trade_logger.propagate = False
trade_logger.addHandler(trade_handler)
trade_logger.addHandler(console_handler)

perf_logger = logging.getLogger("advanced_trading_algorithm.performance")
perf_logger.setLevel(logging.INFO)
perf_logger.propagate = False
perf_logger.addHandler(perf_handler)


# Filtro para el logger de operaciones
class TradeFilter(logging.Filter):
    def filter(self, record):
        return "TRADE" in record.getMessage()


trade_handler.addFilter(TradeFilter())

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

        logger.info(
            f"Algoritmo inicializado con {len(base_data) if base_data is not None else 0} registros históricos"
        )
        logger.debug(
            f"Configuración inicial: SL={self.stop_loss_pct*100}%, TP={self.take_profit_pct*100}%"
        )

    def prepare_features(self, data=None):
        """
        Prepara características para los modelos de machine learning

        Args:
            data: DataFrame con datos (usa self.data si no se proporciona)

        Returns:
            DataFrame con características preparadas
        """
        start_time = time.time()
        logger.debug("Iniciando preparación de características")

        df = data if data is not None else self.data

        if df is None:
            logger.error("No hay datos disponibles para procesar características")
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

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Columnas necesarias no encontradas: {missing_cols}")
            raise ValueError(
                f"Las columnas {missing_cols} no están presentes en los datos"
            )

        # Crear características adicionales
        features = pd.DataFrame(index=df.index)

        logger.debug("Calculando indicadores relativos al precio")
        # 1. Indicadores relativos al precio
        features["close_sma20_ratio"] = df["close"] / df["sma_20"]
        features["close_sma50_ratio"] = df["close"] / df["sma_50"]
        features["close_sma200_ratio"] = df["close"] / df["sma_200"]

        logger.debug("Calculando métricas de Bandas de Bollinger")
        # 2. Distancia a Bandas de Bollinger (normalizada)
        bb_width = df["bb_upper"] - df["bb_lower"]
        features["bb_position"] = (df["close"] - df["bb_lower"]) / bb_width

        # Continuar con el resto de características...
        logger.debug("Calculando indicadores de volatilidad")
        # 3. Volatilidad relativa
        features["atr_close_ratio"] = df["atr"] / df["close"]

        logger.debug("Calculando indicadores de momentum")
        # 4. Momentum y fuerza de tendencia
        features["macd_hist"] = df["macd"] - df["macd_signal"]
        features["rsi_momentum"] = df["rsi"].diff(3)

        # Más características (siguiendo código original)...

        # Llenar NaN
        nan_before = features.isna().sum().sum()
        features = features.fillna(0)

        time_elapsed = time.time() - start_time
        perf_logger.info(
            f"Preparación de características: {time_elapsed:.2f}s | Filas: {len(features)} | NaN rellenados: {nan_before}"
        )
        logger.debug(f"Características generadas: {list(features.columns)}")

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
            logger.error("No hay datos disponibles para preparar variable objetivo")
            raise ValueError("No hay datos disponibles")

        # Calcular retorno futuro
        logger.debug(f"Preparando variable objetivo con horizonte={horizon}")
        future_return = self.data["close"].shift(-horizon) / self.data["close"] - 1

        # Convertir a variable binaria (1 si sube, 0 si baja)
        target = (future_return > 0).astype(int)

        pos_ratio = target.mean()
        logger.debug(
            f"Distribución de clases: {pos_ratio:.2%} positivas, {1-pos_ratio:.2%} negativas"
        )

        return target

    def advanced_trading_algorithm(self, current_data, higher_tf_data=None):
        """
        Implementación del algoritmo avanzado de trading

        Args:
            current_data: DataFrame con datos del timeframe actual
            higher_tf_data: DataFrame con datos de timeframe superior

        Returns:
            Dict con señal, confianza y niveles de gestión de riesgo
        """
        start_time = time.time()
        logger.debug(f"Ejecutando algoritmo en {datetime.datetime.now()}")

        if current_data is None or len(current_data) < 200:
            logger.warning(
                f"Datos insuficientes: {len(current_data) if current_data is not None else 0} registros"
            )
            return {"signal": 0, "confidence": 0}

        # Obtener filas actuales y previas
        current_row = current_data.iloc[-1]
        prev_row = current_data.iloc[-2]

        logger.debug(
            f"Analizando precio actual: {current_row['close']:.2f}, previo: {prev_row['close']:.2f}"
        )
        logger.debug(
            f"Fecha/hora actual: {current_row.name if hasattr(current_row, 'name') else 'N/A'}"
        )

        try:
            # Preparar características para predicción
            features = self.prepare_features(current_data).iloc[-1]
        except Exception as e:
            logger.error(f"Error preparando características: {str(e)}", exc_info=True)
            return {"signal": 0, "confidence": 0, "error": str(e)}

        # 1. Analizar tendencia en timeframe superior
        higher_tf_trend = 0
        if higher_tf_data is not None:
            try:
                higher_tf_trend = self.analyze_higher_timeframe(higher_tf_data)
                logger.debug(f"Tendencia en timeframe superior: {higher_tf_trend:.2f}")
            except Exception as e:
                logger.error(
                    f"Error analizando timeframe superior: {str(e)}", exc_info=True
                )

        # 2. Obtener señales de estrategias base
        try:
            # Estrategia 1: Cruce de medias móviles
            signal_ma_cross = 0
            if (
                current_row["sma_20"] > current_row["sma_50"]
                and prev_row["sma_20"] <= prev_row["sma_50"]
            ):
                signal_ma_cross = 1
                logger.debug("Detectado cruce alcista SMA20 > SMA50")
            elif (
                current_row["sma_20"] < current_row["sma_50"]
                and prev_row["sma_20"] >= prev_row["sma_50"]
            ):
                signal_ma_cross = -1
                logger.debug("Detectado cruce bajista SMA20 < SMA50")

            # Estrategia 2: Breakout de Bollinger Bands
            signal_bb = 0
            if (
                current_row["close"] <= current_row["bb_lower"]
                and current_row["volume"]
                > current_data["volume"].rolling(20).mean().iloc[-1] * 1.5
                and current_row["rsi"] < 30
            ):
                signal_bb = 1
                logger.debug("Detectado breakout alcista en bandas de Bollinger")
            elif (
                current_row["close"] >= current_row["bb_upper"]
                and current_row["rsi"] > 70
            ):
                signal_bb = -1
                logger.debug("Detectado breakout bajista en bandas de Bollinger")

            # Estrategia 3: Divergencias RSI
            signal_rsi_div = 0
            # Código original para divergencia...
        except Exception as e:
            logger.error(f"Error procesando estrategias base: {str(e)}", exc_info=True)
            return {"signal": 0, "confidence": 0, "error": "Error en estrategias base"}

        # 3. Obtener predicciones de modelos ML
        ml_prob = 0.5  # Neutral por defecto
        lstm_prob = 0.5  # Neutral por defecto

        if self.ml_model is not None:
            try:
                ml_prob = self.predict_random_forest(features)
                logger.debug(f"Predicción Random Forest: {ml_prob:.4f}")
            except Exception as e:
                logger.error(
                    f"Error en predicción Random Forest: {str(e)}", exc_info=True
                )

        if self.lstm_model is not None:
            try:
                # Preparar secuencia para LSTM
                sequence = self.prepare_features(current_data).iloc[-10:].values
                lstm_prob = self.predict_lstm(sequence)
                logger.debug(f"Predicción LSTM: {lstm_prob:.4f}")
            except Exception as e:
                logger.error(f"Error en predicción LSTM: {str(e)}", exc_info=True)

        # 4. Obtener datos externos (sentimiento y on-chain)
        sentiment = 0
        on_chain = {}
        symbol = current_data.get("symbol", "BTC")
        if isinstance(symbol, pd.Series):
            symbol = symbol.iloc[0] if len(symbol) > 0 else "BTC"

        try:
            # Extraer el símbolo base
            base_symbol = symbol.split("/")[0] if "/" in symbol else symbol
            logger.debug(f"Obteniendo datos externos para {base_symbol}")

            sentiment = self.get_sentiment_score(base_symbol)
            on_chain = self.get_onchain_metrics(base_symbol)
            logger.debug(f"Sentimiento para {base_symbol}: {sentiment:.4f}")
            logger.debug(f"Métricas on-chain: {on_chain}")
        except Exception as e:
            logger.error(f"Error obteniendo datos externos: {str(e)}", exc_info=True)

        # 5. Calcular señal compuesta y confianza
        try:
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

            # Calcular score compuesto
            composite_score = (
                weights["ma_cross"] * signal_ma_cross
                + weights["bb"] * signal_bb
                + weights["rsi_div"] * signal_rsi_div
                + weights["ml"] * (ml_prob * 2 - 1)
                + weights["lstm"] * (lstm_prob * 2 - 1)
                + weights["higher_tf"] * higher_tf_trend
            )

            # Añadir sentimiento y otros factores
            composite_score += weights["sentiment"] * sentiment

            # Añadir factores on-chain según el código original

            logger.info(f"Score compuesto calculado: {composite_score:.4f}")

            # Determinar señal
            signal = 0
            confidence = 0.5

            if composite_score > 0.2:
                signal = 1  # Comprar
                confidence = 0.5 + min(0.5, composite_score)
                trade_logger.info(
                    f"TRADE | COMPRA | Score: {composite_score:.4f} | Confianza: {confidence:.4f} | Precio: {current_row['close']:.2f}"
                )
            elif composite_score < -0.2:
                signal = -1  # Vender
                confidence = 0.5 + min(0.5, abs(composite_score))
                trade_logger.info(
                    f"TRADE | VENTA | Score: {composite_score:.4f} | Confianza: {confidence:.4f} | Precio: {current_row['close']:.2f}"
                )
            else:
                logger.info("No se genera señal (score en zona neutral)")
        except Exception as e:
            logger.error(f"Error calculando señal compuesta: {str(e)}", exc_info=True)
            return {"signal": 0, "confidence": 0, "error": "Error en cálculo de señal"}

        # 6. Verificar filtro tendencia
        if (
            signal == 1
            and higher_tf_trend < -0.5
            and current_row["close"] < current_row["sma_200"]
        ):
            logger.warning("Filtro de tendencia activado: tendencia bajista fuerte")
            original_confidence = confidence
            confidence = max(0.5, confidence - 0.2)
            logger.debug(
                f"Confianza reducida: {original_confidence:.4f} -> {confidence:.4f}"
            )

            if confidence <= 0.55:
                trade_logger.info(
                    f"TRADE | CANCELADA | Señal de compra cancelada por filtro de tendencia"
                )
                signal = 0
                confidence = 0.5

        # 7. Calcular gestión de riesgo
        risk_levels = None
        position_size = 0

        if signal != 0:
            try:
                position_size = self.calculate_position_size(
                    confidence=confidence, volatility=current_row["atr"]
                )

                risk_levels = self.set_risk_management_levels(
                    entry_price=current_row["close"],
                    signal=signal,
                    atr=current_row["atr"],
                )

                trade_logger.info(
                    f"TRADE | GESTIÓN_RIESGO | Pos: {position_size:.6f} | "
                    f"SL: {risk_levels['stop_loss']:.2f} | "
                    f"TP: {risk_levels['take_profit']:.2f} | "
                    f"R:R: {(risk_levels['take_profit']-current_row['close'])/(current_row['close']-risk_levels['stop_loss']):.2f}"
                )
            except Exception as e:
                logger.error(f"Error en gestión de riesgo: {str(e)}", exc_info=True)

        # Resultado final
        components = {
            "ma_cross": signal_ma_cross,
            "bb": signal_bb,
            "rsi_div": signal_rsi_div,
            "ml_prob": ml_prob,
            "lstm_prob": lstm_prob,
            "higher_tf_trend": higher_tf_trend,
            "sentiment": sentiment,
        }

        result = {
            "signal": signal,
            "confidence": confidence,
            "position_size": position_size,
            "risk_levels": risk_levels,
            "composite_score": composite_score,
            "components": components,
        }

        execution_time = time.time() - start_time
        perf_logger.info(
            f"Tiempo ejecución algoritmo: {execution_time:.4f}s | Señal: {signal} | Confianza: {confidence:.4f}"
        )

        return result
