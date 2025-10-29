#!/usr/bin/env python3
"""
MARKET DATA CACHE V5
====================
Sistema de caché de datos de mercado con LRU cache y persistencia opcional

Features:
- LRU cache en memoria para consultas rápidas
- Persistencia opcional con pickle
- TTL configurable por tipo de dato
- Thread-safe
"""

import os
import pickle
import logging
from functools import lru_cache
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
import yfinance as yf
import numpy as np
import pandas as pd
from threading import Lock

logger = logging.getLogger(__name__)


class MarketDataCache:
    """
    Caché de datos de mercado con LRU cache y persistencia opcional
    """

    def __init__(self, cache_dir: str = "./cache", enable_persistence: bool = True):
        """
        Args:
            cache_dir: Directorio para archivos de caché persistente
            enable_persistence: Habilitar persistencia en disco
        """
        self.cache_dir = cache_dir
        self.enable_persistence = enable_persistence
        self.memory_cache = {}
        self.cache_metadata = {}
        self.lock = Lock()

        if enable_persistence and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            logger.info(f"Directorio de caché creado: {cache_dir}")

    def get_cached_data(self, symbol: str, data_type: str = "historical",
                       period: str = "2mo", ttl_minutes: int = 60) -> Optional[Any]:
        """
        Obtiene datos del caché (memoria o disco)

        Args:
            symbol: Símbolo del ticker
            data_type: Tipo de dato ('historical', 'info', 'indicators')
            period: Periodo de datos históricos
            ttl_minutes: Time-to-live en minutos

        Returns:
            Datos cacheados o None si no existen o expiraron
        """
        cache_key = f"{symbol}_{data_type}_{period}"

        with self.lock:
            # 1. Buscar en memoria
            if cache_key in self.memory_cache:
                metadata = self.cache_metadata.get(cache_key, {})
                cached_time = metadata.get('timestamp')

                if cached_time and self._is_cache_valid(cached_time, ttl_minutes):
                    logger.debug(f"Cache HIT (memoria): {cache_key}")
                    return self.memory_cache[cache_key]
                else:
                    # Cache expirado
                    del self.memory_cache[cache_key]
                    logger.debug(f"Cache EXPIRED (memoria): {cache_key}")

            # 2. Buscar en disco (si está habilitado)
            if self.enable_persistence:
                cached_data = self._load_from_disk(cache_key, ttl_minutes)
                if cached_data is not None:
                    # Cargar a memoria
                    self.memory_cache[cache_key] = cached_data
                    self.cache_metadata[cache_key] = {'timestamp': datetime.now()}
                    logger.debug(f"Cache HIT (disco): {cache_key}")
                    return cached_data

        logger.debug(f"Cache MISS: {cache_key}")
        return None

    def set_cached_data(self, symbol: str, data: Any, data_type: str = "historical",
                       period: str = "2mo"):
        """
        Guarda datos en el caché (memoria y disco)

        Args:
            symbol: Símbolo del ticker
            data: Datos a cachear
            data_type: Tipo de dato
            period: Periodo de datos
        """
        cache_key = f"{symbol}_{data_type}_{period}"

        with self.lock:
            # Guardar en memoria
            self.memory_cache[cache_key] = data
            self.cache_metadata[cache_key] = {'timestamp': datetime.now()}

            # Guardar en disco (si está habilitado)
            if self.enable_persistence:
                self._save_to_disk(cache_key, data)

            logger.debug(f"Cache SET: {cache_key}")

    def clear_cache(self, symbol: Optional[str] = None):
        """
        Limpia el caché (memoria y opcionalmente disco)

        Args:
            symbol: Si se especifica, solo limpia ese símbolo. Si es None, limpia todo
        """
        with self.lock:
            if symbol:
                # Limpiar solo ese símbolo
                keys_to_remove = [k for k in self.memory_cache.keys() if k.startswith(symbol)]
                for key in keys_to_remove:
                    del self.memory_cache[key]
                    if key in self.cache_metadata:
                        del self.cache_metadata[key]

                    if self.enable_persistence:
                        filepath = self._get_cache_filepath(key)
                        if os.path.exists(filepath):
                            os.remove(filepath)

                logger.info(f"Cache limpiado para {symbol}")
            else:
                # Limpiar todo
                self.memory_cache.clear()
                self.cache_metadata.clear()

                if self.enable_persistence and os.path.exists(self.cache_dir):
                    for filename in os.listdir(self.cache_dir):
                        filepath = os.path.join(self.cache_dir, filename)
                        if os.path.isfile(filepath):
                            os.remove(filepath)

                logger.info("Cache completamente limpiado")

    def _is_cache_valid(self, cached_time: datetime, ttl_minutes: int) -> bool:
        """Verifica si el caché sigue siendo válido"""
        elapsed = datetime.now() - cached_time
        return elapsed.total_seconds() < (ttl_minutes * 60)

    def _get_cache_filepath(self, cache_key: str) -> str:
        """Obtiene la ruta del archivo de caché"""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")

    def _save_to_disk(self, cache_key: str, data: Any):
        """Guarda datos en disco con pickle"""
        try:
            filepath = self._get_cache_filepath(cache_key)
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'data': data,
                    'timestamp': datetime.now()
                }, f)
        except Exception as e:
            logger.warning(f"Error guardando caché en disco {cache_key}: {e}")

    def _load_from_disk(self, cache_key: str, ttl_minutes: int) -> Optional[Any]:
        """Carga datos desde disco con pickle"""
        try:
            filepath = self._get_cache_filepath(cache_key)
            if not os.path.exists(filepath):
                return None

            with open(filepath, 'rb') as f:
                cached = pickle.load(f)

            cached_time = cached.get('timestamp')
            if cached_time and self._is_cache_valid(cached_time, ttl_minutes):
                return cached.get('data')
            else:
                # Eliminar archivo expirado
                os.remove(filepath)
                return None
        except Exception as e:
            logger.warning(f"Error cargando caché desde disco {cache_key}: {e}")
            return None

    def get_cache_stats(self) -> Dict:
        """Obtiene estadísticas del caché"""
        with self.lock:
            memory_size = len(self.memory_cache)

            disk_size = 0
            if self.enable_persistence and os.path.exists(self.cache_dir):
                disk_size = len([f for f in os.listdir(self.cache_dir)
                               if os.path.isfile(os.path.join(self.cache_dir, f))])

            return {
                'memory_entries': memory_size,
                'disk_entries': disk_size,
                'cache_dir': self.cache_dir,
                'persistence_enabled': self.enable_persistence
            }


# Funciones helper con LRU cache para cálculos repetitivos
@lru_cache(maxsize=128)
def calculate_indicators_cached(prices_tuple: tuple, volumes_tuple: tuple) -> Dict:
    """
    Calcula indicadores técnicos con LRU cache

    Args:
        prices_tuple: Tupla de precios (para hacerlo hashable)
        volumes_tuple: Tupla de volúmenes

    Returns:
        Dict con indicadores calculados
    """
    prices = np.array(prices_tuple)
    volumes = np.array(volumes_tuple)

    # RSI
    delta = np.diff(prices)
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)

    avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
    avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0

    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs)) if rs != 0 else 50

    # MACD
    ema_12 = _ema(prices, 12)
    ema_26 = _ema(prices, 26)
    macd = ema_12 - ema_26
    signal = _ema(np.array([macd]), 9)
    macd_diff = macd - signal

    # ADX (simplificado)
    adx = 20.0  # Placeholder

    return {
        'rsi': float(rsi),
        'macd': float(macd),
        'macd_signal': float(signal),
        'macd_diff': float(macd_diff),
        'adx': float(adx)
    }


def _ema(data: np.ndarray, period: int) -> float:
    """Calcula EMA"""
    if len(data) < period:
        return float(np.mean(data))

    multiplier = 2 / (period + 1)
    ema = np.mean(data[:period])

    for price in data[period:]:
        ema = (price * multiplier) + (ema * (1 - multiplier))

    return float(ema)


if __name__ == "__main__":
    # Test básico
    logging.basicConfig(level=logging.DEBUG)

    cache = MarketDataCache(enable_persistence=True)

    # Simular guardado
    test_data = {'price': 10.5, 'volume': 1000000}
    cache.set_cached_data('TEST', test_data, 'info')

    # Recuperar
    cached = cache.get_cached_data('TEST', 'info', ttl_minutes=60)
    print(f"Cached data: {cached}")

    # Stats
    stats = cache.get_cache_stats()
    print(f"Cache stats: {stats}")
