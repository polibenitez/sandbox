#!/usr/bin/env python3
"""
ALTERNATIVE DATA V5 - DATOS ALTERNATIVOS
=========================================

Features:
- Reddit sentiment analysis
- Short borrow rate tracking
- Integración con APIs opcionales
- Fallback a datos locales CSV
"""

import logging
import pandas as pd
import os
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger('alternative_data')


class AlternativeDataProvider:
    """
    Proveedor de datos alternativos (Reddit sentiment + Short borrow rate)
    """

    def __init__(self, use_api: bool = False, local_data_path: str = "./data"):
        """
        Args:
            use_api: Si True, intenta usar APIs. Si False, usa datos locales
            local_data_path: Path para archivos CSV locales
        """
        self.use_api = use_api
        self.local_data_path = local_data_path

        # Cache de datos
        self.reddit_cache = {}
        self.short_rate_cache = {}

        logger.info(f"AlternativeDataProvider inicializado - API: {use_api}")

    def get_reddit_sentiment(self, symbol: str) -> Dict:
        """
        Obtiene sentimiento de Reddit para un símbolo

        Args:
            symbol: Ticker del símbolo

        Returns:
            Dict con sentimiento:
            {
                'mentions': int,
                'sentiment_score': float (-1 a +1),
                'sentiment': 'bullish' | 'neutral' | 'bearish',
                'trending': bool,
                'source': 'api' | 'local' | 'default'
            }
        """
        # Buscar en cache
        if symbol in self.reddit_cache:
            return self.reddit_cache[symbol]

        # Intentar obtener datos
        if self.use_api:
            data = self._fetch_reddit_from_api(symbol)
        else:
            data = self._fetch_reddit_from_local(symbol)

        # Si no hay datos, usar defaults
        if data is None:
            data = self._get_default_reddit_sentiment(symbol)

        # Cachear
        self.reddit_cache[symbol] = data

        return data

    def get_short_borrow_rate(self, symbol: str) -> Dict:
        """
        Obtiene tasa de préstamo de shorts para un símbolo

        Args:
            symbol: Ticker del símbolo

        Returns:
            Dict con datos de short:
            {
                'borrow_rate_pct': float,
                'availability': 'easy' | 'moderate' | 'hard_to_borrow',
                'short_squeeze_risk': 'low' | 'medium' | 'high',
                'source': 'api' | 'local' | 'default'
            }
        """
        # Buscar en cache
        if symbol in self.short_rate_cache:
            return self.short_rate_cache[symbol]

        # Intentar obtener datos
        if self.use_api:
            data = self._fetch_short_rate_from_api(symbol)
        else:
            data = self._fetch_short_rate_from_local(symbol)

        # Si no hay datos, usar defaults
        if data is None:
            data = self._get_default_short_rate(symbol)

        # Cachear
        self.short_rate_cache[symbol] = data

        return data

    def get_combined_alternative_data(self, symbol: str) -> Dict:
        """
        Obtiene datos alternativos combinados

        Returns:
            Dict con Reddit + Short rate combinados
        """
        reddit = self.get_reddit_sentiment(symbol)
        short_rate = self.get_short_borrow_rate(symbol)

        # Score combinado (0-100)
        combined_score = self._calculate_combined_score(reddit, short_rate)

        return {
            'symbol': symbol,
            'reddit': reddit,
            'short_borrow': short_rate,
            'combined_score': combined_score,
            'timestamp': datetime.now()
        }

    def _fetch_reddit_from_api(self, symbol: str) -> Optional[Dict]:
        """
        Obtiene sentimiento de Reddit desde API
        (Placeholder - implementar con API real como PRAW o similar)
        """
        logger.debug(f"Intentando obtener Reddit sentiment para {symbol} desde API")

        # TODO: Implementar integración con API de Reddit
        # Ejemplo con PRAW:
        # import praw
        # reddit = praw.Reddit(...)
        # submissions = reddit.subreddit('wallstreetbets').search(symbol, limit=100)
        # ...

        return None  # No implementado aún

    def _fetch_reddit_from_local(self, symbol: str) -> Optional[Dict]:
        """Obtiene sentimiento de Reddit desde archivo CSV local"""
        try:
            filepath = os.path.join(self.local_data_path, "reddit_sentiment.csv")

            if not os.path.exists(filepath):
                logger.debug(f"No existe archivo local de Reddit sentiment: {filepath}")
                return None

            df = pd.read_csv(filepath)

            # Buscar símbolo
            row = df[df['symbol'] == symbol]

            if len(row) == 0:
                return None

            row = row.iloc[0]

            return {
                'mentions': int(row.get('mentions', 0)),
                'sentiment_score': float(row.get('sentiment_score', 0)),
                'sentiment': str(row.get('sentiment', 'neutral')),
                'trending': bool(row.get('trending', False)),
                'source': 'local'
            }

        except Exception as e:
            logger.warning(f"Error leyendo Reddit sentiment local para {symbol}: {e}")
            return None

    def _fetch_short_rate_from_api(self, symbol: str) -> Optional[Dict]:
        """
        Obtiene short borrow rate desde API
        (Placeholder - implementar con API real)
        """
        logger.debug(f"Intentando obtener short rate para {symbol} desde API")

        # TODO: Implementar integración con API de borrow rates
        # Opciones: Fintel, Ortex, etc.

        return None  # No implementado aún

    def _fetch_short_rate_from_local(self, symbol: str) -> Optional[Dict]:
        """Obtiene short borrow rate desde archivo CSV local"""
        try:
            filepath = os.path.join(self.local_data_path, "short_borrow_rates.csv")

            if not os.path.exists(filepath):
                logger.debug(f"No existe archivo local de short rates: {filepath}")
                return None

            df = pd.read_csv(filepath)

            # Buscar símbolo
            row = df[df['symbol'] == symbol]

            if len(row) == 0:
                return None

            row = row.iloc[0]

            borrow_rate = float(row.get('borrow_rate_pct', 0))

            return {
                'borrow_rate_pct': borrow_rate,
                'availability': str(row.get('availability', 'moderate')),
                'short_squeeze_risk': self._assess_squeeze_risk(borrow_rate),
                'source': 'local'
            }

        except Exception as e:
            logger.warning(f"Error leyendo short rate local para {symbol}: {e}")
            return None

    def _get_default_reddit_sentiment(self, symbol: str) -> Dict:
        """Datos default de Reddit si no hay información"""
        return {
            'mentions': 0,
            'sentiment_score': 0.0,
            'sentiment': 'neutral',
            'trending': False,
            'source': 'default'
        }

    def _get_default_short_rate(self, symbol: str) -> Dict:
        """Datos default de short rate si no hay información"""
        return {
            'borrow_rate_pct': 10.0,  # Default moderado
            'availability': 'moderate',
            'short_squeeze_risk': 'medium',
            'source': 'default'
        }

    def _assess_squeeze_risk(self, borrow_rate: float) -> str:
        """Evalúa riesgo de short squeeze basado en borrow rate"""
        if borrow_rate >= 50:
            return 'high'
        elif borrow_rate >= 20:
            return 'medium'
        else:
            return 'low'

    def _calculate_combined_score(self, reddit: Dict, short_rate: Dict) -> float:
        """
        Calcula score combinado de datos alternativos (0-100)

        Lógica:
        - Reddit bullish + high borrow rate = Score alto
        - Trending en Reddit = Bonus
        """
        score = 50.0  # Base

        # Reddit sentiment
        sentiment_score = reddit['sentiment_score']
        score += sentiment_score * 15  # -15 a +15

        # Reddit trending
        if reddit['trending']:
            score += 10

        # Mentions (más menciones = más interés)
        if reddit['mentions'] > 100:
            score += 10
        elif reddit['mentions'] > 50:
            score += 5

        # Short borrow rate (más alto = más potencial de squeeze)
        borrow_rate = short_rate['borrow_rate_pct']
        if borrow_rate >= 50:
            score += 15
        elif borrow_rate >= 20:
            score += 10
        elif borrow_rate >= 10:
            score += 5

        # Normalizar a 0-100
        score = max(0, min(100, score))

        return score

    def clear_cache(self):
        """Limpia el cache de datos alternativos"""
        self.reddit_cache.clear()
        self.short_rate_cache.clear()
        logger.info("Cache de datos alternativos limpiado")


def create_sample_local_files():
    """Crea archivos CSV de ejemplo para testing"""
    data_dir = "./data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Reddit sentiment
    reddit_data = {
        'symbol': ['BYND', 'COSM', 'XAIR', 'RANI', 'CLOV'],
        'mentions': [150, 45, 200, 80, 120],
        'sentiment_score': [0.6, -0.2, 0.8, 0.4, 0.3],
        'sentiment': ['bullish', 'bearish', 'bullish', 'bullish', 'neutral'],
        'trending': [True, False, True, False, False]
    }

    reddit_df = pd.DataFrame(reddit_data)
    reddit_df.to_csv(os.path.join(data_dir, 'reddit_sentiment.csv'), index=False)

    # Short borrow rates
    short_data = {
        'symbol': ['BYND', 'COSM', 'XAIR', 'RANI', 'CLOV'],
        'borrow_rate_pct': [35.5, 15.2, 60.8, 25.0, 18.5],
        'availability': ['hard_to_borrow', 'moderate', 'hard_to_borrow', 'hard_to_borrow', 'moderate']
    }

    short_df = pd.DataFrame(short_data)
    short_df.to_csv(os.path.join(data_dir, 'short_borrow_rates.csv'), index=False)

    logger.info(f"Archivos de ejemplo creados en {data_dir}")


if __name__ == "__main__":
    from logging_config_v5 import setup_logging
    setup_logging(level="INFO")

    # Crear archivos de ejemplo
    create_sample_local_files()

    # Test del provider
    provider = AlternativeDataProvider(use_api=False, local_data_path="./data")

    # Test Reddit sentiment
    reddit = provider.get_reddit_sentiment('BYND')
    print("\n" + "="*70)
    print("REDDIT SENTIMENT - BYND")
    print("="*70)
    print(f"Mentions: {reddit['mentions']}")
    print(f"Sentiment: {reddit['sentiment']} ({reddit['sentiment_score']:.2f})")
    print(f"Trending: {reddit['trending']}")
    print(f"Source: {reddit['source']}")

    # Test Short borrow rate
    short = provider.get_short_borrow_rate('BYND')
    print("\n" + "="*70)
    print("SHORT BORROW RATE - BYND")
    print("="*70)
    print(f"Borrow Rate: {short['borrow_rate_pct']:.1f}%")
    print(f"Availability: {short['availability']}")
    print(f"Squeeze Risk: {short['short_squeeze_risk']}")
    print(f"Source: {short['source']}")

    # Test Combined
    combined = provider.get_combined_alternative_data('BYND')
    print("\n" + "="*70)
    print("COMBINED ALTERNATIVE DATA - BYND")
    print("="*70)
    print(f"Combined Score: {combined['combined_score']:.1f}/100")
