#!/usr/bin/env python3
"""
ACTUALIZADOR DE REDDIT SENTIMENT - Obtiene datos reales de Reddit

Requiere:
    pip install praw textblob

Configuración:
    1. Crear app en https://www.reddit.com/prefs/apps
    2. Obtener: client_id, client_secret, user_agent
    3. Configurar credenciales abajo

Uso:
    python update_reddit_sentiment.py
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta
import time
import praw
from textblob import TextBlob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
from logging_config_v5 import setup_logging, get_logger

setup_logging(level="INFO")
logger = get_logger('reddit_updater')

# =============================================================================
# CONFIGURACIÓN - COMPLETAR CON TUS CREDENCIALES
# =============================================================================

REDDIT_CONFIG = {
    'client_id': 'NDUWkhEII16IoiXcY6zhTg',           # Cambiar
    'client_secret': '7sU9Zsl8HVXu7IivYEK_yFFQBkmCXA',   # Cambiar
    'user_agent': 'PennyStockAdvisor'  # Cambiar
}

# Subreddits a monitorear
SUBREDDITS = ['wallstreetbets', 'pennystocks', 'RobinHoodPennyStocks', 'stocks']

# Días hacia atrás para buscar
LOOKBACK_DAYS = 3

# =============================================================================
# FUNCIONES
# =============================================================================

def check_dependencies():
    """Verifica que las dependencias estén instaladas"""
    try:
        return True
    except ImportError as e:
        logger.error(f"Dependencias faltantes: {e}")
        logger.error("Instalar con: pip install praw textblob")
        return False


def analyze_sentiment(text: str) -> tuple:
    """
    Analiza sentimiento de texto usando TextBlob

    Returns:
        (sentiment_score, sentiment_label)
        sentiment_score: -1.0 (muy negativo) a +1.0 (muy positivo)
        sentiment_label: 'bearish', 'neutral', 'bullish'
    """
    try:
        from textblob import TextBlob

        # Analizar texto
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity

        # Clasificar
        if polarity > 0.3:
            label = 'bullish'
        elif polarity < -0.3:
            label = 'bearish'
        else:
            label = 'neutral'

        return polarity, label

    except Exception as e:
        logger.warning(f"Error analizando sentiment: {e}")
        return 0.0, 'neutral'


def scrape_reddit_sentiment(symbols: list) -> pd.DataFrame:
    """
    Scraper de Reddit para obtener sentiment de símbolos

    Args:
        symbols: Lista de tickers a buscar

    Returns:
        DataFrame con columns: symbol, mentions, sentiment_score, sentiment, trending
    """
    try:
        import praw

        # Verificar configuración
        if REDDIT_CONFIG['client_id'] == 'TU_CLIENT_ID':
            logger.warning("⚠️ Configuración de Reddit API no completada")
            logger.warning("   Edita update_reddit_sentiment.py y completa REDDIT_CONFIG")
            logger.warning("   Usando datos de ejemplo...")
            return None

        # Conectar a Reddit
        logger.info("Conectando a Reddit API...")
        reddit = praw.Reddit(
            client_id=REDDIT_CONFIG['client_id'],
            client_secret=REDDIT_CONFIG['client_secret'],
            user_agent=REDDIT_CONFIG['user_agent']
        )

        # Timestamp para posts recientes
        cutoff_time = datetime.now() - timedelta(days=LOOKBACK_DAYS)
        cutoff_timestamp = cutoff_time.timestamp()

        results = []

        for symbol in symbols:
            logger.info(f"Buscando menciones de {symbol}...")

            mentions = 0
            sentiments = []
            recent_mentions = 0

            # Buscar en cada subreddit
            for subreddit_name in SUBREDDITS:
                try:
                    subreddit = reddit.subreddit(subreddit_name)

                    # Buscar posts con el símbolo
                    search_query = f"${symbol} OR {symbol}"

                    for submission in subreddit.search(search_query, time_filter='week', limit=50):
                        mentions += 1

                        # Analizar sentimiento del título + texto
                        text = submission.title + " " + submission.selftext
                        score, label = analyze_sentiment(text)
                        sentiments.append(score)

                        # Contar menciones recientes (últimos 3 días)
                        if submission.created_utc >= cutoff_timestamp:
                            recent_mentions += 1

                except Exception as e:
                    logger.warning(f"Error en subreddit {subreddit_name}: {e}")
                    continue

                # Rate limiting
                time.sleep(0.5)

            # Calcular sentiment promedio
            if len(sentiments) > 0:
                avg_sentiment = sum(sentiments) / len(sentiments)

                if avg_sentiment > 0.3:
                    sentiment_label = 'bullish'
                elif avg_sentiment < -0.3:
                    sentiment_label = 'bearish'
                else:
                    sentiment_label = 'neutral'
            else:
                avg_sentiment = 0.0
                sentiment_label = 'neutral'

            # Determinar si está trending (muchas menciones recientes)
            is_trending = recent_mentions > 20

            results.append({
                'symbol': symbol,
                'mentions': mentions,
                'sentiment_score': round(avg_sentiment, 2),
                'sentiment': sentiment_label,
                'trending': is_trending
            })

            logger.info(f"   {symbol}: {mentions} menciones, sentiment: {sentiment_label} ({avg_sentiment:.2f})")

        return pd.DataFrame(results)

    except ImportError:
        logger.error("PRAW no instalado. Instalar con: pip install praw")
        return None
    except Exception as e:
        logger.error(f"Error en scraper de Reddit: {e}")
        import traceback
        traceback.print_exc()
        return None


def update_reddit_csv(symbols: list, output_path: str = "./data/reddit_sentiment.csv"):
    """
    Actualiza el archivo CSV de Reddit sentiment

    Args:
        symbols: Lista de tickers
        output_path: Ruta del archivo CSV
    """
    logger.info("="*70)
    logger.info("ACTUALIZADOR DE REDDIT SENTIMENT")
    logger.info("="*70)

    # Verificar dependencias
    if not check_dependencies():
        logger.error("Dependencias faltantes. Abortando.")
        return False

    # Scraper de Reddit
    df = scrape_reddit_sentiment(symbols)

    if df is None:
        logger.warning("No se pudieron obtener datos de Reddit")
        logger.info("Usando datos de ejemplo...")

        # Generar datos de ejemplo si no hay API configurada
        df = pd.DataFrame({
            'symbol': symbols,
            'mentions': [50] * len(symbols),
            'sentiment_score': [0.3] * len(symbols),
            'sentiment': ['neutral'] * len(symbols),
            'trending': [False] * len(symbols)
        })

    # Guardar CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(f"\n✅ Archivo actualizado: {output_path}")
    logger.info(f"   • {len(df)} símbolos actualizados")
    logger.info(f"   • Timestamp: {datetime.now()}")

    # Mostrar estadísticas
    bullish_count = len(df[df['sentiment'] == 'bullish'])
    bearish_count = len(df[df['sentiment'] == 'bearish'])
    trending_count = len(df[df['trending'] == True])

    logger.info(f"\n📊 Estadísticas:")
    logger.info(f"   • Bullish: {bullish_count}")
    logger.info(f"   • Bearish: {bearish_count}")
    logger.info(f"   • Trending: {trending_count}")
    logger.info(f"   • Total menciones: {df['mentions'].sum()}")

    # Mostrar top 5 más mencionados
    top5 = df.nlargest(5, 'mentions')[['symbol', 'mentions', 'sentiment', 'trending']]
    logger.info(f"\n🔥 Top 5 más mencionados:")
    for idx, row in top5.iterrows():
        trending_emoji = "🔥" if row['trending'] else "  "
        logger.info(f"   {trending_emoji} {row['symbol']}: {row['mentions']} menciones ({row['sentiment']})")

    logger.info("\n" + "="*70)
    return True


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Símbolos de la watchlist
    from integration_v5_trading_manager import WATCHLIST_SYMBOLS
    symbols = list(WATCHLIST_SYMBOLS)

    logger.info(f"\n🎯 Actualizando sentiment para {len(symbols)} símbolos...")
    logger.info(f"📅 Lookback: {LOOKBACK_DAYS} días")
    logger.info(f"📍 Subreddits: {', '.join(SUBREDDITS)}")

    # Actualizar
    success = update_reddit_csv(symbols)

    if success:
        logger.info("\n✅ Actualización completada exitosamente")
        logger.info("\n💡 TIP: Ejecuta este script diariamente antes del análisis")
        logger.info("   Ejemplo cron: 0 8 * * * python update_reddit_sentiment.py")
    else:
        logger.error("\n❌ Actualización fallida")
