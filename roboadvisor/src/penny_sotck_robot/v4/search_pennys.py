#!/usr/bin/env python3
"""
PENNY STOCKS SCREENER - VERSIÓN MEJORADA
=========================================

MEJORAS IMPLEMENTADAS:
1. ✅ Análisis de sentimiento de Reddit usando snscrape
2. ✅ Detección de nivel de hype (MUY ALTO, ALTO, MEDIO, BAJO, NINGUNO)
3. ✅ Scoring mejorado: Técnica (50%) + Twitter (20%) + Reddit (30%)
4. ✅ Guardado con timestamp: pennys_YYYY-MM-DD_HH-MM-SS.csv
5. ✅ Análisis de sentimiento positivo/negativo/neutral
6. ✅ Reporte detallado en consola y Telegram

Uso:
    python search_pennys.py

Output:
    - Archivo CSV: pennys_2025-10-23_14-30-45.csv
    - Reporte por consola
    - Mensaje a Telegram (si está configurado)

Autor: Actualizado el 23 de Octubre, 2025
"""

import yfinance as yf
import pandas as pd
import numpy as np
import subprocess
import requests
from datetime import datetime
import re
import json

# ----------------------------------------------------------
# CONFIGURACIÓN
# ----------------------------------------------------------
PRICE_LIMIT = 5
CHANGE_PCT_MIN = 5
VOLUME_RATIO_MIN = 2
TOP_N = 10

# Configuración de Telegram (opcional)
TELEGRAM_TOKEN = "7050816856:AAGoXtA5P7kVuGalV52zAvS71CLYrBTAi6k"
TELEGRAM_CHAT_ID = "Polibeni_bot"

# ----------------------------------------------------------
# FUNCIONES BASE
# ----------------------------------------------------------

def get_all_tickers2(limit=300):
    """Obtiene una lista amplia de tickers del mercado de EE. UU."""
    url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/refs/heads/main/all/all_tickers.txt"
    
    df = pd.read_csv(url)
    return df["Symbol"].dropna().unique().tolist()[:limit]


def get_all_tickers(limit=30000):
    """Obtiene una lista amplia de tickers del mercado de EE. UU."""
    url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/refs/heads/main/all/all_tickers.txt"
    response = requests.get(url)
    tickers = response.text.splitlines()
    print(f"✅ Obtenidos {len(tickers)} tickers del mercado de EE. UU.")
    return tickers[:limit]
    

def analyze_ticker(symbol):
    """Analiza métricas técnicas básicas de un ticker."""
    try:
        data = yf.Ticker(symbol).history(period="10d", interval="1d")
        if data.empty or len(data) < 5:
            return None

        last_close = data["Close"].iloc[-1]
        prev_close = data["Close"].iloc[-2]
        change_pct = (last_close - prev_close) / prev_close * 100
        avg_vol_5d = data["Volume"].tail(5).mean()
        avg_vol_20d = data["Volume"].mean()
        vol_ratio = avg_vol_5d / avg_vol_20d if avg_vol_20d > 0 else 0

        if (
            last_close < PRICE_LIMIT and
            change_pct > CHANGE_PCT_MIN and
            vol_ratio > VOLUME_RATIO_MIN
        ):
            return {
                "symbol": symbol,
                "price": round(last_close, 3),
                "change_pct": round(change_pct, 2),
                "volume_ratio": round(vol_ratio, 2),
                "avg_volume": int(avg_vol_5d)
            }
    except Exception:
        pass


def analyze_sentiment_simple(text):
    """
    Análisis de sentimiento simple basado en palabras clave.

    Returns:
        int: 1 (positivo), 0 (neutral), -1 (negativo)
    """
    text_lower = text.lower()

    positive_words = [
        'moon', 'bullish', 'buy', 'rocket', 'squeeze', 'breakout',
        'pump', 'gain', 'profit', 'long', 'call', 'up', 'rally',
        'bounce', 'support', 'strong', 'winning', 'great', 'amazing'
    ]

    negative_words = [
        'bear', 'bearish', 'sell', 'dump', 'crash', 'tank', 'drop',
        'fall', 'short', 'put', 'down', 'bag', 'holder', 'loss',
        'scam', 'fraud', 'avoid', 'warning', 'risky'
    ]

    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)

    if positive_count > negative_count:
        return 1
    elif negative_count > positive_count:
        return -1
    else:
        return 0


def get_twitter_mentions(symbol, since_days=2):
    """Cuenta menciones recientes en Twitter usando snscrape."""
    try:
        cmd = [
            "snscrape", "--max-results", "100",
            "twitter-search", f"${symbol} since:{(datetime.now() - pd.Timedelta(days=since_days)).date()}"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        lines = result.stdout.splitlines()
        return len(lines)
    except Exception:
        return 0


def get_reddit_sentiment(symbol, max_results=50):
    """
    Analiza menciones y sentimiento en Reddit usando snscrape.

    Args:
        symbol: Ticker del símbolo (ej: "CLOV")
        max_results: Número máximo de posts a analizar

    Returns:
        dict: {
            'mentions': int,
            'sentiment_score': float (-1 a 1),
            'positive': int,
            'neutral': int,
            'negative': int,
            'hype_level': str
        }
    """
    try:
        # Buscar en subreddits relevantes
        # Nota: snscrape reddit-search busca en todo Reddit
        cmd = [
            "snscrape",
            "--jsonl",
            "--max-results", str(max_results),
            "reddit-search",
            f"{symbol}"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            return {
                'mentions': 0,
                'sentiment_score': 0,
                'positive': 0,
                'neutral': 0,
                'negative': 0,
                'hype_level': 'NINGUNO'
            }

        lines = result.stdout.strip().split('\n')

        mentions = 0
        sentiments = []
        positive = 0
        neutral = 0
        negative = 0

        for line in lines:
            if not line.strip():
                continue

            try:
                post = json.loads(line)

                # Extraer texto del post
                text = ""
                if 'title' in post:
                    text += post['title'] + " "
                if 'content' in post:
                    text += post.get('content', '')

                if text.strip():
                    mentions += 1

                    # Analizar sentimiento
                    sentiment = analyze_sentiment_simple(text)
                    sentiments.append(sentiment)

                    if sentiment == 1:
                        positive += 1
                    elif sentiment == -1:
                        negative += 1
                    else:
                        neutral += 1

            except json.JSONDecodeError:
                continue

        # Calcular score de sentimiento promedio
        sentiment_score = np.mean(sentiments) if sentiments else 0

        # Determinar nivel de hype
        if mentions >= 30 and sentiment_score > 0.3:
            hype_level = 'MUY ALTO'
        elif mentions >= 20 and sentiment_score > 0.2:
            hype_level = 'ALTO'
        elif mentions >= 10:
            hype_level = 'MEDIO'
        elif mentions > 0:
            hype_level = 'BAJO'
        else:
            hype_level = 'NINGUNO'

        return {
            'mentions': mentions,
            'sentiment_score': round(sentiment_score, 2),
            'positive': positive,
            'neutral': neutral,
            'negative': negative,
            'hype_level': hype_level
        }

    except subprocess.TimeoutExpired:
        print(f"⚠️  Timeout al buscar {symbol} en Reddit")
        return {
            'mentions': 0,
            'sentiment_score': 0,
            'positive': 0,
            'neutral': 0,
            'negative': 0,
            'hype_level': 'NINGUNO'
        }
    except Exception as e:
        print(f"⚠️  Error al buscar {symbol} en Reddit: {e}")
        return {
            'mentions': 0,
            'sentiment_score': 0,
            'positive': 0,
            'neutral': 0,
            'negative': 0,
            'hype_level': 'NINGUNO'
        }


def send_telegram_message(message):
    """Envía mensaje a Telegram."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    requests.post(url, data=data)


# ----------------------------------------------------------
# SCREENER PRINCIPAL
# ----------------------------------------------------------

def run_screener():
    """
    Screener principal con análisis de Reddit + Twitter
    """
    tickers = get_all_tickers()
    print(f"🔍 Analizando {len(tickers)} tickers...")
    print(f"📊 Incluye: Análisis técnico + Twitter + Reddit\n")

    results = []
    for i, symbol in enumerate(tickers, 1):
        data = analyze_ticker(symbol)
        if data:
            # Análisis de Twitter
            twitter_mentions = get_twitter_mentions(symbol)
            data["twitter_mentions"] = twitter_mentions

            # Análisis de Reddit (NUEVO)
            print(f"  🔎 {symbol}: Analizando Reddit...", end="\r")
            reddit_data = get_reddit_sentiment(symbol)
            data["reddit_mentions"] = reddit_data['mentions']
            data["reddit_sentiment"] = reddit_data['sentiment_score']
            data["reddit_positive"] = reddit_data['positive']
            data["reddit_neutral"] = reddit_data['neutral']
            data["reddit_negative"] = reddit_data['negative']
            data["reddit_hype"] = reddit_data['hype_level']

            # Score ponderado mejorado:
            # - Técnica (50%): cambio de precio
            # - Twitter (20%): menciones
            # - Reddit (30%): menciones + sentimiento
            technical_score = data["change_pct"] * 0.5
            twitter_score = np.log1p(twitter_mentions) * 5 * 0.2
            reddit_score = (
                np.log1p(reddit_data['mentions']) * 8 +
                reddit_data['sentiment_score'] * 10
            ) * 0.3

            data["score"] = technical_score + twitter_score + reddit_score

            # Score individual de hype (para referencia)
            total_mentions = twitter_mentions + reddit_data['mentions']
            data["total_social_mentions"] = total_mentions
            data["hype_score"] = twitter_score + reddit_score

            results.append(data)

            print(f"  ✓ {symbol}: Precio {data['price']}, Reddit {reddit_data['mentions']} menciones ({reddit_data['hype_level']})" + " " * 20)

        if i % 50 == 0:
            print(f"\n📈 Progreso: {i}/{len(tickers)} tickers analizados...")

    if not results:
        print("\n⚠️ No se encontraron penny stocks prometedoras hoy.")
        return

    df = pd.DataFrame(results)
    df = df.sort_values(by="score", ascending=False).head(TOP_N)

    # Guardar resultados con fecha y hora (NUEVO FORMATO)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"pennys_{timestamp}.csv"
    df.to_csv(file_name, index=False)

    print("\n" + "="*80)
    print("🔥 PENNY STOCKS CON MAYOR POTENCIAL")
    print("="*80)
    print(df[['symbol', 'price', 'change_pct', 'volume_ratio',
              'twitter_mentions', 'reddit_mentions', 'reddit_sentiment',
              'reddit_hype', 'score']].to_string(index=False))
    print("="*80)

    # Reporte detallado
    print("\n📋 REPORTE DETALLADO:")
    for idx, row in df.iterrows():
        print(f"\n{idx+1}. 💰 ${row['symbol']} - ${row['price']}")
        print(f"   📈 Cambio: {row['change_pct']:+.2f}%")
        print(f"   📊 Volumen ratio: {row['volume_ratio']}x")
        print(f"   🐦 Twitter: {row['twitter_mentions']} menciones")
        print(f"   🤖 Reddit: {row['reddit_mentions']} menciones | Sentimiento: {row['reddit_sentiment']:+.2f} | Hype: {row['reddit_hype']}")
        print(f"   💯 Score total: {row['score']:.2f}")

    # Enviar por Telegram
    message = "🔥 *Penny Stocks PRO - Top Picks*\n"
    message += f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    for i, (_, row) in enumerate(df.iterrows(), 1):
        message += f"{i}. 💰 `${row['symbol']}` (${row['price']})\n"
        message += f"   📈 {row['change_pct']:+.2f}% | Vol: {row['volume_ratio']}x\n"
        message += f"   🐦 Twitter: {row['twitter_mentions']} | 🤖 Reddit: {row['reddit_mentions']}\n"
        message += f"   🧠 Sentimiento: {row['reddit_sentiment']:+.2f} | Hype: {row['reddit_hype']}\n\n"

    message += f"📁 Guardado en: `{file_name}`"
    send_telegram_message(message)

    print(f"\n💾 Resultados guardados en: {file_name}")
    print(f"📱 Reporte enviado por Telegram\n")

# ----------------------------------------------------------
# EJECUCIÓN
# ----------------------------------------------------------

if __name__ == "__main__":
    run_screener()
