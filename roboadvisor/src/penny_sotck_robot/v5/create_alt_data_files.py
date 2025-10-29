#!/usr/bin/env python3
"""
Script para crear archivos CSV de datos alternativos
"""
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from logging_config_v5 import setup_logging, get_logger

setup_logging(level="INFO")
logger = get_logger('alt_data_creator')

# Crear directorio data si no existe
data_dir = "./data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    logger.info(f"Directorio creado: {data_dir}")

# Símbolos de la watchlist
symbols = [
    "OPEN", "CHPT", "LCFY", "SIRI", "XAIR",
    "HTOO", "CTMX", "CLOV", "ALBT", "ADIL",
    "BYND", "AKBA", "OPAD", "AIRE", "YYAI",
    "RANI", "WOK", "AREB", "BENF", "CJET", "SBEV", "ISRG", "VTYX",
    "RGC", "RVPH", "ONDS", "ADTX", "CLSK", "BITF", "IREN", "WGRX", "ADAG", "QLGN",
    "VIVK", "ASNS", "DFLI", "DVLT", "ASST", "PROP", "DGXX", "BKYI", "SLGB"
]

# Reddit sentiment data (ejemplo con datos variados)
reddit_data = {
    'symbol': symbols,
    'mentions': [45, 120, 30, 80, 200, 15, 25, 150, 90, 60,
                180, 70, 35, 50, 40, 55, 20, 10, 5, 15, 30, 0, 0,
                40, 25, 30, 45, 85, 75, 65, 20, 35, 50,
                280, 35, 40, 15, 350, 25, 180, 95, 120],
    'sentiment_score': [0.2, 0.5, 0.1, 0.3, 0.8, -0.1, 0.0, 0.6, 0.4, 0.3,
                       0.5, 0.4, 0.2, 0.3, 0.1, 0.3, -0.2, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0,
                       0.3, 0.1, 0.2, 0.3, 0.5, 0.4, 0.4, 0.1, 0.2, 0.3,
                       0.9, 0.2, 0.3, 0.1, 0.85, 0.2, 0.75, 0.6, 0.7],
    'sentiment': ['neutral', 'bullish', 'neutral', 'bullish', 'bullish', 'neutral', 'neutral',
                 'bullish', 'bullish', 'bullish', 'bullish', 'bullish', 'neutral', 'bullish',
                 'neutral', 'bullish', 'bearish', 'neutral', 'neutral', 'neutral', 'neutral',
                 'neutral', 'neutral', 'bullish', 'neutral', 'neutral', 'bullish', 'bullish',
                 'bullish', 'bullish', 'neutral', 'neutral', 'bullish', 'bullish', 'neutral',
                 'bullish', 'neutral', 'bullish', 'neutral', 'bullish', 'bullish', 'bullish'],
    'trending': [False, True, False, False, True, False, False, True, False, False,
                True, False, False, False, False, False, False, False, False, False, False,
                False, False, False, False, False, False, True, False, False, False, False,
                False, True, False, False, False, True, False, True, True, True]
}

reddit_df = pd.DataFrame(reddit_data)
reddit_path = os.path.join(data_dir, 'reddit_sentiment.csv')
reddit_df.to_csv(reddit_path, index=False)
logger.info(f"✅ Reddit sentiment guardado: {reddit_path}")
logger.info(f"   {len(reddit_df)} símbolos con datos de Reddit")

# Short borrow rates data
short_data = {
    'symbol': symbols,
    'borrow_rate_pct': [12.5, 25.0, 8.5, 15.0, 60.8, 5.0, 10.0, 18.5, 22.0, 14.0,
                       35.5, 20.0, 11.0, 13.5, 9.0, 25.0, 7.5, 6.0, 4.5, 8.0, 9.5, 3.0, 2.5,
                       16.0, 12.0, 11.5, 19.0, 42.0, 38.0, 35.0, 10.5, 14.5, 17.5,
                       75.0, 15.5, 16.5, 8.0, 80.0, 12.0, 65.5, 48.0, 52.5],
    'availability': ['moderate', 'hard_to_borrow', 'easy', 'moderate', 'hard_to_borrow',
                    'easy', 'moderate', 'moderate', 'hard_to_borrow', 'moderate',
                    'hard_to_borrow', 'hard_to_borrow', 'moderate', 'moderate', 'easy',
                    'hard_to_borrow', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy',
                    'moderate', 'moderate', 'moderate', 'moderate', 'hard_to_borrow',
                    'hard_to_borrow', 'hard_to_borrow', 'moderate', 'moderate', 'moderate',
                    'hard_to_borrow', 'moderate', 'moderate', 'easy', 'hard_to_borrow',
                    'moderate', 'hard_to_borrow', 'hard_to_borrow', 'hard_to_borrow']
}

short_df = pd.DataFrame(short_data)
short_path = os.path.join(data_dir, 'short_borrow_rates.csv')
short_df.to_csv(short_path, index=False)
logger.info(f"✅ Short borrow rates guardado: {short_path}")
logger.info(f"   {len(short_df)} símbolos con datos de short rates")

print("\n" + "="*70)
print("ARCHIVOS CSV DE DATOS ALTERNATIVOS CREADOS")
print("="*70)
print(f"\n📂 Ubicación: {os.path.abspath(data_dir)}/")
print(f"\n📄 Archivos:")
print(f"   • reddit_sentiment.csv ({len(reddit_df)} símbolos)")
print(f"   • short_borrow_rates.csv ({len(short_df)} símbolos)")

print(f"\n📊 Ejemplos de datos:")
print(f"\n   ASST (Momentum Puro detectado):")
asst_reddit = reddit_df[reddit_df['symbol'] == 'ASST'].iloc[0]
asst_short = short_df[short_df['symbol'] == 'ASST'].iloc[0]
print(f"      • Mentions: {asst_reddit['mentions']}")
print(f"      • Sentiment: {asst_reddit['sentiment']} ({asst_reddit['sentiment_score']:.2f})")
print(f"      • Trending: {asst_reddit['trending']}")
print(f"      • Borrow rate: {asst_short['borrow_rate_pct']:.1f}%")

print(f"\n   VIVK (Momentum Puro detectado):")
vivk_reddit = reddit_df[reddit_df['symbol'] == 'VIVK'].iloc[0]
vivk_short = short_df[short_df['symbol'] == 'VIVK'].iloc[0]
print(f"      • Mentions: {vivk_reddit['mentions']}")
print(f"      • Sentiment: {vivk_reddit['sentiment']} ({vivk_reddit['sentiment_score']:.2f})")
print(f"      • Trending: {vivk_reddit['trending']}")
print(f"      • Borrow rate: {vivk_short['borrow_rate_pct']:.1f}%")

print("\n" + "="*70)
