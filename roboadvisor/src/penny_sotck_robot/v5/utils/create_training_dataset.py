#!/usr/bin/env python3
"""
CREATE_TRAINING_DATASET.PY — Penny Stock Advisor V5
====================================================

Crea un dataset real de entrenamiento con features técnicos y etiquetas 'exploded'
a partir de datos históricos de penny stocks.

Features calculados:
- bb_width: Ancho de Bandas de Bollinger
- adx: Average Directional Index (real)
- vol_ratio: Ratio volumen actual / promedio
- rsi: Relative Strength Index
- macd_diff: Diferencia MACD - Signal
- atr_ratio: ATR / Precio
- short_float: % del float en corto (de yfinance)
- compression_days: Días consecutivos en compresión
- volume_dry: 1 si volumen bajo promedio, 0 si no
- price_range_pct: % rango de precio en ventana

Target:
- exploded: 1 si precio subió 15%+ en próximos 5 días, 0 si no

Uso:
    python create_training_dataset.py --months 24 --min_samples 50
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "../data")
OUTPUT_PATH = os.path.join(DATA_DIR, "penny_stock_training.csv")

os.makedirs(DATA_DIR, exist_ok=True)

# =========================================================
# INDICADORES TÉCNICOS
# =========================================================

def calculate_rsi(prices, period=14):
    """Calcula RSI correctamente"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    # Evitar división por cero
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(50)  # RSI neutral si no hay datos


def calculate_adx(df, period=14):
    """
    Calcula ADX (Average Directional Index) real

    ADX mide la fuerza de la tendencia (no la dirección)
    ADX < 20: Sin tendencia (lo que queremos para compresión)
    ADX > 25: Tendencia fuerte
    """
    high = df['High']
    low = df['Low']
    close = df['Close']

    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    # Directional Movement
    up_move = high - high.shift()
    down_move = low.shift() - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    plus_dm_smooth = pd.Series(plus_dm, index=df.index).rolling(window=period).mean()
    minus_dm_smooth = pd.Series(minus_dm, index=df.index).rolling(window=period).mean()

    # Directional Indicators
    plus_di = 100 * (plus_dm_smooth / atr)
    minus_di = 100 * (minus_dm_smooth / atr)

    # ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()

    return adx.fillna(20)  # Default neutral


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calcula MACD"""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_diff = macd - macd_signal

    return macd, macd_signal, macd_diff


def calculate_bollinger_bands(prices, period=20):
    """Calcula Bandas de Bollinger"""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()

    upper = sma + (2 * std)
    lower = sma - (2 * std)
    bb_width = (upper - lower) / sma

    return bb_width.fillna(0.1)


def calculate_compression_days(df, window=5, max_range_pct=8):
    """
    Calcula días consecutivos en compresión
    Compresión = rango de precio < max_range_pct en ventana
    """
    compression = []

    for i in range(len(df)):
        if i < window:
            compression.append(0)
            continue

        days_compressed = 0
        for j in range(i, max(i - 20, 0), -1):  # Mirar hasta 20 días atrás
            if j < window:
                break

            prices_window = df['Close'].iloc[j - window:j].values
            if len(prices_window) < window:
                break

            price_range_pct = ((prices_window.max() - prices_window.min()) /
                              prices_window.min() * 100)

            if price_range_pct <= max_range_pct:
                days_compressed += 1
            else:
                break

        compression.append(days_compressed)

    return pd.Series(compression, index=df.index)


def compute_all_features(df):
    """
    Calcula todos los features necesarios para el modelo

    Returns DataFrame con columnas:
    - bb_width, adx, vol_ratio, rsi, macd_diff, atr_ratio,
      short_float, compression_days, volume_dry, price_range_pct
    """
    # Validar columnas requeridas
    required_cols = ["High", "Low", "Close", "Volume"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Columnas faltantes. Requeridas: {required_cols}")

    df = df.copy()

    # 1. Bollinger Bands Width
    df["bb_width"] = calculate_bollinger_bands(df["Close"], period=20)

    # 2. ADX (real)
    df["adx"] = calculate_adx(df, period=14)

    # 3. Volume Ratio
    avg_volume = df["Volume"].rolling(window=20).mean()
    df["vol_ratio"] = df["Volume"] / avg_volume
    df["vol_ratio"] = df["vol_ratio"].fillna(1.0)

    # 4. RSI
    df["rsi"] = calculate_rsi(df["Close"], period=14)

    # 5. MACD
    macd, macd_signal, macd_diff = calculate_macd(df["Close"])
    df["macd"] = macd
    df["macd_signal"] = macd_signal
    df["macd_diff"] = macd_diff

    # 6. ATR y ATR Ratio
    tr = pd.concat([
        df["High"] - df["Low"],
        abs(df["High"] - df["Close"].shift()),
        abs(df["Low"] - df["Close"].shift())
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(window=14).mean()
    df["atr_ratio"] = df["atr"] / df["Close"]
    df["atr_ratio"] = df["atr_ratio"].fillna(0.05)

    # 7. Short Float (intentar obtener de yfinance, si no usar default)
    # Nota: Este es un valor estático, no histórico
    df["short_float"] = 0.15  # Default - se actualizará si hay datos

    # 8. Compression Days
    df["compression_days"] = calculate_compression_days(df, window=5, max_range_pct=8)

    # 9. Volume Dry (volumen bajo promedio)
    df["volume_dry"] = (df["Volume"] < avg_volume * 0.80).astype(int)

    # 10. Price Range %
    df["price_range_pct"] = (
        (df["Close"].rolling(5).max() - df["Close"].rolling(5).min()) /
        df["Close"].rolling(5).min() * 100
    )
    df["price_range_pct"] = df["price_range_pct"].fillna(5.0)

    # Eliminar NaN
    df = df.dropna()

    return df


def get_short_interest(symbol):
    """
    Intenta obtener short interest real de yfinance

    Returns:
        float: % de short interest (0-1)
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        short_pct = info.get('shortPercentOfFloat', None)
        if short_pct is not None and short_pct > 0:
            return float(short_pct)

        # Alternativa
        short_ratio = info.get('shortRatio', None)
        if short_ratio is not None and short_ratio > 0:
            return min(float(short_ratio) * 0.05, 0.50)  # Estimación aproximada

        return 0.15  # Default

    except:
        return 0.15  # Default


# =========================================================
# DETECCIÓN DE SETUPS Y ETIQUETADO
# =========================================================

def detect_setups(df, symbol, explosion_threshold=0.15, lookahead_days=5):
    """
    Detecta días que podrían ser buenos setups de entrada

    Criterios más flexibles que V4 para generar más samples:
    - Compresión detectada (compression_days > 3 o bb_width < 0.15)
    - Volumen elevado (vol_ratio > 1.5) o comenzando a elevarse
    - RSI no sobrecomprado (< 75)
    - Estructura favorable

    Args:
        df: DataFrame con features calculados
        symbol: Ticker del símbolo
        explosion_threshold: % de ganancia para considerar 'exploded' (default 15%)
        lookahead_days: Días futuros para verificar explosión

    Returns:
        DataFrame con setups detectados
    """
    setups = []

    # Necesitamos al menos lookahead_days al final
    for i in range(len(df) - lookahead_days):
        row = df.iloc[i]

        # Criterios de setup (más flexibles)
        is_compressed = (row["bb_width"] < 0.15 or row["compression_days"] >= 3)
        has_structure = (row["price_range_pct"] < 10)
        not_overbought = (row["rsi"] < 75)
        volume_interest = (row["vol_ratio"] > 1.5)  # Volumen comenzando a subir

        # Setup válido si cumple condiciones básicas
        if is_compressed and has_structure and not_overbought:
            close_today = row["Close"]

            # Verificar si explotó en los próximos días
            future_highs = df["High"].iloc[i + 1 : i + lookahead_days + 1]

            if len(future_highs) > 0:
                max_future = future_highs.max()
                gain_pct = ((max_future - close_today) / close_today)
                exploded = int(gain_pct >= explosion_threshold)

                setups.append({
                    "symbol": symbol,
                    "date": df.index[i],
                    "bb_width": float(row["bb_width"]),
                    "adx": float(row["adx"]),
                    "vol_ratio": float(row["vol_ratio"]),
                    "rsi": float(row["rsi"]),
                    "macd_diff": float(row["macd_diff"]),
                    "atr_ratio": float(row["atr_ratio"]),
                    "short_float": float(row["short_float"]),
                    "compression_days": int(row["compression_days"]),
                    "volume_dry": int(row["volume_dry"]),
                    "price_range_pct": float(row["price_range_pct"]),
                    "exploded": exploded,
                    "gain_pct": float(gain_pct * 100)  # Para análisis
                })

    return pd.DataFrame(setups)


# =========================================================
# DESCARGA Y PROCESAMIENTO
# =========================================================

def fetch_data(symbol, months=12):
    """
    Descarga datos históricos de yfinance

    Returns:
        DataFrame con columnas: Open, High, Low, Close, Volume
    """
    try:
        period = f"{months}mo"
        df = yf.download(symbol, period=period, interval="1d", progress=False)

        if df.empty:
            raise ValueError(f"Sin datos para {symbol}")

        # Corregir columnas si son MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        # Normalizar nombres de columnas
        df.columns = [str(col).capitalize() for col in df.columns]

        # Verificar columnas mínimas
        required = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in df.columns for col in required):
            raise ValueError(f"Columnas incompletas para {symbol}")

        return df

    except Exception as e:
        raise ValueError(f"Error descargando {symbol}: {e}")


def process_symbol(symbol, months=12, explosion_threshold=0.15):
    """
    Procesa un símbolo completo: descarga, calcula features, detecta setups

    Returns:
        DataFrame con setups o None si falla
    """
    try:
        # Descargar datos
        df = fetch_data(symbol, months)

        if len(df) < 50:  # Mínimo de datos
            print(f"  ⚠️  {symbol}: Insuficientes datos ({len(df)} días)")
            return None

        # Calcular features
        df_feat = compute_all_features(df)

        # Obtener short interest real
        short_float = get_short_interest(symbol)
        df_feat["short_float"] = short_float

        # Detectar setups
        df_setups = detect_setups(df_feat, symbol, explosion_threshold)

        if len(df_setups) == 0:
            print(f"  ⚠️  {symbol}: No se detectaron setups")
            return None

        print(f"  ✅ {symbol}: {len(df_setups)} setups ({df_setups['exploded'].sum()} explotaron)")

        return df_setups

    except Exception as e:
        print(f"  ❌ {symbol}: Error - {e}")
        return None


def build_training_dataset(symbols, months=12, explosion_threshold=0.15, min_samples=50):
    """
    Construye dataset de entrenamiento completo

    Args:
        symbols: Lista de símbolos a procesar
        months: Meses de histórico
        explosion_threshold: % para considerar explosión
        min_samples: Mínimo de samples para guardar dataset

    Returns:
        DataFrame con dataset completo
    """
    print(f"\n{'='*70}")
    print(f"GENERANDO DATASET DE ENTRENAMIENTO")
    print(f"{'='*70}")
    print(f"Símbolos: {len(symbols)}")
    print(f"Periodo: {months} meses")
    print(f"Threshold explosión: {explosion_threshold*100:.0f}%")
    print(f"{'='*70}\n")

    all_data = []

    for sym in tqdm(symbols, desc="Procesando símbolos"):
        df_setups = process_symbol(sym, months, explosion_threshold)
        if df_setups is not None:
            all_data.append(df_setups)

    if not all_data:
        raise RuntimeError("❌ No se generaron datos para ningún símbolo")

    # Concatenar todo
    dataset = pd.concat(all_data, ignore_index=True)

    # Validación
    if len(dataset) < min_samples:
        print(f"\n⚠️  ADVERTENCIA: Solo {len(dataset)} samples generados (mínimo: {min_samples})")
        print("   Considera:")
        print("   - Aumentar el periodo (--months)")
        print("   - Agregar más símbolos a la watchlist")
        print("   - Reducir el threshold de explosión")

    # Estadísticas
    print(f"\n{'='*70}")
    print(f"DATASET GENERADO")
    print(f"{'='*70}")
    print(f"Total samples: {len(dataset)}")
    print(f"Explosiones (1): {dataset['exploded'].sum()} ({dataset['exploded'].mean()*100:.1f}%)")
    print(f"No explosiones (0): {(~dataset['exploded'].astype(bool)).sum()} ({(1-dataset['exploded'].mean())*100:.1f}%)")
    print(f"\nGanancia promedio en explosiones: {dataset[dataset['exploded']==1]['gain_pct'].mean():.1f}%")
    print(f"Ganancia promedio en no explosiones: {dataset[dataset['exploded']==0]['gain_pct'].mean():.1f}%")
    print(f"{'='*70}\n")

    return dataset


# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Genera dataset de entrenamiento para Penny Stock Advisor V5"
    )
    parser.add_argument(
        "--months",
        type=int,
        default=24,
        help="Meses de histórico (default: 24)"
    )
    parser.add_argument(
        "--explosion-threshold",
        type=float,
        default=0.15,
        help="Threshold de ganancia para 'exploded' (default: 0.15 = 15%%)"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=50,
        help="Mínimo de samples requeridos (default: 50)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_PATH,
        help=f"Ruta de salida (default: {OUTPUT_PATH})"
    )

    args = parser.parse_args()

    # Watchlist - Puedes modificar esta lista
    WATCHLIST_SYMBOLS = [
        # Originales
        "OPEN", "CHPT", "LCFY", "SIRI", "XAIR",
        "HTOO", "CTMX", "CLOV", "ALBT", "ADIL",
        "BYND", "AKBA", "OPAD", "AIRE", "YYAI",
        "RANI", "WOK", "AREB", "BENF", "CJET",
        "SBEV", "ISRG", "VTYX", "RGC", "RVPH",
        "ONDS", "ADTX", "CLSK", "BITF", "IREN",
        "WGRX", "ADAG", "QLGN", "VIVK", "ASNS",
        "DFLI", "DVLT",
        # Adicionales populares en penny stocks
        "COSM", "MULN", "SNTI", "GME", "AMC",
        "PLUG", "SOFI", "PLTR", "NIO", "RIVN",
        "LCID", "WISH", "HOOD", "DWAC"
    ]

    # Eliminar duplicados
    WATCHLIST_SYMBOLS = sorted(set(WATCHLIST_SYMBOLS))

    print(f"\n🔍 Watchlist: {len(WATCHLIST_SYMBOLS)} símbolos")
    print(f"📁 Output: {args.output}")

    try:
        # Generar dataset
        dataset = build_training_dataset(
            symbols=WATCHLIST_SYMBOLS,
            months=args.months,
            explosion_threshold=args.explosion_threshold,
            min_samples=args.min_samples
        )

        # Reordenar columnas para que coincidan con el modelo
        column_order = [
            'symbol', 'date', 'bb_width', 'adx', 'vol_ratio',
            'rsi', 'macd_diff', 'atr_ratio', 'short_float',
            'compression_days', 'volume_dry', 'price_range_pct',
            'exploded', 'gain_pct'
        ]
        dataset = dataset[column_order]

        # Guardar
        dataset.to_csv(args.output, index=False)
        print(f"✅ Dataset guardado exitosamente en: {args.output}")

        # Mostrar preview
        print(f"\n📊 Preview del dataset:\n")
        print(dataset.head(10).to_string())

        print(f"\n{'='*70}")
        print("✅ PROCESO COMPLETADO")
        print(f"{'='*70}")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
