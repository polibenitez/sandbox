# penny_v4.py
import numpy as np
import pandas as pd
import yfinance as yf
import ta

from typing import Dict, Tuple, Optional

# --------------------------
# Configuración / Presets
# --------------------------
DEFAULT_CONFIG = {
    "min_avg_volume": 200_000,
    "price_min": 0.2,
    "price_max": 10.0,
    "bbwidth_threshold": 0.12,    # < 12% -> compresión
    "adx_threshold": 20,
    "vol_ratio_trigger": 2.5,     # vol_today / avg_vol_20
    "vol_accel_min": 1.3,         # vol_today / vol_prev_day
    "rsi_entry": 55,
    "rsi_hold_min": 50,
    "macd_hist_cross": 0.0,
    "confirm_vol_pct": 0.6,       # day2 vol >= 60% vol_trigger_day
    "max_market_drop_pct": -1.5,   # if market drops more than this, avoid longs (percent)
    "entry_min_score": 0.70,
    "aggressive_entry_score": 0.80,
}

# --------------------------
# Utilities
# --------------------------
def download_history(ticker: str, period="6mo", interval="1d") -> pd.DataFrame:
    """Descarga histórico de yfinance y prepara DataFrame"""
    t = yf.Ticker(ticker)
    hist = t.history(period=period, interval=interval)
    if hist.empty:
        raise ValueError("No data for ticker")
    hist = hist.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume"
    })
    hist = hist[["open", "high", "low", "close", "volume"]].copy()
    hist.index = pd.to_datetime(hist.index)
    return hist

# --------------------------
# Technical indicators (pandas + ta)
# --------------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade columnas útiles:
      - bb_width (Bollinger bands width)
      - adx (14)
      - sma20, sma50
      - rsi14
      - macd, macd_signal, macd_hist
      - atr14
      - avg_vol_20
    """
    out = df.copy()

    # Bollinger Bands (20, 2)
    bb = ta.volatility.BollingerBands(close=out["close"], window=20, window_dev=2)
    out["bb_upper"] = bb.bollinger_hband()
    out["bb_lower"] = bb.bollinger_lband()
    out["bb_mid"] = bb.bollinger_mavg()
    out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / out["bb_mid"]

    # ADX
    adx_ind = ta.trend.ADXIndicator(high=out["high"], low=out["low"], close=out["close"], window=14)
    out["adx"] = adx_ind.adx()

    # SMA
    out["sma20"] = out["close"].rolling(20).mean()
    out["sma50"] = out["close"].rolling(50).mean()

    # RSI
    out["rsi14"] = ta.momentum.rsi(close=out["close"], window=14)

    # MACD
    macd = ta.trend.MACD(close=out["close"], window_slow=26, window_fast=12, window_sign=9)
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_hist"] = macd.macd_diff()

    # ATR
    atr = ta.volatility.average_true_range(high=out["high"], low=out["low"], close=out["close"], window=14)
    out["atr14"] = atr

    # Average volume 20
    out["avg_vol_20"] = out["volume"].rolling(20).mean()

    # previous day values for comparisons
    out["vol_prev_day"] = out["volume"].shift(1)
    out["close_prev_1d"] = out["close"].shift(1)
    out["high_prev_3d"] = out["high"].rolling(3).max().shift(1)  # previous 3-day high (exclude today)
    out["low_prev_3d"] = out["low"].rolling(3).min().shift(1)

    return out

# --------------------------
# SETUP Filter (vectorizable)
# --------------------------
def setup_filter(df: pd.DataFrame, config=DEFAULT_CONFIG) -> pd.Series:
    """
    Devuelve boolean Series con True = cumple condiciones de compresión / candidato.
    """
    cond_price = (df["close"] >= config["price_min"]) & (df["close"] <= config["price_max"])
    cond_vol = df["avg_vol_20"] >= config["min_avg_volume"]
    cond_bb = df["bb_width"] <= config["bbwidth_threshold"]
    cond_adx = df["adx"] <= config["adx_threshold"]

    candidate = cond_price & cond_vol & cond_bb & cond_adx
    return candidate.fillna(False)

# --------------------------
# TRIGGER - entry logic (row-wise)
# --------------------------
def score_trigger_row(row, config=DEFAULT_CONFIG) -> Dict:
    """
    Calcula un score para la fila (día).
    Retorna dict con detalle y score 0..1.
    """

    score_components = {}
    # Volumen relativo
    avg_vol = row.get("avg_vol_20", np.nan)
    vol_today = row.get("volume", np.nan)
    vol_prev = row.get("vol_prev_day", np.nan)

    vol_ratio = (vol_today / avg_vol) if (avg_vol and avg_vol > 0) else 0
    vol_accel = (vol_today / vol_prev) if (vol_prev and vol_prev > 0) else 1.0

    comp_vol = min(1.0, vol_ratio / config["vol_ratio_trigger"])  # 0..1
    comp_accel = min(1.0, (vol_accel - 1.0) / (config["vol_accel_min"] - 1.0)) if config["vol_accel_min"] > 1 else 0.0

    score_components["volume_score"] = float(np.clip(0.6 * comp_vol + 0.4 * comp_accel, 0, 1))

    # Price breakout vs recent high
    close = row.get("close", 0)
    high_prev_3d = row.get("high_prev_3d", 0)
    breakout_pct = ((close - high_prev_3d) / high_prev_3d) if high_prev_3d and high_prev_3d > 0 else -1.0
    breakout_score = 1.0 if breakout_pct > 0.02 else (0.6 if breakout_pct > 0.0 else 0.0)
    score_components["breakout_score"] = breakout_score

    # Momentum: RSI and MACD
    rsi = row.get("rsi14", 50)
    macd_hist = row.get("macd_hist", 0)
    rsi_score = 1.0 if rsi >= config["rsi_entry"] else (0.5 if rsi > (config["rsi_entry"] - 10) else 0.0)
    macd_score = 1.0 if macd_hist > 0 else 0.0
    score_components["momentum_score"] = float(np.clip(0.6 * rsi_score + 0.4 * macd_score, 0, 1))

    # Composite weighting (dynamic can be added using market regime)
    weights = {"volume_score": 0.35, "momentum_score": 0.30, "breakout_score": 0.35}
    composite = 0.0
    for k, w in weights.items():
        composite += score_components.get(k, 0.0) * w

    return {
        "score": float(composite),
        "components": score_components,
        "vol_ratio": float(vol_ratio) if not np.isnan(vol_ratio) else 0.0,
        "vol_accel": float(vol_accel) if not np.isnan(vol_accel) else 1.0,
        "breakout_pct": float(breakout_pct)
    }

# Vectorized wrapper to produce a series of scores
def score_trigger(df: pd.DataFrame, config=DEFAULT_CONFIG) -> pd.DataFrame:
    results = df.apply(lambda r: score_trigger_row(r, config), axis=1)
    # results is a Series of dicts; convert to DataFrame
    scored = pd.DataFrame(list(results), index=df.index)
    # explode components to columns
    comps = pd.json_normalize(scored["components"]).set_index(scored.index)
    scored = pd.concat([scored.drop(columns=["components"]), comps], axis=1)
    return scored

# --------------------------
# CONFIRM / MANAGEMENT
# --------------------------
def confirm_and_manage(df: pd.DataFrame, entry_idx, config=DEFAULT_CONFIG) -> Dict:
    """
    Dado un índice de entrada (Timestamp), verifica la confirmación los siguientes días
    y devuelve una gestión simple: should_hold, stop_price, take_profit_levels, exit_reason
    """
    row = df.loc[entry_idx]
    entry_price = float(row["close"])
    atr = float(row.get("atr14", entry_price * 0.05))

    # baseline stop - basado en ATR
    stop_price = entry_price - 1.8 * atr

    # take profit levels (percent-based example)
    tp1 = entry_price * 1.10
    tp2 = entry_price * 1.20
    tp3 = entry_price * 1.35

    # Check day+1 confirmation
    try:
        nxt = df.loc[entry_idx:].iloc[1]  # siguiente día
    except Exception:
        return {
            "should_hold": False,
            "reason": "No hay día siguiente",
            "stop_price": stop_price,
            "tps": [tp1, tp2, tp3]
        }

    vol_trigger = row["volume"]
    vol_next = nxt["volume"]
    vol_ok = vol_next >= config["confirm_vol_pct"] * vol_trigger

    close_next = nxt["close"]
    close_ok = close_next >= entry_price * 0.95

    rsi_next = nxt.get("rsi14", 50)
    rsi_ok = rsi_next >= config["rsi_hold_min"]

    should_hold = vol_ok and close_ok and rsi_ok

    reason = "confirmed" if should_hold else "failed confirmation"
    if not vol_ok:
        reason = "volume dropped"
    elif not close_ok:
        reason = "price dropped"
    elif not rsi_ok:
        reason = "rsi weakness"

    return {
        "should_hold": bool(should_hold),
        "reason": reason,
        "stop_price": float(stop_price),
        "tps": [float(tp1), float(tp2), float(tp3)]
    }

# --------------------------
# Entry point: scan symbol and produce signals
# --------------------------
def scan_symbol_for_signals(ticker: str, period="6mo", config=DEFAULT_CONFIG) -> Dict:
    df = download_history(ticker, period=period)
    df = compute_indicators(df)

    # Mark setup candidates
    df["is_candidate"] = setup_filter(df, config)

    # Score triggers for all rows
    scored = score_trigger(df, config)
    df = pd.concat([df, scored], axis=1)

    # Identify potential entries (candidate & score >= threshold & breakout > 0)
    entries = df[(df["is_candidate"]) & (df["score"] >= config["entry_min_score"]) & (df["breakout_pct"] > 0.0)]

    signals = []
    for idx, row in entries.iterrows():
        # Minimal market regime check (example: use SPY)
        # Here you would fetch market index; for simplicity, skip unless passed externally
        # confirm & manage
        mgmt = confirm_and_manage(df, idx, config)
        signals.append({
            "ticker": ticker,
            "entry_date": idx,
            "entry_price": float(row["close"]),
            "score": float(row["score"]),
            "vol_ratio": float(row["vol_ratio"]),
            "vol_accel": float(row["vol_accel"]),
            "breakout_pct": float(row["breakout_pct"]),
            "management": mgmt
        })

    return {
        "ticker": ticker,
        "df": df,
        "signals": signals
    }

# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    # Ejemplo rápido: analiza un ticker (modifica a tu gusto)
    WATCHLIST_SYMBOLS = [
        "OPEN", "CHPT", "LCFY", "SIRI", "XAIR",
        "HTOO", "CTMX", "CLOV", "ALBT", "ADIL",
        "BYND", "AKBA", "OPAD", "AIRE", "YYAI",
        "RANI", "WOK", "AREB", "BENF", "CJET", "SBEV", "ISRG", "VTYX", 
        "RGC", "RVPH", "ONDS", "ADTX", "CLSK", "BITF", "IREN", "WGRX", "ADAG", "QLGN",
        "VIVK", "ASNS", "DFLI"
    ]
    WATCHLIST_SYMBOLS = set(WATCHLIST_SYMBOLS)
    for TICKER in WATCHLIST_SYMBOLS:
        out = scan_symbol_for_signals(TICKER, period="3mo")
        print(f"Señales encontradas para {TICKER}: {len(out['signals'])}")
        for s in out["signals"]:
            print("---")
            print("Entry:", s["entry_date"], "Price:", s["entry_price"], "Score:", s["score"])
            print("Management:", s["management"])
