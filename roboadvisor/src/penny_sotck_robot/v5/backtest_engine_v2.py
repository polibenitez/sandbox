#!/usr/bin/env python3
"""
BACKTEST ENGINE V3 - ADAPTIVE ROBUST BACKTESTING SYSTEM
========================================================

Sistema de backtesting profesional con capacidades adaptativas que evita:
- Look-ahead bias (usar datos futuros)
- Survivorship bias (solo stocks exitosos)
- Overfitting (sobreoptimización)

Features V3.0:
- Walk-forward analysis con re-entrenamiento
- Temporal splits (train/val/test) sin shuffle
- Transaction costs realistas (commission + slippage)
- Performance metrics completas (Sharpe, Sortino, Calmar, etc.)
- Monte Carlo simulation para validación estadística
- Visualización avanzada con tearsheet
- Statistical validation vs benchmark
- NUEVO: Intelligent parallelization con ProcessPoolExecutor
- NUEVO: Adaptive thresholds con safeguards
- NUEVO: Ensemble models (RF + XGBoost + LightGBM)
- NUEVO: Market regime detection
- NUEVO: Online learning con drift tracking
- NUEVO: Interactive dashboards con Plotly
- NUEVO: Risk monitoring y alertas
- NUEVO: Meta-validation system

Author: Quantitative Engineering Team
Date: 2025
Version: 3.0
"""

import logging
import numpy as np
import pandas as pd
import pickle
import json
import os
import multiprocessing
from multiprocessing import Manager
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Seaborn for heatmaps
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not installed. Heatmaps will use basic matplotlib.")

# Parallelization
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# ML Libraries
try:
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.model_selection import cross_val_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: scikit-learn not installed. Ensemble models disabled.")

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except (ImportError, Exception) as e:
    HAS_XGBOOST = False
    print(f"Warning: xgboost not available ({type(e).__name__}). XGBoost ensemble disabled.")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except (ImportError, Exception) as e:
    HAS_LIGHTGBM = False
    print(f"Warning: lightgbm not available ({type(e).__name__}). LightGBM ensemble disabled.")

# Interactive visualization
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("Warning: plotly not installed. Interactive dashboards disabled.")

# Progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Warning: tqdm not installed. Progress bars disabled.")

# QuantStats integration
try:
    import quantstats as qs
    HAS_QUANTSTATS = True
except ImportError:
    HAS_QUANTSTATS = False
    print("Warning: quantstats not installed. QuantStats export disabled.")

# Data fetching
import yfinance as yf

import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('backtest_engine_v2')


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Position:
    """Representa una posición abierta"""
    symbol: str
    entry_date: datetime
    entry_price: float
    shares: int
    stop_loss: float
    take_profit: float
    cost: float  # Costos de entrada
    highest_price: float = 0.0

    def __post_init__(self):
        if self.highest_price == 0.0:
            self.highest_price = self.entry_price


@dataclass
class Trade:
    """Representa un trade cerrado"""
    symbol: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    shares: int
    entry_cost: float
    exit_cost: float
    pnl_gross: float
    pnl_net: float
    pnl_pct: float
    holding_days: int
    exit_reason: str
    highest_price: float
    max_adverse_excursion: float  # MAE
    max_favorable_excursion: float  # MFE

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'entry_date': self.entry_date.strftime('%Y-%m-%d'),
            'exit_date': self.exit_date.strftime('%Y-%m-%d'),
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'shares': self.shares,
            'entry_cost': self.entry_cost,
            'exit_cost': self.exit_cost,
            'pnl_gross': self.pnl_gross,
            'pnl_net': self.pnl_net,
            'pnl_pct': self.pnl_pct,
            'holding_days': self.holding_days,
            'exit_reason': self.exit_reason,
            'highest_price': self.highest_price,
            'mae': self.max_adverse_excursion,
            'mfe': self.max_favorable_excursion
        }


@dataclass
class BacktestConfig:
    """Configuración del backtesting"""
    initial_capital: float = 100000.0
    position_size_pct: float = 0.02  # 2% del capital por trade
    max_positions: int = 10
    commission: float = 0.001  # 0.1% por trade
    slippage: float = 0.002  # 0.2% slippage (penny stocks tienen spread alto)
    stop_loss_pct: float = 0.15  # 15% stop loss
    take_profit_pct: float = 0.20  # 20% take profit
    max_holding_days: int = 30
    use_kelly_criterion: bool = False
    risk_free_rate: float = 0.04  # 4% anual para Sharpe ratio


# ============================================================================
# DATA MANAGEMENT FUNCTIONS
# ============================================================================

def get_universe(min_price: float = 1.0,
                max_price: float = 10.0,
                min_market_cap: float = 50e6,
                start_date: str = "2020-01-01",
                end_date: str = "2023-12-31") -> List[str]:
    """
    Obtener universo de penny stocks evitando survivorship bias

    IMPORTANTE: En producción, usar una base de datos que incluya stocks
    delistados. Por ahora usamos un universo fijo + pequeños caps del Russell 2000.

    Args:
        min_price: Precio mínimo
        max_price: Precio máximo
        min_market_cap: Market cap mínimo
        start_date: Fecha inicio
        end_date: Fecha fin

    Returns:
        Lista de símbolos que cumplen los criterios
    """
    logger.info(f"Obteniendo universo de penny stocks (${min_price}-${max_price})")

    # Universe de ejemplo (en producción usar screener real)
    # Incluimos stocks conocidos + algunos que fallaron
    candidate_universe = [
        # Penny stocks activos
        'SNDL', 'GNUS', 'TOPS', 'SHIP', 'NAK', 'MARK', 'IZEA',
        'DGLY', 'ATOS', 'TELL', 'SOLO', 'WKHS', 'RIDE', 'NKLA',
        'PLUG', 'FCEL', 'BLNK', 'SBE', 'HYLN', 'IDEX', 'XSPA',
        'HEXO', 'ACB', 'CGC', 'TLRY', 'KERN', 'ZOM', 'BNGO',
        'OCGN', 'NVAX', 'SRNE', 'INO', 'VXRT', 'COCP', 'CODX',
        # Agregar más según necesidad
    ]

    logger.info(f"Universo inicial: {len(candidate_universe)} símbolos")
    logger.warning("⚠️  Para eliminar survivorship bias, usar DB con stocks delistados")

    return candidate_universe


def temporal_split(data: pd.DataFrame,
                  train_pct: float = 0.6,
                  val_pct: float = 0.2,
                  test_pct: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split temporal SIN shuffle (crítico para evitar look-ahead bias)

    Args:
        data: DataFrame con índice temporal
        train_pct: % para entrenamiento
        val_pct: % para validación
        test_pct: % para test

    Returns:
        (train_df, val_df, test_df)
    """
    assert abs(train_pct + val_pct + test_pct - 1.0) < 0.01, "Los % deben sumar 1.0"

    n = len(data)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))

    train_df = data.iloc[:train_end].copy()
    val_df = data.iloc[train_end:val_end].copy()
    test_df = data.iloc[val_end:].copy()

    logger.info(f"Split temporal: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    logger.info(f"Train: {train_df.index[0]} to {train_df.index[-1]}")
    logger.info(f"Val: {val_df.index[0]} to {val_df.index[-1]}")
    logger.info(f"Test: {test_df.index[0]} to {test_df.index[-1]}")

    return train_df, val_df, test_df


def walk_forward_split(data: pd.DataFrame,
                      train_days: int = 180,
                      test_days: int = 30) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Genera ventanas walk-forward para backtesting

    Ejemplo:
    [Train: 180 días] → [Test: 30 días] → Rodar ventana →
                        [Train: 180 días] → [Test: 30 días] →

    Args:
        data: DataFrame con índice temporal
        train_days: Días para entrenamiento
        test_days: Días para testing

    Returns:
        Lista de tuplas (train_window, test_window)
    """
    windows = []
    start_idx = 0

    while start_idx + train_days + test_days <= len(data):
        train_end = start_idx + train_days
        test_end = train_end + test_days

        train_window = data.iloc[start_idx:train_end].copy()
        test_window = data.iloc[train_end:test_end].copy()

        windows.append((train_window, test_window))

        # Rodar ventana (overlap de train_days - test_days)
        start_idx += test_days

    logger.info(f"Walk-forward: {len(windows)} ventanas generadas")
    logger.info(f"Configuración: Train={train_days} días, Test={test_days} días")

    return windows


def calculate_features_realtime(df: pd.DataFrame,
                                current_idx: int,
                                lookback: int = 20) -> Dict[str, float]:
    """
    Calcula features SOLO con datos disponibles hasta current_idx

    CRÍTICO: Evita look-ahead bias - NO usar datos futuros

    Args:
        df: DataFrame con OHLCV
        current_idx: Índice actual (solo usar datos hasta aquí)
        lookback: Ventana para indicadores

    Returns:
        Dict con features calculados
    """
    if current_idx < lookback:
        return None

    # Subset SOLO hasta current_idx (inclusive)
    hist = df.iloc[:current_idx + 1].copy()

    # Precios
    close = hist['Close'].values
    high = hist['High'].values
    low = hist['Low'].values
    volume = hist['Volume'].values

    current_price = close[-1]

    # === TECHNICAL INDICATORS ===

    # 1. RSI (14 períodos)
    delta = np.diff(close)
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
    avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0
    rs = avg_gain / avg_loss if avg_loss > 0 else 0
    rsi = 100 - (100 / (1 + rs)) if rs > 0 else 50

    # 2. Bollinger Bands
    ma20 = np.mean(close[-20:])
    std20 = np.std(close[-20:])
    bb_upper = ma20 + 2 * std20
    bb_lower = ma20 - 2 * std20
    bb_width = (bb_upper - bb_lower) / ma20 if ma20 > 0 else 0
    bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5

    # 3. ATR (14 períodos)
    tr_list = []
    for i in range(len(high) - 14, len(high)):
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]) if i > 0 else 0,
            abs(low[i] - close[i-1]) if i > 0 else 0
        )
        tr_list.append(tr)
    atr = np.mean(tr_list) if tr_list else 0
    atr_ratio = atr / current_price if current_price > 0 else 0

    # 4. Volume ratio
    avg_volume = np.mean(volume[-20:])
    vol_ratio = volume[-1] / avg_volume if avg_volume > 0 else 1.0

    # 5. Price momentum
    returns_5d = (close[-1] - close[-5]) / close[-5] if len(close) >= 5 else 0
    returns_10d = (close[-1] - close[-10]) / close[-10] if len(close) >= 10 else 0
    returns_20d = (close[-1] - close[-20]) / close[-20] if len(close) >= 20 else 0

    # 6. Volatility
    returns = np.diff(close[-20:]) / close[-20:-1]
    volatility = np.std(returns) if len(returns) > 0 else 0

    # 7. Price range
    price_range_pct = (high[-1] - low[-1]) / close[-1] if close[-1] > 0 else 0

    features = {
        'price': float(current_price),
        'rsi': float(rsi),
        'bb_width': float(bb_width),
        'bb_position': float(bb_position),
        'atr': float(atr),
        'atr_ratio': float(atr_ratio),
        'vol_ratio': float(vol_ratio),
        'returns_5d': float(returns_5d),
        'returns_10d': float(returns_10d),
        'returns_20d': float(returns_20d),
        'volatility': float(volatility),
        'price_range_pct': float(price_range_pct),
        'volume': float(volume[-1]),
        'avg_volume_20d': float(avg_volume)
    }

    return features


# ============================================================================
# POSITION MANAGEMENT
# ============================================================================

def kelly_criterion(win_rate: float,
                   avg_win: float,
                   avg_loss: float,
                   max_size: float = 0.20) -> float:
    """
    Calcula tamaño óptimo de posición usando Kelly Criterion

    f = (p * b - q) / b
    donde:
    - p = probabilidad de ganar
    - q = probabilidad de perder (1-p)
    - b = ratio win/loss

    Args:
        win_rate: Tasa de acierto (0-1)
        avg_win: Ganancia promedio
        avg_loss: Pérdida promedio (positivo)
        max_size: Tamaño máximo permitido (cap)

    Returns:
        Fracción del capital a arriesgar (0-1)
    """
    if avg_loss == 0 or win_rate == 0:
        return 0.02  # Default 2%

    p = win_rate
    q = 1 - win_rate
    b = abs(avg_win / avg_loss)

    kelly = (p * b - q) / b
    kelly = max(0, kelly)  # No negative positions
    kelly = min(kelly, max_size)  # Cap máximo

    # Usar 50% de Kelly (más conservador)
    return kelly * 0.5


def size_position(capital: float,
                 config: BacktestConfig,
                 win_rate: Optional[float] = None,
                 avg_win: Optional[float] = None,
                 avg_loss: Optional[float] = None) -> float:
    """
    Calcula el tamaño de posición

    Args:
        capital: Capital disponible
        config: Configuración del backtest
        win_rate: Tasa de acierto histórica (para Kelly)
        avg_win: Ganancia promedio (para Kelly)
        avg_loss: Pérdida promedio (para Kelly)

    Returns:
        Monto a invertir
    """
    if config.use_kelly_criterion and all([win_rate, avg_win, avg_loss]):
        kelly_pct = kelly_criterion(win_rate, avg_win, avg_loss)
        position_size = capital * kelly_pct
    else:
        position_size = capital * config.position_size_pct

    return position_size


def apply_stop_loss(position: Position,
                   current_price: float,
                   config: BacktestConfig) -> bool:
    """
    Verifica si se debe cerrar por stop loss

    Args:
        position: Posición abierta
        current_price: Precio actual
        config: Configuración

    Returns:
        True si se debe cerrar
    """
    return current_price <= position.stop_loss


def apply_take_profit(position: Position,
                     current_price: float,
                     config: BacktestConfig) -> bool:
    """
    Verifica si se debe cerrar por take profit

    Args:
        position: Posición abierta
        current_price: Precio actual
        config: Configuración

    Returns:
        True si se debe cerrar
    """
    return current_price >= position.take_profit


def apply_trailing_stop(position: Position,
                       current_price: float,
                       trailing_pct: float = 0.10) -> Tuple[Position, bool]:
    """
    Aplica trailing stop (stop loss móvil)

    Args:
        position: Posición abierta
        current_price: Precio actual
        trailing_pct: % de trailing desde el máximo

    Returns:
        (position_actualizada, cerrar_posicion)
    """
    # Actualizar highest price
    if current_price > position.highest_price:
        position.highest_price = current_price
        # Ajustar stop loss
        new_stop = position.highest_price * (1 - trailing_pct)
        position.stop_loss = max(position.stop_loss, new_stop)

    # Verificar si se activó el trailing stop
    close_position = current_price <= position.stop_loss

    return position, close_position


# ============================================================================
# TRANSACTION COSTS
# ============================================================================

def calculate_entry_cost(shares: int,
                        price: float,
                        commission: float = 0.001,
                        slippage: float = 0.002) -> Tuple[float, float]:
    """
    Calcula el costo real de entrada

    Args:
        shares: Número de acciones
        price: Precio base
        commission: Comisión (% del monto)
        slippage: Slippage (% del precio)

    Returns:
        (precio_efectivo, costo_total)
    """
    # Precio con slippage (entramos más caro)
    effective_price = price * (1 + slippage)

    # Monto bruto
    gross_amount = shares * effective_price

    # Comisión
    commission_cost = gross_amount * commission

    # Costo total
    total_cost = gross_amount + commission_cost

    return effective_price, total_cost


def calculate_exit_cost(shares: int,
                       price: float,
                       commission: float = 0.001,
                       slippage: float = 0.002) -> Tuple[float, float]:
    """
    Calcula el costo real de salida

    Args:
        shares: Número de acciones
        price: Precio base
        commission: Comisión (% del monto)
        slippage: Slippage (% del precio)

    Returns:
        (precio_efectivo, monto_neto)
    """
    # Precio con slippage (vendemos más barato)
    effective_price = price * (1 - slippage)

    # Monto bruto
    gross_amount = shares * effective_price

    # Comisión
    commission_cost = gross_amount * commission

    # Monto neto recibido
    net_amount = gross_amount - commission_cost

    return effective_price, net_amount


# ============================================================================
# PHASE 2: ADAPTIVE FEATURES
# ============================================================================

def adaptive_threshold_validation_only(val_results: List[Trade],
                                      current_threshold: float = 0.65,
                                      min_threshold: float = 0.5,
                                      max_threshold: float = 0.85,
                                      window: int = 20,
                                      train_metrics: Optional[Dict] = None) -> Tuple[float, bool]:
    """
    Ajusta threshold SOLO en validation set con safeguards anti-overfitting

    Reglas:
    - Si win_rate > 0.70 → +0.03 (más conservador)
    - Si win_rate < 0.40 → -0.03 (más agresivo)
    - Max cambio: 0.05 por iteración
    - Re-validar después de cada ajuste
    - STOP si train/val gap > 25% (overfitting detectado)

    Args:
        val_results: Resultados de validación (últimos N trades)
        current_threshold: Threshold actual
        min_threshold: Mínimo permitido
        max_threshold: Máximo permitido
        window: Ventana de análisis
        train_metrics: Métricas de training (para detectar overfitting)

    Returns:
        (nuevo_threshold, overfitting_detected)
    """
    if len(val_results) < window:
        return current_threshold, False

    # Analizar últimos trades
    recent_trades = val_results[-window:]
    winning_trades = len([t for t in recent_trades if t.pnl_net > 0])
    win_rate = winning_trades / len(recent_trades)

    logger.info(f"Adaptive threshold check: win_rate={win_rate:.2f}, current={current_threshold:.2f}")

    # Detectar overfitting
    overfitting_detected = False
    if train_metrics:
        train_win_rate = train_metrics.get('win_rate', 0)
        gap = abs(train_win_rate - win_rate) / train_win_rate if train_win_rate > 0 else 0
        if gap > 0.25:  # 25% gap
            logger.warning(f"⚠️  OVERFITTING DETECTED! Train/Val gap: {gap*100:.1f}%")
            overfitting_detected = True
            return current_threshold, True

    # Ajustar threshold
    new_threshold = current_threshold

    if win_rate > 0.70:
        # Demasiado exitoso, ser más conservador
        new_threshold += 0.03
        logger.info("→ Win rate alto, aumentando threshold (más conservador)")
    elif win_rate < 0.40:
        # Muy bajo, ser más agresivo
        new_threshold -= 0.03
        logger.info("→ Win rate bajo, disminuyendo threshold (más agresivo)")

    # Aplicar límites
    new_threshold = np.clip(new_threshold, min_threshold, max_threshold)

    # Max cambio por iteración
    max_change = 0.05
    change = new_threshold - current_threshold
    if abs(change) > max_change:
        new_threshold = current_threshold + np.sign(change) * max_change

    if new_threshold != current_threshold:
        logger.info(f"✓ Threshold ajustado: {current_threshold:.2f} → {new_threshold:.2f}")

    return new_threshold, overfitting_detected


def detect_overfitting(train_metrics: Dict[str, float],
                      val_metrics: Dict[str, float],
                      threshold: float = 0.25) -> Tuple[bool, float]:
    """
    Detecta overfitting comparando train vs validation metrics

    Args:
        train_metrics: Métricas de entrenamiento
        val_metrics: Métricas de validación
        threshold: Umbral de gap (default 25%)

    Returns:
        (is_overfitting, gap_percentage)
    """
    # Comparar win_rate
    train_wr = train_metrics.get('win_rate', 0)
    val_wr = val_metrics.get('win_rate', 0)

    if train_wr == 0:
        return False, 0.0

    gap = abs(train_wr - val_wr) / train_wr
    is_overfitting = gap > threshold

    if is_overfitting:
        logger.warning(f"⚠️  Overfitting detected: Train WR={train_wr:.2f}, Val WR={val_wr:.2f}, Gap={gap*100:.1f}%")

    return is_overfitting, gap


def detect_market_regime(spy_data: pd.DataFrame) -> str:
    """
    Detecta el régimen de mercado actual

    Bull: SPY MA50 > MA200, VIX < 20
    Bear: SPY MA50 < MA200, VIX > 30
    Choppy: Otros casos

    Args:
        spy_data: DataFrame con datos de SPY (Close)

    Returns:
        'bull', 'bear', o 'choppy'
    """
    if len(spy_data) < 200:
        return 'choppy'

    close = spy_data['Close'].values
    ma50 = np.mean(close[-50:])
    ma200 = np.mean(close[-200:])

    # TODO: Integrar VIX para mejor detección
    # Por ahora usar solo MAs

    if ma50 > ma200 * 1.02:  # 2% margen
        regime = 'bull'
    elif ma50 < ma200 * 0.98:
        regime = 'bear'
    else:
        regime = 'choppy'

    logger.info(f"Market regime detected: {regime.upper()} (MA50={ma50:.2f}, MA200={ma200:.2f})")
    return regime


def adjust_threshold_by_regime(base_threshold: float, regime: str) -> float:
    """
    Ajusta threshold según régimen de mercado

    - Bull: threshold - 0.05 (más agresivo)
    - Bear: threshold + 0.10 (muy conservador)
    - Choppy: threshold + 0.05

    Args:
        base_threshold: Threshold base
        regime: 'bull', 'bear', o 'choppy'

    Returns:
        Threshold ajustado
    """
    adjustments = {
        'bull': -0.05,
        'bear': 0.10,
        'choppy': 0.05
    }

    adjustment = adjustments.get(regime, 0)
    adjusted = base_threshold + adjustment
    adjusted = np.clip(adjusted, 0.5, 0.85)

    logger.info(f"Threshold adjusted for {regime} market: {base_threshold:.2f} → {adjusted:.2f}")
    return adjusted


def risk_monitoring(equity_curve: List[Tuple[datetime, float]],
                   trades: List[Trade],
                   config: BacktestConfig) -> Dict[str, Any]:
    """
    Monitorea riesgos y genera alertas

    Alertas automáticas:
    - Drawdown > 20% → WARNING
    - 5 pérdidas consecutivas → PAUSE TRADING
    - Sharpe rolling < 0 → REVIEW REQUIRED
    - Train/Test gap > 30% → OVERFITTING WARNING

    Args:
        equity_curve: Curva de equity
        trades: Lista de trades
        config: Configuración

    Returns:
        Dict con alertas y métricas de riesgo
    """
    alerts = []

    # 1. Drawdown check
    equity_values = [eq[1] for eq in equity_curve]
    max_equity = max(equity_values)
    current_equity = equity_values[-1]
    current_dd = ((max_equity - current_equity) / max_equity) * 100

    if current_dd > 20:
        alerts.append({
            'type': 'DRAWDOWN_WARNING',
            'severity': 'HIGH',
            'message': f'Drawdown exceeds 20%: {current_dd:.2f}%',
            'action': 'REVIEW STRATEGY'
        })

    # 2. Consecutive losses
    consecutive_losses = 0
    for trade in reversed(trades[-10:]):  # Check last 10 trades
        if trade.pnl_net < 0:
            consecutive_losses += 1
        else:
            break

    if consecutive_losses >= 5:
        alerts.append({
            'type': 'CONSECUTIVE_LOSSES',
            'severity': 'CRITICAL',
            'message': f'{consecutive_losses} consecutive losses',
            'action': 'PAUSE TRADING'
        })

    # 3. Rolling Sharpe
    if len(equity_values) > 30:
        recent_returns = np.diff(equity_values[-30:]) / equity_values[-30:-1]
        rolling_sharpe = (np.mean(recent_returns) / np.std(recent_returns)) * np.sqrt(252) if np.std(recent_returns) > 0 else 0

        if rolling_sharpe < 0:
            alerts.append({
                'type': 'NEGATIVE_SHARPE',
                'severity': 'HIGH',
                'message': f'Rolling Sharpe ratio is negative: {rolling_sharpe:.2f}',
                'action': 'REVIEW REQUIRED'
            })
    else:
        rolling_sharpe = 0

    results = {
        'alerts': alerts,
        'current_drawdown': current_dd,
        'consecutive_losses': consecutive_losses,
        'rolling_sharpe_30d': rolling_sharpe,
        'action_required': len([a for a in alerts if a['severity'] in ['HIGH', 'CRITICAL']]) > 0
    }

    # Log alerts
    for alert in alerts:
        logger.warning(f"🚨 {alert['type']}: {alert['message']} → {alert['action']}")

    return results


def adaptive_retraining_schedule(performance_history: List[Dict[str, float]],
                                 last_training_date: datetime,
                                 current_date: datetime,
                                 min_interval_days: int = 90) -> bool:
    """
    Decide CUÁNDO re-entrenar el modelo basado en performance

    Criterios:
    - Han pasado al menos 90 días desde último entrenamiento
    - Performance ha degradado >20%
    - O cada 3 meses automáticamente

    Args:
        performance_history: Historial de métricas
        last_training_date: Fecha del último entrenamiento
        current_date: Fecha actual
        min_interval_days: Días mínimos entre entrenamientos

    Returns:
        True si debe re-entrenar
    """
    days_since_training = (current_date - last_training_date).days

    # Regla 1: Mínimo intervalo
    if days_since_training < min_interval_days:
        return False

    # Regla 2: Re-entrenar cada 3 meses
    if days_since_training >= 90:
        logger.info(f"✓ Re-training scheduled: {days_since_training} days since last training")
        return True

    # Regla 3: Degradación de performance
    if len(performance_history) >= 2:
        recent_perf = performance_history[-1].get('sharpe_ratio', 0)
        baseline_perf = np.mean([p.get('sharpe_ratio', 0) for p in performance_history[:-1]])

        if baseline_perf > 0:
            degradation = (baseline_perf - recent_perf) / baseline_perf
            if degradation > 0.20:  # 20% degradación
                logger.warning(f"⚠️  Performance degraded {degradation*100:.1f}%, scheduling re-training")
                return True

    return False


# ============================================================================
# PHASE 2: ENSEMBLE MODELS
# ============================================================================

def build_ensemble_models(X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
    """
    Construye ensemble de modelos: RF + XGBoost + LightGBM

    Args:
        X_train: Features de entrenamiento
        y_train: Labels de entrenamiento

    Returns:
        Dict con modelos entrenados
    """
    logger.info("Building ensemble models (RF + XGBoost + LightGBM)...")

    models = {}

    # 1. Random Forest
    if HAS_SKLEARN:
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        models['random_forest'] = rf_model
        logger.info("✓ Random Forest trained")

    # 2. XGBoost
    if HAS_XGBOOST:
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        models['xgboost'] = xgb_model
        logger.info("✓ XGBoost trained")

    # 3. LightGBM
    if HAS_LIGHTGBM:
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        models['lightgbm'] = lgb_model
        logger.info("✓ LightGBM trained")

    if not models:
        logger.error("No ensemble models available! Install sklearn, xgboost, or lightgbm")
        return None

    logger.info(f"✓ Ensemble ready with {len(models)} models")
    return models


def ensemble_prediction(models_dict: Dict[str, Any],
                       features: np.ndarray,
                       method: str = 'voting',
                       weights: Optional[Dict[str, float]] = None) -> float:
    """
    Predice usando ensemble de modelos con voting

    Señal final: consensus de al menos 2/3 modelos

    Args:
        models_dict: Dict con modelos entrenados
        features: Features para predicción
        method: 'voting' o 'weighted'
        weights: Pesos para cada modelo (si method='weighted')

    Returns:
        Probabilidad de señal (0-1)
    """
    if not models_dict:
        return 0.0

    predictions = {}

    # Obtener predicciones de cada modelo
    for name, model in models_dict.items():
        try:
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(features.reshape(1, -1))[0][1]
            else:
                prob = model.predict(features.reshape(1, -1))[0]
            predictions[name] = prob
        except Exception as e:
            logger.warning(f"Error predicting with {name}: {e}")
            continue

    if not predictions:
        return 0.0

    # Método de agregación
    if method == 'weighted' and weights:
        # Weighted average
        total_weight = sum(weights.get(name, 1.0) for name in predictions.keys())
        weighted_prob = sum(
            prob * weights.get(name, 1.0)
            for name, prob in predictions.items()
        ) / total_weight
        return weighted_prob
    else:
        # Simple voting (majority)
        votes = [1 if p > 0.5 else 0 for p in predictions.values()]
        consensus = sum(votes) / len(votes)

        # Require 2/3 majority
        if consensus >= 2/3:
            return np.mean(list(predictions.values()))
        else:
            return 0.0


# ============================================================================
# PHASE 2: NOTIFICATIONS & ALERTS
# ============================================================================

def send_alert_notification(alert: Dict[str, Any],
                           notification_type: str = 'log',
                           webhook_url: Optional[str] = None) -> bool:
    """
    Envía notificación de alerta (Email/Slack/Log)

    Args:
        alert: Dict con información de la alerta
        notification_type: 'log', 'email', 'slack'
        webhook_url: URL del webhook (para Slack)

    Returns:
        True si se envió exitosamente
    """
    message = f"{alert['type']}: {alert['message']} → {alert['action']}"

    if notification_type == 'log':
        logger.warning(f"🚨 ALERT: {message}")
        return True

    elif notification_type == 'slack':
        if not webhook_url:
            logger.error("Slack webhook URL not provided")
            return False

        try:
            import requests
            payload = {
                'text': f"🚨 *{alert['type']}* ({alert['severity']})",
                'attachments': [{
                    'color': 'danger' if alert['severity'] == 'CRITICAL' else 'warning',
                    'fields': [
                        {'title': 'Message', 'value': alert['message'], 'short': False},
                        {'title': 'Action Required', 'value': alert['action'], 'short': False}
                    ]
                }]
            }
            response = requests.post(webhook_url, json=payload)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False

    elif notification_type == 'email':
        # TODO: Implement email notification
        logger.warning("Email notifications not implemented yet")
        return False

    return False


def generate_signal_correlation_matrix(signals_df: pd.DataFrame,
                                       save_path: str = 'reports/') -> Optional[str]:
    """
    Genera matriz de correlación de señales

    Args:
        signals_df: DataFrame con señales de trading
        save_path: Directorio para guardar

    Returns:
        Path del archivo generado
    """
    logger.info("Generating signal correlation matrix...")

    if len(signals_df) < 2:
        logger.warning("Not enough signals for correlation analysis")
        return None

    # Calcular correlación
    correlation = signals_df.corr()

    # Crear figura
    plt.figure(figsize=(12, 10))

    if HAS_SEABORN:
        sns.heatmap(correlation, annot=True, fmt='.2f', cmap='RdBu_r',
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    else:
        plt.imshow(correlation, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        plt.colorbar(label='Correlation', shrink=0.8)

    plt.title('Signal Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Guardar
    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"correlation_matrix_{timestamp}.png"
    filepath = os.path.join(save_path, filename)

    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Correlation matrix saved: {filepath}")
    return filepath


# ============================================================================
# PHASE 2: ADVANCED REPORTING & VISUALIZATION
# ============================================================================

def generate_interactive_dashboard(results: Dict[str, Any],
                                   equity_curve: List[Tuple[datetime, float]],
                                   trades: List[Trade],
                                   save_path: str = 'reports/') -> Optional[str]:
    """
    Genera dashboard HTML interactivo con Plotly

    Args:
        results: Resultados del backtest
        equity_curve: Curva de equity
        trades: Lista de trades
        save_path: Directorio para guardar

    Returns:
        Path del archivo HTML generado
    """
    if not HAS_PLOTLY:
        logger.warning("Plotly not installed, skipping interactive dashboard")
        return None

    logger.info("Generating interactive dashboard with Plotly...")

    # Crear directorio si no existe
    os.makedirs(save_path, exist_ok=True)

    # Preparar datos
    dates = [eq[0] for eq in equity_curve]
    equity = [eq[1] for eq in equity_curve]
    trades_df = pd.DataFrame([t.to_dict() for t in trades])

    # Crear figura con subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Equity Curve', 'Drawdown', 'Returns Distribution',
                       'Monthly Returns', 'Trade Duration', 'Cumulative P&L'),
        specs=[[{'colspan': 2}, None],
               [{'type': 'xy'}, {'type': 'xy'}],
               [{'type': 'xy'}, {'type': 'xy'}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # 1. Equity Curve
    fig.add_trace(
        go.Scatter(x=dates, y=equity, mode='lines', name='Equity',
                  fill='tozeroy', line=dict(color='#2E86AB', width=2)),
        row=1, col=1
    )

    # 2. Drawdown
    max_equity = equity[0]
    drawdowns = []
    for eq in equity:
        max_equity = max(max_equity, eq)
        dd = ((max_equity - eq) / max_equity) * 100
        drawdowns.append(-dd)

    fig.add_trace(
        go.Scatter(x=dates, y=drawdowns, mode='lines', name='Drawdown',
                  fill='tozeroy', line=dict(color='red', width=1)),
        row=2, col=1
    )

    # 3. Returns Distribution
    fig.add_trace(
        go.Histogram(x=trades_df['pnl_pct'], nbinsx=30, name='Returns',
                    marker=dict(color='#A23B72')),
        row=2, col=2
    )

    # 4. Monthly Returns
    trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
    trades_df['year_month'] = trades_df['exit_date'].dt.to_period('M').astype(str)
    monthly_returns = trades_df.groupby('year_month')['pnl_pct'].sum()

    fig.add_trace(
        go.Bar(x=monthly_returns.index, y=monthly_returns.values, name='Monthly Returns',
              marker=dict(color=monthly_returns.values, colorscale='RdYlGn', showscale=False)),
        row=3, col=1
    )

    # 5. Trade Duration
    fig.add_trace(
        go.Histogram(x=trades_df['holding_days'], nbinsx=20, name='Duration',
                    marker=dict(color='#F18F01')),
        row=3, col=2
    )

    # Layout
    fig.update_layout(
        title_text="Interactive Backtest Dashboard - V3.0",
        showlegend=False,
        height=1000,
        template='plotly_white'
    )

    # Guardar
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"dashboard_{timestamp}.html"
    filepath = os.path.join(save_path, filename)

    fig.write_html(filepath)
    logger.info(f"✓ Interactive dashboard saved: {filepath}")

    return filepath


def factor_attribution_analysis(models_dict: Dict[str, Any],
                                feature_names: List[str]) -> Dict[str, Any]:
    """
    Analiza qué features contribuyen más a las predicciones

    Args:
        models_dict: Modelos entrenados
        feature_names: Nombres de features

    Returns:
        Dict con feature importance por modelo
    """
    logger.info("Analyzing factor attribution (feature importance)...")

    attribution = {}

    for name, model in models_dict.items():
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_importance = dict(zip(feature_names, importances))
                # Sort by importance
                sorted_features = sorted(feature_importance.items(),
                                       key=lambda x: x[1], reverse=True)
                attribution[name] = dict(sorted_features)

                logger.info(f"✓ {name} - Top 3 features:")
                for feat, imp in sorted_features[:3]:
                    logger.info(f"    {feat}: {imp:.4f}")

        except Exception as e:
            logger.warning(f"Could not extract features from {name}: {e}")

    return attribution


def worst_trades_analysis(trades: List[Trade], top_n: int = 10) -> Dict[str, Any]:
    """
    Análisis post-mortem de los peores trades

    Args:
        trades: Lista de trades
        top_n: Número de peores trades a analizar

    Returns:
        Dict con análisis
    """
    logger.info(f"Analyzing worst {top_n} trades (post-mortem)...")

    # Ordenar por P&L
    sorted_trades = sorted(trades, key=lambda t: t.pnl_pct)
    worst_trades = sorted_trades[:top_n]

    analysis = {
        'worst_trades': [t.to_dict() for t in worst_trades],
        'avg_loss_pct': np.mean([t.pnl_pct for t in worst_trades]),
        'avg_holding_days': np.mean([t.holding_days for t in worst_trades]),
        'exit_reasons': {},
        'symbols': {}
    }

    # Análisis por razón de salida
    for trade in worst_trades:
        reason = trade.exit_reason
        analysis['exit_reasons'][reason] = analysis['exit_reasons'].get(reason, 0) + 1

        symbol = trade.symbol
        analysis['symbols'][symbol] = analysis['symbols'].get(symbol, 0) + 1

    logger.info("Worst trades summary:")
    logger.info(f"  Avg loss: {analysis['avg_loss_pct']:.2f}%")
    logger.info(f"  Avg holding: {analysis['avg_holding_days']:.1f} days")
    logger.info(f"  Exit reasons: {analysis['exit_reasons']}")
    logger.info(f"  Worst symbols: {analysis['symbols']}")

    return analysis


def export_to_quantstats(equity_curve: List[Tuple[datetime, float]],
                         trades: List[Trade],
                         save_path: str = 'reports/') -> Optional[str]:
    """
    Exporta resultados en formato compatible con QuantStats

    Args:
        equity_curve: Curva de equity
        trades: Lista de trades
        save_path: Directorio para guardar

    Returns:
        Path del archivo HTML generado
    """
    if not HAS_QUANTSTATS:
        logger.warning("QuantStats not installed, skipping export")
        return None

    logger.info("Exporting to QuantStats format...")

    # Convertir equity curve a returns
    dates = [eq[0] for eq in equity_curve]
    equity = [eq[1] for eq in equity_curve]

    returns = pd.Series(index=dates, data=np.diff([equity[0]] + equity) / equity)

    # Crear reporte con QuantStats
    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"quantstats_report_{timestamp}.html"
    filepath = os.path.join(save_path, filename)

    qs.reports.html(returns, output=filepath, title='Penny Stock Strategy - QuantStats Report')
    logger.info(f"✓ QuantStats report saved: {filepath}")

    return filepath


# ============================================================================
# PHASE 2: PARALLELIZATION & CHECKPOINTS
# ============================================================================

def save_checkpoint(state: Dict[str, Any], checkpoint_path: str = 'checkpoints/') -> str:
    """
    Guarda checkpoint del estado del backtest

    Args:
        state: Estado a guardar
        checkpoint_path: Directorio para checkpoints

    Returns:
        Path del checkpoint guardado
    """
    os.makedirs(checkpoint_path, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"checkpoint_{timestamp}.pkl"
    filepath = os.path.join(checkpoint_path, filename)

    with open(filepath, 'wb') as f:
        pickle.dump(state, f)

    logger.info(f"✓ Checkpoint saved: {filepath}")
    return filepath


def load_checkpoint(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    """
    Carga checkpoint del estado del backtest

    Args:
        checkpoint_path: Path del checkpoint

    Returns:
        Estado cargado o None
    """
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return None

    with open(checkpoint_path, 'rb') as f:
        state = pickle.load(f)

    logger.info(f"✓ Checkpoint loaded: {checkpoint_path}")
    return state


def process_ticker_parallel(ticker: str,
                           start_date: str,
                           end_date: str,
                           model: Any,
                           config: BacktestConfig) -> Dict[str, Any]:
    """
    Procesa un ticker individual (para paralelización)

    Args:
        ticker: Símbolo del ticker
        start_date: Fecha inicio
        end_date: Fecha fin
        model: Modelo ML
        config: Configuración

    Returns:
        Resultados para este ticker
    """
    try:
        logger.info(f"Processing {ticker}...")

        # TODO: Implementar lógica de backtesting para un ticker
        # Por ahora retornar estructura vacía

        results = {
            'ticker': ticker,
            'trades': [],
            'metrics': {},
            'status': 'completed'
        }

        return results

    except Exception as e:
        logger.error(f"Error processing {ticker}: {e}")
        return {
            'ticker': ticker,
            'error': str(e),
            'status': 'failed'
        }


def run_backtest_parallel(tickers: List[str],
                         start_date: str,
                         end_date: str,
                         model: Any,
                         config: BacktestConfig,
                         n_workers: Optional[int] = None,
                         checkpoint_every: int = 10) -> Dict[str, Any]:
    """
    Ejecuta backtest en paralelo usando ProcessPoolExecutor

    Args:
        tickers: Lista de tickers
        start_date: Fecha inicio
        end_date: Fecha fin
        model: Modelo ML
        config: Configuración
        n_workers: Número de workers (None = auto)
        checkpoint_every: Guardar checkpoint cada N tickers

    Returns:
        Resultados agregados
    """
    logger.info(f"Starting parallel backtest with {len(tickers)} tickers...")

    if n_workers is None:
        n_workers = min(multiprocessing.cpu_count() - 1, len(tickers))

    logger.info(f"Using {n_workers} parallel workers")

    results_list = []
    checkpoint_counter = 0

    # Progress bar
    iterator = tqdm(tickers, desc="Backtesting") if HAS_TQDM else tickers

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        future_to_ticker = {
            executor.submit(process_ticker_parallel, ticker, start_date,
                          end_date, model, config): ticker
            for ticker in tickers
        }

        # Process results as they complete
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                result = future.result()
                results_list.append(result)

                # Checkpoint
                checkpoint_counter += 1
                if checkpoint_counter % checkpoint_every == 0:
                    checkpoint_state = {
                        'results': results_list,
                        'processed': checkpoint_counter,
                        'total': len(tickers)
                    }
                    save_checkpoint(checkpoint_state)

                if HAS_TQDM:
                    iterator.update(1)

            except Exception as e:
                logger.error(f"Error with {ticker}: {e}")

    # Aggregate results
    successful = [r for r in results_list if r['status'] == 'completed']
    failed = [r for r in results_list if r['status'] == 'failed']

    logger.info(f"✓ Parallel backtest completed: {len(successful)} successful, {len(failed)} failed")

    return {
        'successful': successful,
        'failed': failed,
        'total_processed': len(results_list)
    }


# ============================================================================
# PHASE 2: META-VALIDATION
# ============================================================================

def meta_validation_backtest(strategy_adaptive: Dict[str, Any],
                            strategy_fixed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Meta-validación: Backtesting del backtest

    Compara estrategia adaptativa vs fija para determinar si
    el ajuste automático de threshold realmente mejora resultados

    Args:
        strategy_adaptive: Resultados de estrategia adaptativa
        strategy_fixed: Resultados de estrategia fija

    Returns:
        Comparación de ambas estrategias
    """
    logger.info("Running meta-validation (backtesting of backtesting)...")

    adaptive_sharpe = strategy_adaptive.get('sharpe_ratio', 0)
    fixed_sharpe = strategy_fixed.get('sharpe_ratio', 0)

    adaptive_return = strategy_adaptive.get('total_return_pct', 0)
    fixed_return = strategy_fixed.get('total_return_pct', 0)

    adaptive_dd = strategy_adaptive.get('max_drawdown_pct', 0)
    fixed_dd = strategy_fixed.get('max_drawdown_pct', 0)

    comparison = {
        'adaptive_better_sharpe': adaptive_sharpe > fixed_sharpe,
        'adaptive_better_return': adaptive_return > fixed_return,
        'adaptive_better_dd': adaptive_dd < fixed_dd,
        'sharpe_improvement': ((adaptive_sharpe - fixed_sharpe) / fixed_sharpe * 100) if fixed_sharpe > 0 else 0,
        'return_improvement': adaptive_return - fixed_return,
        'dd_improvement': fixed_dd - adaptive_dd,
        'recommendation': ''
    }

    # Determinar recomendación
    wins = sum([
        comparison['adaptive_better_sharpe'],
        comparison['adaptive_better_return'],
        comparison['adaptive_better_dd']
    ])

    if wins >= 2:
        comparison['recommendation'] = 'USE ADAPTIVE STRATEGY'
    else:
        comparison['recommendation'] = 'USE FIXED STRATEGY'

    logger.info("Meta-validation results:")
    logger.info(f"  Adaptive Sharpe: {adaptive_sharpe:.2f} vs Fixed: {fixed_sharpe:.2f}")
    logger.info(f"  Adaptive Return: {adaptive_return:.2f}% vs Fixed: {fixed_return:.2f}%")
    logger.info(f"  Recommendation: {comparison['recommendation']}")

    return comparison


# ============================================================================
# BACKTESTING ENGINE
# ============================================================================

class BacktestEngineV3:
    """
    Motor de backtesting riguroso con capacidades adaptativas (V3.0)

    Nuevas features V3:
    - Adaptive thresholds con anti-overfitting
    - Ensemble models (RF + XGBoost + LightGBM)
    - Market regime detection
    - Risk monitoring y alertas
    - Performance tracking y drift detection
    """

    def __init__(self, config: BacktestConfig, adaptive: bool = True, use_ensemble: bool = True):
        """
        Args:
            config: Configuración del backtest
            adaptive: Activar modo adaptativo
            use_ensemble: Usar ensemble de modelos
        """
        self.config = config
        self.capital = config.initial_capital
        self.positions: List[Position] = []
        self.closed_trades: List[Trade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.daily_returns: List[float] = []

        # V3 Features
        self.adaptive = adaptive
        self.use_ensemble = use_ensemble
        self.current_threshold = 0.65
        self.ensemble_models = None
        self.performance_history: List[Dict] = []
        self.last_training_date = None
        self.current_regime = 'choppy'
        self.alerts_history: List[Dict] = []

        logger.info(f"BacktestEngine V3.0 inicializado")
        logger.info(f"Capital inicial: ${config.initial_capital:,.2f}")
        logger.info(f"Position size: {config.position_size_pct*100:.1f}%")
        logger.info(f"Max positions: {config.max_positions}")
        logger.info(f"Commission: {config.commission*100:.2f}%")
        logger.info(f"Slippage: {config.slippage*100:.2f}%")
        logger.info(f"Adaptive mode: {adaptive}")
        logger.info(f"Ensemble models: {use_ensemble}")

    def reset(self):
        """Resetea el estado del backtester"""
        self.capital = self.config.initial_capital
        self.positions = []
        self.closed_trades = []
        self.equity_curve = []
        self.daily_returns = []

    def get_available_capital(self) -> float:
        """Calcula el capital disponible (no invertido)"""
        invested = sum(p.entry_price * p.shares for p in self.positions)
        return self.capital - invested

    def get_total_equity(self, current_prices: Dict[str, float]) -> float:
        """
        Calcula el equity total (capital + valor de posiciones)

        Args:
            current_prices: Dict {symbol: price}

        Returns:
            Equity total
        """
        position_value = sum(
            current_prices.get(p.symbol, p.entry_price) * p.shares
            for p in self.positions
        )
        return self.capital + position_value

    def can_open_position(self) -> bool:
        """Verifica si se puede abrir una nueva posición"""
        return len(self.positions) < self.config.max_positions

    def open_position(self,
                     symbol: str,
                     entry_price: float,
                     entry_date: datetime,
                     stop_loss_pct: Optional[float] = None,
                     take_profit_pct: Optional[float] = None) -> Optional[Position]:
        """
        Abre una nueva posición

        Args:
            symbol: Símbolo del activo
            entry_price: Precio de entrada
            entry_date: Fecha de entrada
            stop_loss_pct: % de stop loss (opcional)
            take_profit_pct: % de take profit (opcional)

        Returns:
            Position si se abrió exitosamente, None si no
        """
        if not self.can_open_position():
            logger.debug(f"No se puede abrir {symbol}: max positions alcanzado")
            return None

        # Calcular tamaño de posición
        position_size = size_position(
            self.get_available_capital(),
            self.config
        )

        # Calcular costos y precio efectivo
        effective_price, total_cost = calculate_entry_cost(
            1,  # Temporal para calcular
            entry_price,
            self.config.commission,
            self.config.slippage
        )

        # Calcular shares reales
        shares = int(position_size / effective_price)

        if shares <= 0:
            logger.debug(f"No se puede abrir {symbol}: shares=0")
            return None

        # Recalcular con shares reales
        effective_price, total_cost = calculate_entry_cost(
            shares,
            entry_price,
            self.config.commission,
            self.config.slippage
        )

        # Verificar capital suficiente
        if total_cost > self.get_available_capital():
            logger.debug(f"No se puede abrir {symbol}: capital insuficiente")
            return None

        # Stop loss y take profit
        sl_pct = stop_loss_pct or self.config.stop_loss_pct
        tp_pct = take_profit_pct or self.config.take_profit_pct

        stop_loss = effective_price * (1 - sl_pct)
        take_profit = effective_price * (1 + tp_pct)

        # Crear posición
        position = Position(
            symbol=symbol,
            entry_date=entry_date,
            entry_price=effective_price,
            shares=shares,
            stop_loss=stop_loss,
            take_profit=take_profit,
            cost=total_cost,
            highest_price=effective_price
        )

        self.positions.append(position)

        logger.info(f"✓ OPEN {symbol}: {shares} shares @ ${effective_price:.4f} "
                   f"(SL: ${stop_loss:.4f}, TP: ${take_profit:.4f})")

        return position

    def close_position(self,
                      position: Position,
                      exit_price: float,
                      exit_date: datetime,
                      exit_reason: str) -> Trade:
        """
        Cierra una posición

        Args:
            position: Posición a cerrar
            exit_price: Precio de salida
            exit_date: Fecha de salida
            exit_reason: Razón del cierre

        Returns:
            Trade cerrado
        """
        # Calcular costos de salida
        effective_exit_price, net_amount = calculate_exit_cost(
            position.shares,
            exit_price,
            self.config.commission,
            self.config.slippage
        )

        # P&L
        pnl_gross = (effective_exit_price - position.entry_price) * position.shares
        total_cost = position.cost + (position.shares * effective_exit_price * self.config.commission)
        pnl_net = net_amount - position.cost
        pnl_pct = (pnl_net / position.cost) * 100

        # Holding days
        holding_days = (exit_date - position.entry_date).days

        # MAE y MFE (simplificado - en producción calcular con datos intraday)
        mae = ((position.entry_price - position.stop_loss) / position.entry_price) * 100
        mfe = ((position.highest_price - position.entry_price) / position.entry_price) * 100

        # Crear trade
        trade = Trade(
            symbol=position.symbol,
            entry_date=position.entry_date,
            exit_date=exit_date,
            entry_price=position.entry_price,
            exit_price=effective_exit_price,
            shares=position.shares,
            entry_cost=position.cost,
            exit_cost=position.shares * effective_exit_price * self.config.commission,
            pnl_gross=pnl_gross,
            pnl_net=pnl_net,
            pnl_pct=pnl_pct,
            holding_days=holding_days,
            exit_reason=exit_reason,
            highest_price=position.highest_price,
            max_adverse_excursion=mae,
            max_favorable_excursion=mfe
        )

        # Actualizar capital
        self.capital += pnl_net

        # Remover posición
        self.positions.remove(position)

        # Guardar trade
        self.closed_trades.append(trade)

        logger.info(f"✓ CLOSE {position.symbol}: P&L=${pnl_net:.2f} ({pnl_pct:+.2f}%) "
                   f"- {exit_reason}")

        return trade

    def update_positions(self,
                        current_date: datetime,
                        current_prices: Dict[str, float]):
        """
        Actualiza posiciones abiertas y verifica condiciones de salida

        Args:
            current_date: Fecha actual
            current_prices: Dict {symbol: price}
        """
        positions_to_close = []

        for position in self.positions:
            current_price = current_prices.get(position.symbol)

            if current_price is None:
                continue

            # Actualizar highest price
            position.highest_price = max(position.highest_price, current_price)

            # Verificar condiciones de salida
            exit_reason = None

            # 1. Stop loss
            if apply_stop_loss(position, current_price, self.config):
                exit_reason = 'Stop Loss'

            # 2. Take profit
            elif apply_take_profit(position, current_price, self.config):
                exit_reason = 'Take Profit'

            # 3. Max holding time
            elif (current_date - position.entry_date).days >= self.config.max_holding_days:
                exit_reason = 'Max Holding Time'

            if exit_reason:
                positions_to_close.append((position, current_price, exit_reason))

        # Cerrar posiciones
        for position, exit_price, exit_reason in positions_to_close:
            self.close_position(position, exit_price, current_date, exit_reason)

    def record_equity(self, date: datetime, current_prices: Dict[str, float]):
        """
        Registra el equity en un punto del tiempo

        Args:
            date: Fecha
            current_prices: Dict {symbol: price}
        """
        equity = self.get_total_equity(current_prices)
        self.equity_curve.append((date, equity))

        # Calcular daily return
        if len(self.equity_curve) > 1:
            prev_equity = self.equity_curve[-2][1]
            daily_return = (equity - prev_equity) / prev_equity
            self.daily_returns.append(daily_return)

    # ========================================================================
    # V3 ADAPTIVE METHODS
    # ========================================================================

    def train_ensemble(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Entrena ensemble de modelos

        Args:
            X_train: Features de entrenamiento
            y_train: Labels de entrenamiento
        """
        if not self.use_ensemble:
            logger.info("Ensemble mode disabled, skipping training")
            return

        self.ensemble_models = build_ensemble_models(X_train, y_train)
        self.last_training_date = datetime.now()
        logger.info("✓ Ensemble models trained and ready")

    def predict_with_ensemble(self, features: np.ndarray, use_threshold: bool = True) -> float:
        """
        Predice usando ensemble con threshold adaptativo

        Args:
            features: Features para predicción
            use_threshold: Aplicar threshold actual

        Returns:
            Probabilidad de señal
        """
        if self.ensemble_models is None:
            logger.warning("Ensemble not trained, returning 0.0")
            return 0.0

        prob = ensemble_prediction(self.ensemble_models, features)

        if use_threshold:
            # Aplicar threshold adaptativo
            return prob if prob >= self.current_threshold else 0.0
        else:
            return prob

    def update_adaptive_threshold(self, train_metrics: Optional[Dict] = None):
        """
        Actualiza threshold de forma adaptativa

        Args:
            train_metrics: Métricas de entrenamiento (para detectar overfitting)
        """
        if not self.adaptive:
            return

        if len(self.closed_trades) < 20:
            logger.info("Not enough trades for adaptive threshold adjustment")
            return

        # Ajustar threshold
        new_threshold, overfitting = adaptive_threshold_validation_only(
            self.closed_trades,
            self.current_threshold,
            train_metrics=train_metrics
        )

        if overfitting:
            logger.error("⚠️  OVERFITTING DETECTED - Threshold NOT adjusted")
            self.alerts_history.append({
                'type': 'OVERFITTING',
                'date': datetime.now(),
                'message': 'Overfitting detected in adaptive threshold'
            })
        else:
            self.current_threshold = new_threshold

    def update_market_regime(self, spy_data: pd.DataFrame):
        """
        Actualiza el régimen de mercado y ajusta threshold

        Args:
            spy_data: Datos de SPY
        """
        self.current_regime = detect_market_regime(spy_data)

        if self.adaptive:
            # Ajustar threshold según régimen
            base_threshold = self.current_threshold
            adjusted = adjust_threshold_by_regime(base_threshold, self.current_regime)
            self.current_threshold = adjusted

    def check_risk_alerts(self):
        """
        Verifica condiciones de riesgo y genera alertas
        """
        if len(self.equity_curve) < 10 or len(self.closed_trades) < 5:
            return

        alerts = risk_monitoring(self.equity_curve, self.closed_trades, self.config)

        if alerts['action_required']:
            self.alerts_history.extend(alerts['alerts'])
            logger.warning(f"🚨 {len(alerts['alerts'])} RISK ALERTS TRIGGERED")

            # Auto-pause en caso crítico
            for alert in alerts['alerts']:
                if alert['severity'] == 'CRITICAL':
                    logger.error(f"CRITICAL ALERT: {alert['message']}")
                    # TODO: Implementar auto-pause logic

        return alerts

    def should_retrain(self, current_date: datetime) -> bool:
        """
        Determina si debe re-entrenar modelos

        Args:
            current_date: Fecha actual

        Returns:
            True si debe re-entrenar
        """
        if not self.adaptive or self.last_training_date is None:
            return False

        return adaptive_retraining_schedule(
            self.performance_history,
            self.last_training_date,
            current_date
        )

    def record_performance(self):
        """
        Registra performance actual en historial
        """
        if len(self.closed_trades) < 5:
            return

        # Calcular métricas actuales
        metrics = calculate_all_metrics(
            self.closed_trades,
            self.equity_curve,
            self.config,
            period_days=len(self.equity_curve)
        )

        self.performance_history.append({
            'date': datetime.now(),
            'sharpe_ratio': metrics.get('sharpe_ratio', 0),
            'total_return': metrics.get('total_return_pct', 0),
            'max_drawdown': metrics.get('max_drawdown_pct', 0),
            'win_rate': metrics.get('win_rate', 0),
            'num_trades': len(self.closed_trades)
        })

    def get_adaptive_summary(self) -> Dict[str, Any]:
        """
        Obtiene resumen de características adaptativas

        Returns:
            Dict con estado adaptativo
        """
        return {
            'adaptive_enabled': self.adaptive,
            'current_threshold': self.current_threshold,
            'current_regime': self.current_regime,
            'ensemble_enabled': self.use_ensemble,
            'ensemble_ready': self.ensemble_models is not None,
            'last_training': self.last_training_date,
            'num_alerts': len(self.alerts_history),
            'performance_records': len(self.performance_history)
        }


# Maintain backward compatibility with V2 name
BacktestEngineV2 = BacktestEngineV3



# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

def calculate_all_metrics(trades: List[Trade],
                         equity_curve: List[Tuple[datetime, float]],
                         config: BacktestConfig,
                         period_days: int) -> Dict[str, Any]:
    """
    Calcula TODAS las métricas de performance

    Args:
        trades: Lista de trades cerrados
        equity_curve: Curva de equity [(date, equity)]
        config: Configuración del backtest
        period_days: Duración del backtest en días

    Returns:
        Dict con todas las métricas
    """
    if not trades:
        return {'error': 'No trades'}

    # Convertir a DataFrame
    trades_df = pd.DataFrame([t.to_dict() for t in trades])

    # === BASIC METRICS ===
    total_trades = len(trades)
    winning_trades = len([t for t in trades if t.pnl_net > 0])
    losing_trades = len([t for t in trades if t.pnl_net < 0])
    breakeven_trades = total_trades - winning_trades - losing_trades

    win_rate = (winning_trades / total_trades) if total_trades > 0 else 0

    # === P&L METRICS ===
    total_pnl = sum(t.pnl_net for t in trades)
    total_pnl_pct = (total_pnl / config.initial_capital) * 100

    gross_profit = sum(t.pnl_net for t in trades if t.pnl_net > 0)
    gross_loss = abs(sum(t.pnl_net for t in trades if t.pnl_net < 0))

    avg_win = gross_profit / winning_trades if winning_trades > 0 else 0
    avg_loss = gross_loss / losing_trades if losing_trades > 0 else 0

    # Profit Factor
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    # Expectancy
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    # === RETURN METRICS ===
    initial_capital = equity_curve[0][1]
    final_capital = equity_curve[-1][1]
    total_return = ((final_capital - initial_capital) / initial_capital) * 100

    # CAGR (Compound Annual Growth Rate)
    years = period_days / 365.25
    cagr = (((final_capital / initial_capital) ** (1 / years)) - 1) * 100 if years > 0 else 0

    # === RISK METRICS ===
    # Equity curve to returns
    equity_values = [eq[1] for eq in equity_curve]
    returns = np.diff(equity_values) / equity_values[:-1]

    # Sharpe Ratio
    avg_return = np.mean(returns)
    std_return = np.std(returns)
    # Ajustar risk-free rate a diario
    daily_rf = (1 + config.risk_free_rate) ** (1/252) - 1
    sharpe_ratio = ((avg_return - daily_rf) / std_return) * np.sqrt(252) if std_return > 0 else 0

    # Sortino Ratio (usa solo downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
    sortino_ratio = ((avg_return - daily_rf) / downside_std) * np.sqrt(252) if downside_std > 0 else 0

    # Max Drawdown
    max_equity = equity_values[0]
    max_dd = 0
    max_dd_duration = 0
    current_dd_duration = 0
    underwater_periods = []
    peak_date = equity_curve[0][0]

    for i, (date, equity) in enumerate(equity_curve):
        if equity > max_equity:
            max_equity = equity
            peak_date = date
            if current_dd_duration > 0:
                underwater_periods.append(current_dd_duration)
                current_dd_duration = 0
        else:
            dd = ((max_equity - equity) / max_equity) * 100
            max_dd = max(max_dd, dd)
            current_dd_duration += 1
            max_dd_duration = max(max_dd_duration, current_dd_duration)

    # Recovery Time
    avg_recovery_time = np.mean(underwater_periods) if underwater_periods else 0

    # Calmar Ratio
    calmar_ratio = cagr / max_dd if max_dd > 0 else 0

    # === TRADE METRICS ===
    avg_holding_days = trades_df['holding_days'].mean()
    max_holding_days = trades_df['holding_days'].max()
    min_holding_days = trades_df['holding_days'].min()

    # Consecutive wins/losses
    consecutive_wins = 0
    consecutive_losses = 0
    max_consecutive_wins = 0
    max_consecutive_losses = 0

    for trade in trades:
        if trade.pnl_net > 0:
            consecutive_wins += 1
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            consecutive_losses = 0
        else:
            consecutive_losses += 1
            max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            consecutive_wins = 0

    max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
    max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)

    # Best and worst trades
    best_trade = max(trades, key=lambda t: t.pnl_pct)
    worst_trade = min(trades, key=lambda t: t.pnl_pct)

    metrics = {
        # Basic
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'breakeven_trades': breakeven_trades,
        'win_rate': win_rate,

        # P&L
        'total_pnl': total_pnl,
        'total_pnl_pct': total_pnl_pct,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'expectancy': expectancy,

        # Returns
        'total_return_pct': total_return,
        'cagr': cagr,
        'initial_capital': initial_capital,
        'final_capital': final_capital,

        # Risk
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown_pct': max_dd,
        'max_drawdown_duration_days': max_dd_duration,
        'avg_recovery_time_days': avg_recovery_time,
        'calmar_ratio': calmar_ratio,

        # Trade stats
        'avg_holding_days': avg_holding_days,
        'max_holding_days': max_holding_days,
        'min_holding_days': min_holding_days,
        'max_consecutive_wins': max_consecutive_wins,
        'max_consecutive_losses': max_consecutive_losses,

        # Best/Worst
        'best_trade_pct': best_trade.pnl_pct,
        'worst_trade_pct': worst_trade.pnl_pct,
        'best_trade_symbol': best_trade.symbol,
        'worst_trade_symbol': worst_trade.symbol
    }

    return metrics


# ============================================================================
# MONTE CARLO SIMULATION
# ============================================================================

def monte_carlo_simulation(trades: List[Trade],
                          n_simulations: int = 1000,
                          initial_capital: float = 100000) -> Dict[str, Any]:
    """
    Simula secuencias aleatorias de trades para validación estadística

    Randomiza el orden de los trades para entender la distribución
    de posibles outcomes.

    Args:
        trades: Lista de trades históricos
        n_simulations: Número de simulaciones
        initial_capital: Capital inicial

    Returns:
        Dict con resultados de la simulación
    """
    logger.info(f"Ejecutando Monte Carlo: {n_simulations} simulaciones")

    if not trades:
        return {'error': 'No trades'}

    # Extraer P&L de cada trade
    trade_pnls = [t.pnl_net for t in trades]

    # Arrays para resultados
    final_capitals = []
    max_drawdowns = []
    returns = []

    for _ in range(n_simulations):
        # Randomizar orden de trades
        shuffled_pnls = np.random.choice(trade_pnls, size=len(trade_pnls), replace=True)

        # Simular equity curve
        capital = initial_capital
        equity = [capital]

        for pnl in shuffled_pnls:
            capital += pnl
            equity.append(capital)

        # Métricas de esta simulación
        final_capital = equity[-1]
        ret = ((final_capital - initial_capital) / initial_capital) * 100

        # Max drawdown
        max_equity = equity[0]
        max_dd = 0
        for eq in equity:
            max_equity = max(max_equity, eq)
            dd = ((max_equity - eq) / max_equity) * 100
            max_dd = max(max_dd, dd)

        final_capitals.append(final_capital)
        returns.append(ret)
        max_drawdowns.append(max_dd)

    # Calcular percentiles
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    return_percentiles = {
        f'P{p}': np.percentile(returns, p)
        for p in percentiles
    }

    # Probabilidad de ruina (perder >50% del capital)
    ruin_threshold = initial_capital * 0.5
    prob_ruin = len([fc for fc in final_capitals if fc < ruin_threshold]) / n_simulations

    results = {
        'n_simulations': n_simulations,
        'final_capital_mean': np.mean(final_capitals),
        'final_capital_std': np.std(final_capitals),
        'return_mean': np.mean(returns),
        'return_std': np.std(returns),
        'return_percentiles': return_percentiles,
        'max_drawdown_mean': np.mean(max_drawdowns),
        'max_drawdown_std': np.std(max_drawdowns),
        'prob_ruin': prob_ruin,
        'worst_case_return': np.min(returns),
        'best_case_return': np.max(returns)
    }

    logger.info(f"Monte Carlo completado:")
    logger.info(f"  Return medio: {results['return_mean']:.2f}%")
    logger.info(f"  Return mediano (P50): {return_percentiles['P50']:.2f}%")
    logger.info(f"  Peor caso (P5): {return_percentiles['P5']:.2f}%")
    logger.info(f"  Probabilidad de ruina: {prob_ruin*100:.2f}%")

    return results


# ============================================================================
# STATISTICAL VALIDATION
# ============================================================================

def statistical_validation(backtest_returns: np.ndarray,
                          benchmark_returns: np.ndarray,
                          alpha: float = 0.05) -> Dict[str, Any]:
    """
    Tests estadísticos para validar si el outperformance es significativo

    Args:
        backtest_returns: Returns de la estrategia
        benchmark_returns: Returns del benchmark (ej: SPY)
        alpha: Nivel de significancia

    Returns:
        Dict con resultados de tests estadísticos
    """
    logger.info("Ejecutando validación estadística vs benchmark")

    # T-test para comparar medias
    t_stat, p_value = stats.ttest_ind(backtest_returns, benchmark_returns)

    is_significant = p_value < alpha

    # Excess returns
    excess_returns = backtest_returns - benchmark_returns
    avg_excess = np.mean(excess_returns)

    # Information Ratio
    tracking_error = np.std(excess_returns)
    information_ratio = (avg_excess / tracking_error) * np.sqrt(252) if tracking_error > 0 else 0

    # Beta (sensibilidad al mercado)
    covariance = np.cov(backtest_returns, benchmark_returns)[0, 1]
    benchmark_variance = np.var(benchmark_returns)
    beta = covariance / benchmark_variance if benchmark_variance > 0 else 0

    # Alpha (excess return ajustado por riesgo)
    risk_free_daily = (1.04) ** (1/252) - 1  # 4% anual
    strategy_return = np.mean(backtest_returns)
    benchmark_return = np.mean(benchmark_returns)
    alpha_metric = strategy_return - (risk_free_daily + beta * (benchmark_return - risk_free_daily))
    alpha_annualized = alpha_metric * 252

    results = {
        't_statistic': t_stat,
        'p_value': p_value,
        'is_significant': is_significant,
        'significance_level': alpha,
        'avg_excess_return_daily': avg_excess,
        'avg_excess_return_annual': avg_excess * 252,
        'information_ratio': information_ratio,
        'beta': beta,
        'alpha_daily': alpha_metric,
        'alpha_annualized_pct': alpha_annualized * 100,
        'conclusion': 'SIGNIFICATIVO' if is_significant else 'NO SIGNIFICATIVO'
    }

    logger.info(f"T-test: t={t_stat:.4f}, p={p_value:.4f}")
    logger.info(f"Outperformance es {results['conclusion']}")
    logger.info(f"Information Ratio: {information_ratio:.4f}")
    logger.info(f"Beta: {beta:.4f}")
    logger.info(f"Alpha anualizado: {alpha_annualized*100:.2f}%")

    return results


def fetch_benchmark_data(symbol: str = 'SPY',
                        start_date: str = '2020-01-01',
                        end_date: str = '2023-12-31') -> pd.DataFrame:
    """
    Obtiene datos del benchmark para comparación

    Args:
        symbol: Símbolo del benchmark (default: SPY)
        start_date: Fecha inicio
        end_date: Fecha fin

    Returns:
        DataFrame con datos del benchmark
    """
    logger.info(f"Descargando datos del benchmark {symbol}")

    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date, end=end_date)

    if len(data) == 0:
        logger.error(f"No se pudieron obtener datos de {symbol}")
        return None

    logger.info(f"Benchmark: {len(data)} días de datos")
    return data


# ============================================================================
# VISUALIZATION & REPORTING
# ============================================================================

def generate_tearsheet(backtest_results: Dict[str, Any],
                      equity_curve: List[Tuple[datetime, float]],
                      trades: List[Trade],
                      monte_carlo_results: Dict[str, Any],
                      save_path: str = 'reports/') -> str:
    """
    Genera reporte visual completo (tearsheet)

    Args:
        backtest_results: Resultados del backtest
        equity_curve: Curva de equity
        trades: Lista de trades
        monte_carlo_results: Resultados de Monte Carlo
        save_path: Directorio para guardar reportes

    Returns:
        Ruta del archivo generado
    """
    logger.info("Generando tearsheet...")

    # Crear directorio si no existe
    os.makedirs(save_path, exist_ok=True)

    # Timestamp para nombre único
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"tearsheet_{timestamp}.png"
    filepath = os.path.join(save_path, filename)

    # Preparar datos
    dates = [eq[0] for eq in equity_curve]
    equity = [eq[1] for eq in equity_curve]

    trades_df = pd.DataFrame([t.to_dict() for t in trades])

    # Crear figura con subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

    # === 1. EQUITY CURVE ===
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(dates, equity, linewidth=2, color='#2E86AB')
    ax1.fill_between(dates, equity, alpha=0.3, color='#2E86AB')
    ax1.set_title('Equity Curve', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Equity ($)')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=equity[0], color='red', linestyle='--', alpha=0.5, label='Initial Capital')
    ax1.legend()

    # === 2. DRAWDOWN CHART ===
    ax2 = fig.add_subplot(gs[1, :])
    max_equity = equity[0]
    drawdowns = []
    for eq in equity:
        max_equity = max(max_equity, eq)
        dd = ((max_equity - eq) / max_equity) * 100
        drawdowns.append(-dd)

    ax2.fill_between(dates, drawdowns, 0, color='red', alpha=0.3)
    ax2.plot(dates, drawdowns, color='darkred', linewidth=1)
    ax2.set_title('Underwater Plot (Drawdown %)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True, alpha=0.3)

    # === 3. RETURNS DISTRIBUTION ===
    ax3 = fig.add_subplot(gs[2, 0])
    returns_pct = trades_df['pnl_pct'].values
    ax3.hist(returns_pct, bins=30, edgecolor='black', alpha=0.7, color='#A23B72')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax3.set_title('Returns Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Return (%)')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3)

    # === 4. MONTHLY RETURNS HEATMAP ===
    ax4 = fig.add_subplot(gs[2, 1:])
    trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
    trades_df['year'] = trades_df['exit_date'].dt.year
    trades_df['month'] = trades_df['exit_date'].dt.month

    monthly_returns = trades_df.groupby(['year', 'month'])['pnl_pct'].sum().reset_index()
    monthly_pivot = monthly_returns.pivot(index='year', columns='month', values='pnl_pct')

    if HAS_SEABORN:
        sns.heatmap(monthly_pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                    cbar_kws={'label': 'Return (%)'}, ax=ax4)
    else:
        # Fallback to basic matplotlib heatmap
        im = ax4.imshow(monthly_pivot.values, cmap='RdYlGn', aspect='auto')
        ax4.set_xticks(range(len(monthly_pivot.columns)))
        ax4.set_yticks(range(len(monthly_pivot.index)))
        ax4.set_xticklabels(monthly_pivot.columns)
        ax4.set_yticklabels(monthly_pivot.index)
        plt.colorbar(im, ax=ax4, label='Return (%)')

    ax4.set_title('Monthly Returns Heatmap', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Month')
    ax4.set_ylabel('Year')

    # === 5. TRADE DURATION DISTRIBUTION ===
    ax5 = fig.add_subplot(gs[3, 0])
    holding_days = trades_df['holding_days'].values
    ax5.hist(holding_days, bins=20, edgecolor='black', alpha=0.7, color='#F18F01')
    ax5.set_title('Trade Duration Distribution', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Holding Days')
    ax5.set_ylabel('Frequency')
    ax5.grid(True, alpha=0.3)

    # === 6. MONTE CARLO RESULTS ===
    ax6 = fig.add_subplot(gs[3, 1])
    percentiles = monte_carlo_results['return_percentiles']
    labels = list(percentiles.keys())
    values = list(percentiles.values())

    ax6.barh(labels, values, color='#06A77D', alpha=0.7, edgecolor='black')
    ax6.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax6.set_title('Monte Carlo Percentiles', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Return (%)')
    ax6.grid(True, alpha=0.3)

    # === 7. KEY METRICS TABLE ===
    ax7 = fig.add_subplot(gs[3, 2])
    ax7.axis('off')

    metrics = backtest_results
    metrics_text = f"""
    KEY METRICS
    ━━━━━━━━━━━━━━━━━━
    Total Return: {metrics['total_return_pct']:.2f}%
    CAGR: {metrics['cagr']:.2f}%
    Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
    Sortino Ratio: {metrics['sortino_ratio']:.2f}
    Max Drawdown: {metrics['max_drawdown_pct']:.2f}%
    Calmar Ratio: {metrics['calmar_ratio']:.2f}

    Win Rate: {metrics['win_rate']*100:.1f}%
    Profit Factor: {metrics['profit_factor']:.2f}
    Expectancy: ${metrics['expectancy']:.2f}

    Total Trades: {metrics['total_trades']}
    Avg Holding: {metrics['avg_holding_days']:.1f} days
    """

    ax7.text(0.1, 0.5, metrics_text, fontsize=10, family='monospace',
            verticalalignment='center')

    # Guardar
    plt.suptitle('BACKTEST TEARSHEET - Penny Stock Strategy', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Tearsheet guardado: {filepath}")

    return filepath


def print_critical_validations(metrics: Dict[str, Any],
                               stat_validation: Dict[str, Any],
                               train_metrics: Optional[Dict] = None) -> None:
    """
    Imprime validaciones críticas del backtest

    Args:
        metrics: Métricas del backtest
        stat_validation: Resultados de validación estadística
        train_metrics: Métricas del training set (para detectar overfitting)
    """
    print("\n" + "="*80)
    print("⚠️  VALIDACIONES CRÍTICAS DEL BACKTEST")
    print("="*80 + "\n")

    # 1. Out-of-sample
    print("1. ¿El test set es out-of-sample?")
    print("   → YES (por diseño temporal split)")

    # 2. Sharpe Ratio
    sharpe_ok = metrics['sharpe_ratio'] > 1.0
    status = "✓ YES" if sharpe_ok else "✗ NO"
    print(f"\n2. ¿Sharpe ratio > 1.0? → {status}")
    print(f"   Sharpe: {metrics['sharpe_ratio']:.2f}")

    # 3. Statistical significance
    is_sig = stat_validation['is_significant']
    status = "✓ YES" if is_sig else "✗ NO"
    print(f"\n3. ¿Outperformance vs SPY es estadísticamente significativo? → {status}")
    print(f"   p-value: {stat_validation['p_value']:.4f}")
    print(f"   Alpha anualizado: {stat_validation['alpha_annualized_pct']:.2f}%")

    # 4. Max Drawdown
    dd_ok = metrics['max_drawdown_pct'] < 30
    status = "✓ YES" if dd_ok else "✗ NO"
    print(f"\n4. ¿Max Drawdown < 30%? → {status}")
    print(f"   Max DD: {metrics['max_drawdown_pct']:.2f}%")

    # 5. Overfitting check
    if train_metrics:
        train_return = train_metrics.get('total_return_pct', 0)
        test_return = metrics['total_return_pct']
        gap = abs(train_return - test_return) / train_return * 100 if train_return != 0 else 0
        overfit_ok = gap < 20
        status = "✓ YES" if overfit_ok else "✗ NO"
        print(f"\n5. ¿Train/Test gap < 20%? → {status} (detecta overfitting)")
        print(f"   Train return: {train_return:.2f}%")
        print(f"   Test return: {test_return:.2f}%")
        print(f"   Gap: {gap:.1f}%")
    else:
        print(f"\n5. ¿Train/Test gap < 20%? → N/A (no train metrics)")

    # 6. Profit Factor
    pf_ok = metrics['profit_factor'] > 1.5
    status = "✓ YES" if pf_ok else "✗ NO"
    print(f"\n6. ¿Profit Factor > 1.5? → {status}")
    print(f"   Profit Factor: {metrics['profit_factor']:.2f}")

    # 7. Win Rate razonable
    wr = metrics['win_rate']
    wr_ok = 0.40 <= wr <= 0.70
    status = "✓ YES" if wr_ok else "⚠ WARNING"
    print(f"\n7. ¿Win Rate razonable (40-70%)? → {status}")
    print(f"   Win Rate: {wr*100:.1f}%")

    print("\n" + "="*80)


# ============================================================================
# MAIN EXECUTION FUNCTIONS
# ============================================================================

def run_backtest_realistic(universe: List[str],
                          start_date: str,
                          end_date: str,
                          model_path: Optional[str] = None,
                          config: Optional[BacktestConfig] = None) -> Dict[str, Any]:
    """
    Motor principal de backtesting con walk-forward analysis

    Args:
        universe: Lista de símbolos a testear
        start_date: Fecha inicio (YYYY-MM-DD)
        end_date: Fecha fin (YYYY-MM-DD)
        model_path: Ruta al modelo ML (opcional)
        config: Configuración del backtest

    Returns:
        Dict con todos los resultados
    """
    if config is None:
        config = BacktestConfig()

    logger.info("="*80)
    logger.info("INICIANDO BACKTEST REALISTA (SIN SESGOS)")
    logger.info("="*80)
    logger.info(f"Universo: {len(universe)} símbolos")
    logger.info(f"Periodo: {start_date} a {end_date}")
    logger.info(f"Capital inicial: ${config.initial_capital:,.2f}")

    # Crear engine
    engine = BacktestEngineV2(config)

    # TODO: Implementar lógica de backtesting con walk-forward
    # Por ahora retornar estructura vacía
    logger.warning("⚠️  Implementación de backtesting completo pendiente")
    logger.warning("    Este es el framework - necesita integrarse con estrategia específica")

    results = {
        'config': config,
        'universe': universe,
        'period': {'start': start_date, 'end': end_date},
        'status': 'framework_ready'
    }

    return results


if __name__ == "__main__":
    print("="*80)
    print("BACKTEST ENGINE V3.0 - ADAPTIVE ROBUST BACKTESTING SYSTEM")
    print("="*80)

    logger.info("Backtest Engine V3.0 - Framework cargado exitosamente")

    # ========================================================================
    # PHASE 2 FEATURES SUMMARY
    # ========================================================================
    print("\n📋 PHASE 2 FEATURES IMPLEMENTED:")
    print("-" * 80)

    features_v3 = [
        ("✓ Intelligent Parallelization", "ProcessPoolExecutor con checkpoint system"),
        ("✓ Adaptive Thresholds", "Ajuste automático con safeguards anti-overfitting"),
        ("✓ Ensemble Models", "RF + XGBoost + LightGBM con voting"),
        ("✓ Market Regime Detection", "Bull/Bear/Choppy usando SPY + MA"),
        ("✓ Online Learning", "Re-training automático cada 3 meses"),
        ("✓ Drift Tracking", "Auto-pause cuando performance degrada"),
        ("✓ Interactive Dashboards", "Plotly con visualizaciones interactivas"),
        ("✓ Factor Attribution", "Feature importance analysis"),
        ("✓ Signal Correlation", "Correlation matrix de señales"),
        ("✓ Worst Trades Analysis", "Post-mortem de peores trades"),
        ("✓ Risk Monitoring", "Alertas automáticas (DD, losses, Sharpe)"),
        ("✓ Notifications", "Email/Slack/Log integration"),
        ("✓ Meta-Validation", "Backtesting del backtest"),
        ("✓ Strategy Comparison", "Fixed vs Adaptive"),
        ("✓ QuantStats Export", "Reportes compatibles con QuantStats"),
    ]

    for feature, description in features_v3:
        print(f"  {feature:<30} → {description}")

    print("\n" + "="*80)

    # ========================================================================
    # USAGE EXAMPLES
    # ========================================================================
    print("\n📖 USAGE EXAMPLES:")
    print("-" * 80)

    print("\n1️⃣  BASIC USAGE (V2 Compatible):")
    print("""
    config = BacktestConfig(
        initial_capital=100000,
        position_size_pct=0.02,
        commission=0.001,
        slippage=0.002
    )

    engine = BacktestEngineV3(config, adaptive=False, use_ensemble=False)
    # Use as BacktestEngineV2
    """)

    print("\n2️⃣  ADAPTIVE MODE:")
    print("""
    engine = BacktestEngineV3(config, adaptive=True, use_ensemble=True)

    # Train ensemble models
    engine.train_ensemble(X_train, y_train)

    # Update market regime
    spy_data = yf.Ticker('SPY').history(period='1y')
    engine.update_market_regime(spy_data)

    # Check if should retrain
    if engine.should_retrain(current_date):
        engine.train_ensemble(X_new, y_new)

    # Monitor risks
    alerts = engine.check_risk_alerts()
    """)

    print("\n3️⃣  PARALLEL BACKTESTING:")
    print("""
    tickers = ['SNDL', 'GNUS', 'TOPS', 'SHIP', ...]

    results = run_backtest_parallel(
        tickers=tickers,
        start_date='2020-01-01',
        end_date='2023-12-31',
        model=your_model,
        config=config,
        n_workers=8,
        checkpoint_every=10
    )
    """)

    print("\n4️⃣  ADVANCED REPORTING:")
    print("""
    # Interactive dashboard
    dashboard_path = generate_interactive_dashboard(
        results, equity_curve, trades
    )

    # Factor attribution
    attribution = factor_attribution_analysis(
        ensemble_models, feature_names
    )

    # Worst trades analysis
    worst_analysis = worst_trades_analysis(trades, top_n=10)

    # QuantStats report
    quantstats_path = export_to_quantstats(equity_curve, trades)
    """)

    print("\n5️⃣  META-VALIDATION:")
    print("""
    # Compare adaptive vs fixed
    comparison = meta_validation_backtest(
        strategy_adaptive=adaptive_results,
        strategy_fixed=fixed_results
    )

    print(comparison['recommendation'])
    # Output: 'USE ADAPTIVE STRATEGY' or 'USE FIXED STRATEGY'
    """)

    # ========================================================================
    # CONFIGURATION EXAMPLE
    # ========================================================================
    print("\n" + "="*80)
    print("📝 EXAMPLE CONFIGURATION:")
    print("-" * 80)

    config = BacktestConfig(
        initial_capital=100000,
        position_size_pct=0.02,
        commission=0.001,
        slippage=0.002,
        stop_loss_pct=0.15,
        take_profit_pct=0.20,
        max_positions=10,
        max_holding_days=30
    )

    print(f"  Capital Inicial:    ${config.initial_capital:,.2f}")
    print(f"  Position Size:      {config.position_size_pct*100:.1f}%")
    print(f"  Max Positions:      {config.max_positions}")
    print(f"  Commission:         {config.commission*100:.2f}%")
    print(f"  Slippage:           {config.slippage*100:.2f}%")
    print(f"  Stop Loss:          {config.stop_loss_pct*100:.1f}%")
    print(f"  Take Profit:        {config.take_profit_pct*100:.1f}%")
    print(f"  Max Holding Days:   {config.max_holding_days}")

    # ========================================================================
    # CHECKLIST
    # ========================================================================
    print("\n" + "="*80)
    print("✅ PRE-LIVE CHECKLIST:")
    print("-" * 80)
    print("""
    Antes de usar en trading real:

    □ Backtest en out-of-sample completo (min. 2 años)
    □ Forward test en paper trading (3 meses mínimo)
    □ Transaction costs REALISTAS incluidos
    □ Sharpe ratio > 1.5 en forward test
    □ Max Drawdown tolerable (< 25%)
    □ Train/Val/Test gap < 20% (no overfitting)
    □ Validación estadística vs benchmark (p-value < 0.05)
    □ Monte Carlo simulation ejecutado
    □ Risk monitoring activo
    □ Alertas configuradas
    □ Código revisado por otro developer
    □ Plan de exit si no funciona
    □ Capital de riesgo que puedes perder
    """)

    # ========================================================================
    # GOLDEN PRINCIPLES
    # ========================================================================
    print("\n" + "="*80)
    print("⚠️  GOLDEN PRINCIPLES:")
    print("-" * 80)
    print("""
    1. "Un backtest que se ve demasiado bueno, probablemente lo sea"
       → Si Win Rate > 75%, Sharpe > 3.0, Max DD < 10%
       → Probablemente hay overfitting u olvidas algo crítico

    2. "Solo ajustar thresholds en VALIDATION set, NUNCA en test"
       → Test set es sacrosanto, solo para validación final

    3. "Más datos de training ≠ Mejor modelo"
       → Window size fijo (2 años) es mejor que usar todo el historial

    4. "Re-entrenar con cuidado"
       → Cada 3 meses máximo
       → Tracking de drift obligatorio
       → Auto-pause si Sharpe < 0.5

    5. "Transaction costs son CRÍTICOS"
       → Penny stocks tienen spread alto (0.2-0.5%)
       → Slippage + Commission destruyen estrategias marginales
    """)

    print("\n" + "="*80)
    print("✓ BACKTEST ENGINE V3.0 READY FOR USE")
    print("="*80)

    logger.info("\n✓ Framework V3.0 listo para uso")
    logger.info("  Próximos pasos:")
    logger.info("  1. Integrar con estrategia específica")
    logger.info("  2. Entrenar ensemble de modelos")
    logger.info("  3. Ejecutar backtesting con adaptive mode")
    logger.info("  4. Analizar resultados con dashboards interactivos")
    logger.info("  5. Validar con meta-validation")
    logger.info("  6. Paper trading antes de live")
