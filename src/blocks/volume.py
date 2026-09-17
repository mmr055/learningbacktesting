"""Volume-based building blocks."""
from __future__ import annotations
import pandas as pd


def volume_zscore(df: pd.DataFrame, window: int = 50) -> pd.Series:
    v = df["volume"]
    mean = v.rolling(window).mean()
    std = v.rolling(window).std(ddof=0)
    return (v - mean) / (std + 1e-9)


def volume_spike(df: pd.DataFrame, window: int = 50, z_threshold: float = 2.0) -> pd.Series:
    """True where volume is an outlier spike vs its rolling distribution —
    used as a confirmation filter on structure breaks (a break on
    below-average volume is a much weaker signal)."""
    return volume_zscore(df, window) > z_threshold


def on_balance_volume(df: pd.DataFrame) -> pd.Series:
    direction = pd.Series(0, index=df.index, dtype=float)
    direction[df["close"] > df["close"].shift(1)] = 1
    direction[df["close"] < df["close"].shift(1)] = -1
    return (direction * df["volume"]).cumsum()


def obv_divergence(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Simple divergence proxy: sign mismatch between price slope and OBV
    slope over `window` bars. Positive = bullish divergence."""
    price_slope = df["close"].diff(window)
    obv = on_balance_volume(df)
    obv_slope = obv.diff(window)
    return (price_slope < 0) & (obv_slope > 0)  # bullish divergence, boolean
