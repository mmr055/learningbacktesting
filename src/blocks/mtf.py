"""
Multi-timeframe (MTF) fractal alignment blocks.

Higher-timeframe context is computed on a resampled frame, then re-aligned
(merge_asof, backward) onto the base timeframe index so there is zero
lookahead: at base-bar t, the HTF value used is the last HTF bar that had
FULLY CLOSED at or before t.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.data.loader import resample
from src.blocks.structure import swing_levels, fractal_high, fractal_low


def _align_htf_to_base(base_ts: pd.Series, htf_df: pd.DataFrame, htf_cols: list[str]) -> pd.DataFrame:
    htf = htf_df[["timestamp"] + htf_cols].copy()
    base = pd.DataFrame({"timestamp": base_ts})
    merged = pd.merge_asof(base, htf, on="timestamp", direction="backward")
    return merged[htf_cols]


def htf_trend_bias(df: pd.DataFrame, htf: str = "4h", width: int = 2) -> pd.Series:
    """
    +1 if the higher timeframe's last confirmed swing structure is making
    higher highs & higher lows (uptrend bias), -1 if lower highs & lower
    lows, 0 otherwise. Used as a directional filter so the base-timeframe
    setup only fires with higher-timeframe alignment.

    Implementation note: swing_high/low only change value on the (rare)
    bars where a new fractal confirms, so comparing consecutive bars
    directly would require a higher-high AND higher-low to land on the
    exact same bar — almost never happens and silently kills every signal.
    Instead we track each series' last-move direction as PERSISTENT state
    (forward-filled only on actual change bars) so "still in an uptrend
    structure" holds between confirmations, not just on the instant both
    legs move together.
    """
    htf_df = resample(df, htf)
    lv = swing_levels(htf_df, width)
    htf_df = htf_df.join(lv)

    hh_diff = htf_df["swing_high"].diff()
    hl_diff = htf_df["swing_low"].diff()
    hh_state = np.sign(hh_diff).replace(0, np.nan).ffill()
    hl_state = np.sign(hl_diff).replace(0, np.nan).ffill()

    bias = pd.Series(0, index=htf_df.index)
    bias[(hh_state == 1) & (hl_state == 1)] = 1
    bias[(hh_state == -1) & (hl_state == -1)] = -1
    htf_df["bias"] = bias

    aligned = _align_htf_to_base(df["timestamp"], htf_df, ["bias"])
    return aligned["bias"].fillna(0)


def htf_fractal_confluence(df: pd.DataFrame, htf: str = "1h", width: int = 2) -> pd.Series:
    """
    True where the base-timeframe bar's close sits within `width`*ATR of an
    HTF fractal high/low level — i.e. base-TF setup lines up with an HTF
    reaction zone. Cheap proxy for multi-timeframe structural confluence.
    """
    htf_df = resample(df, htf)
    htf_df["fh"] = fractal_high(htf_df, width)
    htf_df["fl"] = fractal_low(htf_df, width)
    htf_df["level"] = htf_df["high"].where(htf_df["fh"]).combine_first(htf_df["low"].where(htf_df["fl"]))
    htf_df["level"] = htf_df["level"].ffill()

    aligned = _align_htf_to_base(df["timestamp"], htf_df, ["level"])
    atr = (df["high"] - df["low"]).rolling(14).mean()
    dist = (df["close"] - aligned["level"]).abs()
    return (dist <= atr).fillna(False)
