"""
Price-action / market-structure building blocks.

Every block is a pure function: (df, **params) -> pd.Series (bool or float),
aligned to df.index. Genome nodes reference these by name + params so the
evolutionary search can mix and match them freely.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def fractal_high(df: pd.DataFrame, width: int = 2) -> pd.Series:
    """Bill Williams-style fractal high: a high with `width` lower highs on
    each side. Returns bool series marking fractal-high bars."""
    h = df["high"]
    is_max = pd.Series(True, index=df.index)
    for shift in range(1, width + 1):
        is_max &= h > h.shift(shift)
        is_max &= h > h.shift(-shift)
    return is_max.fillna(False)


def fractal_low(df: pd.DataFrame, width: int = 2) -> pd.Series:
    l = df["low"]
    is_min = pd.Series(True, index=df.index)
    for shift in range(1, width + 1):
        is_min &= l < l.shift(shift)
        is_min &= l < l.shift(-shift)
    return is_min.fillna(False)


def swing_levels(df: pd.DataFrame, width: int = 2) -> pd.DataFrame:
    """Forward-filled last-confirmed swing high/low levels (a fractal is only
    'confirmed' `width` bars after it forms — this avoids lookahead bias)."""
    fh = fractal_high(df, width)
    fl = fractal_low(df, width)
    swing_high = df["high"].where(fh).shift(width)  # confirmed only after width bars
    swing_low = df["low"].where(fl).shift(width)
    out = pd.DataFrame(index=df.index)
    out["swing_high"] = swing_high.ffill()
    out["swing_low"] = swing_low.ffill()
    return out


def break_of_structure(df: pd.DataFrame, width: int = 2, direction: str = "long") -> pd.Series:
    """True on the bar where close breaks the last confirmed swing
    high (long) or swing low (short)."""
    levels = swing_levels(df, width)
    if direction == "long":
        return (df["close"] > levels["swing_high"]) & (df["close"].shift(1) <= levels["swing_high"].shift(1))
    return (df["close"] < levels["swing_low"]) & (df["close"].shift(1) >= levels["swing_low"].shift(1))


def opening_range(df: pd.DataFrame, session_start_hour: int, duration_min: int, tf_minutes: int = 1) -> pd.DataFrame:
    """Per-session opening range high/low, computed causally (only known
    once the range window has closed for that session)."""
    ts = df["timestamp"]
    session_date = ts.dt.tz_convert("UTC").dt.date
    minute_of_day = ts.dt.hour * 60 + ts.dt.minute
    range_start = session_start_hour * 60
    range_end = range_start + duration_min
    in_range = (minute_of_day >= range_start) & (minute_of_day < range_end)

    grp = pd.DataFrame({"date": session_date, "high": df["high"], "low": df["low"], "in_range": in_range})
    agg = grp[grp["in_range"]].groupby("date").agg(or_high=("high", "max"), or_low=("low", "min"))

    out = pd.DataFrame(index=df.index)
    out["date"] = session_date
    out = out.join(agg, on="date")
    # only valid AFTER the range window has closed for that session
    range_closed = minute_of_day >= range_end
    out.loc[~range_closed.values, ["or_high", "or_low"]] = np.nan
    out[["or_high", "or_low"]] = out.groupby("date")[["or_high", "or_low"]].ffill()
    return out[["or_high", "or_low"]]


def range_width_pct(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Rolling (high-low)/close range as a volatility-regime proxy."""
    rng = (df["high"].rolling(window).max() - df["low"].rolling(window).min())
    return rng / df["close"]
