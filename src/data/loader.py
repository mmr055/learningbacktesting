"""
Loads cleaned XAUUSD data and produces a strictly non-overlapping
in-sample / out-of-sample split. This module is the ONLY place the split
boundary is decided — src/evolution.py must only ever receive the
in-sample DataFrame from here, never the raw clean file directly.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict

import pandas as pd

CLEAN_PATH = "data/clean/xauusd_m1.parquet"


@dataclass
class SplitInfo:
    insample_start: str
    insample_end: str
    oos_start: str
    oos_end: str
    insample_rows: int
    oos_rows: int


def load_clean(path: str = CLEAN_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    # float32 instead of float64: halves memory for price/volume columns.
    # Gold prices to 3 decimal places are nowhere near float32's precision
    # limit, so this costs us nothing numerically but matters a lot on
    # memory-constrained machines with multi-million-row OOS sets.
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype("float32")
    return df.sort_values("timestamp").reset_index(drop=True)


def resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """timeframe e.g. '5min', '15min', '1h', '4h', '1D' (pandas offset aliases)."""
    d = df.set_index("timestamp")
    out = d.resample(timeframe).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open", "high", "low", "close"])
    return out.reset_index()


def load_split(
    insample_days: int = 365,
    path: str = CLEAN_PATH,
    run_id: str | None = None,
    frozen_dir: str = "results/frozen",
) -> tuple[pd.DataFrame, pd.DataFrame, SplitInfo]:
    """
    Returns (insample_df, oos_df, split_info).

    oos = most recent `insample_days` of the dataset (held out as the
    "future" the strategy never saw during evolution).
    insample = everything strictly before that window (the "past" used
    for training/evolution).

    If run_id is given, the split boundary is written immutably to
    results/frozen/<run_id>_split.json so it's auditable that evolution
    never saw oos bars.
    """
    df = load_clean(path)
    end = df["timestamp"].max()
    cutoff = end - pd.Timedelta(days=insample_days)

    # OOS = most recent `insample_days` window (the "future" relative to training).
    # In-sample = everything strictly before that window (the "past" the strategy
    # is trained on). This mirrors real deployment: evolve on historical data,
    # then check performance on data that comes chronologically AFTER training
    # and that the strategy could never have seen.
    insample = df[df["timestamp"] <= cutoff].reset_index(drop=True)
    oos = df[df["timestamp"] > cutoff].reset_index(drop=True)

    info = SplitInfo(
        insample_start=str(insample["timestamp"].min()),
        insample_end=str(insample["timestamp"].max()),
        oos_start=str(oos["timestamp"].min()),
        oos_end=str(oos["timestamp"].max()),
        insample_rows=len(insample),
        oos_rows=len(oos),
    )

    if run_id:
        os.makedirs(frozen_dir, exist_ok=True)
        with open(os.path.join(frozen_dir, f"{run_id}_split.json"), "w") as fh:
            json.dump(asdict(info), fh, indent=2)

    return insample, oos, info


def load_oos_window(start: str, end: str, path: str = CLEAN_PATH) -> pd.DataFrame:
    """Explicit OOS window loader used ONLY by src/validation/*, never by
    src/evolution.py. Kept as a separate entry point so a code review can
    grep for who imports this."""
    df = load_clean(path)
    mask = (df["timestamp"] >= pd.Timestamp(start, tz="UTC")) & (df["timestamp"] <= pd.Timestamp(end, tz="UTC"))
    return df[mask].reset_index(drop=True)
