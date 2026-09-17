"""
Walk-forward test: re-runs the FROZEN genome (no re-tuning) across
consecutive rolling windows of the OOS data (e.g. 4 x 12-month windows in
a 4-year OOS set) and checks stability window-by-window. This is the
check the IVB report itself flags as not yet done — we make it mandatory.
"""
from __future__ import annotations
import pandas as pd

from src.backtest import run_backtest, summarize


def walk_forward(oos_df: pd.DataFrame, genome: dict, window_days: int = 365) -> dict:
    start = oos_df["timestamp"].min()
    end = oos_df["timestamp"].max()

    windows = []
    cur = start
    while cur < end:
        window_end = cur + pd.Timedelta(days=window_days)
        chunk = oos_df[(oos_df["timestamp"] >= cur) & (oos_df["timestamp"] < window_end)]
        if len(chunk) > 100:
            trades = run_backtest(chunk, genome)
            metrics = summarize(trades)
            windows.append({"start": str(cur), "end": str(window_end), **metrics})
        cur = window_end

    profitable_windows = sum(1 for w in windows if w["net_pnl"] > 0)
    return {
        "n_windows": len(windows),
        "profitable_windows": profitable_windows,
        "profitable_window_pct": profitable_windows / len(windows) if windows else 0.0,
        "windows": windows,
    }
