"""
Cost-adjusted re-run (IVB report §08) + slippage stress test (their
"next checks" item, made mandatory here).
"""
from __future__ import annotations
import pandas as pd

from src.backtest import run_backtest, summarize


def cost_adjusted_run(
    df: pd.DataFrame,
    genome: dict,
    commission_per_trade: float = 5.20,
    slippage_ticks: float = 1.0,
) -> dict:
    trades = run_backtest(df, genome, commission_per_trade=commission_per_trade, slippage_ticks=slippage_ticks)
    return {"metrics": summarize(trades), "trades": trades}


def slippage_stress(df: pd.DataFrame, genome: dict, commission_per_trade: float = 5.20) -> dict:
    results = {}
    for ticks in (0, 1, 2, 3):
        trades = run_backtest(df, genome, commission_per_trade=commission_per_trade, slippage_ticks=ticks)
        results[f"{ticks}_tick_slippage"] = summarize(trades)
    return results
