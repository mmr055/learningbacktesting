"""
Monte Carlo simulation (IVB report §07).

Resamples full trade sequences with replacement, walks each simulated
equity path, and records max drawdown + terminal P&L per path. Output
percentiles are the realistic live-risk budget — NOT the single backtest
drawdown.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from src.validation.permutation import max_drawdown


def monte_carlo(trades: pd.DataFrame, n_simulations: int = 20_000, seed: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    pnl = trades["pnl"].values
    n = len(pnl)

    max_dds = np.empty(n_simulations)
    terminals = np.empty(n_simulations)
    for k in range(n_simulations):
        sample = pnl[rng.integers(0, n, size=n)]
        max_dds[k] = max_drawdown(sample)
        terminals[k] = sample.sum()

    pct = lambda arr, p: float(np.percentile(arr, p))
    return {
        "n_simulations": n_simulations,
        "trades_per_sim": n,
        "max_dd_percentiles": {p: pct(max_dds, p) for p in (50, 75, 90, 95, 99)},
        "terminal_pnl_percentiles": {p: pct(terminals, p) for p in (5, 25, 50, 75, 95)},
        "p_maxdd_gte_realized": float((max_dds >= max_drawdown(pnl)).mean()),
    }
