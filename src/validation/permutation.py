"""
Shuffled-trade analysis (IVB report §05).

Reshuffles the ORDER of the realized trade P&L series (not the set) and
re-walks the equity curve each time. Tells us whether the realized max
drawdown was a favorable/unfavorable accident of sequencing, or is close
to what any ordering of these exact trades would produce.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def max_drawdown(pnl_sequence: np.ndarray) -> float:
    equity = np.cumsum(pnl_sequence)
    running_max = np.maximum.accumulate(equity)
    dd = running_max - equity
    return float(dd.max()) if len(dd) else 0.0


def permutation_test(trades: pd.DataFrame, n_permutations: int = 1000, seed: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    pnl = trades["pnl"].values
    realized_dd = max_drawdown(pnl)

    shuffled_dds = np.empty(n_permutations)
    for k in range(n_permutations):
        shuffled = rng.permutation(pnl)
        shuffled_dds[k] = max_drawdown(shuffled)

    percentile_of_realized = float((shuffled_dds < realized_dd).mean() * 100)
    return {
        "n_permutations": n_permutations,
        "realized_max_dd": realized_dd,
        "shuffled_dd_min": float(shuffled_dds.min()),
        "shuffled_dd_max": float(shuffled_dds.max()),
        "shuffled_dd_median": float(np.median(shuffled_dds)),
        "shuffled_dd_p95": float(np.percentile(shuffled_dds, 95)),
        "realized_dd_percentile": percentile_of_realized,
        "interpretation": (
            "realized drawdown sits near the optimistic (low) end of plausible "
            "orderings" if percentile_of_realized < 40 else
            "realized drawdown is a typical, not tail, outcome under reshuffling"
            if percentile_of_realized < 90 else
            "realized drawdown sits near the pessimistic (high) end — treat live "
            "risk budget with extra caution"
        ),
    }
