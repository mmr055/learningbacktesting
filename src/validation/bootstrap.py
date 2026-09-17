"""
Bootstrap expected-value test (IVB report §06).

Resamples trades WITH replacement to build a distribution of EV ($/trade
and R/trade), giving a 95% CI and P(EV <= 0) as the edge-significance
test.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def bootstrap_ev(trades: pd.DataFrame, n_resamples: int = 20_000, seed: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    pnl = trades["pnl"].values
    r_mult = trades["r_multiple"].values
    n = len(pnl)

    idx = rng.integers(0, n, size=(n_resamples, n))
    ev_dollar = pnl[idx].mean(axis=1)
    ev_r = r_mult[idx].mean(axis=1)

    return {
        "n_trades": n,
        "ev_dollar_point": float(pnl.mean()),
        "ev_r_point": float(r_mult.mean()),
        "ev_dollar_ci95": [float(np.percentile(ev_dollar, 2.5)), float(np.percentile(ev_dollar, 97.5))],
        "ev_r_ci95": [float(np.percentile(ev_r, 2.5)), float(np.percentile(ev_r, 97.5))],
        "p_ev_dollar_leq_0": float((ev_dollar <= 0).mean()),
        "p_ev_r_leq_0": float((ev_r <= 0).mean()),
    }
