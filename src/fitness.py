"""
Composite fitness = profit factor and win-rate stability dominate, per
your spec. Complexity is penalized lightly so the search doesn't drift
toward strategies with more filters than the sample can support.

fitness = 0 if n_trades < MIN_TRADES  (hard floor — no statistical basis)
        = pf_score * stability_score * (1 - complexity_penalty)

pf_score        : profit factor mapped through a soft cap (avoids runaway
                  scores from a handful of huge trades on tiny samples)
stability_score : 1 - normalized dispersion of win rate across N
                  sub-periods of the in-sample window (low dispersion =
                  the edge holds up piece by piece, not just in aggregate)
complexity_penalty : fraction of optional filters enabled (volume, mtf),
                     penalized because each extra filter is another
                     degree of freedom the evolutionary search could be
                     exploiting rather than a genuinely useful signal
"""
from __future__ import annotations
import numpy as np
import pandas as pd

MIN_TRADES = 30
PF_SOFT_CAP = 3.0
N_SUBPERIODS = 4


def win_rate_stability(trades: pd.DataFrame, n_subperiods: int = N_SUBPERIODS) -> float:
    if len(trades) < n_subperiods * 5:
        return 0.0
    trades = trades.sort_values("entry_time").reset_index(drop=True)
    # np.array_split() on a DataFrame is unreliable across numpy/pandas
    # versions (it can fall back to raw positional indexing that doesn't
    # understand DataFrame semantics, raising an IndexError on the
    # remainder chunk). Splitting by explicit position ranges avoids that
    # entirely and is just as correct for our purposes.
    n = len(trades)
    boundaries = [round(i * n / n_subperiods) for i in range(n_subperiods + 1)]
    chunks = [trades.iloc[boundaries[i]:boundaries[i + 1]] for i in range(n_subperiods)]
    win_rates = [float((c["pnl"] > 0).mean()) for c in chunks if len(c) > 0]
    if len(win_rates) < 2:
        return 0.0
    dispersion = float(np.std(win_rates))
    # dispersion of 0 -> stability 1.0; dispersion of 0.3+ -> stability ~0
    return float(max(0.0, 1.0 - dispersion / 0.3))


def complexity_penalty(genome: dict) -> float:
    n_optional = int(genome.get("use_volume_filter", False)) + int(genome.get("use_mtf_filter", False))
    return 0.10 * n_optional  # each optional filter costs 10% of fitness


def multi_window_fitness(genome: dict, insample_df: pd.DataFrame, n_windows: int = 3) -> dict:
    """
    Splits the in-sample period into n_windows consecutive, non-overlapping
    time chunks, backtests the genome on EACH separately, and combines
    results so that a genome which only works in one lucky sub-period
    scores worse than one that holds up across several.

    final_fitness = mean(per_window_fitness) * (1 - cross_window_penalty)

    cross_window_penalty grows with the dispersion of per-window fitness
    scores — a genome with fitness [0.6, 0.6, 0.55] beats one with
    [0.9, 0.1, 0.05] even though the second has a higher single-window peak
    and might even have a higher naive average with different weighting.
    This is deliberately stricter than the old single-window approach.
    """
    from src.backtest import run_backtest  # local import avoids a circular import at module load time

    insample_df = insample_df.sort_values("timestamp").reset_index(drop=True)
    n = len(insample_df)
    boundaries = [round(i * n / n_windows) for i in range(n_windows + 1)]

    window_reports = []
    for i in range(n_windows):
        chunk = insample_df.iloc[boundaries[i]:boundaries[i + 1]]
        if len(chunk) < 100:
            window_reports.append({"fitness": 0.0, "n_trades": 0})
            continue
        trades = run_backtest(chunk, genome)
        window_reports.append(fitness(genome, trades))

    window_scores = [r["fitness"] for r in window_reports]
    mean_score = float(np.mean(window_scores))
    dispersion = float(np.std(window_scores))
    cross_window_penalty = min(0.8, dispersion / 0.3)  # capped so it can't zero out a genuinely good genome by itself
    combined = mean_score * (1.0 - cross_window_penalty)

    total_trades = sum(r.get("n_trades", 0) for r in window_reports)
    return {
        "fitness": float(combined),
        "mean_window_fitness": mean_score,
        "cross_window_dispersion": dispersion,
        "cross_window_penalty": cross_window_penalty,
        "n_windows": n_windows,
        "n_trades_total": total_trades,
        "window_reports": window_reports,
    }


def fitness(genome: dict, trades: pd.DataFrame) -> dict:
    if trades is None or len(trades) < MIN_TRADES:
        return {"fitness": 0.0, "reason": f"n_trades < {MIN_TRADES}", "n_trades": 0 if trades is None else len(trades)}

    gross_profit = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    gross_loss = -trades.loc[trades["pnl"] < 0, "pnl"].sum()
    pf = gross_profit / gross_loss if gross_loss > 0 else PF_SOFT_CAP
    pf_score = min(pf, PF_SOFT_CAP) / PF_SOFT_CAP

    stability = win_rate_stability(trades)
    penalty = complexity_penalty(genome)

    score = pf_score * stability * (1.0 - penalty)
    return {
        "fitness": float(score),
        "profit_factor": float(pf),
        "win_rate_stability": stability,
        "complexity_penalty": penalty,
        "n_trades": len(trades),
    }
