"""
Smoke test only — generates synthetic M1 XAUUSD-like data and runs a tiny
version of the full pipeline (small population/generations) to confirm
every module imports and executes without error. NOT a substitute for
running on real data; just proves the wiring works.
"""
import os
import numpy as np
import pandas as pd


def make_synthetic_data(days=30, out_path="data/clean/xauusd_m1.parquet"):
    rng = np.random.default_rng(42)
    n_bars = days * 24 * 60
    start = pd.Timestamp("2024-01-01", tz="UTC")
    ts = pd.date_range(start, periods=n_bars, freq="1min")

    # random walk with mild drift + intraday vol pattern, gold-ish scale ~2000
    rets = rng.normal(0, 0.0004, n_bars)
    price = 2000 * np.exp(np.cumsum(rets))
    close = price
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + rng.uniform(0, 0.5, n_bars)
    low = np.minimum(open_, close) - rng.uniform(0, 0.5, n_bars)
    volume = rng.integers(1, 200, n_bars)

    df = pd.DataFrame({"timestamp": ts, "open": open_, "high": high, "low": low, "close": close, "volume": volume})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path


if __name__ == "__main__":
    path = make_synthetic_data(days=30)
    print(f"Synthetic data written to {path}")

    from src.data.loader import load_split
    insample, oos, info = load_split(insample_days=10, path=path)
    print(f"in-sample rows={len(insample)}, oos rows={len(oos)}")

    from src.evolution import evolve
    pop = evolve(insample, population_size=8, generations=3, seed=1)
    best = pop[0]
    print("Best genome:", best.genome)
    print("Best fitness report:", best.fitness_report)

    from src.backtest import run_backtest, summarize
    trades = run_backtest(oos, best.genome)
    print("OOS trades:", len(trades))
    print("OOS summary:", summarize(trades))

    if len(trades) >= 5:
        from src.validation.permutation import permutation_test
        from src.validation.bootstrap import bootstrap_ev
        from src.validation.monte_carlo import monte_carlo
        print("Permutation test:", permutation_test(trades, n_permutations=100, seed=1))
        print("Bootstrap EV:", bootstrap_ev(trades, n_resamples=500, seed=1))
        print("Monte Carlo:", monte_carlo(trades, n_simulations=500, seed=1))
    else:
        print("Too few OOS trades on this tiny synthetic sample to run full stats — expected for a smoke test.")

    print("\nSMOKE TEST COMPLETE — no exceptions raised.")
