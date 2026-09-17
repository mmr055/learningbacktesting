"""
End-to-end orchestration:

  1. Load in-sample / OOS split (data/loader.py) — boundary frozen to disk.
  2. Evolve on in-sample ONLY (src/evolution.py).
  3. Freeze the top candidate genome (immutable JSON, timestamped).
  4. Run the full validation battery against OOS data:
       - out-of-sample backtest metrics
       - permutation (shuffle) test
       - bootstrap EV
       - Monte Carlo
       - walk-forward across rolling OOS windows
       - cost-adjusted re-run + slippage stress
  5. Apply explicit pass/fail thresholds -> verdict.
  6. Write results/latest.json (dashboard reads this) and an archived,
     timestamped copy under results/runs/.

Run: python -m src.pipeline --generations 40 --population 60
"""
from __future__ import annotations
import argparse
import json
import os
import gc
from datetime import datetime, timezone

from src.data.loader import load_split, load_clean
from src.evolution import evolve
from src.backtest import run_backtest, summarize
from src.validation.permutation import permutation_test
from src.validation.bootstrap import bootstrap_ev
from src.validation.monte_carlo import monte_carlo
from src.validation.walk_forward import walk_forward
from src.validation.cost_adjust import cost_adjusted_run, slippage_stress

# ---- desk thresholds (edit these deliberately, not silently) ----
THRESHOLDS = {
    "min_oos_trades": 100,
    "min_profit_factor": 1.20,          # IVB used 1.20 as the cost floor
    "min_return_over_maxdd": 3.0,       # IVB desk's first-gate screen was 3.0
    "max_p_ev_leq_0": 0.05,             # bootstrap edge significance
    "min_profitable_walkforward_pct": 0.60,
}


def run_id() -> str:
    return datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")


def apply_verdict(oos_metrics: dict, bootstrap: dict, wf: dict) -> dict:
    checks = {
        "n_trades_ok": oos_metrics["n_trades"] >= THRESHOLDS["min_oos_trades"],
        "profit_factor_ok": oos_metrics["profit_factor"] >= THRESHOLDS["min_profit_factor"],
        "return_over_maxdd_ok": (
            oos_metrics["net_pnl"] / oos_metrics["max_dd"] if oos_metrics["max_dd"] > 0 else 0
        ) >= THRESHOLDS["min_return_over_maxdd"],
        "bootstrap_significance_ok": bootstrap["p_ev_dollar_leq_0"] <= THRESHOLDS["max_p_ev_leq_0"],
        "walk_forward_ok": wf["profitable_window_pct"] >= THRESHOLDS["min_profitable_walkforward_pct"],
    }
    passed = all(checks.values())
    return {"passed": passed, "checks": checks, "thresholds": THRESHOLDS}


def main(generations: int, population: int, insample_days: int, seed: int | None):
    rid = run_id()
    frozen_dir = "results/frozen"
    runs_dir = "results/runs"
    os.makedirs(runs_dir, exist_ok=True)

    print(f"=== Run {rid} ===")
    print("Loading in-sample / OOS split...")
    insample_df, oos_df, split_info = load_split(insample_days=insample_days, run_id=rid, frozen_dir=frozen_dir)
    print(f"  in-sample: {split_info.insample_start} -> {split_info.insample_end} ({split_info.insample_rows} rows)")
    print(f"  OOS:       {split_info.oos_start} -> {split_info.oos_end} ({split_info.oos_rows} rows)")

    print("\nEvolving on in-sample data...")
    population_result = evolve(
        insample_df, population_size=population, generations=generations, seed=seed,
        log_path=os.path.join(runs_dir, f"{rid}_evolution_log.json"),
    )
    best = population_result[0]
    print(f"\nBest in-sample fitness: {best.score:.4f} | {best.fitness_report}")

    # freeze immutably BEFORE touching OOS
    os.makedirs(frozen_dir, exist_ok=True)
    frozen_path = os.path.join(frozen_dir, f"{rid}_genome.json")
    with open(frozen_path, "w") as fh:
        json.dump({"genome": best.genome, "insample_fitness": best.fitness_report}, fh, indent=2)
    print(f"Frozen genome -> {frozen_path}")

    print("\nRunning OOS validation battery...")
    oos_trades = run_backtest(oos_df, best.genome)
    oos_metrics = summarize(oos_trades)
    print(f"  OOS metrics: {oos_metrics}")

    perm = permutation_test(oos_trades, n_permutations=300, seed=seed)
    boot = bootstrap_ev(oos_trades, n_resamples=5_000, seed=seed)
    mc = monte_carlo(oos_trades, n_simulations=5_000, seed=seed)
    wf = walk_forward(oos_df, best.genome, window_days=365)
    gc.collect()
    cost_run = cost_adjusted_run(oos_df, best.genome)
    gc.collect()
    stress = slippage_stress(oos_df, best.genome)
    gc.collect()

    verdict = apply_verdict(oos_metrics, boot, wf)
    print(f"\nVERDICT: {'PASS' if verdict['passed'] else 'FAIL'} -> {verdict['checks']}")

    report = {
        "run_id": rid,
        "split_info": split_info.__dict__,
        "genome": best.genome,
        "insample_fitness": best.fitness_report,
        "oos_metrics": oos_metrics,
        "permutation_test": perm,
        "bootstrap": boot,
        "monte_carlo": mc,
        "walk_forward": wf,
        "cost_adjusted_metrics": cost_run["metrics"],
        "slippage_stress": stress,
        "verdict": verdict,
    }

    out_path = os.path.join(runs_dir, f"{rid}_report.json")
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    with open("results/latest.json", "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"\nWrote {out_path} and results/latest.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=40)
    ap.add_argument("--population", type=int, default=60)
    ap.add_argument("--insample-days", type=int, default=365)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    main(args.generations, args.population, args.insample_days, args.seed)
