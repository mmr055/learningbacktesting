# XAUUSD Evolutionary Strategy Research & Validation Pipeline

Continuous, cloud-scheduled system that:
1. Evolves candidate mechanical trading strategies for XAUUSD out of a
   library of price-action / structure / volume / multi-timeframe-fractal
   building blocks, using an in-sample window.
2. Freezes the best candidate(s) and puts them through an institutional-style
   statistical validation battery (out-of-sample test, permutation test,
   bootstrap EV, Monte Carlo, walk-forward, cost-adjustment) modeled on the
   IVB validation methodology.
3. Publishes results to a dashboard that updates every scheduled run.

## Why this structure

Overfitting is fought architecturally, not by discipline alone:
- The evolutionary loop (`src/evolution.py`) NEVER sees out-of-sample data.
  It only ever calls `backtest()` against the in-sample slice.
- A strategy only "graduates" to `results/validated/` if it passes explicit,
  hard-coded thresholds in `src/validation/pipeline.py` on the OOS data —
  and OOS data is loaded by a completely separate code path
  (`src.data.loader.load_oos_window`) that the evolution module never imports.
- Every validated strategy's genome + parameters are logged immutably before
  validation starts (`results/frozen/<run_id>.json`), so a strategy can never
  be quietly re-tuned after seeing OOS performance.

## Repo layout

```
data/                    Raw + cleaned OHLCV parquet, gitignored (see data/README.md)
src/data/                Loading, integrity checks, in-sample/OOS split, resampling
src/blocks/              Strategy building blocks (structure, volume, fractal/MTF)
src/genome.py            Strategy representation (genome <-> executable rule set)
src/backtest.py          Vectorized backtest engine -> trade log + metrics
src/fitness.py           Composite fitness: profit factor + win-rate stability
src/evolution.py         Genetic algorithm loop (in-sample ONLY)
src/validation/          Permutation, bootstrap, Monte Carlo, walk-forward, cost-adjust
src/pipeline.py          Orchestrates: evolve -> freeze -> validate -> report
dashboard/               Streamlit dashboard reading results/*.json
.github/workflows/       Scheduled GitHub Actions runner (free tier)
results/                 Run outputs (frozen genomes, validation reports, dashboard data)
```

## Data workflow (your part)

1. Download 6-month XAUUSD chunks from Dukascopy (or equivalent) covering the
   last 5 years, save to Google Drive.
2. Sync/download that Drive folder into `data/raw/` locally, or point
   `DATA_DIR` env var at a mounted Drive folder in Actions (see
   `data/README.md` for exact steps and the corruption-check script).
3. Run `python -m src.data.validate` — flags gaps, duplicate timestamps,
   impossible OHLC (high<low, etc.), zero-volume runs, and outlier price
   spikes (z-score on returns) BEFORE any chunk is used.

## Running locally

```bash
pip install -r requirements.txt
python -m src.pipeline --generations 40 --population 60
streamlit run dashboard/app.py
```

## Running on GitHub Actions (free tier)

`.github/workflows/evolve.yml` runs on a schedule (default: nightly),
executes one evolution+validation cycle, and commits results back to the
repo so the dashboard (deployed via Streamlit Community Cloud or GitHub
Pages + static export) always reflects the latest run.
