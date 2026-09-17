# Data ingestion

## Expected raw format

Put Dukascopy exports (CSV, one file per 6-month chunk is fine) in
`data/raw/`. Expected columns (Dukascopy M1/tick CSV default):

```
timestamp, open, high, low, close, volume
```

Timestamps should be UTC. If your export uses tick data instead of M1 bars,
run `python -m src.data.resample_ticks` first (see that module's docstring)
to produce M1 OHLCV before validation.

## Corruption / integrity checks

`python -m src.data.validate --path data/raw` runs, per file and across the
whole 5y set:

- **Duplicate timestamps** — dropped, logged.
- **Non-monotonic timestamps** — file flagged for manual review.
- **Impossible OHLC** — high < low, high < open/close, low > open/close.
- **Zero/negative volume runs** longer than a threshold (default 30 bars) —
  flagged (thin holiday liquidity is normal in short bursts, sustained
  zero-volume usually means bad export).
- **Missing session gaps** — flags gaps that don't line up with the FX
  weekend calendar (i.e., a hole during a normally-live session).
- **Return outliers** — |z-score| > 8 on 1-bar log returns, flagged for
  visual review (real gold gaps happen, e.g. Sunday opens, so this is a
  flag not an auto-drop).

Output: `data/_reports/validation_<timestamp>.json` with per-file pass/fail
and a merged, cleaned parquet at `data/clean/xauusd_m1.parquet` once you've
reviewed flags (`--apply` flag drops the confirmed-bad rows and writes the
merged file).

## In-sample / out-of-sample split

`src/data/loader.py:load_split()` takes `insample_days` (default 365) and
returns:
- `insample`: most recent N days before the OOS cutoff you specify
- `oos`: everything else in the 5y set, strictly non-overlapping

The split boundary is written to `results/frozen/<run_id>_split.json` at
freeze time so it's auditable later — you can always confirm evolution
never touched OOS bars.

## Known data limitation

Dukascopy's free historical feed has occasional full-day gaps unrelated to
market holidays (confirmed via direct re-fetch of the specific date, which
also returns 0 bytes). Observed so far: 2021-02-03, 2022-01-12, 2023-08-29,
2024-06-06, 2024-09-10, 2024-12-09, 2025-05-05. These are accepted as
provider-side gaps, not corruption — the backtest engine operates on bar
indices, not calendar time, so missing days don't corrupt trade logic.
