"""
Integrity checks for raw XAUUSD OHLCV exports (e.g. from Dukascopy).

Run:
    python -m src.data.validate --path data/raw --apply

Without --apply, this only reports issues (data/_reports/validation_*.json).
With --apply, confirmed-bad rows (duplicates, impossible OHLC) are dropped
and a merged clean parquet is written to data/clean/xauusd_m1.parquet.
Return-outlier and gap flags are NEVER auto-dropped — gold has real gap
events (Sunday opens, NFP, etc.) and silently removing them would bias
the backtest. Those are for manual review.
"""
from __future__ import annotations

import argparse
import json
import glob
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

REQUIRED_COLS = ["timestamp", "open", "high", "low", "close", "volume"]
ZSCORE_FLAG_THRESHOLD = 8.0
ZERO_VOLUME_RUN_THRESHOLD = 30  # consecutive bars


@dataclass
class FileReport:
    path: str
    n_rows: int = 0
    duplicates_dropped: int = 0
    non_monotonic: bool = False
    impossible_ohlc_rows: int = 0
    zero_volume_runs: list = field(default_factory=list)
    session_gaps: list = field(default_factory=list)
    return_outliers: list = field(default_factory=list)
    ok: bool = True
    notes: list = field(default_factory=list)


def load_raw_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    # volume is sometimes absent (e.g. dukascopy-node with -v false / some
    # Dukascopy exports); rather than fail, synthesize a neutral placeholder
    # so the pipeline runs, but loudly say so — volume-based blocks
    # (src/blocks/volume.py) are meaningless on a constant column.
    if "volume" not in df.columns:
        print(f"  [WARN] {path}: no volume column found — filling with 1.0. "
              f"Volume-based strategy blocks will not be meaningful for this "
              f"file until you redownload with volumes enabled.")
        df["volume"] = 1.0

    missing = set(REQUIRED_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

    # timestamp may be an epoch (ms or s) or an ISO/human-readable string.
    # dukascopy-node emits epoch milliseconds (13-digit ints).
    ts_raw = df["timestamp"]
    if pd.api.types.is_numeric_dtype(ts_raw):
        sample = int(ts_raw.iloc[0])
        unit = "ms" if sample > 10**12 else "s"
        df["timestamp"] = pd.to_datetime(ts_raw, unit=unit, utc=True)
    else:
        df["timestamp"] = pd.to_datetime(ts_raw, utc=True)

    return df[REQUIRED_COLS]


def check_file(path: str) -> tuple[pd.DataFrame, FileReport]:
    df = load_raw_csv(path)
    report = FileReport(path=path, n_rows=len(df))

    # duplicates
    dupe_mask = df.duplicated(subset="timestamp", keep="first")
    report.duplicates_dropped = int(dupe_mask.sum())
    df = df[~dupe_mask].copy()

    # monotonicity
    if not df["timestamp"].is_monotonic_increasing:
        df = df.sort_values("timestamp").reset_index(drop=True)
        report.non_monotonic = True
        report.notes.append("timestamps were non-monotonic; sorted in place")

    # impossible OHLC
    bad = (
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
    )
    report.impossible_ohlc_rows = int(bad.sum())
    df = df[~bad].copy()

    # zero-volume runs
    zero_vol = (df["volume"] <= 0).astype(int)
    run_id = (zero_vol.diff() != 0).cumsum()
    for _, grp in df.groupby(run_id):
        if grp["volume"].iloc[0] <= 0 and len(grp) >= ZERO_VOLUME_RUN_THRESHOLD:
            report.zero_volume_runs.append(
                {"start": str(grp["timestamp"].iloc[0]), "end": str(grp["timestamp"].iloc[-1]), "len": len(grp)}
            )

    # session gap check: flag gaps > 5 min that are NOT explained by known
    # recurring closures - the daily FX broker rollover break (~21:00-23:10
    # UTC, every weekday) and the weekend closure (Fri ~21:00 -> Sun ~21:00).
    diffs = df["timestamp"].diff().dt.total_seconds().fillna(0) / 60.0
    for idx in np.where(diffs > 5)[0]:
        ts_prev = df["timestamp"].iloc[idx - 1]
        ts_cur = df["timestamp"].iloc[idx]
        gap_minutes = diffs.iloc[idx]

        is_weekend_gap = ts_prev.weekday() == 4 and ts_prev.hour >= 20
        # DST-aware: the broker's rollover is a fixed LOCAL time, so in UTC
        # it lands at hour 21 during northern-hemisphere winter and hour 20
        # during (US) daylight saving time (roughly mid-March to early Nov).
        is_daily_rollover = (
            (ts_prev.hour == 21 and ts_prev.minute >= 55 and gap_minutes <= 75)
            or (ts_prev.hour == 22 and gap_minutes <= 75)
            or (ts_prev.hour == 20 and ts_prev.minute >= 55 and gap_minutes <= 75)
            or (ts_prev.hour == 21 and gap_minutes <= 75 and ts_prev.minute == 59)
        )

        if not (is_weekend_gap or is_daily_rollover):
            report.session_gaps.append(
                {"from": str(ts_prev), "to": str(ts_cur), "minutes": float(gap_minutes)}
            )

    # return outliers
    log_ret = np.log(df["close"]).diff()
    z = (log_ret - log_ret.mean()) / (log_ret.std(ddof=0) + 1e-12)
    outlier_idx = np.where(np.abs(z) > ZSCORE_FLAG_THRESHOLD)[0]
    for idx in outlier_idx:
        report.return_outliers.append(
            {"timestamp": str(df["timestamp"].iloc[idx]), "z": float(z.iloc[idx])}
        )

    report.ok = (
        report.impossible_ohlc_rows == 0
        and len(report.zero_volume_runs) == 0
        and len(report.session_gaps) == 0
    )
    return df, report


def run(path_glob: str, apply: bool, out_dir_clean: str, out_dir_reports: str) -> None:
    os.makedirs(out_dir_reports, exist_ok=True)
    files = sorted(glob.glob(os.path.join(path_glob, "*.csv")))
    if not files:
        print(f"No CSV files found under {path_glob}")
        return

    all_reports = []
    cleaned_frames = []
    for f in files:
        df, report = check_file(f)
        all_reports.append(report.__dict__)
        cleaned_frames.append(df)
        status = "OK" if report.ok else "FLAGGED"
        print(f"[{status}] {f}: {report.n_rows} rows, "
              f"{report.duplicates_dropped} dupes, {report.impossible_ohlc_rows} bad OHLC, "
              f"{len(report.zero_volume_runs)} zero-vol runs, {len(report.session_gaps)} gaps, "
              f"{len(report.return_outliers)} return outliers")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = os.path.join(out_dir_reports, f"validation_{ts}.json")
    with open(report_path, "w") as fh:
        json.dump(all_reports, fh, indent=2, default=str)
    print(f"\nWrote report: {report_path}")

    if apply:
        merged = pd.concat(cleaned_frames, ignore_index=True)
        merged = merged.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
        os.makedirs(out_dir_clean, exist_ok=True)
        clean_path = os.path.join(out_dir_clean, "xauusd_m1.parquet")
        merged.to_parquet(clean_path, index=False)
        print(f"Wrote cleaned merged parquet: {clean_path} ({len(merged)} rows)")
        flagged = [r for r in all_reports if not r["ok"]]
        if flagged:
            print(f"\n⚠ {len(flagged)} file(s) still have unresolved flags (gaps/outliers) "
                  f"— review {report_path} before trusting this data for backtests.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data/raw")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--clean-out", default="data/clean")
    ap.add_argument("--reports-out", default="data/_reports")
    args = ap.parse_args()
    run(args.path, args.apply, args.clean_out, args.reports_out)
