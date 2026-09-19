"""
ORACLE (hindsight) test, NOT a tradable filter and NOT statistically valid:
label choppy trading days using the full day's OHLC, drop trades entered on
those days, compare with removing the same number of random days.
Upper bound for what perfect chop avoidance could add to the unfiltered strategy.
"""
import numpy as np
import pandas as pd

from src.data.loader import load_clean
from src.tv_strategy import build_daily, _day_key

CHOP_SHARE = 0.33
N_DRAWS = 500
PRE = "/home/mahfuj/xauusd_pre2021/xauusd_m1_pre2021.parquet"
MAIN = "data/clean/xauusd_m1.parquet"


def pf(r):
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    return gp / gl if gl > 0 else np.nan


def line(name, r):
    return f"{name:20s} n={len(r):5d} PF={pf(r):.3f} avgR={r.mean():+.4f} netR={r.sum():+.1f}"


def run(label, tag, path):
    m1 = load_clean(path)
    for c in ("open", "high", "low", "close"):
        m1[c] = m1[c].astype("float64")
    daily = build_daily(m1)
    rng_ = (daily["high"] - daily["low"]).replace(0, np.nan)
    eff = (daily["close"] - daily["open"]).abs() / rng_
    thr = eff.quantile(CHOP_SHARE)
    chop = eff <= thr

    tr = pd.read_csv(f"results/tv_strategy/trades_{tag}.csv")
    t = pd.to_datetime(tr["entry_time"], utc=True)
    tr["day"] = _day_key(t)
    tr["chop"] = tr["day"].map(chop).fillna(False).astype(bool)
    tr["year"] = t.dt.year

    print(f"\n===== {label} | choppy = bottom {CHOP_SHARE:.0%} of days by |C-O|/(H-L) (threshold {thr:.3f})")
    print(f"share of trades entered on choppy days: {tr['chop'].mean():.1%}")
    codes = pd.factorize(tr["day"])[0]
    ndays = codes.max() + 1
    n_chop_days = tr.loc[tr["chop"], "day"].nunique()
    rng = np.random.default_rng(11)

    for cost in (0.0, 0.3):
        r = (tr["raw_pts"] - cost) / tr["risk"]
        keep, drop = r[~tr["chop"]], r[tr["chop"]]
        print(f"-- round-trip cost {cost:.2f} USD/oz")
        print(line("all trades", r))
        print(line("kept (non-choppy)", keep))
        print(line("removed (choppy)", drop))
        avgs = []
        for _ in range(N_DRAWS):
            bad = rng.choice(ndays, n_chop_days, replace=False)
            avgs.append(r[~np.isin(codes, bad)].mean())
        avgs = np.array(avgs)
        print(f"random removal of same # of days: avgR mean {avgs.mean():+.4f} p95 {np.percentile(avgs, 95):+.4f}"
              f" | share of random >= oracle: {(avgs >= keep.mean()).mean():.3f}")
        if cost == 0.3:
            ky = keep.groupby(tr.loc[~tr["chop"], "year"]).sum()
            print(f"kept trades: years with positive netR {int((ky > 0).sum())} of {len(ky)}")


def main():
    for label, tag, path in (
        ("2005-2020 rr1.0", "pre_15m_rr1.0", PRE),
        ("2005-2020 rr1.5", "pre_15m_rr1.5", PRE),
        ("2021-2026 rr1.0", "base", MAIN),
        ("2021-2026 rr1.5", "rr1.5", MAIN),
        ("2005-2020 30m rr1.5", "pre_30m_rr1.5", PRE),
        ("2005-2020 1h rr1.5", "pre_1h_rr1.5", PRE),
        ("2005-2020 4h rr1.5", "pre_4h_rr1.5", PRE),
        ("2021-2026 30m rr1.5", "30m_rr1.5", MAIN),
        ("2021-2026 1h rr1.5", "1h_rr1.5", MAIN),
        ("2021-2026 4h rr1.5", "4h_rr1.5", MAIN),
    ):
        run(label, tag, path)


if __name__ == "__main__":
    main()
