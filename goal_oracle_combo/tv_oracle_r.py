"""Oracle regime labels (same-day hindsight, ceiling only). Two trials."""
import numpy as np
import pandas as pd
from src.data.loader import load_clean
from src.tv_strategy import build_daily, daily_trend, _day_key
from src.tv_oracle import pf, line, N_DRAWS, PRE, MAIN


def bad_days(daily, kind):
    """Series of bool, True = day removed."""
    rng_ = daily["high"] - daily["low"]
    if kind == "volband":
        p = rng_.rank(pct=True)
        return (p <= 0.20) | (p >= 0.80)
    if kind == "combo":
        eff = (daily["close"] - daily["open"]).abs() / rng_.replace(0, np.nan)
        return (eff <= eff.quantile(0.33)) | bad_days(daily, "trendagree")
    d_tr = daily_trend(daily, 2, 2)
    prev = pd.Series(np.concatenate([[0], d_tr[:-1]]), index=daily.index)
    sgn = np.sign(daily["close"] - daily["open"])
    return sgn != prev


def run(label, tag, path, kind):
    m1 = load_clean(path)
    for c in ("open", "high", "low", "close"):
        m1[c] = m1[c].astype("float64")
    daily = build_daily(m1)
    bad = bad_days(daily, kind)
    tr = pd.read_csv(f"results/tv_strategy/trades_{tag}.csv")
    t = pd.to_datetime(tr["entry_time"], utc=True)
    tr["day"] = _day_key(t)
    tr["chop"] = tr["day"].map(bad).fillna(False).astype(bool)
    tr["year"] = t.dt.year
    print(f"\n===== {label} | label={kind} | removed days {bad.mean():.1%}")
    print(f"share of trades removed: {tr['chop'].mean():.1%} (total={len(tr)})")
    codes = pd.factorize(tr["day"])[0]
    ndays = codes.max() + 1
    n_bad = tr.loc[tr["chop"], "day"].nunique()
    rng = np.random.default_rng(11)
    for cost in (0.0, 0.3):
        r = (tr["raw_pts"] - cost) / tr["risk"]
        keep, drop = r[~tr["chop"]], r[tr["chop"]]
        print(f"-- round-trip cost {cost:.2f} USD/oz")
        print(line("all trades", r))
        print(line("kept", keep))
        print(line("removed", drop))
        avgs = np.array([r[~np.isin(codes, rng.choice(ndays, n_bad, replace=False))].mean()
                         for _ in range(N_DRAWS)])
        print(f"random removal: avgR mean {avgs.mean():+.4f} p95 {np.percentile(avgs, 95):+.4f}"
              f" | share of random >= oracle: {(avgs >= keep.mean()).mean():.3f}")


def main():
    tags = (
        ("2005-2020 30m rr1.5", "pre_30m_rr1.5", PRE),
        ("2021-2026 30m rr1.5", "30m_rr1.5", MAIN),
        ("2005-2020 15m rr1.5", "pre_15m_rr1.5", PRE),
        ("2021-2026 15m rr1.5", "rr1.5", MAIN),
        ("2005-2020 1h rr1.5", "pre_1h_rr1.5", PRE),
        ("2021-2026 1h rr1.5", "1h_rr1.5", MAIN),
        ("2005-2020 4h rr1.5", "pre_4h_rr1.5", PRE),
        ("2021-2026 4h rr1.5", "4h_rr1.5", MAIN),
    )
    for kind in ("combo",):
        for label, tag, path in tags:
            run(label, tag, path, kind)


if __name__ == "__main__":
    main()
