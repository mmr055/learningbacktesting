"""
Backtest engine: takes a genome (see src/genome.py) and an OHLCV DataFrame,
returns a trade log DataFrame and summary metrics.

Design notes:
- Only one open position at a time (mirrors the IVB reference model).
- Entries are signal-on-close, fill-on-next-bar-open (no same-bar lookahead).
- SL/TP are ATR- and R-multiple-based (not fixed-range, since blocks vary).
- Costs (commission + slippage) are parameters, not hardcoded, so the same
  engine runs both the "pure edge" pass and the "cost-adjusted" pass.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from src.blocks.structure import break_of_structure, opening_range
from src.blocks.volume import volume_spike
from src.blocks.mtf import htf_trend_bias, htf_fractal_confluence


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def build_entry_signal(df: pd.DataFrame, genome: dict) -> pd.Series:
    direction = genome["direction"]
    if genome["entry_block"] == "break_of_structure":
        sig = break_of_structure(df, width=genome["entry_params"]["width"], direction=direction)
    else:
        orb = opening_range(df, genome["entry_params"]["session_start_hour"], genome["entry_params"]["duration_min"])
        if direction == "long":
            sig = (df["close"] > orb["or_high"]) & (df["close"].shift(1) <= orb["or_high"].shift(1))
        else:
            sig = (df["close"] < orb["or_low"]) & (df["close"].shift(1) >= orb["or_low"].shift(1))
        sig = sig.fillna(False)

    if genome.get("use_volume_filter"):
        sig = sig & volume_spike(df, **genome["volume_params"])

    if genome.get("use_mtf_filter"):
        mtf = genome["mtf_params"]
        if mtf["mode"] == "trend_bias":
            bias = htf_trend_bias(df, htf=mtf["htf"], width=mtf["width"])
            wanted = 1 if direction == "long" else -1
            sig = sig & (bias == wanted)
        else:
            sig = sig & htf_fractal_confluence(df, htf=mtf["htf"], width=mtf["width"])

    return sig.fillna(False)


def run_backtest(
    df: pd.DataFrame,
    genome: dict,
    account_size: float = 100_000.0,
    commission_per_trade: float = 0.0,
    slippage_ticks: float = 0.0,
    tick_value: float = 0.10,   # $ value of a 0.01 XAUUSD move per unit lot (adjust to your broker spec)
) -> pd.DataFrame:
    direction = genome["direction"]
    sig = build_entry_signal(df, genome)
    atr = _atr(df, 14)

    stop_mult = genome["stop_atr_mult"]
    tp_r = genome["tp_r_multiple"]
    risk_pct = genome["risk_pct"] / 100.0
    max_bars = genome["max_bars_in_trade"]

    open_arr = df["open"].values
    high_arr = df["high"].values
    low_arr = df["low"].values
    close_arr = df["close"].values
    atr_arr = atr.values
    sig_arr = sig.values
    n = len(df)

    trades = []
    in_position = False
    i = 0
    while i < n - 1:
        if not in_position and sig_arr[i] and not np.isnan(atr_arr[i]) and atr_arr[i] > 0:
            entry_idx = i + 1  # fill next bar open
            entry_price = open_arr[entry_idx] + (slippage_ticks * 0.01 if direction == "long" else -slippage_ticks * 0.01)
            risk_dist = atr_arr[i] * stop_mult
            if direction == "long":
                sl = entry_price - risk_dist
                tp = entry_price + risk_dist * tp_r
            else:
                sl = entry_price + risk_dist
                tp = entry_price - risk_dist * tp_r

            units = (account_size * risk_pct) / max(risk_dist, 1e-6)

            exit_idx, exit_price, outcome = None, None, None
            j_end = min(entry_idx + max_bars, n - 1)
            for j in range(entry_idx, j_end + 1):
                hi, lo = high_arr[j], low_arr[j]
                if direction == "long":
                    hit_sl = lo <= sl
                    hit_tp = hi >= tp
                else:
                    hit_sl = hi >= sl
                    hit_tp = lo <= tp
                if hit_sl and hit_tp:
                    exit_idx, exit_price, outcome = j, sl, "sl"  # conservative: assume SL first if both hit
                    break
                if hit_sl:
                    exit_idx, exit_price, outcome = j, sl, "sl"
                    break
                if hit_tp:
                    exit_idx, exit_price, outcome = j, tp, "tp"
                    break
            if exit_idx is None:
                exit_idx, exit_price, outcome = j_end, close_arr[j_end], "time"

            raw_pnl = (exit_price - entry_price) * units if direction == "long" else (entry_price - exit_price) * units
            pnl = raw_pnl - commission_per_trade
            r_multiple = pnl / (account_size * risk_pct) if account_size * risk_pct > 0 else 0.0

            trades.append({
                "entry_time": df["timestamp"].iloc[entry_idx],
                "exit_time": df["timestamp"].iloc[exit_idx],
                "direction": direction,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "outcome": outcome,
                "bars_held": exit_idx - entry_idx,
                "pnl": pnl,
                "r_multiple": r_multiple,
            })
            i = exit_idx + 1
        else:
            i += 1

    return pd.DataFrame(trades)


def summarize(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n_trades": 0, "profit_factor": 0.0, "win_rate": 0.0, "net_pnl": 0.0,
                "avg_r": 0.0, "max_dd": 0.0, "sharpe_ratio": 0.0}
    gross_profit = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    gross_loss = -trades.loc[trades["pnl"] < 0, "pnl"].sum()
    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf
    equity = trades["pnl"].cumsum()
    running_max = equity.cummax()
    dd = (running_max - equity)
    r_std = trades["r_multiple"].std()
    sharpe = float(trades["r_multiple"].mean() / r_std) if r_std and r_std > 0 else 0.0

    return {
        "n_trades": len(trades),
        "profit_factor": float(pf),
        "win_rate": float((trades["pnl"] > 0).mean()),
        "net_pnl": float(trades["pnl"].sum()),
        "avg_r": float(trades["r_multiple"].mean()),
        "max_dd": float(dd.max()) if len(dd) else 0.0,
        "sharpe_ratio": sharpe,  # per-trade Sharpe: mean(R) / std(R), not annualized
    }
