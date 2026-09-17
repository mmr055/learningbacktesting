"""
Dashboard: run with `streamlit run dashboard/app.py`.
Reads results/latest.json (written by src/pipeline.py) — no live compute
happens here, it's purely a viewer so it stays cheap to host.
"""
import json
import os
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="XAUUSD Evolutionary Strategy — Validation Dashboard", layout="wide")

RESULTS_PATH = "results/latest.json"

st.title("XAUUSD Evolutionary Strategy — Validation Dashboard")

if not os.path.exists(RESULTS_PATH):
    st.warning("No results yet. Run `python -m src.pipeline` first.")
    st.stop()

with open(RESULTS_PATH) as fh:
    report = json.load(fh)

verdict = report["verdict"]
badge = "✅ PASS" if verdict["passed"] else "❌ FAIL"
st.header(f"Run {report['run_id']} — {badge}")

cols = st.columns(5)
oos = report["oos_metrics"]
cols[0].metric("OOS Net PnL", f"${oos['net_pnl']:,.0f}")
cols[1].metric("Profit Factor", f"{oos['profit_factor']:.2f}")
cols[2].metric("Win Rate", f"{oos['win_rate']*100:.1f}%")
cols[3].metric("Max DD", f"${oos['max_dd']:,.0f}")
cols[4].metric("# OOS Trades", oos["n_trades"])

st.subheader("Pass/fail checks")
checks_df = pd.DataFrame([
    {"check": k, "passed": v} for k, v in verdict["checks"].items()
])
st.dataframe(checks_df, use_container_width=True)

st.subheader("Bootstrap edge significance")
boot = report["bootstrap"]
st.write(f"EV/trade: ${boot['ev_dollar_point']:.2f}  |  95% CI: {boot['ev_dollar_ci95']}  |  "
         f"P(EV ≤ 0) = {boot['p_ev_dollar_leq_0']:.4f}")

st.subheader("Monte Carlo — drawdown percentiles (realistic live risk budget)")
mc = report["monte_carlo"]["max_dd_percentiles"]
mc_df = pd.DataFrame({"percentile": list(mc.keys()), "max_dd": list(mc.values())})
fig = go.Figure(go.Bar(x=mc_df["percentile"].astype(str), y=mc_df["max_dd"]))
fig.update_layout(xaxis_title="Percentile", yaxis_title="Max Drawdown ($)")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Walk-forward stability")
wf = report["walk_forward"]
st.write(f"{wf['profitable_windows']}/{wf['n_windows']} rolling OOS windows profitable "
         f"({wf['profitable_window_pct']*100:.0f}%)")
wf_df = pd.DataFrame(wf["windows"])
if not wf_df.empty:
    st.dataframe(wf_df, use_container_width=True)

st.subheader("Slippage stress test")
st.json(report["slippage_stress"])

st.subheader("Frozen genome")
st.json(report["genome"])

st.subheader("Split boundary (audit trail)")
st.json(report["split_info"])
