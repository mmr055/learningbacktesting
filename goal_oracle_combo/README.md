# GOAL: hindsight oracle ceiling (NOT tradable)
Combo label = efficiency bottom 33% OR day close disagrees with daily trend (same-day hindsight, includes outcome leakage).
Run: python -m src.tv_oracle_r  (kind="combo"; needs trades_*.csv from tv_strategy)
Results (rr1.5, kept PF gross): 30m 2.16 (2005-20) / 2.18 (2021+); 15m 2.29 / 2.55; 1h 1.69 / 1.74; 4h unusable.
Net of 0.30 cost: 30m 1.99 / 2.10; 15m 2.05 / 2.40.
Purpose: target to approximate with entry-time-only signals (running efficiency, running agreement, later GEX/news/discretion).
Files: tv_oracle.py (original efficiency oracle, imported by tv_oracle_r.py), tv_oracle_r.py (labels volband/trendagree/combo).
