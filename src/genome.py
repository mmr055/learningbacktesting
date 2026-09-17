"""
Strategy genome: a plain, JSON-serializable dict describing one candidate
strategy built from the blocks in src/blocks/. Kept deliberately simple
(no custom tree/AST) so every genome can be frozen to disk verbatim and
audited later.

Schema
------
{
  "direction": "long" | "short",
  "entry_block": "break_of_structure" | "opening_range",
  "entry_params": {...},
  "use_volume_filter": bool,
  "volume_params": {"window": int, "z_threshold": float},
  "use_mtf_filter": bool,
  "mtf_params": {"htf": "1H"|"4H"|"1D", "mode": "trend_bias"|"confluence", "width": int},
  "stop_atr_mult": float,
  "tp_r_multiple": float,
  "risk_pct": float,
  "max_bars_in_trade": int
}
"""
from __future__ import annotations
import random
import copy

HTF_CHOICES = ["1h", "4h", "1D"]
ENTRY_BLOCKS = ["break_of_structure", "opening_range"]


def random_genome(rng: random.Random | None = None) -> dict:
    r = rng or random
    entry_block = r.choice(ENTRY_BLOCKS)
    if entry_block == "break_of_structure":
        entry_params = {"width": r.choice([2, 3, 4, 5])}
    else:
        entry_params = {
            "session_start_hour": r.choice([7, 8, 9, 13]),  # UTC hours
            "duration_min": r.choice([15, 30, 60]),
        }

    return {
        "direction": r.choice(["long", "short"]),
        "entry_block": entry_block,
        "entry_params": entry_params,
        "use_volume_filter": r.random() < 0.6,
        "volume_params": {"window": r.choice([20, 50, 100]), "z_threshold": round(r.uniform(1.0, 2.5), 2)},
        "use_mtf_filter": r.random() < 0.6,
        "mtf_params": {
            "htf": r.choice(HTF_CHOICES),
            "mode": r.choice(["trend_bias", "confluence"]),
            "width": r.choice([2, 3]),
        },
        "stop_atr_mult": round(r.uniform(0.5, 3.0), 2),
        "tp_r_multiple": round(r.uniform(0.8, 3.0), 2),
        "risk_pct": round(r.uniform(0.5, 2.0), 2),
        "max_bars_in_trade": r.choice([50, 100, 200, 400]),
    }


def mutate(genome: dict, rng: random.Random | None = None, rate: float = 0.3) -> dict:
    r = rng or random
    g = copy.deepcopy(genome)
    if r.random() < rate:
        g["direction"] = r.choice(["long", "short"])
    if r.random() < rate:
        fresh = random_genome(r)
        g["entry_block"], g["entry_params"] = fresh["entry_block"], fresh["entry_params"]
    if r.random() < rate:
        g["use_volume_filter"] = not g["use_volume_filter"]
    if r.random() < rate:
        g["volume_params"]["z_threshold"] = round(max(0.5, g["volume_params"]["z_threshold"] + r.uniform(-0.5, 0.5)), 2)
    if r.random() < rate:
        g["use_mtf_filter"] = not g["use_mtf_filter"]
    if r.random() < rate:
        g["mtf_params"]["htf"] = r.choice(HTF_CHOICES)
    if r.random() < rate:
        g["stop_atr_mult"] = round(max(0.2, g["stop_atr_mult"] + r.uniform(-0.5, 0.5)), 2)
    if r.random() < rate:
        g["tp_r_multiple"] = round(max(0.3, g["tp_r_multiple"] + r.uniform(-0.5, 0.5)), 2)
    if r.random() < rate:
        g["risk_pct"] = round(min(3.0, max(0.1, g["risk_pct"] + r.uniform(-0.3, 0.3))), 2)
    return g


def crossover(a: dict, b: dict, rng: random.Random | None = None) -> dict:
    r = rng or random
    child = {}
    for key in a.keys():
        child[key] = copy.deepcopy(r.choice([a[key], b[key]]))
    return child
