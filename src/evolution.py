"""
Evolutionary search over strategy genomes.

HARD RULE: this module only ever imports/receives the in-sample DataFrame.
It must never import src.data.loader.load_oos_window. That import boundary
is what makes the "we never tuned against OOS" claim auditable rather than
just a promise.
"""
from __future__ import annotations
import random
import json
import os
from dataclasses import dataclass, field

import pandas as pd

from src.genome import random_genome, mutate, crossover
from src.backtest import run_backtest
from src.fitness import multi_window_fitness


@dataclass
class Individual:
    genome: dict
    fitness_report: dict = field(default_factory=dict)

    @property
    def score(self) -> float:
        return self.fitness_report.get("fitness", 0.0)


def evaluate(genome: dict, insample_df: pd.DataFrame, n_windows: int = 3) -> Individual:
    try:
        report = multi_window_fitness(genome, insample_df, n_windows=n_windows)
    except Exception as e:  # a malformed mutated genome should not crash the run
        report = {"fitness": 0.0, "reason": f"error: {e}", "n_trades_total": 0}
    return Individual(genome=genome, fitness_report=report)


def tournament_select(pop: list[Individual], k: int = 3, rng: random.Random | None = None) -> Individual:
    r = rng or random
    contestants = r.sample(pop, k)
    return max(contestants, key=lambda ind: ind.score)


def evolve(
    insample_df: pd.DataFrame,
    population_size: int = 60,
    generations: int = 40,
    elite_frac: float = 0.1,
    mutation_rate: float = 0.3,
    seed: int | None = None,
    log_path: str | None = None,
) -> list[Individual]:
    rng = random.Random(seed)
    population = [evaluate(random_genome(rng), insample_df) for _ in range(population_size)]
    history = []

    for gen in range(generations):
        population.sort(key=lambda ind: ind.score, reverse=True)
        best = population[0]
        avg = sum(ind.score for ind in population) / len(population)
        history.append({"generation": gen, "best_fitness": best.score, "avg_fitness": avg,
                         "best_genome": best.genome, "best_report": best.fitness_report})
        print(f"gen {gen:3d} | best={best.score:.4f} avg={avg:.4f} "
              f"trades={best.fitness_report.get('n_trades_total')}")

        n_elite = max(1, int(population_size * elite_frac))
        next_pop = population[:n_elite]  # elitism: carry best forward unmutated

        while len(next_pop) < population_size:
            parent_a = tournament_select(population, rng=rng)
            parent_b = tournament_select(population, rng=rng)
            child_genome = crossover(parent_a.genome, parent_b.genome, rng)
            child_genome = mutate(child_genome, rng, rate=mutation_rate)
            next_pop.append(evaluate(child_genome, insample_df))

        population = next_pop

    population.sort(key=lambda ind: ind.score, reverse=True)

    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as fh:
            json.dump(history, fh, indent=2, default=str)

    return population
