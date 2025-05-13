import csv
import math
import random
import os
from glob import glob
from copy import deepcopy
from typing import Dict, List, Tuple, Callable

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _round_to_nearest(value: float, precision: float = 0.0001) -> float:
    """Round a float to the nearest precision step (default 1e-4)."""
    return round(value / precision) * precision


def _build_bounds(initial_params: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
    """Generate default optimisation bounds from an initial params dict.

    For each numeric parameter x:
        * if x == 0 -> (-1.0, 1.0)
        * else      -> (x * 0.5, x * 1.5)
    Bounds are widened if upper == lower.
    """
    bounds: Dict[str, Tuple[float, float]] = {}
    for k, v in initial_params.items():
        if not isinstance(v, (int, float)):
            # Skip non-numeric parameters – optimiser only handles floats/ints.
            continue
        if v == 0:
            lower, upper = -1.0, 1.0
        else:
            lower, upper = v * 0.5, v * 1.5
            # Ensure ordering
            if lower > upper:
                lower, upper = upper, lower
            # If bounds collapse, widen slightly
            if abs(upper - lower) < 1e-9:
                lower -= abs(v) * 0.1 or 0.1
                upper += abs(v) * 0.1 or 0.1
        bounds[k] = (lower, upper)
    return bounds


def _generate_random_population(bounds: Dict[str, Tuple[float, float]], size: int = 10) -> List[Dict[str, float]]:
    """Generate a list of random parameter dicts inside provided bounds."""
    population: List[Dict[str, float]] = []
    for _ in range(size):
        individual = {}
        for k, (low, high) in bounds.items():
            individual[k] = _round_to_nearest(random.uniform(low, high))
        population.append(individual)
    return population


def _mutate_from_best(best: Dict[str, float], bounds: Dict[str, Tuple[float, float]], generation: int, size: int = 10) -> List[Dict[str, float]]:
    """Create a new population around *best* with shrinking variance each generation."""
    variance = 0.1 / (generation + 1)  # decrease variance over time
    next_pop: List[Dict[str, float]] = []
    rng = random.Random()
    for _ in range(size):
        mutant: Dict[str, float] = {}
        for k, (low, high) in bounds.items():
            centre = best[k]
            span = high - low
            val = centre + rng.gauss(0, variance) * span
            # Clamp to bounds
            val = max(low, min(high, val))
            mutant[k] = _round_to_nearest(val)
        next_pop.append(mutant)
    return next_pop


def _load_close_prices(path: str) -> List[float]:
    """Load the Close column from a CSV file into a simple list (no pandas)."""
    closes: List[float] = []
    with open(path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                closes.append(float(row["Close"]))
            except (KeyError, ValueError):
                continue
    return closes

# -----------------------------------------------------------------------------
# Core evaluation helpers
# -----------------------------------------------------------------------------

def _safe_execute_trading_strategy(strategy_fn: Callable, params: Dict[str, float], close_prices: List[float]) -> float:
    """Run *strategy_fn* safely on *close_prices* with *params*.

    Returns profit (balance - initial_balance). If execution fails or output is
    invalid, a large negative sentinel value is returned so the optimiser will
    ignore this parameter set.
    """
    # Backup any pre-existing global *close* to restore later
    backup_close = globals().get("close", None)
    globals()["close"] = close_prices  # Provide to strategy as global list

    try:
        result = strategy_fn(params)
        if not (isinstance(result, tuple) and len(result) >= 1):
            raise ValueError("strategy_fn must return a tuple where first element is profit.")
        profit = result[0]
        if not isinstance(profit, (int, float)):
            raise ValueError("Profit part of return value is not numeric.")
        return float(profit)
    except Exception:
        # Silently penalise failures
        return -1e12  # big negative number
    finally:
        # Restore original close if it existed
        if backup_close is not None:
            globals()["close"] = backup_close
        else:
            globals().pop("close", None)


def _evaluate_population(population: List[Dict[str, float]], strategy_fn: Callable, datasets: List[str]) -> List[Tuple[Dict[str, float], float]]:
    """Compute total profit across *datasets* for every param set in *population*."""
    evaluations: List[Tuple[Dict[str, float], float]] = []
    for params in population:
        total_profit = 0.0
        for path in datasets:
            close_prices = _load_close_prices(path)
            total_profit += _safe_execute_trading_strategy(strategy_fn, params, close_prices)
        evaluations.append((params, total_profit))
    return evaluations

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def optimise_parameters(strategy_fn: Callable, initial_params: Dict[str, float], datasets_glob: str = "stockbt/datasets/*.csv", generations: int = 8, population_size: int = 10) -> Tuple[Dict[str, float], float]:
    """Perform a simple evolutionary search to optimise *strategy_fn* parameters.

    Params
    ------
    strategy_fn : Callable
        Reference to the *trading_strategy* function.
    initial_params : dict
        The parameter dictionary produced by get_user_params(). Used to derive
        search space bounds and to seed the first generation.
    datasets_glob : str
        Glob pattern pointing at CSV price datasets.
    generations : int
        Number of optimisation generations to run.
    population_size : int
        Number of individuals per generation.

    Returns
    -------
    (best_params, best_profit)
    """
    dataset_paths = glob(datasets_glob)
    if not dataset_paths:
        raise FileNotFoundError(f"No datasets found matching pattern: {datasets_glob}")

    bounds = _build_bounds(initial_params)

    # Start population: initial_params mutated plus random individuals
    population: List[Dict[str, float]] = [_round_params(initial_params)]
    population += _generate_random_population(bounds, size=population_size - 1)

    best_params = deepcopy(population[0])
    best_profit = float("-inf")

    for gen in range(generations):
        evaluations = _evaluate_population(population, strategy_fn, dataset_paths)
        evaluations.sort(key=lambda x: x[1], reverse=True)

        gen_best_params, gen_best_profit = evaluations[0]
        if gen_best_profit > best_profit:
            best_profit = gen_best_profit
            best_params = deepcopy(gen_best_params)

        print(f"Generation {gen+1}/{generations}: Best Profit = {gen_best_profit:.2f} with {gen_best_params}")

        # Produce next generation around current best
        population = _mutate_from_best(gen_best_params, bounds, gen, size=population_size)

    print("\nOptimisation complete.")
    print(f"Best Profit Overall: {best_profit:.2f}")
    print(f"Best Parameters    : {best_params}")
    return best_params, best_profit

# -----------------------------------------------------------------------------
# Internal util
# -----------------------------------------------------------------------------

def _round_params(params: Dict[str, float]) -> Dict[str, float]:
    """Return a copy of params with each value rounded using _round_to_nearest."""
    return {k: _round_to_nearest(v) if isinstance(v, (int, float)) else v for k, v in params.items()} 