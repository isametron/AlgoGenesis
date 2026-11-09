"""CMA-ES and BIPOP-CMA-ES (baselines), using pycma.

Both start from a uniformly random point with sigma0 = 25% of the box width
(2.5 on [-5, 5]) and keep samples inside the bounds with pycma's default
boundary handling. Settings are otherwise pycma's defaults.

Plain CMA-ES is a single run, so it may stop on its own criteria with budget
left over. BIPOP-CMA-ES restarts until the budget is spent. When the harness
raises BudgetExhausted the run simply ends; the harness keeps the recorded best.

pycma seeds numpy from the clock unless given a seed, so each call draws one
from the generator the harness has already seeded, keeping runs reproducible.
"""
import numpy as np


def _setup(func, dim, bounds):
    lower, upper = (np.asarray(b, dtype=float) for b in bounds)
    state = {"best_f": np.inf, "best_x": None, "history": []}

    def objective(x):
        f = func(x)
        if f < state["best_f"]:
            state["best_f"], state["best_x"] = f, np.array(x)
        state["history"].append(state["best_f"])
        return f

    def start():
        return np.random.uniform(lower, upper).tolist()

    options = {
        "bounds": [lower.tolist(), upper.tolist()],
        "seed": int(np.random.randint(1, 2**31 - 1)),
        "verbose": -9,
        "verb_disp": 0,
        "verb_log": 0,
    }
    return objective, start, 0.25 * float(np.mean(upper - lower)), options, state


def optimize(func, dim, bounds, budget):
    import cma

    objective, start, sigma0, options, state = _setup(func, dim, bounds)
    cma.fmin(objective, start(), sigma0, options=options)
    return state["best_x"], state["best_f"], state["history"]


def optimize_bipop(func, dim, bounds, budget, *, restarts=9):
    import cma

    objective, start, sigma0, options, state = _setup(func, dim, bounds)
    # One BIPOP schedule can finish early on easy functions, so run more until the budget is spent.
    while len(state["history"]) < budget:
        cma.fmin(objective, start, sigma0, options=dict(options), restarts=restarts, bipop=True)
        options["seed"] = int(np.random.randint(1, 2**31 - 1))
    return state["best_x"], state["best_f"], state["history"]
