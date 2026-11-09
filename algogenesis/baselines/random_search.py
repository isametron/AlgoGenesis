"""Random search (baseline): uniform sampling of the box. The floor any method should beat."""
import numpy as np


def optimize(func, dim, bounds, budget):
    lower, upper = (np.asarray(b, dtype=float) for b in bounds)
    best_f, best_x, history = np.inf, None, []
    for _ in range(budget):
        x = np.random.uniform(lower, upper)
        f = func(x)
        if f < best_f:
            best_f, best_x = f, x
        history.append(best_f)
    return best_x, best_f, history
