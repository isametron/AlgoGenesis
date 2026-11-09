"""Differential Evolution, DE/rand/1/bin (baseline).

Population 10 x dim, F = 0.5, CR = 0.9. Mutants are clipped to the bounds, and
a trial replaces its parent when it is at least as good. Each generation's
trials are built in one vectorised step, as in ga.py.
"""
import numpy as np


def optimize(func, dim, bounds, budget, *, pop_factor=10, F=0.5, CR=0.9):
    lower, upper = (np.asarray(b, dtype=float) for b in bounds)
    n = max(4, min(pop_factor * dim, budget))  # rand/1 needs the parent plus 3 others

    pop = lower + np.random.rand(n, dim) * (upper - lower)
    fit = np.full(n, np.inf)
    best_f, best_x, history = np.inf, None, []

    def evaluate(x):
        nonlocal best_f, best_x
        f = func(x)
        if f < best_f:
            best_f, best_x = f, x.copy()
        history.append(best_f)
        return f

    for i in range(min(n, budget)):
        fit[i] = evaluate(pop[i])
    evals = min(n, budget)

    while evals < budget:
        # Three distinct donors per parent, all different from the parent.
        donors = np.argsort(np.random.rand(n, n - 1), axis=1)[:, :3]
        donors += donors >= np.arange(n)[:, None]
        mutants = pop[donors[:, 0]] + F * (pop[donors[:, 1]] - pop[donors[:, 2]])

        # Binomial crossover; one random gene per trial always comes from the mutant.
        cross = np.random.rand(n, dim) < CR
        cross[np.arange(n), np.random.randint(dim, size=n)] = True
        trials = np.clip(np.where(cross, mutants, pop), lower, upper)

        for i in range(min(n, budget - evals)):
            f = evaluate(trials[i])
            evals += 1
            if f <= fit[i]:
                pop[i], fit[i] = trials[i], f

    return best_x, best_f, history
