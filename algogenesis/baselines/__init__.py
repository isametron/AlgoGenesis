from . import cmaes, de, ga, pso, random_search

BASELINES = {
    "PSO": pso.optimize,
    "GA": ga.optimize,
    "DE": de.optimize,
    "CMA-ES": cmaes.optimize,
    "BIPOP-CMA-ES": cmaes.optimize_bipop,
    "Random": random_search.optimize,
}
