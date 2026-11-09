import numpy as np
import pytest

from algogenesis.baselines import BASELINES
from algogenesis.harness import score

ALL = sorted(BASELINES)
RUNS_TO_BUDGET = [name for name in ALL if name != "CMA-ES"]  # plain CMA-ES may stop early by design


@pytest.mark.parametrize("name", RUNS_TO_BUDGET)
def test_uses_exactly_the_budget(name):
    r = score(BASELINES[name], "bbob_f15", dim=5, budget=1500, seed=1, instance=4)
    assert r.ok, r.error
    assert r.evals == 1500


def test_cmaes_stays_within_budget():
    r = score(BASELINES["CMA-ES"], "bbob_f15", dim=5, budget=1500, seed=1, instance=4)
    assert r.ok, r.error
    assert 0 < r.evals <= 1500


@pytest.mark.parametrize("name", ALL)
def test_reproducible_with_a_seed(name):
    a = score(BASELINES[name], "bbob_f3", dim=5, budget=1000, seed=7, instance=4)
    b = score(BASELINES[name], "bbob_f3", dim=5, budget=1000, seed=7, instance=4)
    np.testing.assert_array_equal(a.history, b.history)


@pytest.mark.parametrize("name", ALL)
def test_different_seeds_differ(name):
    a = score(BASELINES[name], "bbob_f3", dim=5, budget=1000, seed=1, instance=4)
    b = score(BASELINES[name], "bbob_f3", dim=5, budget=1000, seed=2, instance=4)
    assert a.best_error != b.best_error


@pytest.mark.parametrize("name", ALL)
def test_stays_inside_the_box(name):
    r = score(BASELINES[name], "bbob_f8", dim=5, budget=1000, seed=3, instance=4)
    assert r.out_of_bounds == 0


def test_cma_family_solves_an_ill_conditioned_ellipsoid():
    """Published behaviour: CMA-ES handles f10 easily, where PSO and GA do not."""
    errors = {n: score(BASELINES[n], "bbob_f10", dim=5, budget=10_000, seed=0, instance=4).best_error
              for n in ("CMA-ES", "BIPOP-CMA-ES", "PSO", "Random")}
    assert errors["CMA-ES"] < 1e-6
    assert errors["BIPOP-CMA-ES"] < 1e-6
    assert errors["CMA-ES"] < errors["PSO"] < errors["Random"]


def test_bipop_keeps_going_after_solving_an_easy_function():
    r = score(BASELINES["BIPOP-CMA-ES"], "bbob_f1", dim=5, budget=10_000, seed=0, instance=4)
    assert r.ok, r.error
    assert r.evals == 10_000
    assert r.best_error < 1e-8
