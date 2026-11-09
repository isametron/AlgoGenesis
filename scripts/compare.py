"""Full comparison: baselines and LLM-found algorithms on every benchmark.

    python scripts/compare.py                                   # all baselines, classic suite
    python scripts/compare.py --candidates data/found/*.py      # add LLM-found algorithms
    python scripts/compare.py --dims 10 30 --runs 30 --reference GA
    python scripts/compare.py --suite bbob --candidates data/found/*.py   # the paper's test phase

--suite sets the defaults, and any flag given explicitly overrides them:
    classic   Sphere and Rastrigin, 10-D and 30-D, 30 runs, budget 1000 x dim
    bbob      all 24 BBOB functions, 5-D, 15 runs (one per test instance), budget 2000 x dim
Runs use the held-out test instances unless --instances train is given.

Each candidate is a .py file defining optimize(func, dim, bounds, budget); its
file name is used as the algorithm name. Candidates run in a separate process
with a time limit, so a broken one is reported and skipped instead of stopping
the comparison.

Outputs (default results/comparison/):
    tables.md                  final error + significance, success/AOCC, compute cost
    table_error.tex            final-error table for the paper (booktabs)
    convergence_<fn>_d<dim>.png
    comparison.json            every run, with subsampled convergence histories
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algogenesis.baselines import BASELINES  # noqa: E402
from algogenesis.bbob import BBOB_FUNCTIONS  # noqa: E402
from algogenesis.benchmarks import BENCHMARKS  # noqa: E402
from algogenesis.harness import TEST_INSTANCES, TRAIN_INSTANCES, evaluate_code, run_suite  # noqa: E402
from algogenesis.report import (  # noqa: E402
    comparison_rows, latex_error_table, markdown_tables, plot_convergence, problems,
)


SUITES = {
    "classic": dict(functions=list(BENCHMARKS), dims=[10, 30], runs=30, budget_per_dim=1000),
    "bbob": dict(functions=list(BBOB_FUNCTIONS), dims=[5], runs=len(TEST_INSTANCES), budget_per_dim=2000),
}
INSTANCES = {"test": TEST_INSTANCES, "train": TRAIN_INSTANCES}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", nargs="*", type=Path, default=[], help="LLM-found algorithm files")
    parser.add_argument("--baselines", nargs="*", default=list(BASELINES), choices=list(BASELINES))
    parser.add_argument("--suite", choices=list(SUITES), default="classic", help="sets the four defaults below")
    parser.add_argument("--functions", nargs="+", choices=list(BENCHMARKS) + list(BBOB_FUNCTIONS))
    parser.add_argument("--dims", nargs="+", type=int)
    parser.add_argument("--runs", type=int)
    parser.add_argument("--budget-per-dim", type=int)
    parser.add_argument("--instances", choices=list(INSTANCES), default="test")
    parser.add_argument("--reference", help="algorithm the significance tests compare against "
                                            "(default: first candidate, or first baseline if there are none)")
    parser.add_argument("--timeout", type=float, default=3600, help="seconds allowed per candidate")
    parser.add_argument("--out", type=Path, default=Path("results/comparison"))
    args = parser.parse_args()
    for key, value in SUITES[args.suite].items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    suite_kwargs = dict(functions=args.functions, dims=args.dims, n_runs=args.runs,
                        budget_per_dim=args.budget_per_dim, instances=INSTANCES[args.instances])
    suites, failed = [], {}

    for path in args.candidates:
        print(f"Running candidate {path.stem} ...", flush=True)
        t0 = time.perf_counter()
        suite = evaluate_code(path.read_text(encoding="utf-8"), name=path.stem, timeout=args.timeout,
                              **suite_kwargs)
        if not suite.error and suite.runs and not any(r.ok for r in suite.runs):
            suite.error = "every run crashed; first error:\n" + suite.runs[0].error
        if suite.error:
            failed[path.stem] = suite.error
            print(f"  FAILED, skipping:\n{suite.error}", flush=True)
        else:
            suites.append(suite)
            print(f"  done in {time.perf_counter() - t0:.0f} s", flush=True)

    for name in args.baselines:
        print(f"Running baseline {name} ...", flush=True)
        t0 = time.perf_counter()
        suites.append(run_suite(BASELINES[name], name=name, **suite_kwargs))
        print(f"  done in {time.perf_counter() - t0:.0f} s", flush=True)

    if not suites:
        sys.exit("Nothing to compare: every algorithm failed.")
    names = [s.name for s in suites]
    if len(set(names)) != len(names):
        sys.exit(f"Algorithm names must be unique, got {names}")
    reference = args.reference or names[0]

    rows = comparison_rows(suites, reference)
    settings = {**suite_kwargs, "reference": reference}
    args.out.mkdir(parents=True, exist_ok=True)

    md = markdown_tables(rows, reference)
    header = (f"# Comparison results\n\nSettings: {settings}. Shifted problem instances; "
              f"every algorithm gets {args.budget_per_dim} × dim evaluations per run.\n\n")
    if failed:
        header += "Candidates that failed to run and are left out: " + ", ".join(failed) + "\n\n"
    (args.out / "tables.md").write_text(header + md, encoding="utf-8")
    (args.out / "table_error.tex").write_text(latex_error_table(rows, reference), encoding="utf-8")
    (args.out / "comparison.json").write_text(json.dumps({
        "settings": settings,
        "rows": rows,
        "failed": failed,
        "algorithms": [s.to_dict() for s in suites],
    }, indent=2, default=float), encoding="utf-8")
    for fn, dim in problems(suites):
        plot_convergence(suites, fn, dim, args.out / f"convergence_{fn}_d{dim}.png")

    print()
    print(md)
    print(f"Saved to {args.out.resolve()}")


if __name__ == "__main__":
    main()
