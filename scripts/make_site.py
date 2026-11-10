"""Build the project explainer page, site/index.html, from the saved results.

    python scripts/make_site.py

Reads paper/stats.json (run scripts/make_paper_assets.py first),
results/test_phase/comparison.json and the LLM run logs in data/. Fills the
data placeholder in site/template.html and writes site/index.html.
No optimizer is run, and no number on the page is typed by hand.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from algogenesis.bbob import BBOB_GROUPS  # noqa: E402

GPT, QWEN = "gpt-oss-120b_A_run2", "qwen-3.8-27b_A_run1"  # same picks as make_paper_assets.py
SELECTED = ["BIPOP-CMA-ES", QWEN, GPT, "CMA-ES", "DE", "PSO", "GA", "Random"]
FUNCTIONS = [f"bbob_f{i}" for i in range(1, 25)]
CURVE_POINTS = 100

# Common names of the 24 BBOB functions (Hansen et al., 2009).
BBOB_NAMES = [
    "Sphere", "Ellipsoidal", "Rastrigin", "Büche-Rastrigin", "Linear slope",
    "Attractive sector", "Step ellipsoidal", "Rosenbrock", "Rosenbrock, rotated", "Ellipsoidal, rotated",
    "Discus", "Bent cigar", "Sharp ridge", "Different powers", "Rastrigin, rotated",
    "Weierstrass", "Schaffer F7", "Schaffer F7, ill-conditioned", "Griewank-Rosenbrock", "Schwefel",
    "Gallagher, 101 peaks", "Gallagher, 21 peaks", "Katsuura", "Lunacek bi-Rastrigin",
]


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def median_curve(runs):
    """Median log10 error across runs at each recorded evaluation count, thinned to CURVE_POINTS."""
    n = min(len(r["history_error"]) for r in runs)
    evals = runs[0]["history_evals"][:n]
    err = np.array([r["history_error"][:n] for r in runs], dtype=float)
    logerr = np.log10(np.clip(err, 1e-8, 1e2))
    med = np.median(logerr, axis=0)
    idx = np.unique(np.linspace(0, n - 1, min(n, CURVE_POINTS)).round().astype(int))
    return {"evals": [int(evals[i]) for i in idx], "log_error": [round(float(med[i]), 3) for i in idx]}


def functions_block(algos):
    group_of = {f"bbob_f{i}": g for g, ids in BBOB_GROUPS.items() for i in ids}
    out = []
    for k, fn in enumerate(FUNCTIONS):
        entry = {"id": fn, "number": k + 1, "name": BBOB_NAMES[k], "group": group_of[fn], "algorithms": {}}
        for name in SELECTED:
            runs = [r for r in algos[name]["runs"] if r["function"] == fn and r.get("history_error")]
            summary = next(s for s in algos[name]["summary"] if s["function"] == fn)
            entry["algorithms"][name] = {
                "aocc": round(float(np.mean([r["aocc"] for r in algos[name]["runs"] if r["function"] == fn])), 4),
                "solved": round(float(summary["success_rate"]), 4),
                "curve": median_curve(runs) if runs else None,
            }
        out.append(entry)
    return out


def runs_block(test_scores):
    out = []
    for path in sorted(glob.glob(str(ROOT / "data/results/llm_runs/*/*/run*.jsonl"))):
        rows = read_jsonl(path)
        first = rows[0]
        run_id = f"{first['model']}_{first['arm']}_run{first['run']}"
        best, calls, code = -1.0, [], {}
        for r in rows:
            fit = r.get("fitness")
            ok = r.get("error_type") is None and fit is not None
            improved = ok and fit > best
            if improved:
                best = fit
                code[str(r["call"])] = r.get("code") or ""
            err = (r.get("error") or "").strip().splitlines()
            calls.append({
                "call": r["call"],
                "name": r.get("name") or "",
                "fitness": round(fit, 4) if ok else None,
                "error_type": r.get("error_type"),
                "error": err[-1][:160] if err else "",
                "best": round(best, 4) if best >= 0 else None,
                "improved": improved,
                "tokens": (r.get("tokens") or {}).get("completion"),
                "latency_s": round(r.get("latency_s") or 0, 1),
            })
        out.append({
            "id": run_id, "model": first["model"], "arm": first["arm"], "run": first["run"],
            "calls": calls, "code": code, "test": test_scores.get(run_id),
        })
    return out


def task_prompt():
    """The plain task prompt (Arm B sends exactly this on every call)."""
    rows = read_jsonl(ROOT / "data/results/llm_runs/gpt-oss-120b/B/run1.jsonl")
    return rows[0]["prompt"]


def main():
    stats = json.loads((ROOT / "paper/stats.json").read_text(encoding="utf-8"))
    comparison = json.loads((ROOT / "results/test_phase/comparison.json").read_text(encoding="utf-8"))
    algos = {a["name"]: a for a in comparison["algorithms"]}
    test_scores = {p["program"]: p for p in stats["programs"]}

    data = {
        "settings": comparison["settings"],
        "selected": SELECTED,
        "picks": {"gpt": GPT, "qwen": QWEN},
        "overall": stats["overall"],
        "friedman": stats["friedman"],
        "pairwise": stats["pairwise"],
        "arms": stats["arms"],
        "calls": stats["calls"],
        "audit": stats["audit"],
        "programs": stats["programs"],
        "failed_in_test": stats["failed_in_test"],
        "groups": {g: list(ids) for g, ids in BBOB_GROUPS.items()},
        "functions": functions_block(algos),
        "runs": runs_block(test_scores),
        "task_prompt": task_prompt(),
    }
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    template = (ROOT / "site/template.html").read_text(encoding="utf-8")
    if "__DATA__" not in template:
        raise SystemExit("site/template.html has no __DATA__ placeholder")
    out = ROOT / "site/index.html"
    out.write_text(template.replace("__DATA__", payload), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
