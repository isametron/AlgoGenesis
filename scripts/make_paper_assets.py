"""Generate every statistic, table and figure in paper/ from the saved results.

    python scripts/make_paper_assets.py

Reads results/test_phase/comparison.json and data/. Writes paper/stats.json,
paper/tables/*.tex and paper/figures/*.pdf. No optimizer is run.
"""
import collections
import glob
import json
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import friedmanchisquare, mannwhitneyu, rankdata, wilcoxon

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from algogenesis.bbob import BBOB_GROUPS  # noqa: E402

PAPER = ROOT / "paper"
GPT, QWEN = "gpt-oss-120b_A_run2", "qwen-3.8-27b_A_run1"  # highest training fitness for each model
BASELINES = ["BIPOP-CMA-ES", "CMA-ES", "DE", "PSO", "GA", "Random"]
LABEL = {GPT: "gpt-oss A2", QWEN: "Qwen A1"}
SELECTED = ["BIPOP-CMA-ES", QWEN, GPT, "CMA-ES", "DE", "PSO", "GA", "Random"]
FUNCTIONS = [f"bbob_f{i}" for i in range(1, 25)]
NEMENYI_Q_8 = 3.031  # two-tailed alpha = 0.05, k = 8 (Demsar 2006)

# Okabe-Ito hues, designed to stay distinct under colour-vision deficiency; line styles double the encoding.
STYLE = {
    "BIPOP-CMA-ES": ("#0072B2", "-"), QWEN: ("#D55E00", "-"), GPT: ("#CC79A7", "--"),
    "CMA-ES": ("#56B4E9", "--"), "DE": ("#009E73", ":"), "PSO": ("#E69F00", "-."),
    "GA": ("#7F7F7F", ":"), "Random": ("#000000", "-."),
}


def holm(ps):
    order, out, running = np.argsort(ps), np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        out[i] = running
    return out


def a12(x, y):
    return float(mannwhitneyu(x, y, alternative="two-sided").statistic) / (len(x) * len(y))


def exact(a, b):
    return float(mannwhitneyu(a, b, alternative="two-sided", method="exact").pvalue)


def label(name):
    return LABEL.get(name, name)


def fmt_p(p):
    return r"$<$0.001" if p < 0.001 else f"{p:.3f}"


def main():
    PAPER.joinpath("figures").mkdir(parents=True, exist_ok=True)
    PAPER.joinpath("tables").mkdir(parents=True, exist_ok=True)
    data = json.loads((ROOT / "results/test_phase/comparison.json").read_text(encoding="utf-8"))
    algos = {a["name"]: a for a in data["algorithms"]}
    programs = [n for n in algos if "_run" in n]

    def per_fn(name):
        groups = collections.defaultdict(list)
        for r in algos[name]["runs"]:
            groups[r["function"]].append(r["aocc"])
        return np.array([np.mean(groups[f]) for f in FUNCTIONS])

    def run_aocc(name):
        return np.array([r["aocc"] for r in algos[name]["runs"]])

    stats = {"failed_in_test": data["failed"]}

    # ---- discovered programs ------------------------------------------------
    rows = []
    for path in sorted(glob.glob(str(ROOT / "data/found/*.py"))):
        stem = Path(path).stem
        header = Path(path).read_text(encoding="utf-8").splitlines()[0]
        match = re.search(r"training fitness ([\d.]+) \(call (\d+) of (\d+)\); name: (.*)", header)
        a = algos.get(stem)
        rows.append({
            "program": stem, "name": match[4], "train": float(match[1]), "best_call": int(match[2]),
            "test": a["fitness"] if a else None,
            "crashed_runs": sum(not r["ok"] for r in a["runs"]) if a else None,
            "solved": float(np.mean([r["best_error"] <= 1e-8 for r in a["runs"]])) if a else None,
        })
    stats["programs"] = rows
    # From each program's self-reported name and a keyword scan of its source; not a behavioural analysis.
    family = {
        "gpt-oss-120b_A_run1": "DE + CMA + pattern search", "gpt-oss-120b_A_run2": "jDE + Nelder--Mead + (1+1)-ES",
        "gpt-oss-120b_A_run3": r"L\'evy + CMA memetic", "gpt-oss-120b_A_run4": "DE + PSO hybrid",
        "gpt-oss-120b_A_run5": "Sobol + CMA + DE", "gpt-oss-120b_B_run1": "Self-adaptive DE",
        "gpt-oss-120b_B_run2": "Hybrid DE", "gpt-oss-120b_B_run3": "Adaptive-mean ES",
        "gpt-oss-120b_B_run4": "Adaptive DE", "gpt-oss-120b_B_run5": "Adaptive DE",
        "qwen-3.8-27b_A_run1": "CMA-ES + restarts", "qwen-3.8-27b_A_run2": "IPOP-CMA-ES",
        "qwen-3.8-27b_A_run3": "IPOP-CMA-ES", "qwen-3.8-27b_A_run4": "CMA-ES + restarts",
    }
    lines = [r"\begin{tabular}{llcc}", r"\toprule", r"Program & Reported design & Train & Test \\", r"\midrule"]
    for r in rows:
        tag = re.sub(r"_run", "", r["program"]).replace("gpt-oss-120b_", "gpt-oss ").replace("qwen-3.8-27b_", "Qwen ")
        test = "hung" if r["test"] is None else f"{r['test']:.3f}" + ("$^\\dagger$" if r["crashed_runs"] else "")
        train = f"{r['train']:.3f}"
        lines.append(f"{tag} & {family[r['program']]} & {train} & {test}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (PAPER / "tables/programs.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- overall and per-group AOCC ----------------------------------------
    groups = {g: [i - 1 for i in fs] for g, fs in BBOB_GROUPS.items()}
    short = ["Sep.", "Mod.", "Ill-c.", "M-str.", "M-weak"]
    lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
             "Algorithm & " + " & ".join(short) + r" & All \\", r"\midrule"]
    overall = {}
    for name in SELECTED:
        f = per_fn(name)
        overall[name] = {"aocc": algos[name]["fitness"],
                         "solved": float(np.mean([r["best_error"] <= 1e-8 for r in algos[name]["runs"]])),
                         "evals": float(np.mean([r["evals"] for r in algos[name]["runs"]])),
                         "groups": {g: float(f[i].mean()) for g, i in groups.items()}}
        cells = [f"{f[i].mean():.3f}" for i in groups.values()] + [f"{algos[name]['fitness']:.3f}"]
        lines.append(f"{label(name)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (PAPER / "tables/groups.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    stats["overall"] = overall

    # ---- Friedman, Nemenyi, Holm-corrected Wilcoxon ------------------------
    matrix = np.array([per_fn(n) for n in SELECTED]).T
    chi2, p = friedmanchisquare(*matrix.T)
    ranks = rankdata(-matrix, axis=1).mean(axis=0)
    k, n = len(SELECTED), len(FUNCTIONS)
    stats["friedman"] = {"chi2": float(chi2), "p": float(p), "df": k - 1,
                         "avg_rank": dict(zip(SELECTED, map(float, ranks))),
                         "nemenyi_cd": float(NEMENYI_Q_8 * np.sqrt(k * (k + 1) / (6 * n)))}

    tests = [(llm, b) for llm in (QWEN, GPT) for b in BASELINES]
    raw = [float(wilcoxon(per_fn(a), per_fn(b)).pvalue) for a, b in tests]
    adj = holm(raw)
    pairs = {f"{a}|{b}": {"wins": int((per_fn(a) > per_fn(b)).sum()), "p": r, "p_holm": float(h),
                          "a12": a12(run_aocc(a), run_aocc(b))}
             for (a, b), r, h in zip(tests, raw, adj)}
    stats["pairwise"] = pairs
    lines = [r"\begin{tabular}{lccc|ccc}", r"\toprule",
             r" & \multicolumn{3}{c|}{Qwen A1} & \multicolumn{3}{c}{gpt-oss A2} \\",
             r"Baseline & W/24 & $p_H$ & $\hat{A}_{12}$ & W/24 & $p_H$ & $\hat{A}_{12}$ \\", r"\midrule"]
    for b in BASELINES:
        q, g = pairs[f"{QWEN}|{b}"], pairs[f"{GPT}|{b}"]
        lines.append(f"{b} & {q['wins']} & {fmt_p(q['p_holm'])} & {q['a12']:.2f} & "
                     f"{g['wins']} & {fmt_p(g['p_holm'])} & {g['a12']:.2f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (PAPER / "tables/pairwise.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- loop versus control (gpt-oss-120b) --------------------------------
    by_stem = {r["program"]: r for r in rows}
    arm = lambda a, key: [by_stem[f"gpt-oss-120b_{a}_run{i}"][key] for i in range(1, 6)]
    tr_a, tr_b, te_a, te_b = arm("A", "train"), arm("B", "train"), arm("A", "test"), arm("B", "test")
    rng = np.random.default_rng(0)
    boot = [rng.choice(te_a, 5).mean() - rng.choice(te_b, 5).mean() for _ in range(10_000)]
    mean_a, mean_b = np.mean([per_fn(f"gpt-oss-120b_A_run{i}") for i in range(1, 6)], axis=0), \
        np.mean([per_fn(f"gpt-oss-120b_B_run{i}") for i in range(1, 6)], axis=0)

    # ---- per-call log statistics -------------------------------------------
    calls = {}
    valid_means = collections.defaultdict(list)
    for path in sorted(glob.glob(str(ROOT / "data/results/llm_runs/*/*/run*.jsonl"))):
        p = Path(path)
        model, a = p.parts[-3], p.parts[-2]
        log = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]
        c = calls.setdefault(f"{model}/{a}", collections.Counter())
        c["runs"] += 1
        c["calls"] += len(log)
        for r in log:
            c[f"err_{r.get('error_type')}"] += 1
            c["prompt_tokens"] += (r["tokens"] or {}).get("prompt", 0) or 0
            c["completion_tokens"] += (r["tokens"] or {}).get("completion", 0) or 0
        valid_means[f"{model}/{a}"].append(float(np.mean([r["fitness"] for r in log if r.get("error_type") is None])))
    stats["calls"] = {k: dict(v) for k, v in calls.items()}
    stats["valid_mean_fitness_per_run"] = valid_means
    va, vb = valid_means["gpt-oss-120b/A"], valid_means["gpt-oss-120b/B"]
    stats["arms"] = {
        "train_A": tr_a, "train_B": tr_b, "test_A": te_a, "test_B": te_b,
        "train_mean": [float(np.mean(tr_a)), float(np.mean(tr_b))],
        "test_mean": [float(np.mean(te_a)), float(np.mean(te_b))],
        "train_p": exact(tr_a, tr_b), "test_p": exact(te_a, te_b),
        "train_a12": a12(tr_a, tr_b), "test_a12": a12(te_a, te_b),
        "test_diff_boot95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "functions_A_higher": int((mean_a > mean_b).sum()),
        "valid_mean_A": float(np.mean(va)), "valid_mean_B": float(np.mean(vb)),
        "valid_p": exact(va, vb), "valid_a12": a12(va, vb),
    }

    # ---- exploit-audit counts ----------------------------------------------
    stats["audit"] = {"programs_tested": len(programs),
                      "out_of_bounds_evals": sum(r["out_of_bounds"] for n in programs for r in algos[n]["runs"]),
                      "program_flags": sum(len(algos[n]["flags"]) for n in programs)}

    # ---- figures -----------------------------------------------------------
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.5,
                         "axes.axisbelow": True})

    grid = np.unique(np.logspace(0, 4, 200).astype(int))
    targets = 10 ** np.linspace(2, -8, 51)
    fig, ax = plt.subplots(figsize=(3.45, 2.5))
    for name in SELECTED:
        reached = np.zeros(len(grid))
        for r in algos[name]["runs"]:
            ev, err = np.array(r["history_evals"]), np.array(r["history_error"])
            idx = np.searchsorted(ev, grid, side="right") - 1
            cur = np.where(idx >= 0, err[idx.clip(0)], np.inf)
            reached += (cur[:, None] <= targets).mean(axis=1)
        colour, ls = STYLE[name]
        ax.plot(grid / 5, reached / len(algos[name]["runs"]), color=colour, ls=ls, lw=1.3, label=label(name))
    ax.set_xscale("log")
    ax.set_xlim(0.2, 2000)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Function evaluations / dimension")
    ax.set_ylabel("Fraction of targets reached")
    ax.legend(frameon=False, fontsize=6.3, loc="upper left", ncol=2, handlelength=2.6, columnspacing=0.8)
    fig.tight_layout(pad=0.3)
    fig.savefig(PAPER / "figures/fig_ecdf.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.45, 2.1))
    qwen = [by_stem[f"qwen-3.8-27b_A_run{i}"]["test"] for i in (1, 2, 4)]
    groups_plot = [("gpt-oss\nloop (A)", te_a, "#D55E00", "o"), ("gpt-oss\ncontrol (B)", te_b, "#0072B2", "s"),
                   ("Qwen\nloop (A)", qwen, "#009E73", "^")]
    rng = np.random.default_rng(1)
    for i, (_, values, colour, marker) in enumerate(groups_plot):
        ax.scatter(i + rng.uniform(-0.09, 0.09, len(values)), values, s=26, c=colour, marker=marker,
                   edgecolors="white", linewidths=0.6, zorder=3)
        ax.plot([i - 0.2, i + 0.2], [np.mean(values)] * 2, color="#333333", lw=1.2, zorder=4)
    for name in ("BIPOP-CMA-ES", "CMA-ES", "DE", "PSO"):
        v = algos[name]["fitness"]
        ax.axhline(v, color="#8a8a8a", lw=0.7, ls="--", zorder=1)
        ax.text(2.97, v + 0.004, name, ha="right", fontsize=6.3, color="#444444")
    ax.set_xticks(range(3), [g[0] for g in groups_plot])
    ax.set_xlim(-0.5, 3.0)
    ax.set_ylim(0.3, 0.62)
    ax.set_ylabel("Held-out test AOCC")
    ax.grid(axis="x", visible=False)
    fig.tight_layout(pad=0.3)
    fig.savefig(PAPER / "figures/fig_arms.pdf")
    plt.close(fig)

    (PAPER / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
