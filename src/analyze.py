"""Analysis: descriptive stats, paired significance tests, effect sizes, diversity, cost, and
figures. Writes results/summary.json and figures/*.png."""
import json, os, math
from collections import defaultdict, Counter
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

np.random.seed(42)
ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
CONDS = ["plain", "cot", "simtom", "persentence"]
LABELS = {"plain": "Plain", "cot": "Generic CoT", "simtom": "SimToM (once)",
          "persentence": "Per-sentence sim\n(intervention)"}


def load(name):
    with open(os.path.join(RES, name)) as f:
        return [json.loads(l) for l in f]


def bootstrap_ci(x, fn=np.mean, n=10000, alpha=0.05):
    x = np.asarray(x, float)
    bs = [fn(np.random.choice(x, len(x), replace=True)) for _ in range(n)]
    return float(fn(x)), float(np.percentile(bs, 100 * alpha / 2)), float(np.percentile(bs, 100 * (1 - alpha / 2)))


def cohen_d_paired(diff):
    diff = np.asarray(diff, float)
    return float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else 0.0


def holm(pvals):
    """Holm-Bonferroni corrected p-values, preserving input order."""
    idx = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(idx):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(running, 1.0)
    return adj.tolist()


def main():
    gens = load("generations/generations.jsonl")
    fit = load("judgments/fit_ratings.jsonl")
    pw = load("judgments/pairwise.jsonl")
    adapt = load("judgments/adaptation.jsonl")

    gidx = {(g["task_id"], g["condition"]): g for g in gens}
    fidx = {(r["task_id"], r["condition"]): r for r in fit}
    task_ids = sorted({g["task_id"] for g in gens})
    summary = {}

    # ---------- 1) Absolute audience-fit / understanding / engagement ----------
    print("\n=== 1. Absolute ratings (audience model role-plays reader) ===")
    metrics = ["fit", "understanding", "engagement"]
    abs_stats = {}
    pvals_collect, pkeys = [], []
    for metric in metrics:
        abs_stats[metric] = {}
        # paired arrays
        vals = {c: [] for c in CONDS}
        for tid in task_ids:
            if all((tid, c) in fidx and fidx[(tid, c)][metric] is not None for c in CONDS):
                for c in CONDS:
                    vals[c].append(fidx[(tid, c)][metric])
        for c in CONDS:
            m, lo, hi = bootstrap_ci(vals[c])
            abs_stats[metric][c] = {"mean": m, "ci": [lo, hi], "n": len(vals[c])}
        # Wilcoxon: intervention vs each baseline (paired)
        for b in ["plain", "cot", "simtom"]:
            d = np.array(vals["persentence"]) - np.array(vals[b])
            try:
                w, p = stats.wilcoxon(vals["persentence"], vals[b])
            except ValueError:
                p = 1.0
            abs_stats[metric][f"vs_{b}"] = {"mean_diff": float(d.mean()),
                                            "cohen_d": cohen_d_paired(d), "p": float(p)}
            pvals_collect.append(float(p)); pkeys.append(("abs", metric, b))
        print(f"  {metric}: " + ", ".join(f"{c}={abs_stats[metric][c]['mean']:.2f}" for c in CONDS))
    summary["absolute"] = abs_stats

    # ---------- 2) Pairwise win-rates (intervention-centric) ----------
    print("\n=== 2. Pairwise win-rate of intervention vs baselines ===")
    pw_stats = {}
    by = defaultdict(list)
    for r in pw:
        by[(r["dimension"], r["baseline"])].append(r["interv_score"])
    for (dim, base), scores in sorted(by.items()):
        m, lo, hi = bootstrap_ci(scores)
        # sign-ish test vs 0.5: Wilcoxon of (score-0.5)
        d = np.array(scores) - 0.5
        try:
            w, p = stats.wilcoxon(d) if np.any(d != 0) else (0, 1.0)
        except ValueError:
            p = 1.0
        pw_stats[f"{dim}_vs_{base}"] = {"winrate": m, "ci": [lo, hi], "p": float(p), "n": len(scores)}
        pvals_collect.append(float(p)); pkeys.append(("pw", dim, base))
        print(f"  {dim:8s} vs {base:7s}: winrate={m:.3f} [{lo:.2f},{hi:.2f}] p={p:.4f} n={len(scores)}")
    summary["pairwise"] = pw_stats

    # ---------- 3) Adaptation gap ----------
    print("\n=== 3. Adaptation gap (matched - crossed audience) ===")
    adapt_by = defaultdict(list)
    concept_gap = defaultdict(dict)
    for r in adapt:
        adapt_by[r["condition"]].append(r["gap"])
        concept_gap[r["concept_idx"]][r["condition"]] = r["gap"]
    adapt_stats = {}
    for c in CONDS:
        m, lo, hi = bootstrap_ci(adapt_by[c])
        adapt_stats[c] = {"mean_gap": m, "ci": [lo, hi], "n": len(adapt_by[c])}
        print(f"  {c:12s}: gap={m:.2f} [{lo:.2f},{hi:.2f}] n={len(adapt_by[c])}")
    # paired: intervention gap vs each baseline gap (by concept)
    concepts = sorted(concept_gap.keys())
    for b in ["plain", "cot", "simtom"]:
        pi = [concept_gap[c]["persentence"] for c in concepts if "persentence" in concept_gap[c] and b in concept_gap[c]]
        pb = [concept_gap[c][b] for c in concepts if "persentence" in concept_gap[c] and b in concept_gap[c]]
        d = np.array(pi) - np.array(pb)
        try:
            w, p = stats.wilcoxon(pi, pb)
        except ValueError:
            p = 1.0
        adapt_stats[f"vs_{b}"] = {"mean_diff": float(d.mean()), "cohen_d": cohen_d_paired(d), "p": float(p)}
        pvals_collect.append(float(p)); pkeys.append(("adapt", "gap", b))
        print(f"    intervention-gap vs {b}-gap: diff={d.mean():.2f} d={cohen_d_paired(d):.2f} p={p:.4f}")
    summary["adaptation"] = adapt_stats

    # ---------- 4) Diversity / homogenization ----------
    print("\n=== 4. Homogenization (lower self-BLEU / higher distinct-2 = more diverse) ===")
    smooth = SmoothingFunction().method1
    div_stats = {}
    for c in CONDS:
        texts = [g["text"] for g in gens if g["condition"] == c and len(g["text"].split()) > 5]
        toks = [t.lower().split() for t in texts]
        # distinct-2
        allbg = [tuple(tk[i:i+2]) for tk in toks for i in range(len(tk) - 1)]
        distinct2 = len(set(allbg)) / max(len(allbg), 1)
        # self-BLEU: each piece vs all others as references (sampled cap for speed)
        sb = []
        for i, hyp in enumerate(toks):
            refs = [toks[j] for j in range(len(toks)) if j != i]
            if refs and len(hyp) > 1:
                sb.append(sentence_bleu(refs, hyp, weights=(0.5, 0.5), smoothing_function=smooth))
        div_stats[c] = {"distinct2": float(distinct2), "self_bleu": float(np.mean(sb)) if sb else None}
        print(f"  {c:12s}: distinct-2={distinct2:.3f}  self-BLEU={np.mean(sb):.3f}")
    summary["diversity"] = div_stats

    # ---------- 5) Cost ----------
    print("\n=== 5. Cost (length & thinking) ===")
    cost_stats = {}
    for c in CONDS:
        wc = [g["word_count"] for g in gens if g["condition"] == c]
        th = [g["thinking_chars"] for g in gens if g["condition"] == c]
        cost_stats[c] = {"word_count_mean": float(np.mean(wc)), "word_count_std": float(np.std(wc)),
                         "thinking_chars_mean": float(np.mean(th))}
        print(f"  {c:12s}: words={np.mean(wc):.0f}±{np.std(wc):.0f}  thinking_chars={np.mean(th):.0f}")
    summary["cost"] = cost_stats

    # ---------- Holm correction across all significance tests ----------
    adj = holm(pvals_collect)
    summary["holm"] = [{"test": list(k), "p_raw": pv, "p_holm": pa}
                       for k, pv, pa in zip(pkeys, pvals_collect, adj)]
    print("\n=== Holm-corrected p-values ===")
    for k, pv, pa in zip(pkeys, pvals_collect, adj):
        star = "*" if pa < 0.05 else " "
        print(f"  {star} {str(k):40s} p_raw={pv:.4f} p_holm={pa:.4f}")

    with open(os.path.join(RES, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    make_figures(summary, gens, fit, adapt)
    print("\nSaved results/summary.json and figures/")


def make_figures(summary, gens, fit, adapt):
    # Fig 1: absolute fit/understanding/engagement bars with CIs
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, metric in zip(axes, ["fit", "understanding", "engagement"]):
        s = summary["absolute"][metric]
        means = [s[c]["mean"] for c in CONDS]
        errs = [[s[c]["mean"] - s[c]["ci"][0] for c in CONDS],
                [s[c]["ci"][1] - s[c]["mean"] for c in CONDS]]
        colors = ["#999", "#6aa", "#69c", "#e55"]
        ax.bar(range(len(CONDS)), means, yerr=errs, capsize=4, color=colors)
        ax.set_xticks(range(len(CONDS))); ax.set_xticklabels([LABELS[c] for c in CONDS], fontsize=7, rotation=15)
        ax.set_title(f"Audience-rated {metric}"); ax.set_ylim(0, 10); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig1_absolute_ratings.png"), dpi=130); plt.close()

    # Fig 2: pairwise win-rate of intervention
    fig, ax = plt.subplots(figsize=(8, 4.2))
    keys = [k for k in summary["pairwise"]]
    means = [summary["pairwise"][k]["winrate"] for k in keys]
    errs = [[summary["pairwise"][k]["winrate"] - summary["pairwise"][k]["ci"][0] for k in keys],
            [summary["pairwise"][k]["ci"][1] - summary["pairwise"][k]["winrate"] for k in keys]]
    ax.barh(range(len(keys)), means, xerr=errs, capsize=4, color="#e55")
    ax.axvline(0.5, color="k", ls="--", label="chance (0.5)")
    ax.set_yticks(range(len(keys))); ax.set_yticklabels(keys, fontsize=8)
    ax.set_xlabel("Intervention win-rate vs baseline"); ax.set_xlim(0, 1); ax.legend()
    ax.set_title("Pairwise: per-sentence audience sim vs baselines")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig2_pairwise.png"), dpi=130); plt.close()

    # Fig 3: adaptation gap
    fig, ax = plt.subplots(figsize=(7, 4.2))
    means = [summary["adaptation"][c]["mean_gap"] for c in CONDS]
    errs = [[summary["adaptation"][c]["mean_gap"] - summary["adaptation"][c]["ci"][0] for c in CONDS],
            [summary["adaptation"][c]["ci"][1] - summary["adaptation"][c]["mean_gap"] for c in CONDS]]
    ax.bar(range(len(CONDS)), means, yerr=errs, capsize=4, color=["#999", "#6aa", "#69c", "#e55"])
    ax.set_xticks(range(len(CONDS))); ax.set_xticklabels([LABELS[c] for c in CONDS], fontsize=7, rotation=15)
    ax.set_ylabel("Adaptation gap (matched − crossed fit)")
    ax.set_title("Does the method actually use the audience?"); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig3_adaptation_gap.png"), dpi=130); plt.close()

    # Fig 4: diversity + cost
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    sb = [summary["diversity"][c]["self_bleu"] for c in CONDS]
    axes[0].bar(range(len(CONDS)), sb, color=["#999", "#6aa", "#69c", "#e55"])
    axes[0].set_xticks(range(len(CONDS))); axes[0].set_xticklabels([LABELS[c] for c in CONDS], fontsize=7, rotation=15)
    axes[0].set_title("Self-BLEU (lower = less homogenized)"); axes[0].grid(axis="y", alpha=0.3)
    wc = [summary["cost"][c]["word_count_mean"] for c in CONDS]
    axes[1].bar(range(len(CONDS)), wc, color=["#999", "#6aa", "#69c", "#e55"])
    axes[1].set_xticks(range(len(CONDS))); axes[1].set_xticklabels([LABELS[c] for c in CONDS], fontsize=7, rotation=15)
    axes[1].set_title("Output length (words)"); axes[1].grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig4_diversity_cost.png"), dpi=130); plt.close()


if __name__ == "__main__":
    main()
