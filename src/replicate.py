"""Robustness replication with a SECOND generator (Llama-3.3-70B), to check the finding is not
specific to gpt-4.1-mini. Explanatory tasks only. Reuses the same clean two-step generation and
the same cross-family judges."""
import json, os, numpy as np
from scipy import stats
import common, generate
from tasks import build_explanatory
from judge import rate_fit, run_pairwise_job
from common import pmap

common.GEN_MODEL = common.GEN_MODEL_ALT
generate.GEN_MODEL = common.GEN_MODEL_ALT  # monkeypatch so gen_one uses the alt model
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "replication.json")


def main():
    tasks = build_explanatory()
    jobs = [(t, c) for t in tasks for c in generate.CONDITIONS]
    print(f"[alt-gen] generating {len(jobs)} pieces with {common.GEN_MODEL}")
    gens = pmap(generate.gen_one, jobs, workers=10, desc="alt-gen")
    idx = {(g["task_id"], g["condition"]): g for g in gens}

    print("[alt-gen] fit ratings")
    fit = pmap(rate_fit, gens, workers=12, desc="alt-fit")
    fidx = {(r["task_id"], r["condition"]): r for r in fit}

    task_ids = sorted({g["task_id"] for g in gens})
    jobs2 = [(tid, b, d, idx) for tid in task_ids for b in ["plain", "cot", "simtom"]
             for d in ["quality", "fit"]]
    print(f"[alt-gen] pairwise {len(jobs2)}")
    pw = pmap(run_pairwise_job, jobs2, workers=12, desc="alt-pw")

    conds = generate.CONDITIONS
    out = {"generator": common.GEN_MODEL, "n_tasks": len(task_ids), "absolute": {}, "pairwise": {}}
    for m in ["fit", "engagement", "understanding"]:
        out["absolute"][m] = {c: float(np.mean([fidx[(t, c)][m] for t in task_ids
                                                if fidx[(t, c)][m] is not None])) for c in conds}
    from collections import defaultdict
    by = defaultdict(list)
    for r in pw:
        by[(r["dimension"], r["baseline"])].append(r["interv_score"])
    for (dim, base), sc in by.items():
        d = np.array(sc) - 0.5
        try:
            _, p = stats.wilcoxon(d) if np.any(d != 0) else (0, 1.0)
        except ValueError:
            p = 1.0
        out["pairwise"][f"{dim}_vs_{base}"] = {"winrate": float(np.mean(sc)), "p": float(p), "n": len(sc)}

    json.dump(out, open(OUT, "w"), indent=2)
    print("\n=== REPLICATION (Llama-3.3-70B generator) ===")
    for m in ["fit", "engagement"]:
        print(f"  {m}:", {c: round(out['absolute'][m][c], 2) for c in conds})
    print("  intervention pairwise win-rates:")
    for k, v in out["pairwise"].items():
        print(f"    {k:20s} {v['winrate']:.3f} (p={v['p']:.4f})")


if __name__ == "__main__":
    main()
