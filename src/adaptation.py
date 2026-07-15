"""Adaptation-gap control (H3 / gap G4): does the method actually USE the audience, or is it
just a generic quality boost?

For each concept we have a child-targeted and an expert-targeted piece per condition. We ask the
audience model to rate each piece BOTH as the child and as the expert:

  matched  = fit(child_text | child judge),  fit(expert_text | expert judge)
  crossed  = fit(child_text | expert judge), fit(expert_text | child judge)

  adaptation_gap = mean(matched) - mean(crossed)

A method that genuinely tailors writing to the reader scores high when judged by the intended
reader and lower when judged by the *other* reader -> large gap. A generic-quality method that
ignores the audience produces text both readers rate similarly -> gap ~ 0.
"""
import json, os, re
from common import chat, pmap, AUDIENCE_MODEL
from tasks import AUDIENCES, CONCEPTS, ADAPT_PAIR
from judge import FIT_PROMPT, _parse_json, load_gens

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "judgments", "adaptation.jsonl")
A_KEY, B_KEY = ADAPT_PAIR  # ("child", "expert")


def rate_as(text, judge_audience_desc):
    prompt = FIT_PROMPT.format(audience=judge_audience_desc, text=text[:2500])
    raw = chat(AUDIENCE_MODEL, [{"role": "user", "content": prompt}],
               temperature=0.0, max_tokens=200)
    d = _parse_json(raw) or {}
    return d.get("fit")


def main():
    rows, idx = load_gens()
    conditions = ["plain", "cot", "simtom", "persentence"]
    jobs = []
    for ci in range(len(CONCEPTS)):
        for cond in conditions:
            a_id = f"exp_{ci:02d}_{A_KEY}"   # child-targeted
            b_id = f"exp_{ci:02d}_{B_KEY}"   # expert-targeted
            if (a_id, cond) in idx and (b_id, cond) in idx:
                jobs.append((ci, cond, idx[(a_id, cond)]["text"], idx[(b_id, cond)]["text"]))

    def run(job):
        ci, cond, child_text, expert_text = job
        # matched
        m1 = rate_as(child_text, AUDIENCES[A_KEY])
        m2 = rate_as(expert_text, AUDIENCES[B_KEY])
        # crossed
        c1 = rate_as(child_text, AUDIENCES[B_KEY])
        c2 = rate_as(expert_text, AUDIENCES[A_KEY])
        vals = [m1, m2, c1, c2]
        if any(v is None for v in vals):
            return None
        matched = (m1 + m2) / 2
        crossed = (c1 + c2) / 2
        return {"concept_idx": ci, "condition": cond,
                "matched": matched, "crossed": crossed, "gap": matched - crossed,
                "child_by_child": m1, "expert_by_expert": m2,
                "child_by_expert": c1, "expert_by_child": c2}

    print(f"Adaptation jobs: {len(jobs)} (x4 ratings each)")
    res = [r for r in pmap(run, jobs, workers=12, desc="adapt") if r]
    with open(OUT, "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(res)} -> {OUT}")


if __name__ == "__main__":
    main()
