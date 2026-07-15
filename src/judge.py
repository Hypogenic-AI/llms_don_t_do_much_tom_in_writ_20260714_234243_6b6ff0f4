"""Evaluate generated pieces.

Two evaluators, both in DIFFERENT families from the generator (gpt-4.1-mini) to avoid
self-preference and shortcut inflation (Shapira et al.):
  - AUDIENCE_MODEL (gemini-2.5-flash): role-plays the target reader -> absolute fit &
    comprehension ratings (1-10), and pairwise "which serves this reader better".
  - JUDGE_MODEL (claude-sonnet-4.5): blind overall writing-quality pairwise judge.

Pairwise calls are order-balanced (both A-first and B-first) to cancel position bias.
Only the extracted final TEXT is shown to judges (never the CoT scaffolding).
"""
import json, os, re
from common import chat, pmap, JUDGE_MODEL, AUDIENCE_MODEL

GEN_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "generations", "generations.jsonl")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "judgments")
os.makedirs(OUT_DIR, exist_ok=True)


def load_gens():
    with open(GEN_PATH) as f:
        rows = [json.loads(l) for l in f]
    idx = {(r["task_id"], r["condition"]): r for r in rows}
    return rows, idx


def _parse_json(raw):
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


# ---------- Absolute audience-fit rating (audience model role-plays the reader) ----------
FIT_PROMPT = """You ARE this reader: {audience}.

Below is a piece of writing intended for you. Judge it purely from your own perspective as
this reader.

PIECE:
\"\"\"{text}\"\"\"

Rate on a 1-10 scale (10 = best) and reply with ONLY a JSON object:
{{"fit": <1-10, how well-pitched this is for YOU specifically — right level, tone, and focus>,
  "understanding": <1-10, how fully YOU would understand it>,
  "engagement": <1-10, how engaging/non-annoying it is to YOU>}}"""


def rate_fit(piece):
    prompt = FIT_PROMPT.format(audience=piece["audience"], text=piece["text"][:2500])
    raw = chat(AUDIENCE_MODEL, [{"role": "user", "content": prompt}],
               temperature=0.0, max_tokens=200)
    d = _parse_json(raw) or {}
    return {"task_id": piece["task_id"], "condition": piece["condition"],
            "audience_key": piece["audience_key"], "kind": piece["kind"],
            "fit": d.get("fit"), "understanding": d.get("understanding"),
            "engagement": d.get("engagement")}


# ---------- Pairwise comparison ----------
PAIR_PROMPTS = {
"quality": """You are an expert writing judge. Two pieces (A and B) address the same task for
the same intended reader. Judge OVERALL WRITING QUALITY: clarity, craft, voice, engagement,
and how non-generic/non-annoying it is.

INTENDED READER: {audience}
TASK: {instruction}

PIECE A:
\"\"\"{a}\"\"\"

PIECE B:
\"\"\"{b}\"\"\"

Reply with ONLY JSON: {{"winner": "A" | "B" | "tie", "reason": "<one sentence>"}}""",

"fit": """You ARE this reader: {audience}. Two pieces (A and B) were written for you on the
same task. Which one is BETTER FOR YOU — better pitched to your level, tone, and interests, and
less annoying to read?

TASK: {instruction}

PIECE A:
\"\"\"{a}\"\"\"

PIECE B:
\"\"\"{b}\"\"\"

Reply with ONLY JSON: {{"winner": "A" | "B" | "tie", "reason": "<one sentence>"}}""",
}


def pairwise(a_piece, b_piece, dimension):
    model = JUDGE_MODEL if dimension == "quality" else AUDIENCE_MODEL
    prompt = PAIR_PROMPTS[dimension].format(
        audience=a_piece["audience"], instruction=a_piece["instruction"],
        a=a_piece["text"][:2500], b=b_piece["text"][:2500])
    raw = chat(model, [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=150)
    d = _parse_json(raw) or {}
    w = d.get("winner", "tie")
    return w if w in ("A", "B", "tie") else "tie"


def run_pairwise_job(job):
    """job = (task_id, baseline_cond, dimension, idx). Runs BOTH orders, returns net result."""
    task_id, base, dim, idx = job
    interv = idx[(task_id, "persentence")]
    basep = idx[(task_id, base)]
    # order 1: A=intervention, B=baseline ; order 2: A=baseline, B=intervention
    w1 = pairwise(interv, basep, dim)
    w2 = pairwise(basep, interv, dim)
    # translate to intervention-centric: did intervention win?
    def interv_win(w, interv_is_A):
        if w == "tie":
            return 0.5
        won_A = (w == "A")
        return 1.0 if (won_A == interv_is_A) else 0.0
    s1 = interv_win(w1, True)
    s2 = interv_win(w2, False)
    return {"task_id": task_id, "baseline": base, "dimension": dim,
            "interv_score": (s1 + s2) / 2, "raw1": w1, "raw2": w2,
            "audience_key": interv["audience_key"], "kind": interv["kind"]}


def main():
    rows, idx = load_gens()
    print(f"Loaded {len(rows)} generations")

    # 1) absolute audience-fit ratings for every piece
    fit = pmap(rate_fit, rows, workers=12, desc="fit")
    with open(os.path.join(OUT_DIR, "fit_ratings.jsonl"), "w") as f:
        for r in fit:
            f.write(json.dumps(r) + "\n")
    print("fit ratings done")

    # 2) pairwise: intervention vs each baseline, on quality + fit
    task_ids = sorted({r["task_id"] for r in rows})
    baselines = ["plain", "cot", "simtom"]
    dims = ["quality", "fit"]
    jobs = [(tid, b, d, idx) for tid in task_ids for b in baselines for d in dims]
    print(f"Pairwise jobs: {len(jobs)} (x2 orders each)")
    pw = pmap(run_pairwise_job, jobs, workers=12, desc="pairwise")
    with open(os.path.join(OUT_DIR, "pairwise.jsonl"), "w") as f:
        for r in pw:
            f.write(json.dumps(r) + "\n")
    print("pairwise done")


if __name__ == "__main__":
    main()
