"""Generate writing under 4 conditions.

Design principle: EVERY condition is told who the reader is (same information). The manipulation
is only *how the model reasons about that reader* — so any difference is attributable to the
reasoning strategy, not to privileged audience information.

  plain       : write directly (egocentric default; reader named but no reasoning asked).
  cot         : generic "think step by step to make it excellent" (token-matched reasoning control).
  simtom      : describe the audience ONCE up front, then write.
  persentence : the INTERVENTION — write sentence-by-sentence, simulating the reader after each.

We extract the clean final text between ===BEGIN===/===END=== for fair downstream judging, and
we record the number of "thinking" tokens (chars before ===BEGIN===) to check the token confound.
"""
import re, json, os, sys
from common import chat, pmap, GEN_MODEL

CONDITIONS = ["plain", "cot", "simtom", "persentence"]

_HEADER = "READER (the intended audience for this piece): {audience}\nTASK: {instruction}\n"

# Two-step design: a REASONING turn (varies by condition) then a WRITE turn that produces ONLY
# the final piece. This guarantees clean final text with zero scaffolding leakage, uniformly
# across conditions (fixes the marker-parsing fairness bug). `plain` has no reasoning turn.
REASON = {
"cot": _HEADER + """
Think step by step about how to make this piece as excellent as possible — its structure, word
choice, opening, flow, and impact. Write out your step-by-step reasoning now (do NOT write the
final piece yet).""",

"simtom": _HEADER + """
Before writing, build a rich model of the reader: their background, what they already know, what
they care about, what would confuse or bore them, and what tone would land with them. Write out
that audience model now (do NOT write the final piece yet).""",

"persentence": _HEADER + """
Plan the piece ONE SENTENCE AT A TIME, actively simulating this specific reader as you go. For
each planned sentence use EXACTLY this format:

[S<n>]: <the sentence>
[READER]: <Simulate THIS specific reader reading the sentence you just wrote. What are they
thinking or feeling right now — confused, bored, skeptical, curious, convinced? What do they
want next? Given that, what should the next sentence do to serve them?>

Do this for about 8-12 sentences (~150 words of eventual prose). Do NOT write the clean final
piece yet — just this per-sentence planning.""",
}

WRITE_TURN = ("Now write ONLY the final piece for this reader (about 150 words). Output the prose "
              "and nothing else — no preamble, no headings, no reasoning, no annotations.")

PLAIN_PROMPT = _HEADER + ("\nWrite the piece now (about 150 words). Output ONLY the finished prose "
                          "— no preamble, no headings, no reasoning.")


def _clean(text):
    """Strip a stray leading meta line like 'Here is the piece:' if present."""
    text = text.strip()
    text = re.sub(r"^(sure|certainly|here(?:'s| is)|okay|of course)[^\n]{0,60}:\s*\n+", "",
                  text, flags=re.I)
    return text.strip()


def gen_one(job):
    task, cond = job
    hdr = dict(audience=task["audience"], instruction=task["instruction"])
    if cond == "plain":
        text = chat(GEN_MODEL, [{"role": "user", "content": PLAIN_PROMPT.format(**hdr)}],
                    temperature=0.7, max_tokens=700)
        reasoning = ""
    else:
        r_prompt = REASON[cond].format(**hdr)
        reasoning = chat(GEN_MODEL, [{"role": "user", "content": r_prompt}],
                         temperature=0.7, max_tokens=1400)
        text = chat(GEN_MODEL, [
            {"role": "user", "content": r_prompt},
            {"role": "assistant", "content": reasoning},
            {"role": "user", "content": WRITE_TURN}], temperature=0.7, max_tokens=700)
    text = _clean(text)
    return {**task, "condition": cond, "raw": reasoning, "text": text,
            "thinking_chars": len(reasoning), "word_count": len(text.split())}


def main():
    from tasks import build_all
    tasks = build_all()
    jobs = [(t, c) for t in tasks for c in CONDITIONS]
    print(f"Generating {len(jobs)} pieces ({len(tasks)} tasks x {len(CONDITIONS)} conditions)")
    out = pmap(gen_one, jobs, workers=12, desc="gen")
    path = os.path.join(os.path.dirname(__file__), "..", "results", "generations", "generations.jsonl")
    with open(path, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    # quick sanity: any empty extractions?
    empty = [r["task_id"] + "/" + r["condition"] for r in out if len(r["text"].split()) < 20]
    print(f"Saved {len(out)} -> {path}. Short/empty extractions: {len(empty)}")
    if empty:
        print("  ", empty[:20])


if __name__ == "__main__":
    main()
