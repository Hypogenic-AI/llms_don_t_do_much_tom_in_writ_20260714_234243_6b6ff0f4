# LLMs Don't Do Much ToM in Writing

Does making an LLM **simulate the target reader in its chain-of-thought every sentence** improve
its writing? We test the hypothesis directly against three baselines with cross-family LLM judges,
paired statistics, and a two-model replication.

## Key findings
- **Hypothesis refuted.** Per-sentence audience simulation is the **worst or near-worst** condition:
  it loses pairwise on overall quality to plain, generic CoT, and one-shot audience modeling
  (win-rates **0.08–0.25**, all p < 0.001), and scores lowest on engagement and audience-fit.
- **But the intuition is vindicated.** Modeling the audience *does* help — a **single rich
  audience model written once up front (SimToM)** is the **best** condition on every metric
  (fit, engagement, genuine audience adaptation, and prose diversity), at lower cost.
- **Adaptation control:** per-sentence sim does *not* adapt to the reader more (its adaptation gap
  is the smallest, significantly below SimToM, d=−1.27, p=0.004). It's locally reactive and loses
  global craft.
- **It's not just "more tokens":** it uses the *most* thinking tokens yet writes worse — matching
  prior work that generic reasoning doesn't help writing.
- **Replicates across generators** (gpt-4.1-mini and Llama-3.3-70B). See **[REPORT.md](REPORT.md)**.

**One-line takeaway:** *think about the reader once, deeply — not reactively every sentence.*

## Reproduce
```bash
source .venv/bin/activate            # or: uv venv && source .venv/bin/activate
uv pip install openai pandas numpy scipy matplotlib nltk
export OPENROUTER_KEY=...             # required
cd src
python generate.py     # 240 pieces, 60 tasks x 4 conditions (cached)
python judge.py        # audience-fit + pairwise quality/fit (cross-family judges)
python adaptation.py   # adaptation-gap control (child vs expert cross-rating)
python analyze.py      # stats, Holm correction, figures -> results/summary.json, figures/
python replicate.py    # optional: Llama-3.3-70B robustness replication
```
All model calls are disk-cached (`results/cache/`), so re-runs are free and fully reproducible
(seed 42).

## Structure
```
planning.md              Research plan, motivation & novelty (Phase 0/1)
REPORT.md                Full report with results, figures, analysis, limitations
src/
  common.py              Cached/retrying OpenRouter client + concurrency + model roster
  tasks.py               60 audience-targeted writing tasks (concepts x audiences + creative)
  generate.py            4 conditions, clean two-step (reason -> write) generation
  judge.py               Audience-fit ratings + order-balanced pairwise judging
  adaptation.py          Adaptation-gap control (matched vs crossed audience)
  analyze.py             Descriptive stats, Wilcoxon, bootstrap CIs, Holm, figures
  replicate.py           Second-generator robustness replication
results/                 generations/, judgments/, summary.json, replication.json, cache/
figures/                 fig1_absolute_ratings, fig2_pairwise, fig3_adaptation_gap, fig4_diversity_cost
```

## Method in one paragraph
Every condition is told who the reader is (same information); only the *reasoning strategy*
differs — none / generic CoT / one-shot audience model / per-sentence audience simulation. A
two-step protocol separates reasoning from the final piece so judges never see CoT scaffolding.
Two separate cross-family models judge (claude for quality, gemini role-playing the reader for
fit), pairwise judging is blind and order-balanced, and an adaptation-gap control (write for a
child vs an expert; each reader rates both) tests whether a method *actually uses* the audience
rather than just producing generically-nice text.
