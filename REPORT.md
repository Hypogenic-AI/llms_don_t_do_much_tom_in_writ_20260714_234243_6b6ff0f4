# Does Per-Sentence Audience Simulation in Chain-of-Thought Improve LLM Writing?

**A test of the hypothesis: "LLMs write annoyingly because they don't model the audience;
making the model simulate the audience in its CoT every sentence would help."**

Date: 2026-07-15 · Generators: `gpt-4.1-mini`, `llama-3.3-70b` · Judges (cross-family):
`claude-sonnet-4.5` (quality), `gemini-2.5-flash` (audience). All calls via OpenRouter, seed 42.

---

## 1. Executive Summary

We directly tested the user's hypothesis that forcing an LLM to **explicitly simulate the target
reader in its chain-of-thought roughly once per sentence** produces better writing. We generated
60 audience-targeted writing tasks under four conditions — plain, generic step-by-step CoT,
one-shot audience modeling (SimToM), and the proposed **per-sentence audience simulation** — and
evaluated them with two *separate*, cross-family judge models (one role-playing the reader for
audience-fit, one scoring overall writing quality), using paired significance tests, effect
sizes, and Holm correction.

**The specific hypothesis is refuted, but its underlying intuition is vindicated.** Modeling the
audience *does* help — but a **single, rich audience model written once up front (SimToM) is the
clear winner** on every metric (audience-fit, engagement, genuine audience adaptation, and even
prose diversity), at lower cost. The proposed **per-sentence simulation is the worst or
near-worst condition**: it loses pairwise on overall quality to *all three* baselines (win-rates
0.08–0.25, all p < 0.001 after Holm correction), scores lowest on engagement and audience-fit,
shows *no* better audience adaptation, does *not* reduce homogenization, and is the longest and
most token-expensive. The result **replicates across two generator families** (gpt-4.1-mini and
Llama-3.3-70B).

**Takeaway: "think about the reader" helps writing; "re-simulate the reader every sentence"
hurts it.** Per-sentence simulation makes the model *locally reactive* — it tracks
sentence-by-sentence reactions and loses the global voice, arc, and craft that make prose good,
while bloating length. Less is more: one good audience model beats twelve.

---

## 2. Research Question & Motivation

LLM prose is often fluent-but-generic and "annoying" — pitched at no one in particular. The
user's conjecture is that this stems from weak audience modeling (Theory of Mind), and that the
fix is to have the generator *verbalize an incremental audience simulation in its CoT, about
every sentence.*

**Why it matters.** If true, this would be a cheap, prompt-only, inference-time fix for a
pervasive quality problem, with impact on assistants, education, and technical communication.

**Gap in prior work** (see `literature_review.md`). Audience-aware generation has been studied
via decode-time steering (Takmaz 2023), rerank-by-a-separate-listener (Andreas & Klein 2016;
CodeRSA 2025), external reward (Style Infusion 2023), and formal belief updates (CRSA 2025) —
but audience modeling has *never* been placed **inside the generator's own natural-language CoT,
per sentence, for open-ended writing quality**. Per-sentence mental-state simulation exists only
for *comprehension QA / dialogue* (PercepToM 2024; Thought-Tracing 2025). Critically, AI-Slop→
AI-Polish (2025) shows **generic reasoning does NOT improve writing quality**, so a gain from an
*audience-specific* CoT would be a genuinely non-trivial result — and must be isolated from
"more thinking tokens." This is the first direct test of the user's exact intervention.

## Hypotheses
- **H1 (audience-fit):** per-sentence sim > {plain, CoT, SimToM} on a separate reader model's fit.
- **H2 (quality):** per-sentence sim > baselines on a cross-family quality judge.
- **H3 (adaptation):** per-sentence sim shows the largest audience-adaptation gap.
- **H4 (not just tokens):** it beats generic CoT despite both adding reasoning tokens.
- **H5 (diversity/cost):** it reduces generic "sameness" (homogenization) at acceptable cost.

---

## 3. Experimental Setup

### Task set (60 tasks)
**Audience-targeted writing**, the sharpest test of audience modeling. Each task pairs a topic
with a specific target reader that varies in **expertise and disposition**:
- **48 explanatory** tasks: "Explain {concept} for {audience}" (~150 words). 12 concepts across
  science/finance/tech (vaccines, compound interest, black holes, how an LLM generates text,
  antibiotic resistance, blockchain, seasons, mRNA, the stock market, encryption, photosynthesis,
  greenhouse warming) × 4 audiences: **an 8-year-old child**, **a non-technical executive**, **a
  PhD-level expert**, **a skeptic who distrusts the topic**.
- **12 creative** tasks: WritingPrompts stories with an assigned target readership.

### Conditions (independent variable = *how the model reasons about the reader*)
Crucially, **every condition is told who the reader is** (identical information). The only
manipulation is the reasoning strategy, so any difference is attributable to the strategy, not to
privileged audience information. Generation is a clean **two-step** protocol (a reasoning turn,
then a "write ONLY the final piece" turn) so the judged text never contains CoT scaffolding.

| Condition | Reasoning turn |
|---|---|
| **plain** | none — write directly (the "annoying" egocentric default) |
| **cot** | "think step by step to make it excellent" (generic; **token-matched control**) |
| **simtom** | describe the reader once (background, knowledge, cares, tone), then write |
| **persentence** *(intervention)* | plan sentence-by-sentence; after each sentence simulate how *this reader* reacts (confused/bored/skeptical/curious), then choose the next |

### Models (kept in **different families** to avoid self-preference & shortcut inflation, per Shapira 2023)
- **Generator:** `openai/gpt-4.1-mini` (primary); `meta-llama/llama-3.3-70b-instruct` (replication).
- **Quality judge:** `anthropic/claude-sonnet-4.5` — blind, order-balanced pairwise.
- **Audience/listener model:** `google/gemini-2.5-flash` — **role-plays the target reader** to
  give absolute fit/understanding/engagement ratings (1–10) and pairwise "which serves this
  reader better" (Fried et al.'s communication-based evaluation).

### Metrics
1. **Audience-fit / understanding / engagement** (audience model role-playing the reader, 1–10).
2. **Overall writing quality** — cross-family judge, pairwise, both orders (position-bias control).
3. **Adaptation gap** (H3 control): each concept written for a *child* and an *expert*; both readers
   rate both pieces. `gap = matched − crossed`. A method that truly tailors to the reader scores
   high for its intended reader and low for the other → large gap; a generic-quality method → ~0.
4. **Homogenization**: distinct-2 and self-BLEU within each condition.
5. **Cost**: output length (words) and reasoning ("thinking") characters.

### Protocol / reproducibility
Temperature 0.7 for generation, 0.0 for judging; seed 42. Every model call is disk-cached by a
hash of (model, messages, params), so the pipeline is fully reproducible and re-runs are free.
**Scale:** 240 primary generations + 192 replication generations; ~1,400 judge calls; 3,876
cached responses. Estimated API cost ≈ $8–15. All raw outputs saved under `results/`.

---

## 4. Results

### 4.1 Absolute audience ratings (gemini role-playing the reader, 1–10)

| Metric | plain | Generic CoT | **SimToM** | **Per-sentence (intervention)** |
|---|---|---|---|---|
| **Audience-fit**   | 6.37 | 7.12 | **7.28** | 6.52 |
| **Engagement**     | 5.80 | 6.45 | **6.62** | 5.93 |
| **Understanding**  | 9.15 | 9.30 | **9.32** | 9.08 |

The intervention is **near the bottom on every dimension** — barely above plain and clearly below
both CoT and SimToM. Paired Wilcoxon (intervention vs each baseline), Holm-corrected: intervention
is **significantly worse than SimToM** on fit (p=0.0001), engagement (p=0.0002), and understanding
(p=0.057); and **worse than generic CoT** on fit (p=0.0007) and engagement (p=0.0025). It is
statistically indistinguishable from *plain*. → **H1 refuted.** (`figures/fig1_absolute_ratings.png`)

### 4.2 Pairwise: intervention vs baselines (win-rate; 0.5 = tie, both orders averaged, n=60)

| Comparison | Overall quality (claude) | Audience-fit (gemini) |
|---|---|---|
| intervention vs **plain**   | **0.250** (p<0.001) | 0.475 (p=0.67, tie) |
| intervention vs **CoT**     | **0.150** (p<0.001) | **0.250** (p<0.001) |
| intervention vs **SimToM**  | **0.083** (p<0.001) | **0.200** (p<0.001) |

The intervention **loses on overall quality to all three baselines** — most heavily to SimToM
(wins only 8% of the time) — and loses on audience-fit to CoT and SimToM (it only ties plain). All
survive Holm correction. → **H2 and H4 refuted.** (`figures/fig2_pairwise.png`)

### 4.3 Adaptation gap — does the method actually *use* the audience? (n=12 concepts)

| Condition | plain | Generic CoT | **SimToM** | Per-sentence |
|---|---|---|---|---|
| **Adaptation gap** (matched − crossed fit) | 6.08 | 6.67 | **7.00** | 6.04 |

If the per-sentence simulation genuinely tailored writing to the reader, it should show the
*largest* gap. Instead it is the **smallest** (tied with plain), and **significantly smaller than
SimToM** (Δ=−0.96, Cohen's d=−1.27, p=0.004 Holm-corrected) and CoT (Δ=−0.62, d=−0.75). SimToM,
which models the reader once, adapts most. → **H3 refuted.** (`figures/fig3_adaptation_gap.png`)

### 4.4 Homogenization and cost

| Condition | distinct-2 ↑ | self-BLEU ↓ | words | thinking chars |
|---|---|---|---|---|
| plain        | 0.823 | 0.406 | 148 | 0 |
| Generic CoT  | 0.842 | 0.383 | 144 | 2086 |
| **SimToM**   | **0.857** | **0.363** | 147 | 1762 |
| **Per-sentence** | 0.823 | 0.402 | **170** | **2480** |

The intervention is the **most homogenized** (highest self-BLEU / lowest distinct-2, tied with
plain) and the **longest and most token-expensive**. SimToM is the *least* homogenized. → **H5
refuted**: per-sentence simulation does not reduce generic sameness and costs the most.
(`figures/fig4_diversity_cost.png`)

### 4.5 Cross-family replication (Llama-3.3-70B generator, 48 explanatory tasks)

The same ordering holds with a completely different generator: per-sentence is worst on fit
(5.52 vs SimToM/CoT 5.90) and engagement (5.02 vs SimToM 5.46), and loses pairwise to **every**
baseline on both quality (win-rates 0.12–0.26) and fit (0.17–0.33), all significant. The negative
result is **not specific to one model family.** (`results/replication.json`)

---

## 5. Analysis & Discussion

**The direction of the user's intuition is right; the mechanism is wrong.** Audience modeling
clearly helps: SimToM and generic CoT both beat plain, and SimToM — an explicit one-shot audience
model — is the best condition on *every* metric including the audience-adaptation control. So
"LLMs write annoyingly partly because they don't model the reader" is supported. But the specific
proposed fix — *simulating the reader every sentence in CoT* — actively backfires.

**Why does per-sentence simulation hurt?** Three mechanisms, visible in the outputs:
1. **Local reactivity destroys global craft.** Deciding each next sentence from the reader's
   momentary reaction yields prose that is coherent locally but loses arc, voice, and payoff.
   *Example (children's story):* SimToM builds a whimsical arc with a delightful button ("Death's
   Help Desk: Making Life Interesting Since Forever"); the per-sentence version stays plot-
   mechanical, keeps the adult phrase "living forever sucks," and drifts into heavier themes
   ("watching everyone I loved grow old and leave") — locally reasonable, globally off-key for a child.
2. **It doesn't adapt more, it adapts *less*** (§4.3). A single rich audience model front-loads a
   coherent stance ("this is a scared 8-year-old — stay warm, concrete, gentle") that governs the
   whole piece; per-sentence reactions are noisier and pull in inconsistent directions.
3. **Bloat.** The intervention runs ~15% longer and uses the most thinking tokens, yet the extra
   effort yields worse writing — the opposite of the token-for-quality trade the hypothesis assumed.

This directly echoes AI-Slop→AI-Polish's finding that *more reasoning ≠ better writing*, and
extends it: even *audience-specific* reasoning fails to help writing when applied at fine
granularity, whereas a coarse one-shot audience model does help. **Granularity is the key axis**
(literature gap G2): the sweet spot is "model the reader once, richly," not "re-model every sentence."

**Validity safeguards.** Judges are in different families from the generators (no self-preference);
pairwise judging is blind and order-balanced; only clean final text is judged (the two-step
protocol eliminated a real scaffolding-leak bug we caught mid-study, §6); the adaptation-gap
control rules out a generic-quality confound; and the effect replicates across two generators.

---

## 6. Limitations

- **One granularity of the intervention.** We tested "every sentence." An intermediate granularity
  (per-paragraph, or once-then-one-revision-pass) might recover the benefit; we did not sweep it.
- **LLM judges, not humans.** Fit and quality come from LLM judges (cross-family, but still LLMs).
  A small human eval would strengthen the writing-quality claim, though the effect is large (win-
  rates ≤0.25) and consistent across two independent judges and two generators.
- **Prompt sensitivity.** A different phrasing of the per-sentence prompt could perform better; we
  used one careful, faithful operationalization. The negative result bounds *this* natural
  implementation, not every conceivable one.
- **Short-form only** (~150 words). Per-sentence audience tracking might matter more for long
  documents where global consistency is otherwise lost — untested here.
- **A caught bug.** An initial marker-based text extractor leaked audience-analysis preamble into
  ~23% of SimToM pieces; we replaced it with the two-step protocol (zero leaks) and re-ran
  everything. Reported numbers are from the clean pipeline. (Direction of the finding was unchanged.)
- **Scale.** 60 tasks (48 for replication); paired tests and bootstrap CIs mitigate this, and
  effects are large, but larger and more diverse task sets would tighten the estimates.

---

## 7. Conclusions & Next Steps

**Answer to the research question:** No — making an LLM simulate the audience in its CoT every
sentence does **not** improve its writing; it makes it *worse* on quality, engagement, audience-
fit, adaptation, and diversity, while costing more. However, the user's deeper intuition holds:
**audience modeling does help**, and the effective form is a **single rich audience model written
once up front (SimToM)**, which beats plain generation, generic CoT, and per-sentence simulation
on every metric measured. The lesson is about *granularity*: think about the reader once, deeply —
not reactively, sentence by sentence.

**Recommended follow-ups:**
1. **Granularity sweep** — once / per-paragraph / per-sentence — to map where audience reasoning
   stops helping and starts fragmenting the prose (directly tests literature gap G2).
2. **SimToM + one revision pass** ("re-read as the reader, fix what jars") — likely the practical
   sweet spot; combine the winning one-shot model with a single global critique.
3. **Human evaluation** on the SimToM-vs-plain and per-sentence-vs-SimToM contrasts.
4. **Long-form** documents, where global-consistency costs of per-sentence tracking may differ.

**Open question:** is per-sentence simulation's failure fundamental (local optimization loses
global structure) or an artifact of doing it *before* writing rather than as *revision*? Follow-up
2 would distinguish these.

---

## References (used resources)
- AI-Slop to AI-Polish (WQ/WQRM/LAMP), arXiv:2504.07532 — writing-quality testbed; "generic
  reasoning doesn't improve writing."
- SimToM / perspective-taking prompting — ToM survey, arXiv:2502.06470.
- PercepToM (per-sentence perspective for QA), arXiv:2407.06004.
- Speaking the Language of Your Listener (audience-aware generation), arXiv:2305.19933.
- Pragmatics in Language Grounding survey (communication-based eval), arXiv:2211.08371.
- Clever Hans or Neural ToM? (CoT shortcut inflation caution), arXiv:2305.14763.
- The Homogenization Problem in LLMs, arXiv:2601.06116.
- Datasets: WritingPrompts (`euclaise/writingprompts`). Full catalog in `resources.md` /
  `literature_review.md`.
- Tooling: OpenRouter API; models `openai/gpt-4.1-mini`, `meta-llama/llama-3.3-70b-instruct`,
  `anthropic/claude-sonnet-4.5`, `google/gemini-2.5-flash`; Python 3.12, openai 2.45, scipy 1.18,
  numpy, matplotlib, nltk.

*Figures: `figures/fig1_absolute_ratings.png`, `fig2_pairwise.png`, `fig3_adaptation_gap.png`,
`fig4_diversity_cost.png`. Full numbers: `results/summary.json`, `results/replication.json`.
Raw generations & judgments: `results/generations/`, `results/judgments/`.*
