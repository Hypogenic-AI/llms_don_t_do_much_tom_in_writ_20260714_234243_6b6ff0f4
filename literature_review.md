# Literature Review: LLMs Don't Do Much ToM in Writing

**Hypothesis under study.** One reason LLMs write annoyingly is that they don't model the
audience — only vaguely, if at all. Having the model *explicitly simulate the audience in
its chain-of-thought (CoT) roughly every sentence* should improve its writing.

## Research Area Overview

The hypothesis sits at the intersection of three literatures:

1. **Theory of Mind (ToM) in LLMs** — whether/when models can represent others' mental
   states. Consensus: LLMs show *some* ToM but it is **brittle and non-robust**, collapsing
   under trivial adversarial perturbations (Ullman 2023; Shapira et al. 2023). This motivates
   the premise: default models do *not* reliably model a mind, so an explicit step might help.

2. **Pragmatics / audience design in generation** — the Rational Speech Acts (RSA) tradition,
   in which a speaker chooses each utterance by simulating how a listener will interpret it
   versus alternatives (Andreas & Klein 2016; Fried et al. 2023 survey). Explicit audience
   modeling **reliably improves communicative success** across reference games, captioning,
   instruction generation, and dialogue.

3. **LLM writing quality & homogenization** — SOTA models produce fluent-but-generic,
   homogenized prose; SOTA judges (including reasoning models) can barely tell good writing
   from bad, and **generic CoT does not fix it** (Chakrabarty et al. 2025; Homogenization 2026).

**The gap.** Every existing "audience-modeling" method operationalizes it *outside* the
generator's own reasoning — as a decode-time gradient (Takmaz 2023), a rerank by a separate
trained listener (Andreas & Klein 2016; CodeRSA 2025), an external reward/discriminator
(Style Infusion 2023), or a formal belief update (CRSA 2025). The specific intervention in
the hypothesis — **the generator itself verbalizing an audience simulation in natural-language
CoT, incrementally, during open-ended prose writing** — is essentially untested. The nearest
analogs (PercepToM 2024; Thought-Tracing 2025; ToM-agent 2025) apply per-sentence/per-turn
mental-state simulation to **comprehension QA or dialogue**, not to writing quality.

## Key Papers (condensed; full structured notes were extracted per paper)

### Closest prior work — audience-aware generation
- **Takmaz et al. 2023 (arXiv:2305.19933)** — "Speaking the Language of Your Listener."
  A frozen speaker steered at decode time by a *simulator* (ToM of the listener). **Audience-
  aware (listener-specific) > self-aware (generic listener) > baseline**: 71.8% vs 65.1% vs
  52.3% in-domain listener accuracy; the OOD gain (26.7% vs ~19%) shows modeling a *specific*
  reader is what pays off. Caveat: LSTMs, hidden-state gradients, referential game — not LLM CoT.
- **Andreas & Klein 2016 (arXiv:1604.00562)** — neural RSA. Sample candidate descriptions,
  rerank by a simulated listener; +17% accuracy, and a tiny fluency weight keeps language
  natural. Discrete, per-utterance analog of the intervention. Reranking-reasoning did **not
  distill** into a feedforward net — explicit inference-time reasoning mattered.
- **CodeRSA 2025 (arXiv:2502.15835)** — reverse-generate "what the audience thinks this says,"
  rerank by distinctiveness. Best in 10/12 code-gen settings; a reusable "reverse-infer reader
  interpretation" idea for prose.
- **CRSA 2025 (arXiv:2507.14063)** — formal per-turn belief over the listener's private state.
  Only marginal gains where that belief stays near-uniform — a caution that audience modeling
  helps most when the audience is *actually* uncertain/asymmetric.
- **Style Infusion 2023 (arXiv:2301.10283)** — audience preference as an external reward. A
  clean **non-CoT, non-ToM baseline** and a linguistic-feature evaluation suite.
- **Fried et al. 2023 survey (arXiv:2211.08371)** — argues exactly the hypothesis' premise and
  recommends **communication-based / self-play evaluation** (feed a sentence to a *separate*
  audience model, score comprehension + efficiency).

### Method analogs — per-sentence / reflective mental-state simulation
- **PercepToM 2024 (arXiv:2407.06004)** — prompt the LLM, per sentence, for *who perceived
  each piece of content*, filter to the target's perspective, then answer. Large false-belief
  gains (e.g. GPT-4o FANToM false-belief ~0.02 → 0.57). The **per-sentence "simulate whose
  view sees this"** template — but for QA, not writing.
- **Thought-Tracing 2025 (arXiv:2502.11881)** — incremental hypothesis-driven mental-state
  tracing as an inference-time agent. The most transferable scaffold for a generation-time
  "reader tracer" (code cloned).
- **ToM-agent 2025 (arXiv:2501.15355)** — infer partner Belief-Desire-Intention + confidence,
  predict their reply, counterfactually reflect. Improves dialogue success/efficiency; turn-level.

### Motivation & cautions — does default LLM ToM work?
- **Ullman 2023 (arXiv:2302.08399)** & **Shapira et al. 2023 (arXiv:2305.14763)** — default
  LLM ToM is brittle; adversarial variants collapse it. **Critical caution:** Shapira et al.
  find **CoT can inflate ToM scores via leaked task structure / shortcuts** — so a positive
  result from an audience-simulation CoT must be checked against *adversarial and true-belief
  controls*, not just averaged accuracy.
- **ToM survey 2025 (arXiv:2502.06470)** — perspective-taking prompting ("Think Twice",
  SimToM) already helps ToM QA; useful prior + baseline.
- **ALTPRAG 2025 (arXiv:2505.18497)** — pragmatic competence grows base→SFT→DPO; supplies a
  human-validated **10-point rubric + pairwise-win LLM-judge** protocol (dataset downloaded).

### Evaluation target — writing quality & homogenization
- **AI-Slop to AI-Polish 2025 (arXiv:2504.07532)** — the key testbed. WQ benchmark (LLMs ≈
  chance), WQRM reward model (74%), LAMP expert-edit corpus. **Generic CoT/reasoning models
  (o1, R1, o3) do NOT improve writing quality** — so if an *audience-simulation* CoT does, that
  is a meaningful, non-trivial result. Span-level self-editing + best-of-N *does* help (a strong
  comparison condition). Code+data cloned.
- **Homogenization 2026 (arXiv:2601.06116)** — formal diversity/"default-collapse" metrics and
  an LLM-judge ensemble with **per-sentence barycenter tracking** — a way to measure the
  "annoying sameness" the hypothesis targets, at the "every sentence or so" granularity.

## Common Methodologies
- **Simulate-then-select/steer**: generate candidate(s) → predict audience reaction → steer or
  rerank (RSA, Takmaz, CodeRSA, Andreas & Klein). Mechanism varies: gradient, likelihood rerank,
  reward, or verbalized reflection.
- **Perspective filtering**: explicitly model whose knowledge covers what, then condition on the
  audience's subset (PercepToM).
- **Reflect-and-update belief**: predict the interlocutor's response, measure the gap, revise the
  inferred mental state (ToM-agent, Thought-Tracing).

## Standard Baselines (for the experiment)
1. Plain generation (egocentric) — the "annoying" default.
2. Plain CoT ("think step by step") — controls for *added reasoning per se* (shown insufficient
   for writing quality in AI-Slop).
3. Perspective-taking prompt (SimToM / "Think Twice") — audience mentioned once, not per sentence.
4. Self-refine / edit-based revision (creativity_eval edit pipeline) + best-of-N.
5. RSA-style rerank by a *separate* listener/audience model (communication-based eval).

## Evaluation Metrics
- **Writing quality**: WQRM score + pairwise LLM-judge on the WQ benchmark; LAMP expert-edit
  agreement; human ratings on a small set (gold standard, per AI-Slop & Andreas & Klein).
- **Audience appropriateness / communication success**: feed each generation to a *dissimilar*
  audience model and measure comprehension / preference / task success (Fried et al.; self-play).
- **Pragmatic-reasoning diagnostic**: ALTPRAG rubric + pairwise win; FANToM set-ALL; ToMBench acc.
- **Homogenization / diversity**: deviance / barycenter entropy / distinct-n / Vendi-style, to
  test whether the intervention reduces generic sameness (Homogenization 2026).
- **Cost/fluency tradeoff**: length, tokens of overhead, fluency rating (RSA λ-tradeoff, Takmaz's
  disfluency warning — audience adaptation can *hurt* fluency if unconstrained).

## Datasets in the Literature (see `datasets/README.md` for what was downloaded)
- Generation: WritingPrompts (downloaded sample), LAMP (cloned), WQ benchmark (cloned).
- Audience/ToM diagnostics: ALTPRAG (downloaded), FANToM/ToMBench/BigToM/ToMi (cloned).
- Grounded audience games (optional): PhotoBook/Takmaz-2020, Colors-in-Context, Abstract Scenes.

## Gaps and Opportunities
- **G1 — The intervention itself is untested for writing.** No paper has the *generator*
  verbalize incremental audience simulation in CoT during open-ended prose and measured writing
  quality. This is the core novel contribution space.
- **G2 — Granularity is unstudied.** "Every sentence or so" vs once-up-front vs per-paragraph is
  an open design axis (PercepToM = per-sentence for QA; SimToM = once).
- **G3 — Generic-CoT confound.** Because plain reasoning doesn't help writing (AI-Slop), the
  experiment must isolate *audience simulation* from *more tokens of thinking*.
- **G4 — Shortcut inflation.** Per Shapira et al., verify gains with adversarial/control audiences
  and a *separate* judge/listener, not a same-family judge.
- **G5 — Fluency/verbosity cost.** Audience adaptation risks disfluent, abstract, or bloated prose
  (Takmaz); measure the tradeoff, not just quality-up.

## Recommendations for Our Experiment
- **Task**: WritingPrompts (or short-form: email/explanation/story) with an *explicitly varied
  target audience* (e.g., expert vs layperson vs child; skeptic vs enthusiast).
- **Conditions**: (i) plain, (ii) plain CoT, (iii) one-shot audience prompt (SimToM), (iv)
  **per-sentence audience-simulation CoT (the intervention)**, (v) self-refine/edit baseline.
  Hold total "thinking tokens" roughly constant between (ii) and (iv) to kill the G3 confound.
- **Metrics**: WQRM + pairwise LLM-judge (writing quality); a *separate* audience model's
  comprehension/preference (communication success); a homogenization/diversity metric; ALTPRAG
  as a mechanism check; fluency & length as cost. Add a small human eval if feasible.
- **Controls**: adversarial/mismatched-audience condition (does the model actually use the
  audience, or is it a generic-quality boost?); true-audience vs wrong-audience swap.
- **Reusable scaffolding**: `thought-tracing` (per-step tracer → adapt to a "reader tracer"),
  `creativity_eval` (WQRM + LLM-judge + LAMP), ALTPRAG rubric, Fried et al.'s self-play eval.
- **Practical**: use the `modal-vllm` skill for a separate open-model *audience/listener* and
  an LLM-judge; keep the generator and judge in different model families (G4).
