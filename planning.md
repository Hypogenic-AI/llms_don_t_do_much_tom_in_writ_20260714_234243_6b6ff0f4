# Planning: Does Per-Sentence Audience Simulation in CoT Improve LLM Writing?

## Motivation & Novelty Assessment

### Why This Research Matters
LLM prose is often fluent-but-generic and "annoying" — it reads as if written for no one in
particular. The user's hypothesis is that this stems from weak, vague audience modeling, and
that forcing the model to *explicitly simulate the audience in its chain-of-thought, roughly
every sentence*, would make its writing land better. If true, this is a cheap, inference-time,
prompt-only fix (no fine-tuning) with broad impact on assistants, education, and communication.

### Gap in Existing Work (from literature_review.md)
Every prior "audience-aware generation" method puts audience modeling **outside** the
generator's own reasoning: a decode-time gradient (Takmaz 2023), rerank by a separate trained
listener (Andreas & Klein 2016; CodeRSA 2025), an external reward (Style Infusion 2023), or a
formal belief update (CRSA 2025). Per-sentence mental-state simulation *inside* natural-language
CoT exists only for **comprehension QA / dialogue** (PercepToM 2024; Thought-Tracing 2025;
ToM-agent 2025), never for **open-ended writing quality**. Meanwhile AI-Slop→AI-Polish (2025)
shows **generic CoT / reasoning models do NOT improve writing quality** — so any gain from an
*audience-simulation* CoT is non-trivial and must be isolated from "more thinking tokens."

### Our Novel Contribution
The first direct test of the user's exact intervention: the **generator itself verbalizing an
incremental, per-sentence audience simulation in its CoT during prose writing**, evaluated for
(a) audience-fit, (b) overall writing quality, and (c) genuine audience *adaptation* (does it
actually use the audience, or is it a generic quality boost?), against token-matched controls.

### Experiment Justification
- **E1 — Main comparison (4 conditions):** isolates the intervention from plain generation,
  generic CoT (token-matched control, kills the "more thinking" confound, G3), and one-shot
  audience prompting (SimToM). Needed to answer the core hypothesis.
- **E2 — Adaptation-gap control:** generate the same concept for audience A vs B; have the
  A-audience judge both. A method that *truly* models the audience should show a larger
  (score_for_A − score_for_B) gap. Isolates real audience use from generic quality (G4).
- **E3 — Homogenization / cost:** distinct-n & self-BLEU (does the intervention reduce generic
  sameness?) and length/thinking-token cost (does audience adaptation bloat or disfluent prose,
  per Takmaz's warning, G5).

## Research Question
Does having an LLM explicitly simulate the target audience in its chain-of-thought roughly once
per sentence produce writing that is (i) better fit to that audience, (ii) higher overall
quality, and (iii) more genuinely adapted to the audience — compared to plain generation,
token-matched generic CoT, and one-shot audience prompting?

## Hypothesis Decomposition
- **H1 (audience-fit):** per-sentence audience-sim > {plain, generic-CoT, SimToM} on a separate
  audience model's fit rating and pairwise preference.
- **H2 (quality):** per-sentence audience-sim > baselines on a cross-family quality judge.
- **H3 (adaptation):** per-sentence audience-sim shows the largest audience-adaptation gap
  (matched-audience minus mismatched-audience score).
- **H4 (not just tokens):** the intervention beats generic-CoT even though both add reasoning
  tokens (the G3 control).
- **H5 (diversity/cost):** effect on homogenization (distinct-n, self-BLEU) and length/tokens.

Independent variable: generation *condition* (prompting strategy). Dependent variables:
audience-fit score, pairwise quality win-rate, adaptation gap, diversity metrics, length/tokens.

## Proposed Methodology

### Task set
Primarily **audience-targeted explanatory writing** — the sharpest test of audience modeling:
"Explain {concept} for {audience}" (~150 words). Concepts span domains (science, finance, tech).
Audiences vary in expertise & disposition: an 8-year-old child; a busy non-technical executive;
a PhD-level domain expert; a skeptic who distrusts the topic. Plus a **creative** subset from
WritingPrompts with an assigned target readership, for breadth.
- ~12 concepts × 4 audiences = 48 explanatory tasks + ~12 creative = ~60 tasks.

### Conditions (all same generator model, temp fixed)
1. **plain** — write directly (the "annoying" egocentric default).
2. **cot** — "think step by step, then write" (generic reasoning; **token-matched control**).
3. **simtom** — describe the audience once up front, then write (audience mentioned once).
4. **persentence (INTERVENTION)** — write sentence-by-sentence; after each sentence, in a
   `<sim>` block simulate how the audience reads/reacts (confused? bored? convinced?), then
   choose the next sentence. Final clean text is extracted for judging.

### Models (kept in different families to avoid self-preference / shortcut inflation, G4)
- **Generator:** `openai/gpt-4.1-mini` (primary); `meta-llama/llama-3.3-70b-instruct` (replication).
- **Quality judge:** `anthropic/claude-sonnet-4.5` (cross-family, blind, order-randomized).
- **Audience model / listener:** `google/gemini-2.5-flash` (role-plays the audience for fit &
  comprehension — a *separate* model per Fried et al.'s communication-based eval).

### Baselines
plain, generic-CoT (token-matched), SimToM (one-shot audience) — per literature_review.md's
standard baseline list. The intervention is compared against all three.

### Evaluation Metrics
1. **Audience-fit (primary, H1):** audience model rates 1–10 fit for the named audience;
   pairwise "which is better for this audience" (both orders).
2. **Overall quality (H2):** cross-family judge pairwise win-rate (blind, order-balanced).
3. **Adaptation gap (H3):** A-audience judge scores same-concept text written for A vs for B.
4. **Homogenization (H5):** distinct-2, self-BLEU within each condition.
5. **Cost (H5):** output word count; reasoning/"thinking" tokens.

### Statistical Analysis Plan
Paired designs across tasks. **Wilcoxon signed-rank** on paired score differences; **bootstrap
95% CIs** for win-rates (H0: 50%) and mean differences; **rank-biserial / Cohen's d** effect
sizes. **Holm–Bonferroni** correction across the multiple condition comparisons. Significance
α = 0.05. Fixed seed = 42; temperature logged; every prompt/response cached & saved.

## Expected Outcomes
Supports H1–H4 if per-sentence sim beats all baselines (esp. generic-CoT) on audience-fit and
shows the biggest adaptation gap. Refuted / partial if it only matches SimToM (audience-once is
enough) or only matches generic-CoT (it's just more tokens) or if it bloats/disfluences prose.

## Timeline & Milestones
1. Env + data + task construction (done/early). 2. Generation harness + 4 conditions (cached,
concurrent). 3. Judge harness (quality + audience-fit + comprehension). 4. Adaptation-gap +
diversity + cost. 5. Stats + figures. 6. REPORT.md + README.md. Buffer ~25% for API flakiness.

## Potential Challenges
- **API flakiness / rate limits** → retry+backoff, disk cache keyed by request hash (reproducible,
  cheap re-runs). - **Judge bias / position bias** → cross-family judge, order-balanced, blind.
- **Token confound** → the generic-CoT control. - **Small N** → keep ≥48 tasks, paired tests,
  bootstrap CIs, report effect sizes not just p. - **CoT leakage inflating scores** (Shapira) →
  judge only the extracted final text, and use the mismatched-audience control.

## Success Criteria
A clear, statistically-supported answer to whether per-sentence audience simulation improves
audience-fit/quality over token-matched controls, with the adaptation-gap control showing
*why* (genuine audience use vs generic boost) — reported honestly whether positive or negative.
