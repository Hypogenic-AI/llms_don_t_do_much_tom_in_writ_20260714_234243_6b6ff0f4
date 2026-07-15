# Resources Catalog

Resources gathered for **"LLMs Don't Do Much ToM in Writing"** — testing whether an LLM
that explicitly simulates the audience in its chain-of-thought (per sentence) writes better.

## Summary
- **Papers downloaded**: 14 (all as PDFs in `papers/`)
- **Datasets available**: 6 (2 downloaded to `datasets/`, 4 bundled in cloned repos)
- **Repositories cloned**: 4 (in `code/`)

## Papers (14)

| # | Title | Year | File | Role |
|---|-------|------|------|------|
| 1 | Speaking the Language of Your Listener (audience-aware, plug-and-play ToM) | 2023 | `papers/2305.19933_*.pdf` | ★ Closest prior work |
| 2 | Reasoning about Pragmatics with Neural Listeners and Speakers | 2016 | `papers/1604.00562_*.pdf` | ★ Seminal neural RSA |
| 3 | Audience-Centric NLG via Style Infusion | 2023 | `papers/2301.10283_*.pdf` | Non-CoT audience baseline |
| 4 | Pragmatic Reasoning Improves LLM Code Generation (CodeRSA) | 2025 | `papers/2502.15835_*.pdf` | RSA rerank method |
| 5 | Collaborative Rational Speech Act (multi-turn) | 2025 | `papers/2507.14063_*.pdf` | Formal per-turn audience belief |
| 6 | Pragmatics in Language Grounding (survey) | 2023 | `papers/2211.08371_*.pdf` | Framing + eval design |
| 7 | ToM-agent: Generative Agents w/ Counterfactual Reflection | 2025 | `papers/2501.15355_*.pdf` | ★ Predict-reader-then-reflect |
| 8 | Perceptions to Beliefs (PercepToM) | 2024 | `papers/2407.06004_*.pdf` | ★ Per-sentence perspective |
| 9 | LLMs Fail on Trivial Alterations to ToM Tasks (Ullman) | 2023 | `papers/2302.08399_*.pdf` | Motivation (brittle ToM) |
| 10 | Clever Hans or Neural ToM? Stress Testing | 2023 | `papers/2305.14763_*.pdf` | Caution: CoT can inflate |
| 11 | A Survey of Theory of Mind in LLMs | 2025 | `papers/2502.06470_*.pdf` | Background + prompting prior |
| 12 | The Pragmatic Mind of Machines (ALTPRAG) | 2025 | `papers/2505.18497_*.pdf` | Judge protocol + dataset |
| 13 | AI-Slop to AI-Polish (WQ/WQRM/LAMP) | 2025 | `papers/2504.07532_*.pdf` | ★ Writing-quality testbed |
| 14 | The Homogenization Problem in LLMs | 2026 | `papers/2601.06116_*.pdf` | Diversity metrics |

Full descriptions: `papers/README.md`. Full synthesis: `literature_review.md`.

## Datasets (6)

| Name | Source | Size | Task | Location | Status |
|------|--------|------|------|----------|--------|
| WritingPrompts | HF `euclaise/writingprompts` | 300 (sample) | open-ended writing | `datasets/writingprompts_sample/` | Downloaded (sample) |
| ALTPRAG | HF `Huangtubaye233/AltPrag` | 1,298 | pragmatic intent inference | `datasets/altprag/` | Downloaded (full test) |
| WQ Benchmark | `salesforce/creativity_eval` | 4,729 pairs | writing-quality preference | `code/creativity_eval/WritingRewards/WQ-benchmark-data/` | Cloned (JSON present) |
| LAMP | `salesforce/creativity_eval` | 1,282 pairs | AI-draft → expert edit | `code/creativity_eval/Writing_Alignment/LAMP/` | Cloned |
| ToMBench | `zhchen18/ToMBench` | 2,860 | ToM multiple-choice | `code/ToMBench/data/` | Cloned |
| FANToM / BigToM / ToMi | `skywalker023/*` | multi | conversational/false-belief ToM | `code/fantom/`, `code/thought-tracing/data/` | Cloned |

Details + download instructions: `datasets/README.md`. Data files are git-ignored
(`datasets/.gitignore`); samples + READMEs are tracked.

## Code Repositories (4)

| Name | URL | Purpose | Location |
|------|-----|---------|----------|
| thought-tracing | github.com/skywalker023/thought-tracing | ★ Incremental verbalized ToM tracer (adapt to a "reader tracer") | `code/thought-tracing/` |
| creativity_eval | github.com/salesforce/creativity_eval | ★ WQ benchmark + WQRM reward model + LAMP + edit pipeline | `code/creativity_eval/` |
| ToMBench | github.com/zhchen18/ToMBench | ToM diagnostic benchmark | `code/ToMBench/` |
| fantom | github.com/skywalker023/fantom | Info-asymmetric conversational ToM | `code/fantom/` |

Details: `code/README.md`.

## Resource Gathering Notes

### Search strategy
The paper-finder service was down (500 errors), so literature was gathered via the arXiv
API, the Semantic Scholar API (relevance + citation ranking), and targeted web searches for
canonical works (RSA, audience-aware adaptation, ToM benchmarks, writing homogenization).
All 14 PDFs were then **deep-read in parallel via a 14-agent workflow** producing structured
notes (contribution, method, datasets, baselines, metrics, results, code, relevance), which
directly populated `papers/README.md` and `literature_review.md`.

### Selection criteria
Prioritized (a) the closest prior work to the exact hypothesis (audience-aware generation,
RSA), (b) per-sentence / reflective ToM *method* analogs, (c) rigorous *writing-quality*
evaluation resources, and (d) motivation/caution papers on LLM ToM robustness. Preferred
papers with released code/data usable by the experiment runner.

### Challenges encountered
- **Paper-finder service unavailable** → fell back to arXiv + Semantic Scholar + web.
- **Semantic Scholar rate-limiting (429)** → added backoff/retry.
- **`uv add` hatchling build error** on the placeholder package → used `uv pip install`.
- **ELI5-category** uses a deprecated script loader (unusable) → dropped in favor of
  WritingPrompts + LAMP + WQ benchmark for the generation/quality side.
- **creativity_eval model weights are Git-LFS** → cloned with `GIT_LFS_SKIP_SMUDGE=1`; all
  JSON datasets present, checkpoints fetchable via `git lfs pull` if needed.

### Gaps and workarounds
No single public dataset offers `(writing task, target audience, quality label)`. Workaround:
combine a generation corpus (WritingPrompts/LAMP) with *author-defined audiences* and score
via the WQRM/LLM-judge + a separate audience/listener model (communication-based eval per
Fried et al.). ALTPRAG/FANToM serve as mechanism diagnostics.

## Recommendations for Experiment Design
1. **Primary datasets**: WritingPrompts + WQ/LAMP (generation & quality); ALTPRAG + FANToM
   (audience-modeling diagnostics).
2. **Baselines**: plain, plain-CoT, one-shot audience (SimToM), self-refine/edit, RSA-rerank —
   with total thinking-tokens matched between plain-CoT and the intervention.
3. **Metrics**: WQRM + pairwise LLM-judge; separate-audience comprehension/preference;
   homogenization/diversity; fluency & length cost; ALTPRAG mechanism check.
4. **Code to reuse**: `thought-tracing` (tracer scaffold), `creativity_eval` (quality eval),
   `modal-vllm`/`modal-training` skills for a separate audience model + judge.
5. **Must-have controls**: mismatched/adversarial audience (isolate audience use from generic
   quality gain); cross-family judge to avoid shortcut inflation (Shapira et al.).
