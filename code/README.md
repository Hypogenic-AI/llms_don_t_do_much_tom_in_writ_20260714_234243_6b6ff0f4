# Cloned Repositories

Code + bundled data for baselines, ToM evaluation, and writing-quality evaluation.
No user-specified `code_references` were given in the topic spec, so these were selected
for relevance to the hypothesis (audience-simulation-in-CoT improves writing). All cloned
`--depth 1` (shallow).

---

## 1. `thought-tracing/` — Hypothesis-Driven ToM Reasoning (method analog) ★
- **URL**: https://github.com/skywalker023/thought-tracing (COLM 2025, arXiv:2502.11881)
- **Purpose**: Inference-time ToM reasoning — the model traces/updates hypotheses about a
  character's mental state step by step (a "tracer" agent) before answering. This is the
  **closest existing mechanism to the hypothesis**: explicit, verbalized, incremental
  mental-state simulation added to a frozen LLM. Swapping "character" → "reader/audience"
  and running it during *generation* is essentially the proposed intervention.
- **Key files**: `hypothesis.py`, `agents/{gpt,gemini,vllm,together_ai}.py`,
  `prompt_templates/*.txt`, `bigtom/test_bigtom.py`, `fantom/test_fantom.py`,
  `mmtom_qa/test_mmtom.py`, `revised_tomi/`.
- **Bundled data** (evaluation-only): `data/fantom/fantom_v1.2.json`,
  `data/bigtom/bigtom_agree90.json`, `data/paraphrased_tomi/`, `data/mmtom_qa/`.
- **Deps**: `environment.yml` (conda) — API keys or vLLM; adaptable to the modal-vllm skill.
- **Entry point**: e.g. `python fantom/test_fantom.py --model gpt-4o --use-tracing --tracer-type multi-tracer`.

## 2. `creativity_eval/` — Writing-quality evaluation (WQ / WQRM / LAMP) ★
- **URL**: https://github.com/salesforce/creativity_eval ("AI-Slop to AI-Polish?", COLM 2025, arXiv:2504.07532)
- **Purpose**: the primary **writing-quality testbed**. Provides the WQ benchmark, the WQRM
  reward model, the LAMP expert-edit corpus, and the edit + best-of-N pipeline.
- **Key files / data**:
  - `WritingRewards/WQ-benchmark-data/*.json` — 4,729 pairwise quality judgments (full JSON present).
  - `WritingRewards/WQRM_inference.py`, `WritingRewards/WQRM_annotations.json` — reward model.
  - `Writing_Alignment/LAMP/LAMP.json` — 1,282 AI-draft → expert-edited pairs with 1-10 scores.
  - `Writing_Alignment/{utils_generate_edits.py, prompts/}` — the 3-step edit pipeline.
  - `Art_or_Artifice/run_llm_eval.py` — LLM-judge writing eval harness.
- **Note**: cloned with `GIT_LFS_SKIP_SMUDGE=1`, so **model weights (LFS) were not pulled**;
  the JSON datasets and Python harnesses are present. Run `git lfs pull` inside the repo
  only if the WQRM/edit-model checkpoints are needed.

## 3. `ToMBench/` — Theory-of-Mind benchmark (diagnostic)
- **URL**: https://github.com/zhchen18/ToMBench (ACL 2024, arXiv:2402.15052)
- **Purpose**: bilingual multiple-choice ToM eval, 2,860 items across 8 tasks / 31 abilities.
- **Data**: `data/*.jsonl` (per-task). **Evaluation only — do not train on it.**

## 4. `fantom/` — FANToM conversational ToM (audience-asymmetry diagnostic)
- **URL**: https://github.com/skywalker023/fantom (EMNLP 2023, arXiv:2310.15421)
- **Purpose**: stress-tests ToM under information asymmetry in multi-party conversation
  (a character leaves, misses info, rejoins) — a direct probe of *audience knowledge modeling*.
- **Key files**: `eval_fantom.py`, `task/`, `agents/`. Data auto-downloads on first run
  (also bundled in `thought-tracing/data/fantom/`).

---

## Related repos NOT cloned (pointers)
- `github.com/nicofirst1/speaker-adaptation` — plug-and-play ToM speaker (arXiv:2305.19933).
- `github.com/jacobandreas/pragma` — neural RSA listener/speaker (arXiv:1604.00562).
- `github.com/CrowdDynamicsLab/StyleInfusion` — audience-centric style infusion (arXiv:2301.10283).
- `github.com/LautaroEst/crsa` — Collaborative RSA multi-turn (arXiv:2507.14063).
- `github.com/salavi/Clever_Hans_or_N-ToM` — adversarial ToM stress tests (arXiv:2305.14763).

## How these support the experiment
- **Intervention scaffold**: adapt `thought-tracing` prompt/agent structure to run a
  per-sentence "simulate the reader" tracer during generation.
- **Baselines**: plain generation, plain CoT, perspective-taking (SimToM / "Think Twice"),
  self-refine/edit (creativity_eval edit pipeline), RSA-style rerank.
- **Writing-quality scoring**: `creativity_eval` WQRM + LLM-judge + LAMP expert edits.
- **Mechanism diagnostics**: ALTPRAG (in `datasets/`), FANToM, ToMBench, BigToM.
