# Datasets

Resources for testing the hypothesis: *does having an LLM explicitly simulate the
audience in its chain-of-thought (roughly per sentence) improve its writing?*

Large data files are **not** committed to git (see `.gitignore`); small samples and
this README are. Follow the download instructions to reproduce locally.

The experiment naturally has two sides:
1. **Generation side** — writing tasks whose output quality we score with/without an
   audience-simulation CoT (WritingPrompts, LAMP, WQ benchmark).
2. **Diagnostic / audience-modeling side** — does the intervention actually improve
   the model's modeling of a reader's mental state? (ALTPRAG, ToMBench, FANToM, BigToM).

---

## 1. WritingPrompts (open-ended creative writing) — DOWNLOADED SAMPLE

- **Source**: HuggingFace `euclaise/writingprompts` (originally Fan et al. 2018, the
  Reddit r/WritingPrompts corpus).
- **Location**: `datasets/writingprompts_sample/train_300.json` (300-row sample, ~1 MB).
- **Format**: JSON list of `{"prompt": str, "story": str}`.
- **Task**: open-ended story generation from a one-line prompt. Good for measuring
  writing quality / homogenization and for defining explicit target audiences.
- **Why relevant**: a clean, audience-agnostic generation task. The intervention can
  add a target reader ("write for a tired parent" / "for a domain expert") and we test
  whether per-sentence audience simulation changes quality & appropriateness.

### Download (full set)
```python
from datasets import load_dataset
ds = load_dataset("euclaise/writingprompts")          # train/validation/test
ds.save_to_disk("datasets/writingprompts_full")
```
### Load the sample
```python
import json
data = json.load(open("datasets/writingprompts_sample/train_300.json"))
```

---

## 2. ALTPRAG — pragmatic intent inference (audience design diagnostic) — DOWNLOADED

- **Source**: HuggingFace `Huangtubaye233/AltPrag` (Bao et al. 2025, arXiv 2505.18497).
- **Location**: `datasets/altprag/altprag_test.jsonl` (1,298 rows, ~1.8 MB) +
  `datasets/altprag/samples.json`.
- **Format**: JSONL. Columns: `context`, `root` (initial utterance),
  `candidate_sentence_1/2` (two pragmatically distinct continuations),
  `candidate_sentence_1/2_intention` (reference intent explanation),
  `human_annotation_sentence_1/2_GM` (Gricean maxim flouted, or None).
- **Task**: given a context and two equally-valid-but-pragmatically-different replies,
  infer the speaker's intent / which Gricean maxim is flouted.
- **Why relevant**: a human-verified, scored measure of *audience design / speaker-intent
  reasoning*. Directly usable to check whether an audience-simulation CoT improves the
  model's pragmatic reasoning (measures the mechanism the hypothesis proposes). Paired
  with the paper's 10-point LLM-as-judge rubric + pairwise win-rate protocol.

### Download
```python
from datasets import load_dataset
ds = load_dataset("Huangtubaye233/AltPrag", split="test")   # only a 'test' split exists
ds.to_json("datasets/altprag/altprag_test.jsonl")
```

---

## 3. Writing Quality (WQ) Benchmark + LAMP corpus — CLONED (in `code/`)

- **Source**: `salesforce/creativity_eval` repo (Chakrabarty et al., "AI-Slop to
  AI-Polish?", COLM 2025, arXiv 2504.07532). Already cloned — see `code/README.md`.
- **Location**:
  - `code/creativity_eval/WritingRewards/WQ-benchmark-data/*.json` — 4,729 pairwise
    writing-quality judgments across 5 sub-datasets (Art-or-Artifice, LAMP-test,
    Style-Mimic, Synthetic-Mirror, LM-Arena). **Full JSON present (not LFS).**
  - `code/creativity_eval/Writing_Alignment/LAMP/LAMP.json` — 1,282
    `<AI-generated, expert-MFA-edited>` paragraph pairs with 1-10 quality scores.
  - `code/creativity_eval/WritingRewards/WQRM_annotations.json` — reward-model annotations.
- **Task / why relevant**: the single most on-point *writing-quality* resource. SOTA
  LLMs (incl. o1/R1 reasoning models) score **near chance** on WQ, and generic CoT does
  **not** help — so it is the ideal testbed to see whether an *audience-simulation* CoT
  beats plain CoT. LAMP supplies expert-edit "gold" revisions and the WQRM reward model
  gives an automatic quality score (74% agreement with experts).
- **Note**: model checkpoints (WQRM, edit model) are Git-LFS and were **not** pulled
  (`GIT_LFS_SKIP_SMUDGE=1`). Pull with `git lfs pull` inside the repo if the reward model
  weights are needed; the JSON data above is sufficient for evaluation-by-LLM-judge.

---

## 4. ToM benchmarks (audience/mental-state modeling diagnostics) — CLONED (in `code/`)

These measure whether the intervention improves the model's Theory-of-Mind, i.e. the
underlying capacity the hypothesis says is missing. All are **evaluation-only** (do not
train on them — contamination risk).

- **ToMBench** — `code/ToMBench/data/*.jsonl`: 2,860 bilingual multiple-choice items,
  8 tasks / 31 social-cognition abilities (Chen et al., ACL 2024, arXiv 2402.15052).
- **FANToM** — `code/fantom/` (+ `code/thought-tracing/data/fantom/fantom_v1.2.json`):
  info-asymmetric multi-party conversation ToM; a character leaves and rejoins unaware
  of what was said. Directly about *audience knowledge asymmetry* (Kim et al., EMNLP 2023,
  arXiv 2310.15421).
- **BigToM** — `code/thought-tracing/data/bigtom/bigtom_agree90.json`: causal-template
  false-belief reasoning (Gandhi et al. 2023).
- **ToMi (paraphrased)** — `code/thought-tracing/data/paraphrased_tomi/`: Sally-Anne
  style first/second-order false-belief QA.
- **MMToM-QA** — `code/thought-tracing/data/mmtom_qa/revised_questions.jsonl`.

---

## Optional / not downloaded (pointers for the experiment runner)

- **PercepToM benchmarks** (Percept-ToMi, Percept-FANToM; arXiv 2407.06004) — per-sentence
  perception-annotation ToM; closest comprehension analog to "simulate the audience per
  sentence." Released CC BY-NC 4.0 (check the paper page for the release URL).
- **PhotoBook / Takmaz-2020 referring expressions** (arXiv 2305.19933) — audience-aware
  referential game; code at `github.com/nicofirst1/speaker-adaptation`.
- **UKPConvArg1 + Cornell Movie-Quotes** (arXiv 2301.10283) — audience-preference pairwise
  data; code+data at `github.com/CrowdDynamicsLab/StyleInfusion`.
- **EmpatheticDialogues / PersuasionForGood** (arXiv 2501.15355) — turn-level ToM dialogue.

## Recommended primary datasets for the experiment
1. **WritingPrompts** (+ author-defined audiences) — generation task.
2. **WQ benchmark / LAMP** — automatic + human writing-quality scoring.
3. **ALTPRAG** — diagnostic that the intervention improves pragmatic/audience reasoning.
4. **FANToM** — diagnostic under explicit audience knowledge-asymmetry.
