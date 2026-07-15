# Downloaded Papers

14 papers on Theory of Mind (ToM), audience-aware / pragmatic generation, and LLM
writing quality — gathered for the hypothesis that having an LLM explicitly simulate the
audience in its chain-of-thought (per sentence) improves its writing.

Grouped by role. ★ = closest to the hypothesis / deep-read priority.

## A. Audience-aware / pragmatic generation (the closest prior work)

1. ★ **Speaking the Language of Your Listener: Audience-Aware Adaptation via Plug-and-Play Theory of Mind**
   — `2305.19933_speaking_language_of_listener_audience_aware_ToM.pdf`
   - Takmaz, Brandizzi, Giulianelli, Pezzelle, Fernández — ACL Findings 2023 — arXiv:2305.19933
   - Speaker adapts referring expressions by a simulator (ToM) predicting listener success;
     steers a frozen LM at decode time (plug-and-play, no fine-tune).
   - **Audience-aware > self-aware > baseline** on listener resolution (71.8/26.7% IND/OOD).
   - Code: https://github.com/nicofirst1/speaker-adaptation
   - Why: the paper closest to the hypothesis, but steers via hidden-state gradients, not CoT.

2. ★ **Reasoning about Pragmatics with Neural Listeners and Speakers**
   — `1604.00562_reasoning_about_pragmatics_neural_listeners_speakers.pdf`
   - Andreas & Klein — EMNLP 2016 — arXiv:1604.00562
   - Seminal neural RSA: sample from literal speaker, rerank by a simulated listener.
     +17% accuracy on reference game; tiny speaker weight preserves fluency.
   - Code: https://github.com/jacobandreas/pragma
   - Why: per-utterance "simulate the audience then pick" — the discrete analog of the intervention.

3. **Audience-Centric Natural Language Generation via Style Infusion**
   — `2301.10283_audience_centric_NLG_style_infusion.pdf`
   - Kumar et al. — EMNLP Findings 2023 — arXiv:2301.10283
   - Audience preference as an external BERT discriminator reward on GPT-2 (persuasive/memorable).
   - Code+data: https://github.com/CrowdDynamicsLab/StyleInfusion
   - Why: a non-CoT, non-ToM audience-modeling baseline + linguistic-feature eval suite.

4. **Pragmatic Reasoning Improves LLM Code Generation (CodeRSA)**
   — `2502.15835_pragmatic_reasoning_improves_llm_code_generation.pdf`
   - arXiv:2502.15835 (2025) — RSA-inspired execution-free reranker: reverse-generate what
     each candidate "says" to the audience, pick the distinctively-supported one.
   - Best in 10/12 settings. Code: none found in paper.
   - Why: operationalizes audience modeling as reverse-inference + rerank (reusable idea for prose).

5. **Collaborative Rational Speech Act: Pragmatic Reasoning for Multi-Turn Dialog (CRSA)**
   — `2507.14063_collaborative_rational_speech_act_multiturn_dialog.pdf`
   - arXiv:2507.14063 (2025) — information-theoretic RSA where the speaker maintains a belief
     over the listener's private meaning each turn. Code: https://github.com/LautaroEst/crsa
   - Why: formal "simulate the audience per turn"; marginal gains where audience belief ~uniform.

6. **Pragmatics in Language Grounding: Phenomena, Tasks, and Modeling Approaches** (survey)
   — `2211.08371_pragmatics_in_language_grounding_survey.pdf`
   - Fried, Tomlin, Hu, Patel, Nematzadeh — arXiv:2211.08371 (2023)
   - Thesis ≈ the hypothesis: LLMs communicate poorly because they don't model audience/context;
     RSA-style audience design helps. Recommends communication-based / self-play evaluation.

## B. ToM-aware generation with reflection/perspective (method analogs)

7. ★ **ToM-agent: LLMs as Theory of Mind Aware Generative Agents with Counterfactual Reflection**
   — `2501.15355_ToM_agent_generative_agents_counterfactual_reflection.pdf`
   - arXiv:2501.15355 (2025) — agent infers partner's Belief-Desire-Intention + confidence,
     predicts their reply, reflects on the gap. Improves dialogue success (SR@t up, turns down).
   - Code: promised post-acceptance, none found. Data: EmpatheticDialogues, PersuasionForGood.
   - Why: "predict the reader's reaction then reflect" — turn-level version of the intervention.

8. ★ **Perceptions to Beliefs: Precursory Inferences for Theory of Mind in LLMs (PercepToM)**
   — `2407.06004_perceptions_to_beliefs_precursory_inferences_tom.pdf`
   - arXiv:2407.06004 (2024) — per-sentence: prompt the LLM to tag who perceived each sentence,
     filter to the target's perspective, then answer. Big false-belief gains.
   - Why: the *per-sentence* "simulate whose perspective sees this" mechanism, for QA.

## C. Does default LLM ToM even work? (motivation + cautions)

9. **Large Language Models Fail on Trivial Alterations to Theory-of-Mind Tasks**
   — `2302.08399_llms_fail_trivial_alterations_tom_tasks.pdf` — Ullman — arXiv:2302.08399 (2023)
   - ToM-preserving perturbations flip GPT-3.5 to wrong beliefs → default ToM is brittle.

10. **Clever Hans or Neural Theory of Mind? Stress Testing Social Reasoning in LLMs**
    — `2305.14763_clever_hans_or_neural_tom_stress_testing.pdf` — Shapira et al. — arXiv:2305.14763
    - 15 LLMs × 6 ToM tasks; adversarial variants collapse performance. **Warns CoT can inflate
      results via shortcuts** → need adversarial/true-belief controls. Code: github.com/salavi/Clever_Hans_or_N-ToM

11. **A Survey of Theory of Mind in LLMs: Evaluations, Representations, and Safety Risks**
    — `2502.06470_survey_theory_of_mind_in_llms.pdf` — AAAI 2025 — arXiv:2502.06470
    - LLM ToM is real but non-robust; perspective-taking prompting ("Think Twice") helps.

12. **The Pragmatic Mind of Machines: Tracing the Emergence of Pragmatic Competence (ALTPRAG)**
    — `2505.18497_pragmatic_mind_of_machines_emergence.pdf` — arXiv:2505.18497 (2025)
    - Pragmatic competence rises base→SFT→DPO. Introduces **ALTPRAG** (downloaded) + a
      human-validated 10-point rubric / pairwise-win LLM-judge protocol.

## D. Writing quality & homogenization (the evaluation target)

13. ★ **AI-Slop to AI-Polish? Aligning LMs through Edit-Based Writing Rewards & Test-time Compute**
    — `2504.07532_ai_slop_to_ai_polish_edit_based_writing_rewards.pdf` — COLM 2025 — arXiv:2504.07532
    - **WQ benchmark** (4,729 pairs; SOTA LLMs ~chance), **WQRM** reward model (74%), **LAMP**
      expert-edit corpus, edit+best-of-N pipeline. **Generic CoT (o1/R1) does NOT help writing.**
    - Code+data: https://github.com/salesforce/creativity_eval (cloned).
    - Why: rigorous writing-quality testbed + strong negative baseline for "just add reasoning."

14. **The Homogenization Problem in LLMs: Towards Meaningful Diversity in AI Safety**
    — `2601.06116_homogenization_problem_in_llms_diversity.pdf` — arXiv:2601.06116 (2026)
    - Formalizes homogenization (collapse to a "default"); LLM-judge ensemble + per-sentence
      barycenter tracking + deviance/diversity metrics reusable to measure "annoying sameness."

See `../literature_review.md` for full synthesis and `../resources.md` for the catalog.
