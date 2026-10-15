# 13. Multimodal and Speech Evaluation: MMMU, MathVista, Video Benchmarks, Image Generation Evals, and Speech, ASR, TTS and Voice-Agent Evals

## Overview

Text evaluation, the subject of most of this Atlas, has one property that makes life easy: input and output are both strings, so a grader can compare them. In multimodal evaluation that property breaks in one of three ways, and each way calls for a different architecture.

1. **Perception in, text out** (image, chart, video or audio understanding). The output is still text, so string-matching and LLM-judge machinery carries over. The new problem is that the *input* may not be needed. A question about an image can often be answered from the question text alone, from the options, or from what the model memorized. MMMU, MathVista, MMStar, MMMU-Pro, MathVerse and Video-MME are mainly answers to this problem.
2. **Text in, pixels or waveforms out** (image, video and speech generation). No reference string exists. The grader has to be a perception model (an object detector, a VQA model, an ASR system, a speaker-verification embedding), a learned preference model, or a human. GenEval, VQAScore, HPS v2, ImageReward, PickScore, VBench, Seed-TTS-Eval, UTMOS and the TTS Arena are the design points here.
3. **Real-time interaction** (voice agents). Here the *timing* of the output is part of correctness. A right answer delivered over the user's speech, or after a two-second silence, is a failure. Full-Duplex-Bench and τ-Voice evaluate by simulating whole conversations with a clock.

This chapter covers all three. It does not repeat generic material from earlier chapters on LLM-as-judge (Chapter 4), arenas (Chapter 5), agent harnesses (Chapter 6) and statistical validity (Chapter 10). It focuses on what changes when the modality changes.

A note on sourcing. Facts in this chapter come mostly from the official repositories and dataset cards that the benchmark authors maintain. These are often more current than the papers: several were updated in 2026, for example MMMU's public test answers, Video-MME-v2, WISE_Verified, the Open ASR Leaderboard's benchmark-fitting scorers, Full-Duplex-Bench v3 and τ-Voice. Numbers are as reported there, and leaderboard figures are snapshots.

```
          THREE EVALUATION ARCHITECTURES BY WHERE THE MODALITY SITS
+-----------------------------------------------------------------------------+
| (1) UNDERSTANDING        media + question --> model --> text --> extractor  |
|     MMMU, MathVista,                                         |              |
|     Video-MME, MMAU         rule match / LLM choice extractor --> accuracy  |
|     risk: question answerable WITHOUT the media (blind solvability)         |
+-----------------------------------------------------------------------------+
| (2) GENERATION           prompt --> generator --> image / video / audio     |
|     GenEval, VQAScore,                                  |                   |
|     HPS, VBench, Seed-TTS   detector | VQA model | ASR+WER | SV-embedding   |
|                             | reward model | MOS predictor | human arena    |
|     risk: grader is itself a model with blind spots; reward hacking         |
+-----------------------------------------------------------------------------+
| (3) INTERACTION          user simulator <==audio, clock==> voice agent      |
|     Full-Duplex-Bench,         |                               |            |
|     tau-Voice                  +-- turn-taking metrics, latency, tool calls,|
|                                    task success (DB state / judge)          |
|     risk: simulator realism; non-determinism of real-time APIs              |
+-----------------------------------------------------------------------------+
```

---

## 13.1 Image understanding: from "can it see?" to "does it need to see?"

### 13.1.1 MMMU: expert-level, discipline-spanning multiple choice

**Problem.** In 2023, VQA-style benchmarks such as VQAv2 tested everyday perception, and large models were saturating them. MMMU (Massive Multi-discipline Multimodal Understanding) set out to test "college-level subject knowledge and deliberate reasoning" [1].

**Design.** MMMU contains 11.5K multimodal questions drawn from college exams, quizzes and textbooks. It covers six disciplines (Art & Design, Business, Science, Health & Medicine, Humanities & Social Science, Tech & Engineering), 30 subjects, 183 subfields, and 32 "highly heterogeneous image types" such as charts, diagrams, maps, tables, music sheets and chemical structures [1]. The release is split three ways: 150 development samples for few-shot prompting, 900 validation samples for debugging and quick evaluation, and 10,500 test samples [1]. The test answers were originally withheld behind an EvalAI server. On 2026-02-12 the maintainers released them so the test set can be scored locally [1]. A held-out set therefore became a public one. Section 13.7 returns to what that means for contamination.

**Harness.** The official pipeline handles both multiple-choice and open-ended questions. Third-party harnesses re-implement it. EvalScope's integration, for example, runs zero-shot against OpenAI-compatible endpoints on the 900-item validation split. The maintainers state plainly that leaderboard-equivalent results require the official pipeline [1]. This "official versus community re-implementation" split recurs across almost every benchmark in this chapter.

**Failure mode that motivated a successor.** Multiple-choice questions with images can leak the answer through the text. Many MMMU items can be solved by a text-only LLM, or narrowed down by eliminating options.

### 13.1.2 MMMU-Pro: three filters against shortcutting

MMMU-Pro (ACL 2025) hardens MMMU with a three-step construction [1][2]:

1. **Filter out text-only-answerable questions.** Questions that text-only models can answer are removed, so every survivor needs the image.
2. **Augment candidate options.** More plausible distractors are added, which lowers the payoff from guessing and elimination.
3. **Vision-only input setting.** The question is *embedded inside the image* (a screenshot or photo of the question with its figure). The model has to "see" and "read" at once, with no text channel.

The authors report that accuracies on MMMU-Pro were "significantly lower than on MMMU, with accuracies ranging from 16.8% to 26.9% across various models". OCR prompts had minimal effect and chain-of-thought generally helped [1]. The vision-only setting is a general idea worth reusing: if the deployed product sees screenshots, test on screenshots.

### 13.1.3 MMStar: measuring the leak directly

MMStar (NeurIPS 2024) names two problems with the existing benchmarks: "visual content is unnecessary for many samples" and "unintentional data leakage exists in LLM and LVLM training" [3]. Starting from 22,401 samples, a coarse filter and then manual review cut the pool to 11,607 candidates and finally to 1,500 "vision-indispensable" items. These are balanced at 250 per core capability across 6 capabilities and 18 detailed axes [3].

The architectural contribution is the **submission protocol**. MMStar asks for *three* result files: the LVLM with visual input, the same LVLM without visual input, and the base LLM without visual input [3]. From these it derives two metrics, **Multi-modal Gain (MG)** and **Multi-modal Leakage (ML)**. MG is how much the image actually helps. ML is how much the multimodal training lifted text-only performance, which would show up as answers given without seeing the image. This is a cheap ablation that any multimodal eval pipeline can adopt.

### 13.1.4 MMBench: CircularEval and LLM choice extraction

MMBench (OpenCompass) contains 2,974 multiple-choice questions across 20 ability dimensions, organized in a three-level taxonomy (Perception and Reasoning at L-1, six L-2 abilities beneath them) [4]. It introduced two widely copied techniques:

- **CircularEval.** For an N-option question the model is queried N times, rotating the options (and the correct letter) each time. It is credited only if it is right in all N passes [4]. This removes position bias and lucky guesses, at N times the inference cost. The VLMEvalKit data format stores one image replica for all passes, while the legacy format stores N replicas [4].
- **LLM-based choice extractors.** Rule-based matching is tried first to save cost. If it fails, ChatGPT is given the question, the options and the free-form prediction and asked to map it to a letter, after which exact matching is retried [4].

```
            MMBench CIRCULAR EVALUATION (4-option item)
  pass 1:  A=cat  B=dog  C=fox  D=owl   gt=B  -> model "B"  ok
  pass 2:  A=dog  B=fox  C=owl  D=cat   gt=A  -> model "A"  ok
  pass 3:  A=fox  B=owl  C=cat  D=dog   gt=D  -> model "D"  ok
  pass 4:  A=owl  B=cat  C=dog  D=fox   gt=C  -> model "B"  FAIL
  item score = AND(all passes) = 0     cost = 4x inference
```

### 13.1.5 MathVista: math-in-pictures with a three-stage grader

**Problem.** Mathematical reasoning in visual contexts (function plots, geometry figures, tables, IQ-test puzzles, scientific figures) had not been studied systematically [5].

**Design.** MathVista (ICLR 2024 oral) contains 6,141 examples derived from 28 existing multimodal math datasets plus 3 new ones: IQTest, FunctionQA and PaperQA [5]. The leaderboard reports results on a 1,000-example *testmini* split, broken down by task (FQA, GPS, MWP, TQA, VQA) and by reasoning skill (algebraic, arithmetic, geometric, logical, numeric, scientific, statistical) [5].

**Grading architecture.** Models increasingly produced long answers instead of short ones, so MathVista settled on a three-stage evaluator [5]:

1. **Response generation.** The prompt includes a type-specific instruction, for example "requiring a floating-point number with two decimal places", or a Python list for list answers.
2. **Answer extraction.** An LLM (GPT-4) extracts the short answer from the free-form response. A preliminary study of 200 examples found the extraction more than 99.5% accurate [5].
3. **Score calculation.** The extracted answer is normalized to the required type (option letter, integer, float precision, list) and compared.

**Trajectory.** In October 2023 GPT-4V scored 49.9% on testmini, against 60.3% for humans (Amazon Mechanical Turk workers with at least a high-school diploma). Multimodal Bard scored 34.8%, and text-only GPT-4 given captions and OCR text scored 33.2-33.9% [5]. GPT-4o passed the human average in May 2024 with 63.8, and OpenAI o1 reached 73.9 in September 2024. By then the README listed eight models above the human baseline [5]. MathVista therefore went from "large headroom" to "saturated relative to crowd humans" in under a year. A crowd-worker baseline is a weak ceiling for an expert-domain benchmark.

**Third-party drift.** The EvalScope MathVista integration documents its own differences: a math parser instead of GPT-4 extraction, numeric-equivalence scoring, and "no same-prediction score-parity report against the official MathVista evaluator" [5]. This is the clearest public example of why the extraction stage has to be pinned when numbers are compared.

### 13.1.6 MathVerse, CharXiv, POPE and HallusionBench: targeted diagnostics

- **MathVerse** found that existing visual-math benchmarks put so much visual content in the *question text* that models could answer "without truly interpreting the input diagrams" [6]. It takes 2,612 diagram problems and has annotators rewrite each into **six versions** that move information between text and image, from text-dominant to vision-only. The result is about 15K test samples [6]. This is MMMU-Pro's vision-only idea taken to its conclusion: the same item rendered at several text-to-image information ratios.
- **CharXiv** (Princeton) uses 2,323 real charts from scientific papers, with descriptive and reasoning questions curated by human experts. On reasoning questions, GPT-4o scored 47.1%, the best open model (InternVL Chat V1.5) 29.2%, and humans 80.5% [7]. The motivation was that template-generated chart QA gave "an over-optimistic measure of progress" [7].
- **POPE** (EMNLP 2023) frames object hallucination as yes/no polling ("Is there a X in the image?"). Ground-truth objects come from annotations or from an automatic segmenter (SEEM). Negatives are sampled three ways, *random*, *popular* and *adversarial* (co-occurring objects), which gives three difficulty tiers [8].
- **HallusionBench** (CVPR 2024) separates *language hallucination* (the language prior overrides the image) from *visual illusion* (the vision module misreads the image). It uses yes/no questions grouped into **question pairs**: the same question asked about similar images, or with and without an image [9]. Its headline metric, question-pair accuracy, is harsh. GPT-4V (Sept 2023) scored 31.42% pair accuracy by human evaluation even though its per-question accuracy was 67.58% [9]. The README reports both human-graded and GPT-graded numbers, and the two differ by a few points (31.42 vs 28.79 for GPT-4V) [9]. That gap is a direct measurement of judge error.

---

## 13.2 Video understanding: duration, modality and consistency

Video adds three variables that images lack: **length** (seconds to hours), **frame sampling** (what the model actually sees), and **extra streams** (subtitles, audio). A video benchmark score that does not state its frames-per-video and subtitle setting cannot be reproduced.

### 13.2.1 Video-MME (2024): duration tiers and a subtitle protocol

Video-MME (CVPR 2025) contains 900 videos totalling 254 hours and 2,700 human-annotated QA pairs. Videos are bucketed as short (<2 min), medium (4-15 min) or long (30-60 min), ranging from 11 seconds to 1 hour, across 6 visual domains and 30 subfields. All data was newly collected and annotated rather than drawn from earlier datasets [10]. 744 of the 900 videos have subtitles, and every long video does [10].

Key protocol choices [10]:

- **Subtitle alignment rule.** In the with-subtitles setting, a model may use only the subtitles that match its sampled frames. With 10 frames, it gets the 10 corresponding subtitle lines. This stops a frame-starved model from reading the whole plot off the transcript.
- **Fixed prompt.** "Select the best answer... Respond with only the letter (A, B, C, or D)". Scoring is rule-based, and "the evaluation does not introduce any third-party models, such as ChatGPT".
- **Frame budget sensitivity.** In a June 2024 refresh, GPT-4o was re-run with 384 frames at 512x512 (instead of 10) and its overall accuracy rose to 71.9% [10]. Frame count is a first-order hyperparameter, not a detail.

Frontier labs adopted Video-MME quickly. OpenAI called it an "industry standard measure" of multimodal long-context ability for GPT-4.1. Google reported 84.8% for Gemini 2.5 Pro and used it again for Gemini 3 Pro, and GPT-5 reported it as well [10]. Section 13.7 discusses what such adoption does to headroom.

### 13.2.2 Video-MME-v2 (2026): grouped non-linear scoring

**Problem.** Video-MME's maintainers observed that "scores on existing benchmarks are saturating, yet a clear gap remains between leaderboard performance and actual user experience" [11].

**Design.** Video-MME-v2 (April 2026) has 800 videos, each with a word-level timestamped subtitle file, built from more than 3,300 human-hours of annotation by 60+ experts [11]. Its capability ladder has three levels: (1) multi-point information aggregation, (2) temporal understanding, (3) temporal complex reasoning. Questions come in **groups of four** that target either *consistency* or *coherence*. Options expand to eight (A-H), including "Cannot be determined" [11]. Subtitles can be supplied in two modes, *concatenated* or *interleaved*. In interleaved mode, words are placed between the frames whose time interval they fall in [11].

**Scoring.** A *non-linear group score* depends on getting all the related questions in a group right, not on scattered hits. The gap is large. Gemini-3-Pro and Gemini-3-Flash average 66.1% and 61.1% per-question accuracy but score 49.4% and 42.5% under group scoring. The ratio of group score to per-question accuracy is reported as a robustness index: about 75% for Gemini-3-Pro, about 56% for InternVL3.5-241B and about 40% for LLaVA-Video-7B [11].

**Findings with design implications** [11]:
- "Thinking" mode does not always help video. It gains more with subtitles than without, and some models regress (KimiVL-16B: -3.3 overall, -4.0 on Level 3 without subtitles).
- Models that take raw audio (omni models) do noticeably better on the frames-plus-audio dimension. Human baselines are measured with raw audio.
- Even the top model scores below 30 on Action & Motion and Physical World Reasoning.

```
     VIDEO-MME-v2 GROUPED SCORING (one group = 4 linked questions)
  Q1 retrieve fact   Q2 order events   Q3 causal link   Q4 inference
     correct            correct           wrong            correct
  per-question accuracy contribution : 3/4 = 0.75
  group (non-linear) score           : heavily discounted, because the
                                       reasoning chain is broken at Q3
  ratio NonLin/Avg across groups     -> robustness index per model
```

### 13.2.3 Long-video and knowledge-acquisition benchmarks

| Benchmark | Scale and format | Distinctive design choice |
|---|---|---|
| EgoSchema [12] | Very long-form egocentric clips, 5-way MCQ | Its authors describe it as "10x to 100x more difficult" in temporal reasoning length than almost all other video datasets. 7B+ open models scored <33% (random 20%), humans ~76% at release |
| MLVU [13] | Videos 3 min to 2 h, 9 tasks (dev); test adds Sports QA and Tutorial QA, 6-option MCQ | Mixes global and local ("needle") tasks. GPT-4o averaged 64.6% on MCQ at release. Test answers released March 2025 |
| LongVideoBench [14] | Interleaved frames plus subtitles | "Referring reasoning" questions that point to a specific moment. Accuracy is reported as a function of input frame count |
| Video-MMMU [15] | 300 lecture videos, 6 disciplines, 30 subjects, 900 QA (3 per video) | Perception, comprehension and adaptation levels, plus **Δknowledge** = (Acc_after_video − Acc_before) / (100% − Acc_before) |
| MVBench [16] | Multi-task temporal-understanding suite (released with VideoChat2) | Short-clip temporal skills. About 30 repeated QAs were later replaced, which the maintainers say may affect results by about 0.5% |

Video-MMMU's **Δknowledge** stands out because it measures *learning from the input* rather than prior knowledge. The model answers an exam question without the video, then again after "watching" a lecture that would let a human solve it. The normalized gain is the score [15]. This removes much of the contamination worry, since a model that already knew the answer gains nothing, and it tests the capability a tutoring or meeting-assistant product actually depends on.

---

## 13.3 Shared infrastructure: lmms-eval and VLMEvalKit

Two open harnesses run most academic multimodal evaluation. Their designs mirror the text-side harnesses in Chapter 3, with a few additions specific to media.

### 13.3.1 lmms-eval (LMMs-Lab)

lmms-eval describes a six-stage pipeline: CLI parsing → model resolution → task loading (YAML configs) → inference → response caching → metric aggregation [17]. The modality-specific engineering is the interesting part [17][18]:

- **Response cache keyed by content hash.** The key is a SHA-256 over schema version, task YAML, generation kwargs and document ID, so any prompt or config change invalidates stale results. Storage is SQLite plus a JSONL write-ahead log, fsynced before the upsert. `None` and empty responses are rejected before they are stored.
- **Media-aware batching.** Requests are clustered by shared image or video and sorted by length so vLLM or SGLang can reuse prefix KV cache. Video decode moved from single-threaded PyAV to multi-threaded TorchCodec, which the v0.7 release notes report as up to 3.58x faster.
- **Inference decoupled from scoring.** Outputs are persisted immediately, and judge-stage scoring can run later or on another machine.
- **Reasoning-tag stripping.** `<think>…</think>` blocks are removed before scoring, and the raw output is kept in a separate field.
- **Statistics (v0.6, Feb 2026).** Confidence intervals, **clustered standard errors** for "multiple questions per video", paired comparison against an anchor model, power analysis, and a stability measure that separates within-question model variance from question difficulty. The release notes warn that clustered SE "can be 3× larger than naive estimates" [18].
- **Throughput.** An adaptive-concurrency API path improved throughput about 7.5x on a fixed LIMIT=100 run (0.33 to 2.46 req/s) [18].
- **Audio.** v0.5 (Oct 2025) added an audio suite that scores with accuracy, WER or a GPT-4 judge [17].

The clustered-SE point matters more for video and audio than for text. Video-MME has 3 questions per video and Video-MME-v2 has groups of 4 per video, so treating items as independent understates uncertainty.

### 13.3.2 VLMEvalKit (OpenCompass)

VLMEvalKit is the official harness for MMBench and the OpenVLM Leaderboard. It uses generation-based evaluation for all models and reports **both exact-match and LLM-extracted** results [19]. 2025 changes show the same pressures from reasoning models: a `split_thinking` hook (strongly recommended for thinking models), TSV output because `.xlsx` cells truncate at 32,767 characters, and routing more multiple-choice items to LLM choice extractors, which "empirically leads to slight performance improvement" [19]. That last change is a reminder that harness updates can move scores without any change to the model. It also added multi-node inference through LMDeploy or vLLM, and Video-MME-v2 support in April 2026 [19].

```
          GENERIC MULTIMODAL UNDERSTANDING HARNESS (lmms-eval / VLMEvalKit)
 task YAML --> loader --> media fetch/decode --> prompt builder (frames, subs,
     |                    (TorchCodec, frame       options, CoT/direct flag)
     |                     sampler: N frames)              |
     |                                                     v
     |                                  model adapter (HF / vLLM / API)
     |                                                     |
     |                    cache (sha256 key) <-------------+
     |                                                     v
     |                    strip <think> --> rule extractor --fail--> LLM extractor
     |                                                     |
     +--> metric: acc | CircularEval | group score | WER | judge score
                     --> CI, clustered SE (by video/image), paired diff
```

---

## 13.4 Image and video generation evaluation

Generation evaluation has no reference answer. In practice the field converged on four grader families, each with known blind spots: (a) **discriminative perception models** that check prompt constraints (detectors, VQA models), (b) **learned preference or reward models** trained on human pairwise choices, (c) **MLLM judges** with rubric prompts, and (d) **human pairwise arenas** (Chapter 5).

### 13.4.1 GenEval: object detection as the grader

**Problem.** FID and CLIPScore give holistic quality or alignment numbers that are "unsuited for fine-grained or instance-level analysis" [20].

**Architecture.** GenEval (Ghosh, Hajishirzi, Schmidt) generates prompts from templates across six tasks: single object, two objects (co-occurrence), counting, colors, position, and color attribution (binding a color to the right object) [20]. The pipeline:

```
  prompt templates --> T2I model (4 images / prompt) --> Mask2Former detector
  (seeded generator)                                       (mmdetection)
                                                               |
                    boxes + masks + classes  ------------------+
                               |
       count check / relative-position check / color classifier on crops
                               |
           per-image {correct, reason} JSONL --> per-task + overall score
```

Color is verified by linking another discriminative vision model into the pipeline to check the detected object. The authors report "strong human agreement" for the detector-based judgments [20]. Published results range from 0.23 (minDALL-E) to 0.61 (IF-XL), with SDXL at 0.55. **Position** (0.03-0.15) and **color attribution** (0.00-0.35) were the weakest tasks for every model [20].

**Failure modes.** The prompt space is small and templated. The detector's class vocabulary (COCO-style) bounds what can be tested. Detector errors on stylized or unusual images count against the generator. A fixed, public, templated set is also easy to overfit, so later models report very high GenEval scores and the community moved to denser prompts (below). GenEval is best treated as a regression test for compositional basics, not as a frontier discriminator.

### 13.4.2 VQAScore and GenAI-Bench: a VQA model as the grader

VQAScore (ECCV 2024) asks an image-to-text model a templated question, for example `Is this figure showing "{prompt}"? Please answer yes or no.`, and uses the **probability of "Yes"** as the alignment score. In the v3.1 trace output this is the geometric-mean probability of the scored answer tokens [21]. The original recommended scorer was CLIP-FlanT5-XXL. The library now also supports Qwen2.5/3-VL, Qwen3.5, Gemma 3, GPT-4o and Gemini. Gemini scoring needs Vertex AI because log-probabilities are required, and the standard API key does not return them [21]. The authors report that VQAScore "significantly outperforms" CLIPScore and PickScore on compositional prompts while being much simpler than pipelines such as TIFA, Davidsonian Scene Graph and VIEScore [21]. Google's Imagen 3 report cited it as the strongest replacement for CLIPScore, and GenAI-Bench (compositional prompts with human ratings) has been used by Google DeepMind (Imagen 3 and 4), ByteDance Seed and NVIDIA [21].

**Design notes.**
- The question template shifts the score slightly. The maintainers ship defaults per model and advise against changing them, for reproducibility [21].
- The legacy v3.0 package is kept for reproducing the paper numbers, while v3.1 targets newer transformers versions [21]. Scorer *versions* must be pinned like any other dependency.
- The same machinery extends to video and, with Qwen3-Omni, to audio [21].

**DPG-Bench** (from Tencent's ELLA) uses a related question-decomposition approach for dense, long prompts. Following Davidsonian Scene Graph (DSG), it generates question-answer pairs per prompt. Users generate 4 images per prompt arranged as a 2x2 grid, and the released evaluation script scores the images against these questions [22].

### 13.4.3 Learned human-preference models: HPS v2, ImageReward, PickScore

| Model | Training signal | Scale | Notes |
|---|---|---|---|
| HPS v2 / v2.1 [23] | Human preference choices on generated images (HPD v2) | 798k choices on 430k images | Benchmark prompts in 4 styles (Animation, Concept-art, Painting, Photo). v2.1 trained on higher-quality data |
| ImageReward [24] | Expert pairwise comparisons | 137k pairs | Beat CLIP by 38.6%, Aesthetic by 39.6% and BLIP by 31.6% at predicting preference. Also used as a training reward (ReFL), and ReFL-tuned SD won 58.4% of human comparisons against the untuned model. Successor VisionReward is multi-dimensional |
| PickScore [25] | Real user choices from the Pick-a-Pic web app | v2 dataset >1M examples | CLIP-based scorer trained on v1 |

These models make "human preference" cheap enough to run on every checkpoint. But the HPS v2 leaderboard shows how compressed the score range can be. Model averages run from 23.55 (GLIDE) to 27.86 (Dreamlike Photoreal 2.0), and the top dozen models fall within about 0.6 points of each other [23]. Without an estimate of the scorer's own noise, small differences mean nothing. A more basic problem is that once a preference model is used as a **training reward**, as ImageReward explicitly proposes with ReFL [24], it stops being a valid **evaluation** metric for models trained against it, a direct case of Goodhart's law. Production teams keep a held-out human evaluation, or a different scorer family, for any model optimized against a reward model.

### 13.4.4 MLLM-judge benchmarks: WISE and WISE_Verified

WISE (ICML 2026) tests **world-knowledge-informed** generation: 1,000 prompts across cultural common sense, spatio-temporal reasoning and natural science, where the image is only correct if the model knows something, not merely if it maps words to pixels [26]. The original release scored images with GPT-4o-2024-05-13. In April 2026 the maintainers released **WISE_Verified** specifically to cut evaluation cost and to remove the dependency on a judge snapshot that might be deprecated [26]:

- The judge became an open model, **Qwen3.5-35B-A3B**, served through vLLM.
- About 200 prompts were refreshed. Some were too easy, and others triggered closed-model policy refusals during generation.
- The graded WiScore was **collapsed to binary 0/1** (correct world-knowledge realization and usable realism), dropping separate realism and aesthetic sub-scores.

This is a clear case of a general pattern. Closed-judge benchmarks drift toward open, pinned judges and coarser, more reliable labels as they mature. The same move appears in EmergentTTS-Eval, which re-ran its leaderboard when the judge `gemini-2.5-pro-preview-05-06` was deprecated (Section 13.5.4).

### 13.4.5 Video generation: VBench, VBench++ and VBench-2.0

VBench (CVPR 2024) breaks "video generation quality" into **16 dimensions**, each with its own prompt suite and its own evaluation method [27]. The dimensions are subject consistency, background consistency, temporal flickering, motion smoothness, dynamic degree, aesthetic quality, imaging quality, object class, multiple objects, human action, color, spatial relationship, scene, temporal style, appearance style, and overall consistency. Human preference annotations are collected per dimension to validate the automatic methods [27]. VBench++ (TPAMI 2025) adds image-to-video, long-video and trustworthiness (fairness, bias, safety) suites. VBench-2.0 targets "intrinsic faithfulness" (commonsense, physics, human motion, creative composition) with 18 fine-grained dimensions in five categories [27]. Six of the original dimensions accept custom prompts and videos. The rest are tied to the curated prompt suites [27].

The architectural lesson is that a single "video quality" number is not useful. A dimension like *dynamic degree* (how much motion) trades off directly against *temporal flickering* and *subject consistency*: a nearly static video scores well on consistency. Only a multi-dimension report shows that trade-off.

### 13.4.6 Human preference arenas for media

GenAI-Arena (TIGER-AI-Lab) reuses the FastChat controller-and-worker architecture from Chatbot Arena. Each generation or editing model (served through ImagenHub wrappers: SDXL, SDXL-Turbo/Lightning, PixArt-α, Playground v2/v2.5, Stable Cascade, InstructPix2Pix and others) runs as its own worker process registered with a controller [28]. The voting and Bradley-Terry machinery is covered in Chapter 5. The integrity mechanisms that a media arena needs are discussed with TTS Arena in Section 13.5.5, which documents them in unusual detail.

---

## 13.5 Speech: ASR and TTS evaluation

### 13.5.1 ASR: WER, normalization and the Open ASR Leaderboard

**Metric.** ASR is scored by word error rate (WER), and normalization decides most of the result. Whisper's model card shows why WER alone is not enough. The model was trained on 680,000 hours of weakly supervised web audio (65% English, about 18% non-English audio with English transcripts, 17% non-English with matching transcripts, covering 98 languages). Its documented failure modes are **hallucination** ("texts that are not actually spoken in the audio input"), **repetition loops**, and **uneven accuracy across languages, accents, and demographic groups** [29]. A single aggregate WER captures none of these.

**The Open ASR Leaderboard (Hugging Face).** This is the public reference design for reproducible ASR evaluation [30]:

```
  model family Docker image (HF Space config)  ---+
                                                  v
  HF Jobs: one job per dataset on fixed hardware (1x H200, 23 vCPU, 256 GB)
         |                 |                  |
   EN short-form      multilingual        long-form (earnings21/22, CORAAL)
   (main test sets)   (FLEURS, MCV, MLS)  chunked sets: concatenate chunk
         |                 |                  preds -> session-level WER
         v                 v                  v
   predictions JSONL --> repo EnglishTextNormalizer --> WER  +  RTFx
         |
         +--> results bucket (public) --> leaderboard + private-set eval
                                        --> benchmark-fitting diagnostics
```

Key properties [30]:
- **Fixed hardware for throughput claims.** Every English and multilingual short-form run is a Docker image on the same HF Jobs hardware, so the speed metric **RTFx** (inverse real-time factor) is comparable across models. The leaderboard plots RTFx against WER as a Pareto frontier rather than ranking on WER alone.
- **Published cost per run.** A full English short-form pass costs about $2.92 (parakeet-tdt-0.6b-v3), $4.75 (whisper-large-v3-turbo) or $5.58 (Qwen3-ASR-1.7B) at $5/hour H200 pricing.
- **API models** run in a local Docker container with per-provider concurrency limits, against the same datasets and normalizer.
- **Private datasets** are evaluated by the maintainers after submission, which guards against tuning to the public sets.
- **Long-form and chunked evaluation.** The Earnings22-Cleaned-AA chunked set splits 6 earnings calls into 341 chunks. Chunk predictions are concatenated per call before WER is computed against the call-level reference.
- **Cleaned references.** For VoxPopuli, 628 clips have human-corrected references (from Artificial Analysis) alongside the original parliamentary-record references.

**Benchmark-fitting diagnostics (2026).** The leaderboard ships two scorers that test whether a model's output "fits a benchmark's reference beyond what transcribing the audio explains". Both run on the published prediction manifests and need no audio or inference [31]:

1. **Reference-error agreement.** At 600 spans where the original VoxPopuli reference disagrees with the human correction, does the model reproduce the *wrong original* or the *correct audio*? The score is `rate = ref / (ref + consensus)`, with exclusions for near-normalization artifacts, very short outputs and third readings [31].
2. **Reference-rendering agreement.** Normalization erases spelling and formatting choices (`colour`/`color`, `Mr.`/`mister`, `twenty`/`20`). How often does the model reproduce the reference's *exact* rendering after getting the words right? [31]

The maintainers state that neither diagnostic "by itself prove[s] training-set contamination" [31]. Together they are a concrete, modality-specific contamination probe. A model that has memorized LibriSpeech or VoxPopuli transcripts will copy their errors and their house style. Text benchmarks have no equivalent signal this clean.

### 13.5.2 TTS objective metrics: WER-via-ASR and speaker similarity

Seed-TTS-Eval (ByteDance) became the de facto objective test set for zero-shot TTS and voice cloning. It has 1,000 English samples from Common Voice and 2,000 Mandarin samples from DiDiSpeech-2, plus a Mandarin "hard case" list and voice-conversion splits [32]. It uses two metrics:

- **Intelligibility as WER.** The synthesized audio is transcribed by Whisper-large-v3 (English) or Paraformer-zh (Mandarin) and compared with the input text.
- **Speaker similarity (SIM).** Cosine similarity between WavLM-large speaker-verification embeddings of the synthesized utterance and the reference clip [32].

```
  (prompt audio, prompt text, target text) --> TTS --> synthesized wav
                                                         |        |
                     ASR (Whisper-large-v3 / Paraformer) |        | WavLM-SV
                                                         v        v
                               WER vs target text    cos(emb_syn, emb_ref)
```

**Failure modes.** The ASR judge has its own error floor and biases: it normalizes away prosody and can "correct" mispronunciations its language model expects. SIM rewards copying the reference's channel and noise as well as the voice. Neither metric measures naturalness or expressiveness, which is why every serious TTS report adds a MOS predictor and human evaluation.

### 13.5.3 MOS prediction and aesthetics: UTMOS and Audiobox Aesthetics

- **UTMOS** (UTokyo-SaruLab, Interspeech 2022) is a mean-opinion-score predictor that ensembles "strong learners" (fine-tuned self-supervised speech models) with "weak learners" (classical regressors on SSL features). It took first place in 10 of 16 metrics at the VoiceMOS Challenge 2022 [33]. Its training data comes from past Blizzard Challenges and Voice Conversion Challenges, so it predicts *listening-test MOS for TTS of that era*. Its calibration on modern near-human systems is an open question.
- **Audiobox Aesthetics** (Meta, 2025) is a single model for speech, music and sound that predicts four axes instead of one MOS: **Content Enjoyment (CE), Content Usefulness (CU), Production Complexity (PC), Production Quality (PQ)** [34]. Splitting "is it pleasant" from "is it clean" from "is it complex" mirrors the move to multi-dimensional evaluation in VBench.

### 13.5.4 MLLM-judge TTS evaluation: EmergentTTS-Eval

As systems saturate WER and SIM, the remaining differences lie in prosody, emotion and hard text normalization. **EmergentTTS-Eval** (Boson AI, NeurIPS 2025 Datasets & Benchmarks) targets six scenarios: emotions, paralinguistics, foreign words, syntactic complexity, complex pronunciation (URLs, formulas) and questions [35].

**Architecture.** The architecture has three parts [35]:
- **Test-case generation.** A small set of human-written seed prompts is iteratively expanded by LLMs to target specific structural, phonetic and prosodic challenges, producing 1,645 cases.
- **Judging.** A Large Audio Language Model, gemini-2.5-pro with a thinking budget of 256, makes pairwise comparisons against a fixed baseline system (gpt-4o-mini-tts, voice Alloy). The result is reported as a **win-rate**, so the baseline sits at 50%. WER is reported alongside.
- **Strong prompting.** Some systems are run with explicit style instructions, marked ⭐ on the leaderboard.

**Results and judge stability.** On the current leaderboard Gemini-2.5-Flash-Preview-TTS leads with about 75% win-rate, and gpt-4o-audio-preview (Ballad, strongly prompted) has about 73%. ElevenLabs Multilingual v2 is at about 34%, and several open models fall below 30% [35]. WER and win-rate diverge: minimax speech-02-hd has one of the lowest WERs (10.07) but only a 36.8% win-rate, because intelligibility and expressiveness are different constructs [35]. When the preview judge was deprecated, the maintainers re-ran everything with the GA judge and reported a Spearman rank correlation of 0.98 between the two leaderboards [35]. This is a useful pattern: whenever a judge changes, publish the rank correlation between old and new judges.

### 13.5.5 TTS Arena: human pairwise preference with integrity engineering

TTS Arena (TTS-AGI) is a crowdsourced blind pairwise benchmark: "type a line, hear two anonymous models read it back, and pick the one that sounds more human" [36]. Its public documentation is a compact reference design for any media arena [37][38][39]:

**Rating.**
- A Bradley-Terry fit over the entire vote history, so ratings do not depend on vote order. Ratings are centered at 1500.
- Models are **sorted by the lower bound** of the confidence interval, so "a model can't shoot to the top off a handful of lucky wins".
- A model needs **100 votes** to appear and is marked *Preliminary* below 300.
- Light regularization toward the mean keeps an undefeated model from rating at infinity, and it fades within a few hundred votes [37].

**Vote validity.**
- Only votes on **unchanged prompts from the Random pool** move the public ratings. Custom typed prompts are allowed for side-by-side listening but never count. This keeps a provider from steering the prompt distribution.
- Listening gates: voting unlocks only after enough of each clip has been played.
- Hugging Face sign-in with accounts at least 30 days old. English only for now. Prompts capped at 1,000 characters [38].

**Anti-abuse.**
- Per-vote behavioral risk scoring. High-risk votes are recorded but **shadow-excluded**.
- A captcha, plus sweeps for coordinated rings (shared IP or device).
- A **statistical recognition test**: for each voter and model, how often the voter picks that model compared with everyone else, counting only battles where the model appeared and excluding the voter's own votes from the baseline. "Picking it almost every time it appears, when the crowd is split, is not a preference - it means you can tell which one it is" [38]. Such votes are excluded and the board is recomputed from the clean set.

**Provider integrity policy.**
- No affiliated voting and no un-blinding (no matching endpoint logs to arena sessions, no audio watermarks).
- The arena is not to be used for internal evaluation.
- Violators are **suspended**: they stay on the board, at the bottom, labeled with the reason and date, permanently [39].

```
              TTS ARENA VOTE PIPELINE
 Random-pool prompt --> 2 anonymous providers (router) --> listen gate
        |                                                     |
   custom prompt? -> listen-only, never counted               v
                                                         vote A/B
                                                              |
       risk score --high--> shadow-exclude (stored, not counted)
              |low
              v
   ring sweep + per-voter recognition test --flag--> retro-exclude
              |
              v
   Bradley-Terry over full clean history --> rank by CI lower bound
   (>=100 votes to list, <300 = Preliminary, suspended models at bottom)
```

The recognition test addresses a threat specific to audio and image arenas. Generated media carries **identifying signatures** (a house voice, a sampling artifact, a watermark), so blindness cannot be assumed the way it can for text. It has to be checked statistically.

---

## 13.6 Spoken dialogue and voice-agent evaluation

### 13.6.1 Audio-understanding benchmarks (speech in, text out)

| Benchmark | Content | Grader | Note |
|---|---|---|---|
| MMAU [40] | 10k clips across speech, environmental sound and music; 27 tasks (12 information-retrieval, 15 reasoning) | MCQ accuracy | Gemini 1.5 scored 66.15% and Qwen2-Audio 55.4% at release. The v05.15.25 revision rewrote ~25% of QAs and refined ~5% of audio, and the authors ask that the version be cited |
| AIR-Bench [41] | Foundation: 19 tasks, ~19k single-choice questions. Chat: ~2k open-ended | GPT-4 (gpt-4-0125-preview) for chat. Optional GPT alignment of free-form answers to ABCD | Covers speech, sound and music |
| VoiceBench [42] | Spoken instructions: AlpacaEval, CommonEval, WildVoice (crowd-recorded, diverse accents), SD-QA (by region accent), IFEval, AdvBench, OpenBookQA, MMSU, BBH | gpt-4o-mini judge for open-ended. Rule-based for QA, IF and safety | Each item runs in `audio` and `text` modality to isolate the cost of speech input |
| URO-Bench [43] | 40 datasets, 20 tasks, EN/ZH paired. Basic and Pro tracks | Mixed | First S2S benchmark covering multilinguality, multi-round dialogue and paralinguistics. 1,000-sample mini-set for quick runs |

VoiceBench's paired `--modality audio` / `--modality text` design is the speech counterpart of MMStar's with/without-image ablation. The gap between them is the **speech-interface tax**, and for cascaded systems it separates ASR error from reasoning error [42]. MMAU's versioning notice is a warning about benchmark revisions. A quarter of the answers changed, so scores across versions are not comparable [40].

### 13.6.2 Full-Duplex-Bench: turn-taking as a measurable behavior

Full-duplex models listen and speak at the same time. The interesting failures are *behavioral*: talking over a user who paused mid-thought, ignoring a "mm-hmm" backchannel, or failing to stop when interrupted. Full-Duplex-Bench (NTU, with NVIDIA advising on v3) has grown through four versions [44]:

- **v1.0 (static, offline).** Pre-recorded user audio streamed to the model in four scenarios: pause handling, backchanneling, smooth turn-taking and user interruption. Metrics: **Turn-Over Rate (TOR)**, where lower is better for pauses and backchannels and higher for smooth turns and interruptions; backchannel frequency; Jensen-Shannon divergence of backchannel timing against human distributions; response latency; and GPT-4o-judged contextual relevance after an interruption [45]. Time-aligned transcripts come from an ASR model. The maintainers switched to nvidia/parakeet-tdt-0.6b-v2 for "more reliable time-aligned transcriptions" [44].
- **v1.5.** Adds overlap scenarios (listener backchannel, side conversation, ambient speech), plus stop latency, response latency, prosodic adaptation and perceived speech-quality metrics [45].
- **v2 (real-time, dynamic).** A Node.js WebRTC/WebSocket orchestrator runs live conversations between the **Examinee** (model under test) and an automated **Examiner** agent, under constraints such as conversation speed and complex system instructions. Scoring uses NeMo ASR and Gemini LLM judges [46].
- **v3 (2026): tool use under disfluency.** 100 human-recorded examples (79 scenarios, 12 speakers, 48 kHz) with five annotated disfluency types: fillers, pauses, hesitations, false starts and self-corrections. The tasks need 1 to 3 chained calls to 12 mock APIs across 4 domains. Agents run in LiveKit rooms, either *native realtime* (GPT Realtime, Gemini Live, Grok, Ultravox) or *cascaded* (Silero VAD + Whisper + GPT-4o + TTS). A latency injector simulates slow APIs [47].

**FDB-v3 metrics** [47]:
- **Tool selection F1.** Precision penalizes extra calls.
- **Argument accuracy.** Exact match, or a GPT-4o semantic judge that handles `$RESULT_0.field` references and aliases.
- **Response accuracy.** Judged by GPT-4o.
- **Strict pass rate.** All expected tools called with correct arguments, none missing, none extra.
- **Three latencies**, all measured from the end of user speech: to the first word, to the first tool call, and to the key information in the response (located by GPT-4o).

### 13.6.3 τ-Voice: full-duplex customer-service simulation

Sierra's τ-bench (Chapter 6) simulates a user, gives the agent domain tools and a database, and scores task success from the end state. **τ-Voice** (2026) extends it to **full-duplex audio** [48][49]:

```
                     tau-VOICE DISCRETE-TIME SIMULATION (tick = 0.2 s)
 LLM user simulator --text--> ElevenLabs TTS persona --> audio effects
   (e.g. gpt-4.1)                                    (noise, bursts, frame drops)
        ^                                                  |
        |                                     G.711 mu-law 8 kHz telephony
        |                                                  v
   transcript  <-- STT (Deepgram nova / gpt-4o-transcribe) |
        |                                                  v
        |                      DiscreteTimeAdapter (per provider)
        |                 OpenAI Realtime | Gemini Live | xAI | Nova Sonic |
        |                 Qwen Omni realtime | LiveKit cascaded STT-LLM-TTS
        |                                                  |
        +------------- agent audio + tool calls <----------+
                                   |
                          domain tools / DB  --> end-state task scoring
  artifacts: stereo both.wav, Audacity label tracks (user, agent, tool calls)
```

Design details [48][49]:
- **Discrete-time adapter.** Every provider's streaming API is bridged to a tick-based simulation with a default tick of 0.2 s and a 600 s cap. A shared base class handles audio buffering and capping, proportional distribution of transcripts across ticks, **barge-in buffer clearing**, tool-result queuing and tick timing. Making real-time APIs fit a discrete clock is what makes runs replayable and comparable.
- **Speech-complexity presets as ablations.** `control` (clean audio, American accents, patient user) versus `regular` (background noise, accents, interruptions). Single-factor presets (`control_audio`, `control_accents`, `control_behavior`) and pairwise combinations attribute degradation to specific causes.
- **Realistic channel.** The user's audio is converted to telephony format (G.711 μ-law, 8 kHz), as in a real contact center.
- **Inspectable artifacts.** Stereo recordings with Audacity label tracks for user speech, agent speech and tool calls, so humans can audit failures.
- **Setup caveat.** The default voice personas are Sierra-internal ElevenLabs voices, so external users must create their own [48]. The simulator's voices are therefore not identical across labs, a reproducibility gap that should be disclosed with results.

### 13.6.4 What the voice-agent designs agree on

The two most rigorous public voice-agent benchmarks were built independently, yet they converge on these points:
1. **Score the task through tools or state, not through the transcript.** FDB-v3's strict pass rate and τ-Voice's database end state both work this way.
2. **Measure latency from the end of user speech**, broken into stages (first word, first tool call, key information).
3. **Inject real-world degradation as controlled factors** (disfluency types, noise, accents, telephony codec, API latency) so the score can be attributed to a cause.
4. **Compare native speech-to-speech with cascaded pipelines on the same harness.** Both benchmarks ship a LiveKit-based cascaded baseline.
5. **Keep an ASR or LLM judge in the loop only for what cannot be checked structurally**, such as response appropriateness or locating "key information".

---

## 13.7 Cross-cutting failure modes

| Failure mode | Where it shows up | Mitigation seen in the sources |
|---|---|---|
| Blind solvability (no media needed) | MMMU, MathVista-era benchmarks | Text-only filtering, vision-only input (MMMU-Pro) [1]; six-version information shifting (MathVerse) [6]; with/without-image submissions and ML metric (MMStar) [3] |
| Answer-format and extraction drift | MathVista, MMBench, VLMEvalKit | Three-stage pipeline with a validated extractor (>99.5% on 200 samples) [5]; rule-first then LLM extraction [4]; report both exact-match and extracted scores [19] |
| Position bias and lucky guesses | MCQ benchmarks | CircularEval [4]; more options (MMMU-Pro, MLVU-Test 6 options, Video-MME-v2 8 options) [1][13][11] |
| Inconsistent partial understanding | HallusionBench, Video-MME-v2 | Question-pair accuracy [9]; grouped non-linear scoring [11] |
| Unstated frame or subtitle budget | Video benchmarks | Frame-aligned subtitle rule [10]; interleaved vs concatenated modes [11]; report frame count |
| Correlated items understating error | Multi-question-per-video sets | Clustered SE, paired tests [18] |
| Grader-model blind spots | GenEval, VQAScore, Seed-TTS WER, UTMOS | Human-agreement validation [20][27]; multiple grader families; pinned scorer versions [21] |
| Reward hacking of preference models | ImageReward / HPS used in training | Keep a held-out human evaluation or a different scorer family [24] |
| Judge deprecation | WISE, EmergentTTS-Eval | Move to open pinned judge [26]; publish old-vs-new rank correlation [35] |
| Memorized references | ASR test sets | Reference-error and rendering agreement scorers [31]; private sets [30] |
| Un-blinded arenas | TTS and image arenas | Recognition test, shadow exclusion, provider policy [38][39] |
| Benchmark revision breaking comparability | MMAU, MVBench | Version strings in citations [40]; disclose replaced items [16] |
| Saturation against crowd baselines | MathVista | Expert baselines (CharXiv humans 80.5%) [7]; successor benchmarks |

**Contamination after test-set release.** MMMU (Feb 2026) and MLVU (Mar 2025) both published their test answers [1][13]. This helps reproducibility, but from that point those splits should be treated as potentially contaminated for models trained afterwards. Grouped scoring (Video-MME-v2), Δknowledge (Video-MMMU) and reference-error probes (Open ASR) are partial defenses, because they reward behavior that memorized answers alone do not produce.

---

## 13.8 Reference architecture: a multimodal eval stack for a product team

Combining the designs above, a team shipping a multimodal or voice product would run four layers:

```
 LAYER 4  Human preference          blind pairwise (arena-style), BT + CI lower
          (release gates)           bound, recognition test, fixed prompt pool
 ------------------------------------------------------------------------------
 LAYER 3  Interactive simulation    tau-Voice / FDB-style: simulated user audio,
          (voice / agent)           telephony codec, noise/accent ablations,
                                    tool-state scoring, latency percentiles
 ------------------------------------------------------------------------------
 LAYER 2  Generation graders        detector checks (GenEval-like), VQAScore,
          (per checkpoint)          ASR-WER + SV-SIM, MOS/aesthetics predictor,
                                    MLLM judge with pinned open model
 ------------------------------------------------------------------------------
 LAYER 1  Understanding suites      MMMU-Pro (vision-only), MathVista, CharXiv,
          (per checkpoint)          Video-MME(-v2) with fixed frames, MMAU,
                                    VoiceBench audio-vs-text; with/without-media
                                    ablation; clustered SE; pinned extractor
 ------------------------------------------------------------------------------
 INFRA    content-hash cache, media-aware batching, decoupled scoring,
          reasoning-tag stripping, fixed hardware for latency/RTFx claims
```

Cost guidance from the sources:
- Rule-based scoring (Video-MME, GenEval, ASR WER) costs nothing beyond inference.
- LLM extraction is invoked only when rules fail (MMBench, VLMEvalKit).
- Moving to open judges (WISE_Verified) is explicitly motivated by cost.
- Fixed-hardware ASR runs cost a few dollars per model for the English short-form suite [30].
- CircularEval multiplies MCQ inference by the number of options [4].
- Video costs grow with the frame budget, and GPT-4o's Video-MME refresh used 384 frames per video [10].

---

## Key takeaways for system designers

1. **Always run the "without media" ablation.** Score every understanding benchmark with and without the image, video or audio, and report the difference (MMStar's MG/ML, VoiceBench's audio-vs-text). A benchmark where the gap is small is measuring language priors.
2. **Test in the input format users actually send.** MMMU-Pro's vision-only setting, MathVerse's six versions and τ-Voice's 8 kHz telephony channel all show scores shifting when the input format changes. If users send screenshots or phone audio, evaluate on screenshots and phone audio.
3. **Pin and report every modality hyperparameter**: frames per video, resolution, subtitle mode and alignment, audio sample rate and codec, ASR normalizer, and extractor or judge version. GPT-4o's Video-MME result changed substantially from 10 to 384 frames.
4. **Prefer consistency-based scoring to per-item accuracy** wherever items can be grouped (CircularEval, HallusionBench pairs, Video-MME-v2 groups). The ratio of group score to item accuracy is a cheap robustness index.
5. **Treat generation graders as models with error bars.** Validate detectors, VQA scorers and MOS predictors against human labels on your own distribution. Pin scorer versions. Use at least two grader families, and never evaluate with the reward model you trained against.
6. **Separate constructs for speech output.** Intelligibility (WER via ASR), identity (SIM), quality or aesthetics (MOS, Audiobox axes) and expressiveness (LALM-judge win-rate, human preference) disagree, as EmergentTTS-Eval's WER-versus-win-rate divergence shows. One "TTS score" hides product-relevant trade-offs.
7. **For voice agents, make time a first-class output.** Measure turn-taking behavior (TOR, backchannels, interruption handling) and stage-wise latency from the end of user speech. Score task success through tool or state checks. Run under a discrete clock with factorized degradation presets so failures can be attributed.
8. **Do not assume an arena is blind; check it.** Media outputs carry fingerprints. Adopt fixed prompt pools, listening or viewing gates, ranking by the CI lower bound, per-voter recognition tests and a public integrity policy (TTS Arena).
9. **Build modality-specific contamination probes.** Copying a test set's reference errors or house style (Open ASR Leaderboard) is a stronger signal than n-gram overlap. Learning-gain metrics (Video-MMMU Δknowledge) are resistant to memorization by construction.
10. **Plan for judge deprecation.** Prefer open, self-hosted judges for long-lived benchmarks. When a judge changes, re-score everything and publish the rank correlation between old and new.

---

## References

1. Xiang Yue, Yu Su, Wenhu Chen et al. (MMMU team). "MMMU Benchmark" repository (evaluation code for MMMU, CVPR 2024, and MMMU-Pro, ACL 2025), 2023-2026. https://github.com/MMMU-Benchmark/MMMU
2. Xiang Yue, Tianyu Zheng, Yuansheng Ni et al. "MMMU-Pro: A More Robust Multi-discipline Multimodal Understanding Benchmark." arXiv, 2024. https://arxiv.org/abs/2409.02813
3. Lin Chen, Jinsong Li et al. "Are We on the Right Way for Evaluating Large Vision-Language Models?" (MMStar), NeurIPS 2024; repository. https://github.com/MMStar-Benchmark/MMStar
4. OpenCompass. "MMBench: Is Your Multi-modal Model an All-around Player?" repository and paper, 2023. https://github.com/open-compass/MMBench
5. Pan Lu et al. "MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts," ICLR 2024; repository and leaderboard, 2023-2026. https://github.com/lupantech/MathVista
6. MathVerse team (CUHK). "MathVerse: Does Your Multi-modal LLM Truly See the Diagrams in Visual Math Problems?" repository, 2024. https://github.com/ZrrSkywalker/MathVerse
7. Zirui Wang, Mengzhou Xia, Luxi He, Howard Chen, ..., Sanjeev Arora, Danqi Chen (Princeton NLP). "CharXiv: Charting Gaps in Realistic Chart Understanding in Multimodal LLMs," repository, 2024. https://github.com/princeton-nlp/CharXiv
8. Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, Ji-Rong Wen. "Evaluating Object Hallucination in Large Vision-Language Models" (POPE), EMNLP 2023; repository. https://github.com/AoiDragon/POPE
9. Tianrui Guan, Fuxiao Liu et al. "HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models," CVPR 2024; repository. https://github.com/tianyi-lab/HallusionBench
10. Chaoyou Fu, Yuhan Dai et al. "Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis," CVPR 2025; repository. https://github.com/MME-Benchmarks/Video-MME
11. Chaoyou Fu, Haozhi Yuan, Yuhao Dong et al. "Video-MME-v2: Towards the Next Stage in Video Understanding Evaluation," arXiv 2604.05015, 2026; repository. https://github.com/MME-Benchmarks/Video-MME-v2
12. Karttikeya Mangalam, Raiymbek Akshulakov, Jitendra Malik. "EgoSchema: A Diagnostic Benchmark for Very Long-form Video Language Understanding," 2023; repository. https://github.com/egoschema/EgoSchema
13. MLVU team. "MLVU: A Comprehensive Benchmark for Multi-Task Long Video Understanding," CVPR 2025; repository. https://github.com/JUNJIE99/MLVU
14. Haoning Wu, Dongxu Li, Bei Chen, Junnan Li. "LongVideoBench: A Benchmark for Long-context Interleaved Video-Language Understanding," 2024; repository. https://github.com/longvideobench/LongVideoBench
15. Kairui Hu, Penghao Wu, Fanyi Pu, Xiang Yue, Yuanhan Zhang, Bo Li, Ziwei Liu et al. "Video-MMMU: Evaluating Knowledge Acquisition from Multi-Discipline Professional Videos," 2025; repository. https://github.com/EvolvingLMMs-Lab/VideoMMMU
16. OpenGVLab. "Ask-Anything" repository (VideoChat2 and MVBench), 2023-2024. https://github.com/OpenGVLab/Ask-Anything
17. LMMs-Lab. "LMMs-Eval: Probing Intelligence in the Real World" repository and documentation (pipeline, caching, trust mechanisms), 2024-2026. https://github.com/EvolvingLMMs-Lab/lmms-eval
18. LMMs-Lab. "lmms-eval v0.6 release notes" (statistical rigor, throughput), 2026. https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/releases/lmms-eval-0.6.md
19. OpenCompass. "VLMEvalKit: A Toolkit for Evaluating Large Vision-Language Models," repository, 2023-2026. https://github.com/open-compass/VLMEvalKit
20. Dhruba Ghosh, Hanna Hajishirzi, Ludwig Schmidt. "GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment," 2023; repository. https://github.com/djghosh13/geneval
21. Zhiqiu Lin, Deepak Pathak, Baiqi Li, Graham Neubig, Pengchuan Zhang, Deva Ramanan et al. "VQAScore: Evaluating Text-to-Visual Generation with Image-to-Text Generation" (ECCV 2024) and "GenAI-Bench" (CVPR 2024 workshop); t2v_metrics repository, 2024-2026. https://github.com/linzhiqiu/t2v_metrics
22. Tencent QQGY Lab. "ELLA: Equip Diffusion Models with LLM for Enhanced Semantic Alignment" repository (DPG-Bench), 2024. https://github.com/TencentQQGYLab/ELLA
23. Xiaoshi Wu et al. "Human Preference Score v2: A Solid Benchmark for Evaluating Human Preferences of Text-to-Image Synthesis," 2023; repository (HPS v2/v2.1). https://github.com/tgxs002/HPSv2
24. THUDM (Zhipu AI / Tsinghua). "ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation," NeurIPS 2023; repository. https://github.com/THUDM/ImageReward
25. Yuval Kirstain et al. "Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation" (PickScore), 2023; repository. https://github.com/yuvalkirstain/PickScore
26. PKU-YuanGroup. "WISE: A World Knowledge-Informed Semantic Evaluation for Text-to-Image Generation" (ICML 2026) and WISE_Verified; repository. https://github.com/PKU-YuanGroup/WISE
27. Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang et al. "VBench" (CVPR 2024), "VBench++" (TPAMI 2025), Dian Zheng et al. "VBench-2.0" (2025); repository. https://github.com/Vchitect/VBench
28. TIGER-AI-Lab. "GenAI-Arena" repository, 2024. https://github.com/TIGER-AI-Lab/GenAI-Arena
29. OpenAI. "Whisper" repository and model card, 2022. https://github.com/openai/whisper/blob/main/model-card.md
30. Vaibhav Srivastav, Steven Zheng, Eric Bezzam et al. (Hugging Face and collaborators). "Open ASR Leaderboard: Towards Reproducible and Transparent Multilingual and Long-Form Speech Recognition Evaluation," arXiv 2510.06961, 2025; repository (as of July 2026). https://github.com/huggingface/open_asr_leaderboard
31. Hugging Face Open ASR Leaderboard maintainers. "Benchmark fitting" scorers documentation, 2026. https://github.com/huggingface/open_asr_leaderboard/blob/main/benchmark_fitting/README.md
32. ByteDance Speech. "seed-tts-eval" (objective test set and metrics for Seed-TTS), 2024. https://github.com/BytedanceSpeech/seed-tts-eval
33. Takaaki Saeki et al. (UTokyo-SaruLab). "UTMOS: UTokyo-SaruLab System for VoiceMOS Challenge 2022," Interspeech 2022; repository. https://github.com/sarulab-speech/UTMOS22
34. Meta AI (FAIR). "Meta Audiobox Aesthetics: Unified Automatic Quality Assessment for Speech, Music, and Sound," 2025; repository. https://github.com/facebookresearch/audiobox-aesthetics
35. Boson AI. "EmergentTTS-Eval: Evaluating TTS Models on Complex Prosodic, Expressiveness, and Linguistic Challenges Using Model-as-a-Judge," NeurIPS 2025 Datasets and Benchmarks; repository and leaderboard. https://github.com/boson-ai/EmergentTTS-Eval-public
36. TTS-AGI. "TTS Arena" repository, 2024-2026. https://github.com/TTS-AGI/TTS-Arena
37. TTS-AGI. "TTS Arena docs: Ranking," 2026. https://github.com/TTS-AGI/TTS-Arena/blob/main/apps/docs/content/docs/ranking.mdx
38. TTS-AGI. "TTS Arena docs: Voting," 2026. https://github.com/TTS-AGI/TTS-Arena/blob/main/apps/docs/content/docs/voting.mdx
39. TTS-AGI. "TTS Arena docs: Integrity policy," 2026. https://github.com/TTS-AGI/TTS-Arena/blob/main/apps/docs/content/docs/integrity.mdx
40. Sakshi S. et al. (GAMMA Lab, UMD). "MMAU: A Massive Multi-Task Audio Understanding and Reasoning Benchmark," 2024; repository (MMAU-v05.15.25). https://github.com/Sakshi113/MMAU
41. Qian Yang et al. (Alibaba / Zhejiang University). "AIR-Bench: Benchmarking Large Audio-Language Models via Generative Comprehension," ACL 2024; repository. https://github.com/OFA-Sys/AIR-Bench
42. Yiming Chen et al. "VoiceBench: Benchmarking LLM-Based Voice Assistants," 2024; repository. https://github.com/MatthewCYM/VoiceBench
43. Ruiqi Yan et al. "URO-Bench: A Comprehensive Benchmark for End-to-End Spoken Dialogue Models," arXiv 2502.17810, 2025; repository. https://github.com/Ruiqi-Yan/URO-Bench
44. Guan-Ting Lin, Hung-yi Lee et al. "Full-Duplex-Bench" (v1.0, v1.5, v2, v3) repository, 2025-2026. https://github.com/DanielLin94144/Full-Duplex-Bench
45. Guan-Ting Lin et al. "Full-Duplex-Bench v1/v1.5: static offline evaluation" documentation, 2025. https://github.com/DanielLin94144/Full-Duplex-Bench/blob/main/v1_v1.5/README.md
46. Full-Duplex-Bench team. "Full-Duplex-Bench-v2: A Multi-Turn Evaluation Framework for Duplex Dialogue Systems with an Automated Examiner" documentation, 2026. https://github.com/DanielLin94144/Full-Duplex-Bench/blob/main/v2/README.md
47. Guan-Ting Lin, Chen Chen, Zhehuai Chen, Hung-yi Lee. "Full-Duplex-Bench-v3: Benchmarking Tool Use for Full-Duplex Voice Agents Under Real-World Disfluency," arXiv 2604.04847, 2026; documentation. https://github.com/DanielLin94144/Full-Duplex-Bench/blob/main/v3/README.md
48. Sierra Research. "τ-bench (tau2-bench): Voice (Full-Duplex)" documentation, accompanying "τ-Voice: Benchmarking Full-Duplex Voice Agents on Real-World Domains," arXiv 2603.13686, 2026. https://github.com/sierra-research/tau2-bench/blob/main/src/tau2/voice/README.md
49. Sierra Research. "τ-bench Audio Native Providers architecture" documentation, 2026. https://github.com/sierra-research/tau2-bench/blob/main/src/tau2/voice/audio_native/README.md
