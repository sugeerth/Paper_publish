# 02. The Benchmark Landscape Through 2026: How Frontier Benchmarks Are Built and Graded

## Overview

A benchmark is a small piece of infrastructure. It has an authoring pipeline that produces items, a filter that throws out items that are too easy or wrong, a split policy that decides what is public and what is held back, a harness that runs the system under test, a grader that turns outputs into scores, and an aggregation rule that turns scores into a leaderboard number. When people say a benchmark is "saturated", "contaminated" or "broken", they almost always mean that one of these stages has failed. The model has rarely just gotten too good.

This chapter goes through the benchmarks that frontier labs and independent evaluators used most from roughly 2023 through October 2026. For each one it covers the problem it was meant to measure, how it is built, how it is graded, and what went wrong. The benchmarks fall into six families:

1. Static knowledge and reasoning exams with closed-form answers: MMLU, MMLU-Pro, GPQA and Humanity's Last Exam (HLE).
2. Fluid-intelligence puzzles with exact-match or efficiency scoring: ARC-AGI-1, -2 and -3.
3. Mathematics with machine-checkable answers: MATH/MATH-500, AIME via MathArena, and FrontierMath.
4. Code generation graded by unit tests: HumanEval and LiveCodeBench.
5. Repository-level and agentic software engineering in containers: SWE-bench, SWE-bench Verified, SWE-Bench Pro and Terminal-Bench.
6. Interactive agents and economic proxies: GAIA, tau-bench, BrowseComp, OSWorld, METR's time-horizon metric and GDPval.

The grading primitives explain most of the design differences, so they come first.

```
             GRADING PRIMITIVES USED ACROSS THE 2023-2026 LANDSCAPE
+----------------------+-----------------------------+------------------------------+
| Primitive            | Examples                    | What can go wrong            |
+----------------------+-----------------------------+------------------------------+
| Letter/option match  | MMLU, MMLU-Pro, GPQA        | label noise, guessing,       |
|                      |                             | answer-extraction regex      |
| Exact grid / string  | ARC-AGI-1/2, AIME, MathArena| ambiguous items, formatting  |
| LLM-judge equivalence| HLE, BrowseComp             | judge errors, parser bugs    |
| Symbolic/scripted    | FrontierMath                | wrong reference answers      |
|   answer check       |                             |                              |
| Unit tests (fn-level)| HumanEval, LiveCodeBench    | weak tests, sandbox flakiness|
| Repo test suites     | SWE-bench family            | too-narrow / too-wide tests  |
| Env-state checks     | tau-bench, OSWorld, TB 2.0  | unreachable goals, flaky VMs |
| Efficiency vs humans | ARC-AGI-3 (RHAE)            | baseline choice, rule changes|
| Expert pairwise      | GDPval                      | cost, rater disagreement     |
| Human-time calibrated| METR time horizon           | small long-task sample       |
+----------------------+-----------------------------+------------------------------+
```

Almost every benchmark below goes through some version of the same lifecycle:

```
  author items --> adversarial filter --> expert review --> split
  (humans,          (drop items frontier    (correctness,     |  public / semi-private /
   scraped repos,    models already solve)   ambiguity)       |  private / commercial
   contests)                                                  v
                                       harness (prompt, tools, container, VM, budget)
                                                              |
                                                              v
                    grader (regex | judge LLM | tests | state hash | humans)
                                                              |
                                                              v
             aggregate (acc, pass@k, pass^k, RHAE, win rate, logistic horizon)
                                                              |
                                                              v
             post-release audit --> errata / v2 / retirement  (the stage most
                                                                designs underbudget)
```

The last stage is the one that most often decides a benchmark's fate in practice. Between 2025 and 2026, most of the benchmarks in this chapter had a public audit, an errata release or a re-versioning: MMLU [3], HLE [9], SWE-bench Verified [36], SWE-Bench Pro [42], tau-bench [35], Terminal-Bench [50], OSWorld [56], FrontierMath [27] and ARC-AGI-3 [18].

## 1. Multiple-choice knowledge exams: MMLU, MMLU-Pro, GPQA

### 1.1 MMLU and its error floor

**Problem.** MMLU was built to measure broad academic knowledge across 57 subjects with four-option multiple-choice questions. Its weakness is now well documented: the ground truth itself is noisy. "Are We Done with MMLU?" (Gema et al.) re-annotated 5,700 questions across all 57 subjects to build MMLU-Redux. The authors estimate that 6.49% of MMLU questions contain errors, and that 57% of the analysed questions in the Virology subset are erroneous [3]. Re-scoring leading models on the corrected subset changed their reported metrics and their ranking [3].

**Lesson.** If roughly 6.5% of the labels are wrong, accuracy differences near the top of the leaderboard are within the label noise. An item set needs an error budget that is small relative to the gaps between the models you want to rank.

### 1.2 MMLU-Pro: more distractors, more reasoning

**Architecture.** MMLU-Pro (TIGER-Lab, NeurIPS 2024 Datasets track) keeps the MMLU format but makes three structural changes [1][2]:

- **Option expansion.** The number of options goes from 4 to 10. GPT-4 generated the extra plausible distractors, which lowers the chance of guessing correctly and makes shortcut elimination harder [1].
- **Source mix.** The 12,000+ questions span 14 disciplines (Biology through Psychology, plus "Others"). They draw on a filtered MMLU, with trivial and ambiguous items removed, together with STEM-website problems and TheoremQA [1][2].
- **Two-phase review.** Human experts first verified the answers. Then Gemini-1.5-Pro flagged likely false negatives (distractors that were actually correct), and humans adjudicated those flags. Some items therefore have fewer than ten options after bad distractors were removed [1].

**Results and properties.** Accuracy drops 16 to 33 points relative to MMLU. Prompt sensitivity, measured across 24 prompt styles, falls from 4-5% on MMLU to about 2%. Unlike on MMLU, chain-of-thought beats direct answering, which the authors take as evidence that the items require more reasoning [2].

**Failure modes.** A model wrote the distractors, so their quality is bounded by that model. An LLM also served as the false-negative detector, which ties the benchmark's correctness to the blind spots of the models it is meant to evaluate.

### 1.3 GPQA: "Google-proof" by construction

**Problem.** Rein et al. wanted questions that a skilled non-expert cannot answer even with unrestricted web access. Such questions are a testbed for scalable oversight, because if a human cannot verify an answer by searching, they need some other way to supervise a model that claims to know it [4].

**Architecture.** The pipeline is an expert-writer / expert-validator / non-expert-validator loop:

```
  domain expert writes Q + 4 options + explanation
          |
          v
  expert validator #1 answers, gives feedback --> writer revises
          |
          v
  expert validator #2 answers (measures objectivity)
          |
          v
  3 non-expert validators (PhDs in OTHER fields), unrestricted web access
          |
          v
  subsets: Extended (all)  >  Main (448)  >  Diamond (198: experts agree,
                                              <=1 of 3 non-experts correct)
```

The main set has 448 questions in biology, physics and chemistry. In the paper, experts reach 65% accuracy (74% once their own clear mistakes are discounted) and web-enabled non-experts reach 34% [4]. Most labs report the 198-question Diamond subset [5]. The repository ships the data in a password-protected zip and adds a canary string, a superset of the BIG-bench canary, so model builders can filter it out of training corpora [5].

**Grading.** Grading is a letter match. OpenAI's simple-evals harness used zero-shot chain-of-thought with an "answer regex tweak" for GPQA [21]. That is a reminder that extracting the answer from free text is part of the grader and can move scores.

**Status by 2026.** Third-party trackers describe Diamond as largely saturated. One tracker lists 24 of 136 models at 90% or above (approximate; a dated snapshot) [6]. With 198 items, one item is worth about 0.5 points, so the binomial confidence interval at 85-90% accuracy is roughly plus or minus 4-5 points. That alone makes small leaderboard gaps meaningless.

## 2. Humanity's Last Exam (HLE)

**Problem.** By late 2024 MMLU-class exams no longer separated frontier models. CAIS and Scale AI set out to build what they framed as the final closed-ended academic benchmark [7].

**Construction pipeline.**

```
  ~1,000 subject experts submit Q (MC or exact short answer, some multimodal)
          |
          v
  LLM difficulty gate: exact-match Qs must stump several frontier LLMs;
  MC Qs must be at or below chance across all LLMs   [10]
          |
          v
  human expert review, two rounds (refine, then approve)
          |
          v
  public set (2,500 Qs) + private held-out set (overfitting detection)
          |
          v
  post-release community bug bounty --> errata; dataset finalised Apr 3, 2025
```

The incentives were explicit: a $500,000 prize pool, paid as $5,000 for each of the top 50 questions and $500 for each of the next 500, plus optional co-authorship [7][10]. The released set has 2,500 questions across dozens of subjects, and the private held-out split detects overfitting [7]. The dataset carries its own canary string, again a superset of BIG-bench's [8].

**Grading architecture.** HLE popularised a judge-based extraction pattern. In the official harness [8]:

- The model under test answers with an explanation, an exact answer and a confidence from 0 to 100%. The README recommends `max_completion_tokens` of at least 8192 for reasoning models, because lower values caused collapse.
- A judge model (default `o3-mini-2025-01-31`; previously `gpt-4o-2024-08-06`) receives the question, the response and the reference answer. It returns a structured object with fields `extracted_final_answer`, `reasoning`, `correct` (yes/no) and `confidence`. The prompt tells the judge not to solve the problem itself and to answer "no" on any inconsistency or ambiguity, with "a small margin of error for numerical problems" allowed.
- Accuracy is reported with a Wald 95% interval.
- **Calibration error** is a binned RMS calibration error. Predictions are sorted by stated confidence and split into bins of 100 consecutive items. The per-bin gap |mean confidence - mean correctness| is squared, weighted by bin size, summed and square-rooted. As written in the reference code, the loop runs over `len(bins) - 1`, so the final bin, which holds the remainder, does not contribute [8].

At launch every frontier model scored below 10% accuracy, with calibration errors above 80%. In other words, models were confidently wrong [7]. A February 2026 leaderboard snapshot put Gemini 3 Pro Preview at about 37.5%, with calibration error still high (approximate) [10].

**Failure mode: adversarial filtering selects for wrong answers.** In July 2025 FutureHouse reported that 29% (plus or minus 3.7%, 95% CI) of HLE's text-only chemistry and biology answers directly conflict with peer-reviewed literature [9]. Their diagnosis is structural. Reviewers were told to spend only about five minutes checking answer correctness. At the same time, the LLM gate rewards questions that models get "wrong", and a question with a wrong reference key is exactly the kind that frontier models will "fail". The HLE team's own follow-up found about 18% of a Bio/Chem subset problematic, and in a three-expert re-review at least one reviewer disagreed 25% of the time [9].

**Lesson.** A filter that keeps only the items models get wrong will preferentially keep items whose answer key is wrong. Model-failure gates need independent verification of the key, budgeted in proportion to how adversarial the gate is.

## 3. ARC-AGI-1, -2 and -3

### 3.1 ARC-AGI-1: program synthesis on grids

Chollet's Abstraction and Reasoning Corpus (2019) targets skill-acquisition efficiency rather than skill [11]. Each task is a JSON file with about three `train` demonstration pairs and usually one `test` pair. A grid is a rectangular matrix of integers 0-9 (shown as colours), from 1x1 up to 30x30. The solver must produce the whole output grid, including its dimensions, and only exact cell-for-cell matches count. The original protocol allowed 3 trials per test input. The public release has 400 training and 400 evaluation tasks [11].

ARC Prize keeps semi-private and private evaluation sets so that leaderboard results cannot be tuned against public data. In December 2024 a pre-release OpenAI o3 scored 75.7% on the 100-task semi-private set within the public leaderboard's $10k compute limit. A high-compute configuration using about 172x more compute scored 87.5%. Secondary reports put that configuration at roughly $4.5k per task (approximate) [12]. The best 2024 Kaggle entry reached 53.5% on the private set. ARC Prize also noted that o3 had been trained on 75% of the public training set [12].

### 3.2 ARC-AGI-2: calibrated difficulty and cost reporting

ARC-AGI-2 (2025) keeps the grid format and fixes ARC-AGI-1's weaknesses [13]:

- **Human calibration.** About 400 people were tested on 1,417 unique tasks in controlled sessions. Every evaluation task was solved by at least two people in two attempts or fewer, and the average test-taker score was about 60% [13][14].
- **pass@2.** Two attempts per test input, with exact grid match [14].
- **Cost as a first-class axis.** The leaderboard plots accuracy against dollars per task, because brute-force search can buy accuracy.
- **Split hygiene.** There are public training, public evaluation, semi-private and private sets. The private set is used for Kaggle, where competitors run offline with limited compute.

How to state the human baseline has been contested. The technical report says final test pairs were solved on average by 75% of the people who attempted them. A LessWrong analysis argues that a figure closer to 53% better reflects the evaluation tasks, and that the larger number is inflated by easier tasks [15]. This is a recurring design question: "human-solvable" (some humans solve each item) and "human average" can differ by tens of points, and the headline claim depends on which one is used.

### 3.3 ARC-AGI-3: interactive environments scored on action efficiency

**Problem.** Static input/output puzzles cannot measure exploration, goal inference, memory or planning over many steps. ARC-AGI-3, launched March 25, 2026, is the series' first fully interactive benchmark [16][17].

**Architecture.**

```
  hand-built turn-based game (no instructions, no stated goal)
     levels L1..Ln, each with a target state
          |
   agent <-- frame (grid + available_actions + levels_completed) --+
     |                                                             |
     +-- action (ACTION1..ACTION7, reset) --------------------------+
          |
  per-level action count A_ai  vs  human baseline A_h
          |
  RHAE_level = min(cap, (A_h / A_ai)^2);  0 if A_ai > 5 * A_h
          |
  average across levels, games, private environments
```

- **Environments.** Human game designers built hundreds of turn-based environments. Agents get no rules or goals and must infer them by exploring [16]. Agents talk to the environments through an API. The open-source `ARC-AGI-3-Agents` toolkit exposes `FrameData` with `levels_completed`, `win_levels` and `available_actions`, and an action space that reaches `ACTION7` [20]. Since January 2026 the environments can also run locally, and third parties can author new ones [20].
- **Scoring: Relative Human Action Efficiency (RHAE).** Each level is scored as (human actions / AI actions) squared. An agent that takes more than 5x the human action count on a level is cut off and scores 0 for that level. Scores are capped, so finding an unintended shortcut earns no bonus [17][18].
- **Human baseline.** Every environment was solved by at least 2 of 10 human testers, usually 5 or more. The original baseline was the action count of the second-best tester, chosen to avoid outliers [19]. A secondary source reports an April 14, 2026 changelog entry that switched to the median human per level and raised the per-level cap from 1.0x to 1.15x. I could not verify this against the official docs, so treat it as unconfirmed [18].
- **Results.** At launch, frontier AI scored 0.26% and humans 100%. Every frontier model was below 1%, partly because exceeding the action cutoff zeroes a level [16].

**Lesson.** Squaring the efficiency ratio and adding a hard cutoff makes brute-force exploration expensive by design. That builds the ARC-AGI-2 "cost per task" idea into the metric itself. Any change to the baseline definition, such as moving from second-best to median, breaks score comparability, so versioning the scoring rule matters as much as versioning the data.

## 4. Mathematics: MATH, AIME/MathArena, FrontierMath

### 4.1 MATH and MATH-500

Hendrycks et al.'s MATH dataset (NeurIPS 2021) collects competition problems with boxed final answers [22]. By the reasoning-model era it was close to saturated. OpenAI's simple-evals moved models from o1 onward to MATH-500, described as "a newer, IID version of MATH" [21]. Grading normally extracts the boxed answer and checks equivalence, which is a source of variance between harnesses. simple-evals, for example, does not document its answer checker [21]. simple-evals itself stopped receiving updates for new models in July 2025, apart from reference implementations of HealthBench, BrowseComp and SimpleQA [21].

### 4.2 AIME and MathArena: evaluate at release time

**Problem.** Contest problems leak into training data within months. MathArena's answer is to evaluate models on a competition immediately after the problems are released, before they can be in anyone's training set [23].

**Architecture.** Each competition becomes a small dataset. AIME 2026 has 30 problems, extracted, converted to LaTeX and verified, and released under CC BY-NC-SA 4.0. Each model runs multiple times per problem; one tracker describes the protocol as 4 runs per problem with mean accuracy (approximate) [23][24]. The integer answers (000-999 for AIME) make exact-match grading unambiguous.

**Failure modes.** Thirty items give a coarse score. One problem is 3.3 points, so models near the ceiling are not statistically separable (trackers listed top models at about 96.7-98.3% on AIME 2026) [24]. The design depends on a steady supply of new contests, which is why MathArena expanded to other competitions.

### 4.3 FrontierMath: research-grade problems with automatic verification

**Architecture.** Epoch AI commissioned research mathematicians to write original, unpublished problems with closed-form answers that can be checked automatically, avoiding "subjective or costly evaluations" [26]. The structure is 300 core problems spanning Tiers 1-3 plus a 50-problem Tier 4 of research-level problems [26]. Almost all problems are private; only a handful of samples are published.

**Access and governance.** OpenAI funded the benchmark. Epoch disclosed this only when o3 was announced in December 2024, under an agreement that had prevented earlier disclosure. OpenAI owned and had access to most problem statements and solutions, apart from a holdout [29]. Epoch's governance response was a measurable holdout. On Tier 4, OpenAI has exclusive access to 28 problems and Epoch holds out 20. In one GPT-5.2 Pro run the model solved 5/28 (18%) of the accessible problems and 10/20 (50%) of the held-out ones, which Epoch read as no sign of overfitting to the accessible set [28].

**Errata.** Secondary sources report that a v2 of Tiers 1-4 was issued in June 2026 after an audit. Frontier models (GPT-5.5, Claude Opus 4.7) flagged candidate errors and mathematicians reviewed them. Reported figures include about 338 problems (295 in Tiers 1-3 and 43 in Tier 4) and errors addressed in roughly 42% of problems. Treat all of these as approximate until confirmed on Epoch's own pages [27].

**Lesson.** Even problems written by professional mathematicians and checked by script carry a substantial error rate once strong models start reading them adversarially. Using frontier models as auditors of a benchmark's reference answers has become standard practice.

## 5. Function-level code: HumanEval and LiveCodeBench

### 5.1 HumanEval and pass@k

OpenAI's HumanEval (2021) is a hand-written set of Python function-completion problems, each with a `task_id`, a `prompt` (signature plus docstring) and hidden unit tests [30]. It introduced the unbiased **pass@k** estimator. The harness recommends 200 samples per task and reports pass@1, @10 and @100, and pass@k is not computed when fewer than k samples exist [30]. It also set the safety norm for execution-based evaluation: the `exec` call is deliberately commented out so users read the sandboxing warning first [30]. A small detail is worth remembering: low-memory hosts can produce "malloc: can't allocate region" failures that make correct programs fail [30]. Infrastructure noise shows up as model error.

### 5.2 LiveCodeBench: time-windowed contamination control

**Problem.** Static code sets leak. LiveCodeBench (Jain et al., ICLR 2025) continuously collects problems from LeetCode, AtCoder and Codeforces contests and tags each with its release date [31].

**Architecture.**

```
  weekly contests (LeetCode / AtCoder / Codeforces)
        |  scrape statement + tests, tag release_date, difficulty
        v
  release_v1 (400 probs, to Mar 2024) ... release_v6 (1055 probs, to Apr 2025)
        |
        v
  scenarios: codegeneration | selfrepair | testoutputprediction | codeexecution
        |
        v
  score only problems with release_date > model cutoff  (--start_date)
```

The default "lite" generation set prunes many test cases to speed up evaluation while giving similar estimates, and `--not_fast` restores the full tests [32]. Generation uses n=10 at temperature 0.2 and reports pass@1 and pass@5. Timeouts can move scores by under 0.5 points [32].

**Evidence of contamination.** DeepSeek-Instruct and GPT-4o performed markedly worse on LeetCode problems released after their cutoff dates, which suggests the older problems were contaminated [31]. That is the clearest published demonstration of why time-windowing matters.

**Lesson.** A benchmark that keeps adding items stays fresh, but scores are only comparable across models on the same time window. A LiveCodeBench number without a date range is underspecified.

## 6. The SWE-bench family

### 6.1 SWE-bench: real issues, real test suites

SWE-bench (Princeton, 2023) turned resolved GitHub issues in popular Python repositories into tasks. The model receives the repository at the pre-fix commit plus the issue text and must produce a patch [25]. Grading uses two test sets taken from the human fix PR [25]:

- **FAIL_TO_PASS**: tests that fail before the reference PR and pass after it, which demonstrates the fix.
- **PASS_TO_PASS**: tests that pass both before and after, which guards against regressions.

A task counts as resolved only if both sets pass. The current harness runs every prediction in Docker containers for reproducibility. It recommends an x86_64 machine with at least 120 GB of free disk, 16 GB of RAM and 8 cores, and caps workers at `min(0.75 * cpu_count, 24)`. Results are cached by `run_id` and `instance_id`, so re-scoring a new patch needs a new `run_id`. The v5 CLI builds images from a separate task repository, and cloud runs go through sb-cli or Modal [33].

### 6.2 SWE-bench Verified (2024) and its retirement (2026)

**Construction.** OpenAI ran a human annotation campaign with 93 professional Python developers over 1,699 random SWE-bench samples. Annotators rated each sample on a 0-3 severity scale for whether the issue was well specified and whether the tests were appropriately scoped, and samples rated 2 or 3 were discarded. The result was the 500-task Verified subset, which OpenAI said supersedes the original test set and SWE-bench Lite [25].

**Retirement.** In February 2026 OpenAI stopped reporting Verified and recommended SWE-Bench Pro instead [36][37]. It cited two findings:

1. **Test validity.** OpenAI audited a 27.6% subset of problems that models often failed (138 problems that o3 consistently failed, per secondary summaries). At least 59.4% had flawed tests that reject functionally correct submissions. Some tests were too narrow and enforced implementation details; others were too wide and tested behaviour the issue never described [36].
2. **Contamination.** Frontier models, including GPT-5.2, Claude Opus 4.5 and Gemini 3 Flash, could reproduce gold patches and verbatim problem details, consistent with training exposure (reported via secondary summaries of the post) [36].

Progress had also slowed: state of the art moved only from 74.9% to 80.9% in six months [36]. Other labs adopted mitigations. Anthropic, for example, reportedly screens SWE-bench variants for memorisation and excludes flagged problems from headline numbers (secondary report) [36].

**Lesson.** At the top of the leaderboard, a test-based benchmark measures the tests' false-negative rate as much as the model. Once remaining failures are dominated by over-specified tests, further "progress" is mostly noise, or worse, it reflects memorised gold patches that happen to satisfy over-specified tests.

### 6.3 SWE-Bench Pro (2025) and V2 (2026)

**Problem.** SWE-Bench Pro was designed for long-horizon, multi-file tasks with contamination resistance built in through licensing and privacy [39][40].

**Architecture.**

```
  41 repos total
  +-----------------------+   +------------------------+   +-----------------------+
  | PUBLIC (731 tasks)    |   | HELD-OUT (858 tasks)   |   | COMMERCIAL (276)      |
  | 11 copyleft (GPL)     |   | 12 separate repos,     |   | 18 proprietary startup|
  | repos, released on HF |   | kept private to detect |   | repos; results public,|
  |                       |   | future overfitting     |   | data never released   |
  +-----------------------+   +------------------------+   +-----------------------+
                \__________________ human-verified, augmented _______________/
                        (issue text + requirements + interface spec)
```

There are 1,865 human-verified tasks in total [39][40]. The choice of copyleft repositories is deliberate: Scale argues that the "viral" nature and legal complexity of GPL-style licences make such code likely to be excluded from training corpora [40]. The commercial set uses startup codebases that cannot be released, but results on it are still reported [39]. At launch, top models scored about 23% on the public set, against 70%+ on Verified [40].

**V2 (released 9/22, per the repository).** Scale's review found 89 of the 731 public tasks invalid and dropped them, leaving 642 tasks across 11 repositories. Another 69 tasks had instructions that contradicted the tests grading them. Those instructions were corrected, and an expert then solved each task from the instruction alone [42]. A 51-task "HARD-51" subset collects the tasks most challenging across models [41][42]. Each V2 task is a self-contained **Harbor** task directory with a verifier, a reference solution and a public container image (`ghcr.io/scaleapi/swe-bench_pro-v2:<instance_id>`). Modal is the recommended runner [41]. The Hugging Face `default` config now silently returns V2, and V1 lives under the `v1` config. A third-party review warns that this makes old and new numbers easy to conflate [43].

Independent audits reached similar conclusions. A secondary report describes a Shanghai AI Lab-led audit of the v1 public split that revised 102 tasks and discarded 17, with some models' scores moving by about 19 points after environments were rebuilt (secondary; approximate) [44]. One secondary source also reports that OpenAI withdrew its Pro recommendation in July 2026 after estimating that about 30% of tasks were broken. I found this only in secondary reporting, so treat it as unconfirmed [38].

**Lesson.** Splitting a benchmark by licence and ownership (public, held-out, commercial) is the most robust contamination control in the coding space. It does not fix test validity, which still needs instruction-only expert re-solves of the kind done for V2.

## 7. Interactive and agentic benchmarks

### 7.1 GAIA: simple for humans, hard for assistants

GAIA (Meta, Hugging Face and AutoGPT; ICLR 2024) poses real-world questions that need reasoning, multimodal handling, web browsing and tool use, yet each has a single short factual answer that allows automatic scoring [45]. There are 466 questions. Questions are public, and answers to 300 are withheld to power the Hugging Face leaderboard [45]. At launch humans scored 92% against 15% for GPT-4 with plugins [45]. GAIA's lasting design idea is to make the *process* hard and the *answer* trivially checkable. BrowseComp later sharpened the same idea.

### 7.2 tau-bench, tau2-bench and tau3-bench: tool-agent-user loops graded by end state

**Problem.** Customer-service agents must follow a domain policy, call tools and converse with a user who reveals information gradually. Sierra's tau-bench (Yao et al., 2024) models this as three parties [34].

```
  +------------------+   natural language   +-----------------+   API calls
  | user simulator   | <------------------> |  agent under    | -------------> +-------+
  | (LLM + scenario, |                      |  test (policy   |                | DB /  |
  |  optional tools) |                      |  doc + tools)   | <------------- | env   |
  +------------------+                      +-----------------+   results      +-------+
                                                                                   |
   reward = DB_state_hash(final) == DB_state_hash(replay(gold_actions))           |
            x  all communicate_info strings present in agent messages    <---------+
```

- **Grading by state, not by path.** The final database state is compared with the state produced by replaying the annotated gold actions on a fresh environment, so any trajectory that reaches an equivalent state passes. A COMMUNICATE check additionally requires required facts to appear in the agent's messages. The default `reward_basis` is `[DB, COMMUNICATE]` and the components multiply. ACTION-matching, which accepts only one trajectory, is used only in a small subset of banking tasks. NL-assertion LLM-judging is marked experimental [35].
- **pass^k.** This is the probability that *all* k independent trials of a task succeed. It measures reliability, the opposite emphasis from pass@k. In the original paper, GPT-4o-class agents succeeded on under 50% of tasks, with pass^8 below 25% in retail [34].
- **Evolution.** tau2-bench adds a telecom domain under a "dual-control" framing, where both the user and the agent act on a shared environment [35]. The tau3 release adds full-duplex voice through realtime providers and a `banking_knowledge` retrieval domain. It also applies 75+ task fixes across airline, retail and banking, based on the SABER analysis, plus a v1.0.1 grading fix for banking that makes pre-1.0.1 scores non-comparable. Affected leaderboard entries were re-graded [35].

**Failure modes.** The user simulator is itself a model (gpt-4-0613 in the original paper), so simulator errors become agent errors [34]. Gold action sequences can encode one policy interpretation, which is the motivation for the task-fix waves.

### 7.3 BrowseComp: hard to find, easy to verify

**Construction.** OpenAI's BrowseComp (April 2025) inverts the usual design. Questions are trivial to verify once the facts are found, but finding them requires navigating many web pages. Human trainers checked that each question could not be solved by GPT-4o, o1 or early Deep Research versions [46]. About 21 tasks with ambiguous or incorrect references were later removed, taking the set from 1,287 to 1,266 [46]. Humans given a two-hour limit solved 29.2% of problems and abandoned 70.8% [46].

**Grading.** The reference implementation stores each problem and answer XOR-encrypted with a SHA-256-derived key, with a per-row `canary` as the password. This is obfuscation against casual scraping, not secrecy [48]. The model must output `Explanation / Exact Answer / Confidence`, and a grader LLM applies a template copied verbatim from HLE's judge prompt [48]. The grader code is a cautionary example. In the reference implementation at the time of writing, `grade_sample` returns `match.group(0)` (e.g. "correct: yes") from the regex `correct: (yes|no)`, and the caller then compares that value to the literal "yes" [48]. Taken literally, that comparison can never succeed, which shows how a one-character grader bug can silently zero a metric. Anyone forking a reference grader should unit-test it against known-correct responses.

**Test-time scaling.** At launch Deep Research scored 51.5%, against 1.9% for GPT-4o with search and 9.9% for o1 [46]. With 64 samples per question, majority voting, confidence-weighted voting and best-of-N by confidence each improved accuracy by 15-25% over a single attempt. The paper attributes this to the asymmetry: if an answer is easy to verify, the model can tell when it is right [47].

### 7.4 Terminal-Bench 1.0 and 2.0

**Architecture.** Each task is an English instruction, a Docker environment, a test script that verifies completion and a human-written oracle solution [49]. Terminal-Bench 1.0 (terminal-bench-core v0.1.1) had about 100 beta tasks run through the `tb` CLI [49]. Terminal-Bench 2.0 (paper arXiv 2601.11868) has 89 harder tasks, each manually checked by three reviewers and given "several hours of human and LM-assisted validation" to confirm it is solvable, realistic and well specified [50][52]. Example domains include assembling proteins for synthesis, debugging async code and resolving security vulnerabilities [52].

**Harbor.** TB 2.0 moved to the Harbor harness, which grew into a general framework. It runs arbitrary installed agents (Claude Code, Codex CLI, OpenHands, the reference Terminus-2 scaffold) across Docker or cloud sandboxes (Daytona, Modal and others) with thousands of environments in parallel, and it produces RL rollouts from the same task format [52][53]. The `--agent oracle` mode, which replays reference solutions, serves as a task-health check before any model is run [52]. Harbor's task format has since been adopted by others, including SWE-Bench Pro V2 [41].

**Scores and versions.** At publication, frontier models resolved under 65% of tasks and small models about 15% [50]. By March 2026 leaderboard entries had reached about 74-78% (approximate; scaffold-dependent) [51]. Terminal-Bench 2.1 (May 6, 2026) keeps the 89 tasks but repairs 28 of them, which generally raises scores [50].

**Lesson.** On agentic benchmarks the leaderboard ranks (model, scaffold) pairs. Publishing a reference scaffold and an oracle agent separates "task broken" from "agent weak".

### 7.5 OSWorld and OSWorld-Verified

**Architecture.** OSWorld (NeurIPS 2024) runs multimodal agents in real virtual machines across Ubuntu, Windows and macOS on 369 tasks involving real web and desktop apps, file I/O and multi-app workflows [54]. Each task record carries a `snapshot` (the initial VM image), a natural-language `instruction`, a `config` of setup scripts that download and open files to build the initial state, `related_apps`, and an `evaluator` pointing to a **task-specific execution-based evaluation script** that inspects the final machine state [55].

```
  VM snapshot --config scripts--> initial state --> agent loop:
      screenshot / a11y tree --> model --> pyautogui action (max_steps budget)
                                                   |
  evaluator script: read files/app state/URLs from VM --> compare to expected --> 0/1
```

Humans completed over 72.36% of tasks. The best model at publication reached 12.24%, held back mainly by GUI grounding and operational knowledge [54]. Eight Google Drive tasks may need manual configuration, so some runs use 361 tasks [54]. Providers include VMware, VirtualBox, Docker with KVM and AWS. On AWS a full parallel evaluation can finish within about an hour [56].

**OSWorld-Verified (July 28, 2025)** fixed community-reported task issues, broadened AWS support and changed the benchmark signals enough that the maintainers ask users to compare against new results only. Verified leaderboard entries require the maintainers to run the agent themselves, or trusted institutions to share trajectories, together with a report explaining the agent's behaviour [56].

## 8. Meta-metrics: METR time horizon and GDPval

### 8.1 METR's 50% time horizon

**Problem.** Benchmark accuracy has no natural units. METR's question is: what is the length of task, measured in *skilled-human time*, that an AI agent completes with 50% reliability? [57]

**Architecture.**

```
  task suites with human baselines (HCAST, RE-Bench, SWAA, ...)
     each task: human_minutes (geo-mean of successful human baselines)
          |
  agent runs --> score_binarized in {0,1}, weighted (diversity-adjusted)
          |
  fit logistic:  P(success) = sigma( a - b * log2(human_minutes) )
          |
  horizon_50 = human_minutes where P = 0.5   (also p80)
  bootstrap over tasks/runs --> CI;  regress log(horizon) on release date
```

The public analysis repository stores runs with `task_id`, `task_family`, `human_minutes`, `score_binarized`, `score_cont` and an `invsqrt_task_weight` diversity weight. It produces p50 and p80 logistic fits and bootstrap confidence intervals [58]. The 2025 headline finding was that frontier time horizon has doubled about every seven months since 2019, possibly faster from 2024 [57].

**Time Horizon 1.1 (January 29, 2026).** The suite grew from 170 to 228 tasks: 73 HCAST tasks added, 15 removed and 53 updated. Tasks estimated at 8 or more human hours rose from 14 to 31. Infrastructure moved from METR's Vivaria to the UK AISI's open-source Inspect [59]. Fourteen models were re-estimated, and most new estimates fell within the old confidence intervals [59]. METR has since said the suite is nearing saturation and that measurements above 16 hours are unreliable [59]. Fitted doubling times depend on the window chosen; trackers report roughly 4-6 months for post-2023 data (approximate) [59][60].

**Failure modes.** The long-task tail is thin (31 tasks of 8+ hours). The logistic form assumes difficulty is monotone in human time. Agent-friendly tasks with crisp, checkable specifications can yield much longer apparent horizons, a point METR itself made when comparing its estimates with Epoch's [59].

### 8.2 GDPval: expert pairwise grading of real work products

**Problem.** OpenAI's GDPval (October 2025) measures model performance on economically valuable deliverables rather than exam questions [61].

**Construction.** It covers 44 occupations in the top nine sectors contributing to US GDP. The full set has 1,320 tasks, at least 30 per occupation. A 220-task "gold" subset, 5 per occupation, is open-sourced [61]. Each task is based on actual work product from an industry professional, and authors average 14 years of experience [61]. Tasks come with reference files, and deliverables are documents, slides, spreadsheets and similar artifacts.

**Grading.**

```
  request + reference files --> model deliverable  \
                            --> expert deliverable  >--> blinded occupational expert
                                                    /    ranks unlabeled outputs
                                                         (>1 hour per comparison)
  metric: win-or-tie rate vs. human expert
  automated grader (experimental, gold subset): 66% agreement with experts
  vs. 71% human inter-rater agreement
```

The automated grader is "faster and cheaper" but sits about 5 points below expert-to-expert agreement [62]. Commentators have argued that this gap matters more than headline win rates, since anyone who uses the automated grader inherits a disagreement rate of about 34% (commentary) [63].

**Lesson.** When the output is an open-ended work artifact, blinded expert pairwise preference is the only grading method with face validity, at a cost of hours per item. An LLM grader with known agreement statistics is an acceptable development proxy, not a replacement.

## 9. Comparison table

| Benchmark | Year (latest ver.) | Items | Answer/output type | Grader | Contamination control | Main failure mode reported |
|---|---|---|---|---|---|---|
| MMLU | 2020 | 57 subjects | 4-way MC | letter match | none | ~6.5% label errors [3] |
| MMLU-Pro | 2024 | 12k+ | 10-way MC | letter match | harder distractors | LLM-generated distractors [1] |
| GPQA Diamond | 2023 | 198 | 4-way MC | letter match | canary, zip password | saturation, small n [5][6] |
| HLE | 2025 | 2,500 + private | MC / exact short | LLM judge + calibration | private set, canary | wrong keys in Bio/Chem [9] |
| ARC-AGI-1 | 2019 | 400+400 public | grid | exact match | semi-private/private | brute-force compute [12] |
| ARC-AGI-2 | 2025 | ~1.4k tested | grid | exact, pass@2, $/task | semi-private/private | baseline definition [15] |
| ARC-AGI-3 | 2026 | 100s of games | actions | RHAE vs humans | private envs | scoring-rule changes [18] |
| MATH-500 | 2021/2024 | 500 | boxed answer | equivalence check | none | saturation [21] |
| AIME (MathArena) | 2025-26 | 30/yr | integer | exact match | eval at release | n=30 granularity [23] |
| FrontierMath | 2024 (v2 2026) | 300 + 50 T4 | closed form | scripted check | private + holdout | reference errors [27] |
| HumanEval | 2021 | function tasks | code | unit tests, pass@k | none | weak tests, leakage [30] |
| LiveCodeBench | 2024-25 | 1,055 (v6) | code | unit tests | release-date windows | window comparability [31] |
| SWE-bench Verified | 2024 | 500 | patch | F2P + P2P tests | none (public repos) | flawed tests, memorised [36] |
| SWE-Bench Pro | 2025 (V2 2026) | 1,865 / 642 V2 | patch | tests in Harbor | copyleft, held-out, commercial | invalid tasks [42] |
| GAIA | 2023 | 466 | short answer | normalised match | 300 answers withheld | tool/harness variance [45] |
| tau-bench (tau2/3) | 2024-26 | domains | dialogue + tools | DB-state hash x comms | simulated user | gold-path errors [35] |
| BrowseComp | 2025 | 1,266 | short answer | LLM judge | encrypted rows | grader code fragility [48] |
| Terminal-Bench 2.0 | 2025 (2.1 2026) | 89 | shell actions | test scripts | new tasks | scaffold dependence [50] |
| OSWorld(-Verified) | 2024/2025 | 369 | GUI actions | per-task eval scripts | VM snapshots | env flakiness [56] |
| METR time horizon | 2025 (TH1.1 2026) | 228 tasks | agent runs | logistic fit on human-time | private tasks | thin long tail [59] |
| GDPval | 2025 | 1,320 / 220 gold | work products | blinded expert pairwise | private 1,100 | cost; 66% auto-agreement [62] |

## 10. Cross-cutting failure modes

1. **The ground truth decays faster than the models improve.** Every major benchmark examined here had a published error rate large enough to matter: about 6.5% for MMLU [3], 18-29% for HLE Bio/Chem [9], 59.4% of audited hard SWE-bench Verified tasks [36], 89/731 invalid tasks in SWE-Bench Pro public [42], 75+ fixes in tau-bench [35], 28/89 repaired tasks in Terminal-Bench [50], and a reported multi-tens-of-percent error rate in FrontierMath [27]. Adversarial difficulty filters make this worse, because they keep the items that frontier models fail, and a mislabeled item is one that frontier models will "fail".
2. **The grader is software and has bugs.** Examples include the answer-extraction regex in GPQA [21], the HLE calibration loop that skips the last bin [8], the BrowseComp `group(0)` comparison [48], and tau2-bench's banking grading fix that invalidated earlier scores [35]. Reference graders need their own test suites.
3. **Infrastructure noise shows up as capability.** Low-memory malloc failures in HumanEval [30], timeout sensitivity in LiveCodeBench [32], 120 GB images in SWE-bench [33], and VM flakiness that pushed OSWorld to run verified submissions in-house [56] all affect scores.
4. **Agentic scores rank (model, scaffold, budget) tuples.** Terminal-Bench, OSWorld and SWE-bench numbers move substantially with the scaffold. ARC-AGI-2 and -3 deal with this by putting cost or action efficiency into the headline number [13][17].
5. **Contamination controls fall into five types.** These are canaries and encryption (GPQA, HLE, BrowseComp); time windows (LiveCodeBench, MathArena); private or held-out splits (ARC, FrontierMath, SWE-Bench Pro, HLE, GDPval); licence-based selection (SWE-Bench Pro's copyleft repositories); and measured holdouts (FrontierMath Tier 4's 28/20 split). Only the last two give you a *measurement* of contamination rather than a hope.
6. **Versioning silently breaks comparability.** Examples are LiveCodeBench releases, SWE-Bench Pro V1 versus V2 under the same Hugging Face load call, OSWorld-Verified, tau2 v1.0.1, Terminal-Bench 2.0 versus 2.1, METR TH1.0 versus TH1.1, and the reported ARC-AGI-3 baseline change. A score without a dataset version, harness version and scoring-rule version cannot be compared with anything.

## Key takeaways for system designers

- **Design the grader before the items.** Choose the cheapest grading primitive that keeps validity: exact match, then scripted checks, then execution tests, then state hashing, then LLM judge, then expert pairwise. Each step up multiplies cost and variance.
- **Grade outcomes, not paths.** tau-bench's replay-and-hash-the-final-state design and OSWorld's per-task state evaluators accept any valid trajectory. ACTION-matching should be reserved for policies that truly have one correct path [35][55].
- **Budget for verification in proportion to adversarial filtering.** If items survive only because models fail them, independently re-derive the answer key (FutureHouse-style literature checks, instruction-only expert re-solves as in SWE-Bench Pro V2, frontier-model-assisted audits as in FrontierMath v2) [9][42][27].
- **Ship an oracle agent and a reference scaffold.** Terminal-Bench's `--agent oracle` and Harbor's task directories (verifier + reference solution + image) make task health testable in CI [52][41].
- **Measure reliability, not just capability.** Report pass^k next to pass@k for agents that will run unattended [34]. For selection methods, note that confidence-weighted aggregation can add 15-25% on verify-easy tasks [47].
- **Put cost or efficiency in the metric.** ARC-AGI-2's dollars per task and ARC-AGI-3's squared action-efficiency ratio with a 5x cutoff stop brute force from looking like intelligence [13][17].
- **Make contamination measurable.** Keep a held-out split whose accuracy you can compare with the public split (FrontierMath Tier 4, SWE-Bench Pro held-out, HLE private). Time-window everything that is scraped [28][39][31].
- **Version everything and publish errata.** Pin dataset, harness, grader and scoring-rule versions in every reported number, and re-grade old submissions when graders change, as tau2-bench did [35].
- **Mind statistical resolution.** Thirty AIME items or 198 GPQA Diamond items cannot separate models a few points apart. Report confidence intervals (HLE's harness reports Wald intervals [8]) and prefer larger or pooled sets for ranking.
- **Plan for retirement.** SWE-bench Verified went from standard to unreported in about 18 months [25][36]. Treat any public benchmark as having a half-life, and have its successor (held-out, versioned, licence-filtered) ready before saturation.

## References

1. Wang, Y. et al. "MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark." TIGER-Lab / arXiv, 2024. https://arxiv.org/html/2406.01574v2
2. TIGER-AI-Lab. "MMLU-Pro" repository README. GitHub, 2024. https://github.com/TIGER-AI-Lab/MMLU-Pro
3. Gema, A. P. et al. "Are We Done with MMLU?" arXiv / NAACL 2025, 2024. https://arxiv.org/abs/2406.04127v1
4. Rein, D. et al. "GPQA: A Graduate-Level Google-Proof Q&A Benchmark." arXiv, 2023. https://arxiv.org/pdf/2311.12022
5. Rein, D. "gpqa" repository README (canary, password-protected data, baselines). GitHub, 2023. https://github.com/idavidrein/gpqa
6. Vals.ai. "GPQA benchmark (archived snapshot)." Vals.ai, 2025. https://www.vals.ai/benchmarks/gpqa-09-08-2025
7. Phan, L. et al. (CAIS and Scale AI). "Humanity's Last Exam." arXiv, 2025. https://arxiv.org/pdf/2501.14249
8. Center for AI Safety. "hle" evaluation repository (README and hle_eval/run_judge_results.py). GitHub, 2025. https://github.com/centerforaisafety/hle
9. FutureHouse. "HLE chemistry/biology answer audit." FutureHouse, July 2025. https://futurehouse.org/research/hle-exam
10. Scale AI. "Humanity's Last Exam leaderboard." Scale Labs, 2025-2026. https://labs.scale.com/leaderboard/humanitys_last_exam
11. Chollet, F. "ARC-AGI-1" repository README and "On the Measure of Intelligence." GitHub / arXiv, 2019. https://github.com/fchollet/ARC-AGI and https://arxiv.org/abs/1911.01547
12. ARC Prize Foundation. "ARC Prize 2024 results." ARC Prize, 2024. https://arcprize.org/2024-results
13. ARC Prize Foundation. "ARC-AGI-2 Technical Report." ARC Prize, 2025. https://arcprize.org/blog/arc-agi-2-technical-report
14. Epoch AI. "ARC-AGI-2 benchmark page." Epoch AI, 2025. https://epoch.ai/benchmarks/arc-agi-2/
15. LessWrong. "ARC-AGI-2 human baseline surpassed (updated)." LessWrong, 2025-2026. https://www.lesswrong.com/posts/DX3EmhmwZjTYp9PBf/arc-agi-2-human-baseline-surpassed-updated
16. ARC Prize Foundation. "ARC-AGI-3 launch." ARC Prize, March 25, 2026. https://arcprize.org/blog/arc-agi-3-launch
17. ARC Prize Foundation. "ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence" (technical report). arXiv, 2026. https://arxiv.org/html/2603.24621v1
18. ARC Prize Foundation. "ARC-AGI-3 Scoring Methodology." ARC-AGI-3 Docs, 2026. https://docs.arcprize.org/methodology ; reported April 2026 baseline change via "Workspace Optimization: How to Train Your Agent," arXiv, 2026. https://arxiv.org/pdf/2605.09650
19. Chollet, F. Post on the ARC-AGI-3 human action-efficiency baseline. X, March 2026. https://x.com/fchollet/status/2036868401843626302
20. ARC Prize Foundation. "ARC-AGI-3-Agents" repository README and changelog. GitHub, 2025-2026. https://github.com/arcprize/ARC-AGI-3-Agents
21. OpenAI. "simple-evals" repository README. GitHub, 2024-2025. https://github.com/openai/simple-evals
22. Hendrycks, D. et al. "Measuring Mathematical Problem Solving With the MATH Dataset." NeurIPS, 2021. https://github.com/hendrycks/math and https://arxiv.org/pdf/2103.03874
23. Balunovic, M. et al. "MathArena: Evaluating LLMs on Uncontaminated Math Competitions." arXiv, 2025. https://arxiv.org/pdf/2505.23281
24. MathArena. "aime_2026" dataset card. Hugging Face, 2026. https://huggingface.co/datasets/MathArena/aime_2026/blob/main/README.md ; leaderboard figures via BenchLM, https://benchlm.ai/md/blog/posts/aime-hmmt-competition-math.md
25. OpenAI. "Introducing SWE-bench Verified." OpenAI, August 2024. https://openai.com/index/introducing-swe-bench-verified/
26. Epoch AI. "FrontierMath Tiers 1-4: About." Epoch AI, 2024-2026. https://epoch.ai/frontiermath/tiers-1-4/about ; Glazer, E. et al. "FrontierMath: A Benchmark for Evaluating Advanced Mathematical Reasoning in AI." arXiv, 2024. https://arxiv.org/pdf/2411.04872
27. Secondary reporting on FrontierMath v2 errata (June 2026). Systems Analysis wiki, 2026. https://systems-analysis.ru/eng/FrontierMath ; Harris, M. "The FrontierMath scandal." Silicon Reckoner, 2025. https://siliconreckoner.substack.com/p/the-frontier-math-scandal
28. Epoch AI. "New record on FrontierMath Tier 4." Epoch AI Substack, 2025-2026. https://epochai.substack.com/p/new-record-on-frontiermath-tier-4
29. The Decoder. "OpenAI quietly funded independent math benchmark before setting record with o3." The Decoder, January 2025. https://the-decoder.com/openai-quietly-funded-independent-math-benchmark-before-setting-record-with-o3/
30. OpenAI. "human-eval" repository README. GitHub, 2021. https://github.com/openai/human-eval
31. Jain, N. et al. "LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code." arXiv / ICLR 2025, 2024. https://arxiv.org/html/2403.07974v2
32. LiveCodeBench. Repository README (release versions, scenarios, date filtering). GitHub, 2024-2025. https://github.com/LiveCodeBench/LiveCodeBench
33. SWE-bench team. "SWE-bench" repository README (Docker evaluation harness). GitHub, 2023-2026. https://github.com/SWE-bench/SWE-bench
34. Yao, S., Shinn, N., Razavi, P., Narasimhan, K. "tau-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains." Sierra / arXiv, 2024. https://export.arxiv.org/pdf/2406.12045
35. Sierra Research. "tau2-bench" repository README and docs/evaluation.md (tau3 release notes, reward_basis). GitHub, 2025-2026. https://github.com/sierra-research/tau2-bench
36. OpenAI. "Why SWE-bench Verified no longer measures frontier coding capabilities." OpenAI, February 2026. https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/ ; secondary summary: https://app.daily.dev/posts/why-swe-bench-verified-no-longer-measures-frontier-coding-capabilities-yitnqurmf ; Anthropic screening practice as cited in "Engineering Reliable Coding Agents," arXiv, 2026. https://arxiv.org/pdf/2608.13867
37. OpenAI Developers. Post recommending SWE-bench Pro over Verified. X, February 2026. https://x.com/OpenAIDevs/status/2026002219909427270
38. CodeSOTA. "Is SWE-bench Verified Contaminated? OpenAI Shifts to SWE-bench Pro." CodeSOTA, 2026. https://www.codesota.com/news/swe-bench-contamination-debate
39. Deng, X. et al. (Scale AI). "SWE-Bench Pro: Can AI Agents Solve Long-Horizon Software Engineering Tasks?" arXiv, 2025. https://arxiv.org/pdf/2509.16941
40. Scale AI. "SWE-Bench Pro" blog post. Scale AI, 2025. https://scale.com/blog/swe-bench-pro
41. Scale AI. "SWE-bench_Pro-os" repository README (V2, HARD-51, Harbor tasks). GitHub, 2025-2026. https://github.com/scaleapi/SWE-bench_Pro-os
42. Scale AI. "SWE-Bench Pro V2: A Cleaner, Harder-to-Game Leaderboard." Scale Labs, 2026. https://labs.scale.com/blog/swe-bench-pro-v2
43. Hysen Labs. "SWE-bench Pro: 642 tasks under a v1 pipeline." Hysen Labs, 2026. https://hysenlabs.com/projects/scaleapi-swe-bench-pro-os
44. Kili Technology. "AI Benchmark Audit: What It Did to the Coding Leaderboard." Kili, 2026. https://kili-technology.com/blog/data-story-ai-benchmark-audit-swe-bench-pro-verified
45. Mialon, G. et al. (Meta, Hugging Face, AutoGPT). "GAIA: a benchmark for General AI Assistants." arXiv / ICLR 2024, 2023. https://arxiv.org/pdf/2311.12983
46. InfoQ and Gigazine. Launch coverage of OpenAI's BrowseComp (scores, human baseline, dataset revision). 2025. https://www.infoq.com/news/2025/05/openai-browsecomp-ai-benchmark and https://www.gigazine.net/gsc_news/en/20250411-openai-browsecomp-benchmark-browsing-agent
47. Wei, J. et al. (OpenAI). "BrowseComp: A Simple Yet Challenging Benchmark for Browsing Agents." arXiv, 2025. https://arxiv.org/pdf/2504.12516
48. OpenAI. "browsecomp_eval.py" reference implementation. GitHub (simple-evals), 2025. https://github.com/openai/simple-evals/blob/main/browsecomp_eval.py
49. Laude Institute. "terminal-bench" repository README. GitHub, 2025. https://github.com/laude-institute/terminal-bench
50. Merrill, M. et al. "Terminal-Bench 2.0" (arXiv 2601.11868), paper summary page. hyper.ai, 2026. https://hyper.ai/en/papers/2601.11868
51. Morph. "Terminal-Bench 2.0 Leaderboard (2026)." Morph, 2026. https://www.morphllm.com/terminal-bench-2
52. Laude Institute. "terminal-bench-2" repository README and FAQ. GitHub, 2025. https://github.com/laude-institute/terminal-bench-2
53. Harbor Framework Team. "Harbor" repository README. GitHub, 2026. https://github.com/laude-institute/harbor
54. Xie, T. et al. "OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments." arXiv / NeurIPS 2024, 2024. https://arxiv.org/abs/2404.07972v2
55. XLANG Lab. "OSWorld evaluation_examples README" (task schema). GitHub, 2024. https://github.com/xlang-ai/OSWorld/blob/main/evaluation_examples/README.md
56. XLANG Lab. "OSWorld" repository README (OSWorld-Verified, VM providers, leaderboard policy). GitHub, 2025. https://github.com/xlang-ai/OSWorld
57. Kwa, T., West, B. et al. (METR). "Measuring AI Ability to Complete Long Tasks." arXiv, 2025. https://arxiv.org/abs/2503.14499
58. METR. "eval-analysis-public" repository README. GitHub, 2025-2026. https://github.com/METR/eval-analysis-public
59. METR. "Time Horizon 1.1." METR, January 29, 2026. https://metr.org/blog/2026-1-29-time-horizon-1-1/ ; "Task-Completion Time Horizons of Frontier AI Models," https://metr.org/time-horizons/ ; METR on X, https://x.com/METR_Evals/status/2042666178540429698
60. AI 2027 Tracker. "METR time horizon doubles every 4 months." 2026. https://ai2027-tracker.com/predictions/metr-doubling/
61. Patwardhan, T. et al. (OpenAI). "GDPval: Evaluating AI Model Performance on Real-World Economically Valuable Tasks." arXiv, 2025. https://arxiv.org/html/2510.04374v1
62. OpenAI. "GDPval" paper PDF (grading protocol, automated grader agreement). OpenAI, 2025. https://cdn.openai.com/pdf/d5eb7428-c4e9-4a33-bd86-86dd4bcf12ce/GDPval.pdf
63. Dasika, P. "OpenAI's GDPval: Why the 66% Automated Grading Problem Matters More Than the 48% Win Rate." Medium, 2025. https://medium.com/@pranil.dasika/openais-gdpval-why-the-66-automated-grading-problem-matters-more-than-the-48-win-rate-a5e542508196
