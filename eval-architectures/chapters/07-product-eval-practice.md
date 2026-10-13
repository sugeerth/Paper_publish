# 07. Product Eval Practice: Eval-Driven Development, Error Analysis, Golden Datasets, Evals in CI and Validating the Validators

## Overview

Chapters 02 to 06 cover benchmarks, harnesses, judges, arenas and agent evaluation as research objects. This chapter covers the practice that grew up around them between 2023 and 2026: how product teams decide what to measure for one application, how they turn real failures into tests, how those tests run on every change, and how they check that their automated graders can be trusted.

Most of this body of practice comes from a small number of practitioners and vendors:

- **Hamel Husain and Shreya Shankar.** Husain's consulting write-ups (Rechat, Honeycomb, NurtureBoss) [1][2][3], their joint FAQ and course material [4], and an open-source set of agent "skills" that encodes their method [5]. Shankar's research at UC Berkeley produced SPADE [7] and EvalGen [6]. EvalGen's title, "Who Validates the Validators?", named the central problem of the field.
- **Eugene Yan.** Task-specific metric guidance [9], a survey of LLM-evaluator effectiveness [10], and an essay framing evals as the scientific method [11]. He is also co-author, with Husain, Shankar and others, of the widely cited "What We've Learned From A Year of Building with LLMs" [8].
- **Model providers.** Anthropic's success-criteria and eval-design documentation [13], its January 2026 engineering guide to agent evals [12], and its statistics guidance [14]. OpenAI's cookbook covers the legacy open-source Evals framework [16], the hosted Evals API [17], an "evaluation flywheel" built directly on Husain and Shankar's open and axial coding [15], and, in 2026, a migration guide from OpenAI Evals to Promptfoo [18].
- **Tooling that turns this into CI.** Promptfoo [19][20], DeepEval [21] and Braintrust's GitHub Action [22] are representative examples.

By 2025 these sources broadly agreed on one position. Generic, off-the-shelf metrics and dashboards are not enough. The useful eval suite for a product comes bottom-up from reading traces. Domain experts label them, the labels are clustered into an application-specific failure taxonomy, and that taxonomy is turned into cheap deterministic assertions where possible and validated LLM judges where necessary. The suite runs at several cadences, from every commit to every major release. The sources still disagree on some questions, such as volume versus depth of grading and Likert versus binary scales. The comparison sections below set those disagreements out.

A note on sourcing: some practitioner sites (hamel.dev, eugeneyan.com, applied-llms.org, arxiv.org, promptfoo.dev) could not be reached from the research environment. Their content was read from verbatim copies of the same posts and papers in public GitHub repositories, including the authors' own repositories where these exist. The references below give the canonical URLs. Numbers that come from illustrative examples rather than measured deployments are labelled as such.

## 1. The product-eval stack at a glance

Practitioner sources describe the same layered system even when they name the layers differently. Husain calls them Level 1/2/3 [1]. Anthropic describes a "Swiss cheese" of complementary methods [12]. Yan describes a scientific-method loop [11]. The composite architecture is:

```
                 +---------------------------------------------------------+
  production --->| trace logging (inputs, tool calls, retrieved docs,     |
  traffic        | intermediate LLM calls, outputs, user feedback)        |
                 +-------------------------+-------------------------------+
                                           | sample (random / outlier /
                                           | failure-driven / stratified)
                                           v
  synthetic  --->+---------------------------------------------------------+
  inputs         | ERROR ANALYSIS: domain expert reads traces             |
  (dimension     |   open coding  -> free-text note on FIRST failure      |
  tuples)        |   axial coding -> 5-10 failure categories, counts      |
                 +-------------------------+-------------------------------+
                                           | rank by frequency x impact
                       +-------------------+-------------------+
                       v                                       v
          +--------------------------+           +--------------------------+
          | direct fix (prompt,      |           | build evaluator only if  |
          | tool, retrieval, code)   |           | failure persists / recurs|
          +------------+-------------+           +------------+-------------+
                       |                     +----------------+---------------+
                       |                     v                                v
                       |       +--------------------------+  +----------------------+
                       |       | code assertion (regex,   |  | LLM judge: binary,   |
                       |       | schema, exec, counts)    |  | one failure mode,    |
                       |       +------------+-------------+  | TPR/TNR-validated    |
                       |                    |                +----------+-----------+
                       v                    v                           v
          +---------------------------------------------------------------------+
          | GOLDEN / REGRESSION DATASET (curated, 100+ cases, versioned)        |
          +----------+-----------------------+-----------------------+----------+
                     | every commit / PR     | scheduled             | release
                     v                       v                       v
              CI gate (cheap,         offline judge runs,       A/B test on
              deterministic,          human spot-checks,        user outcomes
              threshold or diff)      bias-corrected rates
                     |                       |                       |
                     +-----------+-----------+-----------+-----------+
                                 v                       v
                       dashboards over time     new failures feed back
                       (per prompt/model ver.)  into error analysis
```

The rest of the chapter takes each box in turn.

## 2. Eval-driven development (EDD)

### 2.1 Problem

Husain's March 2024 essay "Your AI Product Needs Evals" starts from a diagnosis: unsuccessful LLM products almost always lack a robust evaluation system [1]. Iteration speed depends on three activities: evaluating quality, debugging, and changing the system (prompting, fine-tuning, code). Teams that invest only in the third stall at demo quality. His case study is Lucy, the assistant in Rechat's real-estate SaaS. Prompt engineering made fast early progress and then plateaued, with three symptoms [1]:

1. Fixing one failure caused another ("whack-a-mole").
2. The only view of quality across tasks was vibe checks.
3. Prompts grew long and unwieldy as they tried to cover edge cases.

### 2.2 The three levels (Husain)

| Level | What | Cost | Cadence |
|---|---|---|---|
| 1. Unit tests | Assertions on LLM output, scoped by feature and scenario | Lowest | Every code change |
| 2. Human and model eval | Trace review plus LLM-as-judge | Middle | Set schedule |
| 3. A/B testing | Does the product change user behaviour? | Highest | After significant product changes |

Cost decides cadence. Level 1 should be largely built out before investing in model-based tests, and A/B testing can wait until the product has real users [1].

### 2.3 EDD as the scientific method (Yan)

Yan's April 2025 essay "An LLM-as-Judge Won't Save The Product, Fixing Your Process Will" sets out the loop explicitly [11]:

```
  observe --> annotate --> hypothesize --> experiment --> measure --> iterate
  (look at    (label pass/   (e.g., bad      (prompt /      (vs. a       (ship, or
   the data)   fail, ~50:50)  retrieval,      retrieval /    baseline or   back to
                              conflicting     model change)  control)      error
                              instructions)                                analysis)
```

He defines eval-driven development by analogy with test-driven development: "First, write some evals; then, build systems that pass those evals." Start from a baseline such as a simple prompt, then evaluate every prompt tweak and retrieval change against it [11]. The essay's title makes the main claim: automated evaluators amplify an existing annotation and feedback process and do not replace one. Yan recommends roughly a 50:50 pass/fail balance in annotated sets that spans the input distribution. Binary judges should be calibrated with precision and recall, and pairwise judges by correlation with human preferences [11].

Anthropic's 2026 agent-eval guide reaches the same conclusion from the platform side. It recommends starting early with 20 to 50 simple tasks drawn from real failures, converting manual pre-release checks and user-reported bugs into tasks, and practising eval-driven development as a team discipline. Dedicated eval teams own the infrastructure, and domain experts and product teams contribute tasks [12].

### 2.4 Team and process implications

The "Year of Building with LLMs" authors argue that evals are a strategic investment rather than an operational cost [8]. They recommend writing eval criteria into product specs and budgeting time for building evals and running multiple experiments before launch. They note that evaluation skill is closer to classic ML engineering than to application engineering, and they recommend keeping a domain expert involved throughout. Husain's 2025 "Field Guide" adds a planning rule: count experiments, not features. A fixed feature roadmap assumes feasibility is known, which it rarely is for LLM features. He cites Bryan Bischof's "capability funnel" and Yan's time-boxed feasibility stages: about 2 weeks for data feasibility, 1 month for technical feasibility, and 6 weeks to a prototype A/B test [3].

## 3. Error analysis: open and axial coding

### 3.1 Problem

Teams that start from generic metrics such as "helpfulness", "coherence" or "hallucination score" get what Husain calls a false sense of progress. The numbers move, but they do not point at a fix [3]. The alternative borrows from qualitative social science. The terms *open coding* and *axial coding* come from grounded theory, and Husain and Shankar adapted them to LLM traces [4][5].

### 3.2 Procedure

The FAQ [4] and the `error-analysis` skill [5] give a concrete protocol:

1. **Collect about 100 representative traces.** A trace is the full record: input, intermediate LLM calls, tool calls, retrieved documents, reasoning and final output. At low volume, sample randomly. At high volume, stratify by query type, user segment or feature area. k-means clustering on embeddings is suggested as a way to get diversity. If there is no production data yet, generate synthetic inputs and run them through the real pipeline [5].
2. **Open coding.** A domain expert reads each trace, marks pass or fail, and writes a free-text note on the **first** thing that went wrong, because later errors often cascade from it. Notes record observations, not guesses about causes, and the reviewer does not start from a predefined failure list [4][5].
3. **Axial coding.** After about 30 to 50 traces, not after all 100, the notes are grouped into a taxonomy. The target is 5 to 10 categories, each with a specific name and a one-sentence definition. Notes that look alike but have different root causes are split. "Fabricated property feature" and "fabricated user intent" are different failures [5]. An LLM can propose clusters, but a human must review them, because LLMs tend to group by surface wording [5]. The FAQ calls axial coding "the most important step" [4].
4. **Label every trace against every category** (binary), then compute each category's failure rate and sort descending [5].
5. **Iterate 2 to 3 rounds**, merging, splitting and re-labelling, until **theoretical saturation**. The rough stopping rule is about 100 traces with no new failure type in the last 20 [4][5].

The FAQ reports that teams it works with spend 60 to 80% of development time on error analysis and evaluation. Its minimum viable habit is about 30 minutes reviewing 20 to 50 outputs after every significant change [4].

### 3.3 Sampling strategies

| Strategy | When | Method |
|---|---|---|
| Random | Default start | Uniform over recent traces |
| Outlier | Find unusual behaviour | Sort by length, latency, tool-call count; read extremes |
| Failure-driven | After guardrail hits or complaints | Prioritise flagged traces |
| Uncertainty | Once judges exist | Read judge disagreements and low-confidence cases |
| Stratified | Coverage across segments | Sample within each dimension |

Source: `error-analysis` skill [5].

### 3.4 From taxonomy to action

The skill's decision rule matters as much as the taxonomy: **fix before you measure** [5]. Many failure categories disappear with a direct fix, such as a missing prompt instruction, a tool the model was never given, or a retrieval or parsing bug. An evaluator is worth building only for failures that persist after fixing, weighed by frequency, business impact and whether the evaluator will actually be reused. Even then, code-based checks (regex, parsing, schema validation) come before a judge prompt. Safety or compliance requirements can justify a guardrail even after a fix [5]. Husain makes the same point about cost: assertions are cheap, but LLM judges carry real overhead in labelling, weekly maintenance and coordination, so build expensive evaluators only for failures you will iterate on repeatedly [4].

### 3.5 Case study: NurtureBoss

The Field Guide reports one deployment in numbers [3]. NurtureBoss, an AI assistant for the apartment-leasing industry, found through conversation review that it failed about 66% of the time on date handling, such as rescheduling tours. After targeted tests and fixes, success on that category rose from 33% to 95%. Three issues (conversation flow, hand-off failures, rescheduling) accounted for over 60% of all problems. The case shows the main reason to do axial coding: a frequency table over observed categories identifies the few fixes that remove most failures.

### 3.6 Agents: the transition failure matrix

For multi-step agents, the FAQ separates two phases. Phase 1 measures end-to-end task success with the agent treated as a black box. Phase 2 runs step-level diagnostics: tool choice, parameter extraction, error handling, context retention, efficiency and goal checkpoints. It adds a **transition failure matrix**: rows are the last successful state, columns are the first failed state, and each cell counts traces [4]. In the FAQ's worked example, GenSQL to ExecSQL accounts for 12 failures, while DecideTool to PlanCal accounts for 2. This applies the "first failure" rule of open coding to a state machine. Chapter 06 covers agent evaluation in more depth.

```
               first failing state ->
               DecideTool  PlanCal  GenSQL  ExecSQL  Respond
 last   Start      .          .       .        .        .
 good   Decide     -          2       .        .        .
 state  GenSQL     .          .       -       12        .
   |    ExecSQL    .          .       .        -        .
   v      (illustrative cells only; counts from the FAQ example [4])
```

## 4. Unit-test-style assertions

### 4.1 Design

Level 1 tests are written like pytest assertions and scoped by **feature and scenario** [1]. In Rechat's "Listing Finder" feature, the LLM turns a natural-language search into a CRM query. There are three scenarios, each with its own assertion: exactly one match (result length == 1), multiple matches (> 1) and no match (== 0). Generic assertions run across all features. One example is a regex that checks no UUID (8-4-4-4-12 hex) appears in user-facing text, because CRM results contain IDs that users should never see [1]. Rechat maintained "hundreds" of such tests, updated as new failures surfaced [1].

The same assertions do more than gate releases. They can trigger automatic retries or self-correction, filter synthetic data, and later curate fine-tuning data. Husain estimates that 99% of fine-tuning effort is data assembly, and a mature eval system already is a data generation and curation engine [1].

### 4.2 What to assert

The "Year of Building" authors recommend building unit tests from real production samples, with expected outputs judged on at least three criteria. Fewer criteria may mean the task is under-specified [8]. They start with simple checks: phrases that must or must not appear, and word, item or sentence counts within a range. For code generation they recommend execution-based checks, such as "if `foo` was requested, is `foo` callable afterwards?". They also advise relaxing assertions to the weakest condition any viable answer would still satisfy [8].

SPADE (Shankar et al., 2024) automates the first step [7]. It mines **prompt version histories**, because developers add instructions to prompts to fix observed failures. Each diff is a "prompt delta". From 19 LangChain pipelines the authors built a delta taxonomy of about 35% structural and 65% content-based deltas. GPT-4 categorises deltas (F1 approximately 0.81), then generates natural-language criteria and Python assertion functions, some of which call an LLM. Candidate assertions are then filtered to a minimal set that meets a coverage target and a false-failure-rate (FFR) cap. This is an integer linear program, and the authors show the selection problem is NP-hard. A subsumption-based variant handles low-data settings. On nine pipelines (about 75 examples each, coverage target 0.6, FFR cap 0.25), the subsumption variant selected about 14% fewer assertions and had about 21% lower FFR than a keep-everything-that-passes baseline. That baseline violated the FFR cap on 4 of 9 pipelines. ILP runtimes were under about 1.2 s [7]. The paper reports more than 2,000 runs of a public prototype and states that SPADE has been deployed in LangSmith [7].

### 4.3 Task-specific metrics where assertions are not enough

Yan's March 2024 survey of metrics [9] is the reference for when a numeric metric beats a pass/fail assertion:

| Task | Metrics that work | Avoid | Notes |
|---|---|---|---|
| Classification / extraction | Recall, precision, ROC-AUC, PR-AUC, separation of score distributions | Accuracy alone | PR-AUC baseline = positive rate |
| Summarisation: consistency | NLI model, source as premise, summary as hypothesis | ROUGE, BERTScore, G-Eval | Fine-tuning NLI moved ROC-AUC 0.56 -> 0.85 |
| Summarisation: relevance | Reward model on human preferences | -- | Length via simple counts |
| Translation | chrF, BLEURT, COMET, COMETKiwi (ref-free) | BLEU | BLEU ranked poorly at WMT22/23 |
| Copyright regurgitation | Exact (LCS) and near-exact (edit distance) | -- | Use mid-book prompts |
| Toxicity | Share of outputs with Perspective score >= 0.5 | -- | RealToxicityPrompts, BOLD |

Yan gives a baseline rate for factual inconsistency of about 5 to 10%, with below about 2% hard to reach [9][8]. He also reports that Voiceflow's eval harness caught a roughly 10% drop when moving between gpt-3.5-turbo versions [9]. That kind of silent model-upgrade regression is the main reason teams run evals in CI. The Husain and Shankar FAQ is stricter than Yan on similarity metrics. It calls BERTScore, ROUGE and cosine similarity "generally not useful" for judging LLM outputs, though useful for retrieval and for measuring output diversity [4].

## 5. Golden datasets and synthetic data

### 5.1 What a golden set is in practice

Across sources, the "golden" or regression dataset is a curated, versioned set of inputs, with expected properties and sometimes reference outputs, that the system must keep passing:

- **Husain and Shankar:** CI datasets should be small and curated, at least 100 examples, covering core features, past bugs and edge cases, with a preference for deterministic checks because they run often [4].
- **Anthropic agent guide:** tasks need unambiguous success criteria and a reference solution, such that two domain experts would independently reach the same pass/fail verdict [12]. Suites should be balanced, testing both when a behaviour should occur and when it should not. Anthropic's Claude.ai web-search evals cover both over- and under-triggering [12]. A 0% pass rate across many trials usually means the task is broken, not that the agent is incapable [12].
- **Anthropic docs:** cases should mirror the real task distribution and include edge cases: irrelevant or missing input, overly long input, harmful input, and ambiguous cases where humans also disagree [13]. The worked examples give concrete sizes: 1,000 labelled tweets for an exact-match sentiment eval, 50 paraphrase groups for a consistency eval, 200 articles for ROUGE-L, 100 inquiries for an LLM-Likert tone eval, 500 patient queries for a binary PHI-leak eval, and 100 multi-turn conversations for context use [13].

### 5.2 Capability versus regression suites

Anthropic separates two kinds of suite [12]. **Capability (quality) evals** should start at a low pass rate and give the team a hill to climb. **Regression evals** should sit near 100% and catch backsliding. Capability tasks that saturate "graduate" into the regression suite. Husain makes a related point from the product side: a 100% pass rate is not the goal, the acceptable rate is a product decision, and a suite at about 70% may be a more meaningful stress test than a saturated one [1][4]. Anthropic also warns that a saturated eval only tracks regressions and needs harder tasks [12].

### 5.3 Synthetic generation by dimension tuples

Before a product has real traffic, both the Husain/Shankar method and OpenAI's cookbook use structured synthetic data [2][4][15]:

1. Define **dimensions** aimed at suspected failure modes. Husain's B2C default is features x scenarios x personas [2]. The FAQ's recipe-app example uses dietary restriction x cuisine x query complexity [4]. OpenAI's leasing-assistant example uses Channel {Voice, Chat, Text} x Intent {Tour Scheduling, Maintenance, General Info} x Persona {Prospective Resident, Agency} [15].
2. Hand-write about 20 tuples, then scale up with an LLM [4].
3. **Two-step generation.** Generate tuples first, then turn each tuple into natural language in a separate prompt. Asking an LLM for "N examples" directly gives homogeneous cases [4][15].
4. Add **perturbations** such as irrelevant information, mistakes and slang [15].
5. Generate **inputs only**. Run them through the real system and log the outputs, rather than having the LLM write expected outputs. Ground the inputs in real constraints (Rechat used a test listings database) and check that each case actually triggers its intended scenario [2][3].

Rechat's early example generated 50 contact-creation instructions, each paired with a lookup of the same contact. The assertion is that the CRM returns exactly one result [1].

### 5.4 Volume versus depth

The sources disagree here. Anthropic's documentation says "prioritize volume over quality": more automatically graded questions with somewhat lower signal beat fewer hand-graded ones [13]. Husain, Shankar and Yan put depth of reading first. Their dataset sizes are modest (about 100 traces for saturation, about 100 labelled examples per judge) and they spend effort on expert labels [4][5][11]. The two positions apply to different stages. Volume matters once graders are trustworthy and automated. Depth matters while discovering what to grade and validating the graders. Anthropic's own 2026 agent guide, which tells teams to start with 20 to 50 tasks from real failures and to read the transcripts, is closer to the practitioner position for the discovery phase [12].

## 6. Evals in CI

### 6.1 Cadence and gating

```
 developer change (prompt / model / retrieval / code)
        |
        v
 +--------------+    PR opened     +-------------------------------------+
 | local run    |----------------->| CI job                              |
 | (cached LLM  |                  |  1. run golden set through system   |
 |  calls)      |                  |  2. deterministic asserts (all)     |
 +--------------+                  |  3. validated judges (subset/cache) |
                                   |  4. compare vs baseline experiment  |
                                   |  5. gate: --fail-on-error, or       |
                                   |     pass-rate >= threshold, or      |
                                   |     no regression on key scores     |
                                   +---------+----------------+----------+
                                             |                |
                              PR comment w/ score diffs   JUnit/JSON/HTML
                              (+3pp, 8 up / 4 down)       artifacts, dashboards
```

Husain's 2024 essay already recommended running Level 1 tests in GitHub Actions or GitLab pipelines. It also recommended storing results outside CI, next to test and prompt versions, so trends can be charted. Rechat used Metabase dashboards that showed a specific error's prevalence before and after a fix [1]. The FAQ draws the line between CI and production: CI uses curated sets and deterministic checks, while production samples live traces asynchronously, applies reference-free judges, tracks confidence intervals, and investigates when the lower bound crosses a threshold. New production failure patterns are fed back into CI [4].

### 6.2 Tooling patterns

| Tool | Test definition | Grader types | CI gate | Reporting |
|---|---|---|---|---|
| OpenAI Evals (legacy OSS) [16] | Registry YAML + JSONL samples (`input`, `ideal`) | match, includes, fuzzyMatch, model-graded templates | `oaieval` CLI exit / logs | Logs in `/tmp/evallogs` |
| OpenAI Evals API [17] | Eval (config, schema, `testing_criteria`) + Runs | e.g. `label_model` judge | Compare run report scores (baseline vs candidate) | `report_url` in dashboard |
| Promptfoo [19][20] | `promptfooconfig.yaml`: prompts, providers, tests, asserts | Many deterministic (`contains`, `is-json`, `regex`, `latency`, `cost`, trajectory) and model-graded (`llm-rubric`, `g-eval`, `factuality`, RAG metrics) | `--fail-on-error` or custom pass-rate script (e.g. < 95% fails) | JSON, HTML, JUnit XML, share URL |
| DeepEval [21] | pytest functions building `LLMTestCase` | G-Eval, DAG, faithfulness, tool correctness, etc. (0-1 scores) | `assert_test` with per-metric `threshold`; `deepeval test run` | Optional hosted platform |
| Braintrust eval-action [22] | Eval files in Node/Python/Go | User-defined scorers | Optional `terminate_on_failure` | PR comment: averages, improvements, regressions |

A few design details matter in practice:

- **Weighted scoring and thresholds.** In Promptfoo, each assertion has a weight (default 1.0) and the test score is the weighted mean. A test-level `threshold` can override individual failures. Without a threshold, every non-zero-weight assertion must pass. Weight 0 records a named metric without affecting pass/fail, which is useful for collecting data on a new grader before gating on it. `assert-set` groups assertions with their own fractional threshold [20]. Anthropic describes the same three choices for agents: weighted, binary (all graders must pass) and hybrid [12].
- **Caching and cost.** Promptfoo recommends a persistent cache whose key hashes the prompt files and config, with a TTL (for example 24 h). This cuts cost and avoids rate limits without reusing stale results [19]. Braintrust's action can route calls through a caching proxy [22].
- **Baseline diffs over absolute scores.** Braintrust posts a single PR comment showing averages with deltas (for example "83% (+3pp)") and counts of improved and regressed cases [22]. OpenAI's regression notebook does the same thing by grading a `baseline-run` and a `regression-run` under one Eval. In its example, a deliberately worse prompt (longer summaries) scores clearly lower on a `label_model` judge that checks conciseness [17].
- **Portability.** In June 2026 OpenAI's cookbook added a guide to migrating from OpenAI Evals, which it describes as being wound down, to Promptfoo. The guide stresses that a config file kept in the application repo can run locally or in CI, unlike a dashboard-managed eval. It warns that similarity scores may differ across systems and that LLM-judge graders must be re-validated before they are used for regression decisions [18].

### 6.3 Statistical hygiene for gates

Anthropic's November 2024 statistics guidance applies directly to CI gates, even though it was written for model benchmarks [14]:

- Report a standard error with every score. The 95% CI is mean +/- 1.96 SEM.
- **Cluster** standard errors when cases share a source, such as several questions per document or several turns per conversation. On popular evals, clustered SEs were more than 3x the naive ones.
- Reduce within-case variance by resampling each case several times and averaging per case.
- Compare variants with **paired differences** on the same cases. Per-question score correlations of about 0.3 to 0.7 between frontier models make pairing a nearly free variance reduction.
- Use **power analysis** to choose suite size. A 3-point difference cannot be detected reliably with a few dozen cases.

The agent guide adds a non-determinism metric that matters for customer-facing gates [12]. **pass@k** is the probability of at least one success in k trials and rises with k. **pass^k** is the probability that all k trials succeed and falls with k. A 75% per-trial success rate gives a pass^3 of about 42%. By k = 10, pass@k approaches 100% while pass^k approaches 0%. For a support agent, pass^k is the honest number. The guide also warns about harness hygiene. Each trial must start from a clean, isolated state, because shared state such as git history can leak information and inflate scores [12].

## 7. Who validates the validators?

### 7.1 EvalGen and criteria drift

Shankar, Zamfirescu-Pereira, Hartmann, Parameswaran and Arawjo's UIST 2024 paper starts from a simple observation: LLM-based evaluators inherit the unreliability of the LLMs they grade, so a human has to validate them [6]. EvalGen, built into the open-source ChainForge prompt-engineering tool, is a mixed-initiative workflow:

```
  user context + prompt
          |
          v
  [1] LLM proposes criteria (NL) <---- user edits/adds/removes, picks
          |                             code-based or LLM-based per criterion
          v
  [2] LLM generates candidate assertions per criterion (Python fns or
      judge prompts), executed on sampled outputs in background
          |
          v
  [3] user grades outputs thumbs-up / thumbs-down (no fixed quota)
          |     a thumbs-up output must pass all criteria; candidates that
          |     fail good outputs or pass bad ones are re-weighted
          v
  [4] select per-criterion assertion best aligned with grades
          |
          v
  report card: per-criterion confusion matrix, coverage, FFR
```

The paper defines **coverage** as the fraction of human-labelled bad outputs that the assertion set flags. **False failure rate (FFR)** is the fraction of good outputs it wrongly fails, with a default interface threshold of 20%. **Alignment** is the harmonic mean of coverage and (1 - FFR) [6]. In offline comparisons against SPADE with only 16 human grades per pipeline:

| Pipeline | Method | # assertions | Coverage | FFR | Alignment |
|---|---|---|---|---|---|
| Medical (84 outputs, 27 bad) | EvalGen | 3 | 0.33 | 0.10 | 48.3% |
| Medical | SPADE | 5 | 0.33 | 0.10 | 48.3% |
| Product descriptions (100, 49 bad) | EvalGen | 4 | 0.73 | 0.39 | 66.5% |
| Product descriptions | SPADE | 9 | 0.49 | 0.39 | 54.4% |

Source: [6].

The qualitative user study (9 industry practitioners, 45 to 75 min sessions, named-entity recognition on 100 tweets) produced the paper's main contribution, **criteria drift** [6]. Users need criteria to grade outputs, but grading outputs is what helps them define criteria. Participants found implicit rules the prompt never stated and reinterpreted existing criteria while grading. For example, "all proper nouns" was relaxed to "mostly proper nouns", and graders revised earlier grades. Alignment was also subjective. The same code assertion looked aligned to one participant and misaligned to another, usually because of unstated implicit criteria. Participants wanted to read code assertions directly and were sceptical about maintaining LLM-based ones over time [6].

The design implications became standard practice. Criteria refinement and grading should be interleaved rather than sequential. Grading should be per criterion with per-criterion FFR thresholds. Good and bad examples should be embedded as few-shot examples in judges. Users should be prompted to re-grade when the model, prompt or upstream chain changes. Assertions should go through PR-review-like and CI/CD-like workflows [6]. The paper leaves open whether "alignment" is achievable at all, or whether validating the validators is permanent, ongoing work [6]. The "Year of Building" authors summarise the lesson for practitioners: predefined criteria are usually incomplete and must be revisited as outputs are observed [8].

### 7.2 Critique shadowing (Husain)

Husain's October 2024 method for building a business-aligned judge is effectively an operational version of EvalGen [2]:

1. Name a **principal domain expert**, one or at most two people whose judgment defines success, such as a psychologist for a mental-health assistant or the founder at a small company. Avoid proxies.
2. Build a dataset over features x scenarios x personas, from real data, synthetic inputs, or both.
3. The expert gives a **binary pass/fail plus a written critique**, detailed enough for a new hire to follow and reusable as a few-shot example.
4. **Fix pervasive errors first**, then return to step 3.
5. Write the judge prompt with critiques as few-shot examples, and iterate in a spreadsheet (input, output, expert critique, expert verdict, judge critique, judge verdict) until agreement converges.
6. Run error analysis on unseen data using the judge, broken down by dimension.
7. Build specialised judges only for errors that remain.

The reported outcomes: at Honeycomb (Query Assistant, expert Phillip Carter), judge-expert agreement passed 90% within three iterations, and reading the judge's critiques helped the expert notice inconsistencies in his own labels on edge cases [2]. Husain reports roughly 15 to 20% higher human-LLM agreement when judge prompts include example critiques [3]. In the earlier Rechat work, a colleague labelled 25 to 50 examples per round [1]. The "Year of Building" paper reports an unnamed case where agreement rose from 68% to 94% over three iterations of logging responses, critiques and outcomes and reviewing them with stakeholders [8]. Husain reports little success with automatic prompt optimisers such as DSPy for this step and adjusts prompts by hand [2]. Chapter 04 covers optimiser-based alignment.

### 7.3 Statistical validation: TPR, TNR, splits and bias correction

The 2025-2026 Husain/Shankar material replaces "agreement %" with a classifier-style validation protocol, encoded in the `validate-evaluator` skill [5] and adopted almost word for word in OpenAI's flywheel cookbook [15]:

- **Labels.** About 100 traces per failure mode with binary expert labels, balanced near 50 pass and 50 fail even when real prevalence is skewed. Below about 60 examples, confidence intervals become wide [5].
- **Splits.** Train 10-20% (the only source of few-shot examples, clear-cut cases only), dev 40-45% (iterate on the prompt here), test 40-45% (measure once, never iterate on it). Use stratified splits [5]. OpenAI's cookbook uses about 20/40/40 [15].
- **Metrics.** TPR is the share of human-pass cases the judge passes. TNR is the share of human-fail cases the judge fails. The target is both above 90% on dev, with a floor of both above 80% [5]. Raw accuracy is rejected because imbalance hides failure: OpenAI's cookbook notes that an always-pass judge can score 95% while catching nothing [15]. The skill treats Cohen's kappa as a human-vs-human measure only [5].
- **Bias-corrected prevalence.** The judge's raw production pass rate p_obs is corrected with the Rogan-Gladen estimator, theta = (p_obs + TNR - 1) / (TPR + TNR - 1), clipped to [0, 1]. The skill's example: TPR 0.92, TNR 0.88, raw pass 0.80 gives a corrected 0.85. The estimator is unstable as TPR + TNR approaches 1, that is, a judge no better than chance [5].
- **Uncertainty.** Bootstrap the labelled test set (about 2,000 resamples) and recompute TPR, TNR and the corrected rate, giving a 95% CI. Improving TPR narrows the interval more than improving TNR [5].
- **Operational rules.** Pin judge model versions to dated snapshots. Re-validate after prompt or model changes, or when production intervals widen. Never put dev or test cases in the few-shot pool [5].

```
 expert labels (~100, balanced)
     |
     +--> train 10-20% --> few-shot pool --+
     |                                     v
     +--> dev 40-45% ---> run judge --> TPR/TNR --> inspect FP/FN --> edit prompt
     |                                     ^                              |
     |                                     +------------------------------+
     +--> test 40-45% --> run ONCE --> final TPR, TNR (+ bootstrap CI)
                                              |
 production traces --> judge --> p_obs --> Rogan-Gladen --> corrected rate +/- CI
```

The `eval-audit` skill turns this into a checklist and rates "unvalidated LLM judge in production" as the one critical finding [5]. Its other flags include Likert or letter-grade evaluators, holistic "is this helpful?" judges, LLM judges used for checks code could do, similarity metrics as the primary generation metric, kappa or accuracy as the main alignment metric, missing train/dev/test separation, non-expert annotators, reviewers who see only final outputs instead of full traces, and set-and-forget evaluators [5].

### 7.4 What the literature says about judge reliability

Yan's August 2024 review collects the evidence behind this caution [10]. In the MT-Bench study, judges showed position bias: about 50% for gpt-3.5 and about 70% for claude-v1. Those two models also preferred longer, padded answers more than 90% of the time, and GPT-4 and claude-v1 showed self-preference of roughly 10% and 25%. A factual-inconsistency judge caught more than 95% of consistent summaries but only 30 to 60% of inconsistent ones. On HaluEval the best model detected hallucinations at 58.5% accuracy. Agreement metrics can also mislead: UMbrela reported kappa of 0.3 to 0.5 alongside Kendall tau and Spearman rho of 0.8 to 0.9 against TREC labels [10]. Yan recommends direct scoring for objective criteria, pairwise comparison for subjective ones, binary outputs scored with precision and recall where possible, and fine-tuned classifiers or reward models for low-latency guardrails [10]. The "Year of Building" paper adds practical mitigations: swap positions and count both orders, allow ties, ask for reasoning before the verdict, and compare length-matched pairs [8]. Chapter 04 treats judge biases in full.

## 8. Provider guidance compared

### 8.1 Anthropic

Anthropic's developer documentation [13] frames evals around **success criteria** that are specific, measurable, achievable and relevant. They are usually multidimensional. Its sentiment example sets F1 >= 0.85, 99.5% non-toxic outputs and 95% of responses under 200 ms. Even safety is made quantitative, as in "<0.1% of outputs out of 10,000 trials flagged for toxicity". The grader preference order is code-based (fastest, most reliable), then LLM-based ("test to ensure reliability first then scale"), then human (avoid if possible). LLM-grading tips are detailed rubrics, constrained outputs ("correct"/"incorrect" or 1-5), reasoning before the score, and generally a different model as grader [13].

The January 2026 agent guide [12] adds vocabulary that has become common: task, trial, grader, transcript (trace), outcome (final environment state, not what the agent claims), evaluation harness versus agent harness, and suite. It lists three grader families (code, model, human) with their trade-offs, and recommends grading outcomes rather than rigid step sequences, giving partial credit, offering judges an "Unknown" option, grading each dimension separately, and hardening graders against reward hacking. It also gives concrete examples of grader bugs. Opus 4.5 scored 42% on CORE-Bench until grading and specification fixes raised it to 95% [12]. Read the transcripts before trusting a score.

### 8.2 OpenAI

OpenAI's guidance changed in three stages. The 2023 open-source **Evals** framework [16] used a registry of YAML specs and JSONL samples, basic templates (match, includes, fuzzy match) and model-graded templates. The notebook warns that model grading has an error rate and needs human validation. Its own worked example (text-to-SQL, 25 samples, score 0.8) shows the familiar problem that some "failures" were only syntax differences, so either the prompt or the eval needs changing. The hosted **Evals API** [17] separates an Eval (data schema plus testing criteria such as `label_model` judges) from Runs, which makes baseline-versus-candidate comparison the basic operation. The 2025 **evaluation flywheel** cookbook [15] (Analyze -> Measure -> Improve) adopts the Husain/Shankar method directly, citing their course reader for the flywheel graphic. It runs open coding on about 50 failing traces in a Feedback annotation column and axial coding in a Label column, with illustrative shares of 35% scheduling failures versus 10% formatting. It then adds Python and LLM graders, a prompt optimiser that consumes annotations and graders, tuple-based synthetic data, and TPR/TNR judge alignment with 20/40/40 splits. In 2026, the migration guide moved OpenAI's recommended path for regression evals to a repo-resident Promptfoo config [18].

### 8.3 Comparison

| Dimension | Husain / Shankar | Yan | Anthropic | OpenAI (2025-26) |
|---|---|---|---|---|
| Starting point | Error analysis on ~100 traces | Look at data, annotate 50:50 | Success criteria; 20-50 tasks from real failures | Analyze phase: open/axial coding |
| Preferred scale | Binary + critique; no Likert | Binary (precision/recall); pairwise for subjective | Binary or 1-5; constrained output | Binary graders; label columns |
| Grader priority | Fix first; code asserts; judge last | Task-specific metrics; classifiers for guardrails | Code > LLM > human | Python graders + LLM graders |
| Judge validation | TPR/TNR on held-out test; Rogan-Gladen; bootstrap CI | Precision/recall vs human; correlation for pairwise | Calibrate vs humans; "Unknown" option | TPR/TNR; 20/40/40 splits |
| Dataset size stance | Small, expert-labelled, saturated | Balanced annotated set | Volume over quality (docs); start small (agents) | ~50 rows to start |
| CI stance | 100+ curated, deterministic in CI; judges async in prod | Evaluate every change vs baseline | Regression suites near 100%, every commit | Graders in CI/CD; Promptfoo configs |
| Non-determinism | Re-run after changes; CIs in prod | Run inputs multiple times | pass@k vs pass^k; SEM, clustering, pairing | -- |

Sources: [4][5][9][10][11][12][13][14][15][18].

## 9. Human review tooling and the role of domain experts

Every source treats reading traces as the bottleneck, so the review interface is part of the evaluation architecture, not a convenience. Husain recommends removing all friction from looking at data. Rechat's custom viewer showed the feature and scenario under test, whether the input was synthetic or real, cross-tool filters and deep links to the CRM and trace log, and made outputs editable so that corrections became fine-tuning data [1]. The Field Guide lists the features that matter: all context on one screen, one-click pass/fail, free-text notes, filtering and sorting, and hotkeys. With AI-assisted coding such tools can be built in hours, and a spreadsheet is better than nothing [3]. The FAQ claims teams with custom annotation tools iterate about 10x faster (a practitioner estimate, not a controlled measurement), and cites one interface handling more than 400 results per query [4]. The audit skill flags raw JSON, trace data crammed into spreadsheet cells, and output-only review as problems [5].

On staffing, the "benevolent dictator" model wins for small and medium teams: one domain expert sets the quality bar [2][4]. Multiple annotators require measuring inter-annotator agreement, for example with Cohen's kappa [4]. The Field Guide also argues that domain experts should edit prompts directly in "integrated prompt environments", admin versions of the real product UI, instead of passing knowledge to engineers through slides [3]. Yan and the "Year of Building" authors put trace review into the on-call rotation and recommend daily sampling of production logs, writing an assertion as soon as a new issue appears [8][11].

## 10. Guardrails versus evaluators, and production monitoring

Product-eval sources keep two things apart that often get conflated [4][8]:

- **Guardrails** run inline on the request path and must be fast, with a latency budget of a few milliseconds. They are usually deterministic (regex, schema, PII detection, moderation APIs) and target clear, high-impact failures. False positives are treated as production bugs, so teams aim for a very low false-positive rate. Slow LLM judges are almost never used as synchronous guardrails. A cascade that sends only borderline cases to a slower check is one option [4].
- **Evaluators** run after the response, often asynchronously, and may use LLM judges for subjective criteria [4]. Reference-free evals such as summary consistency or translation quality estimation can double as guardrails when they are fast enough: a low score means the output is withheld or regenerated [8].

In production, the sources recommend [4][8][11]:

- Sample traces asynchronously and run validated, reference-free judges.
- Report bias-corrected rates with confidence intervals.
- Track simple drift signals: input and output length, format compliance, and embedding clusters of topics.
- Keep hold-out sets in line with current user behaviour, including typos.
- Run inputs several times to catch rare anomalies.
- Log missing or blocked outputs too.

Yan adds that implicit feedback from user interactions is worth designing for. Explicit feedback is sparse and biased but still useful [11]. Anthropic's "Swiss cheese" framing covers the same ground: automated evals, production monitoring, A/B tests, user feedback, manual transcript review and systematic human studies each catch failures the others miss [12].

## 11. Failure modes of eval programs

Drawn from the sources above:

1. **Brainstormed rather than observed criteria.** Generic labels such as "helpfulness" or "hallucination" signal skipped error analysis and lead to metrics that move without pointing at fixes [3][5].
2. **Evaluators before fixes.** Building judges for failures a one-line prompt change would remove wastes labelling and maintenance budget [4][5].
3. **Likert scales.** Adjacent points are subjective, need larger samples to detect differences, and encourage hedging toward the middle. Several binary sub-checks are better, for example "4 of 5 expected facts present" [4].
4. **Unvalidated or leaky judges.** Reporting raw judge pass rates, raw accuracy, dev-set performance or point estimates, or drawing few-shot examples from the test set [5][15].
5. **Criteria drift ignored.** Freezing a rubric before seeing outputs, and not re-grading after model or prompt changes [6].
6. **Saturation and Goodhart effects.** Suites at 100% stop discriminating [12]. Over-optimising one eval, such as needle-in-a-haystack recall, can hurt real tasks: the "Year of Building" paper reports about 80% recall for medication questions in doctor-patient transcripts against about 30% for an inserted pizza-topping needle [8].
7. **Broken tasks and graders.** Ambiguous specifications and rigid grading understate capability. The CORE-Bench 42% -> 95% correction is the cautionary example [12]. Shared state between trials inflates scores [12].
8. **Statistical overconfidence.** Unclustered standard errors (more than 3x too small on some evals) and unpaired comparisons produce false "wins" [14].
9. **Floating judge models.** Unpinned judge aliases drift silently [5].
10. **Dev-prod skew.** Test sets that do not match production formatting, casing, typos or topics [8].

## Key takeaways for system designers

- **Make trace review a first-class subsystem.** Log full traces, including tool calls and retrieved context. Build a domain-specific review UI that supports one-click binary labels and notes, and budget expert time for it. Most of the return comes from this loop, not from the judge [1][3][11].
- **Derive the taxonomy bottom-up.** Use open coding on the first failure per trace, axial coding into 5 to 10 actionable categories, and about 100 traces or saturation. Then rank by frequency times impact [4][5].
- **Fix, then assert, then judge.** Prompt and tool fixes come first. Deterministic assertions (regex, schema, counts, execution) cover objective criteria and run in CI. LLM judges are for persistent subjective failures only, one failure mode per judge, binary output [5].
- **Treat each judge as a classifier with its own eval set.** Use about 100 balanced expert labels, train/dev/test splits, TPR and TNR above 80 to 90%, Rogan-Gladen-corrected production rates with bootstrap CIs, pinned model snapshots, and re-validation on change [5][15].
- **Expect criteria drift.** Interleave grading and criterion editing, keep critiques as few-shot material, and re-grade after any model, prompt or upstream change [2][6].
- **Separate capability and regression suites.** Keep capability evals hard (around 70% or lower is healthy) and regression suites near 100% on every commit. Graduate saturated capability tasks into regression, and replace them with harder ones [4][12].
- **Gate CI on diffs with statistics.** Use paired comparisons against a baseline experiment, report SEMs and clustered SEs, size suites with power analysis, use pass^k for consistency-critical agents, and cache LLM calls keyed on prompt and config hashes [14][12][19].
- **Keep eval definitions in the repo.** Portable configs (Promptfoo YAML, pytest-style DeepEval, eval files plus a CI action) outlast hosted dashboards. OpenAI's own 2026 migration path illustrates this [18][19][21][22].
- **Keep guardrails separate from evaluators.** Inline checks must be fast, deterministic and very low in false positives. LLM judges belong in asynchronous monitoring [4].

## References

1. Hamel Husain. "Your AI Product Needs Evals." hamel.dev, 2024. https://hamel.dev/blog/posts/evals/
2. Hamel Husain. "Creating a LLM-as-a-Judge That Drives Business Results." hamel.dev, 2024. https://hamel.dev/blog/posts/llm-judge/
3. Hamel Husain. "A Field Guide to Rapidly Improving AI Products." hamel.dev, 2025. https://hamel.dev/blog/posts/field-guide/
4. Hamel Husain and Shreya Shankar. "LLM Evals: Everything You Need to Know" (AI Evals FAQ). hamel.dev, 2025 (continuously updated). https://hamel.dev/blog/posts/evals-faq/
5. Hamel Husain and Shreya Shankar. "evals-skills" (skills: error-analysis, validate-evaluator, eval-audit, generate-synthetic-data, write-judge-prompt, evaluate-rag, build-review-interface). GitHub, 2026. https://github.com/hamelsmu/evals-skills
6. Shreya Shankar, J.D. Zamfirescu-Pereira, Bjorn Hartmann, Aditya G. Parameswaran, Ian Arawjo. "Who Validates the Validators? Aligning LLM-Assisted Evaluation of LLM Outputs with Human Preferences." UIST 2024. https://arxiv.org/abs/2404.12272
7. Shreya Shankar, Haotian Li, Parth Asawa, Madelon Hulsebos, Yiming Lin, J.D. Zamfirescu-Pereira, Harrison Chase, Will Fu-Hinthorn, Aditya G. Parameswaran, Eugene Wu. "SPADE: Synthesizing Data Quality Assertions for Large Language Model Pipelines." arXiv, 2024. https://arxiv.org/abs/2401.03038
8. Eugene Yan, Bryan Bischof, Charles Frye, Hamel Husain, Jason Liu, Shreya Shankar. "What We've Learned From A Year of Building with LLMs." 2024. https://applied-llms.org/
9. Eugene Yan. "Task-Specific LLM Evals that Do & Don't Work." eugeneyan.com, 2024. https://eugeneyan.com/writing/evals/
10. Eugene Yan. "Evaluating the Effectiveness of LLM-Evaluators (aka LLM-as-Judge)." eugeneyan.com, 2024. https://eugeneyan.com/writing/llm-evaluators/
11. Eugene Yan. "An LLM-as-Judge Won't Save The Product, Fixing Your Process Will." eugeneyan.com, 2025. https://eugeneyan.com/writing/eval-process/
12. Anthropic. "Demystifying evals for AI agents." Anthropic Engineering, 2026. https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents
13. Anthropic. "Define success criteria and build evaluations." Claude Platform Docs, accessed 2026. https://platform.claude.com/docs/en/test-and-evaluate/develop-tests
14. Anthropic. "A statistical approach to model evaluations." Anthropic Research, 2024 (paper: Evan Miller, "Adding Error Bars to Evals," https://arxiv.org/abs/2411.00640). https://www.anthropic.com/research/statistical-approach-to-model-evals
15. OpenAI. "Building resilient prompts using an evaluation flywheel." OpenAI Cookbook, 2025. https://github.com/openai/openai-cookbook/blob/main/examples/evaluation/Building_resilient_prompts_using_an_evaluation_flywheel.md
16. OpenAI. "Getting Started with OpenAI Evals." OpenAI Cookbook, 2024. https://github.com/openai/openai-cookbook/blob/main/examples/evaluation/Getting_Started_with_OpenAI_Evals.ipynb
17. OpenAI. "Evals API use case: regression evaluation" (regression.ipynb). OpenAI Cookbook, 2025. https://github.com/openai/openai-cookbook/blob/main/examples/evaluation/use-cases/regression.ipynb
18. OpenAI. "Moving from OpenAI Evals to Promptfoo." OpenAI Cookbook, 2026. https://github.com/openai/openai-cookbook/blob/main/examples/evaluation/moving-from-openai-evals-to-promptfoo.md
19. Promptfoo. "CI/CD integration." Promptfoo documentation, accessed 2026. https://www.promptfoo.dev/docs/integrations/ci-cd/
20. Promptfoo. "Assertions and metrics (expected outputs)." Promptfoo documentation, accessed 2026. https://www.promptfoo.dev/docs/configuration/expected-outputs/
21. Confident AI. "DeepEval: the LLM evaluation framework" (README). GitHub, accessed 2026. https://github.com/confident-ai/deepeval
22. Braintrust. "Braintrust Eval Action" (README). GitHub, accessed 2026. https://github.com/braintrustdata/eval-action
