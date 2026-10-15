# 01. Evaluation Foundations and Taxonomy: Purposes, Modes, and the Eval Stack

## Overview

"Evals" covers activities that have little in common beyond the word. A frontier lab running GPQA against a pretraining checkpoint, a product team checking whether its support bot still returns valid JSON after a prompt edit, a safety team testing whether a model gives meaningful uplift toward a biological weapon, and an observability platform sampling 5% of production traces for an LLM judge are all "running evals." They differ in who owns the result, what decision depends on it, what counts as ground truth, and how much a wrong answer costs.

This chapter sets out the vocabulary and the coordinate system used in the rest of the Atlas. It covers:

1. A shared vocabulary (task, trial, grader, transcript, outcome, harness, suite), lined up across the major frameworks.
2. Four orthogonal taxonomy axes:
   - **Purpose**: capability, product, or safety.
   - **Timing and data source**: offline or online.
   - **Provenance**: public benchmark or task-specific eval.
   - **Grader**: code, model, or human.
3. A reference architecture for the "eval stack": datasets, subject adapters, runners, graders, storage, analysis and dashboards, and gates. Each layer is grounded in public systems: UK AISI Inspect, OpenAI Evals, Stanford HELM, EleutherAI lm-evaluation-harness, LangSmith, Databricks MLflow 3, and Google Vertex AI.
4. The eval lifecycle, from defining success criteria through saturation and retirement. It draws on Anthropic's 2026 agent-eval roadmap, practitioner methods (Husain, Yan, Shankar), and the academic EDDOps process model.
5. A catalogue of failure modes and lessons for system designers.

Later chapters cover individual components in depth: LLM-as-judge, agent sandboxes, online monitoring, red-teaming infrastructure, and so on. This chapter is the map.

---

## 1. Shared vocabulary

The most precise public vocabulary for modern (especially agentic) evals is Anthropic's January 2026 engineering post "Demystifying evals for AI agents" [1]. It defines:

- **Eval:** a test for an AI system. You give it an input and apply grading logic to its output.
- **Task:** a single test case with defined inputs and success criteria.
- **Trial:** one attempt at a task. Several trials are run because outputs vary between runs.
- **Grader:** logic that scores some aspect of performance. A task can have several graders, each with several assertions.
- **Transcript (trace, trajectory):** the full record of a trial, including outputs, tool calls, reasoning, and intermediate results.
- **Outcome:** the final state of the environment after a trial. For example, whether a reservation actually exists in the database, not whether the agent said it made one.
- **Evaluation harness:** the infrastructure that runs evals end to end. It provides tools, runs tasks concurrently, records steps, grades, and aggregates results.
- **Agent harness (scaffold):** the system that lets a model act as an agent. Evaluating "an agent" means evaluating the harness and the model together.
- **Eval suite:** a set of tasks that measure one capability or behavior.

The **transcript** vs **outcome** distinction matters for architecture. It separates graders that read what the agent *said* from graders that inspect what the agent *did* to the world. The **evaluation harness** vs **agent harness** distinction matters too. A large share of eval bugs come from the eval harness quietly differing from the production agent harness, which is why Anthropic recommends keeping the eval agent "close to production" [1].

Each framework names these concepts differently. The mapping below avoids a lot of confusion when moving between tools:

| Concept | Anthropic [1] | UK AISI Inspect [14] | OpenAI Evals [12] | LangSmith [16] | MLflow 3 [25] |
|---|---|---|---|---|---|
| Test case | Task | Sample (in a Dataset) | Sample (JSONL row) | Example (in a Dataset) | Eval dataset record / trace |
| System under test | Agent harness + model | Solver / Agent | Completion Function | Target app (run) | App passed to `evaluate()` |
| Scoring logic | Grader | Scorer (+ metrics) | Eval template / model-graded spec | Evaluator (returns feedback) | Scorer |
| One execution | Trial | Epoch of a sample | Completion | Run | Trace |
| Collection | Eval suite | Task / Eval set | Registry eval / eval set | Dataset + experiment | Evaluation run |
| Stored result | Transcript + outcome | Eval log (`.eval`/`.json`) | Recorder output (local/Snowflake) | Experiment (outputs, scores, traces) | Traces with attached feedback |

LangSmith also draws a terminological line between **evaluation** and **testing** [16]. Evaluation *measures* performance according to metrics. These are often fuzzy and most useful in relative terms. Testing *asserts* correctness, and a system ships only if it passes every test. Metrics turn into tests when you set a threshold or require "no worse than baseline." This is the same move that Anthropic calls *graduating* a capability eval into a regression eval [1].

---

## 2. Axis 1: Purpose (capability, product, safety)

The first question to ask of any eval is *what decision it informs*. Three families dominate public practice.

### 2.1 Capability evals (model-centric)

**Problem.** Model developers and researchers need to know what a *model* can do across a broad range of skills, comparably across models and over time. Typical questions: does this checkpoint reason better than the last one, and how does it compare with competitors?

**Canonical designs.** Stanford's HELM, introduced in 2022 [10], made two contributions:

- An explicit *taxonomy*. A scenario is a task, a domain, and a language. Each scenario is crossed with seven metric categories: accuracy, calibration, robustness, fairness, bias, toxicity, and efficiency.
- A *concrete* evaluation of 30 models on 42 scenarios. HELM reported that before it, models had been evaluated on average on only 17.9% of its core scenarios, and HELM raised that to 96.0% [10].

The paper says the taxonomy also serves to make explicit what is *missing*, which is a design principle worth copying. The HELM framework [11] still ships standardized datasets (for example MMLU-Pro, GPQA, IFEval, WildBench), a unified model interface, metrics beyond accuracy, a per-instance web UI, and public leaderboards. Its README notes that HELM entered maintenance mode on June 1, 2026 [11]. That is a reminder that eval infrastructure has a life cycle just as models do.

EleutherAI's lm-evaluation-harness [15] is the other workhorse for capability evals. It implements over 60 academic benchmarks with hundreds of subtasks, has YAML-configured tasks and Jinja2 prompt templates, supports backends from Hugging Face `transformers` and vLLM to commercial APIs, and is the backend for Hugging Face's Open LLM Leaderboard. Its README says dozens of organizations use it internally.

**Failure modes.** Anthropic's 2023 "Challenges in evaluating AI systems" [2] is still the most candid account of capability-eval pain:

- **MMLU.** Small formatting changes, such as different option labels or an added space, could move accuracy by about 5%. Contamination and inconsistent few-shot and chain-of-thought (CoT) choices made cross-lab comparisons unreliable.
- **BIG-bench.** Its 204 evaluations needed substantial engineering to install and did not scale to Anthropic's infrastructure.
- **HELM.** Third-party runs did not use Claude's native prompt format, and evaluating new models could take months.

Capability numbers are artifacts of harness choices as much as of models.

### 2.2 Product evals (application-centric, task-specific)

**Problem.** A team shipping an LLM *application* needs to know whether *its system* (prompts, retrieval, tools, model choice, guardrails) does *its job* for *its users*. Public benchmarks barely touch this question.

**Canonical designs.** Practitioner frameworks converge on small, specific, frequently run evals:

- **Husain's three levels** [7]: Level 1 unit tests (assertions), Level 2 model and human evaluation (which requires logged traces), and Level 3 A/B testing. Cost rises with each level, so cadence falls: Level 1 on every code change, Level 3 only after significant product changes.
- **Yan's "Product Evals in Three Simple Steps"** [8]:
  1. Label a small dataset, using binary pass/fail for objective criteria and win/lose/tie for subjective ones. Yan argues 1-5 Likert scales are hard to calibrate for both humans and LLM judges.
  2. Align LLM evaluators to those labels.
  3. Run the experiment-plus-eval harness on every configuration change.
- **Anthropic's developer documentation** [5]: be task-specific (mirror the real task distribution, including edge cases such as empty, overlong, harmful, or ambiguous inputs), automate grading where possible, and prefer volume of automatically graded cases over a few hand-graded ones.

**Industrial examples.**

- **GitHub Copilot.** GitHub describes offline evaluations that run before any change reaches production, mostly on tooling it builds itself [28]. Secondary write-ups of the Copilot evaluation system describe a "harness lib" that checked code completions against real unit tests from open-source repositories. Those are secondary sources, so treat the details as approximate.
- **LinkedIn.** LinkedIn's early generative-AI work hit the scaling wall of product evals: each prompt iteration took hours of inference, and the samples were shared in large spreadsheets for a team of editors to review [29].

**Failure modes.**

- **Criteria drift.** Shankar et al. found that people need criteria to grade outputs, yet grading outputs is how they discover their criteria. Some criteria depend on the specific outputs observed [20]. Product evals therefore cannot be fully specified up front.
- **Pass-rate absolutism.** Husain notes that, unlike unit tests, product evals need not pass 100%. The target pass rate is a product decision [7].

### 2.3 Safety evals (risk-centric)

**Problem.** Safety evals aim to determine whether a system can cause, or can be misused to cause, harm. In frontier settings, they also determine whether a model crosses capability thresholds that trigger stronger safeguards.

**Canonical design: layered sociotechnical evaluation.** Weidinger et al. (Google DeepMind, 2023) [17] propose three layers, each adding context:

- **Capability:** model components tested in isolation. These results are early indicators, not proof of real-world harm.
- **Human interaction:** harm at the point of use, which depends on who the user is, their purpose, and the setting.
- **Systemic impact:** effects on society, the economy, and the environment.

The authors say the layers are "a matter of degree" rather than cleanly separated. They also identify a "context gap": evaluations above the capability layer are rare.

**Frontier safety frameworks** turn capability evals into *governance gates*:

- **Anthropic's Responsible Scaling Policy.** It defines capability thresholds (for example, AI R&D and CBRN). Crossing one requires a higher level of safeguards. The RSP updates page records version 3.4 (effective July 8, 2026) [6]. It notes that the evaluation interval moved from 3 to 6 months "to avoid rushed elicitation," and that autonomy evaluations use tasks taking an expert roughly 2-8 hours. It also acknowledges elicitation gaps in past evaluations, such as missing best-of-N or CoT prompting, which Anthropic now tracks.
- **Google DeepMind's Frontier Safety Framework (2024).** It introduced "early warning evaluations" designed to alert before a model reaches a Critical Capability Level, run often enough to give notice before the threshold is reached [26].
- **OpenAI's Preparedness Framework v2 (April 2025).** It tracks biological and chemical, cybersecurity, and AI self-improvement categories against High and Critical thresholds [27].

**Architectural consequences.** Safety evals differ from product evals in ways that change the infrastructure:

- **Elicitation.** Elicitation effort is part of the measurement. An under-elicited model looks safer than it is, so the runner must support scaffolds, many trials, and tool access.
- **Secure evaluation.** Content is often sensitive: Anthropic's 2023 post cites security clearances, information-sharing limits, and legal risk around controlled information [2]. This drives access-controlled storage and restricted transcripts.
- **Gating, not tracking.** Results feed a *decision record* (for example, a capability determination or risk report [6]), not just a dashboard.

### 2.4 The orthogonal axis: capability vs regression

Anthropic also separates **capability (quality) evals** from **regression evals** [1]. This distinction cuts across all three purposes:

- **Capability evals** ask what the system can do. They should *start with a low pass rate* so the team has a target to climb.
- **Regression evals** ask whether the system still does what it used to. They should sit near 100%.

A capability eval that reaches a high pass rate "graduates" into the regression suite. A saturated eval at 100% protects only against regressions and has stopped measuring progress [1]. This is the cleanest public statement of how suites should move through their life cycle (Section 6).

---

## 3. Axis 2: Offline vs online

**Offline evaluation** runs before deployment, on curated datasets that usually have reference outputs. **Online evaluation** runs on production traffic (traces, runs, threads) that has no reference outputs. LangSmith's documentation [16] states the consequence most clearly:

- Offline evals can check *correctness against expected answers*.
- Online evals can only check *quality patterns, safety, and real-world behavior*, using reference-free techniques.

| Dimension | Offline | Online |
|---|---|---|
| Runs on | Curated dataset examples | Live traces / runs / threads |
| Data available | Inputs, outputs, reference outputs | Inputs, outputs, metadata, user feedback |
| Timing | Batch, pre-deployment (CI, release) | Real-time or near-real-time, sampled |
| Typical uses | Benchmarking, unit/regression tests, backtesting, pairwise model comparison | Monitoring, anomaly detection, harvesting failures, A/B tests |
| Usable graders | Reference-based and reference-free | Reference-free only (plus delayed human labels) |
| Cost control | Dataset size, caching, trials | Sampling rate, filters, spend limits |
| Main risk | Dataset unrepresentative of real usage | Noisy, reactive, no ground truth |

(Adapted from LangSmith [16], MLflow 3 [25], and Anthropic [1].)

### 3.1 Shared graders across modes

A key architectural pattern is **grader reuse across offline and online**:

- **MLflow 3.** Production monitoring runs the *same* scorer functions used in development on a configurable sample of incoming traces. Results are attached as feedback to each trace, so scorers are not rewritten at deployment [25].
- **LangSmith.** Evaluators are workspace-level resources that can attach to many datasets and tracing projects. Sampling rates, filters, and spend limits are set per attachment [16].

Only **reference-free** graders can move between the two modes: safety checks, format validation, quality heuristics, and reference-free LLM judges. Reference-based graders remain offline-only. Designers should therefore write as many criteria as possible in reference-free form.

### 3.2 The closed loop

Every mature source describes the same loop. Failures detected online become offline test cases. Offline evals validate the fix. Online evals confirm that production improved [16]. The EDDOps reference architecture from CSIRO's Data61 [18] formalizes this as a single closed feedback loop that unifies development-time and runtime evaluation. It argues that fixed benchmarks and static test suites miss emergent agent behavior.

### 3.3 Where A/B testing and human review fit

Anthropic places automated evals inside a "Swiss cheese" model of layered defenses [1]:

| Method | Strength | Weakness |
|---|---|---|
| Automated evals | Fast, reproducible, no user impact, can run on every commit | Upfront investment; false confidence if unrepresentative |
| Production monitoring | Real behavior at scale | Reactive, noisy, no ground truth |
| A/B testing | Measures real outcomes, controls confounds | Slow; little insight into *why* |
| User feedback | Surfaces unanticipated problems | Sparse, self-selected, skewed to severe issues |
| Manual transcript review | Builds intuition, catches subtle issues | Doesn't scale |
| Systematic human studies | Gold standard for subjective tasks; calibrates LLM graders | Expensive, slow |

Each layer catches failures the others miss. No single layer is sufficient.

Chatbot Arena [23] is a hybrid: an *online* (live, crowdsourced) evaluation that produces a *benchmark-style* public ranking. Users compare two anonymous models and vote, and rankings come from a Bradley-Terry model. The paper reported over 240K votes at submission. Later ICML slides cited over 1.5 million votes across 100+ models, and the authors reported good agreement between crowd and expert votes.

---

## 4. Axis 3: Benchmarks vs task-specific evals

**Benchmarks** are shared, public, fixed datasets with standardized scoring, built for *comparison across systems*. **Task-specific evals** are private datasets built from one system's real inputs and failure modes, used for *decisions about that system*. Kapoor et al. argue that the benchmarking needs of model developers and downstream developers have been conflated [24]. That conflation is the root error this axis exists to prevent.

### 4.1 Why benchmarks mislead product decisions

- **Construct validity.** Bean et al. (NeurIPS 2025) [22] had 29 expert reviewers examine 445 LLM benchmarks. They found that choices of phenomenon, task, and scoring metric often undermine the validity of claims drawn from benchmarks, especially for abstract properties like "safety" or "robustness." Their eight recommendations include defining the phenomenon, measuring only that phenomenon, representative sampling, contamination planning, sound statistics, and error analysis.
- **Quality variance.** BetterBench [21] scored 24 benchmarks against 46 criteria covering five life-cycle phases (design, implementation, documentation, maintenance, retirement). Quality varied widely, and most benchmarks did not report statistical significance or allow easy replication.
- **Overfitting and cost blindness.** Kapoor et al. [24] found that many agent benchmarks lack adequate holdout sets, which produces agents that take shortcuts. They also showed that simple baselines (for example, retrying a stochastic model) can match complex agents on HumanEval at a fraction of the cost. Their conclusion: agent evaluation should report accuracy against cost, ideally as a Pareto frontier.
- **Saturation.** Anthropic reports SWE-bench Verified scores rising from roughly 40% to over 80% in about a year [1]. A saturated benchmark guards only against regressions.
- **Harness sensitivity.** Opus 4.5 initially scored 42% on CORE-Bench. After grading bugs were fixed and an over-constrained scaffold was loosened, it scored about 95% [1]. Benchmark numbers often measure the harness.

### 4.2 Why task-specific evals are not enough alone

Task-specific evals are cheap to start: Anthropic suggests 20-50 tasks drawn from real failures [1], and LangSmith suggests 5-10 examples per critical component, then 10-20 curated examples [16]. Their weakness is selection bias. They cover the failures a team has already seen. They also need ongoing curation as the product and its traffic change. Benchmarks remain useful as a *broad prior* (model shortlisting) and as a cross-check for regressions in general capability that a narrow product suite would miss.

### 4.3 Practical split

| Use case | Prefer | Why |
|---|---|---|
| Choosing a base model shortlist | Public benchmarks + leaderboards | Cheap broad prior; comparable across vendors |
| Prompt / RAG / tool changes | Task-specific offline suite | Measures *your* distribution and failure modes |
| Release gate for an app | Task-specific regression suite (+ a small general canary set) | Near-100% suite catches breakage |
| Frontier risk thresholds | Purpose-built safety evals, expert-designed, with elicitation | Benchmarks lack threat-model fidelity |
| Post-launch quality | Online reference-free graders + A/B tests | No references in production |

---

## 5. Axis 4: Grader type

Graders are covered in later chapters. Here they are placed in the taxonomy because grader choice shapes every other layer: it sets cost, latency, determinism, and online portability.

| Grader | Methods | Strengths | Weaknesses | Online-portable? |
|---|---|---|---|---|
| Code-based | Exact/fuzzy/JSON match, regex, unit tests, static analysis, outcome/state checks, tool-call verification | Fast, cheap, reproducible, debuggable | Brittle to valid variation; little nuance | Yes if reference-free (format, safety regex) |
| Model-based | Rubrics, natural-language assertions, pairwise, reference-based, multi-judge consensus, adaptive rubrics | Flexible, scales, handles open-ended output | Non-deterministic, costs per call, needs calibration | Yes (reference-free variants) |
| Human | SME review, crowdsourcing, spot checks, annotation queues, A/B, inter-annotator agreement | Gold standard; calibrates other graders | Slow, expensive, hard to scale | Partially (sampled queues) |

(Synthesized from Anthropic [1][5], LangSmith [16], and Vertex AI [19].)

Some details matter in practice:

- **OpenAI Evals templates [12].** The basic templates show how simple code graders are: `Match` is `any(a.startswith(b) for b in B)`, `Includes` is `any(b in a for b in B)`, and `FuzzyMatch` checks containment either way. The model-graded template wraps the completion in an evaluation prompt and parses it into `choice_strings`, mapped to `choice_scores` (unparsable responses become `__invalid__`). Of the `eval_type` options, the docs recommend `cot_classify` (reason, then answer) as the most accurate.
- **Vertex AI Gen AI evaluation service [19].** It recommends *adaptive rubrics* by default. The service generates pass/fail rubric tests per prompt, then judges each response against them. Static rubrics, computation-based metrics (for example ROUGE, BLEU), and custom functions are the alternatives. Search excerpts of Google's managed-metric documentation say one managed "general quality" metric makes about six judge-model calls per item, which is a reminder that model graders multiply inference cost (approximate, per documentation excerpts).
- **Anthropic [1][5] on rubric and judge design.** Grade *outcomes* rather than rigid step sequences. Allow partial credit. Give LLM judges an "Unknown" option. Use a different model as judge where possible. Constrain the judge's output format. Validate parsed scores and fail loudly.
- **Calibration.** EvalGen [20] uses a small set of human grades to choose, among candidate LLM-generated evaluators, those best aligned with the human. Yan's AlignEval frames alignment as two-way: humans also calibrate their criteria against AI output [9].

Scores combine in three ways [1]: *weighted* (the combined score must clear a threshold), *binary* (all graders must pass), or *hybrid*.

---

## 6. The eval stack: a reference architecture

All the public systems surveyed separate concerns into roughly the same layers, despite different names. The reference architecture below is synthesized from Inspect [13][14], OpenAI Evals [12], HELM [11], lm-evaluation-harness [15], LangSmith [16], MLflow 3 [25], Vertex AI [19], and the EDDOps reference architecture [18].

```text
 +--------------------------------------------------------------------------------+
 |                       GOVERNANCE / DECISION LAYER                              |
 |  CI gates | release sign-off | safety thresholds (RSP/FSF/PF) | risk reports    |
 +-----------------------------------^--------------------------------------------+
                                     | metrics, CIs, diffs
 +-----------------------------------+--------------------------------------------+
 |                ANALYSIS & DASHBOARDS                                           |
 |  experiment compare | paired diffs + SEs | slice/cluster views | transcript UI  |
 +-----------------------------------^--------------------------------------------+
                                     | query
 +-----------------------------------+--------------------------------------------+
 |                RESULT STORE (logs, traces, scores, feedback)                   |
 |  per-sample transcripts | scores + rationales | config + versions | lineage    |
 +------^----------------------------^-----------------------------------^--------+
        | write                      | write (re-scorable)                | write
 +------+-----------+   +------------+-------------+   +-----------------+--------+
 |   RUNNER /       |-->|        GRADERS           |   |   ONLINE TAP           |
 |   ORCHESTRATOR   |   | code | LLM judge | human  |<--|  sampled prod traces   |
 | concurrency,     |   | queues | metrics/reducers|   |  filters, spend caps   |
 | retries, cache,  |   +--------------------------+   +-----------^------------+
 | epochs, sandbox  |                                              |
 +---^---------+----+                                              | traffic
     |         | invoke                                            |
     |   +-----v------------------------------+        +-----------+------------+
     |   | SUBJECT ADAPTER (system under test)|        |  PRODUCTION APP        |
     |   | model API | completion fn | solver |        |  (agent harness+model) |
     |   | agent harness | external agent CLI |        +------------------------+
     |   +------------------------------------+
 +---+----------------------------------------------------------------------------+
 |                DATASET / TASK REGISTRY                                         |
 |  versioned samples (input, target/rubric, metadata) | splits | YAML task defs  |
 |  provenance: hand-curated | harvested from prod | synthetic | public benchmark |
 +--------------------------------------------------------------------------------+
```

### 6.1 Dataset and task registry

**Responsibility.** This layer holds versioned test cases (input, target or grading guidance, metadata) and the *task definitions* that bind data to solvers and graders.

**Public designs.**

- **Inspect.** Samples have `input` and `target`. `target` can be an ideal answer *or grading guidance*, which matters for rubric-graded tasks. Datasets load from CSV, JSON, Hugging Face, or memory, and `FieldSpec` maps columns declaratively [13].
- **OpenAI Evals.** Evals are registered as YAML in a registry, with sample data in JSONL stored in Git LFS [12].
- **lm-evaluation-harness.** Tasks are YAML configs with Jinja2 prompt templates, so new benchmarks often need no code [15].
- **LangSmith.** Examples have inputs, *optional* reference outputs (never passed to the app, only to evaluators), and metadata. Datasets are versioned automatically on change, versions can be tagged and pinned in CI, and *splits* support ML-style, category-based, or staged-rollout grouping [16].

**Design lessons.**

- **Pin versions in CI.** Otherwise a dataset edit looks like a model regression [16].
- **Record provenance.** Tag each case as hand-curated, harvested from production failures (negative feedback, high latency, errors, LLM-flagged conversations), synthetic, or public [16].
- **Balance the set.** Anthropic recommends testing both when a behavior *should* and *should not* occur. Its example is "should search" vs "should answer from knowledge" queries, which prevents one-sided optimization [1].
- **Check task solvability.** A 0% pass rate across many trials usually means the task is broken, not the model [1]. Two domain experts should independently reach the same verdict on a task [1].

### 6.2 Subject adapter (system under test)

**Responsibility.** This layer presents the system under test through a uniform interface, so any eval can run against any subject.

**Public designs.**

- **OpenAI Evals' Completion Function Protocol** [12] generalizes a "completion" to any system that maps a text or chat input to a list of text outputs. Prompt chains, browsing agents, and LangChain LLMs can all be registered (as YAML in `registry/completion_fns`) and run against any eval.
- **Inspect's solvers** [13] range from a single `generate()` call to full multi-turn tool-using agents. Inspect can also run *external* agents such as Claude Code, Codex CLI, and Gemini CLI.
- **HELM and lm-evaluation-harness** provide unified model interfaces across providers and local inference [11][15].

**Design lesson.** The adapter boundary decides *what you are measuring*. Anthropic stresses that evaluating "an agent" evaluates the harness plus the model together [1]. Anthropic's 2023 HELM experience shows that a mismatched prompt format alone can depress scores [2]. The adapter should therefore be *the production harness* wherever possible, not a reimplementation.

### 6.3 Runner / orchestrator

**Responsibility.** The runner schedules samples × trials × models. It manages concurrency and provider rate limits, retries, caching, sandboxes, resumption, and cost.

**Public designs (Inspect is the most fully documented).**

- **Eval sets** [14]. `inspect eval-set` runs many tasks across many models. It retries failed evaluations, reuses completed samples from failed tasks, cleans up logs from failed runs, and is *re-entrant*: re-running the same command schedules only unfinished work. You can add models or epochs to an existing set by re-issuing the command. By default it runs `max(10, number of models)` tasks in parallel and balances active tasks across providers.
- **Caching.** Model output is cached on a key of model and base URL, message history, *epoch number* (so trials stay distinct), generation config, and active tools, with a default lifetime of one week [14]. Caching makes grader iteration nearly free.
- **Sandboxes.** Untrusted model code runs in Docker, Kubernetes, Modal, Proxmox, Vagrant, and other systems through an extension API [13].
- **Trials.** Anthropic recommends starting each trial from a clean, isolated environment to avoid correlated failures and unfair advantages [1].

**Scale notes.** Inspect ships over 200 pre-built evaluations and supports more than 20 model providers [13]. MLflow's `mlflow.genai.evaluate()` runs the app and scorers in parallel and records outputs as traces [25].

### 6.4 Graders

**Responsibility.** Graders map each transcript or outcome to scores, then aggregate scores into metrics.

**Architectural pattern: decouple generation from grading.** Inspect can run with `--no-score` and score later with `inspect score`, optionally using a *different* scorer [14]. Its documentation notes this "bypasses the entire generation phase, saving lots of time and inference costs" while a scorer is being developed. Grading should be a pure function of stored transcripts so that graders can be versioned, re-run, and A/B tested without re-sampling the model.

**Aggregation.** Inspect scorers declare metrics, for example `accuracy()` or `mean()` [14]. Multi-trial reduction choices matter for agents, as Anthropic's pass@k and pass^k distinction shows [1]:

- **pass@k** is the probability that at least one of k trials succeeds. It rises with k and fits cases where one success suffices.
- **pass^k** is the probability that *all* k trials succeed. It falls with k and fits customer-facing consistency.
- At 75% per-trial success, pass^3 is about 42% (0.75³).

### 6.5 Result store (logs and traces)

**Responsibility.** The store keeps durable, queryable records of every sample: inputs, full transcripts, scores with rationales, configuration, and versions.

**Public designs.**

- **Inspect logs** [14] are JSON hierarchies stored either as `.eval` (a binary format about one-eighth the size of JSON with incremental sample access) or `.json` (readable, but slow in the viewer above roughly 50 MB). Later versions deduplicate repeated messages and use zstd compression. Inspect reports about 10:1 size reductions on typical agentic benchmarks (SWE-Bench, Cybench), and larger reductions on long-horizon tasks, because naive logging grows O(N²) with transcript length. Logs can go to local disk, Amazon S3, Azure, or Hugging Face buckets.
- **Inspect API logging.** By default only the first few raw model API calls per model are logged, plus all errors. Provider request IDs are always kept, so you can point the provider at a specific request [14].
- **OpenAI Evals** supports optional logging to Snowflake [12].
- **LangSmith and MLflow** store results as *experiments* or *traces with attached feedback* [16][25]. Offline and online results then share one schema, which is what enables the closed loop in Section 3.2.

**Design lesson.** Store full transcripts, not just scores. Anthropic's roadmap makes "read transcripts" an explicit step [1]. Without transcripts you cannot tell whether failures are fair, whether graders are buggy (as in the CORE-Bench case), or whether the model found an exploit.

### 6.6 Analysis, statistics, and dashboards

**Responsibility.** This layer turns scores into *decisions* with known uncertainty.

**Statistical core.** Anthropic's "A statistical approach to model evaluations" (Nov 2024) [3], based on the paper "Adding Error Bars to Evals" [4], recommends:

1. **Treat questions as samples from a question universe.** Report the standard error of the mean (SEM). A 95% CI is mean ± 1.96·SEM.
2. **Cluster standard errors** when questions share a source, such as several questions on one passage. The authors found clustered SEs on popular evals more than three times as large as naive ones.
3. **Reduce within-question variance.** Resample answers several times per question (Inspect's `epochs`), or use next-token probabilities when there is no chain of thought.
4. **Use paired differences** when comparing models on the same questions. Frontier-model question-score correlations were about 0.3-0.7, which makes paired tests much more precise than two-sample tests.
5. **Run a power analysis** to size the eval for the effect you care about, for example a 3-point difference.

BetterBench found most benchmarks report no statistical significance [21]. Dashboards that show bare point estimates are the norm, and they are wrong.

**Dashboards and viewers.**

- Inspect View, a VS Code extension, a log dataframe API, and Inspect Scout for transcript analysis [13][14].
- HELM's per-instance web UI and leaderboards [11].
- LangSmith's side-by-side experiment comparison [16].

**Design lesson.** The primary view should be a *diff between two experiments on the same dataset version*, with paired CIs and drill-down to transcripts. Absolute leaderboards are secondary.

### 6.7 Gates and governance

**Responsibility.** This layer encodes *policy* over metrics: CI pass/fail, release sign-off, and safety-threshold determinations.

**Public designs.**

- **CI gates.** LangSmith turns metrics into tests such as "new version must beat baseline" and supports pytest and Vitest/Jest integration [16].
- **Run cadence.** OpenAI's guidance (per its documentation as surfaced in search) is to run evals on every change, monitor the app for new nondeterminism, and grow the eval set over time [30]. Husain sets cadence by cost level [7].
- **Safety gates.** At the high-stakes end, RSP-style policies bind threshold evaluations to required safeguards and published determinations [6].
- **Ownership.** Anthropic recommends dedicated eval teams owning infrastructure, with domain experts and product teams contributing tasks ("eval-driven development") [1].

---

## 7. Reference systems compared

| System | Org / year | Primary purpose | Unit abstractions | Subject interface | Grading | Storage / UI | Notable design choice |
|---|---|---|---|---|---|---|---|
| HELM [10][11] | Stanford CRFM, 2022 | Capability (holistic, multi-metric) | Scenario × metric | Unified multi-provider client | 7 metric families incl. calibration, efficiency | Web UI + leaderboards | Explicit taxonomy that exposes coverage gaps; maintenance mode from June 2026 |
| lm-evaluation-harness [15] | EleutherAI | Capability (academic benchmarks) | YAML task + Jinja2 prompt | HF, vLLM, SGLang, APIs, many more | Benchmark metrics | Results files; Open LLM Leaderboard backend | Config-only task creation |
| OpenAI Evals (OSS) [12] | OpenAI, 2023 | Capability + custom/product | Registry eval (YAML) + JSONL samples | Completion Function Protocol | Basic templates + model-graded (`cot_classify`) | Local logs; optional Snowflake | Any system as a "completion function"; hosted product later superseded the OSS flow |
| Inspect [13][14] | UK AISI (+ Meridian Labs) | Capability + safety + agentic | Task = dataset + solver + scorer | Solvers, agents, external agent CLIs | Code, model-graded, custom, re-scoring | `.eval` logs, Inspect View, Scout | Re-entrant eval sets, cache keyed by epoch, pluggable sandboxes |
| LangSmith [16] | LangChain | Product (offline + online) | Dataset/example, experiment, run/thread | Traced app | Code, LLM judge, pairwise, human queues | Experiments + traces | Workspace-level evaluators shared across offline and online |
| MLflow 3 GenAI [25] | Databricks | Product (offline + online) | Eval dataset, trace, scorer | `evaluate()` on app | Scorers incl. LLM judges | Traces with feedback | Same scorer code in dev and prod monitoring (beta) |
| Vertex AI Gen AI eval [19] | Google Cloud | Product / model selection | Prompt, response, rubric | `run_inference()` | Adaptive rubrics (default), static rubrics, computation-based | Console + SDK | Per-prompt generated pass/fail rubrics |

A note on hosted platforms: search excerpts of OpenAI's documentation state that its hosted Evals platform becomes read-only for existing users on October 31, 2026 and shuts down on November 30, 2026 [30]. Whatever the reason, the practical lesson is that **eval assets (datasets, grader prompts, transcripts) should be stored in portable formats you control**, because the platforms they run on change faster than the assets lose their value.

---

## 8. The eval lifecycle

Sources describe the lifecycle at three granularities:

- **Per suite:** Anthropic's 0-8 roadmap [1].
- **Per product:** Husain's levels [7], Yan's three steps [8], LangSmith's development → deployment → continuous improvement [16].
- **Per benchmark:** BetterBench's five phases, design through retirement [21].

They compose into one cycle:

```text
        +-------------------------------------------------------------------+
        |                                                                   |
        v                                                                   |
 [1] DEFINE SUCCESS ----> [2] LOOK AT DATA / ERROR ANALYSIS                 |
   task-specific criteria     read traces, label pass/fail,                 |
   edge cases, thresholds     discover criteria (criteria drift)            |
                                   |                                        |
                                   v                                        |
 [4] ALIGN GRADERS <------ [3] BUILD SUITE                                  |
   code first; LLM judge      20-50 real-failure tasks, balanced            |
   calibrated vs human        should/shouldn't cases, reference solns       |
   labels (EvalGen/AlignEval) unambiguous (2 experts agree)                 |
        |                                                                   |
        v                                                                   |
 [5] RUN & COMPARE -------> [6] GATE / DECIDE                               |
   trials, cache, CIs,        CI regression gate, release sign-off,         |
   paired diffs, read         safety threshold determination                |
   transcripts                     |                                        |
                                   v                                        |
 [8] HARVEST & GRADUATE <-- [7] DEPLOY & MONITOR ONLINE                     |
   prod failures -> new       sampled reference-free graders,               |
   offline cases; capability  user feedback, A/B tests,                     |
   evals at ~100% -> regress. annotation queues                             |
        |                                                                   |
        v                                                                   |
 [9] SATURATION REVIEW / RETIRE ---------------------------------------------+
   saturated -> harder tasks; stale/contaminated -> retire
```

### Stage notes and public evidence

1. **Define success.** Be task-specific and cover edge cases (empty, overlong, harmful, or ambiguous inputs) [5]. Decide target pass rates as a product decision [7].
2. **Look at data first.** Husain recommends trace review as the starting point for evaluation, and LangSmith's docs cite that recommendation [7][16]. EvalGen shows criteria emerge *from* grading [20].
3. **Build the suite.** Use 20-50 tasks from real failures, converting manual pre-release checks, bug trackers, and support tickets into cases. Early effect sizes are large, so small samples suffice [1].
4. **Align graders.** Use binary labels to make alignment tractable [8]. Validate LLM judges against human labels [5][20].
5. **Run and compare.** Use multiple trials, paired statistics, and power analysis [3]. Read transcripts [1].
6. **Gate.** Regression suites should sit near 100%; capability suites track progress [1]. Safety gates follow published policy [6].
7. **Monitor online.** Sample and filter traffic, cap spend, and use reference-free graders [16][25].
8. **Harvest and graduate.** Online failures become offline cases [16]. Capability evals that saturate graduate into regression [1].
9. **Review saturation.** A suite at 100% has stopped measuring progress, and saturated evals need harder tasks [1]. Benchmarks need a maintenance and retirement plan [21]. Infrastructure is retired too (HELM maintenance mode [11]).

EDDOps [18] argues that this cycle should be treated as an *architectural* concern rather than a process add-on: evaluation is a first-class runtime component whose results feed back into agent behavior, not a final checkpoint before release.

---

## 9. Cross-cutting failure modes

| Failure mode | Symptom | Documented example | Mitigation |
|---|---|---|---|
| Grader bugs | Score jumps after "no model change" | CORE-Bench 42% → ~95% after fixing grading bugs and loosening scaffold [1] | Read transcripts; re-score stored logs; unit-test graders |
| Harness / format mismatch | Model underperforms vs native use | HELM prompt format vs Claude's format [2]; MMLU formatting shifts of about 5% [2] | Use the production harness as the subject adapter |
| Invalid control | "Perfect" safety/bias score | BBQ bias score 0 because the model wasn't answering at all [2] | Add controls and sanity checks; measure refusal/abstention |
| Broken tasks | 0% across many trials | Anthropic: usually a broken task [1] | Reference solutions; two-expert agreement |
| One-sided suites | Over- or under-triggering | Web-search should/shouldn't balance [1] | Balanced positive/negative cases |
| Overfitting / no holdout | Benchmark gains don't transfer | Agent benchmarks lacking holdouts [24] | Private holdouts; rotate tasks |
| Cost blindness | "Better" agent costs 10× | Simple retry baselines matching complex agents on HumanEval [24] | Report accuracy-cost Pareto |
| Statistical overclaiming | Differences within noise | Clustered SEs about 3× naive [3]; most benchmarks report no significance [21] | SEM, clustering, paired diffs, power analysis |
| Under-elicitation (safety) | Model looks safer than it is | Missing best-of-N / CoT elicitation acknowledged in RSP updates [6] | Elicitation checklists; adequate time per cycle |
| Criteria drift | Judge "accuracy" decays | EvalGen findings [20] | Periodic re-labeling and re-alignment |
| Infra cost of evaluation | Eval engineering crowds out eval science | BBQ took about 2 person-years over 6 months (8 people) [2]; BIG-bench integration abandoned [2] | Reuse open harnesses (Inspect, lm-eval); fund implementations |
| Platform churn | Lost eval assets | Hosted platform retirement [30]; HELM maintenance mode [11] | Portable dataset/log formats you own |

---

## Key takeaways for system designers

1. **Classify every eval on four axes before building it:** purpose (capability, product, or safety), mode (offline or online), provenance (benchmark or task-specific), and grader (code, model, or human). Most design errors come from mixing these, for example using a public benchmark to gate a product release, or trying to run reference-based graders online.
2. **Separate capability suites from regression suites.** Capability suites should start hard (low pass rate). Regression suites should sit near 100%. Graduate cases from the first to the second as they saturate [1].
3. **Build the stack in layers with clean contracts:** a versioned dataset registry, a subject adapter that *is* the production harness, a re-entrant cached runner, graders that are pure functions of stored transcripts, a transcript-complete result store, a paired-statistics analysis layer, and an explicit gate policy.
4. **Decouple generation from grading.** Store transcripts and re-score offline (Inspect's `--no-score` / `inspect score` [14]). Grader iteration then costs almost nothing, and grader bugs become recoverable.
5. **Write criteria in reference-free form wherever you can.** Only reference-free graders port from offline to online, and shared grader code across both modes (MLflow, LangSmith) is what closes the loop [16][25].
6. **Report uncertainty by default:** SEM, clustered SEs, paired differences, and power-based sizing [3]. A dashboard without confidence intervals invites decisions made on noise.
7. **Start small and real:** 20-50 tasks from actual failures [1], with binary labels [8] and judges aligned to human labels [20]. Then grow from production failures.
8. **Measure cost alongside quality.** For agents, accuracy without cost is misleading [24]. Choose pass@k or pass^k deliberately, according to whether one success or consistent success is what users need [1].
9. **Treat safety evals as governance artifacts.** They need elicitation budgets, access-controlled storage, and decision records, not just dashboards [2][6][17].
10. **Own your eval assets.** Frameworks and hosted platforms enter maintenance or shut down (HELM 2026, OpenAI's hosted Evals per its docs). Datasets, grader prompts, and logs should live in formats you control.

---

## References

1. Grace, M., Hadfield, J., Olivares, R., De Jonghe, J. "Demystifying evals for AI agents." Anthropic Engineering, January 2026. https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents
2. Ganguli, D., Schiefer, N., Favaro, M., Clark, J. "Challenges in evaluating AI systems." Anthropic, October 2023. https://www.anthropic.com/research/evaluating-ai-systems
3. Anthropic. "A statistical approach to model evaluations." November 2024. https://www.anthropic.com/research/statistical-approach-to-model-evals
4. Anthropic. "Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations." arXiv:2411.00640, 2024. https://arxiv.org/abs/2411.00640
5. Anthropic. "Create strong empirical evaluations (Develop tests)." Claude Developer Platform documentation, accessed 2026. https://platform.claude.com/docs/en/test-and-evaluate/develop-tests
6. Anthropic. "Responsible Scaling Policy updates" (RSP v3.4, effective July 8, 2026). Accessed October 2026. https://www.anthropic.com/rsp-updates
7. Husain, H. "Your AI Product Needs Evals." hamel.dev, March 2024. https://hamel.dev/blog/posts/evals/
8. Yan, E. "Product Evals in Three Simple Steps." eugeneyan.com, November 2025. https://eugeneyan.com/writing/product-evals/
9. Yan, E. "AlignEval: Building an App to Make Evals Easy, Fun, and Automated." eugeneyan.com, 2024. https://eugeneyan.com/writing/aligneval/
10. Liang, P., et al. "Holistic Evaluation of Language Models." Stanford CRFM, arXiv:2211.09110, 2022. https://arxiv.org/abs/2211.09110
11. Stanford CRFM. "HELM: Holistic Evaluation of Language Models" (GitHub repository and README, incl. maintenance-mode notice). Accessed 2026. https://github.com/stanford-crfm/helm
12. OpenAI. "openai/evals" (README, docs/eval-templates.md, docs/completion-fns.md). GitHub, 2023-2026. https://github.com/openai/evals
13. UK AI Security Institute and Meridian Labs. "Inspect: An open-source framework for large language model evaluations" (docs index). Accessed 2026. https://inspect.aisi.org.uk/ (source: https://github.com/UKGovernmentBEIS/inspect_ai/blob/main/docs/index.qmd)
14. UK AI Security Institute. Inspect documentation: "Log Files," "Eval Sets," "Scoring Workflow," "Caching," "Scorers." GitHub docs sources, accessed 2026. https://github.com/UKGovernmentBEIS/inspect_ai/tree/main/docs
15. EleutherAI (Gao, L., Tow, J., Biderman, S., et al.). "Language Model Evaluation Harness." GitHub, 2021-2026. https://github.com/EleutherAI/lm-evaluation-harness
16. LangChain. "Evaluation concepts." LangSmith documentation, accessed 2026. https://docs.langchain.com/langsmith/evaluation-concepts
17. Weidinger, L., et al. "Sociotechnical Safety Evaluation of Generative AI Systems." Google DeepMind, arXiv:2310.11986, 2023. https://arxiv.org/abs/2310.11986
18. Xia, B., Lu, Q., Zhu, L., Xing, Z., Zhao, D., Zhang, H. "Evaluation-Driven Development and Operations of LLM Agents: A Process Model and Reference Architecture." CSIRO Data61 / UNSW / Adelaide, arXiv:2411.13768, 2024-2025. https://arxiv.org/abs/2411.13768
19. Google Cloud. "Gen AI evaluation service overview." Vertex AI documentation, accessed 2026. https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/evaluation-overview
20. Shankar, S., et al. "Who Validates the Validators? Aligning LLM-Assisted Evaluation of LLM Outputs with Human Preferences." UIST 2024. https://people.eecs.berkeley.edu/~bjoern/papers/shankar-validators-uist2024.pdf
21. Reuel, A., et al. "BetterBench: Assessing AI Benchmarks, Uncovering Issues, and Establishing Best Practices." Stanford, NeurIPS 2024 Datasets and Benchmarks. https://arxiv.org/abs/2411.12990
22. Bean, A. M., et al. "Measuring what Matters: Construct Validity in Large Language Model Benchmarks." University of Oxford et al., NeurIPS 2025 Datasets and Benchmarks. https://proceedings.neurips.cc/paper_files/paper/2025/hash/1967e0fc3aa6cbbace562f5cb8e3954e-Abstract-Datasets_and_Benchmarks_Track.html
23. Chiang, W.-L., Zheng, L., Sheng, Y., et al. "Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference." ICML 2024, arXiv:2403.04132. https://arxiv.org/abs/2403.04132
24. Kapoor, S., Stroebl, B., Siegel, Z. S., Nadgir, N., Narayanan, A. "AI Agents That Matter." Princeton, arXiv:2407.01502, 2024 (TMLR 2025). https://arxiv.org/abs/2407.01502
25. Databricks. "Evaluate and monitor GenAI apps (MLflow 3)" and "Evaluation harness" concepts. Databricks documentation, accessed 2026. https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/concepts/eval-harness
26. Google DeepMind. "Introducing the Frontier Safety Framework." May 2024. https://deepmind.google/discover/blog/introducing-the-frontier-safety-framework/
27. OpenAI. "Preparedness Framework (Version 2)." April 2025. https://cdn.openai.com/pdf/18a02b5d-6b67-4cec-ab64-68cdfbddebcd/preparedness-framework-v2.pdf
28. Adams, C. "How we evaluate AI models and LLMs for GitHub Copilot." GitHub Blog, 2025. https://github.blog/ai-and-ml/generative-ai/how-we-evaluate-models-for-github-copilot/
29. LinkedIn Engineering. "Our learnings from the early days of generative AI." 2023. https://engineering.linkedin.com/blog/2023/our-learnings-from-the-early-days-of-generative-ai
30. OpenAI. "Evaluation best practices." OpenAI API documentation, accessed 2026 (content as surfaced in search excerpts; includes Evals platform retirement notice). https://developers.openai.com/api/docs/guides/evaluation-best-practices
