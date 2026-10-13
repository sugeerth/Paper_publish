# 17. Eval System Design Interview Playbook: Platforms, Product Evals, Metrics, Sampling, and Cost

## Overview

"Design an eval system for X" is now a standard system-design question for applied-AI, ML-platform and research-engineering roles. It is also the question a staff engineer has to answer for real before a launch. The question is hard because no single right architecture exists. The answer depends on the following:

- what decision the eval feeds (ship or block a prompt change, choose a model vendor, detect a production regression, report a capability);
- what can serve as ground truth (unit tests, database state, a reference answer, or only human preference);
- how much variance the system under test has, and how much the grader has;
- how much each verdict costs in tokens, sandbox minutes and expert hours.

This chapter turns the rest of the Atlas into an interview playbook. It gives a reusable answer framework, then five worked designs that cover the most common prompts:

1. a general eval platform;
2. a chatbot / open-ended assistant;
3. a retrieval-augmented generation (RAG) system;
4. a tool-using agent;
5. a coding assistant or coding agent.

Each design covers the problem, the architecture, the key techniques, public numbers and the failure modes. The cross-cutting sections that follow cover metric choice, sampling and statistical power, cost and latency, and the tradeoffs interviewers like to probe.

Every quantitative claim comes from a public primary source: Anthropic engineering posts and documentation, benchmark repositories (SWE-bench, tau-bench, AlpacaEval, Arena-Hard-Auto, MT-Bench, ARES, Ragas, HumanEval, LiveCodeBench, Terminal-Bench), Microsoft Research's Eureka, and the main open-source harnesses. The framework steps, the time budgets and the worked power calculations are the playbook's own synthesis and are labelled as such.

A note on sourcing: the research for this chapter could reach Anthropic, Microsoft Research and GitHub-hosted primary material, but not several well-known practitioner blogs. Practitioner methods are therefore cited here mainly through the primary sources that formalize them. Chapters 1, 4 and 7 cover the practitioner literature directly.

---

## 1. The answer framework (the "eight boxes")

Strong candidates do not start by naming a tool. They walk through a fixed sequence, and they say out loud which decision each box serves. This playbook recommends the following order and rough time budget for a 45-minute round.

| # | Box | Key questions | Time |
|---|-----|---------------|------|
| 1 | Decision and stakes | What decision does the eval gate? Who consumes it? Cost of false pass vs false fail? | 3 min |
| 2 | Success criteria | Specific, measurable, achievable, relevant; multi-dimensional (quality, safety, latency, price) | 4 min |
| 3 | Eval taxonomy | Capability vs regression; offline vs online; code / model / human graders | 4 min |
| 4 | Data | Sources (real failures, logs, synthetic, benchmarks), coverage, balance, contamination | 7 min |
| 5 | Graders | Outcome vs transcript grading; judge design and calibration; partial credit | 8 min |
| 6 | Statistics | Trials per task, CIs, paired comparison, power, pass@k vs pass^k | 6 min |
| 7 | Infrastructure | Harness, sandboxes, isolation, caching, storage, CI gates, dashboards | 8 min |
| 8 | Online loop | Production sampling, monitoring, feedback, incident-driven eval growth | 5 min |

Box 2 has a canonical public formulation. Anthropic's evaluation documentation asks that success criteria be specific, measurable, achievable and relevant. It lists common dimensions: task fidelity, consistency, relevance and coherence, tone and style, privacy preservation, context utilization, latency and price [6]. The same documentation shows a multidimensional target written as one sentence, for example an F1 of at least 0.85, 99.5% non-toxic outputs, and 95% of responses under 200 ms [6]. Quote a target in that form early in the interview. It anchors every later design choice.

Box 3 is where you set up your vocabulary. The most precise public vocabulary is Anthropic's January 2026 agent-eval post [1]. It defines:

- a **task** (one test case with inputs and success criteria);
- a **trial** (one attempt, repeated because outputs vary);
- a **grader** (logic scoring one aspect, possibly with multiple assertions);
- a **transcript** (the full record of a trial);
- an **outcome** (the final state of the environment, such as whether the reservation exists in the database, not whether the agent said it booked one);
- the **evaluation harness** (runs tasks concurrently, records, grades, aggregates) and the **agent harness** (the scaffold that makes a model act as an agent). Evaluating "an agent" means evaluating the model and the scaffold together [1].

The same post separates **capability evals**, which should start at a low pass rate to give a "hill to climb", from **regression evals**, which should sit near 100%. Capability tasks that saturate "graduate" into the regression suite [1]. Interviewers often probe this split. It decides how often each suite runs, which thresholds gate a deploy, and when a suite is retired.

---

## 2. Design A: "Design an evaluation platform"

### Problem

Many teams ship LLM features. Each team needs to run versioned test suites against model, prompt and scaffold variants. They need graders ranging from string match to sandboxed test execution to LLM judges. The platform must compare runs with honest uncertainty, gate CI/CD, and feed production samples back into datasets. This is the platform version of the question, and it is the one most often asked of infrastructure candidates.

### Reference architecture

```text
                +--------------------------------------------------+
                |               Dataset / Task Registry            |
                | versioned tasks, splits (dev / held-out), tags,  |
                | provenance (incident, synthetic, benchmark), PII |
                +-------------------------+------------------------+
                                          |
   Subject config ---------+              v
   (model, prompt,         |    +--------------------+     +--------------------+
    scaffold, tools,       +--->|  Run Orchestrator  |---->|  Sandbox Pool      |
    sampling params)            |  (fan-out trials,  |     |  (containers, fresh|
                                |   retries, quotas, |<----|   state per trial) |
                                |   rate limits)     |     +--------------------+
                                +---------+----------+
                                          | transcripts + final env state
                                          v
                +--------------------------------------------------+
                |                 Grader Service                   |
                |  code graders | LLM judges (versioned prompts) | |
                |  human review queue (calibration + disputes)     |
                +-------------------------+------------------------+
                                          | per-trial scores + rationales
                                          v
       +-------------------+    +--------------------+    +---------------------+
       | Results Store     |--->| Stats Engine       |--->| Gates & Dashboards  |
       | (immutable runs,  |    | SE / clustered SE, |    | CI pass/fail, diffs,|
       |  run_id, lineage) |    | paired diffs, power|    | transcript viewer   |
       +-------------------+    +--------------------+    +----------+----------+
                                                                     |
       +-------------------+                                         |
       | Online Sampler    |-- production traces, feedback --------->+--> new tasks
       +-------------------+                                     (registry)
```

### Components and the design decisions to say out loud

**Task registry.** Tasks are data, not code. OpenAI Evals keeps its registry in Git-LFS and supports private evals built from a team's own data without exposing it publicly [29]. UK AISI's Inspect ships more than 200 pre-built evaluations. OpenAI's simple-evals stopped receiving updates for new models in July 2025 and now only hosts reference implementations for HealthBench, BrowseComp and SimpleQA, a reminder that a platform should not depend on any single external benchmark repository staying maintained [30]. It provides built-in components for prompt engineering, tool use, multi-turn dialog and model-graded scoring, and new scorers can be added through extension packages [28]. Every task should carry its provenance, because the roadmap in [1] recommends seeding suites with 20–50 tasks drawn from real failures, bug reports and support tickets, prioritized by user impact. Let PMs and domain experts add tasks through pull requests [1].

**Run orchestrator and sandboxes.** Trial isolation is not optional. Anthropic reports an internal eval in which Claude could see git history from earlier trials. That shared state gave it an unfair advantage, and shared state can also cause correlated failures [1]. SWE-bench moved to a fully containerized Docker harness for reproducibility, and its README warns that Docker Desktop users need about 120 GB of free virtual disk [25]. Plan the sandbox pool as a capacity problem: image prebuilds, disk, CPU per worker, and concurrency limits on model APIs.

**Caching and idempotency.** Cache keys must cover every input. The SWE-bench harness caches results by `run_id` and `instance_id` only. If you rerun the same instance under the same `run_id` with a different patch, the harness silently reuses the first result [25]. In an interview, propose a content-addressed cache key: a hash of the task version, the subject config, the grader version and the seed.

**Grader service.** Use all three grader families and say when each applies. Code-based graders are "fast, cheap, objective, reproducible" but brittle to valid variation. Model-based graders are flexible but non-deterministic, cost more, and need human calibration. Human graders are the gold standard but slow [1]. Version judge prompts like code. A judge prompt change is a grader change, and it invalidates comparisons with earlier runs.

**Stats engine.** Report a standard error with every score. Cluster standard errors when questions share a source. Compare models with paired differences, not independent two-sample tests [2][3] (see section 8).

**Gates and dashboards.** Gate regression suites on thresholds. Report capability suites as trends with confidence intervals. Always link aggregate numbers to a transcript viewer: "read transcripts" is how you tell agent failures from grader bugs [1].

**Online sampler.** Open-source observability stacks show what a platform should expose. Langfuse supports LLM-as-a-judge, code evaluators, user-feedback collection, manual labelling and datasets for pre-deployment testing [31]. Arize Phoenix offers versioned datasets, experiments that track prompt, model and retrieval changes, and response and retrieval evals [32]. Scorer libraries such as Braintrust's autoevals package LLM-as-a-judge templates (for example a Factuality prompt that returns a score and a rationale) alongside heuristic scorers such as Levenshtein distance [33].

### Failure modes the interviewer wants you to name

- **Grader bugs masquerading as model failures.** Opus 4.5 initially scored 42% on CORE-Bench. Rigid grading rejected "96.12" against a longer expected decimal, some specifications were ambiguous, and some tasks were not reproducible. After fixes and a less constrained scaffold, the score was about 95% [1].
- **Broken tasks.** A 0% pass rate across many trials usually means the task is broken, not that the agent is incapable [1]. Sierra's tau-bench maintainers shipped more than 75 task fixes in tau3-bench: incorrect expected actions, ambiguous instructions and impossible constraints. Their July 2026 grading update states that results produced before and after the fix are not comparable [23].
- **Non-determinism.** Microsoft Research's Eureka study found that very few large models are fully deterministic. Even at temperature zero, most showed visible variation in output and accuracy across repeated runs [14]. Design for multiple trials from day one.
- **Silent example-level regressions.** Eureka also found regression rates above 10% between versions within a model family in several cases, even when overall accuracy rose [14]. Store per-example results so that you can diff runs example by example, not just compare averages.
- **Implementation drift on standard benchmarks.** Anthropic reports that trivial formatting changes can move MMLU accuracy by about 5%, and that labs implement the benchmark inconsistently [7]. Pin prompt templates and few-shot settings as part of the task version.

---

## 3. Design B: "Design evals for a chatbot / general assistant"

### Problem

The outputs are open-ended and there is often no reference answer. Quality has many dimensions: helpfulness, tone, safety, faithfulness to policy and format. Preferences are subjective, and human labels are expensive.

### Architecture

```text
  prompt sources                     candidate A / candidate B
  (prod log sample, curated hard     (model, system prompt, temperature)
   prompts, policy edge cases)                 |
        |                                      v
        +-----------> generation (k samples per prompt, logged) ------+
                                                                      |
          +-----------------------------+-----------------------------+
          |                             |                             |
          v                             v                             v
  deterministic checks          pointwise LLM judges          pairwise LLM judge
  (format, length, PII regex,   (one judge per dimension:     (A vs B, both orders,
   refusal classifier)           tone, policy, accuracy)       style-controlled)
          |                             |                             |
          +---------------> per-dimension scores + rationales <-------+
                                        |
                       human calibration sample (agreement, disputes)
                                        |
                          release gate  +  online A/B / feedback
```

### Key techniques

**Pairwise vs pointwise judging.** MT-Bench supports single-answer grading, where the judge scores each turn out of 10 and the scores are averaged, and pairwise comparison against a baseline or across all pairs. It recommends single-answer grading as the default mode [15]. The MT-Bench authors released 3.3K human annotations of answers from 6 models to 80 questions. They report that humans and the GPT-4 judge agree more than 80% of the time, the same level at which humans agree with each other [15].

**Length and style bias.** AlpacaEval's evaluator leaderboard reports, for each judge, the probability that it prefers the longer output. The values range from about 0.65 to 0.79 for automatic judges, against 0.64 for humans [16]. AlpacaEval 2.0's **length-controlled** win rate raised its Spearman correlation with Chatbot Arena from 0.93 to 0.98 and made the metric much harder to game with length [16]. Arena-Hard-Auto applies "style control" that adjusts for token length and markdown elements, following Chatbot Arena's own method [17].

**Benchmark-quality metrics.** Arena-Hard-Auto argues that correlation with human rankings is not enough. A benchmark also needs **separability**: the ability to tell apart models of similar quality with high confidence, such as different checkpoints from the same training run. It proposes three metrics: separability with confidence, agreement with confidence, and pair-rank Brier score [17]. These questions apply directly to an internal eval as well.

**Rubric design.** Anthropic's documentation gives worked examples for this setting: a 1–5 LLM Likert score for customer-service tone over 100 inquiries, a binary LLM classification of "does the response contain PHI?" over 500 queries, and embedding cosine similarity for consistency across 50 groups of paraphrased FAQ questions. It recommends grading with a different model from the one that generated the output [6]. The agent-eval guidance adds that each dimension should get its own judge, and that judges should be allowed to answer "Unknown" [1].

### Cost and scale numbers

AlpacaEval publishes per-judge cost and speed. Human annotation costs about $300 per 1,000 examples and takes about 36,800 seconds per 1,000 examples, with 65.7% agreement with the human majority. Automatic judges cost between about $0.4 and $13.9 per 1,000 examples, with 65–69% human agreement [16]. Length-controlled AlpacaEval 2.0 runs for under $10 in under 3 minutes [16].

| Annotator (AlpacaEval table) | Human agreement | $/1000 ex. | s/1000 ex. | P(prefer longer) |
|------------------------------|-----------------|------------|------------|------------------|
| humans                       | 65.7            | 300.0      | 36,800     | 0.64             |
| alpaca_eval_gpt4             | 69.2            | 13.6       | 1,455      | 0.68             |
| alpaca_eval_gpt4_turbo_fn    | 68.1            | 5.5        | 864        | 0.65             |
| claude                       | 65.3            | 3.3        | 173        | 0.66             |
| alpaca_eval_llama3_70b_fn    | 67.5            | 0.4        | 209        | 0.79             |

Source: AlpacaEval repository evaluator table [16]. The judges are historical model versions, so read the table for the shape of the tradeoff, not for current prices.

The interview point: a calibrated judge can be roughly 20–750x cheaper than human annotation at comparable agreement with the human majority, but it brings its own biases, such as length preference. Measure those biases and control for them; do not assume them away.

### Online loop

For a consumer chatbot, offline suites must be backed by privacy-preserving production analysis. Anthropic's Clio uses Claude to extract facets from conversations, cluster them semantically, write privacy-scrubbed cluster descriptions, and arrange the clusters in a hierarchy. It enforces minimum-size thresholds so that rare, identifying topics never surface. Clio analysed 1 million conversations. Its flags correlated with Trust & Safety classifiers at r = 0.71, and it helped find both false negatives and false positives in safety classifiers [9]. In an interview, this is the answer to "how do you know your eval set reflects real traffic?" Cluster production traffic, compare the cluster distribution with your eval-set distribution, and write tasks for the clusters that are under-covered.

---

## 4. Design C: "Design evals for a RAG system"

### Problem

A RAG system can fail in the retriever (the right evidence is never fetched), in the generator (the evidence is ignored or contradicted), or in the question-answer fit (the answer is faithful but does not address the question). One end-to-end score hides which part broke.

### Architecture: evaluate the pipeline as decomposed stages

```text
 query -> [retriever] -> top-K chunks -> [reranker] -> top-k -> [generator] -> answer
               |                                  |                       |
               v                                  v                       v
      retrieval metrics                  ranking metrics          generation metrics
      recall@K (needs labeled            context precision        faithfulness (claims
      relevant chunks or a               (relevant chunks          supported by context)
      reference answer)                  ranked high)              answer relevancy
                                                                   correctness vs ref.
                \______________________ RAG triad ___________________/
                  context relevance | groundedness | answer relevance
```

### Key techniques and metric definitions

**Retrieval-only metrics.** Anthropic's Contextual Retrieval work is a clean template. The metric is 1 − recall@20, the share of relevant chunks not retrieved in the top 20, averaged across codebases, fiction, arXiv papers and science papers [8]. Reported results: the baseline failure rate of 5.7% fell to 3.7% with contextual embeddings (−35%), to 2.9% with contextual embeddings plus contextual BM25 (−49%), and to 1.9% with reranking on top (−67%). Reranking took the top 150 candidates down to 20 [8]. The one-time contextualization cost was about $1.02 per million document tokens with prompt caching [8]. This is the kind of quantified, component-level result an interviewer wants: isolate the retriever and report a recall metric whose cost you know.

**Claim-level generation metrics.** Ragas defines **faithfulness** as the number of claims in the response that the retrieved context supports, divided by the total number of claims in the response. **Context recall** is the share of claims in the reference answer that the retrieved context supports. **Context precision** is a rank-weighted mean of precision@k over the retrieved chunks. **Answer relevancy** generates (by default) 3 artificial questions from the response and averages their embedding cosine similarity to the user input [18]. TruLens packages context relevance, groundedness and answer relevance as the "RAG triad" [21].

**Statistically grounded judging (ARES).** ARES fine-tunes lightweight classifiers on synthetic queries to judge context relevance, answer faithfulness and answer relevance. It then uses **prediction-powered inference (PPI)** with a small human-labelled validation set to produce confidence intervals. It asks for at least 50 annotated examples, ideally several hundred [19]. PPI in general combines a small gold-labelled set with a large set of model predictions. The result is valid confidence intervals that are tighter than the gold set alone could give [20]. In an interview, PPI is the rigorous answer to "your judge is imperfect, so how can you trust its aggregate?"

### Failure modes

- **Reference-free metrics measure consistency, not truth.** A faithfulness score of 1.0 can still be wrong if the retrieved document is stale. Pair it with correctness against a reference on a labelled subset.
- **Over-reliance on generic metrics.** Section 11 below returns to this. The agent-eval guidance stresses that framework choice matters less than the quality of the tasks and graders [1].
- **Judge-tool divergence.** Different libraries implement "context relevance" differently. The TruLens README cites third-party benchmarks comparing tools on the same metric names [21]. Validate any off-the-shelf metric against your own human labels before you trust it.

---

## 5. Design D: "Design evals for a tool-using / customer-service agent"

### Problem

The agent holds a multi-turn conversation, calls tools that change state, and must follow a domain policy. Success depends on the final state of the world and on policy compliance, not on any single message. Runs are stochastic, and customers experience the agent's *consistency*, not its best case.

### Architecture (tau-bench pattern)

```text
  task spec: user goal + hidden user info + expected end state + policy doc
        |
        v
  +-------------+    natural-language turns   +----------------+  tool calls  +---------+
  | User        | <-------------------------> | Agent under    | -----------> | Domain  |
  | simulator   |                             | test (model +  | <----------- | DB/APIs |
  | (LLM)       |                             | scaffold)      |   results    | (mock)  |
  +-------------+                             +----------------+              +----+----+
                                                                                   |
     graders:  (1) end-state DB diff vs expected   (2) required outputs to user    |
               (3) policy/transcript rubric        (4) turn / token / latency caps |
                                                                                   v
                                     k independent trials per task -> pass^k
```

τ-bench describes itself as "dynamic conversations between a user (simulated by language models) and a language agent provided with domain-specific API tools and policy guidelines" [22]. τ2/τ3-bench generalizes this into domains (airline, retail, telecom, banking_knowledge), each with a policy, tools, tasks and, optionally, user tools. It adds a knowledge-retrieval domain with configurable RAG pipelines and a full-duplex voice modality [23].

### Key technique: pass^k (reliability) instead of pass@k

Anthropic's guidance defines **pass@k** as the probability of at least one success in k trials, which rises with k. **pass^k** is the probability that all k trials succeed, which falls with k. A 75% per-trial success rate gives pass^3 ≈ 42% [1]. Use pass^k for customer-facing agents, where every interaction has to work [1]. The original τ-bench leaderboard shows how large the gap is in practice:

| Agent (tool-calling) | Domain | pass^1 | pass^2 | pass^3 | pass^4 |
|----------------------|--------|--------|--------|--------|--------|
| claude-3-5-sonnet-20241022 | Airline | 0.460 | 0.326 | 0.263 | 0.225 |
| gpt-4o | Airline | 0.420 | 0.273 | 0.220 | 0.200 |
| claude-3-5-sonnet-20241022 | Retail | 0.692 | 0.576 | 0.509 | 0.462 |
| gpt-4o | Retail | 0.604 | 0.491 | 0.430 | 0.383 |

Source: τ-bench repository leaderboard [22] (outdated tasks; superseded by τ3-bench [23]).

At pass^4, the best airline agent of that generation completes the same task correctly on all four tries only about 22% of the time, even though its single-try rate is 46%. This is the strongest single argument for running multiple trials per task.

### Grading principles

- **Grade outcomes, not paths.** Rigid step-sequence checks are brittle and punish creative valid solutions [1]. Anthropic's research-system team graded the end state for agents that mutate persistent state, and split complex workflows into checkpoints where specific state changes should have happened [4].
- **Multidimensional success.** The refund-agent example in [1] checks that identity was verified, that the refund cap was respected, that a confirmation was sent, and that the conversation took at most 10 turns. That is state checks plus transcript constraints plus an LLM rubric for tone.
- **Valid unexpected solutions.** Opus 4.5 "failed" a τ2-bench flight task by finding a policy loophole that actually served the user better [1]. Graders need a dispute path, which means a human review queue.
- **Balance the problem set.** Test both when a behaviour should occur and when it should not. Anthropic's web-search work iterated to balance undertriggering against overtriggering [1].

### Agent-eval metrics beyond accuracy

Anthropic's tool-design work collected top-level accuracy, the runtime of each tool call and task, the total number of tool calls, total tokens and tool errors. It also used held-out test sets to guard against overfitting tool descriptions to the training tasks [5]. Redundant calls can point to pagination or token-limit problems. Frequent invalid-parameter errors can point to unclear tool descriptions [5].

### Research agents and cost scaling

For the multi-agent research system, Anthropic started with about 20 queries that reflected real usage. Early prompt changes moved success from about 30% to about 80%, and effects that large are visible with a handful of cases [4]. The LLM-judge rubric covered factual accuracy, citation accuracy, completeness, source quality and tool efficiency. It produced a 0.0–1.0 score plus a pass/fail grade. A single judge call with a single prompt was more consistent, and closer to human judgment, than several specialized judges [4]. Human testers caught what the judge missed: early agents favoured SEO content farms over authoritative sources [4]. On BrowseComp, token usage alone explained 80% of the variance in performance, and token usage, tool-call count and model choice together explained 95%. Agents use about 4x the tokens of chat, and multi-agent systems about 15x [4]. Raise the cost implication in the interview: agent eval runs cost roughly an order of magnitude more per task than chat evals.

### Behavioural / safety audits at scale

Anthropic's Petri automates behavioural auditing. An auditor agent turns natural-language seed instructions into multi-turn scenarios with simulated users and tools, and LLM judges score the transcripts on safety-relevant dimensions, surfacing the worst ones for human review. The pilot ran 111 seeds against 14 frontier models. The authors call the metrics reductive and recommend reading the transcripts [13]. It is a good pattern for "how would you evaluate policy adherence at scale?"

---

## 6. Design E: "Design evals for a coding assistant / coding agent"

### Problem

Code is the friendliest domain for evaluation, because execution provides ground truth. The hard parts are elsewhere: sandboxing untrusted code, contamination, the cost of long agentic runs, and the gap between "tests pass" and "a reviewer would merge this."

### Architecture

```text
  task sources                          subject
  - function-level problems (HumanEval) - completion model / chat model
  - time-windowed contest problems      - agent (model + bash/edit tools)
    (LiveCodeBench)                              |
  - real repo issues (SWE-bench)                 v
  - terminal tasks (Terminal-Bench)     +---------------------------+
        |                               | per-task container image  |
        +------------------------------>| repo @ pre-fix commit,    |
                                        | deps, no network          |
                                        +-------------+-------------+
                                                      | patch / final FS state
                                                      v
           graders: hidden unit tests (fail->pass, pass->pass) | static analysis
                    (lint, types, security) | LLM rubric on diff & transcript |
                    required tool-call patterns | cost: turns, tokens, wall time
```

### Task families and what each measures

- **Function-level generation (HumanEval).** Graded by executing tests. The harness exists "to run untrusted model-generated code", and its execution call ships commented out so that users must opt in to running it inside a sandbox [24]. It reports **pass@k** with an unbiased estimator, and it refuses to report pass@k when fewer than k samples exist per problem [24].
- **Contamination-controlled generation (LiveCodeBench).** It continuously collects new problems from LeetCode, AtCoder and CodeForces, and versions its releases by problem publication date. It grew from 400 problems (May 2023–Mar 2024) to 1,055 (May 2023–Apr 2025). It also measures self-repair, code execution and test-output prediction [26]. The design lesson: tag every task with a date and evaluate only on tasks published after the model's training cutoff.
- **Repository-level issue resolution (SWE-bench).** Given a codebase and an issue, the model writes a patch, which is graded by the repository's own tests. SWE-bench Verified is a subset of 500 problems that software engineers confirmed are solvable, built with OpenAI Preparedness [25]. Anthropic's minimal scaffold (a short prompt, a bash tool and a string-replace edit tool that requires absolute paths) scored 49% on Verified in January 2025. Many successful runs took hundreds of turns and more than 100k tokens [11]. Anthropic later reported that frontier scores on Verified rose from about 40% to over 80% within a year, and called the benchmark near saturation [1].
- **End-to-end terminal tasks (Terminal-Bench).** A dataset of tasks plus an execution harness that connects a model to a terminal sandbox. It was in beta with about 100 tasks, and version 2.0 runs on the Harbor framework [27]. The tasks range from building a Linux kernel to training an ML model [1].

### Grader stack for a coding agent

Anthropic's example for coding agents combines unit tests, an LLM rubric, static analysis (ruff, mypy, bandit), state checks and required tool-call patterns. It also tracks turns, tool calls, tokens and latency [1]. The rationale for each layer:

1. **Tests** settle functional correctness, but they can be stricter or looser than the issue requires. Anthropic notes that some SWE-bench failures came from environment setup problems or from install patches being applied twice, not from model errors [11].
2. **Static analysis** catches regressions in quality, type safety and security that tests do not cover.
3. **The LLM rubric on the diff** approximates "would a reviewer merge this?" Anthropic's agent guidance says automated tests give feedback, but human review is still needed to align the code with broader system requirements [12].
4. **Efficiency metrics** guard against regressions that only raise cost.

### Failure modes specific to coding

- **Agents believe they succeeded when they did not.** With hidden tests, models often believe a fix worked when it did not, or fix the problem at the wrong level of abstraction [11]. Never grade on the agent's own claim; grade the outcome [1].
- **One-shot evals miss long-horizon gains.** Qodo first found Opus 4.5 unimpressive because its one-shot evals missed gains on longer tasks, and it built an agentic eval framework in response [1].
- **Missing modalities.** In Anthropic's SWE-bench work, the agent could not view image files, which hurt it on Matplotlib tasks [11]. Check that the harness exposes the observations the task needs.

---

## 7. Metric choice: a decision table

| Output type | Primary metric | Grader | Notes |
|-------------|---------------|--------|-------|
| Classification / extraction | Accuracy, F1 (per class) | Code (exact match) | e.g., sentiment on 1,000 labelled tweets [6] |
| Short factual answer | Exact / normalized match; LLM equivalence check | Code + model | Verifiers must not reject correct answers over formatting [5] |
| Summarization w/ reference | ROUGE-L F1 (cheap), LLM rubric (better) | Code / model | ROUGE-L on 200 articles is a documented baseline [6] |
| Open-ended chat | Pairwise win rate (length-controlled); per-dimension pass/fail | Model + human calibration | Correct for length/style [16][17] |
| RAG | Recall@K; faithfulness; answer relevancy; correctness | Code + model | Decompose by stage [8][18] |
| Code (function) | pass@k | Execution | Unbiased estimator needs n ≥ k samples [24] |
| Code (repo / agent) | % resolved; plus lint, types, rubric, cost | Execution + static + model | Hidden tests [11][25] |
| Customer-facing agent | pass^k; end-state match; policy rubric | Code (state) + model | Reliability over best case [1][22] |
| Safety / policy | Violation rate, refusal rate on benign set | Model + human | Balance should/should-not triggers [1] |
| Ops | p50/p95 latency, cost per task, tokens, tool errors | Code | Track alongside quality [5][6] |

### Binary vs scaled scores

The sources split. Anthropic's documentation shows Likert (1–5) judges for tone and context use [6]. The research-system judge outputs both a 0–1 score and a pass/fail [4]. The agent-eval guidance recommends separate judges per dimension, an "Unknown" option, and partial credit for multi-part tasks [1]. A defensible interview position is to use binary or partial-credit verdicts per *atomic* criterion, aggregated into a score. Atomic binary verdicts are easier to calibrate against humans and easier to turn into regression gates than a single holistic Likert number.

### Judge validation

Treat the judge as a classifier and measure its agreement with a human-labelled sample before trusting its aggregate numbers. MT-Bench reports human–judge agreement at the human–human level (above 80%) [15]. AlpacaEval reports agreement, bias, variance and length preference for each judge [16]. ARES and PPI turn a small labelled set into a confidence interval for the whole population [19][20]. When two domain experts cannot agree on a task's verdict, the task is ambiguous and should be rewritten [1].

---

## 8. Sampling, trials and statistical power

This is where many candidates are weakest, and it is where the public guidance is most concrete. Anthropic's statistical guidance (a research post and the companion paper by Evan Miller) gives five recommendations [2][3]:

1. **Report standard errors.** Treat the eval score as an estimate of the mean over a hypothetical "question universe". Report SEM, and give a 95% CI as mean ± 1.96 × SEM.
2. **Cluster when questions are not independent.** If several questions share a passage or source, cluster the standard errors on the unit of randomization. On popular evals, clustered errors were more than 3x the naive ones.
3. **Reduce within-question variance.** With chain-of-thought, resample several answers per question and average them before computing the SEM. Inspect's `epochs` parameter does this. Without chain-of-thought, use next-token probabilities as the question score.
4. **Use paired differences.** Both models answer the same questions, so test the per-question differences. Frontier models' question-level score correlations on popular evals ranged from 0.3 to 0.7, and higher correlation shrinks the standard error of the difference.
5. **Do power analysis** to size the eval before building it, for example to detect "Model A beats Model B by 3 points".

### Worked sizing example (playbook arithmetic)

The numbers below are illustrative. They apply the standard normal-approximation power formula behind recommendation 5; they are not figures from the sources. For per-question binary scores near p = 0.7, the per-model variance is about 0.21. With a question-level correlation of ρ between the two models, the variance of the per-question difference is about 2 × 0.21 × (1 − ρ). The number of questions needed for 80% power at two-sided α = 0.05 is about ((1.96 + 0.84) × σ_d / δ)².

| Target difference δ | ρ = 0 (unpaired-equivalent) | ρ = 0.5 (paired) | ρ = 0.7 (paired) |
|---------------------|----------------------------|------------------|------------------|
| 3 points | ~3,660 questions | ~1,830 | ~1,100 |
| 5 points | ~1,320 | ~660 | ~400 |
| 10 points | ~330 | ~165 | ~100 |
| 50 points (30%→80%) | ~13 | ~7 | ~4 |

How to read the table in an interview:

- The last row is why "start with 20–50 tasks" [1] and "about 20 queries" [4] are sound advice early on. Large effects need few samples.
- The first row is why mature products need suites of thousands of tasks, or several trials per task, to resolve small regressions. Pairing roughly halves the required n at ρ = 0.5.
- Clustering can raise the SE by more than 3x [2]. That raises the required n by roughly the square of the inflation factor, so collect many independent sources rather than many questions per source.

### Trials per task

Run k trials when the subject is stochastic, which is almost always the case [14]. Averaging k trials per question reduces within-question variance [2]. The k trials also give pass@k and pass^k for free [1]. Remember that SWE-bench-style agentic trials can each take hundreds of turns [11]. Budget k by suite: low k for the large regression suite on every commit, higher k for release candidates.

### Production sampling

- **Stratify** by traffic cluster (Clio-style facets [9]), by user tier and by risk category. A uniform sample under-represents the rare, high-stakes slices.
- **Oversample signals.** Thumbs-down, `/bug` reports and escalations are a cheap way to find failures. Anthropic's postmortem lists continued user-feedback channels, plus privacy-preserving tooling for working with that feedback, among its fixes [10].
- **Label a small gold slice and use PPI** to turn judge scores on the full sample into valid intervals [20].

---

## 9. Cost and latency engineering

| Cost lever | Public anchor | Design response |
|------------|--------------|-----------------|
| Human labels | ~$300 / 1,000 pairwise annotations [16] | Reserve for calibration, disputes and gold sets |
| LLM judges | ~$0.4–$13.9 / 1,000 annotations across judges [16] | Use a cheap judge for screening and a strong judge on disagreements |
| Full benchmark-style chat eval | LC AlpacaEval < $10, < 3 min [16] | Fast suites gate every PR |
| Agentic trials | Hundreds of turns, > 100k tokens per success [11]; agents ~4x and multi-agent ~15x chat tokens [4] | Tiered schedule; small k on PRs; nightly full |
| Retrieval preprocessing | ~$1.02 / M document tokens for contextualization [8] | One-time; cache by corpus version |
| Sandbox infra | ~120 GB disk for SWE-bench Docker images [25] | Prebuild images; pool containers |

A **tiered eval schedule** is a strong talking point:

```text
  every commit / PR   : deterministic checks + small regression suite (minutes, cents)
  nightly             : full regression + capability suites, k>=3 trials, judges
  release candidate   : held-out sets, k>=5, human calibration sample, safety audit
  continuous (prod)   : sampled online judges, feedback mining, drift dashboards
```

Anthropic's September 2025 postmortem explains why the last tier is required. Three infrastructure bugs degraded output quality. One of them misrouted up to 16% of Sonnet 4 requests in the worst hour, and about 30% of Claude Code users had at least one message misrouted. The evaluations did not catch the degradation, partly because Claude often recovers well from isolated mistakes. The team also relied too heavily on noisy evaluations [10]. Its fixes include more sensitive evals that better separate working from broken implementations, and continuous quality evals on production systems [10]. Tell the interviewer that offline evals test the *model*, while production evals test the *serving system*, and that both are needed.

---

## 10. Tradeoffs interviewers probe (and defensible answers)

| Tension | Option A | Option B | Defensible default |
|---------|----------|----------|--------------------|
| Volume vs label quality | Many auto-graded items | Few expert-graded items | Volume first [6], but calibrate the grader on an expert sample [1] |
| Outcome vs process grading | End-state checks | Step-by-step path checks | Outcomes, with checkpoints for long workflows [1][4] |
| Single judge vs judge panel | One holistic call | One judge per dimension | Per-dimension for diagnosis [1]; Anthropic found one call more consistent for research outputs [4], so test both |
| Pointwise vs pairwise | Absolute scores, stable over time | Sensitive comparisons | Pointwise for regression gates; pairwise (length-controlled) for model selection [15][16] |
| Public benchmark vs custom | Comparable, cheap | Matches your distribution | Custom for decisions; public for sanity checks; watch saturation [1] |
| Realistic vs sandbox env | Production-like data | Simple mocks | Realistic enough to stress the tools [5]; isolated per trial [1] |
| pass@k vs pass^k | Capability ceiling | Reliability | pass^k for user-facing agents [1] |
| Fresh vs frozen tasks | Contamination-resistant | Comparable over time | Date-stamp tasks; keep a frozen regression core [26] |
| Eval speed vs fidelity | Small, fast suite | Large, slow suite | Tiered schedule (section 9) |

---

## 11. Red flags and common failure stories

These are short, sourced anecdotes you can cite in an interview to show operational experience.

1. **The "useless but perfect" model.** Some Claude models scored 0 on the BBQ bias metric. A control check showed they were "completely useless" because they were not answering the questions [7]. Always pair a safety metric with a helpfulness or answer-rate control.
2. **Benchmarks are software, and software has bugs.** BIG-bench needed heavy engineering, scaled poorly and contained bugs. HELM's prompt format made its results misleading for Claude [7]. CORE-Bench grading bugs cost more than 50 points [1].
3. **Specs that punish correct behaviour.** METR found time-horizon tasks that required exceeding a threshold the prompt stated only as a goal, which penalized models that followed the instructions [1].
4. **Graders can be gamed.** Design tasks so that passing requires actually solving the problem [1]. Length-controlled metrics exist because raw win rates reward verbosity [16].
5. **Saturation hides progress.** At 100% an eval stops measuring anything. SWE-bench Verified above 80% is the cited example [1]. Retire saturated capability evals into the regression suite.
6. **Evals alone give false confidence.** Anthropic recommends a "Swiss cheese" stack of automated evals, production monitoring, A/B tests, user feedback, transcript review and systematic human studies [1].
7. **Version churn breaks comparability.** τ-bench results from before and after v1.0.1 cannot be compared [23]. Eureka found that example-level regressions above 10% are common across model versions [14]. Store per-example results and pin versions of tasks and graders.

---

## 12. Practice prompts with answer skeletons

**P1. "Design an eval system for a customer-support agent that can issue refunds."**
Decision: block deploys that raise policy violations or lower resolution reliability. Criteria: resolution (state check), refund cap respected, identity verified, at most 10 turns, tone rubric [1]. Data: 20–50 tasks from historical tickets [1], balanced between should-refund and should-refuse cases, and growing from incidents. Harness: user simulator, mock DB, fresh state per trial [1][23]. Metrics: pass^k with k ≥ 3, cost per task, turns. Statistics: paired comparison against the production agent, with CIs [2]. Online: sampled transcripts scored by a calibrated judge, plus thumbs-down mining.

**P2. "Our RAG assistant's answers got worse after a re-index. How would you find out why, and prevent it?"**
Decompose by stage. Compare recall@K on a labelled query set before and after the re-index [8], then faithfulness and answer relevancy [18]. Version the corpus and embedding model as part of the eval config. Add a retrieval regression gate to the indexing pipeline. Use PPI on a small human-labelled slice so that judge-based scores come with valid intervals [19][20].

**P3. "Pick a model vendor for a coding assistant."**
Use a custom repo-level suite (SWE-bench-style containers with hidden tests [25]) on your own repositories. Add date-filtered contest problems to control contamination [26], plus static analysis and a diff rubric [1]. Run k trials and report paired differences with CIs [2]. Include cost: tokens and turns per resolved task [11]. Warn about saturated public numbers [1].

**P4. "Design the evaluation platform for a 50-team company."**
Use the section 2 architecture. Emphasize tasks-as-data with provenance [29], containerized isolation [1][25], content-addressed caching (the SWE-bench `run_id` pitfall [25]), versioned judges, a stats engine with clustered and paired SEs [2], a transcript viewer, tiered schedules, an online sampler [31][32], and an ownership model in which an eval infra team owns the harness and product teams own the tasks [1].

**P5. "How many examples do we need?"**
Ask what effect size matters. Show the power table in section 8. Note that early effects are large, so small sets suffice [1][4]. Mature products need hundreds to thousands of items, or repeated trials. Mention clustering inflation [2].

---

## Key takeaways for system designers

1. **Start from the decision, not the tool.** Every eval should state which decision it gates and give a specific, measurable, multidimensional target [6]. Framework choice matters less than task and grader quality [1].
2. **Separate capability from regression.** Capability suites start low and climb. Regression suites sit near 100% and gate deploys. Saturated capability tasks graduate into the regression suite [1].
3. **Grade outcomes in isolated environments.** Check the end state (DB rows, passing hidden tests), give partial credit for checkpoints, and never trust the agent's own claim of success [1][4][11].
4. **Treat graders as models that need validation.** Measure judge–human agreement [15][16], control for length and style bias [16][17], use per-dimension judges with an "Unknown" option [1], and use PPI to get valid intervals from small gold sets [19][20].
5. **Do the statistics.** Report SEs, cluster correlated items (clustered SEs can be more than 3x larger), use paired differences, and size suites with power analysis [2][3]. Run multiple trials because models are non-deterministic even at temperature zero [14].
6. **Choose the metric to match the user experience.** Use pass@k for "one success suffices" and pass^k for customer-facing reliability. Single-try success can overstate reliability by 2x or more [1][22].
7. **Engineer for cost.** Human labels cost hundreds of dollars per thousand. Judges cost cents to dollars [16]. Agentic trials cost hundreds of turns [11] and 4–15x chat tokens [4]. Tier the schedule by cost and by how much signal each run gives.
8. **Keep tasks fresh and versioned.** Date-stamp tasks against contamination [26], pin task and grader versions [23], and store per-example results to catch hidden regressions [14].
9. **Close the loop with production.** Offline evals miss serving-stack failures [10]. Add continuous production quality evals, privacy-preserving traffic clustering [9] and feedback mining, and turn every incident into a regression task [1].
10. **Read transcripts.** Aggregates point you to the problem, and transcripts explain it. The largest eval errors in the public record (CORE-Bench, BBQ, the τ-bench task fixes) were found by people reading outputs [1][7][23].

---

## References

1. Grace, M., Hadfield, J., Olivares, R., De Jonghe, J. "Demystifying evals for AI agents." Anthropic Engineering, 2026. https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents
2. Anthropic. "A statistical approach to model evaluations." Anthropic Research, 2024. https://www.anthropic.com/research/statistical-approach-to-model-evals
3. Miller, E. "Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations." arXiv:2411.00640, 2024. https://arxiv.org/abs/2411.00640
4. Hadfield, J., Zhang, B., Lien, K., Scholz, F., Fox, J., Ford, D. "How we built our multi-agent research system." Anthropic Engineering, 2025. https://www.anthropic.com/engineering/multi-agent-research-system
5. Aizawa, K. et al. "Writing effective tools for agents — with agents." Anthropic Engineering, 2025. https://www.anthropic.com/engineering/writing-tools-for-agents
6. Anthropic. "Define success criteria and build evaluations" (Test and evaluate documentation). Claude Developer Platform docs, accessed 2026. https://platform.claude.com/docs/en/test-and-evaluate/develop-tests
7. Anthropic. "Challenges in evaluating AI systems." Anthropic Research, 2023. https://www.anthropic.com/research/evaluating-ai-systems
8. Anthropic. "Introducing Contextual Retrieval." Anthropic, 2024. https://www.anthropic.com/news/contextual-retrieval
9. Anthropic. "Clio: Privacy-preserving insights into real-world AI use." Anthropic Research, 2024. https://www.anthropic.com/research/clio
10. Anthropic. "A postmortem of three recent issues." Anthropic Engineering, 2025. https://www.anthropic.com/engineering/a-postmortem-of-three-recent-issues
11. Schluntz, E. "Raising the bar on SWE-bench Verified with Claude 3.5 Sonnet." Anthropic Engineering, 2025. https://www.anthropic.com/engineering/swe-bench-sonnet
12. Schluntz, E., Zhang, B. "Building effective agents." Anthropic Engineering, 2024. https://www.anthropic.com/engineering/building-effective-agents
13. Anthropic. "Petri: An open-source auditing tool to accelerate AI safety research." Anthropic Research, 2025. https://www.anthropic.com/research/petri-open-source-auditing
14. Balachandran, V., Chen, J., Joshi, N., Nushi, B., Palangi, H., Salinas, E., Vineet, V., Woffinden-Luey, J., Yousefi, S. "Eureka: Evaluating and understanding progress in AI." Microsoft Research Blog, 2024. https://www.microsoft.com/en-us/research/blog/eureka-evaluating-and-understanding-progress-in-ai/
15. LMSYS (Zheng, L. et al.). "LLM Judge / MT-Bench" README, FastChat repository; paper "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena," arXiv:2306.05685, 2023. https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/README.md
16. Tatsu Lab (Dubois, Y. et al.). "AlpacaEval" repository README, including length-controlled win rates (arXiv:2404.04475, 2024). https://github.com/tatsu-lab/alpaca_eval
17. LMArena (Li, T. et al.). "Arena-Hard-Auto" repository README; paper arXiv:2406.11939, 2024. https://github.com/lmarena/arena-hard-auto
18. Ragas project. "Faithfulness / Context Precision / Context Recall / Answer Relevancy" metric documentation, Ragas repository, accessed 2026. https://github.com/vibrantlabsai/ragas/blob/main/docs/concepts/metrics/available_metrics/faithfulness.md
19. Saad-Falcon, J. et al. (Stanford FutureData). "ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems" repository README; paper arXiv:2311.09476, 2023. https://github.com/stanford-futuredata/ARES
20. Angelopoulos, A. N. et al. "ppi_py: Prediction-powered inference" repository README; paper arXiv:2301.09633, 2023. https://github.com/aangelopoulos/ppi_py
21. TruEra / Snowflake. "TruLens" repository README (RAG triad), accessed 2026. https://github.com/truera/trulens
22. Sierra Research (Yao, S. et al.). "τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains" repository README; paper arXiv:2406.12045, 2024. https://github.com/sierra-research/tau-bench
23. Sierra Research. "τ2/τ3-bench" repository README and release notes (paper arXiv:2506.07982), 2025–2026. https://github.com/sierra-research/tau2-bench
24. OpenAI (Chen, M. et al.). "HumanEval" repository README; paper "Evaluating Large Language Models Trained on Code," arXiv:2107.03374, 2021. https://github.com/openai/human-eval
25. Jimenez, C. E. et al. / SWE-bench team. "SWE-bench" repository README (Docker harness, Verified, Multimodal); paper arXiv:2310.06770, 2023–2026. https://github.com/SWE-bench/SWE-bench
26. LiveCodeBench team (Jain, N. et al.). "LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code" repository README, 2024–2025. https://github.com/LiveCodeBench/LiveCodeBench
27. Laude Institute. "Terminal-Bench" repository README, 2025. https://github.com/laude-institute/terminal-bench
28. UK AI Security Institute. "Inspect" repository README, accessed 2026. https://github.com/UKGovernmentBEIS/inspect_ai
29. OpenAI. "OpenAI Evals" repository README, accessed 2026. https://github.com/openai/evals
30. OpenAI. "simple-evals" repository README (deprecation notice, July 2025). https://github.com/openai/simple-evals
31. Langfuse. "Langfuse" repository README (evaluations, datasets), accessed 2026. https://github.com/langfuse/langfuse
32. Arize AI. "Phoenix" repository README (evaluation, datasets, experiments), accessed 2026. https://github.com/Arize-ai/phoenix
33. Braintrust. "autoevals" repository README (LLM-as-a-judge and heuristic scorers), accessed 2026. https://github.com/braintrustdata/autoevals
