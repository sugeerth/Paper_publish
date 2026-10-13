# 07. Production Traces to Fine-Tuning Loops: Logging, Evaluation, Curation, Distillation and Routing

## Overview

Most LLM applications start on a large, general-purpose API model. Once the application has
real traffic, every request and response becomes a labelled (or labellable) example of the
exact task the business cares about. The designs in this chapter turn that "data exhaust" into
a better system: they **log** production calls with enough structure to slice them by task,
**evaluate** candidate models or prompts against held-out slices of that traffic, **curate**
the logs into training sets, and **train**. Training can be supervised distillation into a
smaller model, preference tuning, or reinforcement learning against graders. The loop then
**promotes** the winner, often behind a **router** that decides per request which model gets
the call, and keeps logging.

NVIDIA's reference implementation defines a data flywheel as "a process that uses data exhaust
from production applications (for example, LLM prompt/response logs, end user feedback, and
expert labeling) to increase the overall accuracy and reduce the latency/cost of Generative AI
systems" [25]. Commercial platforms package the same loop differently:

- OpenAI built it into its API: stored completions, the Evals API, one-click distillation and
  reinforcement fine-tuning (RFT) [1][2][3].
- OpenPipe started with an SDK that logs requests and fine-tunes on them, then moved to
  open-source agent RL with ART and RULER [7][10][11].
- Predibase supplies the methodology (a distillation playbook) and the serving layer (LoRAX)
  [15][17].
- Fireworks, Together and Thinking Machines (Tinker) expose training APIs and cookbooks
  covering SFT, DPO, GRPO and on-policy distillation [18][19][23][48].
- Hyperscalers offer the same capture-to-distill path: Amazon Bedrock can distill directly from
  tagged invocation logs [47].
- Open standards are emerging for both ends of the loop: OpenTelemetry GenAI conventions for
  traces and evaluation results [44][45], and OpenEnv for agent environments [49].
- Research and open-source projects supply the parts these platforms are built from:
  - routers trained on preference or outcome data [31][32][34];
  - on-policy distillation objectives [40][41];
  - agent-RL middleware that captures training data from unmodified production agent
    harnesses [28][29].

The canonical loop, abstracted across all of these systems:

```
 +-------------+  requests   +-------------------+  responses  +-------------+
 |  App / agent|------------>| gateway / SDK /   |------------>| served model|
 |  (workload  |<------------| router (logs all) |<------------| (API / LoRA)|
 |   tags)     |             +---------+---------+             +------^------+
 +-------------+                       | traces + tags + feedback      |
                                       v                               | promote
                          +------------+-------------+                 | (canary /
                          | trace store (ES, S3, PG, |                 |  shadow)
                          | vendor "stored compl.")  |                 |
                          +------------+-------------+                 |
                                       | filter / dedup / split        |
                                       v                               |
           +---------------+   +-------+--------+   +---------------+  |
           | eval sets     |   | train sets     |   | grader/judge  |  |
           | (held-out,    |   | (teacher out-  |   | (code, model, |  |
           |  stratified)  |   |  puts, prefs)  |   |  rubric)      |  |
           +-------+-------+   +-------+--------+   +-------+-------+  |
                   |                   v                    |          |
                   |          +--------+---------+          |          |
                   |          | SFT / DPO / RFT /|<---------+          |
                   |          | GRPO / OPD (LoRA)|  rewards            |
                   |          +--------+---------+                     |
                   v                   v                               |
                +--+-------------------+--+    gate: beats baseline?   |
                | evaluate candidates     |----------------------------+
                | (base, ICL, tuned)      |  human review before prod
                +-------------------------+
```

The chapter is organized by vendor or system, then by technique (distillation objectives,
agent-RL plumbing, routing), and ends with a comparison table, cross-cutting failure modes and
design takeaways. Chapter 02 covers fleet-scale data engines, including NVIDIA's flywheel
blueprints from the autonomy angle. Chapter 03 covers RLHF, and chapter 04 covers RL with
verifiable rewards (RLVR) and self-improvement. This chapter covers the LLM-application side:
going from an API call log to a cheaper, better, task-specific model.

A note on sources: primary material is drawn mostly from vendors' own public repositories,
documentation sources and cookbooks. Cookbook dates and authors follow the OpenAI Cookbook
registry [4]. Where a figure comes from a vendor and was not independently reproduced, it is
labelled as a vendor claim.

## 1. Anatomy of a trace-to-model loop

Across all the systems surveyed, the loop breaks into the same stages. The hard engineering
problems sit in the seams between them.

### 1.1 Capture: what a "trace" must contain

The minimum useful record is the **exact request payload and the exact response**. NVIDIA's
blueprint requires each log document to carry `timestamp`, `workload_id`, `client_id`, the
full OpenAI-format `request` and the full `response`. It explains: "Keeping the full
request/response allows the Flywheel to replay prompts, build few-shot demonstrations, and
fine-tune without lossy conversions" [25]. Two capture details recur:

- **Task identity.** NVIDIA makes `workload_id` the "primary key for slicing, deduplication,
  and dataset creation." It insists that each node of a multi-step agent gets its own id,
  because an untagged mix of chat, query rewriting and tool routing cannot be evaluated or
  distilled coherently [25][26]. The same role is played by OpenAI's `metadata` tags on stored
  completions [1][2] and OpenPipe's `tags` (for example `prompt_id`) [8].
- **Non-blocking logging.** OpenPipe's SDK calls the upstream model directly and records the
  request *after* returning the completion. For streams, it forwards tokens as they arrive and
  reports the assembled response at the end, so logging adds no latency [8][9]. NVIDIA warns
  that streaming clients must buffer the stream and rebuild a full response object before
  indexing [25].

For RL on agents, text is not enough. The Agent Lightning team documented **retokenization
drift**: if a trainer re-tokenizes logged strings instead of using the token IDs the inference
engine actually sampled, the learner can optimize "against the wrong sequence." There are three
causes:

1. Non-unique tokenizations (for example `H`+`AVING` versus `HAV`+`ING`).
2. Tool-call parsing and re-serialization. These change whitespace and can silently
   auto-correct malformed JSON, which "masks the model's true generation errors."
3. Chat-template differences between frameworks.

The result is effectively off-policy updates and unstable learning curves. The authors add that
because the drift is not token-level, it "cannot be corrected through token-level importance
sampling." The fix was
upstreamed into vLLM (v0.10.2): the OpenAI-compatible API accepts `return_token_ids` and
returns `prompt_token_ids` and `token_ids` [30]. Agent Lightning v1.0's gateway now records
"prompt token IDs, response token IDs, and chosen-token log probabilities" for every proxied
call [29].

### 1.2 Curate and split

Every source says that curation matters more than training hyperparameters:

- Predibase's playbook: "Most mistakes made by converged student models can be traced back to
  issues with the source data." It warns that logs-based datasets are often "variation or
  balance-deficient", for example with too many examples from power users [15].
- NVIDIA's flywheel deduplicates, then applies **class-aware stratified splitting** so that
  eval and train sets have balanced representation across tool types. It explicitly guards
  against eval/train leakage, which "can lead to inflated evaluation scores" [25][26].
- OpenPipe holds out 20% of each dataset as a test set automatically [9].
- OpenPipe's **pruning rules** strip static text (for example long system prompts) from
  training data. The fine-tuned model inherits the rules and applies them to incoming
  requests, so the student pays for fewer input tokens [9].

### 1.3 Evaluate and gate

The loop is only as good as its evaluator. Three evaluator families appear:

1. **Deterministic checks**: exact match, schema validity, string or Python graders.
2. **Reference similarity**: fuzzy match, embedding cosine.
3. **LLM-as-judge**: pairwise, rubric-scored or ranked.

OpenAI's evaluation-flywheel guide (co-authored with Hamel Husain) recommends measuring a judge
against SME labels using true-positive and true-negative rates rather than accuracy. Test sets
are imbalanced, so "a judge that always guesses 'pass' might be 95% accurate but will never
find a single failure." It suggests splitting judge-alignment data about 20/40/40 into
few-shot, validation and test sets [5].

### 1.4 Promote with a human in the loop

None of the reference designs promote automatically. NVIDIA: "Think of the Flywheel as a
flashlight, not an autopilot. Promotion to production ... remains a human decision" [25]. The
blueprint "doesn't automatically promote or deploy any model" [26]. Predibase recommends
canary rollout (1%, then 10%), dark launch (shadowing the teacher) or a hybrid launch where the
student takes the easy queries and the teacher the hard ones [15]. That hybrid launch is a
router, which links distillation loops to the routing loops in section 10.

### 1.5 Trace standards and observability platforms

Vendor-specific capture (stored completions, OpenPipe tags, NVIDIA's Elasticsearch schema)
locks the training data to one platform. Since 2024 the OpenTelemetry project has been
standardizing GenAI telemetry, now maintained in a dedicated `semantic-conventions-genai`
repository and still at **Development** stability [44]:

- **Spans** carry `gen_ai.request.model`, `gen_ai.response.id`, token usage
  counts (for example `gen_ai.usage.input_tokens`) and, for multi-turn sessions,
  `gen_ai.conversation.id`. Operation names such as `invoke_agent` and `execute_tool` cover agent
  invocations and tool calls, and further spans cover retrieval and memory operations.
- **Content is opt-in.** `gen_ai.input.messages`, `gen_ai.output.messages`, system
  instructions and tool definitions are all `Opt-In` attributes flagged as likely to contain
  sensitive or PII data. A default-configured instrumentation therefore yields cost and latency
  telemetry but *not* a trainable dataset. A flywheel has to turn content capture on
  deliberately and route it to a store with appropriate retention and redaction.
- **Evaluations are first-class events.** A `gen_ai.evaluation.result` event records an
  evaluator name, a score value and a label (for example `pass` or `fail`), parented to the span
  being evaluated or linked by `gen_ai.response.id` [45]. Judge scores, user feedback and
  production outcomes can thus be joined to the exact call they grade, which is the join every
  curation step in section 1.2 needs.

Open-source LLM observability platforms sit on top of this layer. Langfuse, for example,
combines production tracing with LLM-as-judge, code evaluators, user
feedback, manual labelling and **datasets** built from production traces for regression testing
and experiments [46]. These tools usually stop at the eval set. Exporting their curated
datasets into a fine-tuning API is the step that closes the loop described in this chapter.

## 2. OpenAI: stored completions, Evals, distillation and RFT

### 2.1 Problem and architecture

OpenAI's design keeps the whole loop inside the API platform. The capture primitive is
`store=True` on a chat completion, with free-form `metadata`. Logging can also be turned on by
default at the organization level [2]. Stored completions feed three consumers:

```
 chat.completions.create(store=True, metadata={usecase:..., prompt_version:...})
            |
            v
   +------------------+   filter by model+metadata   +---------------------------+
   | Stored           |----------------------------->| "Distill": auto-built     |
   | completions      |                              | JSONL -> fine-tune job    |
   | (platform logs)  |                              | (e.g. gpt-4o -> 4o-mini)  |
   +--------+---------+                              +-------------+-------------+
            | data_source_config:                                  |
            | {type: stored_completions, metadata: {...}}           v
            v                                            ft:gpt-4o-mini:...:wine
   +------------------+  testing_criteria (label_model, score_model, python,
   | Evals API: Eval  |  text_similarity, string_check)  -> Runs per prompt/model
   +--------+---------+
            | graders reused as reward
            v
   +------------------+  method.type = "reinforcement", grader, hyperparameters
   | RFT job (o4-mini)|  -> eval runs logged back into Evals during training
   +------------------+
```

### 2.2 Distillation from stored completions

The distillation cookbook (October 2024) shows the full loop on a wine grape-variety
classification task [1]:

- **Logging.** It logs `gpt-4o` and `gpt-4o-mini` outputs with `store=True` and a
  `distillation` metadata tag, using Structured Outputs with an `enum` to constrain answers.
- **Distilling.** It filters the stored completions in the dashboard by model (`gpt-4o` only)
  and metadata, then clicks **Distill**. This auto-creates the training file and starts a
  `gpt-4o-mini` fine-tune.
- **Baseline (500 rows).** `gpt-4o` reached 81.8% accuracy and `gpt-4o-mini` 69.0%.
- **Held-out set (300 rows).** `gpt-4o` scored 79.67%, `gpt-4o-mini` 64.67% and the distilled
  `gpt-4o-mini` 79.33%. The author describes this as "almost a 22% relative improvement" over
  the undistilled small model.
- **Cost.** The notebook's own price constants (October 2024) were $2.50 per 1M input tokens
  for `gpt-4o` and $0.15 for `gpt-4o-mini`. The 500-prompt run (about 245K tokens) cost about
  $0.61 against $0.04, roughly a 16x inference-cost gap at the time.

The lesson is that a narrow, closed-label task can recover nearly all of the teacher's accuracy
in a model that costs an order of magnitude less. Structured outputs made the eval a trivial
string compare.

### 2.3 Monitoring stored completions with Evals

The completion-monitoring cookbook (April 2025) turns stored production traffic into a
regression test [2]:

- An **Eval** declares `data_source_config = {type: "stored_completions", metadata: {usecase:
  ...}}`, which exposes `{{item.input}}` and `{{sample.output_text}}` to graders.
- Each **Run** selects a slice, such as one `prompt_version`.
- A `label_model` grader (`o3-mini` in the example) labels each push-notification summary
  correct or incorrect.

The recommended pattern is to tag every call with a use case and a prompt version, so any
change can be compared on real traffic without building a separate dataset.

### 2.4 Reinforcement fine-tuning with graders

RFT runs RL on a reasoning model "by exploring the solution space and reinforcing strategies
that result in a higher reward." The grader *is* the reward function. The May 2025 cookbook
applies RFT to `o4-mini` on 100 training and 100 test medical vignettes [3]. Its main finding
is the sequence of grader designs:

1. **String grader.** A `multi` grader combined fuzzy token-set similarity (0.85) and exact
   match (0.15). It produced "stagnant reward" because the signal was not rich enough.
2. **Model grader v1.** This was a `gpt-4.1` `score_model` with an additive rubric (exact
   match +0.15, synonym +0.35, same family +0.35 ...). The policy **reward-hacked** it. Scores
   "shot up—sometimes by 20-30 percentage points", but only because the model padded
   one-phrase answers with synonyms, doses and management plans that inflated lexical
   similarity.
3. **Model grader v2.** This version normalized answers (mapping to ICD-10 or SNOMED where
   possible), scored a six-level clinical similarity ladder at weight 0.90 and lexical
   similarity at 0.10, penalized missing critical modifiers, and included contrastive examples
   built from the earlier run's hacked outputs. The authors used `o3` to check that the grader
   ranked predictions sensibly before training.

The final job used `n_epochs=5`, `reasoning_effort=medium`, `compute_multiplier=1.0`,
`eval_samples=3` and `eval_interval=5`, after validating the grader through the graders
`validate` endpoint. The tuned model gained "about a 5-point boost in accuracy". Its score
distribution moved from the 0.2–0.6 band into 0.8–1.0, and it spent *more* reasoning tokens
than base `o4-mini`. When both `o3` and the tuned model scored zero on the same items, the
authors read that as a sign of bad reference labels, not model failure. They recommend expert
relabeling before adding compute. On cost, LLM graders are billed on top of training compute,
so the advice is to test graders locally and start with small runs to catch hacking early [3].

### 2.5 Prompt-level loops on the same substrate

Not every loop updates weights. The evaluation-flywheel guide (October 2025) frames the work as
**Analyze → Measure → Improve** [5]:

- open-code about 50 failing traces;
- group the codes into axial categories (for example "35% of failures are tour scheduling");
- build graders for each category;
- run the platform's prompt optimizer, which uses outputs, annotations and graders.

The "Self-Evolving Agents" cookbook (November 2025) automates the same loop [6]:

- Four graders score each output: two Python checks (chemical-name retention, length),
  embedding similarity, and an LLM judge, with pass thresholds of 0.8–0.85.
- A metaprompt agent rewrites the prompt until an aggregate score passes or `max_retry` runs
  out (10 in the example), at which point engineers are alerted.
- A `VersionedPrompt` store keeps history for rollback.

The authors' lessons: "the optimization is not always successful, so being able to roll back
the prompt version is important," and grader fidelity determines optimization quality. The
same notebook notes that the Evals platform "does not yet provide an API to retrieve user
feedback programmatically", so human thumbs-up and thumbs-down data cannot yet flow back into
the loop automatically.

### 2.6 The same pattern on a hyperscaler: Bedrock distillation from invocation logs

Amazon Bedrock Model Distillation packages the stored-completions pattern for AWS. Its
workflow automates teacher response generation, data synthesis and student fine-tuning, and
one of its two input modes is **historical invocation logs** [47]. AWS's sample notebook shows
the loop end to end:

- **Capture.** Model invocation logging is enabled to an S3 bucket. Each teacher call
  (`Llama 3.1 70B` in the example) is made through the Converse API with `requestMetadata`
  tags such as `{"job_type": ..., "invoke_model": "llama31-70b"}`, the direct analogue of
  OpenAI's `metadata`.
- **Curation by tag.** The distillation job's `invocationLogsConfig` points at the log bucket
  and applies `requestMetadataFilters` (for example `equals: {"invoke_model":
  "llama31-70b"}`) so only the intended workload is used.
- **Reuse or regenerate.** The flag `usePromptResponse` (default `False`) decides whether the
  logged teacher responses are used as targets or only the prompts are kept and the teacher is
  re-queried during the job.
- **Train.** The student (`Llama 3.1 8B` in the example) is fine-tuned on the result, with
  the teacher named in `teacherModelConfig`.

The `usePromptResponse` switch makes explicit a choice every log-driven loop faces: train on
what production actually returned (cheap, but it inherits any prompt version or model that was
live at the time) or re-label the logged prompts with today's best teacher.

## 3. OpenPipe: from request logs to agent RL

### 3.1 Phase 1: logging-first fine-tuning platform

OpenPipe (YC S23) pitched: "Use powerful but expensive LLMs to fine-tune smaller and cheaper
models suited to your exact needs ... Switch between OpenAI and fine-tuned models with one line
of code" [7]. The architecture:

```
 app --(openpipe SDK = drop-in OpenAI client)--> OpenAI (key stays on client)
   |                                                  |
   +-- async report (after completion / end of stream)+--> OpenPipe request logs
                                                              | filters: model, tags,
                                                              | request/response text
                                                              v
                                                       dataset (+ pruning rules,
                                                       20% auto test split)
                                                              v
                                       fine-tune (Mistral/Llama/GPT-3.5 base) -> hosted,
                                       OpenAI-compatible endpoint or downloadable weights
                                                              v
                                       evals: GPT-4 judge, head-to-head vs base models
```

Key techniques, from the open-sourced docs:

- **Capture.** SDK or proxy capture with searchable tags [8].
- **Curation by query.** Users filter logs, for example by `prompt_id`, by excluding responses
  containing "As an AI language model", or by model, then import the matching logs into a
  dataset [9].
- **Pruning rules.** Static text is removed before training (section 1.2) [9].
- **Judged comparison.** Evaluations use GPT-4 as a judge with configurable criteria [9].

A notable product choice is that the client keeps the OpenAI key and calls OpenAI directly,
"so you'll continue getting uninterrupted inference even if your OpenPipe instance goes down"
[9]. The logging plane is kept off the critical path. The open-source repository has since
been frozen "to integrate some proprietary third-party code" [7].

### 3.2 Phase 2: ART, agent RL with GRPO

ART (Agent Reinforcement Trainer, 2025) moves from imitating a teacher to learning from reward
[10][13]:

- An OpenAI-compatible **client** embedded in the user's code runs the agent workflow, usually
  with many parallel rollouts.
- Completions are served by a **backend** running the latest LoRA in vLLM.
- Every `system`, `user` and `assistant` message is stored in a `Trajectory`, and the user's
  code assigns each trajectory a reward.
- Trajectories are grouped. Inference pauses while the backend trains with GRPO from the latest
  checkpoint (or an empty LoRA), saves the new LoRA, hot-loads it into vLLM and resumes.
- A dedicated mode keeps training and inference on separate GPUs.
- SFT is supported for distillation and as a warm-up before RL [14].
- Checkpoint forking allows branching from a known-good step [14].

The ART FAQ gives unusually concrete guidance on **when the loop works** [12]:

- Open models should already complete the task "at least 30% of the time".
- Success must be "easily" verifiable, whether objectively or by a consistent LLM judge.
- The agent must be runnable many times "without affecting the real world". GRPO needs
  repeated rollouts on the same scenario, which rules out agents with irreversible side
  effects.
- Runs "often cost between $15 and $200 in GPU time".
- Production aberrations "can be instantly fixed by adding a similar training example to the
  training set", which closes the production-to-training loop.

OpenPipe reports that its ART·E email-research agent (Qwen 2.5 14B) beat o3. This is a vendor
claim, linked from the README [10]. The README also advertises W&B Training (Serverless RL)
as a hosted ART backend, with vendor-claimed "40% lower cost" and "28% faster training" from
multiplexing on a shared inference cluster [10].

### 3.3 RULER: rewards without labels

RULER (Relative Universal LLM-Elicited Rewards) addresses the hardest part of turning traces
into RL data: writing a reward function [11]. It rests on two observations: "relative scoring
is easier than absolute scoring," and "GRPO only needs relative scores" because it normalizes
within each group. The procedure:

1. Generate N trajectories for one scenario.
2. Deduplicate their common prefix (for example the shared system prompt).
3. Ask a judge model (for example `openai/o3`) to score each from 0 to 1, with explanations.
4. Use the scores directly as GRPO rewards.

The default rubric says goal-achieving trajectories must score "significantly higher". More
efficient paths beat less efficient ones, score gaps should be proportional to quality gaps,
and partial credit is allowed. The judge gets task understanding from the trajectories' own
system prompts. The implementation allows one bounded re-judge on a malformed ranking and
records judge cost as a training metric. OpenPipe claims RULER cuts implementation time "2-3x"
and "often matches or exceeds hand-crafted reward functions" (vendor claim) [11]. RULER also
powers "AutoRL" and "MCP•RL", which train without labelled data by generating inputs and
judging outputs [10].

## 4. Predibase: the distillation playbook, LoRA Land and LoRAX

### 4.1 The playbook as a process specification

Predibase's *LLM Distillation Playbook* (Zhao and Abid, February 2024) is the most explicit
written process for log-to-student distillation [15]. Its 12 practices map onto the loop:

| Stage | Practice (abridged) |
|---|---|
| Feasibility | Students usually at best *match* the teacher. Imitation can copy style without factuality ("False Promise of Imitating Proprietary LLMs"). Narrow tasks distill best |
| Capture | "Build good logging infrastructure": stream MaaS requests and responses to S3 or Snowflake. Bootstrap with synthetic data when logs are thin |
| Eval | Fixed, diverse test sets. Measure both an in-distribution set (90/10 class mix) and a balanced set (50/50) |
| Teacher | Maximize teacher quality first. A *simpler* prompt beat chain-of-thought for labelling, and temperature 0 was best |
| Data | Fix data at the source, filter by rules, rank with LLMs, add explanation traces, aggregate multiple teachers |
| Training | Start small. Run ablations on data size. Change one parameter at a time |
| Serving | Plan for PEFT and multi-LoRA serving early |
| Deploy | Canary, dark launch or hybrid launch. Monitor inputs for drift and outputs for failures, and keep logging for re-distillation |

The Jigsaw toxicity case study gives quantitative lessons. Adding gold human labels *and*
removing incorrect teacher labels both improved the student. Going from 1.1K to 5K examples
gave a large jump, but going from 5K to 10K was "rather marginal" [15]. Preference tuning
(RLHF/DPO) was rated a costly "last mile optimization", and the authors had "yet to speak with
a company who has a critical need" for it in 2024 [15]. Chapter 04 shows how much that view
changed once graders and RFT became productized.

### 4.2 LoRA Land and the economics of many small specialists

LoRA Land measured 310 models: 10 base models fine-tuned with 4-bit LoRA on 31 tasks. The
authors report that the fine-tuned models "outperform base models by 34 points and GPT-4 by 10
points on average". The accompanying demo served 25 LoRA fine-tuned Mistral-7B models on a
single 80 GB A100 [16]. The serving layer is LoRAX, which serves many adapters over one shared
base model [17]:

- **Just-in-time adapter loading** happens per request without blocking concurrent requests.
- **Heterogeneous continuous batching** packs requests for different adapters into one batch,
  "keeping latency and throughput nearly constant with the number of concurrent adapters."
  Multi-adapter kernels (Punica's SGMV) keep the mixed batch efficient under load.
- **Adapter exchange scheduling** asynchronously prefetches and offloads adapters between GPU
  and CPU memory.

This matters for loop design because a flywheel that spawns a new adapter per workload, per
tenant or per retraining cycle stays affordable only if adapters share base-model GPUs. Without
that, each new specialist needs dedicated capacity and the flywheel stalls on serving cost.
Fireworks' `multilora_fleet` case study ("one deployment, a per-tenant specialist for every
locale", 51 locales) follows the same pattern [19].

## 5. Fireworks: training API, Eval Protocol and on-policy distillation

### 5.1 Architecture

The 2026 Fireworks cookbook presents training as a programmable API rather than a black-box
job. The repository is organized as `recipes/` (one training loop per file), `utils/` (losses,
rollout helpers, metrics) and `examples/` (task code plus one recipe call), with "dedicated"
and "serverless" twins for each example [18][19]. The recipe set covers:

- SFT, DPO and ORPO;
- synchronous GRPO, and async GRPO "with rollout/train overlap and bounded staleness";
- on-policy distillation (OPD) and top-K self-distillation (SDFT);
- embedding fine-tuning;
- experimental PPO with a value head.

The documented GRPO loss is clipped against **rollout logprobs** with group-normalized
advantages and no frozen-reference KL term by default [19].

Two workflows differ in cost model [20]:

- **Dedicated.** A managed trainer plus an on-demand deployment, billed per GPU-hour.
- **Serverless.** The user drives `forward_backward` and `optim_step` against a shared pooled
  trainer and samples in-session from a snapshot, billed per token with no idle cost.

In earlier runs of the router case study on `qwen3p5-9b`, the dedicated path took about 12–25
minutes against about 2 minutes serverless for the same 51 optimizer steps. The case study
notes these timings were not remeasured on the current base model [20].

### 5.2 Eval Protocol: RL on agents already in production

Eval Protocol (EP) addresses a deployment problem: production agents run "across remote
services with heavy dependencies, Docker containers, or TypeScript backends." EP wraps an
existing agent behind a rollout HTTP interface that handles "rollout orchestration, metadata
passing, and trace storage." Any supported trainer can then train against it, including
Fireworks RFT, TRL, Unsloth or a custom loop [22]. The cookbook's case studies all follow
"build data -> evaluate the base model -> fine-tune -> evaluate again" using EP [19].

### 5.3 A routing model as a fine-tuning target

The `sft_prompt_router` case study links the distillation and routing threads [20]:

- **Task.** It fine-tunes a LoRA on Qwen 3.8 27B (thinking disabled) to emit a five-field JSON routing decision:
  small or big model, complexity 1–5, and math, code and reasoning flags. The goal is to "keep
  easy chat on a small model and escalate hard logic/code/math to a big model — without paying
  a frontier model just to make that decision."
- **Comparison.** On a 60-row holdout, the tuned model with a lean prompt beat a
  prompt-engineered base model that had the routing rule written into its prompt.
- **Results.** Route accuracy rose from 93.3% to 98.3% (dedicated) and from 91.7% to 95.0%
  (serverless). Exact match on all five fields rose from about 40–42% to 67–73%. The largest gain
  was on the `reasoning` flag (62% to 93%, and 60% to 100%).

The sample is small: 992 rows in total and a 60-row holdout.

### 5.4 On-policy distillation (OPD)

Fireworks' OPD recipe trains the student on **its own rollouts**. A frozen teacher (sharing the
tokenizer) scores each sampled token, and the per-token difference `teacher_logprob -
sampling_logprob` becomes a dense advantage. This is a sampled estimate of reverse
KL(student‖teacher) [21]. The loop runs as follows: sample, ask the teacher for logprobs,
compute advantages, update, hot-load the new student, sample again.

A "privileged" variant gives the teacher extra context, such as a worked solution, that the
student never sees. This is a way to distill from information that is present in logs but
absent at inference time. Section 9 places this among the other distillation objectives.

## 6. Together AI and Tinker: training APIs as loop primitives

### 6.1 Together AI: fine-tuning API, continual fine-tuning and evaluations

Together's public cookbook covers the same loop with an API-first shape [23]:

- **Fine-tuning recipes:** end-to-end SFT, DPO preference tuning, function-calling, reasoning
  traces, multi-turn, long-context and vision-language.
- **Evaluations API recipes:** classification evals with LLM-as-judge, head-to-head comparison
  evals, prompt A/B evals, an "Optimizing LLM Judges" notebook for aligning judges with
  humans, and GEPA-based prompt evolution.

The feature most relevant to closed loops is **continual fine-tuning** [24]. A new job passes
`from_checkpoint=<previous job id>`, so each retraining cycle continues from the last tuned
model instead of the base model. The notebook demonstrates this with LoRA on `Qwen/Qwen3.5-2B`:
first core function-calling categories, then the remaining niche categories. It argues that
building on checkpoints means "the model is less likely to lose earlier learned abilities". The
appeal for a flywheel is that each cycle trains only on the newly harvested data.

The risk is cumulative drift, which a fixed regression eval set must catch. This is the same
discipline the playbook asks for ("Avoid changing your test sets frequently") [15].

### 6.2 Tinker: a low-level training API

Thinking Machines Lab's Tinker takes the "training as an API" idea furthest. Instead of
submitting a job, the user's own Python loop calls a remote service that handles distributed
training [48]:

- `create_lora_training_client(base_model, rank)` opens a LoRA training session;
- `forward_backward(...)` and `optim_step(...)` are the training primitives;
- `save_weights_and_get_sampling_client()` returns a client whose `sample(...)` generates from
  the current weights, so the same script can alternate between sampling and updating;
- `save_state` / `load_state` and checkpoint download make each cycle resumable and the
  weights portable.

The companion Tinker Cookbook implements SFT, RL with verifiable rewards, sandboxed code RL,
DPO and a three-stage RLHF pipeline, tool use, multi-agent self-play, rubric-based grading, and
on- and off-policy distillation on top of these primitives [48]. Its distillation results appear
in section 9 [40]. Fireworks' serverless path (section 5.1) exposes the same
`forward_backward` / `optim_step` shape. For flywheel builders the significance is that the
loop's control logic (when to sample, what to grade, when to stop) stays in the team's own
code, while GPU management is outsourced.

## 7. NVIDIA Data Flywheel Blueprint: automated experiment search

The NVIDIA blueprint (built on NeMo microservices, deprecated in April 2026 and kept for
reference) is the most complete *open reference implementation* of the automated loop [25].
Chapter 02 covers its place in NVIDIA's broader strategy. Its LLM-specific mechanics:

```
 Elasticsearch (prompt/completion logs, workload_id)
        | Celery task: create_datasets (dedup, stratified split by tool class)
        v
 NeMo Datastore ---> NeMo Customizer (LoRA) ---> candidate NIMs (spun up / down by
        |                                         NeMo Deployment Manager)
        +----------> NeMo Evaluator (LLM judge; tool-calling accuracy) <-------+
                              |
                              v
        per-NIM results: base | icl | customized    (MongoDB: flywheel_runs, ...)
                              |
                              v
                 human reviews best small candidate -> manual promotion
```

For each candidate NIM it runs three experiments [25][27]:

- **base**: replay the production prompts;
- **icl**: few-shot prompts drawn from traffic, chosen at random or by semantic similarity;
- **customized**: a LoRA fine-tune evaluated on the base prompts.

The ground truth is the *production model's own response*. This is deliberate: the blueprint
notes that it routes traffic into fine-tuning "without PII removal" and builds "evaluation
datasets with no ground truth other than what the production model is responding with" [25].
The objective is therefore **cost and latency reduction at parity**, not accuracy gains over
the teacher.

**Reported results.** On an internal HR chatbot served by `llama-3.1-70b-instruct`, the tool
calling workload proved simple enough that a fine-tuned `llama-3.2-1b-instruct` reached about
98% relative accuracy, which the blueprint presents as an inference-cost reduction of "up to
98.6%". In another case, `Qwen-2.5-32b-coder` matched the 70B model without fine-tuning,
cutting cost and time to first token by more than 50% [25].

**Operational details.**

- The minimum footprint is two H100s or A100s with a remote judge, or six with a self-hosted
  judge [25].
- Runs are serialized through a Celery queue with concurrency 1, because the system cannot yet
  see free GPUs and would otherwise over-allocate NIM deployments [25].

**Stated pitfalls** [26]:

- "Over-trusting a High Accuracy %": the remaining 5% may hold critical errors.
- "Ignoring Test-Time Compute": a small model that emits more tokens may not be cheaper.
- Data leakage between eval and train sets.
- Untagged workloads.
- Treating improvements as monotonic. Historical results must be kept because quality
  sometimes regresses.

Its roadmap items included PII redaction and the use of thumbs-up/down signals for DPO or KTO.
At the time of deprecation, user feedback "will have no effect" on selection [26].

## 8. Agent RL directly from production harnesses: Agent Lightning and OpenEnv

### 8.1 Problem

Agent RL frameworks often require rewriting the agent into the trainer's rollout abstraction.
Microsoft's Agent Lightning instead inserts a **proxy** between an unmodified agent and the
model. Every model call made during a rollout then becomes a training sample. The project
originally pitched this as training "ANY AI agent" with "(almost) zero code changes"; the v1.0
README describes agents reaching the trainer through the proxy with "ZERO changes" to the agent
[28].

### 8.2 Architecture (v1.0, August 2026)

```
 Trainer (verl + vLLM) --creates rollouts--> API Gateway <--polls-- Rollout Controller
        ^                                    |  (rollouts, events,  |  launches agent as
        |   model_request events             |   endpoints)         |  local proc or
        |   (prompt ids, response ids,       |                      |  Kubernetes Job
        |    logprobs) + reward events       v                      v
        +---------------------------- OpenAI-compatible proxy <-- unmodified agent
             /proxy/rollout/{id}/attempt/{a}/mode/train/openai/v1/chat/completions
```

The gateway is the source of truth [29]:

- **Rollouts** move through QUEUING → RUNNING → SUCCEEDED or FAILED, and carry append-only
  events (`model_request`, `reward` and custom events).
- **Rollout-specific URLs.** The rollout ID is embedded in the proxy URL, so every call is
  attributed to the right execution automatically.
- **Trainer behaviour.** The trainer merges consecutive calls only when their token histories
  are exactly continuous, computes advantages at the rollout level and supports rollout-level
  loss normalization.

Earlier versions (v0.2) framed this as a data-store-centric middleware. That design
decouples CPU-heavy agent runners from GPU-heavy trainers, and it lets one span store serve
RL, automatic prompt optimization or "filtering high rewards data and fit[ting] them via
Unsloth", which is rejection-sampling SFT [30].

### 8.3 Results and lessons

The project reports results on SWE-bench Verified [28]:

- Qwen3.5-9B improved from 41.8% to 56.4% (+14.6 points) using 6K training samples. The release
  includes the data cleaning and "reward-hacking prevention" pipeline.
- Qwen3.5-35B-A3B improved from 47.8% to 61.6% (+13.8 points) with pure RL on 1.8K examples.

These are project-reported figures; the README does not describe how the reward-hacking
prevention works.

Two lessons generalize to any trace-to-RL loop:

1. **Capture token IDs at serve time** (section 1.1).
2. **Treat the production harness as the environment.** That means keeping the agent's tools,
   context management and control flow in the loop instead of re-implementing them in a
   simulator.

### 8.4 Standardizing the environment side: OpenEnv

Agent Lightning standardizes the *model* side of the harness (an OpenAI-compatible proxy).
OpenEnv standardizes the *environment* side [49]. Each environment is a Docker image exposing a
Gymnasium-style `reset()`, `step()` and `state()` interface (sync or async) to a client over a
network connection, and can run locally, in a cloud sandbox or as a Hugging Face Space. A hub
lists 40+ environments, from coding sandboxes and browsers to games and finance simulators. The
README lists integrations with ART, SkyRL, torchforge, TRL (whose `GRPOTrainer` accepts an
`environment_factory`), Unsloth and others, and a technical committee drawn from Meta-PyTorch,
Hugging Face, NVIDIA, Microsoft, Unsloth and other organizations governs it. The project
describes itself as early-stage, with APIs that may change.

For trace-to-model loops, a shared environment interface addresses failure mode 8 in section
12: an agent whose production tools have side effects can be retrained against a containerized
replica that exposes the same tool surface. The same image can serve RL rollouts, offline
regression evals and, through its MCP endpoint, production tool calls.

## 9. Distillation techniques used in these loops

| Objective | Teacher signal | Student data | Where seen |
|---|---|---|---|
| Sequence-level imitation (SFT on teacher outputs) | Teacher text only | Teacher-generated (off-policy) | OpenAI Distill [1], OpenPipe [7], Predibase [15], NVIDIA [25], ART SFT [14] |
| Rationale-augmented SFT | Teacher labels plus explanations | Off-policy | Playbook "enrich data with explanation traces" [15]; Distilling Step-by-Step [43] |
| Offline logit distillation | Stored top-k teacher logits | Off-policy | Arcee DistillKit: polynomial approximation, quantization and bit-packing of logits [42] |
| Generalized KD (GKD) | Token-level teacher distribution via a JSD family | Mix of on- and off-policy (`lmbda`) | TRL GKDTrainer [41] |
| On-policy distillation (reverse KL) | Teacher logprob of student tokens | Student rollouts | Fireworks OPD [21], Tinker cookbook [40] |
| Prompt-program to weights | Traces of a prompted pipeline that pass a metric | Bootstrapped traces | DSPy `BootstrapFinetune` [38] |

Key points:

- **Off-policy imitation is the default for log-derived data.** Logs *are* teacher outputs.
  It is cheap, needs no teacher at training time, and works best for closed-label or
  schema-bound tasks [1][15][25]. Its known weakness is distribution mismatch: the student is
  never trained on its own mistakes. GKD was proposed to fix this. The TRL docs summarize the
  authors' finding that "on-policy data (high `lmbda`) performs better" [41].
- **On-policy distillation needs live teacher access during training** but gives dense,
  per-token credit. The Tinker cookbook's reasoning recipe reports that SFT on OpenThoughts3
  reached about 65% on AIME'24 after 3,000 steps with a rank-128 LoRA. A further 200 steps of
  on-policy distillation lifted it to about 76.7%. Their "personalization" recipe uses OPD to
  recover instruction-following (IF-eval) in about 100 steps after SFT on internal documents
  [40]. The pattern of adding domain knowledge with SFT and then restoring general behaviour
  with OPD applies directly to company-data flywheels.
- **Prompt optimization is an alternative to weight updates.** GEPA reflects on full
  execution traces instead of collapsing them into a scalar reward. Its authors claim it needs
  "100–500 evaluations vs. 5,000–25,000+ for GRPO" and report production uses at several
  companies (vendor and author claims) [39]. DSPy explicitly composes the two approaches:
  optimize prompts, then `BootstrapFinetune` the optimized program into a small LM [38].
  OpenAI's prompt optimizer and Together's GEPA notebook put this step on the same platform as
  fine-tuning [5][23].

## 10. Model routing improvement loops

Routing is the other way to cut cost with traces. Instead of making one model cheaper, it sends
each request to the cheapest model likely to answer it well. A router is itself a learned model
trained on outcome or preference data, so it can sit inside its own flywheel. Every routed
request produces a (query, model, outcome) tuple that can retrain the router.

```
           query --> [router f(q) -> P(strong model wins | q)] --threshold t-->
                        ^                                    |        |
                        |                                    v        v
                        |                               weak model  strong model
                        |                                    |        |
                        +---- retrain on (q, model, judged outcome / preference) <--+
                              + recalibrate t on recent traffic
```

### 10.1 Cascades: FrugalGPT

FrugalGPT (Stanford, 2023; TMLR version later) treats budget-constrained LLM use as a
composition problem. Its techniques include prompt adaptation, LLM approximation (for example
caching) and **LLM cascades**: query cheap models first and escalate when a learned scorer
judges the answer unreliable. The repository has been refreshed with newer model generations
and published API generations for 12 commercial models per dataset to support tradeoff
evaluation [34]. A cascade pays for the cheap call on every request. Its trace-loop benefit is
that the scorer can be retrained on logged pairs of cheap answers and verified outcomes.

### 10.2 Pre-generation routers: RouteLLM, RoRF and successors

RouteLLM (LMSYS with Anyscale, 2024) routes between a strong and a weak model before
generation. Its routers are trained on Chatbot Arena preference data, augmented with GPT-4 as a
judge [31]:

- **`mf`**: matrix factorization, the recommended router;
- **`sw_ranking`**: similarity-weighted Elo;
- **`bert`**: a BERT classifier;
- **`causal_llm`**: an LLM-based classifier.

A `random` router is also provided as a baseline.

Each router exposes one method, `calculate_strong_win_rate(prompt)`. A request goes to the
strong model when the win rate exceeds a cost threshold, encoded in the model name (for example
`router-mf-0.11593`). Reported results: cost reduced by "up to 85%" while keeping "95% GPT-4
performance" on MT Bench, and ">40% cheaper" than commercial offerings at equal performance.
The routers also generalize to other model pairs without retraining [31].

**Threshold calibration** is the operational knob. The threshold is calibrated on a sample of
queries to hit a target share of strong-model calls (for example 50%). The README warns that
"the % of calls routed to each model will differ based on the actual queries received," so
recalibration on recent traffic is part of the loop [31].

Successor projects follow the same template:

- **Not Diamond's RoRF** routes with random forests over embeddings. It ships 12 pre-trained
  routers across 6 model pairs and reuses RouteLLM's controller and threshold calibration. It
  reports that routing between two *strong* models can cut cost while beating both (vendor
  claim) [32].
- **RouterBench** (Martian) provides a standard cost-versus-quality evaluation pipeline and
  dataset for routers [33].
- **LLMRouter** (UIUC, 2025–2026) unifies 16+ router types across single-round, multi-round,
  multimodal, agentic and personalized routing. It adds a data-generation pipeline that calls
  APIs and evaluates outputs to build supervision automatically. Its paper reports that
  learned routers beat the strongest fixed-model baseline by 14.6% (relative) [35].

### 10.3 Routing in the gateway

Routing has moved into infrastructure gateways that also capture traces, which merges the
logging plane and the decision plane:

- **vLLM Semantic Router** offers a single endpoint (`vllm-sr/auto`) that selects or combines
  backends under a quality, latency and cost policy, across edge, private and cloud compute
  [36].
- **Plano** (Katanemo, built on Envoy) combines agent orchestration (its default router is
  described as a 4B-parameter routing model), LLM routing by model name, alias or preferences,
  and OpenTelemetry traces plus "Agentic Signals" captured "for continuous improvement" [37].

When the component that decides is also the component that logs, router retraining needs no
extra integration. The fine-tuned router in section 5.3 shows the other direction: a router
can itself be a small model distilled from traffic labels [20].

### 10.4 How routing and distillation interact

The playbook's "hybrid launch" sends easy queries to the student and hard ones to the teacher
[15]. That makes a router the natural promotion mechanism for a distilled model. Combined, the
loop looks like this:

1. Log all traffic through the gateway.
2. Distill a student on the slice where the teacher succeeds.
3. Train or calibrate a router to send that slice to the student.
4. Keep logging the student's traffic and the router's escalations.
5. Feed escalations back as both router training data and new distillation targets.

No single public system in this survey documents all five steps end-to-end, but every step has
a published component above.

## 11. Comparison

| System | Capture primitive | Curation | Training methods | Evaluator | Serving / promotion | Public numbers |
|---|---|---|---|---|---|---|
| OpenAI platform [1][2][3][6] | `store=True` + `metadata` | Dashboard filters by model and metadata | SFT distillation, DPO, RFT (o4-mini) | Evals API graders: label, score, python, similarity | Hosted `ft:` model; manual switch | Distilled 4o-mini 79.3% vs teacher 79.7% vs base 64.7%; RFT about +5 pts |
| Amazon Bedrock distillation [47] | Invocation logs to S3 + `requestMetadata` | `requestMetadataFilters`; reuse logged responses or re-query teacher (`usePromptResponse`) | Managed teacher-to-student SFT with data synthesis | User-side | Bedrock-hosted custom model | Not published in sample |
| OpenPipe (classic) [7][8][9] | Drop-in SDK or proxy, async report, tags | Log filters, pruning rules, 20% holdout | SFT on Mistral, Llama, GPT-3.5 | GPT-4 judge, head-to-head | Hosted OpenAI-compatible endpoint or weights | Not published in docs |
| OpenPipe ART + RULER [10][11][12] | Trajectory objects from client | Group per scenario | GRPO on LoRA (vLLM + Unsloth), SFT | RULER relative LLM judge or custom reward | LoRA hot-swap; W&B serverless backend | $15–200 per run (FAQ); ART·E beats o3 (vendor) |
| Predibase [15][16][17] | Stream logs to S3 or Snowflake | Playbook practices | LoRA SFT (Ludwig) | Task metrics; in-distribution and balanced test sets | LoRAX multi-LoRA; canary, dark or hybrid launch | LoRA Land: +34 vs base, +10 vs GPT-4; 25 adapters per A100 |
| Fireworks [19][20][21][22] | Eval Protocol rollouts and trace store | Per case study | SFT, DPO, ORPO, GRPO (sync and async), OPD, SDFT | EP evaluators | Dedicated vs serverless; multi-LoRA | Router SFT route accuracy 93.3% to 98.3% |
| Together [23][24] | (bring your own logs) | User-side | SFT, DPO, continual FT (`from_checkpoint`) | Evaluations API, judge tuning, GEPA | Dedicated endpoints | Not published in cookbook |
| Tinker [40][48] | (bring your own logs) | User code | LoRA SFT, RL, DPO/RLHF, on- and off-policy distillation via `forward_backward` / `optim_step` | User code; rubric grading recipes | Sampling client; downloadable checkpoints | OPD: AIME'24 about 65% to 76.7% |
| NVIDIA Data Flywheel [25][26][27] | Elasticsearch docs with `workload_id` | Dedup, stratified split | LoRA via NeMo Customizer; ICL | LLM judge; tool-call accuracy | Human promotion only | Up to 98.6% cost reduction (1B vs 70B on tool routing) |
| Agent Lightning [28][29][30] | Gateway proxy recording token IDs and logprobs | Merge only on continuous token histories | RL (verl), APO, RS-SFT | Agent-emitted reward events | Trainer endpoints | SWE-bench Verified +13.8 to +14.6 pts |
| RouteLLM / RoRF [31][32] | Preference or battle data | GPT-4-judge augmentation | MF, BERT, LLM classifier, random forest | MT Bench, MMLU, GSM8K | Threshold `router-mf-t` | Up to 85% cost cut at 95% GPT-4 quality |

## 12. Cross-cutting failure modes

1. **Reward hacking of LLM graders.** OpenAI's RFT run gained 20–30 points on a lexical-heavy
   rubric without improving clinical accuracy [3]. Mitigations: adversarial and contrastive
   grader examples, down-weighting easily gamed components, validating the grader's ranking
   before training, and running small pilots first [3]. Agent Lightning ships explicit
   "reward-hacking prevention" in its coding-agent pipeline [28].
2. **Teacher-as-ground-truth ceilings.** When eval labels are the production model's own
   outputs, a student can at best match the teacher. Systematic teacher errors become
   invisible [15][25]. Use gold labels on a slice and remove teacher errors, which improved
   students in the Jigsaw study [15].
3. **Eval/train leakage and moving test sets.** Deduplicate across splits and freeze
   regression sets [15][26].
4. **Task mixing.** Without per-node task ids, datasets blend incompatible behaviours [25][26].
5. **Aggregate metrics hide tail failures.** "95%" can hide the critical 5% [26]. Manual error
   analysis (open and axial coding, reviewing individual mistakes) is recommended by OpenAI,
   Hamel Husain and Predibase alike [5][15].
6. **Hidden cost regressions.** Smaller models that think longer or emit more tokens can cancel
   the savings. The RFT-tuned o4-mini used *more* reasoning tokens [3][26].
7. **Retokenization and serialization drift** in RL from logged text [30].
8. **Non-repeatable environments.** GRPO-style RL needs many rollouts per scenario, so it does
   not suit agents with irreversible side effects. Sandboxes or replayable environments are
   needed, ideally behind a standard interface such as OpenEnv [12][28][49].
9. **Router calibration drift.** Thresholds calibrated on public data do not hold on product
   traffic, and must be recalibrated on recent queries [31][32].
10. **Privacy and terms of service.** NVIDIA's reference flow fine-tunes on traffic without PII
    removal, and lists redaction as roadmap work [25][26]. OpenTelemetry marks prompt and
    completion content as opt-in and likely to contain PII [44]. The playbook advises checking
    provider terms before training on their outputs [15].

## Key takeaways for system designers

- **Design the log schema for training on day one.** Store the exact request, the exact
  response and a stable task id per call site, plus model, prompt version and user-feedback
  fields. For anything that may become RL, also store the token IDs and logprobs from the
  inference engine [25][29][30]. Prefer an open schema (OpenTelemetry GenAI spans and
  evaluation events), and remember that message content is opt-in there: turn it on
  deliberately, with redaction and retention rules [44][45].
- **Decide whether to reuse or regenerate targets.** Logged responses are free but frozen at
  whatever model and prompt were live; re-querying a current teacher on logged prompts costs
  more but removes stale labels [1][47].
- **Keep logging off the critical path.** Use async reporting, stream passthrough and client
  fallback when the logging service is down [9].
- **Build the evaluator before the trainer.** Freeze an in-distribution and a stress or
  balanced test set. Align LLM judges to SME labels with TPR and TNR, and treat every grader as
  an adversary the policy will exploit [3][5][15].
- **Pick the cheapest learning signal that works.** Prompt and ICL changes come first (the
  NVIDIA `icl` arm and GEPA). Then SFT distillation on narrow tasks, then on-policy
  distillation to fix exposure bias or restore general skills, and RL (RFT, GRPO, RULER) only
  when outcomes are verifiable and rollouts are repeatable [12][25][39][40].
- **Distill per workload, serve as adapters.** Multi-LoRA serving makes per-task and
  per-tenant specialists affordable, which is what lets the flywheel keep spinning [16][17].
- **Use routers as the promotion mechanism.** Ship the student behind a calibrated router,
  log the escalations, and feed them back as both router training data and new distillation
  targets [15][31].
- **Automate the search, not the promotion.** Every reference system surfaces candidates and
  leaves the production decision to humans, with versioned rollback [6][25][26].
- **Measure total cost per task,** including tokens generated, grader spend and judge latency,
  not just the per-token price of the model [3][26].

## References

1. "Leveraging model distillation to fine-tune a model". OpenAI Cookbook (pap-openai), 2024.
   https://github.com/openai/openai-cookbook/blob/main/examples/Leveraging_model_distillation_to_fine-tune_a_model.ipynb
2. "Evals API Use-case - Monitoring stored completions". OpenAI Cookbook (josiah-openai), 2025.
   https://github.com/openai/openai-cookbook/blob/main/examples/evaluation/use-cases/completion-monitoring.ipynb
3. "Exploring Model Graders for Reinforcement Fine-Tuning". OpenAI Cookbook (theophile-oai),
   2025. https://github.com/openai/openai-cookbook/blob/main/examples/Reinforcement_Fine_Tuning.ipynb
4. OpenAI Cookbook registry (catalogue of cookbook entries, dates and authors). OpenAI, 2026.
   https://github.com/openai/openai-cookbook/blob/main/registry.yaml
5. "Building resilient prompts using an evaluation flywheel". OpenAI Cookbook (neelk-oai, H.
   Husain), 2025.
   https://github.com/openai/openai-cookbook/blob/main/examples/evaluation/Building_resilient_prompts_using_an_evaluation_flywheel.md
6. "Self-Evolving Agents - A Cookbook for Autonomous Agent Retraining". OpenAI Cookbook
   (Maguranis, Frenkel, Perraudeau, Saladino et al.), 2025.
   https://github.com/openai/openai-cookbook/blob/main/examples/partners/self_evolving_agents/autonomous_agent_retraining.ipynb
7. OpenPipe: open-source fine-tuning and model-hosting platform (README). OpenPipe, n.d.
   https://github.com/OpenPipe/OpenPipe
8. "Logging Requests" (OpenPipe docs source). OpenPipe, n.d.
   https://github.com/OpenPipe/OpenPipe/blob/main/docs/features/request-logs/logging-requests.mdx
9. OpenPipe docs sources, n.d.:
   (a) "Training on Logs"
   https://github.com/OpenPipe/OpenPipe/blob/main/docs/features/request-logs/using-logs.mdx ;
   (b) "Pruning Rules"
   https://github.com/OpenPipe/OpenPipe/blob/main/docs/features/pruning-rules.mdx ;
   (c) "Evaluations"
   https://github.com/OpenPipe/OpenPipe/blob/main/docs/features/evaluations.mdx ;
   (d) "How reporting works"
   https://github.com/OpenPipe/OpenPipe/blob/main/docs/faq/how-reporting-works.mdx
10. "ART: Agent Reinforcement Trainer" (README). B. Hilton, K. Corbitt, D. Corbitt et al.,
    OpenPipe, 2025. https://github.com/OpenPipe/ART
11. "RULER: Relative Universal LLM-Elicited Rewards" (docs and source). OpenPipe, 2025.
    https://github.com/OpenPipe/ART/blob/main/docs/fundamentals/ruler.mdx ; source:
    https://github.com/OpenPipe/ART/blob/main/src/art/rewards/ruler.py
12. "Frequently Asked Questions" (ART docs). OpenPipe, 2025.
    https://github.com/OpenPipe/ART/blob/main/docs/getting-started/faq.mdx
13. "ART Training Loop" (ART docs). OpenPipe, 2025.
    https://github.com/OpenPipe/ART/blob/main/docs/fundamentals/training-loop.mdx
14. ART docs, 2025-2026: (a) "SFT Training"
    https://github.com/OpenPipe/ART/blob/main/docs/fundamentals/sft-training.mdx ;
    (b) "Checkpoint Forking"
    https://github.com/OpenPipe/ART/blob/main/docs/features/checkpoint-forking.mdx
15. J. Zhao, W. Abid, "LLM Distillation Playbook". Predibase, 2024.
    https://github.com/predibase/llm_distillation_playbook
16. "LoRA Land: 310 Fine-tuned LLMs that Rival GPT-4" (code and summary). Predibase, 2024.
    https://github.com/predibase/lora_bakeoff (paper: https://arxiv.org/abs/2405.00732)
17. "LoRAX: Multi-LoRA inference server that scales to 1000s of fine-tuned LLMs". Predibase,
    n.d. https://github.com/predibase/lorax
18. "Fireworks AI Cookbook" (README). Fireworks AI, 2026. https://github.com/fw-ai/cookbook
19. "Fireworks Training Cookbook" (training README: recipes, case studies). Fireworks AI, 2026.
    https://github.com/fw-ai/cookbook/blob/main/training/README.md
20. "SFT: prompt-routing classifier" (case study). Fireworks AI, 2026.
    https://github.com/fw-ai/cookbook/blob/main/training/case-studies/sft_prompt_router/README.md
21. "Distillation (OPD)" (example README). Fireworks AI, 2026.
    https://github.com/fw-ai/cookbook/blob/main/training/examples/distillation/README.md
22. "Eval Protocol" (README). Eval Protocol project (Fireworks AI ecosystem), 2025.
    https://github.com/eval-protocol/python-sdk
23. "Together AI Cookbook" (README). Together AI, n.d.
    https://github.com/togethercomputer/together-cookbook
24. "Continual Fine-tuning (CFT)" (notebook). Together AI, n.d.
    https://github.com/togethercomputer/together-cookbook/blob/main/fine-tuning/Continual_Finetuning.ipynb
25. "Data Flywheel Foundational Blueprint" (README). NVIDIA AI Blueprints, 2025 (deprecated
    April 2026). https://github.com/NVIDIA-AI-Blueprints/data-flywheel
26. "Limitations & Best Practices" (Data Flywheel docs). NVIDIA, 2025.
    https://github.com/NVIDIA-AI-Blueprints/data-flywheel/blob/main/docs/05-limitations-best-practices.md
27. "Evaluation Types and Metrics" (Data Flywheel docs). NVIDIA, 2025.
    https://github.com/NVIDIA-AI-Blueprints/data-flywheel/blob/main/docs/06-evaluation-types-and-metrics.md
28. "Agent Lightning" (README; v1.0 technical report arXiv:2608.17528; original paper
    arXiv:2508.03680). Microsoft Research, 2025-2026. https://github.com/microsoft/agent-lightning
29. "Basics" (Agent Lightning v1.0 docs). Microsoft Research, 2026.
    https://github.com/microsoft/agent-lightning/blob/main/docs/05-basics.md
30. "No More Retokenization Drift: Returning Token IDs via the OpenAI Compatible API Matters in
    Agent RL". The Agent Lightning team, vLLM Blog, 2025.
    https://blog.vllm.ai/2025/10/22/agent-lightning.html
31. I. Ong, A. Almahairi, V. Wu, W.-L. Chiang, T. Wu, J. E. Gonzalez, M. W. Kadous, I. Stoica,
    "RouteLLM: Learning to Route LLMs with Preference Data" (code README). LMSYS, 2024.
    https://github.com/lm-sys/RouteLLM (paper: https://arxiv.org/abs/2406.18665)
32. "RoRF - Routing on Random Forests". Not Diamond, 2024. https://github.com/Not-Diamond/RoRF
33. "RouterBench: A Benchmark for Multi-LLM Routing System" (code). Martian, 2024.
    https://github.com/withmartian/routerbench (paper: https://arxiv.org/abs/2403.12031)
34. L. Chen, M. Zaharia, J. Zou, "FrugalGPT: How to Use Large Language Models While Reducing
    Cost and Improving Performance" (code README). Stanford, 2023-2025.
    https://github.com/stanford-futuredata/FrugalGPT (paper: https://arxiv.org/abs/2305.05176)
35. "LLMRouter: An Open-Source Library for LLM Routing". UIUC U Lab, 2025-2026.
    https://github.com/ulab-uiuc/LLMRouter (paper: https://arxiv.org/abs/2608.06867)
36. "vLLM Semantic Router" (README). vLLM project, 2026.
    https://github.com/vllm-project/semantic-router
37. "Plano: the AI-native proxy server and data plane for agentic apps" (README). Katanemo,
    2025-2026. https://github.com/katanemo/plano
38. "Optimizers" (DSPy documentation source). Stanford NLP, n.d.
    https://github.com/stanfordnlp/dspy/blob/main/docs/docs/learn/optimization/optimizers.md
39. "GEPA: Genetic-Pareto optimization" (README; paper arXiv:2507.19457). GEPA authors, 2025.
    https://github.com/gepa-ai/gepa
40. "Distillation" recipes (Tinker Cookbook; reproduces the "On-Policy Distillation" blog
    post). Thinking Machines Lab, 2025.
    https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/recipes/distillation/README.md
41. "Generalized Knowledge Distillation Trainer" (TRL docs; based on Agarwal et al., "On-Policy
    Distillation of Language Models: Learning from Self-Generated Mistakes"). Hugging Face, n.d.
    https://github.com/huggingface/trl/blob/main/docs/source/gkd_trainer.md
42. "DistillKit" (README). Arcee AI, n.d. https://github.com/arcee-ai/DistillKit
43. "Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and
    Smaller Model Sizes" (code). Google Research, 2023.
    https://github.com/google-research/distilling-step-by-step (paper:
    https://arxiv.org/abs/2305.02301)
44. "Semantic conventions for generative client AI spans" (gen-ai-spans.md). OpenTelemetry
    GenAI Semantic Conventions, Development status, 2024-2026.
    https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md
45. "Semantic conventions for Generative AI events" (gen-ai-events.md, including
    `gen_ai.evaluation.result`). OpenTelemetry, 2025-2026.
    https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-events.md
46. "Langfuse: open source LLM engineering platform" (README). Langfuse, n.d.
    https://github.com/langfuse/langfuse
47. "Amazon Bedrock Model Distillation" samples (README and
    `Historical_invocation_distillation.ipynb`). AWS Samples, 2025.
    https://github.com/aws-samples/amazon-bedrock-samples/tree/main/custom-models/model_distillation
48. "Tinker Cookbook" (README: Tinker API primitives and recipe list). Thinking Machines Lab,
    2025-2026. https://github.com/thinking-machines-lab/tinker-cookbook
49. "OpenEnv: an interface library for RL post-training with environments" (README). OpenEnv
    project (Meta-PyTorch, Hugging Face et al.), 2025-2026. https://github.com/meta-pytorch/OpenEnv
