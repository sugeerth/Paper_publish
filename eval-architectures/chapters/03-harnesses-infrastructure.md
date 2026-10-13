# 03. Eval Harnesses and Platforms: The Architecture of Fourteen Systems

## Overview

Chapter 01 set out the layers of an "eval stack": datasets, subject adapters, runners, graders, storage, analysis, and gates. This chapter shows how fourteen real systems build those layers. They come in two families, and the split matters more than any single feature.

**Model-centric harnesses** answer "how good is this model on this benchmark?" The subject under test is a model endpoint or a set of weights. These harnesses make prompts reproducible, batch inference efficiently, and score against fixed references. This family includes EleutherAI's lm-evaluation-harness, Stanford HELM, OpenAI's `evals` and `simple-evals`, the UK AI Security Institute's Inspect, Hugging Face lighteval, and Shanghai AI Lab's OpenCompass. Most are open-source Python command-line tools that write results to files.

**Application-centric platforms** answer "did this change to my prompt, retriever, or agent make the product better or worse?" The subject is an arbitrary function in a customer's codebase. These platforms are built around tracing, versioned datasets, experiment comparison, human annotation, and online scoring of production traffic. This family includes Braintrust, LangSmith, promptfoo, Arize Phoenix, W&B Weave, Humanloop, Langfuse, and Patronus. Most are SaaS products with a self-hosting option. Their hardest engineering problems are in storage and ingestion, not in scoring.

The two families are converging. Inspect has added agents, sandboxes, and a transcript viewer, so it now covers ground the platforms used to own. lighteval made Inspect its preferred backend [25]. promptfoo added trajectory assertions over OpenTelemetry traces [40]. Phoenix and Langfuse run LLM-as-judge evaluators over traces. The 2025-2026 period also brought consolidation. Promptfoo states that it "is now part of OpenAI" [39]. Langfuse states that it has been "part of ClickHouse" since January 2026 [52]. HELM "entered maintenance mode on June 1, 2026" [6]. OpenAI's simple-evals stopped receiving new models or results in July 2025 [12].

For each system, this chapter covers the problem it targets, its components and data flow, its key techniques, any public scale numbers, its failure modes, and lessons for designers. A comparison table and a set of design takeaways follow.

A note on sources: this chapter relies mainly on the systems' own repositories and documentation, much of it read directly from GitHub-hosted docs sources. Where a number comes from a vendor's own claim rather than independent measurement, the text says so.

---

## 1. The common skeleton

Despite different vocabularies, every system in this chapter instantiates the same pipeline:

```
  +-----------+    +-------------+    +-----------+    +----------+    +-----------+
  |  Dataset  |--->|   Prompt /  |--->|  Subject  |--->|  Scorer  |--->| Aggregate |
  | (samples, |    |  adaptation |    | (model or |    | (code,   |    | + store + |
  |  targets) |    |  (template, |    |  app fn,  |    |  judge,  |    |  view /   |
  +-----------+    |  few-shot)  |    |  agent)   |    |  human)  |    |  compare  |
                   +-------------+    +-----------+    +----------+    +-----------+
                         ^                  |                               |
                         |         cache / retry / rate-limit               |
                         +---- config (YAML / Python / UI) -----------------+
```

The vocabulary differs from system to system:

| Concept | lm-eval-harness | HELM | Inspect | OpenCompass | Braintrust | LangSmith | promptfoo | Weave |
|---|---|---|---|---|---|---|---|---|
| Unit of test | doc / Instance | Instance | Sample | dataset row | data record | Example | test case | dataset row |
| Prompt adaptation | doc_to_text + Jinja | Adapter (AdapterSpec) | Solver chain | infer_cfg prompt template | inside `task` fn | inside target fn | `prompts` + `vars` | `preprocess_model_input` |
| Subject | `LM` subclass | Client via Executor | Model provider / Agent | model config | `task` fn | target fn | provider | `Model` / `weave.op` |
| Grader | metric_list + filters | Metric -> Stat | Scorer | eval_cfg evaluator | scorer (autoevals) | evaluator | assert | scorer |
| Run grouping | task group | RunSpec / suite | Task / eval set | model x dataset Task | Experiment | Experiment | eval | Evaluation run |

The real design differences lie in five places:

1. How the subject is abstracted: a probability interface or a chat or agent interface.
2. How prompts are made reproducible.
3. Where concurrency and caching sit.
4. What the result artifact looks like: an aggregate number, a per-sample log, or a full trace tree.
5. Whether evaluation is an offline batch job or a continuous process over production traffic.

---

## 2. EleutherAI lm-evaluation-harness

### Problem

The harness began as a framework for "few-shot evaluation of language models." It targets reproducible scoring of base and instruction-tuned models on academic benchmarks. It is "the backend for Hugging Face's popular Open LLM Leaderboard" [1]. The repository advertises "over 60 standard academic benchmarks for LLMs, with hundreds of subtasks and variants implemented," and has about 14.2k GitHub stars [1].

### Architecture

The harness separates **tasks**, declared in YAML, from **models**, written as Python classes. The two meet only through a narrow request protocol. Every model backend subclasses `lm_eval.api.model.LM` and implements three methods. Each method receives a list of `Instance` objects whose `request_type` matches the method name [2]:

- `loglikelihood(requests)`: `args = (context, target)`. Returns `(ll, is_greedy)`, where `is_greedy` says whether greedy decoding would have produced the target.
- `loglikelihood_rolling(requests)`: `args = (text,)`. Scores a full text conditioned only on the end-of-text token, for perplexity.
- `generate_until(requests)`: `args = (input_string, gen_kwargs)`, with stop sequences in `until` and `max_gen_toks`. Returns strings.

```
  task YAML ---> TaskConfig ---> Task.build_requests()
   (dataset_path, doc_to_text,         |  list[Instance(request_type, args)]
    doc_to_target, output_type,        v
    filter_list, metric_list)   +----------------------------------------+
                                | Evaluator: group by request_type,     |
                                | Reorderer sorts by length (desc),     |
                                | optional SQLite response cache        |
                                +----------------------------------------+
                                         |                    |
                              LM.loglikelihood()     LM.generate_until()
                              (HF, vLLM, SGLang,     (API backends, local-
                               NeMo, GGUF, ...)       chat-completions, ...)
                                         |                    |
                                         v                    v
                               filters (regex, take_first, majority_vote)
                                         |
                               metrics (acc, acc_norm, exact_match, f1, ...)
                               -> aggregation (mean, perplexity) -> groups
```

The task YAML schema is rich [3]:

- **Dataset fields:** `dataset_path` (a Hugging Face Hub name), `dataset_name`, `dataset_kwargs`, split selectors, and `process_docs` hooks.
- **Prompt fields:** `doc_to_text`, `doc_to_target`, and `doc_to_choice`. Each accepts a Jinja2 template, a string, or a Python function via `!function`.
- **Formatting fields:** `fewshot_delimiter`, `target_delimiter`, and `gen_prefix`.
- **`output_type`:** one of `generate_until`, `loglikelihood`, `loglikelihood_rolling`, or `multiple_choice`.
- **`repeats`:** runs each sample several times, for self-consistency.
- **`filter_list`:** named post-processing pipelines over model outputs.
- **`metric_list`:** metrics with aggregation and `higher_is_better` settings.
- **`include`:** inherits a base YAML, so variants override only the fields that differ.

Group configs bundle tasks and set `aggregate_metric_list` with `weight_by_size`, which switches between micro- and macro-averaging [3].

### Key techniques

- **Multiple-choice as log-likelihood.** A `multiple_choice` task becomes one `loglikelihood` request per choice. The argmax gives `acc`, and a length-normalized variant gives `acc_norm`. This needs logits, which is why many API-only models can run only generative tasks.
- **Filter pipelines.** Several filters can run over the same outputs. The documented GSM8K self-consistency config uses `repeats: 64` and defines three pipelines, `score-first`, `maj@64`, and `maj@8`, over a single set of generations [3]. This separates the cost of generation from the choice of scoring rule.
- **Length-sorted batching.** The model guide recommends `lm_eval.utils.Reorderer` to process inputs in descending length. Out-of-memory errors then surface early, and runtime estimates are conservative [2]. `--batch_size auto:N` re-tunes batch size N times during a run [4].
- **Caching keyed by tokenizer and chat template.** `--use_cache` stores responses in a SQLite file. The cache key includes `tokenizer_name`, so a transcript rendered with one chat template is never reused for another [2][4]. The Jinja chat template itself is saved with results [2].
- **Chat-era adaptations.** `--apply_chat_template`, `--fewshot_as_multiturn`, and `--system_instruction` turn few-shot prompts into multi-turn conversations [2][4]. A July 2025 change added `think_end_token` to strip reasoning traces before scoring for `hf`, `vllm`, and `sglang` [1].
- **Plugin architecture.** In September 2026, model backends, filters, metrics, and aggregations became registrable from external packages through `lm_eval.*` entry points or a `--plugins` module [1]. A December 2025 CLI refactor added `run`, `ls`, and `validate` subcommands. The same release moved `torch` and `transformers` into extras such as `lm_eval[hf]` [1].
- **Seed control.** `--seed` takes four values, for Python, NumPy, torch, and few-shot sampling. The default is `0,1234,1234,1234` [4].

### Failure modes and lessons

The maintainers' paper, "Lessons from the Trenches on Reproducible Evaluation of Language Models" [5], argues that scores are highly sensitive to implementation details: prompt formatting, normalization, answer extraction, and few-shot selection. It recommends sharing exact evaluation code and prompts and reporting uncertainty. The harness's design follows from this. Prompts live in versioned YAML, chat templates are saved with results, `--log_samples` stores every input and output, and `--check_integrity` runs task tests [1]. Other failure modes are visible in the docs:

- `--limit` is "intended for testing only," so partial runs are not comparable [4].
- `generation_kwargs` are passed to Hugging Face `generate` and "may not work" with other model classes [3].
- Log-likelihood tasks are unavailable for chat APIs that hide logits.

**Lesson:** a narrow model interface of three methods made it possible to support more than 20 backends, from Megatron-LM to llama.cpp to LiteLLM [1]. The price is that any capability outside those three methods, such as tool use, multi-turn agents, or sandboxes, is out of scope.

---

## 3. Stanford HELM

### Problem

HELM (Holistic Evaluation of Language Models) set out to "improve the transparency of language models" by measuring many models on a shared, standardized grid of scenarios and metrics, instead of each lab reporting a different subset. In the original paper's abstract, HELM measures 7 metrics (accuracy, calibration, robustness, fairness, bias, toxicity, and efficiency) on 16 core scenarios, plus 26 targeted evaluations, across 30 prominent models. The paper says this raised standardized benchmark coverage of those models from 17.9% to 96.0% [8]. It was published in TMLR in 2023 [6].

### Architecture

HELM's developer guide divides classes into three groups [7]:

- **Specifications**, written by users: `RunSpec`, `AdapterSpec`, `ExecutionSpec`, plus `ObjectSpec`, a class name and an argument dictionary used to build Scenarios and Metrics.
- **States**, generated and serializable: `Instance`, `ScenarioState`, `Request`, and `RequestResult`.
- **Controllers**, holding logic but not serialized: `Scenario`, `Adapter`, `Executor`, `Metric`, and `Runner`.

```
 RunSpec (built by a @run_spec_function in run_specs.py)
   |-- ScenarioSpec --> Scenario.get_instances()
   |                     -> Instances(input, References[CORRECT_TAG], split)
   |-- DataAugmenterSpec --> DataPreprocessor (ids + one perturbation per instance)
   |-- AdapterSpec --> Adapter -> ScenarioState[RequestState(Request, Instance, ...)]
   |-- ExecutionSpec --> Executor -> RequestResult (via model client + cache)
   '-- MetricSpecs --> Metric -> Stats (aggregate; can hold several values)
 Runner loops over RunSpecs  ->  helm-summarize  ->  helm-server (web UI, :8000)
```

The CLI pipeline has three steps [6]:

1. `helm-run --run-entries mmlu:subject=philosophy,model=openai/gpt2 --suite ... --max-eval-instances N` runs the evaluations.
2. `helm-summarize` aggregates a suite.
3. `helm-server` serves a leaderboard plus per-instance prompts and responses.

Each model also needs a `WindowService`, which handles tokenization and context-window truncation for that model. A `WindowServiceFactory` maps models to window services [7]. `static/schema.yaml` defines how scenarios group into leaderboard subgroups, and `static/contamination.yaml` records which models were trained on which scenarios' data [7].

### Key techniques

- **Perturbations as first-class data.** Robustness and fairness come from the `DataAugmenterSpec`/`PerturbationSpec` path, which applies transformations such as typos or dialect changes to instances before adaptation [7]. That is how one scenario yields both accuracy and robustness numbers.
- **The Adapter as the unit of standardization.** Settings such as in-context example count, temperature, and prompt format live in `AdapterSpec`. Every model on a scenario therefore sees exactly the same adaptation, which is the main point of the "holistic" comparison.
- **A family of derived leaderboards.** The same engine powers HELM Capabilities and HELM Safety (the flagships), VHELM (vision-language), HEIM (text-to-image), MedHELM (medical, published in Nature), audio-language evaluation, Image2Struct, enterprise benchmarks, and table-reasoning robustness (ToRR) [6]. Methodological follow-ups such as "Efficient Benchmarking of Language Models" and "Reliable and Efficient Amortized Model-based Evaluation" also build on it [6].

### Failure modes and lessons

Holistic coverage is expensive. Every new model must be run across the full grid of scenarios and metrics, and every new scenario across all models, so maintenance cost grows with the product of the two. The developer guide itself warns that it "was last modified more than ten months ago" [7]. The project entered maintenance mode on June 1, 2026 [6]. **Lesson:** the separation of spec, state, and controller, together with serializable intermediate states, is a clean reference design for reproducibility, because every `RequestState` can be inspected. Keeping a central, curated leaderboard current, however, takes ongoing institutional funding that a framework alone does not provide.

---

## 4. OpenAI evals and simple-evals

### openai/evals: registry plus completion functions

OpenAI open-sourced `evals` as "a framework for evaluating LLMs and LLM systems, and an open-source registry of benchmarks" [9]. It has about 19.6k stars [9]. It has two notable abstractions.

**1. Templated evals defined by data and YAML.** For an existing template, you supply JSONL data and a YAML spec, with no evaluation code [9]. The basic templates compare a completion `a` with references `B` [10]:

- `Match`: `a` starts with some reference in `B`.
- `Includes`: some reference appears anywhere in `a`.
- `FuzzyMatch`: either string contains the other.
- `JsonMatch`: semantically equal JSON. Key order is ignored, and invalid JSON never matches.

`ModelBasedClassify` wraps a completion in a grading prompt. The grader must answer from a closed set of `choice_strings`, and any other answer becomes `__invalid__`. Optional `choice_scores` map choices to numbers, and `eval_type` sets the order of reasoning and answer: `cot_classify` (reason first, the recommended default), `classify_cot`, or `classify` [10]. Example graders include `fact` (five-way relation to a reference: subset, superset, equal, disagreement, or differences that don't matter), `closedqa`, and `battle` (pairwise) [10]. This closed-set, reason-then-answer judge design reappears almost unchanged in Braintrust's autoevals, Phoenix's `create_classifier`, and promptfoo's `model-graded-closedqa`.

**2. The Completion Function Protocol.** The subject is not a model but a "completion function" that takes a prompt (text or chat) and returns a list of strings. Any chain, tool-using agent, or browser-augmented system that implements this interface can run against any eval [11]. Completion functions are registered in YAML under `evals/registry/completion_fns`, with a `class` import path and `args`. External registries plug in through `--registry_path` [11].

**Data management.** Registry data lives in Git LFS [9]. **Status:** the README now points out that evals "can now be configured and run directly in the OpenAI Dashboard." Contributions of custom-code evals are not accepted [9].

### simple-evals: a deliberately minimal reference

`simple-evals` was released "to be transparent about the accuracy numbers" OpenAI publishes with its models [12]. Its design philosophy is an explicit position on prompting. It rejects few-shot and role-play prompts as carryovers from the base-model era and uses "zero-shot, chain-of-thought" prompts with simple instructions [12]. It includes MMLU, MATH, GPQA, DROP, MGSM, HumanEval, SimpleQA, BrowseComp, and HealthBench, with samplers for the OpenAI and Claude APIs [12]. As of July 2025 it is no longer updated for new models or results. It remains the reference implementation for HealthBench, BrowseComp, and SimpleQA [12].

**Lesson:** OpenAI's two repositories show two sides of a trade-off. `evals` aimed to be a general, crowd-sourced registry and drifted toward a hosted product. `simple-evals` aimed to be the smallest code that reproduces a published number. For reporting headline numbers, the minimal approach turned out to be the easier one to keep credible.

---

## 5. UK AISI Inspect

### Problem

The UK AI Security Institute built Inspect for frontier-model evaluation. That work needs multi-turn agents, tool use, untrusted code execution, and many models evaluated under identical conditions. Inspect is MIT-licensed, has about 3.0k stars and nearly 8,000 commits, and links to a collection of more than 200 pre-built evaluations [13]. The `inspect_evals` library is maintained by Generality Labs, with contributions from AISI, Arcadia Impact, and the Vector Institute [23].

### Architecture

The core abstraction is a `Task`, which combines a `Dataset`, a `Solver`, and a `Scorer` [14]:

```python
@task
def my_eval():
    return Task(
        dataset=json_dataset("questions.json"),
        solver=[system_message(SYS), chain_of_thought(), generate()],
        scorer=model_graded_qa(),
        sandbox="docker",
    )
```

```
                         inspect eval / eval_set()
                                   |
     Dataset (HF/CSV/JSON, FieldSpec) --> Sample(input, target, files, setup)
                                   |  (one async coroutine per sample, epochs)
                                   v
   +----------------------------------------------------------------------+
   | TaskState(messages, output, tools, metadata, store, completed)       |
   |   solver chain: system_message -> chain_of_thought -> use_tools      |
   |                 -> generate() / react() agent / handoff()            |
   |        |                     |                        |              |
   |   Model API (20+ providers,  |  tools: bash, python,  |  sandbox():  |
   |   max_connections/provider,  |  text editor, web,     |  docker/k8s/ |
   |   retries, local cache)      |  computer, MCP         |  modal/...   |
   +----------------------------------------------------------------------+
                                   |
                         Scorer(s) -> Score -> metrics (accuracy, mean, ...)
                                   |
                .eval log (zstd, per-sample, attachments) -> inspect view / S3
```

**Solvers** are async functions `solve(state: TaskState, generate) -> TaskState`. `generate` calls the model, appends the reply to `state.messages`, and sets `state.output` [15]. Built-in solvers include `system_message`, `prompt_template`, `chain_of_thought`, `self_critique` (optionally using a different model), `multiple_choice`, and `use_tools`. They compose through lists or `chain()`. Setting `state.completed = True`, or reaching a `message_limit`, ends a sample early [15].

**Agents** implement a narrower protocol, `execute(AgentState) -> AgentState`. That makes them reusable in four ways: as a top-level solver, run standalone with `run()`, exposed as a string-in, string-out tool with `as_tool()`, or given control with `handoff()`, which passes the full conversation history to another agent [20]. The built-in `react()` agent runs a reason-act-observe loop. An "Agent Bridge" runs agents written in the OpenAI Agents SDK, LangChain, or Pydantic AI inside Inspect, and the Inspect SWE package runs software-engineering agents such as Claude Code and Codex CLI [20]. Token, message, and time limits bound agent runs [20].

**Sandboxes** give each sample its own isolated environment, even when the sandbox is declared at the task level, so samples cannot interfere with each other [16]. Docker and `local` are built in. External packages provide `k8s`, `daytona`, `modal`, `ec2`, `proxmox`, `vagrant`, and `openshell` [16]. Two details matter for security. First, tool code runs in the evaluation process, and only work requested through `sandbox()` runs in the container. Second, Inspect's generated Compose config sets `network_mode: none`, but a custom Compose file replaces it, so isolation must be set again by hand [16]. Defaults cap file reads at 100 MB and exec output at 10 MB. Built-in providers truncate the *front* of each stream, so `exec()` can return incomplete output without an error. The docs recommend writing output to a file and reading it back for complete results [16].

### Concurrency, caching, and resumption

- **Async by default.** Inspect starts many samples at once and relies on limits to avoid saturation. Its async code is built on AnyIO, with asyncio as the default and Trio optional [18].
- **Per-provider limits.** `max_connections` is per model provider, so different providers do not compete for the same connection budget. `max_tasks` defaults to 1, and `max_subprocesses` respects the container's CPU limits [18].
- **Dataset paging.** `max_dataset_memory` pages large datasets to disk [18].
- **Named concurrency for external services.** Calls to other services are wrapped in `concurrency("name", n)` [18].
- **Model output cache.** The cache key combines model name and base URL, message history, epoch, generate config, and active tools and `tool_choice`. Default expiry is one week. The docs warn that model aliases can produce stale hits, so versioned model names are recommended [21].
- **Eval sets** are the production runner. They retry failed tasks (10 attempts by default), reuse completed samples from failed tasks, and resume. Re-running the same command schedules only unfinished work, and adding models or epochs to the command extends a completed set [19].

### Logs as a first-class artifact

Since v0.3.46, the default `.eval` log format is a binary format "optimised for size and speed." It is about one-eighth the size of JSON and lets the viewer read samples incrementally [17]. From v0.3.206, repeated messages across model events are deduplicated and zstd-compressed. The docs report about 10:1 reductions on agentic benchmarks [17]. Large content such as images is stored once as an attachment. Raw provider requests and responses are logged for the first few calls per model and for all errors, and provider request IDs are always recorded [17]. The API supports header-only reads, sample summaries, and generators that yield one sample at a time. With these, analysts can work across thousands of logs without loading full transcripts [17]. Scoring can also run after generation: `inspect score` re-scores existing logs, and scores can be edited with an audit trail of score-edit events [17][22].

### Lessons

Inspect's architecture shows how the field has shifted. The unit of evaluation is no longer a prompt-completion pair but a transcript: a sequence of model calls, tool calls, and sandbox actions, which has to be stored efficiently, browsed, re-scored, and resumed. Its main failure modes are operational: leaked containers after interruption (handled by `inspect sandbox cleanup`), silent output truncation, and misconfigured network isolation [16]. Inspect has also become a backend for other harnesses, as section 6 shows.

---

## 6. Hugging Face lighteval

### Problem and architecture

lighteval is Hugging Face's evaluation toolkit, covering "1000+ evaluation tasks" across many domains and languages. It runs on many backends from one task definition [24]. Entry points [24]:

- `lighteval eval`: inspect-ai backend, the preferred entry point.
- `lighteval accelerate`: local CPU or GPUs via Accelerate.
- `lighteval nanotron`: distributed evaluation with Nanotron, Hugging Face's training framework.
- `lighteval vllm` and `lighteval sglang`: high-throughput local inference.
- `lighteval endpoint`: Inference Endpoints, TGI, LiteLLM, and Inference Providers.
- `lighteval custom`: custom models.
- A Python API (`Pipeline`, `EvaluationTracker`) for a model already in memory, for example during training.

A task is a `LightevalTaskConfig`. Its fields include `name`, `prompt_function`, `hf_repo` and `hf_subset`, splits, `few_shots_split` and `few_shots_select`, `metrics`, `generation_size`, and `stop_sequence` [26]. The prompt function maps a dataset row to a `Doc` with `query`, `choices`, and `gold_index`. Metrics are `SampleLevelMetric` objects with a `sample_level_fn` and a `corpus_level_fn` [26]. Task strings carry the few-shot count, as in `gsm8k|5` [25][26].

### Key techniques

- **Prediction cache in Parquet.** Generations, logits, and probabilities are cached per sample under `~/.cache/huggingface/lighteval/predictions/`, organized by model name, a hash of the model configuration, and task. The hash forces a fresh cache when weights, revision, or generation parameters change [27].
- **Sample-level outputs pushed to the Hub.** Results are saved sample by sample and can be published as Hub datasets [24].
- **Delegation to Inspect.** With `lighteval eval`, provider access, retries (`--max-retries`, `--retry-on-error`), connection limits, epochs and reducers, and the log viewer all come from inspect-ai. Logs go to `lighteval-logs` and open in `inspect view` [25]. The docs warn that "not all tasks are compatible with inspect-ai's API as of yet" [25].

**Lesson:** lighteval's move is a sign of commoditization. A harness owned by a model hub chose to reuse a government lab's runner and log format rather than maintain its own API-provider layer. That suggests Inspect's runtime and `.eval` logs are becoming a shared substrate.

---

## 7. OpenCompass

### Problem and architecture

OpenCompass, from Shanghai AI Laboratory, offers "a complete workflow spanning dataset and model configuration, inference, evaluation, and result summarization" over 100+ datasets. It has about 7.5k stars [28]. Its architecture is the most explicitly *distributed-systems* design among the model-centric harnesses [29]:

```
 config.py (MMEngine read_base: models=[...], datasets=[...], infer, eval, summarizer)
      |
      v
 Partitioner: split each (model x dataset) pair into parallelizable Tasks
      |
      v
 Runner: execute Tasks locally or on a cluster      --> predictions/
      |        (inference stage: PPL or Gen inferencer)
      v
 Runner: evaluation Tasks (rule-based, LLM judge,   --> results/
      |    cascade)
      v
 Summarizer: merge subset scores into summary tables --> summary/
 (stages resumable: --reuse; selectable: --mode eval | viz)
```

Configs are Python files composed through MMEngine's `read_base()` [29]. A dataset config has three parts:

- `reader_cfg`: input and answer fields, and splits.
- `infer_cfg`: prompt template, few-shot retriever, and inference method, either **PPL** (perplexity-style candidate scoring) or **Gen** (generation).
- `eval_cfg`: post-processing and metrics [29].

Model backends include Hugging Face, LMDeploy, and vLLM, plus APIs such as the OpenAI Responses API, LiteLLM, and the Gemini and Anthropic SDKs [28].

### Key techniques

- **Stage-separated persistence.** Predictions and results are written to separate directories. A failed evaluation stage can therefore be rerun without repeating inference, using `--reuse` [29].
- **Composable evaluators.** `GenericLLMEvaluator` (LLM-as-judge), `MATHVerifyEvaluator`, and `CascadeEvaluator` chain evaluators in sequence, for example a cheap rule-based check first and an LLM judge only for unresolved cases [28].
- **Raw prompt passthrough.** `RawPromptTemplate` sends a benchmark's original prompts without reformatting [28].
- **Operational robustness.** The README lists concurrent inference with evaluation watching and heartbeat monitoring of workers [28]. VLMEvalKit has been integrated for multimodal datasets since 2026-08-25 [28].

**Lesson:** partitioning the model x dataset matrix into independent tasks, and saving each stage's outputs, is what makes thousand-cell sweeps restartable on shared clusters. The cost is configuration complexity. Inheritance chains in Python configs make it hard to see the final prompt actually sent to a model. That is the same reproducibility risk the lm-eval authors warn about [5].

---

## 8. Braintrust

### Problem

Braintrust targets product teams that iterate on prompts, retrieval, and agents. It combines offline experiments, production logging, and online scoring around one data store.

### Architecture

The SDK reduces an eval to three parts. `data` supplies records with input and expected values. `task` is the function under test. `scores` is a list of scorers. Running `Eval("name", data=..., task=..., scores=[...])` through the `braintrust eval` CLI produces an *experiment* that can be compared with earlier ones [30][31]. Scorers come from the open-source **autoevals** library, which gives every scorer the same `(input, output, expected)` interface [31]:

- LLM-as-judge: Factuality, ClosedQA, Battle, Humor, Moderation, Security, Summarization, SQL, Translation.
- RAG metrics: context precision, recall, and relevancy, faithfulness, and answer relevancy and correctness.
- Heuristics: Levenshtein, exact match, numeric difference, JSON diff.
- Embedding similarity.

LLM classifiers map labeled choices to scores through `choice_scores`, can reason first with `use_cot=True`, and return a `Score(name, score in [0,1] or None, metadata.rationale)` [31]. This is the openai/evals `ModelBasedClassify` pattern [10], packaged as a library.

The self-hosted **data plane** shows the server-side architecture. The Terraform module for AWS provisions [32]:

```
            Braintrust control plane (UI, auth)            customer AWS account
                        |                       +--------------------------------+
   SDK / OTel spans ----+---------------------->|  API (ECS or Lambda): api,      |
                                                |  ingest, background services   |
                                                |      |            |            |
                                                |  Redis       Brainstore EC2    |
                                                |  (ElastiCache) fleets: reader, |
                                                |               fast-reader,     |
                                                |               writer, automation|
                                                |      |            |            |
                                                |  database (migrations via      |
                                                |  Lambda)     S3 buckets        |
                                                |  quarantine VPC: user code as  |
                                                |  Lambdas; sandbox VPC (no      |
                                                |  egress) for the Loop agent    |
                                                +--------------------------------+
```

Points to note [32]:

- **Brainstore** is Braintrust's own log and trace store. It runs as separate reader, fast-reader, writer, and automation-writer fleets, and reader fleets are rolled out before API updates.
- **User-defined functions**, such as custom scorers, run as Lambdas in a *quarantine VPC*. This applies the same isolation principle as Inspect's sandboxes, but to customer scoring code.
- An optional sandbox VPC with no outbound route hosts the Loop agent runtime. Model calls go through a private AI gateway or a hosted proxy.

The open-source **AI proxy** offers an OpenAI-compatible endpoint across providers. Setting a `seed` "activates the proxy's cache" [33]. That makes eval reruns cheap and deterministic where the provider allows it. The proxy can be self-hosted on Vercel, Cloudflare Workers, AWS Lambda, or Express [33].

### Failure modes and lessons

The `task` function can be any code, so reproducibility depends on the customer pinning prompts, models, and retrieval state. The platform records what happened but cannot freeze it. **Lesson:** the decisive investments are a log store built for this workload, isolation for customer scoring code, and a caching proxy. The scorer library itself, autoevals, is open source and a commodity.

---

## 9. LangSmith (LangChain)

### Architecture and concepts

LangSmith's evaluation model is defined in its concepts documentation [36]:

- **Datasets** contain **examples**, each with inputs, optional reference outputs (seen only by evaluators), and metadata. **Splits** may overlap: one example can belong to several. **Versions** are created automatically when examples change, and CI can pin a tagged version so that dataset edits do not break pipelines.
- An **experiment** is one application version run over a dataset, with outputs, evaluator feedback, and traces.
- **Evaluators** are workspace-level resources that return feedback as a key, a score or value, and a comment. The documented techniques are human review, code heuristics, LLM-as-judge, "decision models" (typed answers such as probabilities or ordered scores), and pairwise comparison.
- **Annotation queues** handle structured human review, either single-run (scored against rubric items) or pairwise (A/B). Annotated runs can be exported back to datasets.
- **Offline** evaluation runs over datasets with references. **Online** evaluation runs over production runs and threads without references, so it relies on reference-free checks.

The SDKs for Python and JavaScript are MIT-licensed [34]. Tracing uses `@traceable` and `wrap_openai`. The underlying data model is a `RunTree` of runs with types such as `llm`, `chain`, and `tool`, built with `create_child()`, `end()`, `post()`, and `patch()` [35]. A pytest plugin ships with the SDK and is enabled by default, so datasets and evaluations can be written as test cases [35].

### Deployment architecture

The BYOC (bring your own cloud) architecture [37] separates the two planes:

- **Control plane**, on LangChain's cloud: authentication, organization configuration, billing, API keys, and static frontend assets. It "does not hold any sensitive application data."
- **Data plane**, in the customer's AWS account: traces, prompts, datasets, evaluators, experiments, and secrets, on a private EKS cluster with Istio ingress and KEDA. The stores are RDS PostgreSQL, ElastiCache Redis, S3 for trace data, and ClickHouse with backups in S3.
- **Connectivity:** the two planes communicate over AWS PrivateLink, and "sensitive data does not leave your VPC." The browser loads the UI from LangChain's cloud but sends data requests directly to the customer's data plane.

The Helm chart lists further services [38]: frontend, backend, a platform backend whose internal gRPC listener serves "online evaluations on trajectories," queue workers, a sandboxed playground, an `ace-backend`, and "SmithDB" components (query, ingestion, compaction, and cluster manager). The chart pins ClickHouse 25.12 [38].

**Lesson:** LangSmith, Braintrust, and Langfuse have each arrived at the same split. Tenant metadata and the UI live in the vendor's cloud. High-volume traces live in a columnar or purpose-built analytical store, increasingly inside the customer's account. Evaluating whole agent trajectories online is heavy enough to need its own dedicated service.

---

## 10. promptfoo

### Problem and architecture

promptfoo is an open-source (MIT) CLI and library for "evaluating and red-teaming LLM apps." It has about 25.9k stars and says it is "used by OpenAI and Anthropic." It is now "part of OpenAI" and remains open source [39]. Its design is local-first: "LLM evals run 100% locally - your prompts never leave your machine" (apart from calls to the providers under test) [39].

Its central abstraction is a declarative matrix. `promptfooconfig.yaml` declares `prompts`, `providers`, and `tests` (with `vars` and `assert`). `promptfoo eval` runs **every prompt against every provider on every test case** and shows the results in a matrix view. List-valued `vars` multiply the number of runs [43].

```
  prompts[i] x providers[j] x tests[k](vars) --render--> provider call
       (disk cache ~/.promptfoo/cache, TTL 14d, 429 backoff)
                                   |
             transformResponse -> options.transform -> per-assert transform
                                   |
   asserts: deterministic | model-graded | trace/trajectory | custom code
                                   |
   weighted average -> threshold / all-must-pass -> derived metrics (mathjs)
                                   |
                 promptfoo view (matrix UI) / CI exit code / share
```

### Key techniques

- **A large assertion vocabulary** [40]:
  - Deterministic: `equals`, `contains`, `regex`, `is-json`, `is-sql`, schema checks, ROUGE, BLEU, METEOR, Levenshtein.
  - Operational: `latency`, `cost`, `perplexity`, `is-refusal`.
  - Custom code: JavaScript, Python, Ruby, webhook.
  - Trace and trajectory: `trace-span-count`, `trace-error-spans`, and `trajectory:*` checks on tool use, arguments, sequence, and step count.
  - Model-graded: `llm-rubric`, `g-eval`, `factuality`, `model-graded-closedqa`, RAG metrics, `select-best`.
  - Any assertion can be negated with a `not-` prefix.
- **Weighted scoring.** A test's score is the weighted average of its assertions, with weight 1.0 by default. A test-level `threshold` can pass a test even if some assertions fail. Without a threshold, every assertion with non-zero weight must pass. `weight: 0` records a metric without gating on it. `assert-set` groups give "A or B" logic, and derived metrics computed after the run, such as MAPE, can use `__count` [40].
- **Cache semantics suited to CI.** Successful responses are cached on disk for 14 days by default. Errors and empty responses are not cached, so failures retry. With repeats, each repeat index gets its own cache namespace [42].
- **Red-team architecture** [41]. **Plugins** generate adversarial inputs per vulnerability type, such as PII exposure, BOLA, or harmful content. **Strategies** deliver them, from encodings to multi-turn attacks. **Targets** are the systems under test. Probe generation runs in promptfoo's cloud service. The client runs each attack, asks for follow-ups with context from earlier attempts until a vulnerability is found or the attempt limit is reached, and submits a summary [41].

**Failure modes and lessons.** In the prompts x providers x tests matrix, the number of runs grows multiplicatively, so model-graded assertions can dominate cost. The docs offer `weight: 0` and assert-sets to control what gates a build. Note also that red-team attack generation is a cloud call, even though plain evals are local [41]. **Lesson:** for developer adoption, treating evals as declarative test files that return a CI exit code beat notebook-style workflows. promptfoo's growth to roughly 26k stars and its acquisition both support this.

---

## 11. Arize Phoenix

Phoenix is Arize's open-source, self-hostable observability and evaluation platform, licensed under the Elastic License 2.0. It has about 11.8k stars [44]. Its architecture is **OpenTelemetry-native**: traces are collected over OTel using the **OpenInference** semantic conventions, with instrumentors for frameworks such as LangGraph, LlamaIndex, and CrewAI and for model providers [44]. On top of the traces it offers:

- versioned datasets and experiments;
- a playground that can replay traced LLM calls;
- prompt management with version control and tags;
- code evaluators that run in sandboxes such as E2B, Daytona, Modal, or Docker;
- an in-product debugging agent (PXI);
- a remote MCP server at `/mcp`, so coding agents such as Claude Code and Cursor can query traces, datasets, and experiments [44].

Deployment options include pip, conda, or `uvx` with `phoenix serve`; Docker in root, nonroot, and debug image variants; a Helm chart; and cloud templates. Phoenix can run "fully air-gapped" and supports OAuth2, LDAP, role-based access, and trace retention policies [46].

The **phoenix-evals** library (v2+) is built around `create_classifier`, which takes a prompt template and labeled choices mapped to scores. It ships pre-built evaluators for faithfulness, correctness, retrieval relevance, refusal, conciseness, completeness, tool use, user friction, and PII detection, plus code metrics [45]. Results are `Score(name, score, label, explanation)`. The library claims "up to 20x speedup" from built-in concurrency and batching. That is the project's own figure, with no published benchmark [45]. The evaluators are themselves traced with OTel [45], so a judge's prompts and latencies can be inspected like any other span.

**Lesson:** building on OTel and OpenInference instead of a proprietary SDK makes Phoenix a sink any instrumented application can send to, and lowers lock-in. Exposing an MCP endpoint is an early sign of the next consumer of eval data: coding agents that read failing traces and propose fixes.

---

## 12. W&B Weave

Weave, from Weights & Biases, is a "toolkit for tracing and evaluating AI agents," licensed under Apache-2.0 [47]. Its architecture is built on the `@weave.op` decorator. Any decorated function becomes a traced, versioned operation, and once `weave.init()` is called, OpenAI calls inside an op nest beneath it [47]. Agent traces from the OpenAI Agents SDK, Claude Agent SDK, and Google ADK are captured automatically. Custom loops use the `start_conversation`, `start_turn`, `start_llm`, and `start_tool` context managers [47]. The repository's layout follows the architecture: `weave/trace` and `weave/trace_server` for capture and storage, and `weave/flow` for evaluations [47].

An `Evaluation` combines a dataset (a `Dataset` or a list of dicts) with scorers. Python scorers receive `output` plus any dataset columns they name. `evaluation.evaluate(model)` runs a `Model` subclass or an op over every row, and each call is one "evaluation run" [48]. `trials` repeats each example, with each trial scored independently. `preprocess_model_input` reshapes inputs for the model while scorers still see the original row. `EvaluationLogger` offers an imperative alternative for workflows that do not fit the declarative pattern [48].

**Lesson:** because evaluation runs are themselves traced ops, every score can be traced back to the exact model call and code version that produced it. The decorator does the versioning automatically.

---

## 13. Humanloop

Humanloop was an early application-centric platform built around **Files**: Prompts, Agents, and other versioned assets deployed to **Environments**, producing **Logs**, scored by **Evaluators** in **Evaluations** made of **Runs** over versioned **Datasets** of **Datapoints** [49][50]. Two design ideas stand out:

- **Prompts as files in version control.** `humanloop pull` syncs Prompts and Agents into a local `humanloop/` directory. With `use_local_files=True`, calls resolve paths such as `humanloop/examples/chat/basic.prompt`, so prompt changes go through code review [49].
- **Logs carrying cost and latency.** A log records inputs, output, provider latency, token counts, cost, and finish reason [49].

The SDK repositories were last updated in May 2025, and the company published a `humanloop-export` tool (last updated July 2025). It exports Files, Versions, Environments, Logs, Evaluations, Datasets, and Evaluator Logs to JSON and JSONL, resumes interrupted exports, and retries failed requests up to 5 times. Accounts with more than 1 million Logs need a custom S3 export [50][51]. None of the repositories reviewed here states the reason for the export tool. Readers should treat Humanloop as a historical design and check its current status independently.

**Lesson:** a platform-specific "eval asset graph" is a lock-in risk. An export path for every entity, including evaluator judgments, is part of a responsible design.

---

## 14. Langfuse

### Problem and architecture

Langfuse is an open-source LLM engineering platform. It is MIT-licensed except for its `ee` folders, has about 35.7k stars, and has been "part of ClickHouse" since January 2026 [52]. Its features cover tracing (SDKs, `@observe()`, and integrations), evaluations (LLM-as-judge, code evaluators, user feedback, manual labeling, and datasets for experiments), and prompt management [52]. It deploys through Docker Compose, Kubernetes via Helm (the preferred production setup), or Terraform templates for AWS, Azure, and GCP [52].

Its public v3 infrastructure write-up [54] is one of the most detailed accounts of scaling an eval and observability backend:

```
  SDK batch --> Web (auth + schema check only)
                  | write raw event --> S3 (durable, replayable)
                  | enqueue reference --> Redis queue
                  v
               Worker: tokenize, compute cost, merge events by object id
                  |
                  v
           ClickHouse (ReplacingMergeTree; wide, mostly immutable observations)
  Prompt / auth endpoints: separate deployment + Redis read-through cache
  SDKs: serve last-fetched prompt immediately, refresh in background
```

### Scale numbers and techniques (from [54])

- **Starting point.** v2 was a single container with one Postgres database. With spiky traffic in summer 2023, ingestion API response times reached up to 50 seconds. Prompt retrieval p95 reached 7 seconds during heavy ingestion, and dashboards slowed because row-oriented Postgres storage was costly to scan.
- **Async ingestion.** The API only checks authentication and body format. A worker handles tokenization, cost calculation, and writes. Redis held data in memory and had network limits, so raw events moved to S3, with only references in Redis. This allowed replays and reduced web CPU spent on serialization.
- **ClickHouse with insert-only updates.** Updates are expensive in ClickHouse, so they became inserts into a `ReplacingMergeTree` table, deduplicated in the background. Reading ClickHouse before each write is not reliably consistent at scale, so the worker caches acknowledged events and rebuilds each row from all events for that object ID. Traffic analysis showed about 90% of updates arrive within 10 seconds.
- **Prompt path isolation.** The prompt API moved to a Redis cache and a dedicated deployment with its own load balancer targets. SDKs serve the last-fetched prompt version immediately and refresh it in the background. Prompt API p99 fell from 7 seconds to 100 ms.
- **Query latency targets.** The targets were p99 of 1 second over the past week of data and 4 seconds for longer ranges. They were met with ordering keys that start with project ID and date, skip indexes, fewer joins, filters pushed down through CTEs, and `FINAL` or distinct aggregations for deduplication.

The self-hosting docs summarize the reasons [53]: spike tolerance, durability before database writes, rejecting unauthorized requests from cached API keys without touching the database, fast prompt delivery, join-free analytics, direct-to-S3 multimodal uploads, and non-blocking background migrations.

### Lessons

The team's stated lessons are that careful upfront schema design paid off (only one data import from Postgres was needed), that running reads side by side on both databases gave fast feedback, and that projects should be capped at about a month with more milestones [54]. For evaluation architects the broader lesson is this: **once online evaluation scores production traffic, the eval system becomes an ingestion system**, and the usual high-volume event pipeline (object-store buffer, queue, workers, OLAP store) applies.

---

## 15. Patronus AI

Patronus AI's public footprint is mostly *evaluation models and benchmarks* rather than open harness code. Its hosted evaluation API and SDK are not covered in the sources reviewed here. The pieces relevant to architecture are:

- **Lynx**, an open hallucination-detection judge model. The repository credits Rebecca Qian, Sunitha Ravi, and Bartosz Mielczarek and documents fine-tuning and serving with MosaicML CLI and vLLM [58]. The accompanying paper is "Lynx: An Open Source Hallucination Detection Model" (2024) [59]. Model sizes and benchmark results are reported there and were not independently checked for this chapter.
- **GLIDER** (arXiv 2412.14140), a judge model for rubric-based grading. It was trained in two phases, supervised fine-tuning followed by alignment with RLAIF data, using FSDP. It is released under CC-BY-NC-4.0 [57].
- **TRAIL** ("Trace Reasoning and Agentic Issue Localization", Deshpande et al., 2025, arXiv 2505.08638). It has 148 annotated agent traces containing 841 errors, classified as reasoning, execution, or planning errors, drawn from GAIA and SWE-Bench-style tasks. The best model reaches only 11% accuracy at localizing errors in traces [55].
- **FinanceBench** (Islam, Kannappan, Kiela, Qian, Scherrer, Vidgen, arXiv 2311.11944). It has 10,231 questions about public companies, with 150 open-sourced. In tests of 16 configurations, GPT-4-Turbo with retrieval answered incorrectly or refused 81% of questions [56].

**Architectural significance:** Patronus represents the "judge-as-a-model" design. Instead of prompting a frontier model as a grader, a smaller specialized evaluator is fine-tuned and served behind an API, which makes grading cheaper, faster, and more consistent. TRAIL measures the next frontier for such judges: scoring whole agent trajectories instead of single responses. Its 11% result shows that this remains largely unsolved.

---

## 16. Comparison

| System | Family | Primary subject | Config surface | Concurrency / scale mechanism | Result artifact | Storage / deployment | License / status (2026) |
|---|---|---|---|---|---|---|---|
| lm-eval-harness | model | weights or API; 3-method `LM` | YAML + Jinja + `!function` | length-sorted batches; vLLM/SGLang DP/TP; SQLite cache | aggregate JSON + `--log_samples` | local files, HF Hub | MIT; plugins added 2026-09 |
| HELM | model | model via Client/Executor | Python RunSpec functions | per-request cache (implementation not covered here) | per-instance states + leaderboard | `helm-server` static UI | Apache-2.0 (assumed); maintenance mode since 2026-06-01 |
| openai/evals | model or system | Completion Function | YAML + JSONL registry | thread pool (details not covered) | JSONL logs | Git LFS data; dashboard successor | MIT; contributions limited |
| simple-evals | model | sampler (OpenAI/Claude) | Python | minimal | tables | local | MIT; frozen since 2025-07 |
| Inspect | model + agent | Model / Agent + sandbox | Python `@task` | AnyIO async; per-provider `max_connections`; eval-set retry and resume | `.eval` binary log (zstd, attachments) | local/S3; `inspect view` | MIT; 200+ evals |
| lighteval | model | many backends; inspect-ai | Python task config | inspect-ai runtime or vLLM/Nanotron | per-sample Parquet; Hub | local/Hub | MIT (assumed); 1000+ tasks |
| OpenCompass | model | HF/LMDeploy/vLLM/API | MMEngine Python configs | Partitioner + Runner (local/cluster) | predictions/ results/ summary | local or cluster FS | Apache-2.0 (assumed); 100+ datasets |
| Braintrust | app | arbitrary `task` fn | Python/TS `Eval()` | hosted; caching AI proxy | experiments + logs | Brainstore; customer-VPC data plane | SDK OSS; platform commercial |
| LangSmith | app | traced target fn | SDK + UI; pytest | queue workers; SmithDB | experiments, run trees, feedback | Postgres, Redis, ClickHouse, S3; BYOC | SDK MIT; platform commercial |
| promptfoo | app | provider/target | `promptfooconfig.yaml` | prompts x providers x tests matrix; disk cache | matrix view; CI exit code | local-first; cloud red-team generation | MIT; part of OpenAI |
| Phoenix | app | OTel/OpenInference spans | SDK + UI | async evals (claimed up to 20x) | spans + eval annotations | self-host, Helm; air-gapped | ELv2 |
| Weave | app | `@weave.op` fns / Model | Python/TS `Evaluation` | trials; hosted trace server | traced eval runs | W&B cloud/server | Apache-2.0 |
| Humanloop | app | Prompt/Agent Files | UI + local `.prompt` files | hosted | Logs, Evaluations | hosted; export tool | SDKs inactive since 2025 |
| Langfuse | app | traces via SDK/OTel | SDK + UI | S3 + Redis queue + worker | traces, scores, datasets | Postgres + ClickHouse + Redis + S3 | MIT (core); part of ClickHouse |
| Patronus | judge models | (judges any output) | API | hosted judge models | scores + explanations | hosted | models: mixed open licenses |

"Assumed" marks a license not confirmed from the sources fetched for this chapter.

---

## 17. Cross-cutting design patterns

**1. The subject interface sets the ceiling of the harness.** lm-eval's three-method `LM` makes log-likelihood evaluation exact and backend-agnostic, but it cannot express tools [2]. openai/evals' completion function added "any system" but kept a single-string-in, list-out shape [11]. Inspect's `TaskState`, `generate`, and Agent protocol can represent multi-agent handoffs and sandboxes [15][20]. Application platforms accept any function and give up control of determinism in exchange.

**2. Prompt adaptation needs its own versioned layer.** Each system isolates the "instance to prompt" step: Jinja YAML [3], `AdapterSpec` [7], solver chains [15], `infer_cfg` [29], and prompt files under version control [49]. Mature systems save the rendered chat template or prompt next to the result [2][17].

**3. Caching is keyed on everything that can change the output.** Inspect's cache key covers model, base URL, messages, epoch, generate config, and tools [21]. lm-eval keys on tokenizer and chat template [2]. lighteval hashes the model configuration [27]. promptfoo namespaces by repeat and refuses to cache errors [42]. Braintrust caches only when a seed is set [33]. The shared failure is aliasing: a cache keyed on `gpt-x-latest` silently serves stale outputs [21].

**4. Separating generation from scoring.** lm-eval's multiple filter pipelines [3], OpenCompass's `predictions/` and `results/` split [29], Inspect's `inspect score` re-scoring [22], and lighteval's prediction cache [27] all let the scoring rule change without paying for inference again. This matters most for LLM-as-judge scorers, whose prompts change more often than the subject does.

**5. Logs are the product.** As evaluations become agentic, the per-sample transcript outweighs the aggregate number. Inspect built a compressed binary log format with attachment deduplication and streaming readers (about 8x smaller than JSON, plus about 10:1 from deduplication on agentic runs) [17]. Platforms built trace stores on ClickHouse [37][38][54] or a custom engine such as Brainstore [32].

**6. Isolation applies to the subject, the tools, and the scorers.** Inspect gives each sample its own sandbox with networking disabled by default [16]. Braintrust runs customer scorer code in a quarantine VPC [32]. Phoenix runs code evaluators in E2B, Daytona, or Modal sandboxes [44]. LangSmith runs its playground and `ace-backend` as sandboxed components [38].

**7. Split control and data planes for enterprise deployment.** LangSmith BYOC [37], the Braintrust data plane [32], and Langfuse self-hosting [53] all keep traces and datasets in the customer's cloud and centralize auth, billing, and UI assets.

**8. Closed-set, reason-first LLM judges.** The `choice_strings` + `choice_scores` + `cot_classify` pattern [10] appears again in autoevals [31], phoenix-evals [45], and promptfoo model-graded assertions [40]. Patronus goes further and fine-tunes dedicated judge models [57][58].

---

## Key takeaways for system designers

1. **Pick the family before the tool.** If the subject is a model and the question is a benchmark number, use a model-centric harness: Inspect for agentic or sandboxed work; lm-eval or lighteval for log-likelihood and leaderboard parity; OpenCompass for very large model x dataset sweeps on a cluster. If the subject is your application, use a tracing-first platform and treat benchmarks as a side input.
2. **Design the subject interface deliberately.** It is the hardest thing to change later. Probability-level access, a chat-level `generate`, and an agent-level `execute(state)` are different contracts. Supporting all three, as Inspect does through solvers and agents, costs complexity but avoids rewriting the harness later.
3. **Version and save the rendered prompt, not just the template.** Configs that use inheritance or includes (`include:` in lm-eval YAML, `read_base()` in OpenCompass) hide the final prompt. Log it per sample.
4. **Make caches complete and alias-safe.** Include model version (not an alias), base URL, the full message list, generation config, tools, epoch or repeat index, and the chat template. Never cache errors.
5. **Keep generation and scoring as separate, resumable stages.** Save raw outputs, and let scorers, filters, and judges be re-run offline. Use eval-set style retry with sample reuse for long runs.
6. **Plan log storage from day one.** Agentic transcripts are large and repetitive. Deduplicate messages, store attachments once, compress (zstd), and support header-only and streaming reads. For production traces, use an object-store buffer, a queue, workers, and a columnar store with insert-only updates (Langfuse's ReplacingMergeTree pattern).
7. **Isolate everything that executes model-influenced code**, including custom scorers. Default to no network egress, and remember that a custom container config can silently undo the default (the Inspect Compose caveat).
8. **Watch for silent truncation and partial runs.** Inspect's front-truncation of `exec()` output and lm-eval's `--limit` are documented examples of results that look valid but are not comparable.
9. **Isolate latency-sensitive paths from evaluation load.** Langfuse's prompt API went from 7 s to 100 ms p99 by moving prompt serving to a separate deployment with a cache and stale-while-revalidate SDKs.
10. **Expect consolidation and plan for exit.** In 2025-2026, promptfoo joined OpenAI, Langfuse joined ClickHouse, HELM entered maintenance mode, simple-evals froze, and Humanloop's SDKs went inactive and the company released a full-data export tool. Prefer open formats (OTel and OpenInference spans, `.eval` logs, JSONL datasets) and make sure every entity can be exported, including judge outputs.
11. **Treat judges as models under evaluation.** Use closed answer sets with reasoning before the choice, log rationales, and measure judge accuracy. TRAIL's 11% best-model accuracy on trace-level error localization is a warning against trusting automated trajectory judges without calibration.

---

## References

1. EleutherAI. "lm-evaluation-harness: A framework for few-shot evaluation of language models" (README, news through 2026). GitHub, 2020-2026. https://github.com/EleutherAI/lm-evaluation-harness
2. EleutherAI. "New Model Guide" (model_guide.md). GitHub, 2024-2026. https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/model_guide.md
3. EleutherAI. "Task Configuration Guide" (task_guide.md). GitHub, 2024-2026. https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md
4. EleutherAI. "User Guide / Command-line Interface" (interface.md). GitHub, 2024-2026. https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md
5. Biderman, S., Schoelkopf, H., Sutawika, L., et al. "Lessons from the Trenches on Reproducible Evaluation of Language Models." arXiv:2405.14782, 2024. https://arxiv.org/abs/2405.14782
6. Stanford CRFM. "HELM: Holistic Evaluation of Language Models" (README; maintenance-mode notice). GitHub, 2022-2026. https://github.com/stanford-crfm/helm
7. Stanford CRFM. "HELM developer guide: code structure" (docs/code.md). GitHub. https://github.com/stanford-crfm/helm/blob/main/docs/code.md
8. Liang, P., Bommasani, R., Lee, T., et al. "Holistic Evaluation of Language Models." arXiv:2211.09110, 2022 (TMLR 2023). https://arxiv.org/abs/2211.09110
9. OpenAI. "Evals" (README). GitHub, 2023-2025. https://github.com/openai/evals
10. OpenAI. "Eval templates" (docs/eval-templates.md). GitHub, 2023. https://github.com/openai/evals/blob/main/docs/eval-templates.md
11. OpenAI. "Completion Functions" (docs/completion-fns.md). GitHub, 2023. https://github.com/openai/evals/blob/main/docs/completion-fns.md
12. OpenAI. "simple-evals" (README). GitHub, 2024-2025. https://github.com/openai/simple-evals
13. UK AI Security Institute. "Inspect: A framework for large language model evaluations" (inspect_ai repository). GitHub, 2024-2026. https://github.com/UKGovernmentBEIS/inspect_ai
14. UK AI Security Institute. "Inspect documentation: Welcome / Getting started" (docs/index.qmd). GitHub. https://github.com/UKGovernmentBEIS/inspect_ai/blob/main/docs/index.qmd
15. UK AI Security Institute. "Inspect documentation: Solvers" (docs/solvers.qmd). GitHub. https://github.com/UKGovernmentBEIS/inspect_ai/blob/main/docs/solvers.qmd
16. UK AI Security Institute. "Inspect documentation: Sandboxing" (docs/sandboxing.qmd). GitHub. https://github.com/UKGovernmentBEIS/inspect_ai/blob/main/docs/sandboxing.qmd
17. UK AI Security Institute. "Inspect documentation: Eval Logs" (docs/eval-logs.qmd). GitHub. https://github.com/UKGovernmentBEIS/inspect_ai/blob/main/docs/eval-logs.qmd
18. UK AI Security Institute. "Inspect documentation: Parallelism" (docs/parallelism.qmd). GitHub. https://github.com/UKGovernmentBEIS/inspect_ai/blob/main/docs/parallelism.qmd
19. UK AI Security Institute. "Inspect documentation: Eval Sets" (docs/eval-sets.qmd). GitHub. https://github.com/UKGovernmentBEIS/inspect_ai/blob/main/docs/eval-sets.qmd
20. UK AI Security Institute. "Inspect documentation: Agents" (docs/agents.qmd). GitHub. https://github.com/UKGovernmentBEIS/inspect_ai/blob/main/docs/agents.qmd
21. UK AI Security Institute. "Inspect documentation: Caching" (docs/caching.qmd). GitHub. https://github.com/UKGovernmentBEIS/inspect_ai/blob/main/docs/caching.qmd
22. UK AI Security Institute. "Inspect documentation: Scorers" (docs/scorers.qmd). GitHub. https://github.com/UKGovernmentBEIS/inspect_ai/blob/main/docs/scorers.qmd
23. Generality Labs, UK AISI, et al. "Inspect Evals" (inspect_evals repository). GitHub, 2024-2026. https://github.com/UKGovernmentBEIS/inspect_evals
24. Hugging Face. "lighteval" (README). GitHub, 2024-2026. https://github.com/huggingface/lighteval
25. Hugging Face. "lighteval: Using inspect-ai as backend" (docs/source/inspect-ai.mdx). GitHub. https://github.com/huggingface/lighteval/blob/main/docs/source/inspect-ai.mdx
26. Hugging Face. "lighteval: Adding a custom task" (docs/source/adding-a-custom-task.mdx). GitHub. https://github.com/huggingface/lighteval/blob/main/docs/source/adding-a-custom-task.mdx
27. Hugging Face. "lighteval: Caching" (docs/source/caching.mdx). GitHub. https://github.com/huggingface/lighteval/blob/main/docs/source/caching.mdx
28. Shanghai AI Laboratory / OpenCompass. "OpenCompass" (README). GitHub, 2023-2026. https://github.com/open-compass/opencompass
29. OpenCompass. "Overview of the framework" (docs/en/user_guides/framework_overview.md). GitHub. https://github.com/open-compass/opencompass/blob/main/docs/en/user_guides/framework_overview.md
30. Braintrust. "braintrust-sdk" (README). GitHub, 2023-2026. https://github.com/braintrustdata/braintrust-sdk
31. Braintrust. "autoevals" (README). GitHub, 2023-2026. https://github.com/braintrustdata/autoevals
32. Braintrust. "terraform-aws-braintrust-data-plane" (README). GitHub, 2024-2026. https://github.com/braintrustdata/terraform-aws-braintrust-data-plane
33. Braintrust. "braintrust-proxy: AI proxy" (README). GitHub, 2023-2025. https://github.com/braintrustdata/braintrust-proxy
34. LangChain. "langsmith-sdk" (README). GitHub, 2023-2026. https://github.com/langchain-ai/langsmith-sdk
35. LangChain. "LangSmith Python SDK README" (python/README.md). GitHub. https://github.com/langchain-ai/langsmith-sdk/blob/main/python/README.md
36. LangChain. "LangSmith evaluation concepts" (docs source). GitHub, 2025-2026. https://github.com/langchain-ai/docs/blob/main/src/langsmith/evaluation-concepts.mdx
37. LangChain. "LangSmith BYOC architecture" (docs source). GitHub, 2025-2026. https://github.com/langchain-ai/docs/blob/main/src/langsmith/byoc-architecture.mdx
38. LangChain. "LangSmith Helm chart" (charts/langsmith README). GitHub, 2024-2026. https://github.com/langchain-ai/helm/tree/main/charts/langsmith
39. promptfoo. "promptfoo: Test your prompts, agents, and RAGs" (README; "now part of OpenAI" notice). GitHub, 2023-2026. https://github.com/promptfoo/promptfoo
40. promptfoo. "Assertions and metrics" (site/docs/configuration/expected-outputs/index.md). GitHub. https://github.com/promptfoo/promptfoo/blob/main/site/docs/configuration/expected-outputs/index.md
41. promptfoo. "Red team architecture" (site/docs/red-team/architecture.md). GitHub. https://github.com/promptfoo/promptfoo/blob/main/site/docs/red-team/architecture.md
42. promptfoo. "Caching" (site/docs/configuration/caching.md). GitHub. https://github.com/promptfoo/promptfoo/blob/main/site/docs/configuration/caching.md
43. promptfoo. "Configuration guide" (site/docs/configuration/guide.md). GitHub. https://github.com/promptfoo/promptfoo/blob/main/site/docs/configuration/guide.md
44. Arize AI. "Phoenix: AI observability and evaluation" (README). GitHub, 2023-2026. https://github.com/Arize-ai/phoenix
45. Arize AI. "phoenix-evals" (packages/phoenix-evals README). GitHub, 2025-2026. https://github.com/Arize-ai/phoenix/tree/main/packages/phoenix-evals
46. Arize AI. "Phoenix self-hosting" (docs/phoenix/self-hosting.mdx). GitHub. https://github.com/Arize-ai/phoenix/blob/main/docs/phoenix/self-hosting.mdx
47. Weights & Biases. "Weave" (README). GitHub, 2023-2026. https://github.com/wandb/weave
48. Weights & Biases. "Weave evaluations guide" (weave/guides/core-types/evaluations.mdx). GitHub. https://github.com/wandb/docs/blob/main/weave/guides/core-types/evaluations.mdx
49. Humanloop. "humanloop-python: Official Python SDK for the Humanloop API." GitHub, 2024-2025. https://github.com/humanloop/humanloop-python
50. Humanloop. "humanloop-export" (README). GitHub, 2025. https://github.com/humanloop/humanloop-export
51. Humanloop. "Humanloop GitHub organization." GitHub, accessed 2026. https://github.com/humanloop
52. Langfuse. "Langfuse: Open source LLM engineering platform" (README). GitHub, 2023-2026. https://github.com/langfuse/langfuse
53. Langfuse. "Self-hosting overview" (content/self-hosting/index.mdx, docs source). GitHub, 2025-2026. https://github.com/langfuse/langfuse-docs/blob/main/content/self-hosting/index.mdx
54. Langfuse. "Langfuse v3 infrastructure evolution" (blog post source, December 2024). GitHub. https://github.com/langfuse/langfuse-docs/blob/main/content/blog/2024-12-langfuse-v3-infrastructure-evolution.mdx
55. Deshpande, D., Gangal, V., Mehta, H., Krishnan, J., Kannappan, A., Qian, R. (Patronus AI). "TRAIL: Trace Reasoning and Agentic Issue Localization" (trail-benchmark repository; arXiv:2505.08638). GitHub, 2025. https://github.com/patronus-ai/trail-benchmark
56. Islam, P., Kannappan, A., Kiela, D., Qian, R., Scherrer, N., Vidgen, B. (Patronus AI). "FinanceBench: A New Benchmark for Financial Question Answering" (repository; arXiv:2311.11944). GitHub, 2023. https://github.com/patronus-ai/financebench
57. Patronus AI. "GLIDER" evaluation code and test sets (arXiv:2412.14140). GitHub, 2024. https://github.com/patronus-ai/glider
58. Qian, R., Ravi, S., Mielczarek, B. (Patronus AI). "Lynx-hallucination-detection" repository. GitHub, 2024. https://github.com/patronus-ai/Lynx-hallucination-detection
59. Ravi, S., Mielczarek, B., Kannappan, A., Kiela, D., Qian, R. "Lynx: An Open Source Hallucination Detection Model." arXiv:2407.08488, 2024. https://arxiv.org/abs/2407.08488
