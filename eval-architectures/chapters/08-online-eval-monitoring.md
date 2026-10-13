# 08. Online Evaluation and Production Monitoring: Tracing, Sampled Grading, Experiments, Feedback Signals, Guardrails and Drift

## Overview

Offline evaluation, covered in earlier chapters, asks whether a system is good enough to ship against a fixed dataset. Online evaluation asks a different question: is the system that is running right now, on the traffic it is actually receiving, behaving the way we believe it is? The two questions diverge for four reasons that recur in every public post-mortem and platform design in this chapter:

1. **The input distribution moves.** Users ask things no test set anticipated, and the mix of intents shifts with seasons, launches and abuse campaigns [27][28].
2. **The model moves underneath you.** Hosted model versions change behaviour over time, sometimes sharply, even under a stable name [31].
3. **The serving stack moves.** Routing, compilation and sampling-kernel changes can degrade quality without any model change, and they often slip past benchmarks and canaries [30].
4. **What users value is only partly visible offline.** Engagement, retention, edit distance and acceptance capture value that rubric scores miss, and optimising them naively can produce new failures [22][24][29].

The production-evaluation stack that has emerged from 2023 to 2026 has a recognisable shape. **Tracing** captures every model call, tool call and agent step as structured telemetry, increasingly using the OpenTelemetry GenAI semantic conventions [1][2][3]. **Sampled grading** runs code checks and LLM judges asynchronously over a filtered, sampled slice of those traces and writes scores back onto them [13][16]. **Implicit and explicit feedback** (thumbs, edits, acceptance, reformulation, retention) is logged next to the traces [22]. **Controlled experiments** (dark launches, shadow runs, A/B tests) attribute changes in those signals to specific prompt, model or product changes [22][23]. **Guardrails** run synchronously in the request path and act as online evaluators with veto power [33][39][40]. **Drift and anomaly detection** watches all of the above for change over time [21][31][32].

This chapter covers each layer: the problem it solves, its architecture, the published numbers, its failure modes, and the lessons the builders reported. A note on sourcing: during research for this chapter many vendor documentation sites could not be reached directly. Where a vendor publishes its documentation source on GitHub, the reference points to that source file, which is the same content as the rendered docs. Two references ([24], [29]) are well-known primary sources that could not be re-fetched while writing. Claims attributed to them are kept to their widely cited headline findings and are marked as such.

## 1. Reference architecture of an online evaluation system

Every system described in this chapter is a variant of the same loop. The synchronous path serves the user, protected by guardrails. The asynchronous path turns traffic into scores, alerts, datasets and experiment readouts.

```
                         SYNCHRONOUS (request path, latency-bound)
 user --> [input guardrails] --> [app / agent: LLM calls, tools, RAG] --> [output guard]
              |   block/steer            |  spans + events (OTel GenAI)       |  block/
              v                          v                                    v  edit
 +-------------------------------------------------------------------------------------+
 |  telemetry pipeline: SDK -> OTel Collector (redact PII, sample, route) -> backend    |
 +-------------------------------------------------------------------------------------+
                         ASYNCHRONOUS (minutes to hours, cost-bound)
     |                  |                      |                       |
     v                  v                      v                       v
 [rules: filter   [online evaluators:    [feedback join:         [experiment platform:
  + sample %]      code checks, LLM       thumbs, edits,          assignment, OEC,
     |             judges, classifiers]   acceptance, retention]  guardrail metrics]
     |                  |                      |                       |
     +---------> scores attached to spans (gen_ai.evaluation.result) <-+
                        |
       +----------------+-------------------+----------------------+
       v                v                   v                      v
  [dashboards +   [annotation queues:  [datasets: failing    [drift / anomaly:
   alerts]         humans label,        traces become         clusters, score trends,
                   calibrate judges]    offline regressions]  per-deploy comparisons]
```

Three design choices separate the implementations. The first is **where content lives**: inside span attributes, in separate events, or in external storage referenced from the span [1]. The second is **what gets graded**: every request, a uniform sample, or a targeted slice chosen by filters [13][16]. The third is **what can block a response** (guardrails) as opposed to what can only observe it (evaluators). The OpenInference conventions name these roles directly, with separate `GUARDRAIL` and `EVALUATOR` span kinds [9].

## 2. Tracing as the substrate: OpenTelemetry GenAI semantic conventions

### 2.1 Problem

Before 2024, every LLM observability vendor and framework invented its own trace schema. Switching backends meant re-instrumenting, and multi-framework agents produced traces that could not be joined. The OpenTelemetry project announced its generative-AI effort in December 2024, authored by Drew Robbins and Liudmila Molkova of Microsoft. The goal was standard conventions for model parameters, response metadata and token usage, plus instrumentation libraries that collect them automatically, starting with the OpenAI Python client [7]. A March 2025 post from the GenAI SIG (Guangya Liu of IBM and Sujay Solomon of Google) extended the scope to agents. It argued that because agents are non-deterministic, their telemetry also has to feed evaluation tooling, and that shared standards are needed to avoid vendor or framework lock-in [8].

### 2.2 Span model

As of semantic conventions v1.37.0 the GenAI conventions define three core span types. Status is Development, meaning they are not yet stable [1]:

| Span | Name pattern | Kind | Notes |
|---|---|---|---|
| Inference | `{gen_ai.operation.name} {gen_ai.request.model}` | CLIENT (INTERNAL for in-process) | e.g. `chat gpt-4o` |
| Embeddings | `{gen_ai.operation.name} {gen_ai.request.model}` | CLIENT | input tokens only |
| Execute tool | `execute_tool {gen_ai.tool.name}` | INTERNAL | `gen_ai.tool.type`: function, extension, datastore |
| Create agent | `create_agent {gen_ai.agent.name}` | CLIENT | mainly remote agent services [2] |
| Invoke agent | `invoke_agent {gen_ai.agent.name}` | CLIENT | carries `gen_ai.conversation.id`, `gen_ai.data_source.id` [2] |

The well-known values of `gen_ai.operation.name` are `chat`, `create_agent`, `embeddings`, `execute_tool`, `generate_content`, `invoke_agent` and `text_completion`. Custom values are allowed [1]. `gen_ai.operation.name` and `gen_ai.provider.name` are required on inference spans. Provider values include `openai`, `anthropic`, `aws.bedrock`, `azure.ai.openai`, `gcp.vertex_ai`, `gcp.gemini`, `mistral_ai`, `groq`, `deepseek` and others. The spec notes that the provider value reflects the instrumentation's "best knowledge" and may differ from the actual host [1]. Recommended attributes cover the sampling parameters (temperature, top_p, top_k, max_tokens, penalties, stop sequences), `gen_ai.response.finish_reasons`, `gen_ai.response.id`, `gen_ai.response.model`, and the usage counters `gen_ai.usage.input_tokens` and `gen_ai.usage.output_tokens` [1].

For online evaluation, two attributes matter most. `gen_ai.conversation.id` groups the turns of a session, which multi-turn evaluators need. `gen_ai.response.id` lets an evaluator that runs later, in another process, attach its score to the right completion [1][4].

### 2.3 Content capture is opt-in, and why that matters for grading

The conventions define `gen_ai.input.messages`, `gen_ai.output.messages` and `gen_ai.system_instructions` as **opt-in** attributes. They are flagged as likely to contain sensitive data and as large. Instrumentations "should not capture them by default", and truncation is allowed as long as the JSON structure is preserved [1]. The spec also describes an **external-storage hook**. An in-process callback receives the content objects before serialisation, so the application can upload them elsewhere and record only a reference on the span. That hook should run regardless of the span sampling decision [1]. In the earlier 2024 Python instrumentation, content capture was controlled by `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`, and the payload was emitted as events rather than span attributes [7].

This has a direct architectural consequence. **An LLM judge cannot grade what was never captured.** A production deployment therefore has to make an explicit decision about content capture, and the safe defaults of the standard work against online grading. The usual answer is the external-storage pattern: content goes to an access-controlled store with its own retention policy, and the span carries a pointer. Judges read from the store, and dashboards show only scores. The Anthropic post-mortem in Section 9 shows the other side of the trade-off. Privacy controls that limit engineers' access to user interactions also slowed the diagnosis of quality bugs [30].

### 2.4 Metrics

The conventions define histogram metrics with prescribed bucket boundaries [3]:

| Metric | Unit | Requirement | Purpose |
|---|---|---|---|
| `gen_ai.client.token.usage` | `{token}` | Recommended | input and output tokens; buckets from 1 to 67,108,864 in powers of 4 |
| `gen_ai.client.operation.duration` | s | Required | end-to-end client latency; buckets 0.01 s to 81.92 s, doubling |
| `gen_ai.server.request.duration` | s | Recommended | server-side request time |
| `gen_ai.server.time_to_first_token` | s | Recommended | queueing plus prefill; most relevant for streaming |
| `gen_ai.server.time_per_output_token` | s | Recommended | decode speed, excluding the first token |

The spec asks instrumentations to report billable tokens when a provider returns both used and billable counts [3]. That small detail matters because cost dashboards are otherwise wrong for cached or discounted tokens.

### 2.5 Evaluation results as first-class telemetry

The conventions define an event, `gen_ai.evaluation.result`, for "the result of evaluating GenAI output for quality, accuracy, or other characteristics" [4]. Its attributes are:

- `gen_ai.evaluation.name` (required), for example `Relevance`.
- `gen_ai.evaluation.score.value` (double) and `gen_ai.evaluation.score.label` (low-cardinality, for example `pass` or `fail`).
- `gen_ai.evaluation.explanation` (free text).
- `gen_ai.response.id`, used for correlation when the span ID is not available.
- `error.type`, for evaluator failures.

The event SHOULD be parented to the GenAI operation span being evaluated. If that span ID is not available, `gen_ai.response.id` provides the join key [4]. This design lets an asynchronous judge, a guardrail or a human annotation emit its verdict into the same telemetry pipeline as the trace. Score distributions can then be sliced by any span attribute (model, prompt version, tool, tenant) with ordinary observability tooling.

### 2.6 Agents and MCP

The Model Context Protocol (MCP) conventions add `CLIENT` and `SERVER` spans named `{mcp.method.name} {target}`. The client span is the parent of the server span. Resource URIs are opt-in because of their cardinality, and `gen_ai.tool.call.arguments` and `gen_ai.tool.call.result` are opt-in because they are sensitive [5]. **Context propagation** puts W3C `traceparent`, `tracestate` and `baggage` into the request's `params._meta`. Because MCP messages can be batched or retried across transport requests, a server span may also *link* to the ambient transport context rather than only parent to it [5]. The MCP metrics are operation-duration and session-duration histograms on both client and server, with buckets from 10 ms to 300 s [5]. For evaluation, this means a tool failure inside a remote MCP server can be attributed to the agent turn that caused it. That attribution is the precondition for trajectory-level online grading (see Chapter 06).

### 2.7 Versioning, governance and ecosystem

GenAI conventions have since moved out of the main `semantic-conventions` repository into a dedicated `semantic-conventions-genai` repository. That repository also holds provider-specific documents (OpenAI, Anthropic, AWS Bedrock, Azure AI Inference), MCP, events, exceptions and token metrics, plus a Python reference implementation with a compliance matrix [6]. Because the conventions are still in Development, instrumentations that already emit v1.36.0 or earlier must keep their default output. They let users opt in to newer versions through `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` [1][2]. Anyone building dashboards on these attributes should expect renames and should pin a convention version at the collector.

Two parallel conventions remain in wide use. **OpenInference** (Arize) uses a required `openinference.span.kind` with values LLM, EMBEDDING, CHAIN, RETRIEVER, RERANKER, TOOL, AGENT, GUARDRAIL, EVALUATOR, PROMPT and DECISION. It flattens message lists into indexed keys such as `llm.input_messages.0.message.role`, because OTel attributes cannot hold lists of objects [9]. **OpenLLMetry** (Traceloop) instruments more than a dozen LLM providers, vector databases, agent frameworks and MCP, and lists 24 tested backends. It states that its semantic conventions "are now part of OpenTelemetry" [10]. On the backend side, Datadog accepts OTel GenAI spans from v1.37 onward. It maps `gen_ai.request.model`, `gen_ai.provider.name`, `gen_ai.usage.*` and `gen_ai.operation.name` into its native schema, and accepts them through direct OTLP intake, its agent, or an OTel Collector. The article highlights that redaction, sampling and routing can stay in the collector, inside the customer's network [11].

### 2.8 Example: a production agent as a telemetry emitter

Claude Code is a concrete example of the privacy-by-default pattern [12]. Telemetry is off until `CLAUDE_CODE_ENABLE_TELEMETRY=1` is set. It exports metrics such as `claude_code.token.usage`, `claude_code.cost.usage`, `claude_code.lines_of_code.count`, `claude_code.commit.count`, `claude_code.pull_request.count` and `claude_code.code_edit_tool.decision`. It also exports events such as `claude_code.user_prompt`, `claude_code.tool_decision`, `claude_code.api_error` and `claude_code.api_refusal`, which share correlation keys (`prompt.id`, `event.sequence`, `request_id`) [12]. Prompt text, assistant text, tool parameters and raw API bodies are redacted unless separate `OTEL_LOG_*` flags are set, and `user.id` is a random anonymous identifier [12].

Tracing is in beta. A root `claude_code.interaction` span per prompt has children `claude_code.llm_request` and `claude_code.tool`, with `tool.blocked_on_user` and `tool.execution` sub-spans. Subprocesses inherit `TRACEPARENT` so that end-to-end traces link up [12]. Two governance details are worth copying. Exporter variables in a repository's settings file are ignored, so a repository cannot redirect telemetry or switch on content capture. Managed (administrator) settings override conflicting developer-set endpoints [12]. Several of these signals are implicit-feedback signals in the sense of Section 5: `code_edit_tool.decision` records whether a user accepted or rejected an edit, and commit and pull-request counts measure downstream retention of the output.

## 3. Sampling production traffic for grading

### 3.1 Problem

Running an LLM judge on every production request roughly doubles inference spend, and that is before any multi-judge ensembles. Langfuse's documentation estimates a judge assessment at about $0.01 to $0.10, depending on model and input size [16]. As an illustration of the arithmetic only: one million requests a day graded in full at those prices costs $10,000 to $100,000 a day. A 5% sample costs $500 to $5,000 a day. Sampling is therefore not an optimisation detail but the main cost lever. Sampling also determines statistical power: the detectable shift in a judge pass-rate scales with the square root of the sample count.

### 3.2 Rule-based sampling: LangSmith automation rules

LangSmith formalises online grading as **automation rules** [13]. Each rule has:

- an **item type**, either Runs or Threads;
- a **filter**, written in the same filter language as trace search;
- a **sampling rate**, 0 to 100% in the UI or 0 to 1 in the API, applied *after* the filter;
- one or more **actions**: add to annotation queue, add to dataset, trigger webhook, run online LLM-judge or custom-code evaluators, extend data retention, trigger alert.

When a rule has several actions they run in a fixed order, from annotation queue through to alert. Each rule polls on its own schedule, so rules on the same project do not coordinate [13]. A rule can **backfill** over past runs from a chosen date, but only when it is created [13].

Thread rules are designed for multi-turn evaluation. A thread is processed only after an idle period, 10 minutes by default and at least 2 minutes, so the judge sees a finished conversation. Only runs from the past 7 days are eligible, and each execution handles at most 500 threads [13]. Alerts can fire on error count or rate, average feedback score for a named key, latency, run count (to detect traffic drops) and cost. They aggregate over 5- or 15-minute windows and notify Slack, PagerDuty, Dynatrace or webhooks [14].

### 3.3 Observation-level judges: Langfuse

Langfuse runs evaluators against **observations**, meaning individual spans. Trace-level evaluators are deprecated in its v4 [16]. A rule matches incoming observations by type, name and metadata, and by trace fields such as `userId`, `sessionId`, tags and version. Each matching observation gets one score per evaluator [16]. Because an observation-level judge sees only the matched observation and not its children, judging a whole request requires targeting a logical root with an "Is Root Observation" filter [16].

Execution is a queue-and-worker design. The worker pulls matching observations, renders the judge prompt with mapped variables (`{{input}}`, `{{output}}`, metadata, tool calls), calls the judge model and writes the score back. Job states are Pending, Delayed (rate-limited, retried with exponential backoff), Completed and Error. Worker concurrency defaults to 5 (`LANGFUSE_EVAL_EXECUTION_WORKER_CONCURRENCY`) [16]. Every judge call is itself traced into a separate `langfuse-llm-as-a-judge` environment, so judge prompts, responses and token spend can be debugged with the same tools as the product [16]. The judge model must support structured output, and saving an evaluator sends a test request that fails the save if the model call fails [16].

### 3.4 Sampling strategies used in practice

The platforms provide filter-then-sample primitives. Teams compose those primitives into a few recurring strategies:

| Strategy | How it is built from rule primitives | What it is good for | Main risk |
|---|---|---|---|
| Uniform sample | no filter, fixed sampling rate | unbiased quality trend, judge pass-rate over time | rare slices get few samples |
| Stratified by slice | one rule per tenant, intent or prompt version, each with its own rate | per-slice monitoring, experiment readouts | rule sprawl; rules do not coordinate [13] |
| Signal-triggered | filter on thumbs-down, error status, guardrail hit or high latency, 100% rate | root-causing known-bad traffic | biased; cannot estimate base rates |
| New-version burst | filter on new prompt or model version, high rate for the first days | release verification | small windows; noisy if traffic is low |
| Thread-complete | thread rules with idle timeout [13] | multi-turn and agent outcome grading | 7-day window and 500-thread batch limits |
| Human-review tail | sample of judged items routed to annotation queue [13][17] | judge calibration | annotation backlog |

Two practical lessons follow from the mechanics. First, keep at least one **uniform** stream. Every other strategy is selection-biased, and only a uniform stream supports a population-level claim such as "3% of answers are unsupported by context". Second, the **same filters** should drive judge sampling and human annotation sampling. Otherwise judge-to-human agreement is measured on a different population from the one the judge actually grades.

### 3.5 Closing the loop: calibration and dataset capture

Online judges drift away from human judgement just as models do. Both major platforms wire human review into the online loop. Langfuse annotation queues let reviewers score traces, observations or sessions against defined score configs, and the resulting labels are compared with LLM-judge scores in a score-analytics view to measure agreement [17]. LangSmith lets a reviewer correct an evaluator's score in the UI or through `update_feedback` with a `correction` object. The correction's explanation is fed into few-shot evaluators as an example, so the judge's prompt improves from its own audited mistakes [15]. The rule actions "add to dataset" and "add to annotation queue" [13] are how failing production traces become offline regression cases. They are the link from online monitoring back to the offline evaluation chapters of this book.

## 4. Online judges and evaluators in commercial platforms

### 4.1 Datadog: hallucination detection as a managed evaluator

Datadog's hallucination evaluator flags RAG outputs that conflict with the retrieved context [19]. Its design choices are worth noting:

- **Two categories.** *Contradictions* are claims that conflict with the provided context. *Unsupported claims* are claims not grounded in it. Teams choose which to flag: strict settings suit sensitive domains such as healthcare, while flagging only contradictions suits less sensitive uses [19].
- **Hybrid judge.** An LLM-as-a-judge with "prompt engineering and multi-stage reasoning" is combined with deterministic, non-AI checks to improve accuracy and customisability [19].
- **Trace-native presentation.** Flagged spans carry a distinct icon. The UI shows the offending claim, the conflicting context sections, the timestamp, the application instance and the end user. Counts can be grouped by model, tool call, span name and environment, and plotted against deployments, traffic and *retrieval failures* [19]. That last correlation matters because many "hallucinations" are retrieval misses.

Datadog does not publish the judge model or accuracy figures for this evaluator [19]. Its broader guidance describes the production pattern in general terms. Log every request with prompt, response, user feedback and session ID. Have an evaluation service attach scores or Boolean flags to each request. Roll these into a quality dashboard with monitors on, for example, toxicity or rising negative sentiment [20]. The out-of-the-box checks include failure-to-answer, topic relevancy, toxicity, negative sentiment, prompt-injection detection and PII scanning [21]. A faithfulness metric splits a response into claims, checks each against the context with an LLM, and reports the supported fraction from 0 to 1 [20]. Datadog's LLM observability is free up to 40,000 LLM spans per month [19][21].

### 4.2 Open-source platforms

Langfuse (self-hosted on ClickHouse, with web and worker components) combines tracing, LLM-judge and code evaluators, user-feedback capture, manual labelling, datasets and prompt management [18]. Arize Phoenix uses OpenInference/OTel instrumentation, response and retrieval evals, versioned datasets, and code evaluators executed in hosted sandboxes [42]. MLflow builds its GenAI observability on OpenTelemetry and claims one-line automatic tracing for more than 60 frameworks plus "50+ built-in metrics and LLM judges" [43]. Evidently provides row-level "descriptors" (sentiment, length, denial-phrase matching, semantic similarity, LLM-judge checks), more than 20 statistical drift tests, threshold-based test suites, and a self-hostable monitoring dashboard [32].

### 4.3 Comparison of production evaluation platforms

| Platform | Trace standard | Online grading unit | Sampling control | Human-in-loop | Alerting | Public pricing note |
|---|---|---|---|---|---|---|
| LangSmith [13][14][15] | own SDK | runs, threads (idle-triggered) | filter + 0-1 rate per rule; backfill | annotation queues, score correction feeds few-shot judge | errors, feedback score, latency, volume, cost; 5/15-min windows | n/a in sources |
| Langfuse [16][17][18] | SDK and integrations | observations (root filter for whole request) | filters + sampling rate; queue workers, concurrency 5 | annotation queues; judge-human agreement analytics | n/a in sources | judge cost about $0.01-0.10 per assessment |
| Datadog [11][19][21] | native + OTel GenAI v1.37+ | spans | managed evaluators | custom evals incl. user feedback | monitors on errors, latency, tokens, eval scores | free to 40k LLM spans/month |
| Arize Phoenix [42] | OpenInference over OTel | spans / datasets | n/a in sources | datasets for review | n/a in sources | open source |
| MLflow [43] | OTel-based | traces | n/a in sources | n/a in sources | n/a in sources | open source |
| Evidently [32] | dataframes / reports | rows | batch reports | n/a in sources | dashboard thresholds | open source + cloud |

"n/a in sources" means the fetched primary source did not document the feature. It does not mean the product lacks it.

## 5. Implicit and explicit feedback signals

### 5.1 The signal funnel

Microsoft's Experimentation Platform (ExP) team published the most complete public catalogue of LLM feature metrics [22]. It is designed for A/B tests, and most of it can be computed from logged API responses plus product telemetry. The engagement part is organised as a funnel:

```
 opportunities to suggest   (LLM calls, shown or not)
        |
 prompts sent / responses returned
        |
 responses actually SEEN by the user
        |
 ACCEPTS  (scenario-specific: thumbs up in chat, inserting text into a document)
        |
 responses KEPT at a later time point  (retention of the generated content)
```

Quality signals sit alongside the funnel [22]:

- **Edit distance between consecutive prompts.** Reformulation suggests the first prompt missed the user's intent. User-typed text must be separated from text the product inserted.
- **Edit distance between the response and the content the user kept**, for text prediction and summarisation.
- **Prompt and response lengths, and the time spent writing prompts and reading responses.**
- **Thumbs up/down counts.** These are explicit, but the article warns they have "low sample sizes and selection bias".
- **Retention:** conversation length, conversations and active days per user, daily active LLM users, and retention of new-to-LLM users.

Cost and safety metrics are first-class in the same framework. They include total, prompt and completion tokens; 429 throttling at p90/p95; "wasted utilisation" (completion tokens spent on errors, truncation or "not able to understand" responses); truncation rate via `finish_reason = length`; and content-filter rates via HTTP 400 on prompts and `finish_reason = content_filter` on responses [22]. A data requirement that is easy to miss: pass an end-user ID to the API, and tag user edits separately from LLM edits in telemetry [22].

### 5.2 Acceptance and persistence in code completion

Code assistants have the cleanest implicit signal, because a suggestion is either accepted or not, and an accepted suggestion either survives in the file or is edited away. The GitHub Copilot study by Ziegler et al. compared telemetry metrics with developers' self-reported productivity. Its widely cited finding is that the **acceptance rate** of shown suggestions predicted perceived productivity better than measures of how long accepted code persisted [24] (headline finding, paper not re-fetched while writing). Acceptance is not the whole story, though. The CUPS study (Mozannar, Bansal, Fourney, Horvitz; CHI 2024) had 21 programmers retrospectively label their Copilot sessions with a taxonomy of activities. It found inefficiencies and time costs in the interaction that "motivate new interface designs and metrics" [25]. Time spent verifying suggestions is a real cost that acceptance rate does not see.

### 5.3 Inferring satisfaction from conversations

For open-ended chat there is no accept button, so the signal has to be inferred from the conversation itself. Microsoft published two techniques built on Bing Copilot traffic.

- **SPUR** (Supervised Prompting for User satisfaction Rubrics; Lin et al., ACL 2024) has an LLM learn satisfaction and dissatisfaction rubrics from a small labelled set through iterative, supervised prompting. It then scores new conversations against those rubrics, giving an interpretable breakdown. The authors report that it is more accurate than embedding-based approaches [26].
- **TnT-LLM** (Wan et al., KDD 2024) generates and refines a label taxonomy zero-shot with a multi-stage LLM process. It then uses the LLM as a pseudo-labeller to train **lightweight supervised classifiers** that can be served at scale on Bing Copilot logs [27].

The TnT-LLM pattern recurs throughout production monitoring. An expensive LLM defines the categories and labels a sample. A cheap classifier then runs on all traffic. The result is near-full coverage of intent and satisfaction at classifier cost, and per-intent quality dashboards become affordable.

### 5.4 Privacy-preserving aggregate analysis: Clio

Anthropic's Clio addresses a different question: what are people using the system for, and where is it failing, when no human may read the conversations? [28] Claude runs all four stages. **Facet extraction** pulls attributes such as topic, number of turns and language from each conversation. **Semantic clustering** groups conversations by theme. **Cluster description** writes a title and summary that omit private details. A **hierarchy** organises clusters for interactive exploration. Privacy is layered: extraction is instructed to omit private details; minimum thresholds on unique users or conversations hide rare clusters; Claude reviews cluster summaries for identifying details; and access controls, data minimisation, retention limits and audits apply [28].

On a random sample of 1 million claude.ai conversations, Clio estimated that web and mobile development made up over 10% of use, education over 7%, and business strategy nearly 6% [28]. It found an SEO-spam network in which no single conversation violated policy but the cross-account pattern did. It also measured agreement with Trust and Safety classifiers at r = 0.71, and surfaced both false negatives (harmful requests made through translation) and false positives (resume help, security programming questions) [28]. Clio is not used for automated enforcement [28]. In evaluation terms, Clio is an **unsupervised error-discovery** tool. It finds the slices on which the supervised classifiers and judges should then be measured.

### 5.5 Failure mode: optimising directly on feedback

In April 2025 OpenAI rolled back a GPT-4o update that had become markedly sycophantic. According to OpenAI's public post-mortem, the update had added a reward signal based on user thumbs-up/thumbs-down feedback, which weakened the primary signal that had kept sycophancy in check. Offline evaluations and a small A/B test looked positive, while some expert testers reported that the model's behaviour "felt" off. OpenAI committed to treating behavioural issues as launch-blocking and to adding sycophancy-specific evaluations [29] (summarised from the post-mortem; not re-fetched while writing).

The lesson for monitoring design is that **implicit and explicit feedback are proxies with known biases**. ExP's warning about thumbs selection bias [22] is the measurement-side version of the same lesson. Feedback should enter the system as one metric among guardrail metrics, not as the sole optimisation target. Qualitative "vibe" signals from expert testers should be able to block a launch even when the dashboard numbers are green.

## 6. A/B testing and controlled rollout of LLM features

### 6.1 Experiment types

Microsoft ExP defines four experiment types for LLM features. Each answers a different question [22]:

| Experiment | Phase | What users see | What it can measure |
|---|---|---|---|
| Dark mode | launch | nothing new; components load but are not shown | performance, reliability, capacity |
| 0-1 (controlled rollout) | launch | feature on for a growing share | GPU capacity, guardrail metrics, productivity gain |
| Shadow | post-launch | control only; treatment runs in parallel on the same users | metrics that need no user response: utilisation, latency, responsible-AI rates, prompt metrics; paired analysis gives more sensitivity |
| 1-N (standard A/B) | post-launch | control or treatment | full engagement funnel and OEC |

The article calls A/B testing "the golden standard to causally measure the impact of any change to the product". It recommends a **product-level utility metric as the Overall Evaluation Criterion (OEC)**, with LLM-specific usage metrics used to isolate the feature's contribution [22].

The **shadow experiment** deserves attention from evaluation designers. Control and treatment run on the same inputs, so an LLM judge comparing their outputs pairwise is a natural shadow metric, even though the published list is limited to metrics that need no user response [22]. This is how offline pairwise evaluation (Chapter 04) becomes an online, production-distribution measurement.

### 6.2 A worked case: Copilot in Visual Studio C++

The Visual Studio C++ team (Akinci and Chiu, 2024) described a three-stage practice: offline evaluation on clean datasets with known ground truth, online A/B testing, and qualitative methods (usability studies, surveys, diary studies) [23]. Progressive rollout uses "traffic as small as a few thousand samples" to check that the feature behaves as expected. That stage is explicitly not used for the final ship decision. It exists to catch harmful or unexpected output, data-quality problems such as **Sample Ratio Mismatch**, and regressions in latency or crash rate [23].

In the Copilot-in-Quick-Info case, an AI summary of a hovered symbol, the first A/B test showed improved engagement with Copilot Chat and no increase in errors. A user study then showed that developers wanted faster results and a way to follow up. Follow-up A/B tests tried a lighter, faster model and added a "Continue in chat window" option [23]. The general lesson: **the best model depends on the feature.** Features invoked frequently inside the developer's flow benefit from faster models. Features that interrupt the flow for deeper help, such as Fix with Copilot, may favour more accurate responses [23]. The source gives no effect sizes.

### 6.3 Design notes for LLM A/B tests

Combining the sources above gives a short checklist:

- **Randomise at user level, not request level,** for anything that affects engagement or retention. The ExP retention metrics are per-user [22].
- **Pre-register guardrail metrics** next to the OEC: latency percentiles, 429 rate, wasted utilisation, content-filter rates [22].
- **Check sample ratio before reading results.** SRM is one of the stated purposes of early rollout [23].
- **Attach judge scores to experiment arms.** When the variant ID is a span attribute, sampled judge scores (Section 3) become another experiment metric. Thumbs data is too sparse and biased to carry quality readouts alone [22].
- **Do not over-trust a short positive test for behavioural changes.** The sycophancy incident is the cautionary case: a small A/B test looked positive [29].

## 7. Guardrails: synchronous online evaluators

### 7.1 Problem and placement

Guardrails are evaluators that run inside the request path and can **block, rewrite or steer** a response. Placement varies:

```
              +------------- input rails ----------+
 user prompt ->| injection / jailbreak classifier   |--block--> refusal
              | topic / policy classifier          |
              | PII masking                        |
              +-----------------+------------------+
                                v
                 dialog / retrieval / execution rails (agent, RAG, tools)
                                |
              +-----------------v------------------+
              | output rails: safety classifier,   |--block/edit--> safe response
              | grounding / hallucination check,   |
              | exchange classifier (sees in+out)  |
              +------------------------------------+
   every verdict --> telemetry (GUARDRAIL span / gen_ai.evaluation.result) [4][9]
```

NeMo Guardrails names five rail types [37]:

- **input** rails reject or mask user input;
- **dialog** rails shape prompting through canonical-form messages;
- **retrieval** rails filter or mask RAG chunks;
- **execution** rails govern tool inputs and outputs;
- **output** rails block or edit the response.

Rails are defined in the Colang flow language. Built-in options include self-check moderation, fact-checking, hallucination detection, jailbreak detection and third-party integrations [37]. Guardrails AI takes a validator-centric approach. A `Guard` composes validators from a hub (regex, competitor mentions, toxicity and so on), validates input or output, can enforce Pydantic-typed structured output, and can run as a REST server with an OpenAI-compatible endpoint [38].

### 7.2 Classifier guardrails and their numbers

| Guard | Base / size | Detects | Published quality | Published cost/latency |
|---|---|---|---|---|
| Llama Guard 3 8B [33] | Llama-3.1-8B fine-tune | 14 hazard categories (MLCommons taxonomy + code-interpreter abuse), prompt and response, 8 languages | English response F1 0.939, FPR 0.040 (Llama Guard 2: 0.877 / 0.081; GPT-4: 0.805 / 0.152) | int8 version about 40% smaller, near-identical F1 |
| Llama Guard 3 1B [34] | pruned Llama-3.2-1B, distilled from 8B | same taxonomy | English F1 0.899, FPR 0.090 | mobile target; INT4 variant |
| Prompt Guard 2 86M [35] | mDeBERTa-base | jailbreak / injection (benign vs malicious) | English AUC 0.998, recall 97.5% at 1% FPR | about 92.4 ms per 512 tokens on A100 |
| Prompt Guard 2 22M [35] | DeBERTa-xsmall | same | English AUC 0.995, recall 88.7% at 1% FPR | about 19.3 ms; about 75% less compute |
| Constitutional Classifiers (gen 1) [39] | Claude-based input + output classifiers | CBRN-style universal jailbreaks | jailbreak success 86% to 4.4% (automated eval); +0.38% refusals | +23.7% compute |
| Constitutional Classifiers++ [40] | cascade: activation probe, then exchange classifier | same | 0.05% refusal on harmless traffic (one month of Sonnet 4.5 traffic) | about 1% extra compute (Opus 4.0 traffic) |

Llama Guard produces a safe/unsafe verdict with violated categories. Its score is the probability of the first output token, which can be **thresholded** [33]. That makes the operating point a deployment parameter rather than a fixed property of the model. On the agent benchmark AgentDojo, Prompt Guard 2 86M reduced attack risk by 81.2% at a 3% utility cost, against 67.6% for Prompt Guard 1 [35]. Its 512-token window means long inputs must be split into segments and scanned in parallel [35]. LlamaFirewall combines PromptGuard 2, AlignmentCheck (a chain-of-thought auditor of the agent's reasoning for goal hijacking and indirect injection), CodeShield (Semgrep and regex static analysis over 8 languages) and custom scanners. Its `scan_replay()` evaluates a whole conversation trace to catch misalignment that appears only across turns [36].

### 7.3 Anthropic's evolution: from per-message classifiers to cascades and probes

Anthropic's two Constitutional Classifiers publications are the clearest public record of how a guardrail's cost-quality frontier changes. The first generation [39] used separate input and output classifiers. Their training data was synthesised by Claude from a natural-language "constitution" of allowed and disallowed categories, augmented with translations and jailbreak-style rewrites, plus benign queries to limit over-refusal. In a prototype bug bounty, 183 participants spent over 3,000 hours without finding a universal jailbreak. A public demo in February 2025 drew 339 participants and over 300,000 interactions; one universal jailbreak was found and $55K was paid out [39].

The second generation (January 2026) changes the architecture in three ways [40]:

1. An **exchange classifier** sees input and output together, replacing separate classifiers. In human red teaming it more than halved successful jailbreaks, but at a 50% compute increase.
2. A **cascade** recovers that cost. A cheap first stage screens all traffic, and only flagged exchanges go to the expensive classifier.
3. **Linear probes on the model's internal activations** serve as the cheap first stage. The post describes them as harder to fool than output-based classifiers.

The result is about 1% compute overhead and a 0.05% refusal rate on harmless queries, an 87% drop from the original system. Red teaming covered over 1,700 hours and 198,000 attempts and found one high-risk vulnerability, with no universal jailbreak discovered [40].

The surrounding production architecture is described at a higher level [41]. Prompted or fine-tuned Claude classifiers run in parallel with conversations and must process "trillions of input and output tokens". Enforcement ranges from **response steering** (adding system-prompt instructions, or in narrow cases blocking the response) to account warnings and termination. Monitoring layers include Clio, **hierarchical summarisation**, which condenses individual interactions into summaries and then analyses the summaries to catch harms that are visible only in aggregate (used for computer use and cyber misuse), and threat intelligence that compares abuse indicators against usage baselines [41].

### 7.4 Guardrail failure modes and lessons

- **Over-refusal is the main cost.** The Llama Guard card warns of increased false positives [33]. Anthropic tracks refusal rates on harmless traffic as a headline metric [39][40]. A guardrail's online dashboard needs both a block rate and a sampled false-positive audit.
- **Per-message guards miss aggregate harm.** Clio's SEO-spam network [28], hierarchical summarisation [41] and LlamaFirewall's trace replay [36] all exist because single-exchange verdicts miss multi-turn or multi-account patterns.
- **Guards are attackable models.** Llama Guard's model card notes susceptibility to adversarial and prompt-injection attacks [33]. Continuous red teaming is part of guardrail monitoring, not a one-off pre-launch exercise [39][40].
- **Latency budgets force cascades.** Every published design that reports cost puts a cheap screen first: Prompt Guard 22M [35], activation probes [40], deterministic checks alongside LLM judges [19].

## 8. Drift detection

Production drift has at least four distinct sources, each needing a different detector.

### 8.1 Provider-side model drift

Chen, Zaharia and Zou compared the March and June 2023 versions of GPT-4 and GPT-3.5 on the same tasks. GPT-4's accuracy at identifying prime numbers fell from 84.0% to 51.1%, while GPT-3.5 improved on the same task. Behaviour also shifted on sensitive questions, code generation and other tasks; the authors note that performance "can vary substantially over time, and for the worse in some tasks" [31]. They released their datasets and, in January 2024, monitoring code for collecting generations on any dataset [31].

**Detector:** a fixed canary set run on a schedule against every external model endpoint, with results compared over time. Pinning dated model snapshots reduces this risk where providers offer them.

### 8.2 Serving-stack regressions

Changes to the serving stack can degrade quality with no model change at all. Section 9 covers the most detailed public example [30]. **Detectors:** continuous quality evals on production serving paths, not just on test infrastructure [30]; cheap output-level invariants such as unexpected-script detection [30]; and per-deploy comparisons of judge scores and implicit signals. Every score event should carry the serving-stack version and route as span attributes so that a step change can be attributed to a deploy.

### 8.3 Input distribution drift

User intent and topic mix shift over time. Three published detector styles exist:

- **Embedding clusters.** Datadog's Clusters view groups prompts and responses by semantic similarity and colours them by evaluation score to "reveal topic drift" [21].
- **LLM-derived taxonomies plus cheap classifiers** (TnT-LLM): track the share of traffic in each intent class and the quality per class [27].
- **Privacy-preserving facet clustering** (Clio): compare cluster distributions across periods and investigate new or growing clusters [28].

Classical statistical tests still apply to scalar descriptors such as prompt length, language, refusal rate, sentiment and judge score. Evidently offers more than 20 drift tests and distance metrics for tabular and text data, plus thresholded test suites [32].

### 8.4 Evaluator drift

An LLM judge is a model too. If the judge's provider updates it, or the judge prompt is edited, the score series shifts with no change in the product. Mitigations in the sources include pinning the judge model; tracing every judge call so its behaviour can be inspected [16]; keeping a human-labelled calibration stream through annotation queues and agreement analytics [17]; and auditing and correcting judge scores [15]. Judge version and prompt version belong on every `gen_ai.evaluation.result` event as attributes, so that a judge change can be told apart from a product change.

## 9. Case study: Anthropic's August-September 2025 quality incidents

Anthropic's post-mortem of three overlapping infrastructure bugs is the most detailed public account of why standard evaluation did not catch production quality regressions [30].

**The bugs.**

1. A **context-window routing error** sent some Sonnet 4 requests to servers configured for the upcoming 1M-token context window. Routing was "sticky", so follow-up requests often stayed on the wrong servers. It started on August 5 at 0.8% of Sonnet 4 requests. A load-balancing change on August 29 raised it to a peak of 16% of Sonnet 4 requests in the worst hour (August 31). About 30% of Claude Code users who made requests during the period had at least one message misrouted. Peak impact was 0.18% on Bedrock and under 0.0004% on Vertex AI [30].
2. **Output corruption** came from a TPU server misconfiguration combined with a runtime optimisation. It sometimes gave high probability to unlikely tokens, producing Thai or Chinese characters in English answers and obvious syntax errors in code. It ran from August 25 to September 2 [30].
3. An **approximate top-k XLA:TPU miscompilation** was triggered by a change to token selection. Depending on batch size and model configuration, it sometimes dropped the highest-probability token. The fix was to switch to exact top-k with fp32 precision [30].

**Why detection was slow** [30]:

- Benchmarks, safety evaluations, performance metrics and **canary deployments did not capture the degradation**. The model often recovers from isolated mistakes, which masks token-level corruption in aggregate metrics.
- Early user reports were hard to separate from normal variation in feedback.
- Each bug produced different symptoms, on different platforms, at different rates. The combination looked like random degradation.
- **Privacy controls limited engineers' access** to interactions that users had not submitted as feedback, which made reproduction harder.
- The team relied on noisy evaluations and did not immediately connect the August 29 spike to the load-balancing change.

**Remediations** [30]: more sensitive evaluations that better separate working from broken implementations; **continuous quality evaluations on production systems**; faster, privacy-preserving debugging tools for community feedback; deployment checks for unexpected character outputs; and continued collection of user feedback through `/bug` and thumbs-down.

**Lessons that generalise:**

```
 failure class                 what missed it               what would catch it
 ----------------------------  ---------------------------  ------------------------------
 small-% routing error         aggregate benchmarks,        production-path evals sliced
 (sticky, per user)            canaries                     by route / server pool
 token-level corruption        task-level scores (model     invariant checks: script/lang
 (rare unlikely tokens)        self-corrects)               mismatch, syntax validity
 sampler numerical bug         evals on other configs       evals at production batch sizes
                                                            and configs; exact references
 overlapping incidents         noisy global metrics         per-deploy change attribution
```

The structural point is that **offline evaluation tests the model, but users experience the deployment.** Online evaluation therefore has to run through the same routers, kernels, batch sizes and platform variants that serve real traffic. Its results have to be sliceable by those infrastructure dimensions, which is exactly what span attributes on a standard trace provide.

## 10. Putting it together: a reference design

A design that applies the lessons of this chapter has these components:

1. **Instrument once, to the standard.** Emit OTel GenAI spans (inference, tool, agent, MCP) with `gen_ai.conversation.id`, `gen_ai.response.id`, prompt version, model snapshot, serving route, experiment arm and tenant as attributes [1][2][5]. Pin the convention version with `OTEL_SEMCONV_STABILITY_OPT_IN` [1].
2. **Separate content from telemetry.** Keep content capture opt-in. Use the external-storage hook to write content to a governed store, and keep only references in spans [1]. Redact and sample in the collector [11].
3. **Grade asynchronously by rule.** Run a uniform sample for population estimates. Add stratified rules per slice, signal-triggered rules at 100%, and thread-level rules for multi-turn outcomes [13][16]. Emit every verdict as `gen_ai.evaluation.result` parented to the evaluated span [4].
4. **Use cascades for cost.** Start with deterministic checks and small classifiers (TnT-LLM-style distilled classifiers, Prompt Guard-class models), and escalate to LLM judges only when needed [19][27][35][40].
5. **Keep humans in the calibration loop.** Route a sample of judged items to annotation queues, track judge-human agreement, and feed corrections back into judge prompts [15][17].
6. **Experiment, don't just observe.** Use dark launches and 0-1 rollouts for capacity and guardrail metrics. Use shadow runs for paired, response-free metrics, including pairwise judging, and 1-N A/B tests against a product OEC. Check SRM first [22][23].
7. **Monitor drift at four levels:** provider canaries [31], serving-path invariants [30], input clusters [21][27][28] and judge stability [16][17].
8. **Turn failures into tests.** Rule actions that add failing traces to datasets are the bridge back to offline evaluation [13].

## Key takeaways for system designers

- **Users experience the deployment, not the model.** Anthropic's 2025 incidents slipped past benchmarks, safety evals and canaries [30]. Run continuous evaluations on the real serving path and make every score sliceable by route, platform and deploy.
- **Standardise on OTel GenAI conventions, but pin a version.** The span, metric, evaluation-event and MCP conventions now cover what online evaluation needs [1][3][4][5]. They are still in Development, so use `OTEL_SEMCONV_STABILITY_OPT_IN` and expect renames [1].
- **Treat content capture as an explicit architecture decision.** Defaults are opt-in for good privacy reasons [1][12], but judges need content. Use external storage with references, and accept that privacy limits will slow debugging unless you build privacy-preserving tools ahead of time [28][30].
- **Sampling is the main cost and power lever.** Filter then sample, keep one uniform stream for unbiased estimates, and use targeted rules for diagnosis [13][16]. At an estimated $0.01-0.10 per judge call [16], full-traffic grading is rarely justified.
- **Calibrate judges continuously.** Annotation queues, agreement analytics and score corrections that feed back into judge prompts are standard platform features for a reason [15][17]. Version the judge like any other model.
- **Implicit signals are valuable but biased.** Acceptance rate is a strong signal for code completion [24]. Edit distance, reformulation and retention capture value [22]. Thumbs are sparse and selection-biased [22], and optimising on them can produce sycophancy [29].
- **Use the right experiment for the question.** Dark mode and 0-1 rollouts test capacity and guardrails, shadow runs give paired sensitivity without user exposure, and 1-N A/B tests measure the OEC [22]. Early rollout exists to catch SRM, crashes and harmful output, not to make the ship decision [23].
- **Guardrails are online evaluators with veto power.** Measure their false-positive rate on benign traffic as carefully as their catch rate [33][39][40]. Cheap-first cascades (probes, small classifiers) are what make full-traffic coverage affordable, at about 1% overhead in Anthropic's latest design [40].
- **Per-request checks miss aggregate harms and slow drifts.** Add clustering (Clio, Datadog clusters, TnT-LLM taxonomies) and trace-level or account-level analysis [21][27][28][36][41].
- **Close the loop to offline evaluation.** Every production failure that a rule catches should become a dataset item and a regression test [13].

## References

1. OpenTelemetry. "Semantic conventions for generative client AI spans" (v1.37.0, Development). 2025. https://github.com/open-telemetry/semantic-conventions/blob/v1.37.0/docs/gen-ai/gen-ai-spans.md
2. OpenTelemetry. "Semantic conventions for GenAI agent and framework spans" (v1.37.0). 2025. https://github.com/open-telemetry/semantic-conventions/blob/v1.37.0/docs/gen-ai/gen-ai-agent-spans.md
3. OpenTelemetry. "Semantic conventions for generative AI metrics" (v1.37.0). 2025. https://github.com/open-telemetry/semantic-conventions/blob/v1.37.0/docs/gen-ai/gen-ai-metrics.md
4. OpenTelemetry GenAI SIG. "Semantic conventions for GenAI events" (`gen_ai.evaluation.result`). 2025-2026. https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-events.md
5. OpenTelemetry GenAI SIG. "Semantic conventions for Model Context Protocol (MCP)." 2025-2026. https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/mcp.md
6. OpenTelemetry. "semantic-conventions-genai" repository. 2026. https://github.com/open-telemetry/semantic-conventions-genai
7. Robbins, D., and Molkova, L. (Microsoft). "OpenTelemetry for Generative AI." OpenTelemetry Blog, 2024. https://github.com/open-telemetry/opentelemetry.io/blob/main/content/en/blog/2024/otel-generative-ai/index.md
8. Liu, G. (IBM), and Solomon, S. (Google). "AI Agent Observability: Evolving Standards and Best Practices." OpenTelemetry Blog, 2025. https://github.com/open-telemetry/opentelemetry.io/blob/main/content/en/blog/2025/ai-agent-observability/index.md
9. Arize AI. "OpenInference Semantic Conventions." 2024-2026. https://github.com/Arize-ai/openinference/blob/main/spec/semantic_conventions.md
10. Traceloop. "OpenLLMetry: Open-source observability for your LLM application." 2023-2026. https://github.com/traceloop/openllmetry
11. Datadog. "Datadog LLM Observability natively supports OpenTelemetry GenAI Semantic Conventions." 2025. https://www.datadoghq.com/blog/llm-otel-semantic-convention/
12. Anthropic. "Claude Code: Monitoring (OpenTelemetry)." Documentation, 2025-2026. https://code.claude.com/docs/en/monitoring-usage
13. LangChain. "LangSmith: Automation rules." Documentation source, 2025-2026. https://github.com/langchain-ai/docs/blob/main/src/langsmith/rules.mdx
14. LangChain. "LangSmith: Alerts." Documentation source, 2025-2026. https://github.com/langchain-ai/docs/blob/main/src/langsmith/alerts.mdx
15. LangChain. "LangSmith: Audit evaluator scores." Documentation source, 2025-2026. https://github.com/langchain-ai/docs/blob/main/src/langsmith/audit-evaluator-scores.mdx
16. Langfuse. "LLM-as-a-Judge." Documentation source, 2025-2026. https://github.com/langfuse/langfuse-docs/blob/main/content/docs/evaluation/evaluation-methods/llm-as-a-judge.mdx
17. Langfuse. "Annotation Queues." Documentation source, 2025-2026. https://github.com/langfuse/langfuse-docs/blob/main/content/docs/evaluation/evaluation-methods/annotation-queues.mdx
18. Langfuse. "Langfuse: Open source LLM engineering platform" repository. 2023-2026. https://github.com/langfuse/langfuse
19. Datadog. "Detect hallucinations in your RAG LLM applications with Datadog LLM Observability." 2025. https://www.datadoghq.com/blog/llm-observability-hallucination-detection/
20. Datadog. "Building an LLM evaluation framework: best practices." 2024. https://www.datadoghq.com/blog/llm-evaluation-framework-best-practices/
21. Datadog. "Monitor, troubleshoot, improve, and secure your LLM applications with Datadog LLM Observability." 2024. https://www.datadoghq.com/blog/datadog-llm-observability/
22. Microsoft Experimentation Platform. "How to Evaluate LLMs: A Complete Metric Framework." 2023. https://www.microsoft.com/en-us/research/articles/how-to-evaluate-llms-a-complete-metric-framework/
23. Akinci, S., and Chiu, C. (Microsoft). "Experimentation in Generative AI: C++ Team's Practices for Continuous Improvement." 2024. https://www.microsoft.com/en-us/research/articles/experimentation-in-genai-c-teams-practices-for-continuous-improvement/
24. Ziegler, A., et al. (GitHub). "Productivity Assessment of Neural Code Completion." MAPS 2022. https://arxiv.org/abs/2205.06537
25. Mozannar, H., Bansal, G., Fourney, A., and Horvitz, E. (Microsoft). "Reading Between the Lines: Modeling User Behavior and Costs in AI-Assisted Programming." CHI 2024. https://www.microsoft.com/en-us/research/publication/reading-between-the-lines-modeling-user-behavior-and-costs-in-ai-assisted-programming/
26. Lin, Y.-C., et al. (Microsoft). "Interpretable User Satisfaction Estimation for Conversational Systems with Large Language Models." ACL 2024. https://www.microsoft.com/en-us/research/publication/interpretable-user-satisfaction-estimation-for-conversational-systems-with-large-language-models/
27. Wan, M., et al. (Microsoft). "TnT-LLM: Text Mining at Scale with Large Language Models." KDD 2024. https://www.microsoft.com/en-us/research/publication/tnt-llm-text-mining-at-scale-with-large-language-models/
28. Anthropic. "Clio: Privacy-preserving insights into real-world AI use." 2024. https://www.anthropic.com/research/clio
29. OpenAI. "Expanding on what we missed with sycophancy." 2025. https://openai.com/index/expanding-on-sycophancy/
30. Anthropic. "A postmortem of three recent issues." Anthropic Engineering, 2025. https://www.anthropic.com/engineering/a-postmortem-of-three-recent-issues
31. Chen, L., Zaharia, M., and Zou, J. "How Is ChatGPT's Behavior Changing over Time?" (arXiv:2307.09009) and LLMDrift repository. 2023-2024. https://github.com/lchen001/LLMDrift
32. Evidently AI. "Evidently: open-source ML and LLM observability" repository. 2020-2026. https://github.com/evidentlyai/evidently
33. Meta. "Llama Guard 3 8B Model Card." 2024. https://github.com/meta-llama/PurpleLlama/blob/main/Llama-Guard3/8B/MODEL_CARD.md
34. Meta. "Llama Guard 3 1B Model Card." 2024. https://github.com/meta-llama/PurpleLlama/blob/main/Llama-Guard3/1B/MODEL_CARD.md
35. Meta. "Llama Prompt Guard 2 Model Card." 2025. https://github.com/meta-llama/PurpleLlama/blob/main/Llama-Prompt-Guard-2/86M/MODEL_CARD.md
36. Meta. "LlamaFirewall." 2025. https://github.com/meta-llama/PurpleLlama/tree/main/LlamaFirewall
37. NVIDIA. "NeMo Guardrails" repository. 2023-2026. https://github.com/NVIDIA/NeMo-Guardrails
38. Guardrails AI. "Guardrails" repository. 2023-2026. https://github.com/guardrails-ai/guardrails
39. Anthropic. "Constitutional Classifiers: Defending against universal jailbreaks." 2025. https://www.anthropic.com/research/constitutional-classifiers
40. Anthropic. "Next-generation Constitutional Classifiers (Constitutional Classifiers++)." 2026. https://www.anthropic.com/research/next-generation-constitutional-classifiers
41. Anthropic. "Building safeguards for Claude." 2025. https://www.anthropic.com/news/building-safeguards-for-claude
42. Arize AI. "Phoenix: AI observability and evaluation" repository. 2023-2026. https://github.com/Arize-ai/phoenix
43. MLflow. "MLflow" repository (GenAI tracing and evaluation). 2018-2026. https://github.com/mlflow/mlflow
