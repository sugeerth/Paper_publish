# 10. ML Platform and MLOps System Design: Data Flywheel, Model Lifecycle, Capacity and Rollout (with Diagrams)

> **How to read this chapter.** The rest of the pack raises MLOps only as follow-up probes. This chapter is the dedicated design chapter for it. It covers training-data pipelines and labelling operations, feature and embedding stores, model registry and versioning (AI Center-style), CI/CD for models and prompts, drift and data-quality monitoring for document and agent models, per-tenant fine-tuning and adapter management, GPU capacity planning, on-prem, air-gapped and multi-region rollout, and rollback and incident response. Each section opens with a plain-text diagram, then gives numbered questions with follow-up probes. It has questions only, with no answers. The diagrams are whiteboard scaffolds that show the parts of the system the questions refer to. They are not model answers. They are plain text so they render the same in Markdown and in a PDF export.
>
> **Tagging system.** Every question carries one of these tags:
> - **[Reported] [n]**: a candidate report or interview-experience post links this question to UiPath. The number points to the References list.
> - **[Likely]**: an inference from the role (Principal Applied Scientist), the level and UiPath's products: AI Center-style model hosting, Document Understanding (DU Modern), IXP and Communications Mining, Context Grounding, the LLM Gateway and AI Trust Layer, agents and Action Center, and Automation Suite for on-prem customers. No candidate has reported it.
> - **[Likely] [n]**: as [Likely], but source [n] confirms the *premise* of the question, such as a product behaviour (for example, published project tags and versions, or AI Trust Layer model policy). It does **not** mean a candidate was asked it.
> - Indented sub-bullets are follow-up probes an interviewer may use to push a strong first answer toward the principal bar.
>
> **"Tag" in this chapter.** UiPath uses "tag" for a published version alias of a Document Understanding or IXP project. For non-pretrained projects a caller must pass exactly one of `version` or `tag` (passing both is rejected), and an unknown tag raises an error that lists the project's available tags [2]. Several questions below test how that tag-as-alias model works for release, rollback and per-tenant pinning. PDF *structure* tags (accessibility tagging) are covered in chapter 08.
>
> **Research limits (please read).** Neither the first pass nor the adversarial review (both 14 October 2026) could run a web search: the workflow's shared budget of 200 searches was already used up. Direct fetches of glassdoor.com, interviewquery.com and docs.uipath.com failed DNS lookup, and earlier passes found teamblind.com, reddit.com, leetcode.com, prepfully.com and uipath.com unreachable. Only GitHub-hosted files could be read. The chapter therefore rests on UiPath's public SDK source and documentation, UiPath's IXP CLI and a general ML system design guide [1]-[15]. Those sources confirm product behaviour and general round format, not interview content. So **no question below is tagged [Reported]**, and the chapter does not invent reports. A later run with search budget should check Glassdoor, Blind, LeetCode Discuss, 1point3acres, Prepfully and Interview Query for UiPath-specific MLOps prompts, and upgrade tags where sources support them.
>
> **Review pass (14 October 2026).** An adversarial review re-fetched all 7 original GitHub sources and confirmed each claim. It corrected the `version`/`tag` wording (passing both is rejected, rather than the tag being ignored), and noted that searching an index during ingestion is opt-in. It removed two references that could not be reached or verified (the docs.uipath.com portal and the UiPath careers page; no question cited them) and two duplicate questions: right-to-deletion for adapters (covered by the consent-revocation probe in section A) and funding evaluation capacity (covered by framing question 5). It reworded one probe that duplicated the vendor-drift monitoring question. It then read 8 more official UiPath GitHub documents [8]-[15] and added 28 principal-level [Likely] questions on gaps: escalation outcomes as labels, redacted traces, heterogeneous tenant schemas, ephemeral indexes and index tiering, release artefacts and dependency locking, identity and lineage, semantic versioning for models, trajectory gates, CI credentials, guardrail and ring-based rollout, cost gates, eval contamination, trace sampling, guardrail-hit monitoring, customer-owned taxonomies, UI drift, adapter poisoning, prefix caching, admission control, per-environment resolution, telemetry opt-out bias, CVE response, rollback without data rollback, build-or-buy and agents that do MLOps.

## Format notes for this round

| Item | Notes | Status |
|---|---|---|
| Where it appears | Probably not as its own round. More likely it is the second half of the ML system design round ("now take it to production"), or a deep dive during research-presentation Q&A ("how did you deploy, monitor and retrain it?"). See chapter 01 | Inferred |
| Duration | 45-60 minutes when it is a full design round; 15-25 minutes when it is the production section of another round | Inferred |
| Interviewer | Principal or staff scientist, ML platform architect, or an engineering director who owns the AI platform | Inferred |
| General round shape | Open-ended, no single right answer. A general guide puts serving, online testing (A/B, shadow, canary) and scaling, monitoring and retraining as the last three steps of a nine-step flow. For 2026 GenAI rounds, it says interviewers weigh cost, latency, guardrails and monitoring more than the architecture diagram, and that "evaluation methodology is the new system design" [1] | Sourced (general, not UiPath-specific) |
| Principal bar | Owns the lifecycle, not just the model. Sets SLOs and release gates. Designs for thousands of tenants with different data, regions and deployment models (cloud, dedicated, on-prem, air-gapped). Puts numbers on GPU, cost and blast radius. Makes the data flywheel (human validation leading to retraining) a deliberate, governed system. Names who owns each failure | Inferred |
| Common failure modes | One global model with no tenant story. "Retrain weekly" with no trigger or gate. Monitoring only latency and errors. No rollback plan for prompts or adapters. Ignoring air-gapped customers. No cost model | Inferred |

### Canvas: the end-to-end model lifecycle (whiteboard scaffold)

```
                         ┌────────────────────────── CONTROL PLANE ──────────────────────────┐
                         │  Model registry  │  Prompt / config registry  │  Eval-set registry │
                         │  (versions, tags,│  (templates, tool schemas, │  (golden sets,     │
                         │   lineage, cards)│   guardrail policy)        │   slices, judges)  │
                         └────────┬─────────┴─────────────┬──────────────┴─────────┬──────────┘
                                  │ promote / pin / roll back                      │ gate
   ┌──────────────┐   ┌──────────▼─────────┐   ┌──────────▼──────────┐   ┌─────────▼─────────┐
   │ DATA SOURCES │──>│ DATA PIPELINE      │──>│ TRAINING / TUNING   │──>│ EVALUATION + CI   │
   │ docs, emails,│   │ ingest, PII redact,│   │ base models, per-   │   │ offline metrics,  │
   │ screens,     │   │ dedupe, split,     │   │ tenant adapters,    │   │ slices, regression│
   │ agent traces │   │ label queue        │   │ GPU scheduler       │   │ cost and latency  │
   └──────▲───────┘   └──────────▲─────────┘   └─────────────────────┘   └─────────┬─────────┘
          │                      │ corrected labels                                │ release
          │           ┌──────────┴─────────┐   ┌─────────────────────┐   ┌─────────▼─────────┐
          │           │ HUMAN VALIDATION   │<──│ MONITORING          │<──│ SERVING           │
          └───────────│ (Action Center-    │   │ drift, data quality,│   │ shadow > canary > │
            new data  │  style review)     │   │ confidence, cost,   │   │ GA; per-tenant,   │
                      └────────────────────┘   │ incidents           │   │ per-region        │
                                               └─────────────────────┘   └───────────────────┘
```

### Framing questions

1. [Likely] Design the ML platform that lets UiPath ship, monitor and retrain document, communications and agent models for thousands of enterprise tenants. Start from the SLOs, not the components.
   - Which three SLOs would you put on the board first, and who in the company owns each?
   - Which parts are shared across tenants, and which must be isolated per tenant?
2. [Likely] Walk me through one model's life, from the first labelled document to its retirement. Where do most production failures come from?
   - Which step is most often skipped in practice, and how would your platform make skipping it impossible?
3. [Likely] What does "MLOps" mean differently for a fine-tuned extractor, a prompt-based LLM agent and a retrieval index? Draw the lifecycle of each next to the others.
   - Which artefacts need versioning in each case: weights, prompts, tool schemas, indexes, eval sets, guardrail policy?
4. [Likely] You join and find each team has its own training scripts, buckets and deployment path. What platform would you build first, and what would you deliberately not build?
   - How do you get adoption without a mandate?
5. [Likely] How would you estimate the business value of a better MLOps platform, so that leadership funds it over a new model feature?
   - Which metric would you track for a year to show it worked?

## A. Training-data pipelines and labelling operations

```
  Production traffic ──> Sampler ──────────> Label queue ──> Annotators / SMEs ──> Adjudication
  (docs, emails,         (uncertainty,         (priority,       (tenant users in      (agreement,
   agent traces)          disagreement,         SLA, cost)       validation, vendor    gold checks)
        │                 drift slices,                          labellers)                │
        │                 random audit)                                                    ▼
        │                                                                     Versioned dataset
        └── consent and tenant policy check ── PII redaction ── dedupe ──>    (snapshot ID, split,
                                                                               lineage, licence)
```

6. [Likely] [2] Design the data flywheel in which human corrections from document validation tasks become training data. Validation tasks block until a person completes them [2]. How do you turn those completed tasks into clean labels?
   - A validator accepts a wrong extraction because they are rushed. How do you detect low-quality corrections?
   - Which corrections can be used for a shared model, and which only for that tenant's model?
7. [Likely] How do you decide which production samples to send for labelling when the labelling budget is 1% of traffic?
   - How do you mix uncertainty sampling, drift-slice sampling and a random audit sample, and why do you need the random part?
   - How do you avoid a feedback loop in which the model only ever gets labels on cases it already finds hard?
8. [Likely] Design a labelling operation for a new document type with no existing labels, where the only experts are the customer's accounts-payable staff.
   - How would you use an LLM to pre-label, and how do you stop annotators from rubber-stamping it?
   - How do you measure and report inter-annotator agreement for key-value fields and tables?
9. [Likely] [6] UiPath's `re` CLI, described as the command-line interface for UiPath IXP (formerly Communications Mining), can download raw messages from sources and datasets together with human-applied annotations [6]. How would you build a reproducible training snapshot from a live, continuously annotated dataset?
   - Annotations change after a snapshot is taken. How do you version them so an old model can still be reproduced?
10. [Likely] How do you split train, validation and test data for documents so that the test score is honest?
    - Same vendor's invoices in train and test: is that leakage? How do you split by vendor, template or time?
    - How do you keep a frozen test set that never leaks into labelling priorities?
11. [Likely] A taxonomy change adds two new fields and merges two classes. How does the data pipeline migrate old labels?
    - Which models must be retrained, and which only re-evaluated?
12. [Likely] Design PII handling in the training pipeline for invoices, IDs and emails, where some customers forbid any use of their data for training.
    - Where does consent live, and how is it enforced at read time rather than by convention?
    - A customer revokes consent. How do you find and remove their data from datasets, adapters and shared models?
13. [Likely] How would you generate synthetic documents to cover rare layouts, and how do you stop the model from learning synthetic artefacts?
    - What share of synthetic data would you allow in an eval set, if any?
14. [Likely] How do you turn agent traces (tool calls, human escalations, final outcomes) into training or evaluation data?
    - The outcome arrives weeks later (an invoice is paid or disputed). How do you join delayed labels to the original run?
15. [Likely] [14] Coded agents suspend on an interrupt such as an Action Center task or escalation, and resume when a person completes it [14]. Design the logging so that every escalation outcome becomes a usable label linked to the exact model, prompt and trace span that caused it.
    - Escalations are routed by priority and to specific users or groups. How does that routing bias the labels you collect, and how do you correct for it?
16. [Likely] [9] Agent traces can redact inputs and outputs (`hide_input`, `hide_output`) or transform them with processors before recording [9]. How do you build evaluation and training sets from traces when the most informative fields are the ones customers redact?
    - Who decides the redaction policy, and how do you stop a later policy change from silently changing what your eval set contains?
17. [Likely] What is the unit cost of a label in your design, and how would you lower it by 5x without losing quality?
    - When is a cheaper, noisier label better than a costly, clean one?
18. [Likely] How would you detect and handle duplicate and near-duplicate documents across tenants without moving tenant data across boundaries?
19. [Likely] Each tenant extends the shared document schema with its own custom fields and names. How do you train and evaluate a shared extractor over heterogeneous, tenant-defined schemas?
    - How do you map "Invoice No.", "Bill number" and "Ref" across tenants without a human curating every mapping?
20. [Likely] Who owns label quality: the scientist, the labelling vendor, the product team or the customer? How would you set that up organisationally?

## B. Feature stores and embedding stores

```
  Raw doc ──> OCR / parser ──> Layout + text features ──┐
                (version v_ocr)   (version v_feat)       ├──> Offline store (training, backfills)
  Message ──> Embedding model ──> Vectors ───────────────┤
                (version v_emb)    (index version)       └──> Online store / vector index (serving)
                                                                │
          Point-in-time correctness ── feature lineage ── per-tenant namespaces ── TTL and deletion
```

21. [Likely] Do document and agent models need a feature store? Argue for and against, then design the smallest one that solves the real problems.
    - What does training-serving skew look like when OCR or the parser is the "feature pipeline"?
22. [Likely] The OCR engine is upgraded. Every downstream extractor sees different tokens and boxes. How does the platform handle this?
    - Do you retrain, re-evaluate or pin the old OCR per model version? What does the registry record?
23. [Likely] [4] Design an embedding store for retrieval over tenant knowledge sources. Context Grounding indexes can be scoped to folders, built from buckets and other connectors, and re-ingested on a schedule; an ingestion request that overlaps a running one is rejected, and searching during ingestion is an explicit opt-in [4]. How do you version and refresh such indexes?
    - You change the embedding model. How do you re-embed millions of chunks per tenant without downtime?
    - How do you serve queries while ingestion is still running, and what consistency do you promise users?
24. [Likely] How do you test that a new embedding model is better for retrieval inside an agent, not just on a public benchmark?
    - Which offline retrieval metrics, and which end-to-end agent metrics, would gate the switch?
25. [Likely] How do you make sure a deleted document disappears from the vector index, caches and any derived summaries within a stated time?
    - How would you prove that to an auditor?
26. [Likely] Design point-in-time-correct features for an agent that predicts whether an invoice exception will be approved.
    - Which features would leak the label if computed at training time from current data?
27. [Likely] One shared vector index with tenant filters, or one index per tenant? Walk through cost, isolation, noisy neighbours and deletion.
    - At what tenant size does your answer change?
28. [Likely] How do you monitor the health of an embedding index: staleness, coverage and duplicate rate?
29. [Likely] [14] Agents can create short-lived (ephemeral) Context Grounding indexes for Deep RAG and batch-transform jobs, and suspend until the job completes [14]. Design the lifecycle for ephemeral indexes: creation, retention, deletion, cost attribution and monitoring.
    - A long-running agent is suspended for three days waiting for a human. Does its ephemeral index survive, and who pays for it?
30. [Likely] Retrieval indexes for 10,000 tenants cost more to store and keep warm than to query. Design hot, warm and cold tiers for vectors and the rules that move an index between them.
    - What query latency do you promise for a cold tenant, and how do you hide the reload from an agent mid-run?
31. [Likely] [4] Context Grounding supports encrypted indexes and a choice of extraction strategy at index creation [4]. How do these options change the re-index, migration and debugging story?

## C. Model registry and versioning (AI Center-style)

```
  Train run ──> Candidate artefact ──> Registry entry ─────────────────────────────────┐
  (code SHA,     (weights, tokenizer,   model_id / version N                          │
   data snapshot, prompt, config)       lineage: data, code, base model, OCR version  │
   hyperparams)                         eval report, model card, licence              │
                                        status: candidate > staging > production >     │
                                                deprecated > retired                   │
                                                                                       ▼
   Consumers pin either   ──  version N      (immutable, reproducible)
                          ──  tag "prod"     (movable alias; promotion = moving the tag)
```

32. [Likely] Design a model registry for UiPath-style ML packages and skills. What exactly is versioned, what is immutable, and what is a movable pointer?
    - What metadata must exist before a version may be promoted?
33. [Likely] [2] In the Documents service, callers of non-pretrained projects must pass exactly one of `version` or `tag` (passing both is rejected), and an unknown tag raises an error listing the project's available tags [2]. What are the trade-offs of tag-based versus version-based pinning for customer automations?
    - A customer's production robot pins tag "prod". You move the tag to a new version. What can break, and how do you warn them?
    - Which customers should pin an exact version, and how does the platform push them to upgrade?
34. [Likely] How do you make any production model exactly reproducible two years later for a regulated customer?
    - What do you store besides weights: data snapshot, code, container image, OCR version, random seeds?
35. [Likely] Design versioning for an LLM-based extractor where the "model" is a vendor model plus a prompt plus a schema plus a post-processor.
    - The vendor offers only an undated alias for one model. How do you protect customers from silent updates when you cannot pin a version?
36. [Likely] [3] LLM Gateway model IDs carry dated version suffixes, such as `gpt-4.1-mini-2025-04-14`, and which models a tenant can use depends on AI Trust Layer policy [3]. How would you manage the deprecation of a third-party model version that hundreds of customer agents depend on?
    - What migration tooling and evaluation do you give customers before the cut-over date?
37. [Likely] How should the registry represent a base model plus hundreds of per-tenant adapters?
    - What happens to the adapters when the base model is upgraded?
38. [Likely] Design the model card and lineage record for a document classifier, for both internal reviewers and customers' risk teams.
39. [Likely] How do you retire a model safely when you cannot see all of its callers (on-prem robots, long-running agents)?
    - How long do you keep serving a deprecated version, and who decides?
40. [Likely] How would a pipeline that retrains, evaluates and conditionally promotes a model (the "full pipeline" pattern) decide on its own whether to promote?
    - Which decisions would you never automate?
41. [Likely] How do you version evaluation sets and judges so that a score from last quarter is comparable with today's?
42. [Likely] [10] `uipath pack` builds a `.nupkg` and by default locks dependencies with `uv.lock`; with `--nolock` each run resolves the latest compatible versions [10]. For a deployed agent, define the immutable release artefact. Which of code, lock file, prompt, model ID, tool schemas, guardrail policy and eval set are inside it, and which are referenced?
    - A locked dependency gets a security fix. How do you patch thousands of locked customer agents without breaking reproducibility?
43. [Likely] [10] Each coded project carries a stable GUID in `uipath.json`; changing it makes the project look like a new, unrelated one and breaks the link to what was published under the old ID [10]. Design the identity model that separates a model or agent's identity, its versions and its deployments across personal-workspace and tenant feeds.
    - How do you keep lineage when a customer forks a UiPath template agent and edits it?
44. [Likely] [11] UiPath's Python SDK follows semantic versioning with release candidates, a deprecation period of at least one minor release and migration guides for breaking changes [11]. Would you apply semantic versioning to models and prompts? Define what counts as a "breaking change" for an extractor or an agent.
    - Is a 2-point accuracy gain that changes 5% of outputs a patch, a minor or a major release?

## D. CI/CD for models and prompts

```
  PR: code / prompt / tool schema / guardrail policy / model version change
     │
     ▼
  Unit + schema tests ──> Offline eval (golden set, slices) ──> Regression diff vs prod ──> Cost / latency check
     │                         │                                      │
     │                         └── LLM-judge + deterministic scorers  └── block if any protected slice drops
     ▼
  Shadow (copy traffic, no effect) ──> Canary (1-5% tenants, opt-in first) ──> Staged GA by region ──> Bake time
     │                                                                                                  │
     └──────────────────────────── automatic rollback on SLO breach ─────────────────────────────────────┘
```

45. [Likely] [5] Design CI/CD for agent prompts and tools. UiPath's SDK runs eval sets with output-based, trajectory-based and custom evaluators through `uipath eval`, with JSON results and a custom run ID [5]. How would you turn that into a release gate?
    - What is the pass rule: mean score, per-slice floors, or no regression on any "must-pass" case?
    - How do you handle non-determinism so that the gate is not flaky?
46. [Likely] A prompt change improves the average score by 3 points and drops one customer's slice by 10. Do you ship? Who decides?
47. [Likely] How do you test a change to a tool schema or guardrail policy that is not itself a model change but changes agent behaviour?
48. [Likely] Design shadow deployment for a document extractor. What do you compare, and how do you label disagreements cheaply?
    - Shadowing doubles inference cost. How do you limit it?
49. [Likely] How do you run a canary when many tenants have little traffic and some have strict change windows?
    - Who gets the canary first: internal tenants, opt-in customers, or the largest tenants?
50. [Likely] [12] UiPath's LLM-judge evaluators return a 0-1 score with a written justification, from a configurable judge model run at temperature 0 by default [12]. How do you use such a judge in CI without the judge's own drift, or its deprecation, breaking your gates?
    - How often do you recalibrate the judge against humans, and what triggers recalibration?
51. [Likely] How long should the full pre-release evaluation take, and what would you cut to keep it under one hour?
52. [Likely] Design an A/B test for a new extractor where the outcome is "fields changed by the human validator".
    - The validator's behaviour changes with the model (automation bias). How does that bias the experiment?
53. [Likely] How do you release a change that must be coordinated across a model, a prompt and a robot workflow that customers own?
54. [Likely] What is your branching and environment model (dev, staging, prod) for prompts that business users can edit themselves?
    - How do you stop a customer's prompt edit from being overwritten by a platform release?
55. [Likely] What release metrics would you report to the VP each week to show that releases are getting safer and faster?
56. [Likely] [13] UiPath's tool-call-order evaluator can require an exact sequence (score 1 or 0) or give partial credit with a longest-common-subsequence score [13]. When is partial credit the wrong rule for a release gate? Design trajectory gates that encode hard business constraints, such as "approve before pay".
    - How do you combine hard trajectory constraints with soft quality scores into one go/no-go decision?
57. [Likely] [10] The CLI supports a non-interactive OAuth client-credentials flow for CI/CD, so a pipeline can run `uipath publish` and `uipath invoke` [10]. Design the promotion pipeline from a developer's personal workspace to a production tenant feed. Who holds which credentials, and how do you enforce separation of duties?
    - How do you prove to a regulated customer that the package in production is the one that passed the gate?
58. [Likely] [8] Guardrails can log, block or modify data, at tool, LLM or agent scope, before or after a call [8]. Design the rollout of a new guardrail policy across tenants without breaking working automations.
    - Would you start in log-only mode as a shadow? Which false-positive rate lets you switch to block, and per which slice?
59. [Likely] [8] Some UiPath AI features are enabled ring by ring behind feature flags (for example, Bring Your Own Guardrail is described as still rolling out) [8]. Design a ring-based rollout for model and prompt changes. How are rings defined, and which ML metrics must hold before you promote to the next ring?
    - Early rings are not representative of large regulated tenants. How do you correct for that?
60. [Likely] A model change keeps quality flat but raises tokens per run by 40% on long documents. Design a cost and latency gate that would have caught it, and say who may override it.
    - How do you set cost budgets per tenant tier and per product line?
61. [Likely] Your golden set leaks into prompts over time: engineers copy failing eval cases into few-shot examples. How do you detect and prevent eval contamination in a prompt CI system?

## E. Drift and data-quality monitoring for document and agent models

```
  Inputs ─────────> Input monitors: file type, page count, language, OCR confidence, layout cluster, schema
     │
  Model ──────────> Output monitors: confidence histograms, null-field rate, class mix, abstention rate
     │
  Humans ─────────> Outcome monitors: validator edit rate per field, escalation rate, time-to-validate
     │
  Agents ─────────> Trace monitors: steps per run, tool error rate, loop rate, token cost, guardrail hits
     │
     └──> Per-tenant and per-slice baselines ──> alert routing (owner, severity) ──> triage runbook
```

62. [Likely] Design drift monitoring for a document extraction model across thousands of tenants when true labels arrive only for the small share of documents that humans validate.
    - Which label-free signals best predict an accuracy drop, and how did you check that they do?
    - How do you set per-tenant baselines for tenants with 50 documents a month?
63. [Likely] A large customer changes its invoice template and field accuracy drops for that tenant only. How does your system detect it, and how fast?
    - Is the right response a retrain, a tenant adapter, a rule, or routing to human validation?
64. [Likely] How do you tell data drift from a data-quality bug upstream, such as a scanner change, a broken OCR language pack or a truncated file?
65. [Likely] Design monitoring for an LLM agent where there is no single accuracy metric. What do you watch, and what pages a human at 3 a.m.?
    - Which agent metrics are leading indicators, and which are lagging?
66. [Likely] Validator edit rate goes down after a release. Is the model better, or are validators trusting it more? How do you find out?
67. [Likely] How do you monitor calibration of confidence scores over time, given that customers set straight-through-processing thresholds on them?
    - Calibration drifts. Do you recalibrate silently, or tell customers their thresholds now mean something else?
68. [Likely] How would you detect that a third-party LLM behind the gateway has changed behaviour, when the provider does not tell you?
69. [Likely] Design the alerting system so that thousands of tenant-level monitors do not cause alert fatigue.
    - How do you aggregate, deduplicate and route alerts to the right owner?
70. [Likely] How do you monitor retrieval quality in production for a RAG-based agent without labelled queries?
71. [Likely] What data-quality checks would you put at the very start of the training pipeline, and which ones block training?
72. [Likely] Over a year, which signals would tell you to retrain on a schedule, and which would justify event-triggered retraining instead?
73. [Likely] [9] Traces of agent runs are visible per Orchestrator job and in Maestro [9]. At millions of runs a day, design trace sampling and retention: what you keep in full, what you sample, and how tail-based sampling keeps the failures.
    - How do you bound trace storage cost per tenant while keeping enough data to debug a one-in-10,000 failure?
74. [Likely] [8] Guardrail block rate for PII doubles the day after a model release [8]. Is the new model leaking more PII, is the guardrail misfiring, or has customer data changed? Design the monitors that tell these apart.
    - Silent blocks show up as failed or escalated runs, not as errors. How do you count them as a quality regression?
75. [Likely] [6] In IXP and Communications Mining, business users keep annotating and the taxonomy changes weekly [6]. How do you monitor model quality when the label set and the model are both moving targets owned by the customer?
    - When validation scores fall, how do you tell a worse model from a harder taxonomy?
76. [Likely] A customer's ERP vendor ships a UI update and a computer-use agent starts failing on one screen. Design drift detection for UI and screen-understanding models, where the "data" is a third party's application.
    - Can you detect the change before the first failed run, for example from screenshots or selectors?

## F. Per-tenant fine-tuning and adapter management

```
                         ┌──────── Base model (shared, version B_k) ────────┐
                         │                                                    │
   Tenant A data ──> Adapter A (LoRA, v3) ──┐                                 │
   Tenant B data ──> Adapter B (LoRA, v1) ──┼──> Adapter registry ──> Multi-adapter server
   Tenant C (no FT) ── base + prompt only ──┘    (base compat,          (hot adapters in GPU,
                                                  eval report,           cold in CPU / disk,
                                                  owner, region)         per-request routing)
```

77. [Likely] When should a tenant get its own fine-tuned adapter, rather than prompt changes, retrieval or a shared model improvement? Design the decision rule.
    - What is the minimum data and the minimum expected gain for a tenant adapter?
78. [Likely] Design serving for 5,000 tenant adapters on a shared base model with a strict p95 latency SLO.
    - How do you decide which adapters stay in GPU memory? What happens on a cold load?
    - How do you batch requests that use different adapters?
79. [Likely] The base model is upgraded. 5,000 adapters were trained on the old base. What is your migration plan?
    - Which adapters do you retrain automatically, and how do you check that each retrained adapter is no worse for its tenant?
80. [Likely] How do you guarantee that tenant A's adapter, data and outputs can never be served to tenant B?
    - Which test would you add to CI to prove it?
81. [Likely] How do you evaluate a tenant adapter when the tenant's labelled test set has only 40 documents?
82. [Likely] A tenant fine-tunes on its own data and quality gets worse. Who is responsible, and what does the product do?
83. [Likely] How do you attribute GPU and storage cost of fine-tuning and adapter serving to tenants and turn that into pricing?
84. [Likely] Can improvements learned from one tenant's corrections flow into the shared base model? Design the consent, aggregation and privacy controls.
    - How would you test for memorisation of one tenant's data in the shared model?
85. [Likely] Design self-service fine-tuning for customers (upload, train, evaluate, deploy). Which guard rails stop a customer from deploying a bad model?
86. [Likely] A tenant's training data for an adapter could be poisoned, for example by an insider planting documents so that an agent approves certain payments. Design the controls on self-service fine-tuning data and the checks on the resulting adapter before it serves agents that take actions.

## G. GPU training and inference capacity planning

```
   Demand ──> interactive (agents, validation UI)   p95 < 2 s     ── reserved GPUs, autoscale
          ──> near-real-time (queue-triggered docs)  minutes       ── shared pool, batching
          ──> batch (backfills, re-embedding, evals) hours         ── preemptible / off-peak
          ──> training and tuning                    days          ── scheduled, quota per team

   Capacity = f(requests/s, tokens or pages per request, model size, batch size, utilisation target, headroom)
```

87. [Likely] Do a back-of-envelope estimate on the board: 20 million pages a month across tenants, a 10x peak in the last week of the quarter, a 2-second p95 target. How many GPUs, of which type, in how many regions?
    - What utilisation target would you plan for, and why not 90%?
88. [Likely] Interactive agent traffic and batch document backfills compete for the same GPUs. Design the scheduler and priority model.
    - How do you stop one tenant's backfill from hurting everyone's latency?
89. [Likely] When would you self-host an open-weights model instead of calling a third-party model through the gateway? Put numbers on the break-even.
    - How do quantization, distillation and speculative decoding change the numbers?
90. [Likely] Plan GPU capacity for training for the next year when the research roadmap is uncertain. How do you split reserved and on-demand capacity?
91. [Likely] How do you allocate scarce GPUs among teams (documents, agents, computer vision for UI automation)? Who decides, and on what basis?
92. [Likely] Design autoscaling for LLM inference, where cold start means loading tens of gigabytes of weights.
    - What do you keep warm, and what does it cost?
93. [Likely] How do you capacity-plan for an on-prem customer who has two GPUs, or none?
    - Which models must have a CPU-only path, and what accuracy do you give up?
94. [Likely] What GPU efficiency metrics would you report, and how would you find the biggest waste in a large training cluster?
95. [Likely] A new model is 4% more accurate and 3x more expensive to serve. How do you decide whether to ship it, and to whom?
96. [Likely] How do you reserve capacity for evaluation and re-embedding jobs so that releases are not blocked by production load?
97. [Likely] Agent prompts share long system prompts and tool schemas across runs. How does prefix (KV) caching change GPU capacity planning, and how do you design prompts and routing to maximise cache hits without mixing tenants?
    - How do you measure the cache hit rate, and what happens to capacity if a release changes the shared prefix?
98. [Likely] [3] LLM Gateway usage consumes Agent Units [3]. At quarter end, demand exceeds both GPU capacity and some tenants' unit budgets. Design admission control and graceful degradation: queueing, smaller models, lower reasoning effort or human routing.
    - Which tenants get priority, and how is that written into the contract and the scheduler?

## H. On-prem, air-gapped and multi-region rollout

```
   Automation Cloud (multi-region)        Dedicated / sovereign cloud        On-prem Automation Suite
   ┌─────────────────────────────┐        ┌───────────────────────┐          ┌─────────────────────────┐
   │ region EU │ region US │ ... │        │ single tenant, pinned │          │ customer infra, maybe   │
   │ data residency per tenant   │        │ region and models     │          │ air-gapped; own package │
   └──────────────┬──────────────┘        └───────────┬───────────┘          │ feed and model mirror   │
                  │                                   │                      └────────────┬────────────┘
                  └──────── signed release bundle: weights + config + eval suite + SBOM ──┘
                                     (cloud: continuous; on-prem: versioned drops)
```

99. [Likely] [7] Air-gapped Automation Suite customers must host their own package index even for Python dependencies [7]. Design how you ship model weights, guardrail models, embedding models and evaluation suites into such an environment and keep them updated.
    - How do you sign, verify and roll back an offline model bundle?
    - How do you support a customer who is three releases behind?
100. [Likely] How do you monitor model quality in an air-gapped deployment where no telemetry leaves the site?
     - What can the customer run locally to tell you whether a regression is the model or their data?
101. [Likely] [3] Some gateway models are not available in some regions, which can cause data-residency errors, and AI Trust Layer policy controls which models a tenant can use [3]. Design model routing across regions under data-residency rules.
     - A model exists only in the US. What do EU tenants get, and how do you keep behaviour consistent?
102. [Likely] How do you roll out a new model region by region, and what decides the order?
     - How do you keep evaluation results comparable when regions use different underlying model versions?
103. [Likely] Design a model release process that serves both the continuous cloud release train and quarterly on-prem releases from one codebase.
     - Which features do you allow to differ between cloud and on-prem, and how do you stop the two from drifting apart?
104. [Likely] How would you run an LLM-based agent on-prem when the customer cannot use any third-party model API?
     - What is the smallest model that keeps the product useful, and how do you prove it to the customer?
105. [Likely] How do you handle a security patch to an inference container across cloud regions, dedicated clouds and on-prem installs?
106. [Likely] A regulator asks which model version processed a given document in a given region on a given date. Design the audit trail that answers this in minutes.
107. [Likely] What is your disaster-recovery plan for the model registry and vector indexes in one region?
108. [Likely] [10] [8] The same agent package can run on Automation Cloud or on an Automation Suite instance with its own URL [10]. Built-in guardrails are backed by the UiPath Guardrails API, powered by Azure Content Safety [8]. Design how model endpoints, guardrails and judge models resolve per environment, and what replaces cloud-hosted guardrails in an air-gapped install.
     - How do you show an air-gapped customer that their local guardrails behave the same as the cloud ones?
109. [Likely] [15] The UiPath CLI collects usage telemetry by default, and users can opt out [15]. If your fleet-level quality and drift metrics come only from tenants that share telemetry, how biased are they, and how do you correct for that?
     - What minimal, privacy-preserving signal would you ask opted-out customers to share, and how would you justify it?
110. [Likely] [7] [10] Air-gapped customers mirror Python packages into their own index, and agent packages lock their dependencies [7] [10]. A critical CVE lands in a library inside hundreds of locked customer agents running on-prem. Design the response, from detection to verified patch.

## I. Rollback and incident response for model regressions

```
   Detect ──────> Triage ────────────> Mitigate ──────────────> Diagnose ─────────> Fix and prevent
   (monitor,      (scope: tenants,     (move tag back,           (diff inputs,       (new eval cases,
    customer       regions, slices;     route to previous         model, prompt,      gate change,
    ticket,        severity; owner)     version, raise human      OCR, gateway        postmortem,
    judge alert)                        review threshold,         version)            customer comms)
                                        disable feature)
```

111. [Likely] [2] A new extractor version is promoted by moving a published tag. Two hours later, three large tenants report wrong totals. Walk me through the first 60 minutes.
     - Rolling back is moving the tag back [2]. What can still go wrong: cached results, documents in flight, adapters trained on the new base?
     - What do you tell customers, and when?
112. [Likely] How do you roll back a prompt or agent configuration when agent runs are long-running and suspended waiting for a human?
     - Should a resumed run continue on the old version or the new one? Why?
113. [Likely] A third-party model provider changes behaviour and your agents start failing a guardrail at 5x the normal rate. You cannot roll back the provider. What are your options?
114. [Likely] Wrong extractions were sent straight through to a customer's ERP system for six hours before detection. Design the remediation: finding affected records, re-processing and customer communication.
     - What would have reduced the blast radius?
115. [Likely] Design the severity levels and on-call model for ML regressions, as distinct from infrastructure outages.
     - Who is on call for "the model is quietly wrong": scientists, ML engineers or SRE?
116. [Likely] Write the outline of a blameless postmortem for a model regression that passed every offline gate. Which process change would you push for?
117. [Likely] How do you design "safe mode" fallbacks (higher human-review thresholds, rule-based extraction, a previous model) that can be switched on per tenant in seconds?
118. [Likely] How do you test your rollback path regularly, so that it works on the day you need it?
119. [Likely] After an incident, how do you decide whether to add the failing cases to the golden set, a separate regression set or the training data?
     - How do you avoid overfitting the release gate to past incidents?
120. [Likely] A new extractor version writes outputs in a new schema to downstream entities and queues. You roll back the model, and consumers now see two schemas. Design releases so that model rollback never needs a data rollback.
     - When do you need a dual-write or versioned-output period, and how long should it last?
121. [Likely] [8] A newly enabled guardrail blocks 30% of legitimate invoice runs for one region [8]. Walk through mitigation: per-tenant kill switch, switching to log-only, threshold changes, and how you stop the same failure in the next policy release.

## J. Principal-level integrative prompts

122. [Likely] You have one quarter and four engineers. Which three MLOps investments would most reduce model-related customer incidents next year? Defend the ranking with data you would collect.
123. [Likely] Design the platform so that a new document type can go from zero labels to a monitored production model in two weeks, mostly self-service by the customer.
     - Where must a UiPath scientist stay in the loop?
124. [Likely] Product wants weekly model releases; large regulated customers want quarterly changes with sign-off. Design a release policy that satisfies both.
125. [Likely] How would you unify the lifecycle for classic ML packages, document models and LLM agents on one platform without slowing any of them down?
     - What would you refuse to unify?
126. [Likely] Pick one MLOps decision from your past work that you would now make differently. What did it cost, and what would you design today?
127. [Likely] Build or buy: would you base UiPath's model registry, experiment tracking and evaluation store on open-source tools (for example MLflow or Kubeflow), a cloud vendor service or an in-house system, given that it must also run on-prem and air-gapped?
     - What would make you change the decision in two years?
128. [Likely] By 2026, agents can triage drift alerts, propose retraining and open release PRs. Which MLOps steps would you let an agent do on its own, which need human approval, and how do you audit the agent doing MLOps?

## References

1. alirezadir, "Machine-Learning-Interviews" (AIMLInterviews), ML System Design chapter: a nine-step flow ending with prediction service, online testing (A/B, bandits, shadow, canary) and scaling, monitoring and updates (distribution shift, retraining cadence, active learning). Its 2026 GenAI notes say interviewers weigh cost, latency, guardrails and monitoring, and that "evaluation methodology is the new system design". General guide, not specific to UiPath (fetched). GitHub. https://raw.githubusercontent.com/alirezadir/Machine-Learning-Interviews/main/src/MLSD/ml-system-design.md
2. UiPath, Documents service source, `uipath-platform` package of `uipath-python`: choose exactly one of `version` or `tag` for non-pretrained projects; when a version is given the tag is ignored; unknown tags raise an error listing the published tags; validation actions block until a person completes them in Action Center; non-blocking IXP validation path (official; fetched). GitHub, 2025-2026. https://raw.githubusercontent.com/UiPath/uipath-python/main/packages/uipath-platform/src/uipath/platform/documents/_documents_service.py
3. UiPath, "Chat Models" documentation, `uipath-langchain-python`: models served through the LLM Gateway and subject to AI Trust Layer policy; some models unavailable in some regions (data-residency errors); dated model version IDs; usage consumes Agent Units (official; fetched). GitHub. https://raw.githubusercontent.com/UiPath/uipath-langchain-python/main/docs/chat_models.md
4. UiPath, Context Grounding service source, `uipath-platform` package: index creation from buckets and connectors with extraction strategy, embeddings and encryption options; scheduled indexers; ingestion conflicts rejected; search during ingestion; ephemeral indexes for Deep RAG and batch transform (official; fetched, first 100,000 characters read). GitHub. https://raw.githubusercontent.com/UiPath/uipath-python/main/packages/uipath-platform/src/uipath/platform/context_grounding/_context_grounding_service.py
5. UiPath, "Agent Evaluations" documentation, `uipath-python`: eval sets, output-based, trajectory-based and custom evaluators, `uipath eval` CLI with parallel workers, JSON output file and custom run IDs (official; fetched). GitHub. https://raw.githubusercontent.com/UiPath/uipath-python/main/packages/uipath/docs/eval/index.md
6. UiPath (Re:infer), `re` CLI README: the command-line interface for UiPath IXP; manages sources, datasets and comments; downloads raw messages with human-applied annotations (official; fetched). GitHub. https://raw.githubusercontent.com/reinfer/cli/master/README.md
7. UiPath, "Automation Suite" documentation, `uipath-python`: air-gapped environments need their own package index (devpi, Artifactory, Nexus and others); restricted environments can use Azure DevOps Artifacts (official; fetched). GitHub. https://raw.githubusercontent.com/UiPath/uipath-python/main/packages/uipath/docs/AutomationSuite.md
8. UiPath, "Guardrails" documentation, `uipath-python`: guardrails at tool, LLM and agent scope; PRE, POST and PRE_AND_POST stages; PII, harmful-content, prompt-attack, intellectual-property, LLM-as-judge and custom validators backed by the UiPath Guardrails API (Azure Content Safety); log, block and custom actions; Bring Your Own Guardrail configured under AI Trust Layer and still rolling out across rings (official; fetched). GitHub. https://raw.githubusercontent.com/UiPath/uipath-python/main/packages/uipath/docs/core/guardrails.md
9. UiPath, "Tracing" documentation, `uipath-python`: the `@traced()` decorator; traces shown per Orchestrator job and in Maestro; `hide_input`, `hide_output` and input/output processors to redact data (official; fetched). GitHub. https://raw.githubusercontent.com/UiPath/uipath-python/main/packages/uipath/docs/core/traced.md
10. UiPath, "CLI Reference" documentation, `uipath-python`: OAuth client-credentials flow for CI/CD; `UIPATH_URL` for Automation Suite; stable project GUID in `uipath.json`; `uipath pack` to `.nupkg` with `uv.lock` dependency locking by default and `--nolock`; `publish` to tenant or personal-workspace feeds; `push`, `pull`, `invoke`, `deploy` (official; fetched). GitHub. https://raw.githubusercontent.com/UiPath/uipath-python/main/packages/uipath/docs/cli/index.md
11. UiPath, "Release Policy" documentation, `uipath-python`: semantic versioning, release candidates (`X.Y.ZrcN`), deprecation for at least one minor release, migration guides for breaking changes (official; fetched). GitHub. https://raw.githubusercontent.com/UiPath/uipath-python/main/packages/uipath/docs/release_policy.md
12. UiPath, "LLM Judge Output Evaluators" documentation, `uipath-python`: judge scores from 0.0 to 1.0 with a justification, configurable judge model, temperature 0.0 by default (official; fetched). GitHub. https://raw.githubusercontent.com/UiPath/uipath-python/main/packages/uipath/docs/eval/llm_judge_output.md
13. UiPath, "Tool Call Order Evaluator" documentation, `uipath-python`: strict exact-sequence scoring (1 or 0) versus non-strict longest-common-subsequence partial credit (official; fetched). GitHub. https://raw.githubusercontent.com/UiPath/uipath-python/main/packages/uipath/docs/eval/tool_call_order.md
14. UiPath, "Human In The Loop" documentation, `uipath-langchain-python`: `interrupt()` models that suspend an agent until an Action Center task, escalation, process job, Deep RAG query, ephemeral index, batch transform or IXP extraction/validation completes (official; fetched). GitHub. https://raw.githubusercontent.com/UiPath/uipath-langchain-python/main/docs/human_in_the_loop.md
15. UiPath, "Environment Variables" documentation, `uipath-python`: folder context, and CLI usage telemetry enabled by default with an opt-out (`UIPATH_TELEMETRY_ENABLED=false`) (official; fetched). GitHub. https://raw.githubusercontent.com/UiPath/uipath-python/main/packages/uipath/docs/core/environment_variables.md
