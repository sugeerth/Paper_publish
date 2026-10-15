# 07. UiPath AI Context: DocPath and Document Understanding, the LLM Gateway and AI Trust Layer, Agentic Automation Strategy and the Peak Acquisition

> **How to read this chapter.** This chapter asks how well you understand UiPath's own AI stack and how you would extend it. You asked for ML system design material, so each section opens with a plain-text block diagram of the UiPath system it covers, followed by numbered questions. Indented sub-bullets are follow-up probes. The chapter has questions only, with no answers. The diagrams are plain text so they look the same in Markdown and in a PDF export.
>
> **Tagging system.** Every question carries one tag:
> - **[Reported] [n]**: a candidate report, question bank or official page links this question to UiPath. The number points to the References list.
> - **[Likely]**: an inference from the role (Principal Applied Scientist), the onsite loop format and UiPath's public product line. No candidate has reported it.
> - *(context: [n])* at the end of a [Likely] question means the product fact in the question comes from a source that was read in the review pass. It does **not** mean a candidate reported the question.
>
> **Research limits (please read).** In the review pass (October 2026) the shared web-search budget was already used up, and direct fetches of uipath.com, docs.uipath.com, Glassdoor, arXiv, Hugging Face and the OSWorld leaderboard site were blocked (DNS failure or proxy denial). Only raw GitHub files could be read. The review read 15 of them, mostly UiPath's own open-source SDK code and documentation (References 1-15). Those files confirm product facts: the LLM Gateway, AI Trust Layer model policy, guardrails, Document Understanding project types, Context Grounding, agent evaluation and human-in-the-loop. None of them is a candidate interview report. So **no question below is tagged [Reported]**, and this chapter does not invent reports. Every question is **[Likely]**.
>
> **Facts still unverified.** The names DocPath and CommPath, the ScreenAgent name and its benchmark standing, and the details of the Peak acquisition (announced in 2025, by general knowledge) could not be checked against any page in this pass. Check them on uipath.com before the interview. One verified update: UiPath's Communications Mining CLI (the former Re:infer `re` tool) now describes itself as the CLI for **UiPath IXP**, and the SDK runs both document and IXP extraction through one Documents service (context: [5][12]).

## Format notes for this round

| Item | Expected shape (inferred, not confirmed) |
|---|---|
| Where it appears | It is rarely a separate round. It runs through the hiring-manager conversation, the system design round and the "why UiPath" part of the leadership or bar-raiser round. |
| Duration | 10-25 minutes of a 45-60 minute round |
| Interviewer | Hiring manager (an AI or science director), a principal scientist on DU, IXP, Agents or the platform, or a product leader |
| What is assessed at principal level | Do you know what UiPath ships and why? Can you name the science gaps in it? Can you reason about the moat (specialised models, enterprise data, governance, the installed robot base) against frontier-model vendors? Can you propose a multi-quarter research agenda that fits the business? |
| Failure modes | Generic "I'd fine-tune an LLM" answers; not knowing the difference between RPA robots and agents; ignoring on-prem, air-gapped and regulated customers; treating governance as an afterthought |

### Reference diagram: where the AI pieces sit in UiPath's platform

```
   ┌───────────────────────────── UiPath Platform (Automation Cloud / Suite) ─────────────────────────────┐
   │                                                                                                      │
   │  Maestro (agentic orchestration: process model, long-running state, SLAs, human/agent/robot routing) │
   │        │                    │                         │                          │                   │
   │   ┌────▼─────┐        ┌─────▼──────┐          ┌───────▼───────┐          ┌───────▼────────┐          │
   │   │ AGENTS   │        │ ROBOTS     │          │ HUMANS        │          │ DECISIONING    │          │
   │   │ Agent    │        │ RPA, API,  │          │ Action Center │          │ (Peak: pricing,│          │
   │   │ Builder, │        │ UI selectors│         │ validation,   │          │ inventory,     │          │
   │   │ coded    │        │ computer-  │          │ approvals     │          │ forecasting)   │          │
   │   │ agents   │        │ use agents │          │               │          │                │          │
   │   └────┬─────┘        └─────┬──────┘          └───────┬───────┘          └───────┬────────┘          │
   │        │                    │                         │ labels / corrections     │                   │
   │   ┌────▼────────────────────▼─────────────────────────▼──────────────────────────▼────────┐          │
   │   │           AI TRUST LAYER / LLM GATEWAY                                                │          │
   │   │  tenant model policy · guardrails (PII, harmful content, prompt attacks, IP, BYOG)    │          │
   │   │  passthrough + normalized APIs · audit · usage (Agent Units) · per-product attribution │          │
   │   └────┬──────────────────────┬────────────────────────┬──────────────────────────────────┘          │
   │   ┌────▼──────┐        ┌──────▼────────┐        ┌──────▼──────────────────────┐                     │
   │   │ DocPath / │        │ IXP (incl.    │        │ Third-party LLMs: OpenAI,   │                     │
   │   │ DU models │        │ Comms Mining) │        │ Bedrock (Anthropic), Vertex │                     │
   │   └───────────┘        └───────────────┘        └─────────────────────────────┘                     │
   │                                                                                                      │
   │  Context: Process Mining · Task Mining · Data Service · Context Grounding (indexes, Deep RAG)         │
   └──────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## 1. Company AI strategy and "why UiPath"

1. [Likely] UiPath has moved from RPA to "agentic automation". In your own words, what does that shift mean technically, and what new science problems does it create?
   - What can an agent do that a robot cannot, and where is a robot still the better choice?
   - How would you explain the shift to a customer who has 500 production robots?
2. [Likely] What is UiPath's defensible advantage against frontier-model labs and hyperscalers that also ship agents?
   - Is the moat the models, the data, the orchestration layer, the installed base or governance? Rank them and defend the order.
   - What would make that moat disappear within two years?
3. [Likely] What do you think is the single biggest open science problem in UiPath's product line today?
   - How would you staff and sequence a team to work on it?
4. [Likely] UiPath sells to regulated enterprises (banking, insurance, healthcare, public sector). How does that change your research priorities compared with a consumer AI company?
   - Which methods would you rule out because of it?
5. [Likely] Where should UiPath build its own models, and where should it rely on third-party LLMs?
   - What evidence would change your mind for a given capability?
   - What is your view on small specialised models against large general models for enterprise automation over the next three years?
6. [Likely] How would you decide whether a research result is ready to go into a product that runs on customers' production processes?
7. [Likely] What should a Principal Applied Scientist own at UiPath that a senior applied scientist does not?
   - Give an example of a cross-team technical decision you would expect to drive in your first year.
8. [Likely] Model calls made through UiPath's agent SDKs consume the customer's "Agent Units" rather than the customer's own provider tokens. How does that pricing model change what the science team should optimise? (context: [3])
   - Who carries the cost of a model that uses three times more tokens for a two-point quality gain: UiPath or the customer?
   - How would you price a call to an in-house specialised model against a call to a frontier model?
9. [Likely] UiPath now runs agents written in LangGraph, LlamaIndex, OpenAI Agents and other frameworks on one runtime, and also hosts customers' MCP servers. Is "be the neutral governed platform for everyone's agents" a stronger strategy than "build the best agents"? What does each strategy demand from applied science? (context: [13][14])

## 2. DocPath and Document Understanding (DU)

### System diagram: the Document Understanding pipeline

```
  Document in (scan, PDF, email attachment, photo)
        │
        ▼
  ┌──────────────┐   ┌──────────────┐   ┌────────────────────┐   ┌──────────────────────┐
  │ Digitize     │──>│ Classify     │──>│ Extract             │──>│ Validate             │
  │ OCR / layout │   │ doc type,    │   │ project types:      │   │ confidence thresholds│
  │ reading order│   │ page split,  │   │ - pretrained        │   │ business rules       │
  │ tables, hand-│   │ results with │   │ - DU Modern         │   │ cross-field checks   │
  │ writing      │   │ confidences  │   │ - IXP (generative)  │   │                      │
  └──────────────┘   └──────────────┘   │ pinned to a project │   └─────────┬────────────┘
                                        │ version tag         │             │
                                        └────────────────────┘             │
                                                          low confidence   │  high confidence
                                                      ┌────────────────────┘──────────────┐
                                                      ▼                                   ▼
                                        ┌──────────────────────┐            ┌────────────────────┐
                                        │ Classification and   │            │ Export to robot /  │
                                        │ extraction validation│            │ agent / ERP        │
                                        │ tasks (Action Center)│            └────────────────────┘
                                        └──────────┬───────────┘
                                                   │ corrected labels
                                                   ▼
                                        ┌──────────────────────┐
                                        │ Active learning /     │
                                        │ retraining, new tag,  │
                                        │ eval sets, promotion  │
                                        └──────────────────────┘
```

10. [Likely] DocPath is positioned as a specialised generative model for documents. Why would a company build a document-specific LLM instead of prompting a general frontier model?
   - On which document types would you expect a general model to win, and on which DocPath?
   - What benchmark would convince a sceptical product leader?
11. [Likely] Design the training data strategy for a document foundation model when most customer documents are confidential and cannot leave the tenant.
   - What can you do with synthetic documents, and where does synthetic data fail?
   - How would you use human validation corrections without breaking data-use commitments?
12. [Likely] How do you produce calibrated per-field confidence scores from a generative extractor so that the validation-threshold logic in the diagram still works?
   - How would you measure calibration per field and per document type?
   - What happens to straight-through-processing rate if the model is overconfident by 5 points?
13. [Likely] Generative extraction can hallucinate a plausible invoice number that is not in the document. How do you detect and prevent that?
   - How would you ground each extracted value to a bounding box or token span?
   - What is your metric for "unsupported" values?
14. [Likely] Compare three extraction approaches for a new document type with 20 labelled samples: a fine-tuned layout model, a prompted generative model and a hybrid. How do you choose?
   - How does the answer change at 2,000 samples?
15. [Likely] UiPath's SDK exposes three extraction project types side by side: pretrained models, DU Modern projects and IXP projects. If you owned the roadmap, would you converge them into one model family or keep them separate? (context: [5])
   - What does each type buy the customer in cost, latency, controllability and labelling effort?
   - How would you migrate a customer's trained DU Modern project to IXP without a drop in accuracy?
16. [Likely] How would you handle multi-page bundles (a loan file with 40 mixed documents) for classification and page splitting?
   - Where do long-context models help here, and where do they hurt?
17. [Likely] The SDK's default extraction path takes several classification results and keeps the one with the highest confidence. When is "take the argmax class" the wrong policy for a document pipeline? (context: [5])
   - How would you handle near-ties, out-of-taxonomy documents and bundles that contain several types?
   - Should a low classification margin open a classification-validation task even when extraction confidence is high?
18. [Likely] Tables, checkboxes, signatures, stamps and handwriting cause most DU errors. How would you prioritise model work across them?
   - What error analysis would you run first?
19. [Likely] Design the evaluation framework for DU across hundreds of customer document types.
   - Field-level F1, document-level accuracy or straight-through rate: which is the north-star metric and why?
   - How do you build a held-out set that reflects the customer distribution without seeing customer data?
20. [Likely] A large customer reports that extraction accuracy fell after a model upgrade, while your global benchmark improved. How do you investigate, and how do you prevent this?
21. [Likely] DU and IXP extraction calls name a published project version by tag (or version number). Design the release process for a customer's project: how a retrained model earns a new tag, how it is shadow-tested against the current tag, and when traffic moves. (context: [5])
   - What offline and online evidence do you require before a customer's production tag moves?
   - How do you roll back if straight-through rate drops after promotion?
22. [Likely] Human review in the SDK is split into classification validation and extraction validation tasks in Action Center. Treat human validation capacity as a fixed budget: how would you allocate it across documents and fields to maximise end-to-end accuracy? (context: [5][10])
   - How does the policy change when the validation queue backs up and SLAs are at risk?
23. [Likely] IXP extraction in the SDK is asynchronous: start the job, then poll for the result. How would you schedule GPU inference for a mix of one-page receipts and 300-page contracts under per-tenant SLAs? (context: [5])
   - Where would you batch, where would you split documents, and how would you decide?
24. [Likely] How should OCR and the generative extractor interact? Would you move to an OCR-free, pixel-input model?
   - What does that do to latency, cost and multilingual coverage?
25. [Likely] Design an active-learning loop that uses human validation in Action Center to improve a tenant's extractor without catastrophic forgetting or label noise.
   - How do you handle validators who disagree or who make systematic mistakes?
26. [Likely] How would you serve DocPath for on-premises and air-gapped customers with limited GPUs?
   - Which compression methods would you try (distillation, quantisation, smaller specialised heads), and how do you show that quality held?
27. [Likely] How do you extend document models to new languages and scripts (Arabic, Japanese, Cyrillic) with little labelled data?
28. [Likely] What role should DU play inside an agent? Should extraction be a tool the agent calls, or should the agent read documents directly?
   - How do confidence and validation signals pass through to the agent's decisions?

## 3. CommPath, Communications Mining and IXP

### System diagram: communications understanding

```
  Email / ticket / chat streams
        │
        ▼
  ┌──────────────┐   ┌────────────────────┐   ┌───────────────────────┐   ┌────────────────────┐
  │ Ingest,      │──>│ CommPath / intent  │──>│ Entity + field        │──>│ Route: robot, agent│
  │ thread, de-  │   │ taxonomy model     │   │ extraction, sentiment,│   │ human queue, or    │
  │ dup, redact  │   │ (multi-label,      │   │ urgency, quality-of-  │   │ auto-reply draft   │
  └──────────────┘   │  hierarchical)     │   │ service signals       │   └────────────────────┘
                     └─────────┬──────────┘   └───────────────────────┘
                               │ taxonomy drift / new intents
                               ▼
                     ┌────────────────────┐
                     │ Few-shot labelling │  <── analyst trains taxonomy in UI
                     │ + discovery        │      (sources, datasets managed via
                     └────────────────────┘       the IXP `re` CLI)
```

29. [Likely] Communications Mining lets business users build their own intent taxonomy with few labels. How would you design the model so a non-expert gets a usable classifier after about 50 labels?
   - How do you choose which messages to show the user next?
30. [Likely] How would you discover new or emerging intents in a stream of customer emails before anyone has labelled them?
   - How do you avoid flooding analysts with junk clusters?
31. [Likely] Why build a communications-specific model (CommPath) rather than prompt a general LLM per message?
   - Answer in terms of cost per million messages, latency and stability of the taxonomy.
32. [Likely] UiPath's Communications Mining tooling now presents itself as part of UiPath IXP, alongside document extraction. What is the science case for one extraction platform over emails and documents, and what breaks when you merge them? (context: [5][12])
   - Can one label schema and one model family cover an email thread with three PDF attachments?
   - How would you migrate existing Communications Mining datasets and taxonomies without retraining every customer's model from scratch?
33. [Likely] How do you evaluate a hierarchical multi-label intent model when the taxonomy changes every month?
34. [Likely] Design the handoff from a classified email to an agent that drafts a reply and a robot that updates the CRM. Where are the human checkpoints?
   - What failure would worry you most in that chain?

## 4. LLM Gateway and AI Trust Layer

### System diagram: governed LLM access

```
  Caller (Agent, coded agent, DU/IXP, Autopilot, customer workflow)
        │  request + tenant + requesting product/feature headers
        ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ AI TRUST LAYER / LLM GATEWAY                                                │
  │                                                                             │
  │  ┌───────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐  │
  │  │ AuthN/Z,  │─>│ Policy check │─>│ Guardrails   │─>│ Router: passthrough│  │
  │  │ tenant    │  │ (allowed     │  │ PRE: PII,    │  │ (native provider   │  │
  │  │ quotas,   │  │ models, data │  │ prompt attack│  │ API) or normalized │  │
  │  │ Agent     │  │ residency)   │  │ BYOG vendors │  │ API; schema cleanup│  │
  │  │ Units     │  └──────────────┘  └──────────────┘  └─────────┬──────────┘  │
  │  └───────────┘                                                │             │
  │  ┌─────────────────────────────┐                     ┌────────▼─────────┐   │
  │  │ Audit log · usage + cost by │<────────────────────│ Guardrails POST: │   │
  │  │ product/feature · traces    │                     │ PII, harmful, IP,│   │
  │  └─────────────────────────────┘                     │ LLM-as-judge     │   │
  │                                                      └──────────────────┘   │
  └─────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
  Model providers: UiPath models (DocPath, CommPath), OpenAI, AWS Bedrock (Anthropic), Google Vertex AI
```

35. [Likely] What should an enterprise LLM gateway do that a direct API call to a model provider does not?
   - Which of those functions need applied-science work rather than plain engineering?
36. [Likely] UiPath's gateway offers two interfaces: passthrough classes that expose each provider's native API, and a normalized interface that is uniform across providers but loses provider-specific features. The SDK docs recommend passthrough. Which would you make the default for UiPath's own agents, and why? (context: [2][3])
   - What does each choice do to portability, to evaluation and to the cost of switching providers later?
37. [Likely] The models a tenant can call depend on that tenant's AI Trust Layer policy. An agent built and evaluated on one model is deployed to a tenant whose policy blocks that model. Design what should happen. (context: [3])
   - How do you guarantee quality on a substitute model you never tested the agent with?
   - Should the platform refuse to deploy, fall back automatically, or re-run the agent's evaluation set first?
38. [Likely] Design PII and PHI detection and masking for prompts going to third-party LLMs.
   - How do you mask without destroying the information the LLM needs (names in a contract, account numbers in an invoice dispute)?
   - What is your recall target, and how do you measure it across languages?
   - How do you re-identify the masked values safely in the response?
39. [Likely] UiPath's PII guardrail takes a per-entity confidence threshold (for example email at 0.5). How would you pick thresholds per entity type, language and customer, and what evidence would you show a compliance officer for each choice? (context: [4])
   - Is one threshold per entity calibrated across languages, or do you need per-language calibration?
40. [Likely] Guardrails can run at three scopes (tool, LLM call, whole agent) and at PRE, POST or both stages. Where would you place each validator for a claims-processing agent, and what latency and cost budget would you set for the guardrail stack? (context: [4][16])
   - Which checks are worth a model call on every step, and which only at agent input and output?
41. [Likely] The user-prompt-attack validator runs only before a call, not after. Most real prompt injection reaches an agent indirectly, through documents, emails, web pages or tool outputs. How would you design detection for that indirect path? (context: [4])
   - Where does that check belong: in the gateway, in the agent runtime, or in both?
   - How do you measure false positives that break legitimate automations?
42. [Likely] UiPath offers an LLM-as-judge guardrail at runtime, alongside fixed classifiers. When is an LLM judge a sound runtime guardrail, and when is it a liability? (context: [4])
   - What happens when the judge and the agent run on the same model family?
   - How do you bound the extra latency and cost per run?
43. [Likely] "Bring Your Own Guardrail" lets an admin plug a third-party guardrail vendor into the Trust Layer. How would you qualify a customer's guardrail vendor, and what should happen when it disagrees with UiPath's built-in validator? (context: [4])
44. [Likely] UiPath's guardrails include an "intellectual property" validator for outputs. What could that mean technically (code licence detection, copyrighted text, trade secrets), and how would you build and evaluate it? (context: [4])
45. [Likely] A guardrail can log, block or modify (redact) the data it inspects. For a multi-step agent, when should a violation block the run, and when should it redact and continue? (context: [4])
   - How does a silent redaction change the agent's later reasoning, and how would you detect harm from it?
46. [Likely] Design a model router that picks between a small specialised model and a frontier model per request.
   - What features does the router use, and what does it optimise (cost, latency, quality, policy)?
   - How do you train and evaluate the router without ground-truth labels for every request?
47. [Likely] Customers can bring their own model or API keys. How do you keep agent and DU quality guarantees when the underlying model is not yours?
   - What compatibility test suite would you require before a customer model is allowed?
48. [Likely] UiPath's gateway rewrites JSON schemas for structured output (it strips titles and forces `additionalProperties: false`), and it handles provider quirks such as reasoning models that reject a temperature setting. How would you guarantee schema-valid structured output across OpenAI, Anthropic and Gemini models that differ in strict-mode support? (context: [2])
   - How would you maintain and test a capability matrix of models against features (tools, schemas, reasoning effort, vision)?
49. [Likely] Every gateway call carries "requesting product" and "requesting feature" headers. Design cost and quality attribution across UiPath products from that telemetry. (context: [2])
   - How would you use it to find which product feature gives the worst quality per dollar?
50. [Likely] The SDK's model list still contains old model versions next to new ones. Design a deprecation process for a model that thousands of customer agents are pinned to. (context: [2])
   - What regression evidence do you need before forcing a migration, and who signs off?
51. [Likely] Design the audit and observability layer so a compliance officer can reconstruct why an agent took an action six months later.
   - What do you log, and what must you not log?
52. [Likely] How would you design semantic caching in the gateway, and when is it unsafe?
53. [Likely] A new frontier model is released. Walk through how you would qualify it before turning it on for all tenants.
   - What regression suites, canary strategy and rollback triggers would you use?
54. [Likely] How would you enforce data residency (EU-only processing, for example) while still giving customers access to the best models?
55. [Likely] What guardrail models would you build in-house (toxicity, groundedness, policy adherence, structured-output validity), and which would you buy?
56. [Likely] How should cost and token budgets be enforced per tenant, per agent and per run, and how do you stop a looping agent from burning budget?

## 5. Agentic automation: Agent Builder, Maestro, Context Grounding and computer use

### System diagram: an enterprise agent run

```
  Trigger (queue item, email, schedule, API)
        │
        ▼
  ┌────────────────── Maestro process instance (long-running state) ──────────────────┐
  │                                                                                    │
  │   ┌────────────┐     plan / act loop      ┌───────────────────────────────────┐    │
  │   │ Agent      │ ───────────────────────> │ Tools                             │    │
  │   │ (prompt,   │ <─────────────────────── │ - RPA workflows (deterministic)   │    │
  │   │  memory,   │      observations        │ - APIs / Integration Service      │    │
  │   │  context   │                          │ - DU / IXP extraction             │    │
  │   │  grounding)│                          │ - Context Grounding: search,      │    │
  │   └─────┬──────┘                          │   Deep RAG, Batch Transform       │    │
  │         │                                 │ - Computer-use agent              │    │
  │         │ low confidence / policy hit     │ - MCP servers                     │    │
  │         ▼                                 └───────────────────────────────────┘    │
  │   ┌────────────┐      ┌───────────────────┐       ┌──────────────────────────┐     │
  │   │ interrupt: │ ───> │ Resume with human │ ───>  │ Evaluate: output and     │     │
  │   │ Action     │      │ decision          │       │ trajectory evaluators,   │     │
  │   │ Center task│      └───────────────────┘       │ LLM judges, outcome KPIs │     │
  │   └────────────┘                                  └──────────────────────────┘     │
  └────────────────────────────────────────────────────────────────────────────────────┘
```

57. [Likely] UiPath frames its approach as agents, robots and humans working together. Design how Maestro should decide which of the three handles each step.
   - Is that decision made at design time, at runtime, or both?
   - How do you learn better routing from historical runs?
58. [Likely] Design the evaluation system for agents built by customers in Agent Builder, who have few test cases and no ML background.
   - How would you generate evaluation cases automatically from process documentation or past runs?
   - When is LLM-as-judge acceptable, and how do you calibrate it against humans?
59. [Likely] UiPath's agent evaluation framework splits evaluators into output-based ones (exact match, contains, JSON similarity, LLM judge) and trajectory-based ones (tool-call order, count, arguments and outputs, LLM judge of the trajectory). When does a trajectory evaluator punish a correct agent, and how would you fix that? (context: [8])
   - How would you score an agent that reaches the right answer by a different but valid tool order?
60. [Likely] The LLM-judge output evaluator returns a score from 0 to 1 with a written justification. How would you turn that score into a pass/fail release gate, and how do you stop the gate drifting when the judge model is upgraded in the gateway? (context: [9])
61. [Likely] How would you make a non-deterministic agent safe enough to run inside a process that auditors expect to be repeatable?
   - What would "deterministic replay" mean for an LLM agent?
62. [Likely] UiPath agents suspend on `interrupt(...)` while a person handles an Action Center task, then resume when it is done. Design the checkpoint for a run that waits three days for a human. (context: [10])
   - What state must be saved, and what must be re-read on resume?
   - What should happen if the prompt, model or policy changed while the run was suspended?
63. [Likely] UiPath's computer-use work has been reported on public desktop-agent benchmarks such as OSWorld (check the current standing before the interview). What do such benchmarks miss about enterprise desktop automation?
   - How would you build an internal benchmark for legacy apps (Citrix, SAP GUI, mainframe terminals)?
64. [Likely] OSWorld now runs a "Verified" leaderboard where the maintainers run the agent code themselves instead of accepting self-reported scores. How should UiPath report computer-use results so that enterprise buyers and researchers trust them? (context: [15])
   - What would you publish beyond a single success rate (cost per task, steps, failure taxonomy, variance across runs)?
65. [Likely] Design a computer-use agent that uses existing RPA selectors and recorded workflows as priors instead of acting from raw pixels.
   - How do you recover when the UI changes and a selector breaks?
   - How do you decide between a grounding model, the accessibility tree and OCR for a given screen?
66. [Likely] How would you turn the large base of existing RPA workflows into training data or tools for agents?
   - What are the licensing and tenant-isolation constraints?
67. [Likely] Design "self-healing" automation: an agent that fixes broken robot workflows at runtime. How do you stop it from silently doing the wrong thing?
68. [Likely] How should Context Grounding (enterprise retrieval) be built so agents get the right policy document and not a stale or out-of-scope one?
   - How do you evaluate retrieval quality per tenant without labelled queries?
69. [Likely] Context Grounding indexes live in Orchestrator folders, and a search can fall back to tenant-wide system indexes when the named index is not found. What permission-leak and wrong-context risks does that create, and how would you design retrieval to enforce document-level permissions? (context: [6][7])
70. [Likely] Context Grounding's Deep RAG runs a research-style synthesis across many documents in an index, with a choice of citation mode. How would you evaluate the faithfulness and citation accuracy of a multi-document synthesis? (context: [6])
   - When should an agent be forced to cite, and what should happen when a claim has no supporting passage?
71. [Likely] Batch Transform applies one LLM prompt to every row of a CSV in an index, adds new output columns, and can optionally ground on web search. Design quality control for row-level LLM transformations over a million rows. (context: [6])
   - How do you sample rows for review, and how do you detect systematic errors?
   - What new risks does web-search grounding bring into an enterprise data pipeline?
72. [Likely] Unified search supports a semantic mode and a tabular mode, and can return partial results while ingestion is still running. How would you route a natural-language query between unstructured and tabular retrieval, and when is it unsafe for an agent to act on partial results? (context: [6])
73. [Likely] Context Grounding can build an ephemeral index for one run's attachments. When is per-run indexing better than putting the whole document set into a long-context model, and how would you decide by cost and quality? (context: [6])
74. [Likely] Autopilot helps developers and business users build automations from natural language. How would you measure whether Autopilot-generated workflows are correct?
   - Would you use execution-based evaluation, and how would you sandbox it?
75. [Likely] How would you use Process Mining and Task Mining data to find which processes are worth turning into agents?
   - How do you estimate the ROI of agentifying a process before building it?
76. [Likely] What are the main failure modes of multi-step enterprise agents you have seen or expect, and how would you instrument for each?
   - Which ones can a better model fix, and which need system design?
77. [Likely] How would you design agent memory across a long-running case that lasts weeks (a claims process, for example)?
78. [Likely] Should UiPath train its own agent or action models, or orchestrate third-party ones? Lay out a decision framework.
79. [Likely] How would you support open agent protocols (such as MCP and agent-to-agent interop) while keeping governance through the Trust Layer? (context: [13])
   - What new attack surface does a customer-hosted MCP server open?

## 6. Peak acquisition and decision intelligence

### System diagram: decision intelligence joined to automation

```
  ERP / POS / supply data ──> ┌────────────────────────┐    recommendations    ┌──────────────────────┐
                              │ Peak-style decision     │ ────────────────────> │ Maestro / agents     │
  Market + price signals  ──> │ models: demand forecast,│                       │ execute: update price│
                              │ pricing, inventory opt. │ <──────────────────── │ in ERP, reorder, flag│
                              └───────────┬────────────┘   outcome feedback     │ human for approval   │
                                          │                                     └──────────────────────┘
                                          ▼
                              ┌────────────────────────┐
                              │ Experimentation / causal│
                              │ impact measurement      │
                              └────────────────────────┘
```

80. [Likely] UiPath announced the acquisition of Peak, a decision-intelligence company focused on areas such as pricing and inventory, in 2025. Why would an automation company buy a decision-intelligence company? (The deal details could not be verified in this pass; check them before the interview.)
   - What does "closing the loop from decision to action" mean in system terms?
81. [Likely] Design a system where a pricing model recommends a price change and an agent carries it out in the ERP. Where do you put humans, guardrails and rollback?
   - How do you stop a feedback loop where the model's own actions corrupt its training data?
82. [Likely] How would you measure the business impact of automated pricing or replenishment decisions when you cannot run a clean A/B test across stores?
   - Which causal-inference designs would you consider?
83. [Likely] How would you combine classical forecasting and optimisation models with LLM agents? Which part should never be done by an LLM?
84. [Likely] What are the main technical integration risks when merging an acquired ML stack (different data platform, model serving and MLOps) into UiPath's platform?
   - What would you unify in the first six months, and what would you leave alone?
85. [Likely] How would you expose Peak-style decision models as tools inside Agent Builder so a business user can use them safely?

## 7. UiPath research and publications

86. [Likely] Have you read any UiPath research papers, technical blog posts or open-source releases? Pick one and critique its method and evaluation. (UiPath's open-source agent SDKs, guardrails and evaluation code on GitHub are a fair target; find UiPath-affiliated papers on arXiv before the interview.) (context: [1][8])
   - What follow-up experiment would you run?
87. [Likely] UiPath is a product company, not a lab. How do you balance publishing with shipping, and when is publishing worth it for the business?
   - How do you protect customer data and competitive advantage in a paper?
88. [Likely] Which public benchmarks matter for UiPath's products (document understanding, GUI and computer-use agents, tool use, enterprise RAG), and which are misleading?
   - Propose a new benchmark UiPath could release, and explain why it would be credible.
89. [Likely] How would you set up a research agenda that shares components across DU, IXP, agents and computer use?
   - Which shared foundation (a multimodal backbone, an eval harness or a data flywheel) gives the most leverage?
90. [Likely] How would you turn a strong arXiv result (a new GUI-grounding method, for example) into a production feature within two quarters?
   - What would make you stop the effort early?

## 8. Cross-cutting ML system design probes on UiPath's stack

### System diagram: the data and evaluation flywheel

```
  Production runs (DU, IXP, agents, robots)
        │ traces, confidences, outcomes
        ▼
  ┌──────────────────┐   ┌─────────────────────┐   ┌───────────────────┐   ┌───────────────────┐
  │ Telemetry +      │──>│ Human corrections   │──>│ Curated eval sets │──>│ Train / fine-tune │
  │ consent filters  │   │ (validation, approval│  │ per task, tenant, │   │ / distil / prompt │
  │ (tenant opt-in)  │   │  outcomes)           │  │ and language      │   │ optimisation      │
  └──────────────────┘   └─────────────────────┘   └───────────────────┘   └─────────┬─────────┘
        ▲                                                                            │
        │                 ┌─────────────────────┐   ┌───────────────────┐            │
        └─────────────────│ Staged rollout:     │<──│ Offline gates +   │<───────────┘
                          │ canary tenants,     │   │ regression suites │
                          │ per-tenant pinning  │   │ (no tenant worse) │
                          └─────────────────────┘   └───────────────────┘
```

91. [Likely] Design the data flywheel in the diagram for a multi-tenant SaaS product where many tenants do not consent to data use for training.
   - What can you learn from non-consenting tenants (aggregate metrics, federated methods), and what is off limits?
92. [Likely] What release gate would you set for a model that serves thousands of tenants: average improvement, or "no tenant gets worse by more than X"?
   - How do you get statistical power per tenant?
93. [Likely] How do you run the same model family across Automation Cloud, dedicated cloud, on-premises Automation Suite and air-gapped deployments?
   - What breaks first when you cannot call home for telemetry?
   - Air-gapped Automation Suite customers must host their own package feeds even for Python agent dependencies. How would you ship model weights, guardrail models and evaluation suites into such an environment, and keep them up to date? (context: [11])
94. [Likely] Pick one UiPath AI product and design its online monitoring: drift, quality, cost and safety. What alerts would page someone at 2 a.m.?
95. [Likely] How would you estimate and lower the cost per automated transaction across OCR, the LLM, retries and human review?
   - Which levers have the largest effect?
96. [Likely] Design human-in-the-loop thresholds that adapt over time as the model improves, without letting error rates creep up unnoticed.
97. [Likely] How would you handle the EU AI Act and similar regulation for agents that make decisions in HR, credit or insurance processes run on UiPath?
   - What does a "high-risk" classification change in your system design?
98. [Likely] Sketch, end to end, an "invoice exception agent". It reads an invoice with DocPath, matches it to a PO through a robot, resolves mismatches by emailing the supplier with CommPath, uses a Peak-style model to decide whether to accept a price variance, and escalates to a human. Where would you invest science effort first?
   - Which component's error dominates end-to-end accuracy, and how would you find out?
   - How does the whole chain behave when the LLM provider has an outage?

## 9. Questions you should be ready to ask the interviewer

99. [Likely] (To ask) How is applied science organised across DU, IXP, Agents and the platform, and where would this role sit?
100. [Likely] (To ask) Which model decisions are centralised (gateway, guardrails, shared backbones), and which are owned by product teams?
101. [Likely] (To ask) How much of the roadmap depends on in-house models (DocPath, CommPath, computer-use models) compared with third-party LLMs over the next 18 months?
102. [Likely] (To ask) How is the Peak team being integrated, and what joint science work is planned?
103. [Likely] (To ask) What does success in the first year look like for a Principal Applied Scientist here?

## References

Sources 1-16 were fetched and read in the review pass (October 2026) as raw GitHub files. They back the product facts behind the *(context: [n])* markers. None of them is a candidate interview report. Sources 17-19 are official UiPath pages that could not be reached from this environment (DNS failure or proxy denial); they are listed as the places to check DocPath, CommPath, the AI Trust Layer admin guide and the Peak acquisition, and no question relies on them.

1. UiPath, "uipath-python" README: Python SDK and CLI with services for processes, assets, buckets, context grounding, jobs, queues and Action Center tasks (official; fetched). https://raw.githubusercontent.com/UiPath/uipath-python/main/README.md
2. UiPath, LLM Gateway service source, `uipath-platform` package: OpenAI-compatible and normalized APIs, model constants, schema cleanup for structured output, requesting-product and requesting-feature headers, reasoning-model handling (official; fetched). https://raw.githubusercontent.com/UiPath/uipath-python/main/packages/uipath-platform/src/uipath/platform/chat/_llm_gateway_service.py
3. UiPath, "Chat Models" documentation, `uipath-langchain-python`: OpenAI, Anthropic on Bedrock and Gemini on Vertex through the LLM Gateway; models subject to AI Trust Layer tenant policy; usage consumes Agent Units; passthrough versus normalized classes (official; fetched). https://raw.githubusercontent.com/UiPath/uipath-langchain-python/main/docs/chat_models.md
4. UiPath, "Guardrails" documentation, `uipath-python`: tool, LLM and agent scopes; PRE and POST stages; PII, harmful content, user prompt attacks, intellectual property, LLM-as-judge, custom and Bring Your Own Guardrail validators; log, block and custom actions (official; fetched). https://raw.githubusercontent.com/UiPath/uipath-python/main/packages/uipath/docs/core/guardrails.md
5. UiPath, Documents service source, `uipath-platform` package: pretrained, DU Modern and IXP project types; published project tags and versions; classification with confidences; asynchronous IXP extraction; classification and extraction validation (official; fetched). https://raw.githubusercontent.com/UiPath/uipath-python/main/packages/uipath-platform/src/uipath/platform/documents/_documents_service.py
6. UiPath, Context Grounding service source, `uipath-platform` package: folder-scoped and system indexes, ephemeral indexes, Deep RAG with citation modes, Batch Transform with optional web-search grounding, unified semantic and tabular search, search during ingestion (official; fetched). https://raw.githubusercontent.com/UiPath/uipath-python/main/packages/uipath-platform/src/uipath/platform/context_grounding/_context_grounding_service.py
7. UiPath, "Context Grounding" documentation, `uipath-langchain-python`: indexes created in Orchestrator folders from storage buckets; ContextGroundingRetriever (official; fetched). https://raw.githubusercontent.com/UiPath/uipath-langchain-python/main/docs/context_grounding.md
8. UiPath, "Agent Evaluations" documentation, `uipath-python`: output-based, trajectory-based and custom evaluators (official; fetched). https://raw.githubusercontent.com/UiPath/uipath-python/main/packages/uipath/docs/eval/index.md
9. UiPath, "LLM Judge Output Evaluators" documentation, `uipath-python`: semantic-similarity and strict-JSON judges with a 0-1 score and justification (official; fetched). https://raw.githubusercontent.com/UiPath/uipath-python/main/packages/uipath/docs/eval/llm_judge_output.md
10. UiPath, "Human In The Loop" documentation, `uipath-langchain-python`: agents suspend on `interrupt(...)` and resume after an Action Center task (official; fetched). https://raw.githubusercontent.com/UiPath/uipath-langchain-python/main/docs/human_in_the_loop.md
11. UiPath, "Automation Suite" documentation, `uipath-python`: package feeds for air-gapped deployments (official; fetched). https://raw.githubusercontent.com/UiPath/uipath-python/main/packages/uipath/docs/AutomationSuite.md
12. UiPath (Re:infer), `re` CLI README: "the official command line interface for UiPath IXP", linked to the Communications Mining developer guide; manages sources and datasets (official; fetched). https://raw.githubusercontent.com/reinfer/cli/master/README.md
13. UiPath, "uipath-mcp-python" README: hosting MCP servers on the UiPath platform (official; fetched). https://raw.githubusercontent.com/UiPath/uipath-mcp-python/main/README.md
14. UiPath, "uipath-langchain-python" README: LangGraph agents on UiPath, human-in-the-loop through Action Center, built on the UiPath Runtime Protocol (official; fetched). https://raw.githubusercontent.com/UiPath/uipath-langchain-python/main/README.md
15. XLANG Lab, "OSWorld" README: OSWorld-Verified release (2025-07-28) and a verified leaderboard run by the maintainers (fetched). https://raw.githubusercontent.com/xlang-ai/OSWorld/main/README.md
16. UiPath, "Guardrails" documentation, `uipath-langchain-python`: middleware and decorator patterns with before/after agent, model and tool-call hooks (official; fetched). https://raw.githubusercontent.com/UiPath/uipath-langchain-python/main/docs/guardrails.md
17. UiPath Documentation portal, docs.uipath.com (Document Understanding, IXP, Communications Mining, AI Trust Layer, Agents and Maestro; not reachable in this pass). https://docs.uipath.com
18. UiPath Blog, uipath.com (product, AI and acquisition announcements, including Peak; not reachable in this pass). https://www.uipath.com/blog
19. UiPath Careers, uipath.com (official careers page; not reachable in this pass). https://www.uipath.com/company/careers
