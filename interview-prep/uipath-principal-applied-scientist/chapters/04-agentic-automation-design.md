# 04. Applied Science System Design: Agentic Automation, Computer-Use Agents, Document Understanding, Communications Mining, Process Mining and Orchestration

> **How to read this chapter.** This is the ML system design chapter for UiPath's own product surface. Each section opens with a plain-text block diagram of the system you are likely to be asked to design. The numbered questions follow, with follow-up probes as indented sub-bullets. The chapter has questions only, with no answers. The diagrams are plain text so they render the same in Markdown and in a PDF export.
>
> **Tagging system.** Every question carries one of two tags:
> - **[Reported] [n]**: a candidate report or interview question bank links this question to UiPath. The number points to the References list.
> - **[Likely]**: an inference from the role, the level and UiPath's products: the agentic automation platform (agents, robots, humans), computer-use agents, Document Understanding and IXP, Communications Mining, Process Mining, Autopilot, Agent Builder and coded agents, Maestro orchestration and the AI Trust Layer. No candidate has reported it. A **[Likely] [n]** tag means the *premise* of the question (a product feature, an SDK behaviour, a benchmark rule) is confirmed by source [n]. It does not mean a candidate was asked it.
> - Indented sub-bullets are follow-up probes an interviewer may use to push a strong first answer toward the principal bar.
>
> **Research limits (please read).** This review pass (October 2026) could not reach any candidate-report site. The shared web-search budget for the workflow was used up. The egress proxy refused connections to glassdoor.com, uipath.com, docs.uipath.com and arxiv.org (HTTP 403 by policy). GitHub raw content was reachable, so the review read UiPath's own public SDK repositories (documentation, samples and source) and the OSWorld repository. Those sources confirm product behaviour, not interview content. So **no question in sections A-L is tagged [Reported]**, and this chapter does not invent reports. A later deepening pass (see "Deepening pass (October 2026)" near the end) found a few candidate reports through search snippets and tags those questions [Reported]. Product names such as Maestro, Agent Builder, Autopilot and IXP should still be checked against current UiPath documentation before the interview.

## Format notes for this round

| Item | Expected shape (inferred, not confirmed) |
|---|---|
| Round name | ML / applied science system design, sometimes merged with a product or "architecture" round |
| Duration | 60 min: about 5 min framing, 35-40 min design, 10-15 min deep-dive probes, 5 min your questions |
| Interviewer | Principal or distinguished scientist, or an AI platform architect; sometimes paired with a product lead |
| Prompt style | Open-ended and UiPath-shaped: "design an agent that processes invoices end to end", "design a computer-use agent for legacy desktop apps", "design the eval system for agents" |
| Principal-level bar | You set the problem framing and success metrics yourself. You separate deterministic (robot / coded function) work from probabilistic (agent / LLM) work. You design evaluation and the human-in-the-loop path as first-class parts. You reason about cost, latency, on-prem and regulated deployment, and governance. You define a multi-quarter roadmap and say what research is still open. |
| Common failure modes | Jumping to "use GPT plus RAG" without workload analysis; no offline/online eval plan; ignoring exception handling, retries and human escalation; ignoring the existing RPA asset base (selectors, workflows, queues) |

### Reference diagram: the agentic automation stack an interviewer has in mind

```
                         ┌───────────────────────────────────────────┐
  Business event ──────> │  ORCHESTRATION (Maestro-style, BPMN-like) │ <── Process Mining:
  (email, doc, ticket,   │  process model, state, SLAs, routing      │     discovered process
   API call, schedule)   └───────┬───────────────┬───────────────┬───┘     + KPIs + conformance
                                 │               │               │
                    ┌────────────▼───┐   ┌───────▼───────┐  ┌────▼────────────┐
                    │  AGENTS        │   │  ROBOTS       │  │  HUMANS         │
                    │  LLM reasoning │   │  deterministic│  │  Action Center/ │
                    │  tool calls,   │   │  RPA, APIs,   │  │  approvals,     │
                    │  context/RAG   │   │  coded funcs  │  │  exceptions     │
                    └───┬────────┬───┘   └───────┬───────┘  └────┬────────────┘
                        │        │               │               │
              ┌─────────▼──┐ ┌───▼──────────┐    │      feedback / labels
              │ Specialised│ │ Computer-use │    │               │
              │ models: DU,│ │ agent (screen│    │               ▼
              │ IXP, Comms │ │ understanding│    │      ┌─────────────────┐
              │ classifiers│ │ + actions)   │    │      │ Eval + learning │
              └────────────┘ └──────────────┘    │      │ loop (traces,   │
                                                 │      │ datasets, judges)│
   ┌─────────────────────────────────────────────▼──────┴─────────────────┴──┐
   │ AI TRUST LAYER: LLM gateway, guardrails, PII masking, audit logs, quotas │
   └─────────────────────────────────────────────────────────────────────────┘
```

## A. Agentic automation: agents + robots + humans

```
   Task ──> Router ──┬──> deterministic? ──> Robot (RPA workflow / coded function / API)
                     ├──> needs judgement? ──> Agent (plan, tools, memory)
                     │                           │  confidence < τ or policy hit
                     │                           ▼
                     └──> high risk / ambiguous ──> Human task (approve / correct)
                                                     │
                                       corrections ──┴──> eval set + fine-tune / prompt update
```

1. [Likely] Design an end-to-end agentic system that resolves inbound customer-service emails for an insurer, using agents, existing RPA robots and human reviewers.
     - How do you decide which steps stay deterministic robots and which become agent steps?
     - What is the unit of work, and how do you define "resolved" as a metric?
     - Where exactly does a human get pulled in, and what context do they see?
2. [Likely] [5] UiPath's SDK separates "coded functions" (deterministic, typed input/output, no LLM) from "coded agents" (an LLM reasoning loop that chooses tools). Give a decision procedure for which one a new automation should be, and design the path for promoting a function to an agent, or demoting an agent back to a function, once you have production traces.
     - What evidence from traces would justify the demotion?
     - How do you keep the typed input/output contract stable for the callers when the implementation changes?
3. [Likely] Given a company with 500 existing RPA workflows, how would you introduce agents without breaking what works?
     - Would agents call robots as tools, or robots call agents? What changes in failure handling?
     - How do you prioritise which workflows to "agentify" first?
4. [Likely] [10] [11] Design the confidence and escalation policy for an agent that takes financial actions (refunds, payments) inside an enterprise process.
     - How do you calibrate an LLM agent's confidence when there are no token-level probabilities you can trust?
     - How do you set thresholds per action type and per customer risk appetite?
     - How do you stop humans from rubber-stamping escalations?
5. [Likely] [11] UiPath publishes an anti-money-laundering triage sample in which the LLM only extracts facts and writes the rationale, while a separate typed classifier makes every decision (red flags, risk level, disposition). When is this "LLM reads and writes, a calibrated model decides" split better than letting the agent decide?
     - How do you calibrate and monitor the decision model, and what happens when the LLM's extraction is wrong upstream?
     - How would you explain one disposition to a regulator?
6. [Likely] [3] [4] How would you design the tool interface layer so that an agent can safely invoke UiPath robots, APIs, Integration Service connectors and MCP servers?
     - How do you describe tools so the model picks the right one among hundreds?
     - How do you handle tool schemas that change under the agent?
     - How do you enforce least privilege per agent and per run?
7. [Likely] [4] UiPath lets customers host their own MCP servers on the platform and expose them to agents. Design the governance for third-party MCP tools: registration, trust review, versioning, sandboxing and per-tenant allow lists.
     - How do you detect that an MCP tool's behaviour changed even though its schema did not?
     - What do you do when a tool's description itself contains an injected instruction?
8. [Likely] Design memory for a long-running enterprise agent that handles a case over days or weeks.
     - What goes into short-term context, case state and long-term memory?
     - How do you prevent stale or poisoned memory from driving actions?
9. [Likely] [8] [2] Design durable suspend and resume for agents that wait days for a human approval, a robot job or an external callback. UiPath's runtime checkpoints agent state at an `interrupt()` and resumes it later.
     - What exactly do you checkpoint, and where, so a resume survives a container restart?
     - The agent's code is redeployed while 3,000 runs are suspended. How do you migrate or pin their checkpoints?
     - How do you guarantee a side effect performed just before the suspend is not repeated on resume?
10. [Likely] [9] Design an event-triggered, long-running agent that sleeps on a connector event (for example, a new email in a watched mailbox), wakes to process it, then suspends again.
     - How do you handle duplicate events, out-of-order events and a burst of 10,000 events after an outage?
     - How do you stop one poison message from blocking the agent forever?
11. [Likely] [13] How would you design a multi-agent system for procure-to-pay, and when is multi-agent (for example, a supervisor routing to specialist agents) worse than a single agent?
     - How do agents hand off context, and who owns the global state?
     - How do you debug a failure that spans three agents and two robots?
12. [Likely] What is your framework for deciding the autonomy level of an agent in a given process?
     - What evidence moves a process from "human approves every action" to "human reviews a sample"?
13. [Likely] [6] How would you make an agentic workflow idempotent and recoverable when a step fails halfway, for example after a payment API call times out?
     - How do you design compensation or rollback for actions an agent already took?
14. [Likely] [7] [12] Design the human-in-the-loop task experience so that corrections become training or evaluation data.
     - What do you log from a human correction, and how do you de-bias it?
     - How do you use corrections without overfitting to one reviewer's preferences?
15. [Likely] [12] Design the human task inbox at scale: 50,000 open approval tasks a day from agents across many processes, with different SLAs and reviewer skills.
     - How do you prioritise and assign tasks, and when do you batch similar approvals?
     - How do you measure reviewer fatigue and the quality of approvals, not just throughput?
16. [Likely] How would you measure ROI for an agentic automation, compared with the RPA baseline and the fully manual baseline?
     - Which metrics would you show a CFO, and which would you show the science team?
17. [Likely] [15] How would you defend an agent against prompt injection that arrives through emails, documents or web pages it processes?
     - How do you separate instructions from data in the agent's context?
     - What runtime guardrails would you add beyond the model itself?
18. [Likely] [3] Design the system that lets an enterprise run the same agent across multiple LLM providers, or on-prem models, with consistent behaviour.
     - How do you detect and manage regression when a provider updates a model?
19. [Likely] [2] [3] [5] UiPath's runtime hosts agents written in LangGraph, LlamaIndex, the OpenAI Agents SDK, Google ADK, PydanticAI, the Claude Agent SDK and its own framework behind one runtime protocol. What must that protocol standardise (inputs/outputs, tracing, errors, human-in-the-loop, resume) and what should it deliberately leave to the framework?
     - How do you evaluate and compare two agents built on different frameworks fairly?
     - What breaks first when a framework releases a breaking change?

## B. Computer-use agents (ScreenAgent-style UI automation)

```
  Goal ("file this expense in SAP GUI")
     │
     ▼
  ┌──────────────┐   screenshot + UI tree / accessibility / selectors
  │ Perception   │ <─────────────────────────────────────────── Desktop / Browser / Citrix
  │ (VLM, OCR,   │                                                     ▲
  │  grounding)  │                                                     │ click/type/scroll
  └──────┬───────┘                                                     │
         ▼                                                             │
  ┌──────────────┐   plan / reflect    ┌──────────────┐   verify   ┌────┴─────────┐
  │ Planner LLM  │ ──────────────────> │ Action model │ ─────────> │ Executor     │
  │ (task graph) │ <── state diff ──── │ (grounded    │            │ (robot runtime│
  └──────────────┘                     │  coordinates)│            │ + safety gate)│
                                       └──────────────┘            └──────────────┘
         └──> trajectory log ──> replay / eval (OSWorld-like suites + customer apps)
```

20. [Likely] Design a computer-use agent that can operate legacy desktop applications (for example SAP GUI or a Citrix-virtualised app) where no API or DOM is available.
     - What inputs do you use: pixels only, accessibility tree, UiPath selectors, OCR? How do you fuse them?
     - How do you ground a natural-language target ("the Submit button") to coordinates reliably?
21. [Likely] How would you combine a generalist computer-use model with UiPath's existing deterministic selectors and recorded workflows?
     - When should the agent "compile" a successful trajectory into a deterministic robot workflow?
     - How would you detect that a compiled workflow has drifted and needs the agent again?
     - If an agent proposes a repaired selector after a UI change, how do you verify the repair before it is committed, so the robot does not silently click the wrong element?
22. [Likely] [21] [22] How would you evaluate a computer-use agent, and how far would you trust public benchmarks such as OSWorld for enterprise use?
     - How do you build an internal benchmark of enterprise apps with verifiable end states?
     - How do you score partial progress versus binary task success?
     - How do you prevent benchmark contamination and overfitting to the leaderboard?
23. [Likely] [21] OSWorld had to be re-released as OSWorld-Verified to fix community-reported task errors, and its maintainers now run verified submissions themselves. Design the governance for an internal computer-use benchmark so you do not repeat that history.
     - How do you detect flaky tasks and environment noise, and separate them from real capability changes?
     - How many reruns per task do you need before you report a score difference, and how do you report variance?
24. [Likely] [22] OSWorld grades each task with an execution-based script that inspects the final machine state. Design state verifiers for enterprise tasks where the true end state lives in a backend you cannot read directly (SAP posting, Salesforce record, mainframe screen).
     - When is a screen-based check acceptable, and when must you query the system of record?
     - How do you keep verifiers correct as the application is upgraded?
25. [Likely] Design the safety layer for a computer-use agent that has a real user's desktop session.
     - Which actions require confirmation, and how does the agent know an action is irreversible?
     - How do you handle unexpected dialogs, pop-ups and login prompts?
26. [Likely] Latency is 5-10 seconds per step and tasks take 40 steps. How do you make this usable and affordable?
     - Where would you use a smaller grounding model versus a frontier planner?
     - Which steps can be cached or replayed without calling a model?
27. [Likely] Build the break-even model: for a given process, at what volume, UI change rate and error cost is a computer-use agent cheaper than a maintained RPA robot, and when is it cheaper than a human?
     - Which inputs to this model would you measure first, and from which telemetry?
28. [Likely] How would you collect training data for UI grounding and action prediction at scale?
     - How would you use existing robot executions and selector data as weak supervision?
     - What are the privacy constraints of using customer screenshots, and how do you design around them?
29. [Likely] How does the agent recover when it gets lost (wrong screen, loop, state it did not expect)?
     - How do you detect loops and lack of progress automatically?
     - When should it give up and hand off to a human, and with what summary?
30. [Likely] How would you handle high-DPI, multi-monitor, localisation (non-English UIs) and theme variation in perception?
31. [Likely] Compare a pixel-only VLM agent with a structured-UI (accessibility tree / DOM) agent for enterprise automation. Where does each break?
32. [Likely] How would you fine-tune or distil a computer-use model for a single customer's application portfolio, and how do you keep that maintainable?

## C. Document Understanding and IXP pipelines

```
  Ingest (scan, PDF, email attachment)
     │
     ▼
  Digitise: OCR / layout ──> Classify (doc type, split multi-doc bundles)
     │                                 │
     ▼                                 ▼
  Extract fields & tables  <── model choice: specialised ML extractor | generative LLM | hybrid
     │
     ▼
  Validate: business rules, cross-doc checks, confidence per field
     │                    │
     │ high confidence    │ low confidence / rule fail
     ▼                    ▼
  Downstream robot     Validation action (human) ──> labelled data ──> retrain / prompt eval
```

33. [Likely] Design an invoice-processing pipeline that handles 1M documents a month across 40 languages and thousands of vendor layouts.
     - Where do you use a trained extractor and where a generative model?
     - How do you hit a straight-through-processing target while keeping field-level accuracy above a contract threshold?
34. [Likely] [20] UiPath's SDK exposes classification, extraction (DU Modern and IXP projects) and human validation actions as separate, non-blocking start/retrieve operations. Design the asynchronous pipeline around them for 1M documents a month.
     - How do you track a document across classify, extract, validate and downstream steps when any of them can fail or time out?
     - What decides that a document goes to a validation action, and at which granularity (document, page, field)?
35. [Likely] Suppose one extraction product must serve both structured documents and unstructured inputs such as emails and long contracts. What shared representation, schema language and confidence model would you design, and what gets worse when you merge the two workloads?
36. [Likely] How would you produce field-level confidence scores that are actually calibrated, for both classic extractors and LLM-based extraction?
     - How do you set per-field thresholds that trade human review cost against error cost?
     - How do you show calibration to a customer?
37. [Likely] Design document classification and splitting for a 300-page mortgage packet containing 25 document types.
     - How do you handle unseen document types at run time?
38. [Likely] Design generative extraction from 200-page commercial contracts where the answer to one field (for example, the termination notice period) depends on definitions, amendments and cross-references spread through the document.
     - How do you chunk, retrieve and resolve conflicting clauses?
     - How do you evaluate this when ground truth needs a lawyer's time?
39. [Likely] How would you extract complex tables (line items spanning pages, merged cells, nested headers) robustly?
     - How do you evaluate table extraction: cell-level, row-level, or downstream reconciliation?
40. [Likely] A customer has 50 labelled examples of a new document type. Design the fastest route to production quality.
     - When is few-shot prompting enough, and when do you fine-tune?
     - How do you use active learning to pick the next documents to label?
41. [Likely] [20] How would you design the human validation loop so that corrections improve the model without introducing label noise?
     - How do you detect inconsistent annotators?
42. [Likely] Handwriting, low-quality scans and stamps degrade OCR. Where in the pipeline do you invest, and how do you measure the gain end to end?
43. [Likely] How would you ground every extracted value back to a location in the document, and why does that matter for audit?
     - How do you stop a generative extractor from producing a plausible value that is not in the document?
44. [Likely] Design the evaluation harness for Document Understanding that runs on every model or prompt change.
     - What is in the golden set, how is it stratified, and how do you keep it from going stale?
45. [Likely] How would you run Document Understanding on-prem or in an air-gapped environment with limited GPU capacity?
     - What do you distil or quantise, and what accuracy loss is acceptable?
46. [Likely] Design cross-document reasoning: match an invoice to a purchase order and a goods receipt and explain mismatches.
47. [Likely] How would you detect drift in incoming document distributions (new vendors, new templates) and trigger action?

## D. Communications Mining (email, chat and ticket understanding)

```
  Channels (email, chat, tickets, call transcripts)
     │
     ▼
  Thread reconstruction + PII masking
     │
     ▼
  Taxonomy (hierarchical labels, user-defined) ── Few-shot / active-learning training UI
     │
     ▼
  Inference: labels + extracted fields + sentiment/tone + quality-of-service signals
     │
     ├──> Analytics (volumes, drivers, SLA)
     └──> Automation trigger: robot / agent / human queue
```

48. [Likely] Design a system that turns a bank's operations mailbox (200k emails per month) into structured requests that trigger automations.
     - How do you handle one email with several requests in it?
     - How do you handle threads where the intent changes mid-thread?
49. [Likely] Customers define their own label taxonomies with very few examples. How do you design the model and training loop?
     - How do you pick examples for the user to label (uncertainty, diversity, coverage)?
     - How do you tell the user when a label is "good enough" to automate on?
50. [Likely] [10] A UiPath ticket-classification sample has the LLM return a label with a self-reported confidence (for example 0.9) and sends it for human approval. Why is verbalised LLM confidence a weak basis for automation thresholds, and what would you put in its place per customer taxonomy?
     - How do you measure calibration per label when some labels have ten examples?
51. [Likely] How would you decide between a fine-tuned encoder classifier, an embedding plus kNN approach, and an LLM prompt for intent classification at this scale?
     - What are the cost, latency and accuracy implications per million messages?
52. [Likely] Month-end volume is 8x the daily average and some message types have a 15-minute SLA. Design capacity, prioritisation and graceful degradation for the inference and automation-trigger path.
53. [Likely] Design field extraction from free-text messages (account numbers, amounts, dates) with validation against systems of record.
54. [Likely] How would you discover new, emerging topics in communications that nobody has labelled yet?
     - How do you present clusters so a business user can act on them?
55. [Likely] How do you evaluate a taxonomy-specific model when every customer's labels differ?
     - What platform-level metrics can you still track across customers?
56. [Likely] How would you measure and improve service quality signals (tone, urgency, chaser emails) in a way that operations teams trust?
57. [Likely] How would you handle multilingual mailboxes and code-switching?
58. [Likely] [9] How does Communications Mining feed an agent rather than a fixed robot, and what changes in the design?

## E. Process mining and task mining

```
  ERP / CRM event logs ──> Event log (case_id, activity, timestamp, attributes)
                                │
          ┌─────────────────────┼─────────────────────────┐
          ▼                     ▼                         ▼
   Process discovery     Conformance checking      Predictive monitoring
   (graph of variants)   (deviations vs model)     (delay, outcome, next step)
          │                     │                         │
          └──────────> Automation opportunities ranked by value ──> Agent / robot design
                                                         ▲
   Task mining (desktop interaction logs, screenshots) ──┘
```

59. [Likely] Design a system that takes ERP event logs and recommends which process steps to automate with robots, agents or neither.
     - How do you estimate the value and feasibility of each opportunity?
     - How do you avoid recommending automation of a step that only looks slow because of data quality?
60. [Likely] How would you use LLMs to help build event logs from messy enterprise data (choosing case IDs, mapping tables to activities)?
     - How do you validate an LLM-proposed data model?
61. [Likely] Design a predictive process monitoring system that forecasts SLA breaches for in-flight cases and triggers interventions.
     - How do you handle censoring and changing process versions?
     - How do you show the intervention actually caused the improvement?
62. [Likely] How would you design task mining that infers repetitive desktop tasks from user interaction logs while respecting privacy?
     - How do you segment continuous interaction streams into tasks?
     - What do you blur, drop or keep on-device?
63. [Likely] How would you close the loop so that process mining measures the impact of agents after deployment?
     - How do you represent agent steps (probabilistic, variable paths) in a process model?
     - How would you use conformance checking of agent trajectories against an approved process as an evaluation signal?
64. [Likely] Design root-cause analysis over process variants that a business analyst can trust.

## F. Autopilot (natural-language assistance across the platform)

```
  User prompt (in Studio / Apps / Assistant / Excel / email)
     │
     ▼
  Intent + context collection (open workflow, project, user permissions)
     │
     ▼
  Generation: workflow XAML / expressions / code / test cases / answers
     │
     ▼
  Validators: compile, static analysis, schema checks, sandbox run
     │                         │ fail
     ▼                         └──> repair loop (bounded) ──> fallback / ask user
  Suggestion shown to user ──> accept / edit / reject telemetry ──> eval + improvement
```

65. [Likely] Design a feature that generates a UiPath automation workflow from a natural-language description.
     - What is the target representation the model generates, and why?
     - How do you validate that the generated workflow is correct before the user runs it?
     - How do you measure quality beyond "it compiles"?
66. [Likely] How would you build an assistant that writes and fixes expressions and code inside Studio, given the project's context?
     - What context do you retrieve, and how do you fit it in the window?
67. [Likely] Design automated test-case generation for automations from requirements documents.
     - How do you measure coverage and usefulness of generated tests?
68. [Likely] How would you use accept/edit/reject telemetry to improve Autopilot without leaking customer data across tenants?
69. [Likely] Autopilot answers questions about a customer's processes and data. How do you design retrieval and permission enforcement?
     - How do you stop the assistant from revealing data the user cannot otherwise access?
70. [Likely] How would you prioritise model spend across Autopilot features with very different value per call?

## G. Agent Builder, coded agents and agent evaluation

```
  Author defines: goal prompt, tools (robots, APIs, MCP, other agents), context (indexes), guardrails
     │
     ▼
  Design-time: eval sets, simulated tools, output + trajectory evaluators, LLM-as-judge
     │
     ▼
  Publish (versioned) ──> Runtime (orchestrated) ──> traces ──> monitoring + online eval
     ▲                                                              │
     └──────────── failure clusters / suggested fixes ◄─────────────┘
```

71. [Likely] Design Agent Builder so that a citizen developer can build a reliable agent without being an ML expert.
     - Which decisions do you take away from the author, and which do you expose?
     - What defaults (models, guardrails, escalation) do you ship?
72. [Likely] [17] [18] Design the evaluation system for agents: test sets, simulated tools, judges and release gates.
     - How do you generate realistic test cases when the author has none?
     - How do you validate an LLM-as-judge, audit it against human labels, and detect judge drift?
     - How do you evaluate the trajectory (tool choice, order, arguments), not just the final answer?
73. [Likely] [17] [19] UiPath's SDK offers output evaluators (exact match, contains, JSON similarity, LLM judge) and trajectory evaluators (tool-call order, count, arguments, outputs, LLM-judge trajectory). For a refund agent, choose the evaluators per behaviour and design how their scores combine into a single release gate.
     - The tool-call-order evaluator gives partial credit through the longest common subsequence. When does that partial credit hide a dangerous ordering error (for example, refund before verification)?
     - Which checks must be hard gates rather than averaged scores?
74. [Likely] [18] How would you simulate tools (robots, APIs) during agent testing so evaluation is safe and repeatable? UiPath's trajectory-simulation evaluator mocks tool responses from instructions while the agent does not know it is being simulated.
     - How do you check that the simulator itself behaved as instructed, so a failure is the agent's fault and not the mock's?
     - If the simulator and the agent use the same model family, what correlated errors do you worry about?
75. [Likely] [17] Customers can register custom evaluators with their own domain logic. How do you let them do that safely, version the evaluators, and keep scores comparable across months?
76. [Likely] [8] [17] LLM evaluations are noisy. Design the continuous-integration gate that runs an agent's eval set on every prompt, tool or model change.
     - How many runs per case, and what statistical test, before you call something a regression?
     - How do you keep the eval cost per change under a budget?
77. [Likely] Design versioning and release management for agents: prompts, tools, models and context indexes all change independently.
     - What triggers a re-evaluation, and what blocks a release?
78. [Likely] How would you automatically cluster agent failures from production traces and suggest fixes to the author?
79. [Likely] [1] Design the context grounding (retrieval) component that agents use over enterprise content.
     - How do you handle permissions, freshness and document-level citations?
     - How do you evaluate retrieval separately from generation?
80. [Likely] How would you optimise an agent's prompt and tool descriptions automatically (prompt optimisation) without overfitting to the test set?
81. [Likely] How do you prevent an author from shipping an agent whose evaluation set is too small or too easy?

## H. Maestro-style orchestration of long-running agentic processes

```
  Process model (BPMN-like) ──> Engine: state machine per case, timers, SLAs
        │                                  │
        │       ┌──────────────────────────┼──────────────────────────┐
        ▼       ▼                          ▼                          ▼
   Agent tasks (probabilistic)   Robot tasks (deterministic)   Human tasks (queues)
        │                                  │                          │
        └───────── events, outputs, confidence, cost ─────────────────┘
                                   │
                                   ▼
             Case analytics + process mining on live instances ──> optimise routing / model
```

82. [Likely] [8] Design an orchestration engine for long-running processes that mix agents, robots and humans, with SLAs measured in days.
     - How do you persist and resume state across failures and deployments?
     - How do you version the process model while thousands of cases are in flight?
83. [Likely] [6] UiPath's docs warn that Maestro may retry a coded function after a transient failure, and advise checking for an existing result by a stable business key before writing. Extend that to agent tasks: on retry, the LLM may choose different tool arguments. How do you get exactly-once business effects?
     - Where do idempotency keys come from when the agent itself decides what to create?
84. [Likely] [6] The same docs recommend returning business errors as typed output fields so the orchestrator can route on them. Design the error and uncertainty taxonomy for agent tasks so that Maestro can route "not sure", "policy blocked", "tool down" and "bad input" differently.
85. [Likely] How would you design dynamic routing, where the orchestrator chooses agent, robot or human for each case at run time?
     - What signals feed the router, and how do you learn the routing policy safely (bandits, offline RL, rules)?
     - How do you evaluate a routing change before it goes live?
86. [Likely] Where should "agentic" decisions live: inside the agent, inside the process model, or in the orchestrator? Defend your boundaries.
87. [Likely] [5] [6] Design observability for an orchestrated agentic process: traces, cost per case, latency per step, quality per step. UiPath surfaces `@traced` spans in job traces and Maestro dashboards.
     - What is the trace schema that links process instance, agent run, LLM calls, robot jobs and human tasks?
     - How do you sample traces and strip PII without losing the ability to debug?
     - What would an on-call engineer look at first when case completion drops 10%?
88. [Likely] How would you handle a mid-process LLM outage or a degraded model in an orchestrated process?
     - What is the fallback per task type?
89. [Likely] How would you let an agent propose changes to the process model itself, and how would you govern that?
90. [Likely] How would you design case-level explainability so an auditor can reconstruct why a case was approved?

## I. Evaluation, reliability and quality across the platform

91. [Likely] Design a single evaluation framework that covers DU extraction, Communications Mining classification, agent trajectories and computer-use tasks.
     - What is shared (datasets, runners, judges, dashboards) and what must stay specific?
92. [Likely] How do you build and maintain golden datasets when customer data cannot leave the tenant?
     - How would you use synthetic data, and how would you check it is representative?
93. [Likely] How would you run online evaluation and A/B tests for agents in enterprise processes where traffic is low and errors are expensive?
     - When do you use shadow mode, interleaving or staged rollout instead of A/B?
94. [Likely] What is your reliability target for an agent step, and how do you compose step reliability into process-level reliability?
     - How does a 95%-reliable step compound across a 20-step process, and what do you do about it?
95. [Likely] How would you detect silent quality regressions in production when there is no ground truth for most cases?
96. [Likely] How would you measure and reduce hallucination in agents that write to systems of record?

## J. Trust, safety, governance and deployment constraints

97. [Likely] [15] Design an AI Trust Layer: a model gateway that enforces PII masking, policy, audit logging and usage quotas for every model call on the platform.
     - Where do you mask PII, and how do you unmask for downstream steps that need it?
     - What latency budget does the gateway get?
98. [Likely] [15] [16] UiPath's guardrails apply at three scopes (tool, LLM call, whole agent), before and/or after execution, with actions such as log or block, and include deterministic rule checks alongside model-based detectors for PII, harmful content and prompt attacks. Design the guardrail architecture for a regulated agent.
     - Which risks belong at which scope, and why?
     - How do you evaluate the guardrails themselves (false-block rate, missed attacks), and who owns that eval?
99. [Likely] [14] Customers can plug in their own guardrail vendor through a connector. How do you keep policy semantics consistent across vendors, and do you fail open or fail closed when the vendor is down?
100. [Likely] [3] How would you support customers who require on-prem, sovereign cloud or "bring your own model" deployments?
     - How do you keep evaluation results valid across different underlying models?
101. [Likely] How would you design permissions so an agent can never do more than the human or robot account it acts for?
102. [Likely] How would you meet EU AI Act or sector rules (financial services, healthcare) for an agentic process?
     - What documentation, logging and human oversight would you build in by default?
103. [Likely] Design the audit log for agent actions so it is complete, tamper-evident and still useful for debugging.
104. [Likely] How would you handle data residency when an agent's model provider is in a different region than the customer data?

## K. Cost, latency and model strategy

105. [Likely] How would you decide between frontier API models, open-weight models you host, and small specialised models for each part of the platform?
     - What would make you train a UiPath-specific model rather than use a third-party one?
106. [Likely] Design model routing (cascades) to cut cost by 60% while keeping quality on agentic tasks.
     - How do you decide when to escalate from a small model to a large one?
107. [Likely] Design per-run budgets for agents (tokens, tool calls, wall-clock time) with hard caps. What should an agent do when it hits a cap mid-task: stop, summarise and escalate, or degrade to a cheaper model?
     - How do you set the caps per agent without starving the hard cases?
108. [Likely] How would you build capacity planning and cost attribution per tenant, per agent and per process?
109. [Likely] How would you cache LLM results safely in agentic workflows where inputs look similar but differ in important ways?
110. [Likely] What is your approach to distilling frontier-model behaviour into a smaller model for DU or Communications Mining, and how do you check you lost nothing important?

## L. Principal-level framing and roadmap probes

111. [Likely] If you owned applied science for agentic automation, what would your 18-month research and product roadmap be?
     - Which bets are high-risk research, and which are engineering execution?
     - How would you staff and sequence them?
112. [Likely] What are the three biggest open research problems for enterprise agents, and how would UiPath's data position help solve them?
113. [Likely] Pick one UiPath AI product. What would you change in its architecture and why?
114. [Likely] How do you decide when a capability should be a platform service versus a product-specific model?
115. [Likely] [3] How do you keep the platform competitive as frontier models get better at computer use and tool use out of the box, and as customers bring agents built in any framework?
     - What is UiPath's durable advantage at each layer (orchestration, data, governance, models)?
116. [Likely] How would you design a feedback flywheel from millions of robot and agent executions while respecting tenant data boundaries?

## Deepening pass (October 2026): newly reported questions

> **Source note.** This pass used five web searches. Direct page fetches were not attempted. Every [Reported] tag below rests only on what the search-result snippets said. The closest match to this role is one September 2026 candidate write-up for a UiPath Machine Learning Engineer role on the Agents team [24]. It describes a 15-minute LLM and agent fundamentals block in a one-hour phone round, followed by a coding problem. No snippet described a principal-level applied-science onsite design round, so treat these as signals about the topics the Agents team probes, not as the onsite prompt itself. Sources [25] and [28] are commercial prep guides, not first-hand reports. They are tagged [Reported] only where the definition at the top of this chapter allows it ("interview question bank").

### Newly reported questions

117. [Reported] [24] How does an LLM actually call a tool, end to end, from the tool schema in the prompt to the parsed call and the returned result?
     - At principal level, how does this change when the platform hosts hundreds of tools (robots, APIs, MCP servers) behind one agent?
118. [Reported] [24] How do you control the format of a model's tool calls so that downstream code can always parse them?
     - When would you use constrained decoding or a JSON grammar, native function-calling APIs, or a validate-and-repair loop, and what does each cost in latency and quality?
119. [Reported] [24] How do you make tool calling stable across runs, prompts and model versions?
     - Which metrics would you track (wrong-tool rate, malformed-argument rate, unnecessary-call rate), and on what eval set?
120. [Reported] [24] A model keeps calling tools incorrectly (wrong tool, wrong arguments, hallucinated tools). How do you diagnose and fix it?
     - When is the fix a better tool description or schema, when is it few-shot examples or routing, and when is it fine-tuning?
     - How would you build the training data for a tool-calling fine-tune from production traces?
121. [Reported] [24] An agent's tool-calling loop runs too long or never ends. How do you detect and handle it?
     - How do you tell legitimate long work apart from a loop that is making no progress?
     - What should the agent hand back to the orchestrator or a human when you cut it off?
122. [Reported] [24] [30] Explain transformer internals, beam search and LoRA as they relate to building agents.
     - Why is beam search rarely used for tool-calling agents, and when would you use it?
     - When is a LoRA adapter per customer or per tool family the right design, and how do you serve many of them?
123. [Reported] [25] How would you design a RAG pipeline that minimises hallucinations in a document-processing workflow?
     - How do you verify that every generated field or answer is supported by a retrieved passage?
124. [Reported] [25] How would you compare embedding models for retrieval over long technical documents?
     - What offline eval would you build, and how would you account for chunking and context length?
125. [Reported] [25] Design a multi-agent system and explain the trade-offs between non-deterministic AI and enterprise reliability.
126. [Reported] [26] What is the time complexity of matrix multiplication, and how would you parallelise it across GPUs? (Reported for a Research Scientist loop in London, 2025.)
     - How does this bear on the latency and cost of serving the models behind an agent?
127. [Reported] [27] The hiring manager gives you a real problem from the team. Design the system end to end, covering the gateway, queues and storage, and justify SQL versus NoSQL. (Reported for an engineering loop in Cluj-Napoca. Expect the same "real team problem" format to be applied to agentic workloads.)
     - For an agent-run store (traces, checkpoints, case state), which data goes in which store, and why?
128. [Reported] [28] How does an AI agent plan, call tools and recover when a step fails, and how do you get DAG-style execution, state persistence and idempotent retries around it?

### Gap-closing questions for the principal bar

129. [Likely] [29] UiPath's agent guidance says the model picks a tool from its name, description and the context in the system prompt, and infers arguments from the input schema. Design an automated "tool description linter and optimiser" that finds descriptions likely to cause wrong tool choice or wrong arguments, before an agent ships.
     - How do you measure that a rewritten description helped without overfitting to the eval set?
130. [Likely] [29] Coded functions can now be exposed as deterministic agent tools. Design how the platform decides, per tool call, whether to let the LLM compute something itself or to force a deterministic tool, and how you measure the effect on errors.
131. [Likely] How would you design the agent's context management for a 60-step task whose tool outputs exceed the context window (summarisation, eviction, scratchpads, retrieval over the run's own history)?
     - How do you evaluate whether compression dropped a fact the agent later needed?
132. [Likely] Design an offline reinforcement-learning or preference-tuning programme that improves tool-calling agents from enterprise traces, with human corrections as reward signals.
     - How do you handle reward hacking and the bias of corrections that only cover escalated cases?
133. [Likely] How would you set and enforce a latency SLO (for example, p95 under 30 seconds) for a conversational agent that calls several robots and APIs in sequence?
     - Which calls can run in parallel or speculatively, and what is the risk of doing so?
134. [Likely] Design a red-teaming programme for agents before release: who runs it, which attack classes (indirect prompt injection, tool abuse, data exfiltration, privilege escalation), and what makes a finding block the release?
135. [Likely] An enterprise customer reports that an agent took a wrong, costly action last week. Walk through the incident review: what you reconstruct from traces, how you assign root cause between model, prompt, tool, data and process design, and what you change platform-wide.
136. [Likely] How would you write the technical strategy document that convinces product and engineering leadership to invest in a UiPath-trained computer-use or tool-calling model rather than relying on frontier APIs?
     - What evidence would you gather in the first 90 days, and what would make you abandon the bet?
137. [Likely] How do you mentor and set the scientific bar for a team of applied scientists shipping agent features on a product cadence (experiment design, eval rigor, negative results)?
138. [Likely] Design the benchmark that measures agent progress across UiPath's product lines quarter over quarter, so leadership can compare agent, robot and human performance on the same processes.
     - How do you keep it stable while the products and customer mix change?

## References

Sources 1-22 were fetched and read in this review pass (October 2026) as raw GitHub content. They confirm product and benchmark facts used in question premises. None of them is a candidate interview report.

1. UiPath. "UiPath Python SDK" README (services including processes, queues, Action Center tasks and context grounding). GitHub. https://raw.githubusercontent.com/UiPath/uipath-python/main/README.md
2. UiPath. "UiPath Runtime" README (runtime contracts: tracing, structured errors, durable execution, human-in-the-loop). GitHub. https://raw.githubusercontent.com/UiPath/uipath-runtime-python/main/README.md
3. UiPath. "UiPath Agents Runtime Integrations" README (Google ADK, LangChain, LlamaIndex, Microsoft Agent Framework, OpenAI Agents integrations). GitHub. https://raw.githubusercontent.com/UiPath/uipath-integrations-python/main/README.md
4. UiPath. "UiPath MCP Python SDK" README (hosting MCP servers on the UiPath platform). GitHub. https://raw.githubusercontent.com/UiPath/uipath-mcp-python/main/README.md
5. UiPath. "Python Coded Agents" (coded agents versus coded functions, supported frameworks, tracing in Maestro dashboards). GitHub. https://raw.githubusercontent.com/UiPath/uipath-python/main/packages/uipath/docs/core/agents.md
6. UiPath. "Coded functions" documentation (Maestro Service Task, typed error outputs, idempotency under Maestro retries). GitHub. https://raw.githubusercontent.com/UiPath/uipath-python/main/packages/uipath/docs/core/functions.md
7. UiPath. "UiPath LangChain Python SDK" README (LangGraph agents, human-in-the-loop through Action Center). GitHub. https://raw.githubusercontent.com/UiPath/uipath-langchain-python/main/README.md
8. UiPath. "Tool-Calling Suspend/Resume Sample" README (interrupt, checkpoint, resume, `uipath eval`). GitHub. https://raw.githubusercontent.com/UiPath/uipath-langchain-python/main/samples/tool-calling-suspend-resume/README.md
9. UiPath. "Email Triage Agent" sample README (long-running agent suspended on an Integration Service event). GitHub. https://raw.githubusercontent.com/UiPath/uipath-langchain-python/main/samples/email-triage-agent/README.md
10. UiPath. "Support Ticket Classification System" sample README (label plus confidence, Action Center approval). GitHub. https://raw.githubusercontent.com/UiPath/uipath-langchain-python/main/samples/ticket-classification/README.md
11. UiPath. "AML Alert Triage" coded agent sample README (LLM extracts and explains, a typed classifier decides). GitHub. https://raw.githubusercontent.com/UiPath/uipath-langchain-python/main/samples/aml-alert-triage-agent/README.md
12. UiPath. "Human-in-the-Loop Inbox Server" sample README. GitHub. https://raw.githubusercontent.com/UiPath/uipath-langchain-python/main/samples/hitl-inbox-server/README.md
13. UiPath. "Multi-Agent Supervisor System" sample README. GitHub. https://raw.githubusercontent.com/UiPath/uipath-langchain-python/main/samples/multi-agent-supervisor-researcher-coder/README.md
14. UiPath. "Joke Agent: Bring Your Own Guardrail" sample README (customer-managed guardrail vendor through Integration Service; agent and LLM scopes). GitHub. https://raw.githubusercontent.com/UiPath/uipath-langchain-python/main/samples/joke-agent-bring-your-own-guardrail/README.md
15. UiPath. "Guardrails" documentation (tool, LLM and agent scopes; pre/post stages; PII, harmful content, prompt attacks). GitHub. https://raw.githubusercontent.com/UiPath/uipath-python/main/packages/uipath/docs/core/guardrails.md
16. UiPath. Deterministic guardrails service, `uipath-core` source. GitHub. https://raw.githubusercontent.com/UiPath/uipath-python/main/packages/uipath-core/src/uipath/core/guardrails/_deterministic_guardrails_service.py
17. UiPath. "Agent Evaluations" documentation (output-based, trajectory-based and custom evaluators). GitHub. https://raw.githubusercontent.com/UiPath/uipath-python/main/packages/uipath/docs/eval/index.md
18. UiPath. "LLM Judge Trajectory Evaluators" documentation (trajectory and tool-simulation judges). GitHub. https://raw.githubusercontent.com/UiPath/uipath-python/main/packages/uipath/docs/eval/llm_judge_trajectory.md
19. UiPath. "Tool Call Order Evaluator" documentation (strict versus LCS-based scoring). GitHub. https://raw.githubusercontent.com/UiPath/uipath-python/main/packages/uipath/docs/eval/tool_call_order.md
20. UiPath. Documents service source, `uipath-platform` (classify, DU Modern / IXP extraction, validation actions). GitHub. https://raw.githubusercontent.com/UiPath/uipath-python/main/packages/uipath-platform/src/uipath/platform/documents/_documents_service.py
21. XLANG Lab. "OSWorld" README (OSWorld-Verified release of 2025-07-28; maintainer-run verified leaderboard). GitHub. https://raw.githubusercontent.com/xlang-ai/OSWorld/main/README.md
22. XLANG Lab. "OSWorld evaluation_examples" README (task schema with execution-based evaluators). GitHub. https://raw.githubusercontent.com/xlang-ai/OSWorld/main/evaluation_examples/README.md
23. UiPath Documentation portal (official product documentation for Document Understanding, IXP, Communications Mining, Process Mining, Agents and Maestro). Not reachable in this pass (egress policy); listed as the place to check product names. https://docs.uipath.com
24. PracHub. "UiPath Machine Learning Engineer Interview Experience: LLM Agent Q&A, a LeetCode 837 Variant, and a Coding Reject" (candidate report, September 2026; Agents team MLE; recruiter, hiring manager and phone rounds). Seen via search snippet only. https://prachub.com/interview-experiences/uipath-machine-learning-engineer-interview-experience-llm-agent-qanda-a-leetcode-837-variant-and-a-coding-reject
25. Dataford. "UiPath AI Engineer interview guide" (commercial prep guide with sample questions; not a first-hand report). Seen via search snippet only. https://dataford.io/interview-guides/uipath/ai-engineer
26. Glassdoor. UiPath interview review, Research Scientist, London, April 2025 (candidate report). Seen via search snippet only. https://www.glassdoor.co.nz/Interview/UiPath-Interview-E1102519-RVW98010961.htm
27. Glassdoor. UiPath interview review, Cluj-Napoca (candidate report; hiring-manager system design round). Seen via search snippet only. https://static.glassdoor.com.au/Interview/UiPath-Interview-E1102519-RVW104635339.htm
28. techinterview.org. "UiPath Interview Guide (2026): RPA and Automation Engineering" (prep guide; not a first-hand report). Seen via search snippet only. https://www.techinterview.org/companies/uipath-interview-guide/
29. UiPath Documentation. "Building effective agent tools" (tool selection from name, description and context; argument inference from the input schema). Seen via search snippet only. https://docs.uipath.com/agents/automation-cloud/latest/user-guide/building-effective-agent-tools
30. 1point3acres. "UiPath Machine Learning Engineering Phone Screen Interview Experience" (candidate report; transformer basics, behavioural and coding). Seen via search snippet only. https://www.1point3acres.com/interview/thread/1140358
