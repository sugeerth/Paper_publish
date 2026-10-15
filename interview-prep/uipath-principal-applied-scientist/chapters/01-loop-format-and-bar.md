# 01. UiPath Principal Applied Scientist Loop: Format, ML System Design Focus, Research Presentation and the Principal Bar

> **How to read this chapter.** You asked for ML system design, so this chapter centres on it. It explains where the ML system design round sits in the UiPath Principal Applied Scientist loop, what the principal-level bar looks like in that round, and the process-level questions you are likely to get there. The research presentation and values rounds are covered only as far as they connect to system design. Diagrams are plain-text block diagrams, so they render the same in Markdown and in a PDF export.
>
> **Tagging system.** Every question carries one of two tags:
> - **[Reported] [n]**: a candidate report, question bank or official page links this question, or this round type, to UiPath. The number points to the References list.
> - **[Likely]**: an inference from the role, the level and UiPath's AI products (agentic automation, Document Understanding, Communications Mining, Autopilot, the AI Trust Layer, computer vision for UI automation). No candidate has reported it.
> - Indented sub-bullets are follow-up probes an interviewer may use to push a strong first answer toward the principal bar.
>
> **Research limits (please read).** This run could not do any live research. The shared web-search budget for this workflow was already used up before the chapter started. Direct fetches of glassdoor.com, teamblind.com and uipath.com also failed, because the egress proxy rejected them and DNS lookups for WebFetch failed. So **no question below is tagged [Reported]**: none could be checked against a candidate write-up, and this chapter does not invent reports. Every question is **[Likely]**. The format notes describe the usual principal-scientist loop at enterprise AI companies of UiPath's size and shape; they are not a confirmed UiPath process. Confirm round names, durations and the presentation format with your recruiter. If you need sourced reports, a follow-up run with search budget can upgrade tags from [Likely] to [Reported].
>
> **Review pass (2026-10-14).** An adversarial review re-checked this chapter. Web search was again unavailable (the shared search budget was used up). Direct fetches of 12 candidate source pages were tried: UiPath careers, hiring process, blog, product and documentation pages, Glassdoor, Blind, Levels.fyi and Interview Query. Every fetch failed with a DNS error or a proxy 403, so no source could be verified and no question was upgraded to [Reported]. The review removed two generic leadership questions that did not test system design (they are covered in chapter 06). It added 21 [Likely] principal-level design prompts on gaps: capacity and cost estimation, idempotent agent actions, process versioning, agent tracing, computer-use evaluation harnesses, multi-agent attribution, data flywheels and tenant isolation, synthetic document data, rule-validated extraction, PII redaction, cross-source retrieval, red-teaming, model retirement, cost attribution, judge drift, delayed-outcome experiments, on-device inference, caching and platform consolidation.

## Format notes

### Typical loop shape (inferred, not confirmed)

| Stage | Typical duration | Interviewer | Status | Principal bar assessed |
|---|---|---|---|---|
| Recruiter screen | 30 min | Recruiter | Likely | Scope match (agents, document AI, comms mining, CV for UI), level calibration, compensation |
| Hiring manager screen | 45-60 min | Director or Head of AI/ML | Likely | Track record of research shipped into product; ability to set direction for a team |
| Technical screen | 45-60 min | Senior or staff scientist | Likely | ML/LLM depth plus a short design discussion or ML coding |
| **Research / technical presentation** | 60 min (about 30-40 min talk plus Q&A) | Panel of scientists, engineers and a product lead | Likely | Depth, rigour, and whether you can turn research into a product system |
| **ML system design** | 60 min | Principal scientist or architect | Likely | End-to-end design of an LLM or agentic system with evaluation, safety, cost and rollout |
| ML depth / applied science | 60 min | Staff or principal scientist | Likely | Modelling choices, evaluation methodology, failure analysis |
| Coding (ML or general) | 45-60 min | Engineer or scientist | Likely | Clean, working code; less weight than at junior levels |
| Cross-functional / product | 45 min | PM or engineering leader | Likely | Turning customer automation problems into ML roadmaps |
| Values / leadership (Humble, Bold, Immersed, Fast) | 45 min | Senior leader | Likely | Influence without authority, judgement, pace, ownership |

### Where ML system design shows up

```
  Recruiter ──> HM screen ──> Tech screen ──> ONSITE LOOP ──> Debrief / calibration
                    │               │               │
                    │               │               ├── Research presentation  (design Q&A on YOUR system)
                    │               │               ├── ML system design       (design a NEW UiPath-shaped system)
                    │               │               ├── ML depth               (evaluation / modelling probes)
                    │               │               ├── Coding
                    │               │               ├── Product / cross-functional (design trade-offs vs roadmap)
                    │               │               └── Values / leadership
                    │               └── short design discussion likely
                    └── "how would you architect X at UiPath" likely
```

System design probably comes up in three places, not one. You may be asked to (a) re-architect your own past system during presentation Q&A, (b) design a new system in the dedicated round, and (c) defend design trade-offs against product constraints in the cross-functional round.

### Canvas for the design round (a whiteboard layout, not an answer)

Use this layout to structure any prompt below, so the interviewer can see every principal-level concern on the board.

```
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │ 1. PROBLEM & SLOs: user, task, success metric, latency, cost/run, accuracy    │
 ├──────────────────────────────────────────────────────────────────────────────┤
 │ 2. DATA: sources (docs, emails, UI screens, logs), labels, PII, tenant bounds │
 ├───────────────┬───────────────────────┬───────────────────────┬──────────────┤
 │ 3. INGEST /   │ 4. MODEL LAYER        │ 5. ORCHESTRATION /    │ 6. HUMAN-IN- │
 │    PREPROCESS │   (LLM, specialist    │    AGENT RUNTIME      │    THE-LOOP  │
 │  OCR, parse,  │    models, retrieval, │  tools, robots, APIs, │  validation, │
 │  chunk, embed │    routing)           │  guardrails, retries  │  escalation  │
 ├───────────────┴───────────────────────┴───────────────────────┴──────────────┤
 │ 7. EVALUATION: offline sets, LLM-as-judge, task success, human review rate    │
 ├──────────────────────────────────────────────────────────────────────────────┤
 │ 8. TRUST & GOVERNANCE: PII masking, audit, policy, model choice per tenant    │
 ├──────────────────────────────────────────────────────────────────────────────┤
 │ 9. DEPLOY & OPERATE: cloud/on-prem, rollout, monitoring, drift, feedback loop │
 └──────────────────────────────────────────────────────────────────────────────┘
```

### What the principal bar likely means in design rounds

- **Problem framing before architecture.** You turn a vague automation goal into measurable SLOs: straight-through processing rate, human review rate, cost per document or per agent run, latency, and error cost.
- **Mix of deterministic and probabilistic parts.** UiPath's core is deterministic RPA robots. A principal candidate is expected to decide what belongs in a model, what belongs in a rule or a robot, and where people approve or correct the result.
- **Evaluation as a first-class part of the system.** Benchmarks, regression suites, LLM-as-judge calibration, and online metrics that tie back to customer ROI.
- **Enterprise constraints.** Multi-tenant isolation, PII and data residency, on-prem and air-gapped deployments, customer choice of model provider, and auditability.
- **Platform thinking.** You design a reusable capability (for example, one extraction or agent-evaluation framework) that many product teams and customers can use, not a one-off model.
- **Technical leadership.** You can explain the roadmap, the staffing, what you would build first, and how you would know to stop or pivot.

## Questions

### A. Process and expectations for the design round

1. [Likely] Walk me through how you approach a 60-minute ML system design interview, and what you clarify in the first ten minutes.
   - Which requirements would you refuse to assume, and ask about instead?
   - How do you decide when to go deep on one component versus staying broad?
2. [Likely] At principal level, what do you think separates a good system design answer from a great one?
   - Can you give an example from your own work where the design choice mattered more than the model choice?
3. [Likely] How would you design a system when the interviewer gives you no numbers? What numbers would you estimate yourself?
   - How would you size cost per agent run or per document for an enterprise customer processing 1M documents a month?
4. [Likely] How do you draw the boundary between what an LLM should do and what a deterministic robot, rule or API call should do?
   - Give a case where you moved logic out of a model and into deterministic code. What did it buy you?
5. [Likely] What does "production ready" mean for an ML system you own, and what checklist would you use before launch?
   - Who signs off, and on what evidence?
6. [Likely] How do you present trade-offs to an interviewer who keeps changing the constraints mid-design?
   - What would you re-do on the board if latency suddenly had to be 10x lower?
7. [Likely] Do a back-of-envelope estimate on the board: 50,000 agent runs a day, about 12 LLM steps per run, about 3,000 input tokens per step. What are tokens per day, peak requests per second and monthly cost at two price tiers?
    - Which assumption dominates the answer, and how would you check it with the customer?
8. [Likely] In the last ten minutes the interviewer asks, "What breaks first at 10x scale?" How do you structure that answer for the system on your board?

### B. Agentic automation system design (UiPath-shaped prompts)

9. [Likely] Design an agentic automation platform that lets enterprise users build agents that call tools, trigger RPA robots and hand off to people.
   - How do agents, robots and people share state across a long-running process?
   - Where do you put guardrails, and what happens when a tool call fails halfway through?
10. [Likely] Design an orchestration layer that coordinates several agents and robots across a multi-step business process (for example, order-to-cash).
   - How do you make the process recoverable and auditable after a crash?
   - How do you stop an agent from looping forever or spending too much?
11. [Likely] Design an evaluation system for enterprise agents whose prompts, tools and underlying LLMs change often.
   - How do you build test sets when customer data is private and differs per tenant?
   - How would you calibrate an LLM-as-judge against human labels, and what agreement level would you accept?
12. [Likely] Design a system that turns a natural-language description of a task into a working automation workflow.
    - How do you check the generated workflow before it runs on production systems?
    - What is your fallback when generation confidence is low?
13. [Likely] Design an agent that operates desktop and web applications from screenshots, without selectors or APIs.
    - How do you combine a vision-language model with UI element detection and existing selectors?
    - How do you measure task success, and how do you make it safe to run against live systems?
14. [Likely] Design a self-healing system for UI automations that break when a target application's interface changes.
    - How do you detect breakage before customers report it?
    - When should the system repair the automation itself, and when should it ask a developer?
15. [Likely] Design a tool-selection and routing layer for an agent that has hundreds of enterprise connectors available.
    - How do you keep the prompt small while keeping recall high on the right tool?
    - How would you evaluate routing errors separately from execution errors?
16. [Likely] Design memory and context management for long-running agents that work on a case over days.
    - What do you store, what do you summarise, and what must never be stored?
17. [Likely] How would you design human-in-the-loop escalation so agents ask for help at the right rate?
    - How do you set confidence thresholds per action type and per customer?
    - How do reviewer corrections flow back into the system?
18. [Likely] Design a cost-and-latency-aware model router that picks between small, large, hosted and self-hosted models per step.
    - What signals decide the route, and how do you stop quality from silently dropping?
19. [Likely] Design idempotency and compensation for agent actions that write to systems of record such as an ERP or CRM.
    - How do you handle a tool call that timed out but may have succeeded?
    - When would you use a saga with compensating actions instead of a human approval step?
20. [Likely] Design a long-running process orchestrator in which modelled process steps mix agent steps, robot steps and human tasks.
    - How do you ship a new process version while thousands of instances of the old one are still running?
21. [Likely] Design tracing and observability for agent runs.
    - What goes into a span, and how do you make a non-deterministic LLM run replayable for debugging?
    - How do you strip PII from traces and still keep them useful?
22. [Likely] Design an evaluation harness for computer-use agents that work in enterprise applications (ERP, CRM, legacy desktop apps).
    - How do you build sandboxed replicas, reset their state between runs and score partial task completion?
23. [Likely] Design a multi-agent system where a planner hands work to specialist agents.
    - When is multi-agent worse than one agent with tools, and how do you attribute a failure to the right agent?
24. [Likely] Design a data flywheel that turns agent traces, reviewer corrections and business outcomes into evaluation and training data.
    - How do you guarantee one tenant's data never improves, or leaks into, another tenant's model?

### C. Document AI and communications system design

25. [Likely] Design an intelligent document processing system that extracts fields from invoices, contracts and forms across many layouts and languages.
    - Where do OCR, layout models, specialised extractors and LLMs each fit?
    - How do you produce calibrated field-level confidence for straight-through processing?
26. [Likely] Design a system that lets a customer build a new document extractor with ten labelled examples.
    - How do you choose between prompting, few-shot retrieval and fine-tuning?
    - How do you tell the customer whether the extractor is good enough to go live?
27. [Likely] Design a document classification and splitting pipeline for multi-document PDFs that arrive in one email.
    - How do you handle page-order errors and mixed scans?
28. [Likely] Design a communications mining system that finds intents, entities and sentiment in high-volume enterprise email and tickets, then triggers automations.
    - How do you discover new intents nobody labelled yet?
    - How would you handle taxonomy drift when the business changes?
29. [Likely] Design an active learning loop for document and message models where every label costs expert time.
    - Which sampling strategy would you use, and how would you show it beats random labelling?
30. [Likely] Design a table extraction system for complex financial documents with nested and multi-page tables.
    - How do you evaluate table structure accuracy as well as cell values?
31. [Likely] How would you decide whether a document understanding product should use one general foundation model or many task-specific models?
    - What would the cost, latency and accuracy data need to show to change your mind?
32. [Likely] Design a synthetic data pipeline (layouts, handwriting, poor scans) to bootstrap extractors for a new document type or language.
    - How do you show that synthetic data helps on the real distribution and is not just inflating offline scores?
33. [Likely] Design schema-constrained LLM extraction with business-rule validation (line items sum to the total, dates are consistent, vendor matches master data).
    - How should rule violations change field-level confidence and routing to human review?
34. [Likely] Design support for handwriting, low-resource languages and right-to-left scripts in a document understanding product.
    - How do you decide which languages get dedicated models and which rely on a general model?
35. [Likely] Design a PII detection and redaction model that runs before any document or email reaches an external LLM.
    - How do you measure its recall, and what happens to a request when recall is uncertain?

### D. Retrieval, grounding and enterprise knowledge

36. [Likely] Design a retrieval-augmented system that answers questions over a customer's processes, documents and automation logs.
    - How do you enforce per-user access rights during retrieval?
    - How do you measure groundedness and catch answers that sound plausible but are wrong?
37. [Likely] Design an assistant that helps automation developers write and debug workflows inside the studio IDE.
    - What context do you retrieve from the project, and how do you keep it fresh?
    - How do you measure developer productivity gains rather than suggestion acceptance alone?
38. [Likely] Design a process mining plus ML system that recommends which processes a customer should automate next.
    - How do you estimate automation ROI from event logs and task-mining data?
    - How do you avoid recommending processes that are frequent but too variable to automate?
39. [Likely] Design an embedding and indexing service that many product teams share.
    - How do you version embeddings when the model changes, without re-indexing every tenant at once?
40. [Likely] Design question answering over semi-structured enterprise sources (spreadsheets, ticket histories, process definitions) where the answer needs a join across sources.
    - When do you use text-to-SQL or tool calls instead of vector retrieval?

### E. Trust, safety, governance and enterprise deployment

41. [Likely] Design an AI trust layer that sits between UiPath products and any LLM provider.
    - Where do PII masking, prompt-injection defence, logging and policy enforcement run?
    - How do you prove to an auditor what the model saw and what it did?
42. [Likely] How would you defend an agent that reads untrusted emails and documents against prompt injection that tries to trigger robot actions?
    - Which actions need human approval no matter what the model's confidence is?
43. [Likely] Design an ML system that must run in the cloud, on-prem and in air-gapped customer environments.
    - What changes in model choice, updates and monitoring for each?
44. [Likely] How would you support customers who bring their own LLM, when your evaluation and prompts were tuned on a different model?
    - How do you certify a new model and roll it out per tenant?
45. [Likely] Design monitoring for model drift and quality regression when you cannot look at customer data.
    - Which privacy-preserving signals would you rely on?
46. [Likely] How would you design data governance for training on customer data that some tenants allow and others forbid?
    - How would you trace which data trained which model version?
47. [Likely] Design an audit and explainability feature for an agent that made a financial decision inside an automated process.
48. [Likely] Design a red-teaming and guardrail evaluation programme for an agent platform.
    - How do you weigh the cost of guardrail false positives (blocked legitimate work) against missed attacks?
49. [Likely] An LLM provider will retire a model that customers' production agents depend on in 90 days. Design the migration process.
    - How do you find every affected agent, re-certify behaviour and handle customers who refuse to move?
50. [Likely] Design per-tenant cost attribution and budget enforcement for LLM usage across several products.

### F. Evaluation, experimentation and rollout

51. [Likely] How would you build an offline benchmark for agentic tasks that predicts customer-facing success?
    - How do you stop the benchmark from going stale or leaking into training?
52. [Likely] Design an A/B or staged rollout for a new extraction or agent model when each customer has a small, different workload.
    - What would you do when the aggregate metric improves but a large customer gets worse?
53. [Likely] Which online metrics would you put on the executive dashboard for an agent product, and which would you keep internal?
54. [Likely] How do you set and defend a "good enough to automate" threshold with a customer whose error tolerance differs from yours?
55. [Likely] Design a regression testing pipeline that runs every time a prompt, tool schema or model version changes.
    - How do you handle non-determinism in LLM outputs during testing?
56. [Likely] How would you measure whether human reviewers are a reliable gold standard?
57. [Likely] Your LLM-as-judge model is upgraded by its provider. How would you detect that its scores have shifted, and what would you do about historical comparisons?
58. [Likely] Design an online experiment for an agent whose true outcome (for example, an invoice paid correctly) is known only weeks later.
    - Which proxy metrics would you use, and how would you validate them against the delayed outcome?

### G. Scaling, efficiency and serving

59. [Likely] Design LLM serving for millions of short agent steps a day with strict cost targets.
    - Where do caching, batching, distillation and quantisation help, and where do they hurt?
60. [Likely] How would you distil a large model into a small specialised model for a high-volume document or classification task?
    - How do you decide the small model is safe to replace the large one?
61. [Likely] Design a fine-tuning platform that adapts base models per customer without a full model copy per tenant.
    - How would you serve many adapters efficiently?
62. [Likely] How do you plan GPU capacity for a product whose usage is spiky and tied to customers' business hours?
63. [Likely] What would you change in your design if per-request latency must drop below one second for an attended (user-facing) automation?
64. [Likely] Design inference for a screen-understanding model that must run on customer desktops with CPU only, for attended robots.
    - What do you trade among model size, accuracy and startup time, and how do you ship model updates?
65. [Likely] Design caching for agent steps (prompt-prefix caching, semantic caching of LLM answers and tool results).
    - How do you prevent stale answers and cross-tenant cache hits?

### H. Research presentation: design Q&A on your own system

66. [Likely] Present a system you designed end to end. What was the hardest design decision, and what did you reject?
    - If you rebuilt it today with current LLMs, what would you delete?
67. [Likely] Draw the architecture of your presented system and mark where it failed in production.
    - How did you find out, and what did you change in the design?
68. [Likely] How did you go from a research result to something a product team could own and operate?
    - What did you hand over: code, eval suite, runbooks, people?
69. [Likely] Which baseline did you compare against, and why should we believe the comparison was fair?
70. [Likely] What is the single number that best describes the impact of your system, and how was it measured?
71. [Likely] How would your presented system map onto a UiPath product such as agents, document understanding or communications mining?
    - What would need to change for multi-tenant enterprise deployment?
72. [Likely] What part of your presented work would you publish, and what would you keep as product advantage?

### I. Principal-level scope and technical leadership in design

73. [Likely] Describe a time you set the technical direction for an ML platform used by several teams.
    - How did you get teams that did not report to you to adopt it?
74. [Likely] How would you write a 12-month applied science roadmap for agentic automation at UiPath?
    - What would you build, buy or partner on, given fast-moving foundation models?
75. [Likely] How do you decide when to invest in a custom model versus using a frontier API?
    - What would trigger you to reverse that decision?
76. [Likely] Tell me about a design review where you blocked or significantly changed a launch.
    - How did you handle the disagreement with product or engineering?
77. [Likely] How do you review another scientist's system design, and what do you look for first?
78. [Likely] Describe a system you designed that turned out to be the wrong bet. How early did you notice, and what did you do?
79. [Likely] You inherit three teams, each with its own document extraction stack. Design the consolidation plan and the metrics that tell you it worked.
    - What would you keep separate on purpose?

### J. Values lens on design decisions (Humble, Bold, Immersed, Fast)

80. [Likely] **Humble:** Tell me about a design of yours that a junior colleague or a customer improved.
    - What did you change in how you run design reviews afterwards?
81. [Likely] **Humble:** When did your evaluation show that your favourite approach was worse than a simple baseline?
82. [Likely] **Bold:** Describe a time you proposed an architecture that most people thought was too ambitious.
    - How did you reduce the risk while keeping the ambition?
83. [Likely] **Bold:** What design bet would you make for UiPath's agentic platform that others might disagree with?
84. [Likely] **Immersed:** How did you learn the customer's real workflow before designing an ML system for it?
    - Have you sat with operators or reviewers who used your system? What did you learn?
85. [Likely] **Immersed:** Tell me about a design detail you only got right because you read the raw data or logs yourself.
86. [Likely] **Fast:** Describe a time you shipped a smaller version of a system early to learn quickly.
    - What did you deliberately leave out of v1, and how did you protect quality?
87. [Likely] **Fast:** How do you keep moving fast when the foundation models underneath your system change every few months?

### K. Questions you should be ready to ask (reverse questions)

88. [Likely] What does the ML system design round expect from a principal candidate compared with a staff candidate?
89. [Likely] Is the research presentation on my own work, a given topic, or a take-home design?
90. [Likely] Which UiPath product area (agents, document AI, communications mining, CV for UI automation, process intelligence) would this role design for first?
91. [Likely] How do applied scientists and platform engineers share ownership of production systems here?
92. [Likely] How does UiPath evaluate agents today, and what is the biggest gap?
93. [Likely] What constraints from on-prem and regulated customers most shape model and architecture choices?

## References

No source pages could be retrieved in this run (see "Research limits" above), so no question cites a source. The official pages below are the right starting points for checking the format notes and product context. They are listed for verification and were not read in this run. A fetch of each was tried again on 2026-10-14 and blocked by the network (DNS failure or proxy 403), so whether they are live could not be confirmed. Both are long-standing official UiPath URLs and are kept only as starting points; no question relies on them.

1. UiPath Careers, uipath.com (official careers page; not retrieved in this run). https://www.uipath.com/company/careers
2. UiPath Blog, uipath.com (official product and AI announcements; not retrieved in this run). https://www.uipath.com/blog
