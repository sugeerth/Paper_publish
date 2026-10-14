# 06. Principal-Level Leadership Through ML System Design: Technical Strategy, Design Reviews, Influence, Research-to-Product and Roadmaps

> **How to read this chapter.** You asked for ML system design only, so this chapter does not cover general behavioral questions. It covers the leadership part of the Principal Applied Scientist bar *as it shows up in ML system design*: setting the technical strategy for an ML platform, running and defending design reviews, getting other teams to adopt a shared architecture, moving a research prototype into a production system, and planning a multi-quarter roadmap for systems such as Document Understanding (PDF and document tagging and extraction), agents and Communications Mining. Diagrams are plain-text block diagrams, so they render the same in Markdown and in a PDF export.
>
> **Tagging system.** Every question carries one of two tags:
> - **[Reported] [n]**: a candidate report, question bank or official page links this question, or this round type, to UiPath. The number points to the References list.
> - **[Likely]**: an inference from the role, the level and UiPath's AI products (agentic automation, Document Understanding, Communications Mining, Autopilot, the AI Trust Layer, computer vision for UI automation). No candidate has reported it.
> - Indented sub-bullets are follow-up probes an interviewer may use to push a strong first answer toward the principal bar.
>
> **Research limits (please read).** In the latest review run the shared web-search budget was used up, and direct fetches of uipath.com, docs.uipath.com, Glassdoor, Indeed, Levels.fyi, LeetCode Discuss and Blind were all blocked (DNS failure or proxy denial). Only GitHub raw files could be read. That means no candidate write-up of a UiPath leadership or design round could be checked, so **no question below is tagged [Reported]**, and this chapter does not invent reports. Every question is **[Likely]**. Where a [Likely] question rests on a fact from a page that *was* read (for example UiPath's public agent SDKs, Action Center human-in-the-loop, context grounding, MCP hosting or runtime tracing), the question ends with *(context: [n])*. That marker points to the source of the product fact. It does not mean a candidate reported the question. The format notes describe the usual principal-scientist loop at enterprise AI companies of UiPath's size; they are not a confirmed UiPath process. Confirm round names and durations with your recruiter.

## Format notes (inferred, not confirmed)

- **Where it happens.** Leadership is unlikely to be tested only in a values round. At principal level it is probably tested *inside* the ML system design round ("now you have a team of six, what do you build first?"), in the research-presentation Q&A ("how did you get this adopted?"), and in a hiring-manager or cross-functional round with a director, head of AI or product leader. Each is typically 45-60 minutes. (Inferred; see chapter 01.)
- **Interviewers.** Likely a principal scientist or architect for the design round, and a director or VP-level AI leader plus a product manager for the strategy and cross-functional rounds.
- **Bar assessed at principal level.** (1) You set direction for a *platform*, not a single model. (2) You make irreversible architecture decisions with evidence and can say when you would reverse them. (3) Other teams adopt your design without reporting to you. (4) You turn research into shipped, measured product value. (5) You grow other scientists' design judgement. UiPath's published values are commonly cited as Humble, Bold, Immersed and Fast; the careers page [1] could not be reached in this run, so check the current wording yourself. Public interview-prep guides also note that GenAI and agent system design rounds in 2025-2026 tend to weigh evaluation, cost, latency, guardrails and human fallback above the architecture diagram itself [11].

### Reference diagram: the leadership layers around an ML system design

```
 ┌────────────────────────────────────────────────────────────────────────────┐
 │ STRATEGY       why this system, which bets, build/buy/partner, 2-3 yr view │
 ├────────────────────────────────────────────────────────────────────────────┤
 │ ROADMAP        milestones, kill criteria, staffing, dependencies           │
 ├────────────────────────────────────────────────────────────────────────────┤
 │ ARCHITECTURE   shared platform pieces vs product-specific pieces           │
 │   ┌──────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
 │   │ Ingest / │->│ Models (LLM, │->│ Orchestration│->│ Human-in-the-loop│   │
 │   │ OCR/parse│  │ specialists) │  │ agents/robots│  │ validation/label │   │
 │   └──────────┘  └──────────────┘  └──────────────┘  └────────┬─────────┘   │
 │        ^                                                     │ feedback    │
 │        └─────────────────────────────────────────────────────┘             │
 ├────────────────────────────────────────────────────────────────────────────┤
 │ EVALUATION & GOVERNANCE   shared benchmarks, release gates, trust layer    │
 ├────────────────────────────────────────────────────────────────────────────┤
 │ PEOPLE & ALIGNMENT  design reviews, RFCs, mentoring, partner teams, PM     │
 └────────────────────────────────────────────────────────────────────────────┘
```

### Reference diagram: a document tagging and extraction (PDF) system, as a leadership prompt

Interviewers can use a familiar system like this to ask leadership questions ("who owns each box, what do you build first?"). This is the shape of the system, not an answer.

```
  PDF / scan / email attachment
          │
          v
 ┌──────────────┐   ┌───────────────┐   ┌────────────────────┐   ┌───────────────┐
 │ Digitize     │-->│ Classify doc  │-->│ Extract fields /   │-->│ Confidence &  │
 │ OCR + layout │   │ type (invoice,│   │ tables (specialist │   │ business-rule │
 │ parse        │   │ ID, contract) │   │ model or LLM)      │   │ checks        │
 └──────────────┘   └───────────────┘   └────────────────────┘   └──────┬────────┘
                                                                        │
                        high confidence ────────────────────────────────┤
                                                                        v
                     ┌─────────────────────┐   corrections   ┌──────────────────┐
                     │ Downstream robot /  │<----------------│ Human validation │
                     │ ERP / agent         │                 │ (tag & correct)  │
                     └─────────────────────┘                 └────────┬─────────┘
                                                                      │ labels
                                                                      v
                                                  ┌────────────────────────────────┐
                                                  │ Labeling store -> retrain/eval │
                                                  │ -> gated release per tenant    │
                                                  └────────────────────────────────┘
```

## Questions

### A. Technical strategy for ML systems

1. [Likely] If you joined as the principal scientist for Document Understanding, what would your one-page technical strategy for the next two years say?
    - Which current architecture choices would you keep, and which would you challenge first?
    - What evidence would you gather in your first 90 days before committing to the strategy?
2. [Likely] Specialist extraction models versus a general LLM or vision-language model for PDF and document extraction: how would you set the long-term direction, and how would you explain it to leadership?
    - Under what cost, accuracy or latency numbers would you reverse that decision?
    - How do you avoid stranding customers who have trained custom models on the old stack?
3. [Likely] How would you decide which parts of UiPath's AI stack should be a shared platform and which should stay inside each product team?
    - Give an example of a component you would centralise (for example, evaluation or labeling) and one you would deliberately not centralise.
4. [Likely] Build, buy or partner: how would you decide whether UiPath should train its own models, fine-tune open models or call frontier model APIs for a given capability?
    - How does customer choice of model provider change the architecture?
    - How do on-prem and air-gapped customers change the answer?
5. [Likely] Describe a technical bet you made that most people disagreed with. How did you structure the system so the bet could be tested cheaply?
    - What was the kill criterion, and did you ever use it?
    - Give one bold architecture change that paid off and one that did not. What separated them?
6. [Likely] How would you set the technical strategy for moving from deterministic RPA toward agentic automation without breaking customer trust in reliability?
    - Where must the system stay deterministic, and who decides?
7. [Likely] How do you decide when an ML system needs a rewrite rather than incremental improvement?
    - How did you make that case to people who had built the old system?
8. [Likely] How would you design a model strategy that survives the frontier-model landscape changing every few months?
    - What abstraction layer would you insist on, and what does it cost you?
9. [Likely] What are the two or three ML system metrics you would put on an executive dashboard for an automation platform, and why those?
    - How do you connect straight-through processing rate or agent task success to customer ROI?
10. [Likely] How would you identify the highest-leverage ML investment across Document Understanding, Communications Mining, agents and screen understanding when you can only staff two?
    - What data would you need, and what would you do if product leaders disagreed with your ranking?
11. [Likely] UiPath's agent SDKs let teams and customers bring agents built in LangGraph, LlamaIndex, OpenAI Agents, Google ADK, Microsoft Agent Framework or PydanticAI onto one platform. Would you keep the platform framework-agnostic, or standardise on one framework for agents built in-house? (context: [5][6])
    - What must the shared runtime contract guarantee so that evaluation, tracing and human review work the same way for every framework?
12. [Likely] How would you set the strategy for retrieval ("context grounding") across products: one shared, permission-aware index service, or retrieval owned by each product team? (context: [4])
    - Who owns index freshness, access control and retrieval quality metrics?
13. [Likely] Would you expose document extraction and classification as MCP tools that third-party agents can call? Make the strategic case for and against. (context: [7])
    - How does that change your trust, metering and abuse-prevention design?
14. [Likely] How would you set a cost-per-document or cost-per-agent-run target for the platform, and drive it down across teams you do not manage?
    - Which architecture levers (model routing, caching, distillation, batching) would you push first?
15. [Likely] How would you set a strategy for evaluating agents that customers build in frameworks you do not control?

### B. Leading design reviews and architecture decisions

16. [Likely] Walk me through how you run a design review for a new ML system at principal level. What do you require in the design document?
    - What sections do people usually skip, and how do you make them write them?
17. [Likely] A senior engineer proposes a document extraction architecture you think will not scale to multi-tenant use. How do you handle the review?
    - How do you separate "I would do it differently" from "this will fail"?
18. [Likely] Tell me about an architecture decision record you wrote that was later proven wrong. What did you change in your process?
    - How did you tell the people who had followed your recommendation?
19. [Likely] How do you decide which design decisions need broad review and which a team can make alone?
    - How do you classify reversible versus irreversible decisions in an ML system?
20. [Likely] In a design review for an agent platform, which failure modes would you always ask about?
    - How do you make sure evaluation, cost and safety are reviewed, not only the model?
21. [Likely] How do you review an ML system design when you are not the expert in its sub-domain (for example, OCR or speech)?
22. [Likely] Describe a time you blocked a launch on design grounds. What was the evidence and how did you communicate it?
    - What would you have needed to see to approve it?
23. [Likely] How do you keep a large ML architecture coherent when ten teams are each shipping changes to it?
    - What guardrails (interfaces, contracts, shared evaluation) would you put in place?
24. [Likely] Two teams have built overlapping document classification services. How would you decide whether to merge them, and how would you lead that change?
    - How do you handle the team whose system is being retired?
25. [Likely] How do you document an ML system so a new scientist can make safe changes in their first month?
26. [Likely] You are reviewing the design of a long-running agent that pauses for human approval in Action Center and resumes hours later. What do you require on state, idempotency and resume semantics before you approve it? (context: [5][8])
    - What happens if the human edits the agent's proposed action instead of approving it?
27. [Likely] A team proposes adding a new agent-framework integration to the shared runtime protocol. What conformance and evaluation checks must it pass before it ships? (context: [6][8])
28. [Likely] You want every agent run to emit traces in one shared format (LLM calls, tool calls, human tasks). How would you get teams to agree on the span schema, and what would you do with teams that have already shipped their own? (context: [8][9])

### C. Influencing without authority across teams

29. [Likely] You want every product team to adopt one shared evaluation framework for agents and extraction models. None of them report to you. How do you get adoption?
    - What would you build first to make adoption cheaper than not adopting?
    - What do you do about a team that refuses?
30. [Likely] How would you convince the platform engineering team to change their serving infrastructure to support a new model architecture your research needs?
    - How do you quantify the benefit in their terms rather than yours?
31. [Likely] Tell me about a time you changed a product leader's mind about an ML system design using data.
    - What data did not persuade them, and why?
32. [Likely] How do you align security, legal and product on PII handling in a document or communications ML pipeline?
    - Where would you put masking in the architecture, and how do you get sign-off?
33. [Likely] A partner team owns the OCR component your extraction quality depends on, and their priorities differ from yours. How do you get the improvements you need?
34. [Likely] How do you build credibility with engineering teams when you join as a principal scientist from outside?
35. [Likely] How do you resolve a disagreement between research and engineering about whether a model is ready for production?
    - What shared release gate would you propose so the argument does not repeat?
36. [Likely] Describe how you would get field and customer-success teams to feed real customer failures back into the ML system.
    - What tooling or process in the system design makes that feedback loop work?
37. [Likely] How do you influence the direction of a system owned by a peer principal who disagrees with you?
38. [Likely] How would you convince the team that owns the human-task (Action Center) experience to change its schema so validator corrections become usable training and evaluation labels? (context: [5])
    - What do you offer them in return?
39. [Likely] A model provider deprecates the model version your extraction pipeline depends on, at short notice. How do you lead the response across your team, the provider and affected customers?
    - What would you change in the architecture so the next deprecation is routine?
40. [Likely] A product team ships an agent feature that bypasses the shared guardrail and audit layer to meet a launch date. How do you respond, and how do you stop it happening again?
41. [Likely] How would you use a customer advisory group to shape the design of a governance feature (model updates, data retention, audit) without letting the loudest customer set the architecture?

### D. Research-to-product for ML systems

42. [Likely] Walk me through a research result you took into production. What changed in the system design between the paper and the product?
    - Which offline metric gains disappeared in production, and why?
43. [Likely] How would you take a new layout-aware or vision-language model for documents from a research prototype to a generally available Document Understanding feature?
    - What gates would you set between prototype, preview and GA?
    - How would you handle customers whose documents look nothing like the training data?
44. [Likely] How do you decide which research directions are worth a production investment and which should stay exploratory?
45. [Likely] How do you design the research environment so that experiments can be promoted to production without being rewritten?
    - What shared components (data loaders, eval harness, model registry) would you standardise?
46. [Likely] How would you run a time-boxed research spike on computer-use agents, and how would you decide whether to productise it?
47. [Likely] Tell me about a research project you stopped. How did you decide, and what did you salvage into the product?
48. [Likely] How do you balance publishing research with protecting UiPath's product advantage?
49. [Likely] How would you move a Communications Mining-style model from per-customer training toward a foundation model, and what organisational changes would that require?
50. [Likely] How do you translate an academic benchmark gain into a business case a product leader will fund?
51. [Likely] How would you take a research technique for agent reliability (for example self-verification or plan-then-execute) to production when it doubles token cost and latency?
    - What evidence of reliability gain would justify the cost, and for which customer processes?
52. [Likely] How would you set up a loop where production failures from agents and extraction models directly set the research agenda?
    - Who triages failures, and how do they reach a researcher within a week?
53. [Likely] How would you lead the move to synthetic documents for training and evaluating extraction models, and how would you get legal and customers comfortable with it?
54. [Likely] How would you decide between per-tenant fine-tuning and a shared model with in-context examples for customer-specific extraction, and what does each mean for tenant isolation?

### E. Roadmap and execution for ML platforms

55. [Likely] Build a four-quarter roadmap for an enterprise agent evaluation platform. What ships in Q1?
    - Which dependencies on other teams are on the critical path?
    - What would you cut if your team were halved?
56. [Likely] Build a roadmap for a document tagging and labeling system that reduces human validation effort by half.
    - How would you sequence active learning, confidence calibration and better base models?
    - Which milestone would tell you the plan is not working?
57. [Likely] How do you plan ML work when model quality gains are uncertain and hard to estimate?
    - How do you communicate uncertainty in a roadmap to executives?
58. [Likely] How do you balance roadmap items for new capabilities against reliability, technical debt and evaluation work?
59. [Likely] A large customer asks for a custom model feature that conflicts with your platform roadmap. How do you decide?
60. [Likely] How would you staff a new ML system: which roles, in which order, and which you would hire versus borrow?
61. [Likely] How do you define success and kill criteria for each milestone of an ML platform roadmap?
62. [Likely] Tell me about a roadmap you owned that slipped badly. What did you change in the system or the plan?
63. [Likely] How do you plan the migration of customers from an older extraction model generation to a new one?
    - How do you prove to customers that the new model is not worse on their documents?
64. [Likely] Build a roadmap that moves customers from robot-only workflows to processes that combine agents, robots and people. What has to be true in the platform before each step?
65. [Likely] A new frontier model release makes two quarters of your team's specialist-model work look obsolete. How do you re-plan, and how do you explain it to your team and to leadership?
    - What part of that work would you keep, and why?
66. [Likely] How would you set and track a reliability target (for example agent task success or straight-through processing rate) as a roadmap commitment rather than a slogan?

### F. Cross-org alignment on platform architecture

67. [Likely] How would you align Document Understanding, agents and Communications Mining teams on a shared model-serving and routing layer?
    - What would the interface contract look like, and who owns it?
68. [Likely] How would you get agreement across product, research and the AI Trust Layer team on what every LLM call must log for audit?
69. [Likely] How do you align cloud and on-prem product lines on one ML architecture when their constraints differ?
70. [Likely] Describe how you would set up a cross-org architecture forum for ML systems. Who attends, and what decisions does it own?
    - How do you stop it becoming a bottleneck?
71. [Likely] How do you handle a situation where two VPs want incompatible ML platform directions?
72. [Likely] How would you align on a single definition of "agent task success" across teams that each measure it differently?
73. [Likely] How do you manage shared labeling and human-validation infrastructure when several products compete for annotator capacity?
74. [Likely] How would you align products and partners on one field schema and label taxonomy for document extraction?
    - Who owns changes to the schema, and how are breaking changes handled?
75. [Likely] How would you get product and research to agree on one offline benchmark suite that gates every model release, when the most useful golden data is customer data under privacy limits?
76. [Likely] A model-provider outage takes down agents, Document Understanding and Communications Mining at once. Who should own failover policy across products, and how would you get it agreed before the next outage?

### G. Mentoring and raising design judgement

77. [Likely] How do you teach a strong senior scientist to think about system design, not only model design?
    - What exercises or review habits do you use?
78. [Likely] Tell me about someone you grew into a technical lead for an ML system. What did you delegate, and when?
79. [Likely] How do you give feedback on a design document that is technically correct but misses the product problem?
80. [Likely] How do you build an evaluation-first culture in a team that is used to shipping on demo quality?
81. [Likely] How do you scale your design influence beyond the meetings you personally attend?
    - What templates, checklists or reference architectures have you created?
82. [Likely] How do you mentor engineers who are new to ML on the failure modes of probabilistic systems in production?
83. [Likely] What separates senior from principal design work in ML systems, and how have you coached someone across that gap?
84. [Likely] How would you coach a strong researcher who resists production limits on latency, cost and on-prem deployment for a document model?
85. [Likely] How would you set up design-review shadowing so junior scientists learn to review agent systems, not only build them?

### H. Judgement, ownership and values in system decisions

86. [Likely] Tell me about a time you shipped an ML system faster than the team thought possible. What did you deliberately leave out of the design?
    - What risks did that create, and how did you contain them?
87. [Likely] Describe a time you took ownership of a production incident in an ML system you did not build.
    - What design changes did you drive afterwards?
88. [Likely] When would you refuse to deploy an LLM-based automation for a customer process, even if the customer wants it?
    - How do you design the human-in-the-loop step so the customer stays accountable?
89. [Likely] How do you handle a customer escalation where your model's extraction error caused a downstream financial mistake?
    - What changes in confidence thresholds, validation and audit would you lead?
90. [Likely] How did you stay close to real customer usage ("immersed") when leading a platform used by thousands of customers?
91. [Likely] How do you make a high-stakes design call when the data is incomplete and the deadline is fixed?
92. [Likely] How do you weigh model cost against quality when the customer, finance and product each want different trade-offs?
93. [Likely] What would you want your first design review at UiPath to accomplish, and how would you prepare for it?
94. [Likely] An agent takes a wrong, irreversible action in a customer's ERP. Walk through how you lead the response and which design changes you drive afterwards.
    - How do you decide, as a company-wide policy, which agent actions need human approval?
95. [Likely] When would you tell leadership that a much-hyped capability, such as fully autonomous computer-use agents, is not ready for enterprise customers, and what evidence would you bring?
96. [Likely] How do you lead on fairness and quality gaps when your document or communications models are much weaker in some languages or regions?

### I. Leadership probes attached to a live design prompt

These are likely to come mid-way through a design question, after you have drawn the architecture.

97. [Likely] You have just designed this document extraction system. You have six people. Who works on which box first, and why?
98. [Likely] Which box in your diagram is the riskiest, and how would you de-risk it in the first month?
99. [Likely] Which parts of this design would you hand to a partner team, and how would you specify the interface?
100. [Likely] If the product manager says the launch date moves up by two months, what do you cut from this design?
    - What do you refuse to cut?
101. [Likely] How would you present this architecture to the CTO in five minutes?
102. [Likely] What would make you scrap this design six months from now?
103. [Likely] How would you measure whether this system is succeeding one year after launch, and who would you report it to?
104. [Likely] Which decisions in this design would you want reviewed by security and legal, and when in the timeline?
105. [Likely] Which component in your diagram would you build now as a shared platform piece, and which would you leave product-specific until a second team needs it?
106. [Likely] If your LLM inference budget were cut by half, what changes in this diagram?
107. [Likely] Which interface in your diagram needs an RFC first, and who must sign it?
108. [Likely] How would you split this design between a small research team and a platform engineering team, and where are the hand-offs?

## References

Sources 1-3 are official UiPath pages. They exist but could not be reached from this environment (blocked by DNS or the proxy), so check them yourself. Sources 4-11 were fetched and read in this review. None of them reports a UiPath interview question; they back the product facts behind the *(context: [n])* markers and the format notes.

1. UiPath Careers, uipath.com (official careers page; blocked in this run, not read). https://www.uipath.com/company/careers
2. UiPath Blog, uipath.com (official product and AI announcements; blocked in this run, not read). https://www.uipath.com/blog
3. UiPath Documentation portal, docs.uipath.com (Document Understanding, Communications Mining, Agents and Maestro documentation; blocked in this run, not read). https://docs.uipath.com
4. UiPath, "uipath-python" README: Python SDK and CLI for UiPath Cloud Platform, with services for processes, assets, buckets, connections, context grounding (semantic search over knowledge indexes), jobs, queues and Action Center tasks (official; fetched). https://raw.githubusercontent.com/UiPath/uipath-python/main/README.md
5. UiPath, "uipath-langchain-python" README: build and deploy LangGraph agents on UiPath, with human-in-the-loop through Action Center (official; fetched). https://raw.githubusercontent.com/UiPath/uipath-langchain-python/main/README.md
6. UiPath, "uipath-integrations-python" README: agent runtimes for Google ADK, LangChain, LlamaIndex, Microsoft Agent Framework, OpenAI Agents and PydanticAI, all implementing the UiPath Runtime Protocol (official; fetched). https://raw.githubusercontent.com/UiPath/uipath-integrations-python/main/README.md
7. UiPath, "uipath-mcp-python" README: hosting MCP servers on the UiPath platform (official; fetched). https://raw.githubusercontent.com/UiPath/uipath-mcp-python/main/README.md
8. UiPath, "uipath-runtime-python" README: the base contracts for agent runtimes, covering execution context, event streaming, tracing, structured errors, durable execution and human-in-the-loop (official; fetched). https://raw.githubusercontent.com/UiPath/uipath-runtime-python/main/README.md
9. UiPath, "uipath-dev-python" README: developer console for running and debugging runtimes, with structured logs and OpenTelemetry traces (official; fetched). https://raw.githubusercontent.com/UiPath/uipath-dev-python/main/README.md
10. alirezadir, "Machine-Learning-Interviews" (AIMLInterviews), Behavioral and Leadership chapter: common behavioral and leadership prompts and the STAR method; general guide, not specific to UiPath (fetched). https://raw.githubusercontent.com/alirezadir/Machine-Learning-Interviews/main/src/behavioral/behavior.md
11. alirezadir, "Machine-Learning-Interviews" (AIMLInterviews), ML System Design chapter: notes that in 2026 GenAI and agent design rounds weigh evaluation, cost, latency, guardrails and human fallback; general guide, not specific to UiPath (fetched). https://raw.githubusercontent.com/alirezadir/Machine-Learning-Interviews/main/src/MLSD/ml-system-design.md
