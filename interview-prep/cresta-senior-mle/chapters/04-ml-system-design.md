# 04. ML System Design: Real-Time Agent Assist, AI Agents, Conversation Intelligence, QA Automation, Knowledge Assist and Multi-Tenant LLM Serving

> **Tagging system.**
> - **[Reported] [n]**: a candidate report or question bank attributes this question, or a close variant, to Cresta. [n] points to the References list.
> - **[Likely]**: inferred from Cresta's products, engineering blog and ML job postings. No candidate has reported it. Where a [Likely] item scales up a coding question reported for another round, the reference points to that coding report.
>
> **Diagram prompts.** Most design questions include a "Sketch" follow-up. It names the diagram the interviewer is likely to ask you to draw on the whiteboard. The prompt names the boxes and arrows to include. It does not give the answer.
>
> **Research limits.** In this review the egress proxy refused every cited domain (cresta.com, prachub.com, Glassdoor, Blind, 1point3acres, Greenhouse, Built In, fireworks.ai), and the shared web-search budget was already used up, so no cited page could be opened and no new candidate reports could be searched for. Instead, each reference was checked against the reviewed reference lists and excerpts in chapters 01, 03 and 07 of this guide. A citation that no excerpt supported was removed (see the note after the References). The only Cresta ML system design prompts on record are still the agent evaluation platform and "How would you deploy a model in production?". Two prompts from adjacent Cresta rounds (RAG and hallucination for FDE; decoder component design) were added as [Reported] variants, and their tags name the round. All other new questions are [Likely]. Confirm the round format with your recruiter.

## Format notes for this round

- **Round exists and lasts about an hour.** A 2025-26 Senior MLE report lists a systems design round after a recruiter call, an LC-medium round and a "vibe coding" round. The sample prompt was "How would you deploy a model in production?" [1]. Senior/Staff SWE reports describe one hour of system design next to one hour of coding and a hiring-manager chat (SWE loop) [3].
- **Prompt flavor: LLM agents and evaluation.** PracHub lists "Design an Evaluation Platform for AI Agents Across Prompt and Model Changes" as a Cresta ML system design question. A linked Cresta SWE interview experience describes building an eval platform for AI agents [5][6].
- **Adjacent reported prompts.** A 2026 FDE technical screen asked "What is RAG, and how do you prevent hallucination?" [26]. PracHub lists "Design sequence decoding with greedy and beam search" for Cresta [28]. A rate-limiter question and a worker-pool/orchestrator question appear in Cresta coding rounds [1], and so does a job dispatcher/executor question (SWE, 2026) [29]. Design interviewers may scale any of these up.
- **Adjacent rounds.** A 2025 Cresta MLE phone screen focused on greedy and beam decoding [7], so design interviewers may probe generation and decoding choices. Other Cresta engineering loops have included a project-presentation round, and the process takes about 25 days on average [2].
- **Interviewer.** Most likely a senior or staff MLE from the target track: Agentic Assist, Agent and System Quality, Voice/ASR or Insights. This is inferred from postings [8][9][10][11].
- **Senior bar (from postings).**
  - End-to-end ownership from research to production [11].
  - Evaluation of non-deterministic systems: offline benchmarks, online experiments and LLM-as-a-judge [9].
  - Diagnosing hallucination, retrieval errors, tool misuse, context drift and prompt brittleness [8][9].
  - Designing for latency, cost, scalability and security, including real-time streaming inference for voice [10].
  - Recruiters have flagged LLM, RAG and agent fundamentals as in scope [4].
- **What to expect (inferred).** Requirements, then an architecture sketch, then one or two deep dives: latency budget, evaluation, multi-tenancy or rollout. Expect probes on how you measure quality without labels and how you ship safely to many enterprise customers.

## Questions

### A. Reported prompts (this round, plus close variants from adjacent Cresta rounds)

1. Design an evaluation platform for AI agents that catches regressions when prompts or models change. [Reported] [5][6]
   - Sketch: how do the test-set store, scenario simulator, agent-under-test runner, judge/evaluator layer, results store and CI gate connect?
   - How do you version prompts, model IDs, tools, knowledge snapshots and test sets so that a run is exactly reproducible?
   - Which checks are deterministic (tool-call arguments, policy rules), and which need an LLM judge?
   - How many reruns per scenario do you need given non-determinism? What statistical test declares a regression?
   - How do you score the trajectory (tool calls, reasoning steps) separately from the final outcome?
   - How would you scale it to thousands of scenarios across hundreds of customer-specific agents? What does a full run cost?
   - Who consumes the results (ML engineers, forward-deployed engineers, customers), and how does that change the UX?
2. How would you deploy a model in production? [Reported] [1]
   - Sketch: what path does a model take from the registry through shadow, canary and full rollout, with monitoring and rollback?
   - What would you do differently for a real-time model (assist suggestions) and a batch model (post-call QA scoring)?
   - What signals in the first 24 hours trigger an automatic rollback?
   - How do you manage per-customer model versions and configuration?
3. What is RAG, and how do you prevent hallucination? (Reported as a conceptual question in a Forward Deployed Engineer technical screen. Expect a senior MLE design round to extend it: build RAG over an enterprise knowledge base for a customer-facing agent and show where hallucinations are caught.) [Reported] [26]
   - Sketch: ingestion, chunking, index, query rewriting, retrieval, re-ranking, generation with citations, groundedness check, fallback ("I don't know" or hand off to a human).
   - At which stage does each hallucination type start (retrieval miss, stale article, generator going beyond the retrieved text), and which check catches it?
   - What do you do when the groundedness check fails mid-conversation on a voice call?
4. Design sequence decoding with greedy and beam search as a reusable component. (PracHub title; the matching 2025 MLE phone screen was for the AI Agent team.) [Reported] [28][7]
   - What interface lets greedy, beam and sampling strategies share code and plug into a serving loop?
   - How does beam width trade quality against latency and memory when the component runs inside a real-time service?
   - How would you add stop conditions, length penalties and constraints (banned phrases, required schema) without forking the code?

### B. Real-time Agent Assist (live guidance for human agents)

5. Design real-time Agent Assist: listen to a live call and show the human agent next-best-action hints, behavior prompts and suggested responses. [Likely]
   - Sketch: telephony/CCaaS audio fork, streaming ASR, transcript bus, inference graph, suggestion service, agent desktop UI. Where are the queues?
   - What is the end-to-end latency budget from spoken words to on-screen hint? How do you split it across ASR, models and UI?
   - How do you handle partial transcripts compared with final ones? Do you trigger on partials? (Cresta publishes both, about once per second [16].)
   - How do you decide when to show a hint and when to stay quiet, so agents don't get alert fatigue?
   - How do you measure impact on handle time, CSAT or conversion when top agents already behave well?
6. Design the "moment detection" layer that fires guidance triggers (for example "customer mentioned cancellation" or "agent forgot disclosure") from a live transcript. [Likely]
   - Which triggers are rules, which are small classifiers, and which are LLM calls? How do you decide?
   - How do you let customers define new triggers without a training run?
   - How do you evaluate trigger precision and recall per customer with few labels?
7. Design suggested-response (smart compose) generation for chat agents. [Likely]
   - Sketch: which context window builder, retrieval, generator, ranker and UI feedback loop would you draw?
   - How do you personalize to a customer's tone and policies: prompting, retrieval of good past replies, or fine-tuning? (See [20].)
   - What offline metric best predicts agent acceptance rate?
   - How do you avoid suggesting something that violates compliance?
8. Design live call summarization that is ready the moment the call ends (auto-notes and dispositions). [Likely]
   - Do you summarize incrementally during the call or once at the end? What does each cost in latency and money?
   - How do you map the summary to the customer's CRM fields and disposition codes?
   - How do you measure factual consistency of summaries at scale?
9. Design an inference graph that runs many models per conversation (ASR, PII redaction, intent, sentiment, hints, summarization) with customer-specific configuration. [Likely]
   - Sketch: a DAG of model nodes with dependencies. Where do streaming nodes end and batch nodes begin? (See [15].)
   - How do you skip nodes a customer has not enabled, and how do you add a node without redeploying the graph?
   - How do you stop one slow node from blocking real-time outputs (timeouts, fallbacks, partial results)?
   - How would you trace a single conversation through the graph for debugging?
10. Design a real-time sentiment and escalation-risk detector for live calls. [Likely]
   - How do you combine acoustic signals (prosody, overlap, silence) with lexical ones?
   - How do you set per-customer alert thresholds and keep them calibrated over time?
11. Design supervisor live-monitoring: a dashboard that shows which of 2,000 concurrent calls need attention right now. [Likely]
   - Which scoring and ranking would you use, and how often do you refresh it?
   - How do you scale the fan-out of updates to many supervisor screens?
12. How would you A/B test a new Agent Assist model when agents, queues and customers differ widely? [Likely]
    - Do you randomize by agent, by call or by team? What about spillover between agents?
    - How long must the test run to detect a 2% handle-time change?
13. Design the delivery layer that pushes live hints, transcripts and knowledge cards to tens of thousands of agent desktops (browser or CCaaS-embedded widget). [Likely]
    - WebSockets, server-sent events or polling? How do you handle reconnects, ordering and duplicate hints?
    - How do you keep the hint tied to the correct call when an agent transfers, holds or conferences?
14. Design assist for chat agents who handle three to five conversations at once: which conversation needs the agent's attention next, and what to suggest there. [Likely]
    - Which signals rank the conversations (customer wait time, sentiment, SLA, open question)?
    - How do you evaluate the ranking offline?
15. Design an offline evaluation of a new Agent Assist hint model by replaying historical calls, before any live A/B test. [Likely]
    - How do you judge whether a hint shown at time t would have helped, when the agent never saw it?
    - How do you combine replay metrics (precision of hint timing, coverage) with a small human review?
16. In suggested-response generation, would you use greedy decoding, beam search or sampling of n candidates followed by a ranker? Justify the choice on latency, diversity and acceptance rate. [Likely] [7]
    - How does the choice change for a fine-tuned small model compared with a frontier API?
17. Design zero-shot intent and moment detection that works for a new customer on day one, then improves as data arrives. [Likely]
    - How do you start with LLM prompts over customer-provided descriptions and move to cheaper models once labels accumulate?
    - How do you detect the point at which the cheap model is good enough to replace the LLM?
18. Design real-time translation for a live call, where the agent works in English and the customer speaks another language. [Likely]
    - Where does translation sit relative to ASR, hints and the agent's reply (text or synthesized speech)?
    - How do you stop translation errors from triggering wrong hints or compliance alerts?

### C. Voice AI agents (autonomous customer-facing agents)

19. Design a voice AI agent that handles inbound customer calls end to end, with responses in under a second. [Likely]
    - Sketch: SIP/telephony, VAD and endpointing, streaming ASR, dialog/LLM orchestrator with tools, streaming TTS, and back. Label the latency of every hop.
    - Which component metrics do you track (ASR final latency, LLM time-to-first-token, TTS time-to-first-byte), and how do you measure end-to-end latency? (See [14].)
    - How do you detect end of turn without cutting off slow speakers?
    - How do you handle barge-in when the customer interrupts mid-sentence?
    - Which network-level choices matter: co-location, connection reuse, streaming APIs, warm pools?
20. Design the tool-use layer for an AI agent that can look up orders, issue refunds and update accounts in customer systems. [Likely]
    - How do you constrain tools so the agent cannot make unauthorized commitments (refund limits, identity verification first)?
    - How do you handle slow or failing downstream APIs during a live voice call? What does the agent say while waiting?
    - How do you test tool calls safely without touching production customer systems?
21. Design the handoff from an AI agent to a human agent. [Likely]
    - What signals trigger escalation, and how do you avoid escalating too early or too late?
    - What context (summary, verified identity, attempted actions) do you pass to the human, and in what format?
22. Design an agent-building system in which a customer's business process (SOPs, flows, policies) becomes a working AI agent. [Likely]
    - Should the agent be a structured flow, a free-form LLM with tools, or a hybrid? What trade-offs does each bring?
    - How can an "agent that builds agents" draft the configuration from historical conversations? (See [25].)
    - How do you keep the agent compliant when the SOP changes?
23. Design guardrails for a customer-facing LLM agent: prompt injection, off-topic requests, PII leakage and policy violations. [Likely]
    - Where in the pipeline do guardrails run (input, planning, output), and what latency does each add?
    - How do you ground responses and detect hallucinations before they are spoken? (See [23].)
    - Which guardrails are small classifiers, and which are LLM checks?
24. Design a multi-agent architecture with a router and specialized sub-agents (billing, tech support, sales). [Likely]
    - When does this beat one agent with many tools?
    - How do you keep shared conversation state consistent across handoffs?
    - How do you debug a failure that spans several agents?
25. How would you keep a voice agent's latency stable during peak traffic (Monday morning, outages that cause call spikes)? [Likely]
    - What degrades first, and what is your graceful-degradation plan (smaller model, cached responses, queueing)?
26. Design multilingual support for a voice agent (language ID, code-switching, per-language ASR and TTS). [Likely]
    - How do you evaluate quality in languages your team does not speak?
27. Design context management for a long call (30-60 minutes, many tool calls) so the voice agent stays coherent without blowing the latency or token budget. [Likely]
    - What do you keep verbatim, what do you summarize, and what do you store as structured state (verified identity, open tasks)?
    - How do you test for context drift late in a long conversation? (See [8][9].)
28. Design identity verification inside a voice AI agent (knowledge-based questions, one-time passcodes, account lookup) before it can take sensitive actions. [Likely]
    - How do you enforce "verify first" in code rather than trusting the prompt?
    - How do you handle ASR errors on spelled names, dates of birth and account numbers?
29. Design detection of social-engineering and fraud attempts against a customer-facing AI agent (account takeover, refund abuse, repeated probing). [Likely]
    - Which signals are per-call, and which need cross-call aggregation per account or phone number?
    - What does the agent do when risk is high but not certain?
30. Design an outbound voice AI agent for campaigns (payment reminders, appointment confirmations), including answering-machine and IVR detection. [Likely]
    - How do you classify human, voicemail and IVR within the first seconds of the call?
    - How do calling-window rules, consent records and retry policies enter the design?
31. Design memory across conversations for returning customers, so an AI agent can pick up an issue from last week. [Likely]
    - What do you store, for how long, and how do you respect retention and deletion rules?
    - How do you stop stale or wrong memories from steering the agent?
32. Design an AI agent for asynchronous messaging channels (SMS, WhatsApp, web chat) where one conversation can span hours or days. [Likely]
    - How do you persist and resume conversation state, and how do you handle a customer who replies after their session has expired?
    - What changes in latency targets, evaluation and handoff compared with voice?

### D. Agent evaluation, simulation and quality

33. Design a customer simulator that generates synthetic callers to test an AI agent before launch. [Likely]
    - Sketch: how do the persona/goal generator, simulated customer LLM, agent under test, conversation transcript and evaluators connect?
    - How do you seed personas and goals from real conversation data so coverage matches production? (See [24].)
    - How do you stop the simulator and the agent from sharing the same blind spots (same base model)?
34. Design a layered ("Swiss-cheese") reliability program for a production AI agent. [Likely]
    - What does each layer catch: unit tests on tools, simulated conversations, human review, production monitors? What does each miss? (See [13].)
    - How do production failures become new regression tests automatically?
35. Design an LLM-as-a-judge evaluator service used for both agent testing and production scoring. [Likely]
    - How do you calibrate judges against expert labels, and what agreement level is good enough? (See [12].)
    - How do you detect judge drift when you upgrade the judge model?
    - How do you stop position bias, verbosity bias and self-preference from skewing results?
    - How do you keep per-conversation judging cost manageable at millions of conversations?
36. Design production monitoring for AI agents across hundreds of enterprise deployments. [Likely]
    - Which label-free signals indicate quality problems (escalation rate, repeated questions, abandonment, tool error rates)?
    - How do you catch a regression that affects only one customer or one intent?
    - How do you alert without paging on noise?
37. How do you measure whether an AI agent "resolved" a call when there is no survey and no disposition code? [Likely]
    - How would you validate an inferred-outcome model without ground truth?
38. Design the release process for a prompt change that ships to 50 customer agents at once. [Likely]
    - Which tests gate it, and how do you stage the rollout across customers?
    - What is your rollback story when a customer has customized the prompt?
39. Design a "shadow mode" launch in which a new AI agent runs silently beside human agents on live traffic, to estimate containment and risk before it speaks to customers. [Likely]
    - How do you compare the agent's proposed actions with what the human actually did?
    - Which disagreements are agent errors, and which are human errors or policy differences?
40. Extend the customer simulator to voice: synthetic callers with TTS voices, accents, background noise, interruptions and silence. [Likely] [24]
    - Which failures appear only in audio (endpointing, barge-in, ASR on spelled data) and never in a text simulation?
    - How do you keep voice simulation cheap enough to run on every release?
41. Design a test harness that measures how robust the LLM layer is to ASR errors. [Likely] [27]
    - How do you inject realistic ASR noise (substitutions on entities, dropped words, wrong partials)?
    - When ASR errors hurt the outcome, do you fix ASR, the prompt, or add a confirmation step?
42. Design the curation pipeline for "golden" evaluation sets taken from production conversations. [Likely]
    - How do you sample (stratified by intent, failure-weighted, new-scenario detection) and redact PII before engineers see the data?
    - How do you keep the set from going stale or leaking into training data?
43. Design an audit and replay system that can reproduce exactly why an AI agent said or did something on a call three months ago, for a regulated customer. [Likely]
    - What must you log (prompt version, model version, retrieved documents, tool inputs and outputs, guardrail decisions), and at what storage cost?
    - How do you replay when the model or the knowledge base has changed since then?

### E. Conversation intelligence and insights

44. Design topic discovery over millions of unlabeled contact-center conversations, producing a hierarchical taxonomy that admins can edit. [Likely]
    - Sketch: a batch pipeline from conversations to embeddings or LLM extraction, clustering, LLM labeling, the taxonomy store, and the assignment service.
    - How do you keep assignments stable when admins merge, split or rename topics? (See [19].)
    - How do you overlay outcomes (resolution, sales, churn) to rank problem areas?
45. Design an "AI analyst" that answers open-ended business questions over a customer's conversation corpus ("Why did cancellations rise last week?"). [Likely]
    - How do you combine per-conversation LLM extraction with SQL-style aggregation?
    - How do you stop it from presenting a weak correlation as a causal finding?
    - How do you bound cost when a question would touch ten million conversations?
46. Design a driver-analysis system that finds which agent behaviors correlate with good outcomes (the basis for coaching and assist hints). [Likely]
    - How do you control for confounders like call type, customer mood and queue?
    - How do you turn the findings into real-time guidance triggers?
47. Design a real-time emerging-issue detector that flags new spikes (an outage, a billing bug) within minutes. [Likely]
    - How do you separate a true spike from seasonality and noise?
    - How do you name the new issue automatically?
48. Design structured extraction from conversations (reason for call, products mentioned, competitor mentions) into a customer's data warehouse. [Likely]
    - How do you enforce schemas and handle extraction failures?
    - How do you backfill when the schema changes?
49. How would you design the storage and indexing layer for searching billions of utterances by keyword, semantic meaning and metadata filters? [Likely]
    - Which parts use a vector index, which use an inverted index, and how do you do hybrid ranking?
    - How do you enforce per-tenant isolation and retention policies in the index?
50. Design the event pipeline that turns live and post-call conversation events into analytics tables, with late-arriving transcripts and reprocessing when a model changes. [Likely]
    - How do you get idempotent, exactly-once results per conversation?
    - How do you recompute a year of a derived metric after an extraction model upgrade without double counting?

### F. QA automation and coaching

51. Design automated QA that scores 100% of calls against each customer's rubric. [Likely]
    - Sketch: how do the rubric definition UI, rubric compiler (prompt or classifier per item), batch scoring workers, evidence highlighting and the QA analyst review queue connect?
    - How do you turn a free-text rubric item ("agent showed empathy") into a reliable, auditable judgment?
    - How do you show evidence (timestamps, quotes) so supervisors trust the score?
    - How do you handle disputes, and feed corrections back into the scorer?
52. How do you measure and improve agreement between automated QA scores and human QA analysts? [Likely]
    - What do you do when the human analysts don't agree with each other?
53. Design a system that lets a customer add a new QA criterion and have it live across past and future calls within an hour. [Likely]
    - Few-shot prompting, a small classifier, or both? How do you validate before going live?
    - What does a backfill over a year of calls cost?
54. How would you stop agents from gaming the automated QA scorer (for example, saying keywords without the real behavior)? [Likely]
55. Design an automated coaching system that picks which behavior each agent should work on and tracks improvement. [Likely]
    - How do you attribute improvement to coaching and not to seasonality or call mix?
56. Design compliance monitoring (required disclosures, prohibited statements) with near-zero miss tolerance in regulated industries. [Likely]
    - How do you trade precision against recall when a miss means regulatory risk?
    - Which checks run in real time, and which run post-call?
57. Design speaker attribution for QA scoring when some customers only provide mono (single-channel) recordings. [Likely]
    - How do diarization errors flow into rubric scores ("agent stated the disclosure"), and how do you guard against them?
    - When would you fall back to an LLM to infer the speaker from content?

### G. Knowledge Assist (RAG in live conversations)

58. Design Knowledge Assist: surface answers from a customer's knowledge base to an agent during a live conversation. [Likely]
    - Sketch: KB connectors, ingestion and chunking, embedding and indexing, query formulation from conversation context, retrieval and re-ranking, grounded answer generation, citation UI, feedback.
    - How do you turn messy, implicit conversation context into a retrieval query? (See [18][20].)
    - When should the system answer proactively, and when should it wait for the agent to ask?
    - What latency is acceptable, and which steps would you cache?
59. How would you serve Knowledge Assist with a fine-tuned model at low latency and cost? [Likely]
    - What are the trade-offs of fine-tuning compared with prompting a frontier model for this task? (See [20][21].)
    - How do you evaluate answer quality and groundedness offline?
60. Design the ingestion pipeline that makes enterprise knowledge "AI-ready": duplicate, stale, contradictory and poorly structured KB articles. [Likely]
    - How do you detect conflicting articles and surface them to the content owner? (See [18].)
    - How do you handle PDFs, tables, and screenshots inside articles?
    - How do you re-index incrementally when articles change?
61. How do you detect and handle knowledge gaps: questions customers ask that the KB cannot answer? [Likely]
    - How would you mine gaps from conversations and propose new articles?
62. Design retrieval evaluation for Knowledge Assist when you have no labeled query-document pairs. [Likely]
    - How would you bootstrap labels with LLMs, and how far can you trust them?
    - Would you fine-tune embeddings per domain? Where would you get hard negatives?
63. How do you adapt one retrieved answer to channel constraints: a short spoken answer for voice, a formatted answer for chat? [Likely]
64. Plan an embedding-model upgrade for Knowledge Assist and conversation search that re-embeds billions of vectors across tenants without downtime or quality loss. [Likely]
    - How do you run the old and new indexes side by side, compare retrieval quality, and cut over per tenant?
    - What does the re-embedding job cost, and how do you schedule it around real-time traffic?

### H. Multi-tenant LLM serving, models and cost

65. Design multi-tenant LLM serving: one shared base model with per-customer adapters, prompts and policies. [Likely]
    - Sketch: API gateway with tenant auth, request router, adapter registry, GPU serving pool with multi-LoRA batching, autoscaler and per-tenant metering.
    - How do you serve hundreds of LoRA adapters on shared GPUs without per-adapter replicas? (See [22].)
    - How do you prevent cross-tenant data leakage in caches, logs and training data?
    - How do you enforce per-tenant rate limits and noisy-neighbor isolation?
66. Design an LLM serving stack that meets p99 latency targets for real-time features. [Likely]
    - What do continuous batching, KV-cache management, quantization and speculative decoding each trade off?
    - Why is CPU-based autoscaling a poor fit for GPU inference, and what signal would you scale on?
    - How do you keep cold-start times acceptable?
67. Design a model router that chooses between a self-hosted small model and a frontier API per request. [Likely]
    - Which features drive the routing decision (task, customer tier, input length, confidence)?
    - How do you prove the router does not hurt quality?
68. Design the domain adaptation of a base LLM for contact centers, in the spirit of Ocean-1. [Likely]
    - Which data would you use (self-instruct, real transcripts, preference data), and how do you handle customer data rights? (See [17].)
    - How do you show it beats a frontier API on quality, latency and cost for your tasks?
    - How do you keep it current as new open base models come out?
69. Plan the migration of one high-volume feature from a frontier LLM API to a distilled, self-hosted model. [Likely]
    - What quality bar must the new model meet, and how is it measured?
    - What is the break-even volume where self-hosting is cheaper?
70. Design per-customer fine-tuning as a product feature: training, evaluation, deployment and rollback for each tenant. [Likely]
    - How do you schedule training jobs fairly and cap cost?
    - How do you stop one customer's adapter from regressing silently after a base-model upgrade?
71. Design a cost-accounting and budgeting system for LLM usage per customer and per feature. [Likely]
    - How do you attribute GPU cost on shared, batched infrastructure?
    - What would you do when one customer's usage spikes 10x?
72. How would you design for data residency and isolation (EU customers, regulated industries) on a shared LLM platform? [Likely]
    - What changes in serving, storage and training pipelines?
73. Design the prompt and configuration management service for hundreds of customer-specific agents and assist features. [Likely]
    - How do base prompts, vertical templates and per-customer overrides inherit from each other?
    - How do you version, review, diff and roll back a configuration, and tie each production response to the exact version that produced it?
74. Scale up the reported rate-limiter coding question: design per-tenant rate limiting and quota enforcement in front of shared LLM serving and third-party LLM APIs. [Likely] [1]
    - Token-based or request-based limits? How do you protect real-time traffic from a tenant's batch backfill?
    - How do you enforce limits consistently across regions and replicas?
75. Design scheduling that shares a GPU fleet between real-time inference (live assist, voice agents) and batch work (post-call QA, insights backfills, evaluation runs). [Likely]
    - How do you pre-empt batch work when live traffic spikes, without losing progress?
    - How do you plan capacity for diurnal traffic across time zones?
76. Design a distillation pipeline in which a frontier LLM labels data and small, fast models serve real-time classifiers and triggers. [Likely]
    - How do you choose which tasks to distill, and how do you refresh the student when the teacher or the customer's taxonomy changes?
    - How do you catch cases where the student confidently disagrees with the teacher?
77. How would you guarantee valid structured outputs (tool-call arguments, CRM fields, QA scores) from the LLM in your serving stack? [Likely] [28]
    - Constrained decoding against a JSON schema, validate-and-retry, or fine-tuning? What does each cost in latency?
    - How does constrained decoding interact with batching and speculative decoding?

### I. Speech and ASR systems

78. Design a streaming ASR service for contact-center audio at thousands of concurrent calls. [Likely]
    - Sketch: audio ingestion (dual-channel), VAD, streaming encoder workers, partial and final hypothesis emitter, and the transcript event bus.
    - How do you balance accuracy and latency (chunk size, lookahead)?
    - How do you pack many streams onto one GPU?
79. Design customer-specific vocabulary adaptation for ASR (product names, account terms). [Likely]
    - Contextual biasing, fine-tuning, or post-correction with an LLM: which and why? (See [27].)
80. Design an ASR quality program across accents, noise and domains. [Likely]
    - How does WER relate to downstream task quality, and which do you optimize?
    - How do you get ground-truth transcripts cheaply and detect a regression for one customer?
81. Design PII redaction for audio and transcripts that runs before anything is stored or used for training. [Likely]
    - How do you redact both the transcript and the matching audio segment?
    - What does a missed credit-card number cost, and how does that set your recall target?

### J. Data, feedback loops and platform

82. Design the feedback flywheel: agent accept, edit and ignore signals on suggestions, flowing back into model improvement. [Likely]
    - What biases does implicit feedback introduce (position, presentation, agent skill), and how do you correct for them?
    - How do you respect customer consent and data-usage terms when you train on their data?
83. Design the training-data platform for many enterprise tenants: collection, labeling, consent, lineage and deletion. [Likely]
    - How do you honor a deletion request for data that has already been used in training?
84. Design a labeling program that combines LLM pre-labels with expert human review for new tasks. [Likely]
    - How do you decide which examples go to humans (active learning, disagreement sampling)?
85. Design the experimentation platform for LLM features (prompt variants, model variants) across enterprise customers who may refuse experiments. [Likely]
    - How do you get statistical power when each customer is small?
86. Design an observability and tracing system for LLM pipelines (prompts, retrieved documents, tool calls, outputs) that is safe for PII. [Likely]
    - What do you sample and store, and for how long?
87. Design onboarding for a new enterprise customer, so that their deployment reaches production quality in weeks instead of months. [Likely]
    - What can be automated from their historical conversations (intents, KB, SOPs, QA rubric)?
88. Scale up the reported worker-pool and job-dispatcher coding questions: design the orchestrator that dispatches AI-agent and evaluation jobs across a worker fleet. [Likely] [1][29]
    - Round-robin, least-loaded or priority scheduling? How do you handle stuck or failed jobs, retries and idempotency?
    - How do you give each tenant a fair share of the fleet?
89. Design tool onboarding: turn a customer's API spec (for example OpenAPI) into tools an AI agent can call, with a sandbox and tests. [Likely] [25]
    - How do you write tool descriptions the model uses correctly, and detect tool misuse after launch?
    - How do you mock customer systems for simulation without drifting from the real APIs?

### K. Cross-cutting deep-dive probes (expect several in any design)

90. What are your functional and non-functional requirements, and which one dominates the design? [Likely]
91. Give a back-of-envelope estimate: concurrent calls, utterances per second, tokens per day, GPUs needed. [Likely]
92. Where is the latency budget spent, and which component would you optimize first? [Likely]
93. What is your offline evaluation, and how do you know it predicts online results? [Likely]
94. Which online metrics would you put on the launch dashboard, and which one is the north star? [Likely]
95. What happens when the LLM provider has an outage or rate-limits you mid-call? [Likely]
96. How does the design change for chat compared with voice, and for 10 customers compared with 1,000? [Likely]
97. Where could the system hallucinate or mislead a human agent, and how do you contain the blast radius? [Likely]
98. How do you handle a customer who demands that their data is never used for training? [Likely]
99. What would make you choose a classic ML model (gradient boosting, small classifier) over an LLM for one component? [Likely]
100. How would you explain a quality regression in this system to an enterprise customer's VP of operations? [Likely]
101. What are the top three ways this system fails in production, and how would you detect each? [Likely]

## References

1. Glassdoor, "Cresta (CA) Interview Questions" (Senior Machine Learning Engineer review with systems design round; "How would you deploy a model in production?"), 2025-2026. https://www.glassdoor.co.in/Interview/Cresta-CA-Interview-Questions-E3180909.htm
2. Glassdoor, "Cresta (CA) Interview Questions" (about 25-day process; SWE project presentation), 2025-2026. https://www.glassdoor.ca/Interview/Cresta-CA-Interview-Questions-E3180909.htm
3. Blind, "Cresta interview process", 2024-2025. https://www.teamblind.com/post/cresta-interview-process-jmfnzjoa
4. Blind, "Anyone interview with Cresta lately?", 2025. https://www.teamblind.com/post/anyone-interview-with-cresta-lately-zvj0uhq2
5. PracHub, "Cresta ML System Design Interview Questions" ("Design an Evaluation Platform for AI Agents Across Prompt and Model Changes"), 2026. https://prachub.com/companies/cresta/categories/ml-system-design
6. PracHub, "Cresta Software Engineer Interview Experience: System Design Round: Build an Eval Platform for AI Agents". https://prachub.com/interview-experiences/cresta-software-engineer-interview-experience-system-design-round-build-an-eval-platform-for-ai-agents
7. 1point3acres, "Cresta Machine Learning Engineer Tech Phone Interview", 2025. https://www.1point3acres.com/interview/thread/1140481
8. Cresta on Greenhouse, "Senior Machine Learning Engineer". https://job-boards.greenhouse.io/cresta/jobs/4249943008
9. Cresta on Greenhouse, "Staff Machine Learning Engineer". https://job-boards.greenhouse.io/cresta/jobs/4741283008
10. Cresta on Greenhouse, "Senior Machine Learning Engineer, Voice Experience". https://job-boards.greenhouse.io/cresta/jobs/5155675008
11. Built In Vancouver, "Senior Machine Learning Engineer, Cresta" (Agentic Assist track). https://builtinvancouver.org/job/senior-machine-learning-engineer/8619914
12. Cresta Blog, "How Cresta Grades Its Agents: Evaluators in an LLM World". https://cresta.com/blog/how-cresta-grades-its-agents-evaluators-in-an-llm-world
13. Cresta Blog, "Why AI Agent Evaluations Fail, and How the Swiss-Cheese Model Prevails". https://cresta.com/blog/why-ai-agent-evaluations-fail----and-how-the-swiss-cheese-model-prevails
14. Cresta Blog, "Engineering for Real-Time Voice Agent Latency". https://cresta.com/blog/engineering-for-real-time-voice-agent-latency
15. Cresta Blog, "Understanding Cresta's Voice Platform: ML Services, Inference Graphs and Real-Time Intelligence", 2025. https://cresta.com/blog/understanding-crestas-voice-platform-ml-services-inference-graphs-and-real-time-intelligence
16. Cresta Blog, "Understanding Cresta's Voice Platform: The Voice Stack", 2025. https://cresta.com/blog/understanding-crestas-voice-platform-the-voice-stack
17. Cresta Blog, "Introducing Ocean-1: World's First Contact Center Foundation Model", 2023. https://cresta.com/blog/introducing-ocean-1-worlds-first-contact-center-foundation-model
18. Cresta Blog, "AI-Ready Knowledge for Contact Centers: Closing the Gap Between KBs and AI", 2025. https://cresta.com/blog/ai-ready-knowledge-for-contact-centers-closing-the-gap-between-kbs-and-ai
19. Cresta Blog, "Introducing the Next Generation of Cresta Insights", 2026. https://cresta.com/blog/introducing-the-next-generation-of-cresta-insights-a-complete-system-for-enterprises-to-understand-their-cx
20. Cresta Blog, "How Fine-Tuned LLMs Power Knowledge Assist, Summarization and Chat Suggestions", 2024. https://cresta.com/blog/how-fine-tuned-llms-power-knowledge-assist-summarization-and-chat-suggestions
21. Fireworks AI Blog, "Story: Cresta Knowledge Assist", 2024. https://www.fireworks.ai/blog/story-cresta-knowledge-assist
22. Fireworks AI Blog, "Multi-LoRA", 2024. https://fireworks.ai/blog/multi-lora
23. Cresta Blog, "Grounding Reality: How Cresta Tackles LLM Hallucinations in Enterprise AI". https://cresta.com/blog/grounding-reality---how-cresta-tackles-llm-hallucinations-in-enterprise-ai
24. Cresta Blog, "Cresta Launches Automated AI Agent Testing: Confidence in Every Conversation". https://cresta.com/blog/cresta-launches-automated-ai-agent-testing-confidence-in-every-conversation
25. Cresta Blog, "Cresta Conductor: The Agent for AI Agent Development". https://cresta.com/blog/cresta-conductor-the-agent-for-ai-agent-development
26. Glassdoor, "Cresta (CA) Interview Experience and Questions" (FDE technical screen: RAG and hallucination), 2026. https://www.glassdoor.com/Interview/Cresta-CA-Interview-Questions-E3180909.htm
27. Cresta on Greenhouse, "Senior Machine Learning Engineer (ASR)". https://job-boards.greenhouse.io/cresta/jobs/5101011008
28. PracHub, "Cresta Interview Questions" (company page; "Design sequence decoding with greedy and beam search"), 2025. https://prachub.com/companies/cresta
29. 1point3acres, "Cresta Fulltime Software Engineer Tech Phone Screen: Job Dispatcher-Executor System Implementation" (title seen in a listing; full text not read), 2026. https://www.1point3acres.com/interview/thread/1172433

*Changed in review:* The Blind thread "Cresta senior backend SWE interview" was removed. The only excerpt of it in this guide supports an online assessment, not the one-hour system design round it was cited for here, and ref 26 now points to the Glassdoor FDE report instead. References 1 (now also cited for the reported rate-limiter and worker-pool coding questions), 7, 28 and 29 match excerpts that chapters 01-03 already reviewed. Three generic probes were removed as filler: "what would you build in week one", "what would you hand to a vendor" and "twice or half the budget". Note on [Reported] tags: only questions 1-4 carry the tag. Questions 1 and 2 are reported for this round. Question 3 was reported for an FDE screen, and question 4 for an MLE phone screen. No question was retagged, because every [Reported] attribution matches an excerpt in the sibling chapters. Page contents could not be re-read live.
