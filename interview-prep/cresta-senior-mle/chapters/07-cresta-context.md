# 07. Cresta Product and Research Context: ML System Design Questions Built on Agent Assist, AI Agent, Conversation Intelligence and Ocean Models

> **Scope.** You asked for ML system design content, so this chapter presents Cresta's products and research as design prompts an interviewer could build from them. Each design question has a **Sketch** follow-up that names the diagram you would draw: the boxes and arrows, not the answer. Questions about the product or the company appear only where they set up a design discussion.
>
> **Tagging system.**
> - **[Reported] [n]**: a candidate report or question bank attributes this question, or a close variant, to Cresta. [n] points to the References list.
> - **[Likely]**: inferred from Cresta's products, engineering blog, partner write-ups and ML job postings. No candidate has reported it. Where a [Likely] question builds on a specific Cresta post, that post is cited as context only; the citation does not mean anyone reported the question.
>
> **Research limits.** No cited page could be opened while this chapter was researched or reviewed. In the review, fetches of 17 cited URLs (cresta.com, fireworks.ai, prachub.com, Glassdoor, 1point3acres, Greenhouse, Built In, CX Today, Scoutify) were all refused by the egress proxy or failed DNS lookup, and the shared web-search budget was already used up, so no new candidate reports could be searched for. Every reference is therefore kept only where search-result excerpts gathered for this guide (chapters 01-07) support it, and nothing here comes from a full read of a page. No candidate has reported a question about Cresta's products as such. The Cresta ML system design prompts on record are the agent evaluation platform and "How would you deploy a model in production?" [1][5][6]. Four prompts from adjacent Cresta rounds (decoding in an MLE AI Agent phone screen, RAG and hallucination in an FDE screen, and the recruiter's LLM/RAG/agents prep note) are added as [Reported] variants, and each tag names its round. Facts about the founders are general background with no source cited here; check them before quoting them in an interview.

## Format notes

- **Where this material appears.** No candidate report describes a separate "Cresta context" round. Product knowledge comes up inside the ML system design round, the hiring-manager conversation and the project deep dive [1][2]. For senior MLEs, Glassdoor lists a recruiter call, a LeetCode-medium round, an AI-assisted ("vibe coding") round and a system design round [1].
- **Interviewer.** Most likely a senior or staff MLE, or an engineering manager, from the track you are matched to: Agentic Assist, Agent and System Quality, Voice/ASR or Insights. This is inferred from the job postings [9][10][12][13].
- **Duration.** About 60 minutes for system design, based on reports from comparable Cresta engineering loops [2]. Hiring-manager chats run 30-60 minutes [2].
- **Recruiter prep signal.** One recruiter email told the candidate to expect "key concepts and fundamentals around LLMs, GenAI, RAG, and AI Agents" [2][3].
- **What the senior bar assesses here.** First, whether you can turn a Cresta product into a measurable ML problem with a latency budget, an evaluation plan and a rollout plan. Second, whether your designs reflect real constraints: streaming voice, many enterprise tenants, non-deterministic LLM output, and customers who must be able to trust the system. The postings name hallucinations, retrieval errors, tool misuse, context drift and prompt brittleness as failure modes to diagnose, and accuracy, faithfulness, task completion, latency, cost and robustness as metrics [9][11][12][13].

### Product map to anchor your designs

| Product area | What it does (as Cresta and partners describe it) | Design themes it implies |
|---|---|---|
| Agent Assist / Knowledge Assist | Real-time guidance and knowledge answers for human agents during live calls and chats [15][16][18] | Streaming ASR, retrieval under latency, when to suggest something, measuring impact |
| AI Agent | Autonomous voice and chat agents that handle customer conversations, with testing and an agent-building assistant (Conductor) [14][19][20][24][25] | Tool use, guardrails, escalation, simulation-based testing, voice latency |
| Conversation Intelligence / Insights | Topic discovery, outcome inference, an AI analyst and QA over all conversations [21] | Batch LLM extraction, clustering, judges, causal pitfalls |
| Voice platform | Voice stack and inference graphs of ML services per conversation [17][18] | Model DAGs, per-customer policies, partial transcripts |
| Ocean models and fine-tuned LLMs | Ocean-1 contact-center foundation model; fine-tuned models matching GPT-4 on specific tasks; multi-LoRA serving at Fireworks [7][15][16][22] | Domain adaptation, distillation, multi-tenant adapter serving |
| Quality and evaluation | Evaluators, the Swiss-cheese reliability model, automated agent testing, grounding against hallucination [14][23][24][26] | LLM-as-a-judge, regression gating, groundedness |

## Questions

### A. Agent Assist (real-time guidance for human agents)

1. Design Cresta's real-time Agent Assist. It listens to a live call and shows the human agent next-best-action hints, behavior reminders and knowledge cards. [Likely]
   - Sketch: telephony/CCaaS audio feed, streaming ASR (partial and final transcripts), conversation state store, trigger/moment detectors, suggestion generator, ranking and filtering, push channel to the agent's desktop, and a logging path for feedback.
   - What is the latency budget from the customer's last word to a card on screen, and how do you split it across ASR, detection, retrieval and generation?
   - Which components run on partial transcripts and which wait for final ones? (Cresta's voice-stack post describes pushing both [18].)
   - How do you decide when not to show a suggestion?
2. Cresta Agent Assist uses "moment" or hint triggers such as "customer mentions cancellation". How would you build a trigger system that a customer's admin can configure without ML expertise? [Likely]
   - Sketch: admin UI for trigger definitions, compiler that turns definitions into classifier or LLM prompts, real-time evaluator, per-tenant config store and alerting.
   - When do you use a trained classifier, a few-shot LLM or a keyword rule?
   - How do you estimate a new trigger's precision before it goes live?
3. How would you measure whether Agent Assist improves agent outcomes such as handle time, conversion or CSAT, rather than just correlating with good agents? [Likely]
   - Sketch: experiment assignment (agent-level or team-level), exposure logging, outcome joins, and the analysis pipeline.
   - Why randomize at agent or team level rather than per conversation?
   - How do you handle spillover when agents learn from suggestions and change their behavior?
4. Design the feedback loop that uses agents' accept, edit and ignore actions on suggestions to improve the models. [Likely]
   - Sketch: UI event capture, sessionization, label derivation, bias correction, training-data store and retraining trigger.
   - What does an ignored suggestion actually tell you?
   - How do you correct for position bias and for agents who click everything?
5. Design chat reply suggestions (smart compose) for chat agents, served from a fine-tuned model. [Likely] [15]
   - Sketch: chat event stream, context builder, candidate generator, safety filter, ranker, UI, and a feedback log.
   - How do you keep suggestions consistent with the brand's voice for each customer?
   - What offline metric best predicts how often agents accept a suggestion?
6. Design automatic after-call summarization that fills CRM fields when the call ends. [Likely] [15]
   - Sketch: final transcript, PII redaction, summarizer, field extractor with schema validation, CRM connector and a human edit loop.
   - How do you evaluate summaries against agents' manual notes?
   - What happens when the model invents a field value?
7. Design a real-time supervisor dashboard that flags live calls going badly (escalation risk, compliance misses) so a supervisor can step in. [Likely]
   - Sketch: per-call scoring stream, aggregation by team, ranking of calls needing attention, alerting with suppression.
   - How do you calibrate alert thresholds so supervisors are not flooded?
8. Partial ASR transcripts get revised as the speaker continues. Design the real-time detection layer so a hint or behavior check does not fire, retract and fire again as the transcript changes. [Likely] [18]
   - Sketch: partial-transcript stream with revision IDs, stability window or debounce, detector, fire-once state per moment, and a correction path when a final transcript contradicts a hint already shown.
   - How do you trade the extra latency of waiting for stable text against the cost of a wrong hint?

### B. Knowledge Assist (RAG in live conversations)

9. Design Knowledge Assist, which answers an agent's question, or one detected in the conversation, from the customer's knowledge base during a live call. [Likely] [15][16][20]
   - Sketch: KB ingestion (connectors, chunking, cleaning), embedding index per tenant, query reformulation from conversation context, retriever and reranker, answer generator with citations, and the channel formatter.
   - How do you turn a messy spoken exchange into a good retrieval query?
   - How do you keep one tenant's documents isolated from another's?
10. Cresta has written about the gap between existing knowledge bases and AI-ready knowledge. How would you build a pipeline that finds and fixes KB gaps (missing, stale or contradictory articles)? [Likely] [20]
    - Sketch: unanswered-question mining from conversations, clustering, coverage check against the KB, LLM-drafted articles, human approval and re-indexing.
    - How do you detect two articles that contradict each other?
11. Fireworks reports that Cresta moved Knowledge Assist to fine-tuned models that matched GPT-4 quality at lower cost and latency. How would you plan and run that migration? [Likely] [15][16]
    - Sketch: production traffic sampling, teacher labeling, student fine-tuning, offline evaluation against the teacher, shadow deployment, canary and cutover.
    - What quality bar must the student meet, and on which slices?
    - How do you keep the system working if the fine-tuned model fails?
12. How would you evaluate retrieval for Knowledge Assist when you have no labeled query-document pairs? [Likely]
    - How would you use LLM-generated queries, agent click signals and expert spot checks?
13. How would you make Knowledge Assist's answers grounded and stop it from answering when the KB has no answer? [Likely] [26]
    - Sketch: retrieval confidence, answerability classifier, generation constrained to citations, post-hoc groundedness check, and an "I don't know" path.
    - How do you measure the false-refusal rate as well as the hallucination rate?
14. What is RAG, and how do you prevent hallucination? Then extend it: draw RAG over a customer's knowledge base for Knowledge Assist and mark every point where a hallucination can be caught. [Reported] [27] (reported as a conceptual question in a Forward Deployed Engineer technical screen; the design extension is [Likely])
15. KB articles change during the business day (a new promotion, a policy update). Design ingestion so Knowledge Assist never serves a superseded answer. [Likely] [20]
    - Sketch: KB change feed or crawler, document versioning, incremental chunk and embed jobs, index alias swap, cache invalidation, and the answer log recording which article version was used.
    - How do you find answers already shown to agents that relied on an article that has since been retracted?
16. Design retrieval over past resolved conversations ("how did other agents solve this?") as a second knowledge source next to the KB. [Likely]
    - Sketch: resolved-case mining, PII scrubbing, case summarization, case index per tenant, retriever, and a ranking step that blends cases with KB articles.
    - How do you stop a wrong workaround that one agent used from spreading to every agent?

### C. AI Agent (autonomous voice and chat agents)

17. Design Cresta's AI Agent: an autonomous voice agent that handles a full customer call for an enterprise, including authentication, tool calls (order lookup, refund) and handoff to a human. [Likely] [19][25]
    - Sketch: telephony, streaming ASR, turn-taking/endpointing, dialogue manager or LLM agent loop, tool gateway with policy checks, TTS, human handoff service, and the trace store.
    - Where are business rules enforced: in the prompt, in the tool gateway or in a separate checker?
    - What context goes to the human agent at handoff?
18. Cresta's latency post discusses engineering a voice agent for low end-to-end latency. Design the pipeline and its latency budget. [Likely] [19]
    - Sketch: audio in, voice activity detection, streaming ASR, endpointing, LLM (time to first token), streaming TTS (time to first byte), audio out, with a timing annotation on each arrow.
    - Which stages can overlap, for example speculative generation before the user finishes speaking?
    - How do you handle barge-in when the customer interrupts?
    - Which network choices matter (connection reuse, region placement, streaming APIs)?
19. Cresta Conductor is described as an agent that helps build AI agents. Design a system in which a business user describes a flow in natural language and receives a working, tested agent configuration. [Likely] [20][25]
    - Sketch: natural-language spec, plan/flow generator, tool and API binding, generated test scenarios, simulation runner, review UI and deploy.
    - How do you stop a generated agent from calling tools it should not have?
    - What does the human review step check?
20. Design the guardrail layer for a customer-facing AI agent: prompt injection, off-topic requests, unauthorized commitments and PII leakage. [Likely]
    - Sketch: input classifier, policy-aware tool gateway, output checker, escalation trigger and audit log.
    - How much latency can the guardrails add on a voice call?
21. How would you decide, per customer and per intent, which conversations the AI Agent should handle and which should go straight to humans? [Likely]
    - Sketch: intent classifier, containment-success predictor, routing policy, and a monitoring loop on outcomes.
    - How do you measure containment without rewarding the agent for hanging up on customers?
22. Design the shift from Agent Assist to AI Agent for one use case, so a customer can move gradually from human-assisted to fully automated conversations. [Likely]
    - Sketch: assist-mode logs, automation-candidate discovery, agent built from those logs, shadow mode, partial automation, full automation.
    - How do you use assist-mode data (what top agents say) to build the agent?
23. A voice AI agent gets an account number wrong because of an ASR error. How would you redesign the pipeline to make this less likely and to recover when it happens? [Likely]
    - What would you change in ASR biasing, confirmation turns, spelling modes and tool-side validation?
24. Walk through how Cresta's AI Agent produces one response step by step, from transcript to spoken output. Where can the decoding strategy (greedy, sampling, beam, constrained decoding) change the agent's behavior? [Reported] [4] (variant of a 2025 MLE phone-screen question for the AI Agent team)
25. Design sequence decoding with greedy and beam search as a reusable component. Then extend it to an AI agent that must emit valid tool-call JSON: when do you use constrained decoding, and when validation with a retry? [Reported] [4][28] (the decoder component is reported; the tool-call extension is [Likely])
    - Sketch: decoder interface (step function, scoring, stop rules), grammar or schema mask, validator, retry and fallback path, and a latency annotation for each.
26. AI agent tools have side effects (refunds, cancellations, address changes). Design the action layer so a retried LLM step, a dropped connection or a misheard confirmation never applies an action twice or to the wrong account. [Likely] [25]
    - Sketch: tool schema registry, intent-to-action confirmation turn, idempotency keys, a pending-action state machine, the customer system's API, and a compensation (undo) path.
    - Which actions need an explicit spoken confirmation, and who decides?
27. Cresta Conductor's examples describe agents built from defined flows with LLM steps inside them. Design a hybrid agent runtime that mixes a deterministic flow graph with free-form LLM nodes. [Likely] [25]
    - Sketch: flow definition (nodes, transitions, required slots), LLM node executor, slot store, policy checks on transitions, and a fallback to free-form handling or to a human.
    - Which parts of a flight-rebooking flow should be deterministic, and which should the LLM own?
28. A customer calls back the next day about the same issue. Design conversation memory for the AI Agent across sessions and channels (voice, chat, SMS). [Likely]
    - Sketch: customer identity resolution, interaction summaries, memory store with retention rules, retrieval at call start, and redaction.
    - What must the agent never remember, and how do you enforce that?

### D. Agent quality: evaluators, testing and the Swiss-cheese model

29. Design an evaluation platform for AI agents that detects regressions across prompt and model changes. [Reported] [5][6]
    - Sketch: scenario store, versioned agent configs (prompt, model, tools), simulator, run orchestrator, deterministic checks, LLM judges, results store, regression dashboard, CI gate.
    - How many reruns does each scenario need given non-determinism, and what statistical test calls a regression?
30. Cresta describes automated AI agent testing with simulated customers. Design the customer simulator. [Likely] [14][24]
    - Sketch: persona and goal generator mined from real conversations, simulator LLM, agent under test, conversation runner, outcome judge.
    - How do you make simulated customers behave realistically (interrupting, going off-script, being vague)?
    - How do you stop the simulator and the agent from sharing the same blind spots when both are LLMs?
31. Cresta's post on evaluators describes how it grades its agents in an LLM world. Design an evaluator service that scores every production conversation against a set of criteria. [Likely] [14]
    - Sketch: criteria registry, per-criterion judge prompts or classifiers, batch scoring jobs, calibration set with expert labels, agreement monitoring, and results surfaced to customers.
    - How do you measure and maintain agreement between judges and human experts?
    - What happens to historical scores when you upgrade the judge model?
32. Apply the Swiss-cheese model to an AI agent: which failures does each layer (pre-launch tests, human review, production monitoring) catch, and which slip through? [Likely] [23]
    - Sketch: the layers as slices, with failure types drawn as arrows that pass through or are stopped.
    - How do production failures become new pre-launch tests?
33. Design hallucination detection and grounding for agent and assist outputs. [Likely] [26]
    - Sketch: claims extraction, evidence lookup (KB, tool results, transcript), entailment check, scoring, and block-or-flag policy.
    - How do you handle a statement that is true but not supported by any available source?
34. Design a release process for prompt and model changes across hundreds of enterprise customers' agents. [Likely]
    - Sketch: change request, offline evaluation per tenant, staged rollout by tenant cohort, online metrics, automatic rollback.
    - How do you catch a regression that affects only one customer?
35. Design the expert-labeling platform used to calibrate evaluators and judges. [Likely] [14]
    - Sketch: sampling service (stratified and uncertainty-based), labeling UI with rubric, inter-annotator agreement tracking, adjudication, gold-set store, and judge-calibration reports.
    - How many expert labels per criterion do you need before you trust a judge for one customer?
36. Design automated red-teaming for a customer-facing voice agent: social engineering to get account access, prompt injection spoken by the caller, and attempts to make the agent promise refunds it cannot give. [Likely] [23][24]
    - Sketch: attack library, adversarial simulator persona, text-to-speech for spoken attacks, agent under test, policy-violation judge, and a findings tracker that feeds release gates.
    - How do you keep the red-team suite from going stale as the agent's prompts change?
37. Design a live-call quality monitor that scores an AI agent conversation while it is still running and can trigger a human takeover mid-call. [Likely] [14][23]
    - Sketch: turn-level signals (repetition, sentiment, tool failures, policy flags), streaming scorer, takeover policy with hysteresis, warm-transfer service, and outcome logging.
    - How do you set the takeover threshold so it rescues failing calls without doubling human workload?

### E. Conversation Intelligence and Insights

38. Design topic discovery over millions of unlabeled conversations: an LLM-generated taxonomy up to three levels deep, which admins can merge, split and rename. [Likely] [21]
    - Sketch: conversation summarization, embedding, clustering, LLM labeling, hierarchy builder, admin edit layer, incremental assignment of new conversations.
    - How do you keep assignments stable when admins edit the taxonomy?
39. Design outcome inference: estimate whether each conversation was resolved, and the customer's experience, without surveys or disposition codes. [Likely] [21]
    - How do you validate inferred outcomes when you have survey data for only a few percent of calls?
40. Design Cresta's "AI Analyst": answer open-ended business questions ("Why did cancellations rise last week?") over a customer's conversation corpus. [Likely] [21]
    - Sketch: question parser, query planner, structured metadata queries, per-conversation LLM extraction over a sample, aggregation, statistical checks, answer with evidence links.
    - How do you avoid presenting a weak correlation as a cause?
    - How do you bound the cost of a question that needs LLM reads of 1M conversations?
41. Design automated QA scoring for all calls against a customer-written rubric. [Likely]
    - Sketch: rubric parser, per-item scorer (rule, classifier or LLM), evidence extraction, score aggregation, dispute and override flow.
    - How do you turn a vague rubric item ("showed empathy") into a reliable judgment?
42. Design a real-time trend detector that flags emerging issues, such as an outage that shows up as a spike in a topic, before dashboards would. [Likely]
    - How do you separate a real spike from seasonality and noise?
43. Design a system that finds what top-performing agents do differently from others, and turns it into coaching guidance. [Likely]
    - Sketch: performance labels, behavior extraction from transcripts, comparison model with controls, insight ranking, coaching content generator.
    - How do you control for call mix, so you do not just learn that top agents get easier calls?
44. Design the batch processing backend that runs dozens of LLM extractions over every conversation for thousands of tenants each day. [Likely]
    - Sketch: ingestion queue, per-tenant scheduling, model router (small model or large model), batching, retries, results store, cost metering.
    - How do you prioritize jobs when GPU capacity is short?
45. Design semantic search over a customer's conversation corpus for analysts ("find calls where the customer mentioned a competitor's price"). [Likely] [21]
    - Sketch: transcript chunking by turn, hybrid lexical and dense index per tenant, metadata filters, reranker, result snippets with audio timestamps.
    - How do ASR errors on brand and product names hurt recall, and how do you compensate?

### F. Voice platform and inference graphs

46. Cresta describes its voice platform as running inference graphs of ML services for each conversation. Design that orchestration layer. [Likely] [17]
    - Sketch: a DAG with ASR, diarization, redaction, intent, sentiment, moment detection, summarization; a per-customer policy that enables nodes; an execution engine; and output sinks.
    - How do you express dependencies and skip models a customer has not enabled?
    - How do you isolate a slow node so it cannot delay real-time outputs?
47. Design the voice stack that ingests audio from many CCaaS platforms and telephony providers and produces speaker-separated streaming transcripts. [Likely] [18]
    - Sketch: connectors (SIPREC, media streams, vendor APIs), audio normalization, channel or speaker separation, streaming ASR, transcript bus.
    - How do you handle mono audio where agent and customer are mixed?
48. Design ASR adaptation for contact-center audio: noisy lines, accents and domain vocabulary such as product names. [Likely] [10]
    - Sketch: data collection, transcription/labeling, fine-tuning or contextual biasing, per-tenant vocabularies, and WER evaluation by slice.
    - When is word error rate the wrong metric for downstream tasks?
49. Design PII and PCI redaction for audio and transcripts that runs before storage and training. [Likely]
    - Sketch: streaming entity detection, audio muting aligned to timestamps, transcript masking, audit sampling.
    - What does a missed credit card number cost, and how does that set your recall target?
50. One recording can contain IVR prompts, hold music, a transfer and two different human agents. Design segmentation that labels these spans so downstream models see only the relevant parts. [Likely] [18]
    - Sketch: audio event detection, speaker change detection, role classifier (customer, agent, IVR, bot), transfer detection, and a span map attached to the transcript.
51. Design the evaluation and selection of text-to-speech for a voice agent: naturalness, correct pronunciation of brand names and numbers, and time to first audio byte. [Likely] [12][19]
    - Sketch: test-utterance set by slice, automatic metrics, listener ratings, pronunciation lexicon per tenant, latency harness, and A/B testing on live calls.
52. Your streaming ASR vendor or a GPU pool degrades in the middle of peak traffic. Design graceful degradation for the real-time inference graph. [Likely] [17]
    - Sketch: health checks per node, secondary ASR path, priority tiers that shed non-critical nodes first, circuit breakers, and backfill of skipped nodes after the call.
    - What does the agent's screen show while some features are degraded?

### G. Ocean models, fine-tuning and serving

53. Ocean-1 was announced as a contact-center foundation model. Design the training pipeline for a domain-adapted LLM like it. [Likely] [7]
    - Sketch: data sources (licensed and consented conversations, synthetic instructions), cleaning and PII removal, continued pretraining or SFT, preference tuning, evaluation suite, release.
    - What data do you need to train instruction-following for contact-center tasks?
    - How do you show it beats a general frontier model on quality, latency and cost?
54. Design the system that adapts a shared base model to each customer: a per-customer fine-tune, a LoRA adapter, prompts, or retrieval. [Likely] [7][16][22]
    - Sketch: customer data intake, eligibility and consent checks, adapter training jobs, evaluation gate, adapter registry, deployment.
    - When is a per-customer adapter worth its cost compared with a better prompt?
55. Fireworks describes serving many LoRA adapters on one base-model deployment. Design multi-tenant serving for thousands of per-customer adapters. [Likely] [22]
    - Sketch: gateway with tenant auth, router, adapter cache (GPU, CPU, storage), mixed-adapter batching, autoscaler, per-tenant metering.
    - What happens to latency when a cold adapter is requested?
    - How does quantization of the base model interact with adapters?
56. Design the model-routing layer that chooses between a small fine-tuned model and a large frontier model per request. [Likely]
    - Sketch: request features, router (rules or learned), model pool, quality feedback, cost dashboard.
    - How do you train the router without labels for which model would have done better?
57. Design the LLM serving stack for Cresta's real-time features with a p99 latency target. [Likely]
    - What do continuous batching, KV-cache reuse, quantization and speculative decoding each trade off?
    - How do you autoscale GPUs for call-center traffic that peaks in business hours across time zones?
58. Design the data governance pipeline that decides which customer conversations can be used for training shared models and which can only be used for that customer. [Likely]
    - Sketch: consent and contract metadata, data tagging, access-controlled training sets, lineage tracking, deletion propagation.
    - What happens when a customer asks you to delete their data after it has been used for training?
59. Design the benchmark suite that compares Ocean models, open-weight models and frontier APIs on contact-center tasks each time a new model is released. [Likely] [7]
    - Sketch: task registry (summarization, extraction, reply suggestion, tool use), held-out tenant data with consent, metrics and judges per task, latency and cost harness, and a leaderboard with slice views.
    - How do you prevent benchmark data from leaking into the training data of the models you evaluate?
60. How would you turn agents' edits to suggested replies into preference pairs for DPO or similar preference tuning, and what can go wrong? [Likely] [15]
    - Sketch: suggestion and edit log, pair builder (suggested versus sent text), filters for trivial edits, per-tenant weighting, training job, and an evaluation gate.
    - How do you avoid learning one agent's personal style as if it were a preference?

### H. Founders' research background and how it shapes the product

These questions draw on public background that is not sourced in this guide. Cresta was founded by Zayd Enam and Tim Shi, both from Stanford AI research; Tim Shi's earlier work includes research on web-based agents (World of Bits) during a period at OpenAI. Verify this before you mention it. The current CEO, Ping Wu, has spoken publicly about how contact centers misunderstand the value of LLMs [8].

61. Early research on web-based agents trained agents to complete tasks in browser environments. How would you apply that idea to an AI agent that operates a contact center's internal tools (CRM, billing) rather than calling APIs? [Likely]
    - Sketch: environment (UI or API), observation encoding, action space, policy (LLM), reward or success check, sandbox.
    - When is UI automation better than an API, and what does it cost in reliability?
62. Cresta's CEO has argued that many contact centers misunderstand where LLMs create value. Where would you put LLMs, and where would you keep smaller models or rules, in a contact-center stack? Draw the stack and defend each choice. [Likely] [8]

### I. Cross-cutting design prompts an interviewer may use

63. How would you deploy a model in production? [Reported] [1]
    - Sketch: model registry, CI evaluation gate, shadow deployment, canary, A/B test, full rollout, monitoring and rollback.
    - What do you watch in the first 24 hours, and what triggers a rollback?
64. Design monitoring for LLM features across hundreds of enterprise customers when you have almost no labels. [Likely]
    - Sketch: per-tenant traffic metrics, output-distribution drift, judge-sampled quality, user-feedback signals, alert routing.
    - Which label-free signals catch quality regressions early?
65. Design the onboarding pipeline for a new enterprise customer: from first data access to Agent Assist live in production. [Likely]
    - Sketch: data connectors, historical conversation backfill, ASR and redaction, model adaptation, evaluation with customer stakeholders, launch.
    - What is the minimum data needed before launch?
66. Design a system that lets a customer's admin define a new metric in natural language (for example, "Did the agent offer the upgrade?") and get it computed over past and future conversations. [Likely]
    - Sketch: definition UI, LLM prompt compiler, sample-based validation with admin feedback, backfill job, streaming scorer.
    - How do you show the admin the metric's accuracy before they rely on it?
67. Design the trace and observability system for multi-step LLM agents, so engineers can debug a failed conversation. [Likely]
    - Sketch: span model (turn, LLM call, tool call, retrieval), trace store, search, replay with a modified prompt.
68. Design a cost model and budget controls for LLM usage per tenant. [Likely]
    - How do you prevent one tenant's batch backfill from starving real-time traffic?
69. Design an A/B testing framework for LLM prompt changes where outcomes (CSAT, resolution) arrive days later. [Likely]
    - Which proxy metrics do you use, and how do you validate them against delayed outcomes?
70. How would you design support for many languages in Agent Assist and AI Agent? [Likely]
    - Sketch: language identification, multilingual ASR, translation or native multilingual LLM, per-language evaluation.
71. Design a system that detects compliance violations (required disclosures not read, prohibited statements) in real time and after the call. [Likely]
    - How do you trade recall against precision when a miss means regulatory risk?
72. Design the data flywheel for the whole company: how do conversations flowing through assist, AI agents and analytics improve each product? [Likely]
    - Sketch: shared conversation store, labeling sources (agent actions, judge scores, outcomes), training pipelines per product, evaluation gates.
73. Explain the key concepts of LLMs, GenAI, RAG and AI agents, then draw how they fit together inside one Cresta product of your choice. [Reported] [2][3] (a recruiter told a candidate to expect these concepts; framing them around a Cresta product is [Likely])
74. A European bank requires its audio, transcripts and model calls to stay in the EU. Design multi-region deployment for the real-time and batch ML stack under data-residency rules. [Likely]
    - Sketch: regional data planes, a global control plane holding only configuration, region-pinned model serving and adapters, regional evaluation jobs, and cross-region telemetry with no conversation content.
    - Which shared models can you still improve with EU data, and how?

### J. Product-sense questions that lead into design

75. Which Cresta product would you work on first, and what is its hardest ML problem? [Likely]
    - How would you sketch the current system and the change you would make?
76. What are the main ML differences between assisting a human agent and replacing one? [Likely]
    - How does each change the latency, accuracy and evaluation requirements?
77. What would you need to believe to build an in-house foundation model rather than fine-tune open models or call frontier APIs? [Likely] [7]
78. If a competitor's AI agent claims 80% containment, how would you check that claim, and how would you measure your own? [Likely]
79. What makes contact-center conversations harder for ML than generic chat data? [Likely]
    - How do ASR errors, long calls, multiple parties and hold music affect your pipeline design?
80. How would you prioritize between improving ASR accuracy and improving the LLM layer, given a fixed team? [Likely]
81. Which Cresta engineering blog post did you read, and what would you do differently in the design it describes? [Likely] [17][18][19]
82. How should a product like Cresta's explain its suggestions or scores to agents and supervisors, and what does that require from the system design? [Likely]
83. Where do you see the biggest risk of customer-visible failure in Cresta's AI Agent, and how would you design against it? [Likely]

### K. Track-specific prompts (by team in the job postings)

84. Agentic Assist track: design an assist agent that can take actions for the human agent (fill forms, look up orders) rather than only suggest text. [Likely] [13]
    - How do you confirm actions with the human before they run?
85. Agentic Assist track: how would you combine pre-LLM ML (classifiers, ranking) with LLMs in one assist pipeline? [Likely] [13]
86. Agent and System Quality track: design the metric set and dashboards used to decide whether an AI agent is ready to launch for a customer. [Likely] [9][11]
    - Which metrics block a launch, and which are only tracked?
87. Agent and System Quality track: how would you diagnose context drift and multi-step reasoning failures in long conversations? [Likely] [11]
88. Voice/ASR track: design the evaluation of a new streaming ASR model against the current one, including latency and downstream task impact. [Likely] [10][12]
    - Sketch: test-set curation by slice, offline WER and latency, downstream replay through assist models, shadow traffic, rollout.
89. Voice Experience track: design natural turn-taking for a voice agent: endpointing, backchannels and interruption handling. [Likely] [12][19]
    - How do you evaluate turn-taking quality offline?
90. Insights track: design a pipeline that keeps conversation-level LLM extractions consistent when the underlying model is upgraded. [Likely] [21]
    - How do you backfill history, or version metrics, so customer dashboards do not jump?
91. Voice Experience track: design measurement of perceived response latency for live voice-agent calls, using the recordings rather than server logs. [Likely] [12][19]
    - Sketch: dual-channel recording, voice activity detection on both sides, gap computation per turn, attribution of each gap to pipeline stages using trace timestamps, and percentile dashboards per tenant.
92. Agentic Assist track: design an "autopilot" mode in chat that drafts a full reply and sends it after a short timeout unless the human agent edits or cancels it. [Likely] [13]
    - Sketch: draft generator, confidence and risk scorer, per-intent eligibility policy, countdown UI, send service, and audit log.
    - Which intents should never be eligible, and how do you decide that from data?
93. Agent and System Quality track: design automatic root-cause clustering of failed AI agent conversations, so engineers see a ranked failure taxonomy instead of single transcripts. [Likely] [11]
    - Sketch: failure detection, trace featurization (step, tool, error type), LLM failure summaries, clustering, ranking by volume and severity, and links to replay.

### L. Questions you can ask the interviewer about Cresta's systems

94. How is the inference graph configured per customer today, and what is the hardest part of operating it? [Likely] [17]
95. How much of the real-time stack runs on self-hosted fine-tuned models and how much on third-party APIs? [Likely] [16][22]
96. How do evaluators and simulated testing gate AI agent releases today? [Likely] [14][24]
97. What is the current state of the Ocean model family, and how does it relate to the per-customer adapters? [Likely] [7]
98. Which parts of Conductor and AI agent building are owned by ML engineers, and which by product engineers? [Likely] [25]

## Deepening pass (October 2026): newly reported questions

> Five web searches were run for this pass: English and Chinese ("面经") queries, plus queries scoped to PracHub, Glassdoor/Blind/1point3acres and Reddit/LeetCode. None turned up a new Cresta ML system design or product-context question for the Senior MLE onsite. Search snippets confirm that PracHub has a Cresta machine learning category page, but the snippets did not show its question text, so nothing was added from it. One new candidate report from an adjacent round was found and is listed below with that round named. No existing [Likely] question was supported strongly enough by a snippet to be retagged.

### Newly reported (adjacent round)

99. Implement a client that calls the Wikipedia API, then discuss current AI trends. Extend it for this chapter: how would that external-API client become a tool an AI Agent calls during a live conversation? [Reported] [29] (reported in a Software Engineer technical phone screen; the AI Agent tool extension is [Likely])
    - How do you handle timeouts, rate limits and pagination when the caller is a real-time voice agent?
    - Which current AI trends do you think matter most for Cresta's products, and how would they change one of the designs in this chapter?

### Gap-closing questions for the senior level

100. An AI Agent prompt change ships and, within an hour, the agent starts promising refunds that policy does not allow for one large customer. Design the kill switch and incident response: how do you limit the blast radius, roll back for one tenant, and find the affected conversations? [Likely] [23] (extends question 34)
     - Sketch: per-tenant feature flags and config pins, policy-violation alerting, rollback service, affected-conversation query over traces, and customer notification workflow.
     - What should the system do automatically, and what needs a human decision?
101. Customers edit their AI Agent prompts and flows, and Cresta also upgrades the base prompts and models underneath. Design config versioning so a platform upgrade does not overwrite or break a customer's customizations. [Likely] [25]
     - Sketch: layered config (platform base, vertical template, tenant overrides), version graph, three-way merge, per-tenant regression run, and a staged opt-in.
     - How do you tell a customer that their override is now the cause of a regression?
102. Should Cresta's voice agent use a cascaded pipeline (ASR, LLM, TTS) or an end-to-end speech-to-speech model? Draw both and compare latency, controllability, tool use, evaluation and compliance logging. [Likely] [19]
     - What would you need to see in an offline evaluation before switching one customer to speech-to-speech?
103. Cresta wins its first customer in a new vertical (for example, healthcare scheduling) with no historical conversations it may use. How would you bootstrap Agent Assist or an AI Agent for them? [Likely]
     - Sketch: synthetic conversation generation from SOPs and KB, cross-vertical transfer, simulated testing, a human-reviewed pilot, and a data collection plan.
     - How do you keep synthetic data from teaching patterns that real customers never show?
104. You must upgrade the embedding model behind Knowledge Assist and conversation search for every tenant. Design the re-indexing and cutover so retrieval quality does not drop and the live product has no downtime. [Likely] [21]
     - Sketch: dual-write or shadow index per tenant, backfill scheduler with cost limits, offline retrieval evaluation per tenant, index alias swap, and rollback.
     - Which tenants do you migrate first, and why?
105. A support call lasts 75 minutes. Design how Insights and summarization handle conversations longer than the model's useful context: long-context models, hierarchical summarization or turn-level extraction. [Likely] [21]
     - How do you evaluate whether facts from the middle of the call are lost?
106. LLM judges show position bias, length bias and a preference for outputs from their own model family. Design an evaluator pipeline that measures and reduces these biases when judges gate AI Agent releases. [Likely] [14]
     - Sketch: swapped-order pairwise runs, length-controlled comparisons, cross-family judge panel, expert calibration set, and bias dashboards.
107. Callers to a bank's voice agent may use synthetic or cloned voices to pass authentication. Design fraud and spoofing detection inside the voice agent's authentication step. [Likely] [12]
     - Sketch: audio anti-spoofing model, behavioral and metadata signals, risk score, step-up verification (one-time code), and escalation to a human.
     - How do you set the false-positive rate when every false alarm adds friction for a real customer?
108. A third-party frontier model that several AI Agent customers depend on is announced for deprecation in 90 days. Plan the migration across all affected tenants. [Likely]
     - Sketch: dependency inventory per tenant, candidate replacement models, per-tenant regression and simulation runs, prompt adaptation, staged cutover, and a fallback.
     - How do you handle a tenant whose agent regresses on the replacement and cannot be fixed in time?
109. As the senior engineer, you disagree with a product manager who wants to launch an AI Agent feature for a customer before its evaluation metrics meet the agreed launch bar. How would you use data and the system design to resolve the disagreement? [Likely] [9][11]
     - What would you offer instead: a narrower scope, shadow mode, or human-in-the-loop launch?

## References

1. Glassdoor, "Cresta (CA) Interview Questions" (Senior Machine Learning Engineer review: recruiter call, LeetCode-medium, vibe-coding and system design rounds; "How would you deploy a model in production?"), 2025-2026. https://www.glassdoor.co.in/Interview/Cresta-CA-Interview-Questions-E3180909.htm
2. Blind, "Cresta interview process", 2024-2025. https://www.teamblind.com/post/cresta-interview-process-jmfnzjoa
3. Blind, "Anyone interview with Cresta lately?", 2025. https://www.teamblind.com/post/anyone-interview-with-cresta-lately-zvj0uhq2
4. 1point3acres, "Cresta Machine Learning Engineer Tech Phone Interview Experience" (MLE, AI Agent direction; greedy and beam-search decoding, how an agent generates a sequence), 2025. https://www.1point3acres.com/interview/thread/1140481
5. PracHub, "Cresta ML System Design Interview Questions" ("Design an Evaluation Platform for AI Agents Across Prompt and Model Changes"), 2026. https://prachub.com/companies/cresta/categories/ml-system-design
6. PracHub, "Cresta Software Engineer Interview Experience: System Design Round: Build an Eval Platform for AI Agents". https://prachub.com/interview-experiences/cresta-software-engineer-interview-experience-system-design-round-build-an-eval-platform-for-ai-agents
7. Cresta Blog, "Introducing Ocean-1: World's First Contact Center Foundation Model", 2023. https://cresta.com/blog/introducing-ocean-1-worlds-first-contact-center-foundation-model
8. CX Today, "Cresta CEO Ping Wu: Many Contact Centers Misunderstand the Value of LLMs". https://www.cxtoday.com/contact-center/cresta-ceo-ping-wu-many-contact-centers-misunderstand-the-value-of-llms/
9. Cresta on Greenhouse, "Senior Machine Learning Engineer". https://job-boards.greenhouse.io/cresta/jobs/4249943008
10. Cresta on Greenhouse, "Senior Machine Learning Engineer (ASR)". https://job-boards.greenhouse.io/cresta/jobs/5101011008
11. Cresta on Greenhouse, "Staff Machine Learning Engineer". https://job-boards.greenhouse.io/cresta/jobs/4741283008
12. Cresta on Greenhouse, "Senior Machine Learning Engineer, Voice Experience". https://job-boards.greenhouse.io/cresta/jobs/5155675008
13. Built In Vancouver, "Senior Machine Learning Engineer, Cresta" (Agentic Assist track). https://builtinvancouver.org/job/senior-machine-learning-engineer/8619914
14. Cresta Blog, "How Cresta Grades Its Agents: Evaluators in an LLM World". https://cresta.com/blog/how-cresta-grades-its-agents-evaluators-in-an-llm-world
15. Cresta Blog, "How Fine-Tuned LLMs Power Knowledge Assist, Summarization and Chat Suggestions", 2024. https://cresta.com/blog/how-fine-tuned-llms-power-knowledge-assist-summarization-and-chat-suggestions
16. Fireworks AI Blog, "Story: Cresta Knowledge Assist", 2024. https://www.fireworks.ai/blog/story-cresta-knowledge-assist
17. Cresta Blog, "Understanding Cresta's Voice Platform: ML Services, Inference Graphs and Real-Time Intelligence", 2025. https://cresta.com/blog/understanding-crestas-voice-platform-ml-services-inference-graphs-and-real-time-intelligence
18. Cresta Blog, "Understanding Cresta's Voice Platform: The Voice Stack", 2025. https://cresta.com/blog/understanding-crestas-voice-platform-the-voice-stack
19. Cresta Blog, "Engineering for Real-Time Voice Agent Latency". https://cresta.com/blog/engineering-for-real-time-voice-agent-latency
20. Cresta Blog, "AI-Ready Knowledge for Contact Centers: Closing the Gap Between KBs and AI", 2025. https://cresta.com/blog/ai-ready-knowledge-for-contact-centers-closing-the-gap-between-kbs-and-ai
21. Cresta Blog, "Introducing the Next Generation of Cresta Insights: A Complete System for Enterprises to Understand Their CX", 2026. https://cresta.com/blog/introducing-the-next-generation-of-cresta-insights-a-complete-system-for-enterprises-to-understand-their-cx
22. Fireworks AI Blog, "Multi-LoRA", 2024. https://fireworks.ai/blog/multi-lora
23. Cresta Blog, "Why AI Agent Evaluations Fail, and How the Swiss-Cheese Model Prevails". https://cresta.com/blog/why-ai-agent-evaluations-fail----and-how-the-swiss-cheese-model-prevails
24. Cresta Blog, "Cresta Launches Automated AI Agent Testing: Confidence in Every Conversation". https://cresta.com/blog/cresta-launches-automated-ai-agent-testing-confidence-in-every-conversation
25. Cresta Blog, "Cresta Conductor: The Agent for AI Agent Development". https://cresta.com/blog/cresta-conductor-the-agent-for-ai-agent-development
26. Cresta Blog, "Grounding Reality: How Cresta Tackles LLM Hallucinations in Enterprise AI". https://cresta.com/blog/grounding-reality---how-cresta-tackles-llm-hallucinations-in-enterprise-ai
27. Glassdoor, "Cresta (CA) Interview Experience and Questions" (Forward Deployed Engineer technical screen: "What is RAG, and how do you prevent hallucination?"), 2026. https://www.glassdoor.com/Interview/Cresta-CA-Interview-Questions-E3180909.htm
28. PracHub, "Cresta Interview Questions" (company page; "Design sequence decoding with greedy and beam search"), 2025. https://prachub.com/companies/cresta
29. 1point3acres, "Cresta Software Engineer Tech Phone Screen Overview" (full-time SWE phone screen: implementing Wikipedia API calls and discussing AI trends; candidate moved to the virtual onsite). https://www.1point3acres.com/interview/thread/1173094

*Changed in review:* No cited page could be opened (proxy refusals and DNS failures for every domain), and no web search was possible, so citations were checked against the search-result excerpts already reviewed in chapters 01-05. Removed as uncited or unsupported: the second Glassdoor regional mirror (glassdoor.ca), the Blind "Cresta senior backend SWE interview" thread (its excerpt supports only an online assessment), the Blind AMA thread (no excerpt in this guide supports it), the 1point3acres company page (uncited here) and the Scoutify guide (a generic template, not a candidate report). The 1point3acres thread title was corrected. Two references were added: the Glassdoor FDE report and the PracHub company page. Both [Reported] tags in the original chapter (agent evaluation platform; deploying a model in production) match reviewed excerpts and were kept. Four [Reported] variants from adjacent Cresta rounds were added, each naming its round. Two questions were removed as duplicates: "extract top agents' behaviors and generalize them" (same as the top-performer coaching design) and "human-in-the-loop assist-then-automate" (same as the assist-to-AI-Agent transition design). All other added questions are [Likely].
