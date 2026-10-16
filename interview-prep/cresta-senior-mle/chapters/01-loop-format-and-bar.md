# 01. Cresta Senior MLE Interview Loop: Format, Rounds, Culture and the Senior Bar

> **How to read this chapter.** This chapter puts the most weight on the ML system design round, as requested. It also covers the rounds around it: screens, coding, AI-assisted coding, deep dive and behavioral.
>
> **Tagging system.** Every question carries one of two tags:
> - **[Reported] [n]**: a candidate report or question bank attributes this question, or this round type, to Cresta. The number points to the References list. Where the report is for another role (SWE, FDE), the tag says so.
> - **[Likely]**: an inference from Cresta's job postings, products and engineering blog. No candidate has reported it. A citation on a [Likely] item points to the product or posting the question is based on, not to a candidate report.
>
> **Research limits.** None of the cited pages could be opened during research or during this review. The egress proxy refused connections to Glassdoor, Blind, 1point3acres, PracHub, Greenhouse, CoderPad and cresta.com, and the web-search budget for this review run was used up. Every [Reported] item therefore rests on search-result excerpts gathered while this bank was being researched (this chapter and chapters 02-07), not on reading the full pages. One reference that no excerpt supported has been replaced (see the end of the References). Confirm the loop with your recruiter.

## Format notes

### What candidates and sources report

- **Senior MLE loop (Glassdoor, 2025-26):** the stages were a recruiter call, a LeetCode-medium technical round, a **"vibe coding" round using an LLM tool**, and a **systems design round**. The sample design prompt was "How would you deploy a model in production?" [1]
- **Live coding setting:** reviewers describe live rounds of about an hour in Python on CoderPad over Zoom, with a software engineer, camera on, and conceptual AI questions mixed in [1].
- **Phone-screen pace:** a 2026 Senior SWE report describes three LeetCode-medium questions in 70 minutes. The reviewer felt the interviewer wanted a specific approach, not just any working solution [31].
- **Senior/Staff SWE loop (Blind, 2024):** one hour of coding, one hour of system design and a hiring-manager chat [3].
- **Mid-level SWE loop (Blind):** one technical screen, three coding interviews and three behavioral rounds, with a mix of LeetCode-style and object-oriented questions. Another candidate's recruiter email said the interview would also cover "key concepts and fundamentals around LLMs, GenAI, RAG, and AI Agents" [3][4].
- **MLE phone screen, AI Agent direction (1point3acres, 2025):** greedy and beam-search decoding over a next-token probability dictionary [28][10].
- **Async first step:** Cresta adopted an asynchronous CoderPad technical assessment as the first engineering filter and runs its technical hiring on CoderPad [11]. A Cresta engineering posting describes a self-paced CoderPad assessment after resume review [26]. Blind posters mention an online assessment (OA) before the recruiter screen [5][6]. One SWE online challenge mixed data structures and algorithms with SQL joins [32].
- **AI-allowed practical round:** a Forward Deployed Engineer (FDE) candidate described a 1.5-hour practical round with AI tools allowed [5]. This matches the "vibe coding" round in the senior MLE report [1].
- **System design with an agentic or eval flavor:** PracHub lists a Cresta ML system design question, "Design an Evaluation Platform for AI Agents Across Prompt and Model Changes" [8][9].
- **Other Cresta engineering loops:** reports include a project presentation round [2], a customer-delivery round plus a leadership round for a Senior FDE [2], an OOP pool/orchestrator coding question with round-robin scheduling [1], and a job dispatcher/executor phone screen (SWE, 2026) [29].
- **Timeline:** about 25 days on average across roles, according to Glassdoor [2].
- **Policy:** Cresta postings say candidates may not record interviews or use AI transcription during them, and that recruiting email comes only from @cresta.ai [12].

### Round-by-round summary

| Stage | Duration | Interviewer | Status | Senior bar assessed |
|---|---|---|---|---|
| Online assessment (CoderPad, async) | Timed | Automated | Reported [11][26][5] | Clean, correct code under time pressure |
| Recruiter screen | ~30 min | Recruiter | Reported [1] | Motivation, team match (Agentic Assist, Agent and System Quality, Voice/ASR, Insights) [12][16] |
| Technical round (LC-medium, CoderPad + Zoom) | ~60-70 min | Engineer | Reported [1][31] | DS&A fluency; for ML, decoding or metric implementations [10][28] |
| AI-assisted ("vibe coding") practical | 60-90 min | Engineer | Reported [1][5] | Directs the LLM tool well, verifies its output, ships working end-to-end code |
| ML system design | ~60 min | Senior or staff MLE | Reported [1][8][9] | End-to-end LLM/ML system: requirements, eval, serving latency, rollout, monitoring |
| Hiring manager / behavioral | 30-60 min | EM | Reported [3] | Ownership, pace, customer orientation, mentoring |
| Project presentation / deep dive | ~60 min | Panel | Reported for SWE [2]; Likely for MLE | Depth on one shipped system; measured impact |

### What the senior bar looks like (from postings and blog)

- **Strong ML foundations from before LLMs, combined with deep LLM and prompting expertise.** The Agentic Assist track names both as requirements, along with "a proven ability to move research into production-grade systems" [16].
- **Evaluation of non-deterministic systems.** Postings name offline benchmarking, online experimentation and LLM-as-a-judge, plus specific failure modes to diagnose: hallucinations, retrieval errors, tool misuse, context drift, prompt brittleness and multi-step reasoning breakdowns [14][12].
- **Quality metrics.** Postings list accuracy, faithfulness, task completion, latency, cost and robustness [14].
- **Scale and production constraints.** Postings ask candidates to design for scalability, latency, cost efficiency, security and maintainability. The Voice posting covers real-time inference and streaming pipelines [15].
- **Culture.** Third-party pages cite these operating principles: "Mission is the boss", "Do too much", "Have the courage to be disliked" and "patient in strategy, impatient in execution". Reviewers describe a high technical bar, ownership, little politics and a fast pace [25][4].

## Questions

### A. ML system design: LLM agents and evaluation (highest priority)

1. Design an evaluation platform for AI agents that detects regressions across prompt changes and model changes. Its users are the teams that build agents. [Reported] [8][9]
   - How do you version prompts, models, tools and test sets together so that a run is reproducible?
   - Which checks are deterministic, and which use an LLM judge? How do you calibrate the judge against human experts?
   - How do you score the agent's reasoning and tool-use path separately from whether the final outcome was correct?
   - How do you handle non-determinism? How many reruns per scenario, and what statistical test decides that something is a regression?
   - How do you gate a release in CI? What is the blocking threshold?
2. How would you deploy a model in production? [Reported] [1]
   - What does your rollout look like: shadow, canary, A/B, then full?
   - What do you monitor in the first 24 hours, and what triggers a rollback?
   - How do you serve it within a real-time latency budget?
   - How do you handle model versioning and per-customer configuration?
3. Design a customer simulator that generates realistic synthetic callers to test an AI agent before launch. [Likely] [17][27]
   - How do you give simulated callers objectives, personalities, account data and deviations from the standard path?
   - How do you mine real conversations so that the test set covers what production actually sees?
   - How do you stop the simulator and the agent from sharing the same blind spots?
4. Design a layered ("Swiss-cheese") reliability strategy for a production AI agent: pre-launch tests, human review and production monitoring. [Likely] [18]
   - Which failure modes does each layer catch, and which does it miss?
   - How do you turn production failures back into regression tests?
5. Design an LLM-as-a-judge system for scoring millions of conversations, with calibration to expert labels. [Likely]
   - How do you measure agreement between the judge and humans, and track it over time?
   - How do you detect judge drift when you upgrade the judge model?
   - What does the judge cost per conversation, and how do you cut it?
6. Design an AI agent that fully handles customer voice or chat for an enterprise, including tool calls (order lookup, refunds) and escalation to a human. [Likely]
   - How do you constrain tool use so the agent cannot make unauthorized commitments?
   - How do you detect and recover from tool misuse or context drift in multi-step flows?
   - How do you design the handoff to a human agent, including the context you pass along?
7. Design a multi-agent orchestration layer, in which a router hands off to specialized sub-agents. [Likely]
   - When does a multi-agent design beat a single agent with many tools?
   - How do you trace and debug a failure that spans several agents?
8. Design a guardrail system against prompt injection and policy violations in a customer-facing agent. [Likely]
   - What does it add to latency, and where in the pipeline does it run?
9. Design an "agent for building agents": a copilot that turns a business owner's natural-language description of a workflow (for example, flight rebooking) into an agent configuration, then tests it. [Likely] [34]
   - How do you validate the generated configuration before it touches real customers?
   - How do you keep a human in the loop without making the copilot pointless?
10. Design a grounding checker that flags agent responses unsupported by the knowledge base or by tool outputs, before or right after they are sent. [Likely] [35]
    - Claim extraction plus entailment, or a single judge call? What does each cost in latency?
    - What do you do when the checker fires in the middle of a live voice turn?

### B. ML system design: real-time and voice

11. Design real-time Agent Assist: listen to a live call and surface next-best-action or knowledge suggestions to the human agent. [Likely] [21]
    - What is the end-to-end latency budget from audio to suggestion, and how is it split across ASR, retrieval and LLM?
    - How do you handle partial versus final transcripts? (Cresta pushes both about once per second [21].)
    - How do you decide when to show a suggestion and when to stay silent?
    - How do you measure the effect on agent outcomes, separating it from agent skill?
12. Design a voice AI agent pipeline that responds in under a second: telephony, streaming ASR, LLM and TTS. [Likely] [19]
    - Which per-component metrics do you track (LLM first-token latency, TTS first-byte latency), and how do you measure end-to-end latency?
    - How do you detect end-of-turn and handle barge-in?
    - Which networking choices matter (connection reuse, streaming APIs, keeping DNS off the critical path)?
13. Design an inference graph that runs several models (ASR, redaction, intent, sentiment, summarization) per conversation, with customer-specific policies. [Likely] [20]
    - How do you express dependencies and skip models a customer has not enabled?
    - How do you isolate a slow model so it cannot block real-time outputs?
14. Design an ASR evaluation and quality-monitoring program for contact-center audio across accents, noise and domain vocabulary. [Likely] [13]
    - How does WER differ from downstream task quality, and which do you optimize?
    - How do you get ground-truth transcripts cheaply?
    - How do you detect ASR regressions for a single customer?
15. Design per-customer ASR adaptation for product names, account IDs and jargon (contextual biasing, custom vocabularies or fine-tuning). [Likely] [13]
    - How do you stop biasing toward one customer's terms from raising errors on ordinary speech?
    - How do you roll out an adapted model to one tenant safely?
16. Design a real-time sentiment or escalation-risk detector for live calls. [Likely]
    - How do you combine acoustic and lexical signals?
    - How do you set alert thresholds per customer?

### C. ML system design: RAG, knowledge and analytics

17. Design Knowledge Assist: RAG over a customer's knowledge base during live conversations. [Likely] [23][37]
    - How do you reformulate ambiguous conversational queries before retrieval?
    - Would you fine-tune embeddings for the domain, and how would you mine hard negatives?
    - How do you adapt answers by channel (voice, chat, email)?
    - How do you handle stale or contradictory documents?
18. Design topic discovery over millions of unlabeled conversations: an LLM-built taxonomy with up to three levels, which admins can merge, split and rename. [Likely] [24]
    - How do you keep cluster assignments stable when the taxonomy is edited?
    - How do you overlay outcomes (resolution, sales) to rank problem areas?
19. Design a system that infers conversation outcomes (resolved or not, customer experience) without surveys or disposition codes. [Likely] [24]
    - How do you validate inferred outcomes without ground truth?
20. Design a research agent ("AI Analyst") that answers open-ended business questions over a customer's conversation corpus. [Likely] [24]
    - How do you combine per-conversation LLM extraction with aggregate statistics?
    - How do you stop the agent from presenting a weak correlation as a finding?
21. Design automated QA scoring for 100% of calls against a customer-defined rubric. [Likely]
    - How do you turn a free-text rubric item into a reliable judgment?
    - How do you stop agents from gaming the scorer?
22. Design a real-time trend detector that flags emerging issues before dashboards would show them. [Likely]
    - How do you separate a true spike from seasonality or noise?
23. Design post-call summarization that writes into each customer's CRM format, using fine-tuned models rather than a frontier API. [Likely] [36]
    - How do you evaluate summaries when every customer wants different fields?
    - When does fine-tuning beat prompting a larger model for this task?

### D. ML system design: models, serving, experiments and cost

24. Design the domain adaptation of a base LLM for contact centers, in the spirit of Cresta's Ocean-1: data, SFT, preference tuning and serving. [Likely] [22]
    - Self-instruct and RLHF/DPO: what data do you need, and how do you avoid overfitting to one customer?
    - How do you prove it beats a frontier API on quality, latency and cost?
25. Design a migration from a frontier LLM API to a smaller self-hosted model for one high-volume feature. [Likely]
    - Which quality bar must be met, and how is it measured?
    - Distillation or fine-tuning: which, and on what data?
26. Design multi-tenant serving: one base model with per-customer adapters or prompts. [Likely] [38]
    - How do you serve hundreds of LoRA adapters efficiently?
    - How do you prevent data leakage across tenants?
27. Design an LLM serving stack for real-time features with p99 latency targets. [Likely]
    - What do batching, KV cache, quantization and speculative decoding each trade off?
    - How do you autoscale when CPU-based autoscaling (HPA) is meaningless for GPUs?
28. Design monitoring for production LLM features across hundreds of enterprise customers. [Likely]
    - Which label-free signals catch quality regressions?
    - How do you catch a regression that affects only one customer?
29. Design the feedback flywheel: agent accept, edit and ignore signals on suggestions feed model improvement. [Likely]
    - What biases does this implicit feedback introduce, and how do you correct them?
30. Design a PII redaction pipeline for audio and transcripts that runs before storage and training. [Likely]
    - How do you trade recall against precision, and what does a miss cost?
31. Design an online experiment for a new Agent Assist model when a handful of large customers dominate traffic and agents work in teams. [Likely]
    - Randomize by conversation, by agent or by team? What interference do you expect?
    - Which outcome metric moves fast enough to decide within two weeks?
32. Design the training-data pipeline for customer-specific models under contractual data limits: opt-outs, retention windows and deletion requests. [Likely]
    - How do you prove a deleted customer's data no longer influences a deployed model?

### E. Coding rounds (online assessment, technical screen and onsite)

33. Implement greedy decoding given a Python dictionary that maps each token to its candidate next tokens and their probabilities, starting from a start token and stopping at an end token or a maximum length. Then extend it to beam search with beam width k. [Reported] [10][28]
    - How do you add length normalization?
    - How would you add top-k / top-p sampling and temperature?
    - What is the time and space complexity in beam width and vocabulary size?
34. Design the decoder as a reusable component: what interface lets greedy and beam search share code? [Reported] [30]
34. Walk through how an AI agent generates a sequence step by step. Where can the decoding strategy change the agent's behavior? [Reported] [28]
35. Solve LeetCode-medium problems in the technical round, possibly three in 70 minutes. [Reported] [1][31]
    - Can you state the complexity and test edge cases before running the code?
    - If the interviewer is steering toward a specific approach, how do you find out early?
36. Implement a worker pool and an orchestrator that assigns tasks round-robin (reported for the FDE AI Agent role); a SWE variant asks for a job dispatcher that hands jobs to a pool of executors. [Reported] [1][29]
    - How do you handle worker failures and retries?
    - How do you make it thread-safe?
37. Implement a rate limiter in your language of choice. [Reported] [1]
    - Extend it to per-tenant limits for shared LLM API calls. Token bucket or sliding window, and why?
38. Call a public API (Wikipedia was used in a reported SWE phone screen), then process and return the results. [Reported] [7]
    - How do you handle pagination, rate limits and errors?
40. Implement autocomplete: given a list of words and a prefix, return the matching suggestions (reported for SWE). [Reported] [32]
    - How do you rank suggestions by frequency, and what changes at a million words?
41. Convert a flat list of records with parent references into a tree (reported for SWE). [Reported] [32]
42. Solve a grid or matrix problem framed as a puzzle game in a timed online challenge (reported for SWE). [Reported] [32]
43. Write SQL with joins in the online challenge, for example: from `conversations` and `annotations` tables, find agents whose suggestion-acceptance rate dropped week over week. [Reported for SQL joins; the example prompt is Likely] [32]
44. Solve object-oriented design problems mixed with LeetCode-style problems (mid-level SWE loop). [Reported] [3]
45. Implement word error rate (edit distance) between a reference transcript and a hypothesis. [Likely]
46. Given a stream of utterances with speaker labels and timestamps, merge them into turns and emit partial results in real time. [Likely]
47. Implement precision, recall and F1 from scratch for multi-label intent classification, plus micro and macro averages. [Likely]
48. Implement cosine-similarity top-k retrieval over an embedding matrix in NumPy, without loops. [Likely]
    - Which ANN index would you choose at 100M vectors?
49. Implement a dynamic batching queue that maximizes GPU throughput under a max-latency constraint. [Likely]
50. Implement a streaming chunker that splits a live transcript into token-budgeted segments without cutting sentences. [Likely]
51. Count the top-k most frequent intents over a sliding time window on an unbounded stream, with bounded memory. [Likely]

### F. AI-assisted ("vibe coding") practical round

52. Build a small working feature end to end with an LLM coding tool, within the time limit. [Reported] [1][5]
    - How do you break the task down before prompting the tool?
    - How do you verify generated code? Which tests do you write yourself?
    - When do you stop prompting and write the code by hand?
53. Build a minimal RAG or agent prototype (for example, an agent that answers from a document set or calls a mock API). [Likely]
    - How do you evaluate it in the time available?
54. Build a tool-calling loop for a flight-rebooking agent against mock flight and booking APIs, including a confirmation step before any change. [Likely] [34]
55. Build a small evaluation harness: run an LLM over a labeled set of transcripts, enforce a JSON output schema, and report accuracy with confidence intervals. [Likely]
56. Debug an LLM-generated pipeline that passes its tests but gives wrong results. [Likely]

### G. LLM / GenAI fundamentals (flagged by recruiters)

57. Explain the key concepts and fundamentals of LLMs, GenAI, RAG and AI agents. [Reported] [3][4]
58. What is RAG, and how do you prevent hallucination? [Reported for an FDE technical screen] [33]
    - In what ways does RAG still fail?
    - How do you measure faithfulness or groundedness automatically?
59. What is an AI agent, as opposed to a chain or a workflow? When would you not use an agent? [Likely]
60. How does function calling / tool use work, and how do you test multi-turn agent trajectories? [Likely]
61. What drives LLM inference latency and cost, and which levers reduce them? [Likely]
62. How do you handle conversations longer than the context window? [Likely]
63. Compare SFT, DPO and RLHF. When is each worth its cost? [Likely]
64. How do you make prompts robust across model versions, and how do you regression-test them? [Likely]
65. Which ML fundamentals from before LLMs still matter here (calibration, class imbalance, active learning, experiment design)? [Likely] [16]
    - Why does calibration matter for a confidence-gated suggestion?
66. How do ASR errors propagate into downstream NLP, and how do you make classifiers robust to them? [Likely]
67. How do you evaluate a generative model when there is no single correct answer? [Likely]
    - How do you design a rubric for human raters and measure agreement between them?
68. QA labels from different customers' reviewers disagree systematically. How do you detect and model annotator bias before training on the labels? [Likely]

### H. Project deep dive / presentation

69. Present a significant project you built end to end. [Reported for SWE] [2]
    - What was the baseline, and how did you prove the improvement?
    - What did you personally own?
70. Walk through an LLM or ML system you shipped to production. What broke first? [Likely]
71. Tell me about a model that worked offline but failed online. How did you find the gap? [Likely]
72. Describe the evaluation you built for your project. What did your metrics miss? [Likely]
73. How did you handle latency, cost or scaling constraints? What did a request cost, and how did you bring it down? [Likely]
74. Describe a decision where you chose a simpler model over a more sophisticated one. [Likely]
75. Tell me about a time you took a research idea or paper into a production system. What had to change on the way? [Likely] [16]
76. If we gave you this problem in the contact-center domain, what would change? [Likely]

### I. Hiring manager, behavioral and culture

77. Walk through your background, focusing on ML you shipped. Why Cresta and contact-center AI? [Likely]
78. Which track appeals to you: Agentic Assist, Agent and System Quality, Voice/ASR, or Insights? [Likely] [12]
    - Placement depends on your strengths and on business needs. Where would you have the most impact?
79. Tell me about a time you "did too much": went well beyond scope to get something shipped. [Likely] [25]
80. Tell me about a time you had the courage to be disliked: pushing back on a PM, a customer or a leader with data. [Likely] [25]
81. Tell me about a time you were patient on strategy but impatient on execution. [Likely] [25]
82. Describe working directly with customers or forward-deployed teams to define what an ML feature should do. [Likely]
    - How did you turn a vague ask into measurable requirements?
83. Describe a disagreement over quality versus shipping speed. How was it resolved? [Likely]
84. Tell me about mentoring engineers or raising your team's technical bar. [Likely]
85. Tell me about changing another team's roadmap or technical direction without having authority over them. [Likely]
86. Tell me about the hardest production incident you were involved in, and what the postmortem changed. [Likely]
87. How do you decide which new LLM research is worth adopting? [Likely]
88. If evaluation for a key model turned out to be weak, what would you do in your first 90 days? [Likely]
89. When would you build a capability in-house rather than use a third-party model provider? [Likely]
90. How would you explain a model regression to an enterprise customer? [Likely]

### J. Questions to ask your interviewers

91. Which team would I join, and what are its goals for the next two quarters? [Likely]
92. How does the team gate LLM changes before they reach customers? [Likely]
93. What is the split between self-hosted models and third-party APIs, and where is it heading? [Likely]
94. How much direct customer contact do MLEs have? [Likely]
95. How is the AI-assisted coding round scored: on the final product, or on how I direct and check the tool? [Likely]
96. What separates Senior from Staff MLE here? [Likely]

## Deepening pass (October 2026): newly reported questions

> **Result of this pass.** Five targeted web searches (English and Chinese, covering Glassdoor, Blind, 1point3acres and PracHub) found no new Cresta question that a candidate reported, beyond those already in this chapter. The only new Cresta-specific excerpt is that PracHub tags its Cresta questions for both Machine Learning Engineer and Software Engineer roles [39]. That supports the existing PracHub-sourced items (questions 1 and 33-34), but the excerpt does not name any new prompt. A third-party prep guide also describes an ML deep-dive or paper-discussion stage. It reads as a generic template, so this pass does not cite it as a Cresta report. The questions below are therefore all [Likely]. They cover gaps for the senior level that sections A-J leave open.

97. Write pseudocode for a ReAct-style agent loop that alternates reasoning and tool calls for a customer-service task, with a step limit and a fallback to a human. [Likely]
    - How do you detect that the loop is stuck or repeating the same tool call?
    - Where would you add structured-output validation for tool arguments?
98. A new enterprise customer signs and wants a working intent or outcome model in two weeks, with almost no labels. How do you cold-start it? [Likely]
    - Zero-shot LLM labels, active learning, or transfer from other tenants' models: which, in what order?
    - How do you decide when the model is good enough to switch on in production?
99. Your third-party LLM provider has an outage or a latency spike during peak call hours. Design the fallback path for real-time features. [Likely]
    - Which features degrade gracefully, and which must fail closed?
    - How do you test the fallback before you need it?
100. Design per-customer cost controls for LLM features: token budgets, model routing by difficulty, and caching. [Likely]
     - How do you route a request to a cheaper model without hurting quality on hard cases?
     - What cost per conversation would make a feature unprofitable, and how would you show that to leadership?
101. Design the expansion of an English-first voice agent and Agent Assist stack to Spanish and to code-switched calls. [Likely]
     - How do you evaluate quality in a language that no one on the team speaks?
     - Where do ASR errors and translation errors compound?
102. Design end-to-end evaluation for a voice agent at the audio level: simulated callers with speech, noise and interruptions, not just text transcripts. [Likely]
     - Which failures appear only with audio (barge-in, end-of-turn errors, misheard account numbers)?
     - How do you score latency and naturalness alongside task success?
103. Design versioning and staged rollout for per-tenant agent configurations (prompts, tools, knowledge sources), so that one customer's change can never break another's. [Likely]
     - How do you diff two configuration versions in terms of behavior, not just text?
104. Design memory for an AI agent that serves the same customer across several calls and channels. [Likely]
     - What do you store, for how long, and how do you respect deletion requests?
     - How do you stop stale memory from overriding the facts in the current conversation?
105. You are asked to review another engineer's design doc for a new LLM feature. What do you look for first, and how do you push back on a weak evaluation plan? [Likely]
106. What would you do differently from today's practice if you had to cut the latency of a real-time suggestion feature in half within one quarter? [Likely]
     - How would you find the biggest contributor before changing anything?

## References

1. Glassdoor, "Cresta (CA) Interview Questions" (Senior Machine Learning Engineer loop with vibe-coding and system design rounds; FDE reviews; CoderPad and Zoom Python round; rate limiter), 2024-2026. https://www.glassdoor.co.in/Interview/Cresta-CA-Interview-Questions-E3180909.htm
2. Glassdoor, "Cresta (CA) Interview Questions" (25-day average; SWE project presentation; Senior FDE loop), 2025-2026. https://www.glassdoor.ca/Interview/Cresta-CA-Interview-Questions-E3180909.htm
3. Blind, "Cresta interview process", 2024-2025. https://www.teamblind.com/post/cresta-interview-process-jmfnzjoa
4. Blind, "Anyone interview with Cresta lately?", 2025. https://www.teamblind.com/post/anyone-interview-with-cresta-lately-zvj0uhq2
5. Blind, "Interview for forward deployment engineer at Cresta", 2025. https://www.teamblind.com/post/interview-for-forward-deployment-engineer-at-cresta-ngswbj44
6. Blind, "Cresta senior backend SWE interview", 2025. https://www.teamblind.com/post/cresta-senior-backend-swe-interview-6v1tmjld
7. 1point3acres, "cresta interview questions" company page (SWE/MLE phone screen and online test reports, including the API phone screen), 2024-2026. https://www.1point3acres.com/interview/company/cresta
8. PracHub, "Cresta ML System Design Interview Questions" ("Design an Evaluation Platform for AI Agents Across Prompt and Model Changes"), 2026. https://prachub.com/companies/cresta/categories/ml-system-design
9. PracHub, "Cresta Software Engineer Interview Experience: System Design Round: Build an Eval Platform for AI Agents". https://prachub.com/interview-experiences/cresta-software-engineer-interview-experience-system-design-round-build-an-eval-platform-for-ai-agents
10. PracHub, "Cresta Machine Learning Interview Questions" ("Implement greedy and beam decoding"). https://prachub.com/companies/cresta/categories/machine-learning
11. CoderPad, "How Cresta consolidated all technical hiring on CoderPad to scale eng hiring" (case study). https://coderpad.io/case-study/how-cresta-consolidated-all-technical-hiring-on-coderpad-to-scale-eng-hiring/
12. Cresta on Greenhouse, "Senior Machine Learning Engineer". https://job-boards.greenhouse.io/cresta/jobs/4249943008
13. Cresta on Greenhouse, "Senior Machine Learning Engineer (ASR)". https://job-boards.greenhouse.io/cresta/jobs/5101011008
14. Cresta on Greenhouse, "Staff Machine Learning Engineer". https://job-boards.greenhouse.io/cresta/jobs/4741283008
15. Cresta on Greenhouse, "Senior Machine Learning Engineer, Voice Experience". https://job-boards.greenhouse.io/cresta/jobs/5155675008
16. Built In Vancouver, "Senior Machine Learning Engineer, Cresta" (Agentic Assist track). https://builtinvancouver.org/job/senior-machine-learning-engineer/8619914
17. Cresta Blog, "How Cresta Grades Its Agents: Evaluators in an LLM World". https://cresta.com/blog/how-cresta-grades-its-agents-evaluators-in-an-llm-world
18. Cresta Blog, "Why AI Agent Evaluations Fail, and How the Swiss-Cheese Model Prevails". https://cresta.com/blog/why-ai-agent-evaluations-fail----and-how-the-swiss-cheese-model-prevails
19. Cresta Blog, "Engineering for Real-Time Voice Agent Latency". https://cresta.com/blog/engineering-for-real-time-voice-agent-latency
20. Cresta Blog, "Understanding Cresta's Voice Platform: ML Services, Inference Graphs and Real-Time Intelligence", 2025. https://cresta.com/blog/understanding-crestas-voice-platform-ml-services-inference-graphs-and-real-time-intelligence
21. Cresta Blog, "Understanding Cresta's Voice Platform: The Voice Stack", 2025. https://cresta.com/blog/understanding-crestas-voice-platform-the-voice-stack
22. Cresta Blog, "Introducing Ocean-1: First Contact Center Foundation Model", 2023. https://cresta.com/blog/introducing-ocean-1-worlds-first-contact-center-foundation-model
23. Cresta Blog, "AI-Ready Knowledge for Contact Centers: Closing the Gap Between KBs and AI", 2025. https://cresta.com/blog/ai-ready-knowledge-for-contact-centers-closing-the-gap-between-kbs-and-ai
24. Cresta Blog, "Introducing the Next Generation of Cresta Insights", 2026. https://cresta.com/blog/introducing-the-next-generation-of-cresta-insights-a-complete-system-for-enterprises-to-understand-their-cx
25. JobsByCulture, "Cresta: Frontend Engineer" (culture and operating-principles summary; third-party). https://jobsbyculture.com/jobs-at/cresta/frontend-engineer
26. a16z Jobs, "Cresta: Senior Product Engineer" (posting describing the self-paced CoderPad assessment). https://jobs.a16z.com/jobs/cresta/5288477008--senior-product-engineer
27. Cresta Blog, "Cresta Launches Automated AI Agent Testing: Confidence in Every Conversation". https://cresta.com/blog/cresta-launches-automated-ai-agent-testing-confidence-in-every-conversation
28. 1point3acres, "Cresta Machine Learning Engineer Tech Phone Interview Experience" (MLE, AI Agent direction; greedy and beam-search decoding), 2025. https://www.1point3acres.com/interview/thread/1140481
29. 1point3acres, "Cresta Fulltime Software Engineer Tech Phone Screen: Job Dispatcher-Executor System Implementation" (title seen in a listing; full text not read), 2026. https://www.1point3acres.com/interview/thread/1172433
30. PracHub, "Cresta Interview Questions" (company page; "Design sequence decoding with greedy and beam search"), 2025. https://prachub.com/companies/cresta
31. Glassdoor, "Cresta (CA) Senior Software Engineer Interview Questions" (three LeetCode-medium questions in 70 minutes), 2026. https://www.glassdoor.ca/Interview/Cresta-CA-Senior-Software-Engineer-Interview-Questions-EI_IE3180909.0,9_KO10,34.htm
32. Glassdoor, "Cresta (CA) Software Engineer Interview Questions" (autocomplete, list-to-tree, grid puzzle and SQL-joins online challenge), 2024-2026. https://www.glassdoor.sg/Interview/Cresta-CA-Software-Engineer-Interview-Questions-EI_IE3180909.0,9_KO10,27.htm
33. Glassdoor, "Cresta (CA) Interview Experience and Questions" (FDE technical screen: RAG and hallucination), 2026. https://www.glassdoor.com/Interview/Cresta-CA-Interview-Questions-E3180909.htm
34. Cresta Blog, "Cresta Conductor: The Agent for AI Agent Development". https://cresta.com/blog/cresta-conductor-the-agent-for-ai-agent-development
35. Cresta Blog, "Grounding Reality: How Cresta Tackles LLM Hallucinations in Enterprise AI". https://cresta.com/blog/grounding-reality---how-cresta-tackles-llm-hallucinations-in-enterprise-ai
36. Cresta Blog, "How Fine-Tuned LLMs Power Knowledge Assist, Summarization and Chat Suggestions", 2024. https://cresta.com/blog/how-fine-tuned-llms-power-knowledge-assist-summarization-and-chat-suggestions
37. Fireworks AI Blog, "Story: Cresta Knowledge Assist", 2024. https://www.fireworks.ai/blog/story-cresta-knowledge-assist
38. Fireworks AI Blog, "Multi-LoRA", 2024. https://fireworks.ai/blog/multi-lora
39. PracHub, "Cresta Machine Learning Engineer Interview Questions" (position page; the excerpt says its Cresta questions are tagged for Machine Learning Engineer and Software Engineer roles), 2026. https://prachub.com/companies/cresta/positions/machine-learning-engineer

*Removed in review:* the 1point3acres thread 1173094 (no search excerpt in this bank supported it; replaced by the company page, ref 7); the Blind company-posts page and the Scoutify guide (neither was cited, and Scoutify is a generic template).
