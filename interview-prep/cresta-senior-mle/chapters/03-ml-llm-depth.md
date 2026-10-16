# 03. ML/LLM Depth for Contact-Center AI: NLP, Speech, Fine-Tuning, Evaluation, RAG, Agents and Latency

## Format notes for this round

- **Where depth gets tested.** A Glassdoor review of a Senior ML Engineer loop lists four steps: a recruiter call, a LeetCode-medium coding round, a round done with an LLM coding tool ("vibe coding"), and a system design round [1]. No source reports a standalone "ML deep dive" round for this role. Expect the depth questions below inside the phone screen, the system design round and the hiring-manager conversation.
- **Duration and interviewer.** The reported MLE phone screen (AI Agent direction) was a live Python decoding problem, with follow-ups on how an agent generates sequences [6][4][5]. Interviewers are probably ML engineers from the team you would join. This is an inference from the postings for the Voice Experience, ASR, Agentic Assist and Staff tracks [22][23][24][25][26].
- **Topics recruiters flag in advance.** A recruiter email to a candidate said the interviews would cover "key concepts and fundamentals around LLMs, GenAI, RAG, and AI Agents" [3].
- **Senior-level bar, as inferred from Cresta's postings.**
  - Debug hallucinations, retrieval errors, tool misuse, context drift, prompt brittleness and multi-step reasoning failures [22][23].
  - Evaluate with offline benchmarks, online experiments and LLM-as-a-judge [22][26].
  - Design and deploy enterprise RAG [22].
  - Make trade-offs across the real-time voice stack: ASR, turn detection, LLM and TTS [13][24].
- **Products and tech to anchor your answers.**
  - Ocean-1, Cresta's contact-center foundation model [11].
  - Fine-tuned models that power Knowledge Assist, summarization and chat suggestions [8][9].
  - Thousands of per-customer LoRA adapters served on one base-model cluster at Fireworks [9][10].
  - The voice stack and inference graphs [14][15].
  - LLM-judge evaluators, automated agent testing and the "Swiss-cheese" evaluation model [16][17][19].
  - Hallucination grounding [18], Cresta Conductor [20] and Insights [21].
- **Research caveat.** During this review, live page fetches failed for every cited domain (DNS and proxy errors) and the web-search budget was used up, so no URL could be re-opened. Each reference below was cross-checked against the reviewed reference lists in chapters 01, 04, 05 and 07 of this bank. Unverifiable or generic sources were removed (see the note at the end of the References). Only questions a candidate source attributes to Cresta are tagged [Reported]. Everything else is [Likely], inferred from Cresta's published engineering work and postings. Confirm the round content with your recruiter.

## Decoding and LLM fundamentals

1. Implement greedy decoding and beam search over a next-token probability table given as a Python dictionary (a directed token graph). [Reported] [4][5][6]
   - How do you handle sequences that hit an end token at different lengths? Would you apply length normalization?
   - What is the time and memory complexity in terms of beam width, vocabulary size and depth?
   - When does beam search do worse than greedy decoding for open-ended dialogue?
2. Walk through how an AI agent generates a sequence step by step. Where can decoding strategy change the agent's behavior? [Reported] [6]
   - How would you add constraints, such as forcing a valid tool-call JSON schema?
3. Explain the key concepts and fundamentals of LLMs, GenAI, RAG and AI agents. Then go one level deeper: what in the transformer architecture makes in-context learning and tool use possible? [Reported] [3]
   - Where do pretraining, instruction tuning and preference tuning each change model behavior?
4. Compare temperature, top-k, top-p and min-p sampling. Which would you use for agent-assist suggestions, and which for summaries? [Likely]
   - How do you make outputs reproducible for regression tests?
5. Explain speculative decoding. When would a draft model trained on contact-center data pay off? [Likely] [10]
   - What acceptance rate makes it worthwhile, and how does batch size change that?
6. Explain the KV cache and how its memory grows. How do paged attention and prefix caching apply to long call transcripts? [Likely]
   - Estimate the KV-cache memory for a 7B-class model with 8k context and 200 concurrent sessions. Which knobs (GQA, FP8 cache, context truncation) would you turn first?
7. How do rotary position embeddings work, and how would you extend a model's context window for hour-long calls? What quality risks come with context extension? [Likely]
8. Tokenizers often split account numbers, SKUs, phone numbers and dates badly. How does this hurt extraction and generation, and how would you mitigate it? [Likely] [12]

## NLP for conversations

9. What is RAG, and how do you prevent hallucination? [Reported] [2] (reported for a Forward Deployed Engineer technical screen)
   - In what ways does RAG still fail?
   - How would you detect a hallucination that got past your safeguards?
10. How would you build intent or topic discovery over millions of unlabeled call transcripts? [Likely] [21]
    - How do you pick the number of clusters and keep them stable across retrains?
    - LLM labeling of every conversation is expensive at Insights scale. Where would you use embeddings and clustering, and where an LLM?
11. Design a classifier for conversation "moments", such as a missed upsell or a compliance phrase not said. Would you choose a fine-tuned encoder or a prompted LLM? [Likely] [8]
    - How do you handle severe class imbalance and labels that change by customer?
12. You need training labels for a new per-customer classifier and have only a few hundred human labels. How would you use LLM-generated (weak) labels, and how do you measure and correct their noise? [Likely] [8]
13. A model's confidence threshold works for one customer and over-fires for another. How do you calibrate scores per customer, and how do you set thresholds when you have few positives? [Likely]
14. How would you predict escalation, churn risk or negative sentiment early in a live conversation, from partial transcripts? [Likely] [14]
    - How do you trade earliness against precision, and how do you evaluate a prediction made at minute 3 of a 20-minute call?
15. How do transcription errors from ASR propagate into NLP models trained on clean text? How would you make the downstream models robust to them? [Likely] [15]
16. How would you do PII detection and redaction on streaming transcripts with low latency? [Likely]
    - What precision/recall trade-off would you choose, and why?
    - How do you align redaction spans back to audio timestamps, so the recording is redacted too?
17. How do you model speaker roles and turn structure in a two-party transcript for summarization or extraction? [Likely]
18. How would you handle multilingual calls, and calls that switch language mid-conversation? [Likely]
19. You fine-tuned a model on one customer's data and it looks great offline. How do you detect overfitting, leakage across time or agents, and memorization of that customer's PII? [Likely]

## Speech and ASR

20. Explain CTC, RNN-T and attention encoder-decoder ASR. Which suits streaming contact-center audio, and why? [Likely] [25]
21. Streaming ASR returns partial transcripts every ~0.5 to 1.5 seconds and revises them. How should downstream models consume these unstable partials? [Likely] [15]
    - When do you trigger an LLM on partial text versus final text?
    - How do you upsert or reconcile results that were computed on revised partials?
22. How do you measure ASR quality beyond WER for downstream tasks such as entity capture (account numbers, names, SKUs)? [Likely] [25]
    - How would you build a domain-specific test set, and how often would you refresh it?
23. Spoken digits, dates, currency and spelled-out emails must become structured text. How would you design and evaluate inverse text normalization and entity formatting for ASR output? [Likely] [25]
24. How would you adapt an ASR model to customer-specific vocabulary: contextual biasing, fine-tuning, or LM rescoring? [Likely] [25]
25. You must choose between third-party ASR vendors and an in-house model (for example, a fine-tuned Whisper-class model). How would you run the bake-off? [Likely] [15][25]
    - How do you catch failure modes WER hides, such as hallucinated text during silence or hold music?
26. Design end-of-turn (endpointing) detection for a voice AI agent. Compare silence-based VAD, acoustic models and semantic models. [Likely] [13][24]
    - How do you measure false barge-ins versus added latency?
    - How would you distill a large turn-detection model into one small enough for real time?
27. Handling 8kHz telephony audio, noise and cross-talk: what augmentation and front-end choices matter? [Likely]
28. Speaker diarization on stereo versus mono call recordings: how do the approaches differ, and how do you evaluate them? [Likely]
29. How would you evaluate and choose a TTS system for a voice agent? Cover naturalness, pronunciation of names and alphanumerics, latency to first audio, and streaming behavior. [Likely] [13][24]
30. What changes when you use speech-to-speech models instead of a cascaded ASR, LLM and TTS stack? [Likely] [24]
    - How do you evaluate quality and controllability in each design?
31. The ASR WebSocket drops in the middle of a call. How would you replay or recover audio so the models don't lose context? [Likely] [15]

## LLM fine-tuning and adaptation

32. Cresta fine-tuned its own models to match GPT-4 quality on Knowledge Assist at lower cost and latency. How would you set up that project end to end? [Likely] [8][9]
    - How do you decide a fine-tuned model is "as good as" the frontier model?
33. Ocean-1 is described as a contact-center foundation model. When would you do continued pretraining on domain conversations rather than only instruction tuning? How would you choose the data mix? [Likely] [11]
34. Compare full fine-tuning, LoRA, QLoRA and prompt tuning for per-customer adaptation. [Likely] [9][10]
    - How do you choose LoRA rank and target modules?
35. You serve thousands of per-customer LoRA adapters on one base model. What are the training, versioning and serving implications? [Likely] [10]
    - How do you handle a customer with too little data for its own adapter?
    - What breaks if you upgrade the base model?
    - How would you automatically evaluate and gate thousands of adapters before each release? [17]
36. How do you build SFT data from contact-center transcripts? Cover label sources, agent-quality filtering and deduplication. [Likely] [8]
37. When would you train on synthetic conversations generated by a stronger LLM? How do you keep synthetic data diverse, and how do you detect that it is degrading the model? [Likely] [17][20]
38. When would you use preference optimization (DPO, RLHF, or RLAIF) for agent-assist suggestions? Where would the preference signal come from? [Likely]
    - Can you use agent accept/reject clicks as rewards? What biases do they carry?
39. How do you prevent catastrophic forgetting of instruction following when you fine-tune on narrow domain data? [Likely] [11]
40. Walk through the GPU memory math for fine-tuning a 7B and a 70B model. When do you need FSDP or ZeRO, activation checkpointing or mixed precision? [Likely] [23]
41. How would you fine-tune an embedding model for contact-center terminology, product SKUs and policy language? [Likely] [12]
    - How do you mine hard negatives?
42. When would you call a frontier API rather than run a self-hosted fine-tuned model? Discuss cost, latency, data residency and quality. [Likely] [27]

## Distillation and model compression

43. Explain knowledge distillation for LLMs: logit-level versus sequence-level (training on teacher outputs). [Likely]
    - Which would you use when the teacher is a closed API?
44. Design a pipeline that distills a large teacher into a small model for real-time summaries or suggestions. [Likely] [8]
    - How do you choose teacher prompts and filter teacher outputs?
    - How do you know when the student is good enough to ship?
45. An LLM judge is too slow and expensive to score every production conversation. How would you distill it into a small classifier for online monitoring, and how do you keep the two aligned over time? [Likely] [16]
46. Compare quantization methods (INT8, FP8, 4-bit GPTQ/AWQ) in terms of quality loss and throughput. How does quantization interact with LoRA adapters? [Likely] [10]
47. When would you prune or use a smaller architecture rather than distill? [Likely]
48. How do you detect regressions in rare but high-stakes behaviors (such as compliance disclosures) after compression? [Likely]

## Retrieval-augmented generation

49. Design a RAG system for Knowledge Assist that answers agent questions mid-call from a customer's knowledge base. [Likely] [8][12][22]
    - How do you reformulate the query from a noisy, partial conversation?
    - How do you choose chunk size and structure for knowledge-base articles, FAQs and policy PDFs?
50. Knowledge Assist can surface answers before the agent asks. How do you decide when to trigger retrieval proactively, and how do you keep false triggers from distracting agents? [Likely] [9]
51. Compare hybrid retrieval (BM25 plus dense) with dense only for domain jargon and SKUs. [Likely] [12]
52. Where would you add a reranker (cross-encoder or LLM-based), and how do you fit it into a real-time latency budget? [Likely] [12][13]
53. How do you evaluate retrieval separately from generation? Which metrics, and where do labels come from? [Likely]
    - How do you detect knowledge-base gaps, where the answer simply doesn't exist?
54. Knowledge bases are often stale or contradictory. How do you make them "AI-ready", and how do you handle conflicting sources at answer time? [Likely] [12]
55. How do you keep the index fresh when articles change daily? What does upgrading the embedding model cost, and how do you re-index without downtime? [Likely] [12]
56. How do you enforce grounding and citation? When should the model refuse or fall back? [Likely] [18]
57. How do you adapt answer length and format by channel (voice, chat, email)? [Likely] [12]
58. Diagnose this: RAG answers look fluent but agents report they are wrong 15% of the time. Walk through your debugging. [Likely] [22]
    - How do you tell retrieval failures from generation failures?
59. When would you use long-context stuffing instead of retrieval? [Likely]
60. How do you handle multi-tenant isolation in a vector index, so no customer can retrieve another's documents? [Likely]

## Agents and tool use

61. Explain function calling end to end. How do you make tool-call arguments reliable? [Likely] [22]
62. Tool calls fail, time out or are retried. How do you design tool schemas and error handling so the agent recovers, and so non-idempotent actions (refunds, cancellations) never run twice? [Likely] [23]
63. Design a customer-facing AI agent that authenticates a caller, looks up an order and processes a refund. [Likely]
    - Where do deterministic guardrails sit, and where does the LLM decide?
    - How do you prevent a refund from running without verified identity?
64. A customer's standard operating procedure has many branches. Would you encode it as an explicit workflow graph, a single long prompt, or a router over sub-agents? Which fails least in a regulated setting? [Likely] [26]
65. What are the most common agent failure modes in production? How would you detect each one (tool misuse, context drift, looping, multi-step reasoning breakdown)? [Likely] [22][23]
66. Explain AI agent frameworks and planning patterns (ReAct, plan-and-execute, state machines). Which fits a regulated contact center, and why? [Likely] [22]
67. How do you manage memory and context across a 30-minute call, including summarization of earlier turns? [Likely]
68. When do you hand off from the AI agent to a human, and how do you design and evaluate that escalation policy? [Likely]
69. How do you defend a voice or chat agent against prompt injection from customers? [Likely]
70. Cresta Conductor is described as an agent that helps build AI agents. How would you evaluate an LLM that writes or edits another agent's prompts and configuration? [Likely] [20]
    - What stops it from shipping an edit that passes tests but breaks real conversations?
71. Agentic assist (agent-facing) versus AI agent (customer-facing): how do the risk tolerance, latency budget and evaluation differ? [Likely] [22][24][26]

## Evaluation

72. Design an evaluation framework for an AI agent: trajectory evaluation versus final-outcome evaluation. [Likely] [16]
    - What would you check deterministically, and what would you hand to an LLM judge?
73. Explain a layered ("Swiss-cheese") approach to agent evaluation. Which layers would you build first, and how do you know which failures slip through every layer? [Likely] [19]
74. How do you build and calibrate LLM-as-a-judge evaluators against human experts? [Likely] [16][17][18]
    - How do you measure judge-human agreement, and what threshold is acceptable?
    - How do you detect drift in a judge over time, or when you swap its model?
    - How do you control position, verbosity and self-preference bias?
75. A prompt change moves your eval score from 82% to 84% on 150 test conversations, and the agent is non-deterministic. Is that a real improvement? How would you decide, statistically, whether to ship? [Likely] [7]
76. Design a customer simulator to test AI agents before launch. How do you make simulated customers realistic and diverse? [Likely] [16][17]
    - Why run each simulated conversation several times? How do you report variance?
77. How do you measure the hallucination rate, and how do you use it to choose models and prompts? [Likely] [18]
78. How would you evaluate LLM-extracted structured fields (contact reason, outcome, sentiment) across millions of conversations, when there is no ground truth for most of them? [Likely] [21]
79. Offline metrics improved but online agent adoption fell. What could explain this, and how would you investigate? [Likely] [22]
80. How would you design an online A/B test for agent-assist suggestions in a contact center? Consider randomization unit, interference between agents, and business metrics such as AHT and CSAT. [Likely]
81. How do you evaluate summarization quality at scale: reference-based metrics, LLM judges, or human ratings? [Likely] [8]
82. How do you evaluate a voice agent end to end, including latency, interruptions and task success? [Likely] [13][24][25]
83. How do you build regression suites that stop a prompt or model change from breaking a customer's agent? [Likely] [7][17]
84. How would you monitor data drift and model quality in production for per-customer models? [Likely]

## Latency optimization and serving

85. How would you deploy a model in production? [Reported] [1]
    - Go deeper for an LLM: which serving engine, precision and batching settings would you choose, and how would you canary and roll back a new model?
86. A voice agent must respond within about one second of the caller finishing. Break down the latency budget across telephony, ASR, endpointing, LLM, TTS and network. [Likely] [13]
    - Which components would you measure individually, and how would you run end-to-end tests?
87. How do you cut time-to-first-token and time-to-first-audio? Discuss streaming LLM output into TTS, filler phrases and prefetching. [Likely] [13]
88. How would you speculatively start LLM generation before end-of-turn is confirmed? What happens if the user keeps talking? [Likely] [13][24]
89. Explain continuous batching, and the trade-off between throughput and tail latency (p99) for real-time assist. [Likely]
90. Long transcripts make prefill expensive. Explain chunked prefill and prefill/decode disaggregation. When would each help a real-time assist workload? [Likely]
91. Design an inference graph that runs several models (ASR, intent, moment detection, suggestion LLM) per utterance. How do you orchestrate them under customer-specific policies? [Likely] [14]
    - How do you avoid redundant calls and handle partial failures?
92. Multi-LoRA serving: how do adapter swapping and batching across adapters affect latency? What limits remain, such as quantization compatibility? [Likely] [10]
93. Estimate the GPUs needed to serve real-time suggestions for 10,000 concurrent calls. State your assumptions about utterance rate, tokens per request and throughput per GPU. [Likely] [14]
94. How would you cut cost per conversation with model routing or cascades (a small model first, escalating to a larger one)? How do you choose the escalation signal? [Likely] [8][27]
95. How would you autoscale GPU inference for contact-center traffic with strong daily peaks? [Likely]
96. When would you cache LLM outputs or retrieval results in real-time assist, and what invalidates them? [Likely]

## ML system design prompts (depth-heavy)

97. Design real-time agent assist: streaming audio in, suggestions and knowledge cards out. Draw the full pipeline. [Likely] [14][15]
    - Where does state live per call, and how do you scale to tens of thousands of concurrent calls?
98. Design automatic post-call summarization and disposition for every call, with quality guarantees. [Likely] [8]
99. Design a conversation analytics or "AI analyst" system that answers natural-language questions over millions of conversations. [Likely] [18][21]
    - How do you keep aggregate statistics accurate when an LLM produces them?
100. Design per-customer model customization (adapters, prompts, policies) with safe rollout and rollback. [Likely] [10][14]
101. Design an evaluation platform for AI agents that catches regressions across prompt and model changes. Go deep on the metrics, judges and statistical gates. [Reported] [7][28]
102. Design a feedback loop that turns agent accept/reject actions and supervisor reviews into training data. [Likely] [17]

## Research, papers and ML foundations

103. Discuss a recent paper on LLM evaluation, distillation or speech that you found useful. What are its limitations for contact-center data? [Likely]
104. What are "strong pre-LLM ML foundations" good for in an LLM-first team? Give an example where a classical model beat an LLM. [Likely] [22]
105. Explain backpropagation through attention. Why does layer normalization placement matter for training stability? [Likely]
106. How do you explain to a customer why an AI suggestion or score was produced? [Likely]
107. How do you handle data privacy, retention and consent when you train on customer conversations? [Likely] [27]

## Deepening pass (October 2026): newly reported questions

Search notes: five targeted searches (Glassdoor/Blind/PracHub/1point3acres/interviewing.io, English and Chinese phrasing) found no new MLE-specific onsite report. Two adjacent Cresta reports surfaced, both from Software Engineer loops, so treat them as signals of shared rounds rather than confirmed MLE content. Q101 now also cites the September 2026 PracHub onsite report [28].

108. In the "evaluation platform for AI agents" design round, the users are teams building agents that analyze contact-center conversations. Beyond task accuracy, how would the platform measure tool usage and response speed when a system prompt or model changes? [Reported] [28] (Software Engineer onsite, system design round)
     - How would you show a team, per change, which conversations regressed and why?
     - How do you compare latency and tool-call counts fairly across models with different output lengths?
109. Which AI trends are you following right now, and which do you think matter most for contact-center products? [Reported] [29] (Software Engineer tech phone screen, alongside a coding task)
     - Which recent trend would you not bet on for Cresta, and why?

### Gap-closing questions for the senior level

110. A fine-tuning run's loss spikes and then diverges midway through training. How would you debug it (data batch, learning rate schedule, precision, gradient norms), and how would you decide whether to restart from a checkpoint? [Likely] [23]
111. Your eval set is built from past conversations, and the base model may have seen similar public data. How do you detect test-set contamination or leakage between training transcripts and evaluation transcripts? [Likely] [7]
112. How would you design constrained or structured decoding (JSON schema, grammar) for extraction at scale? What does it cost in latency and quality compared with free generation plus validation and retry? [Likely] [22]
113. How would you estimate uncertainty for an LLM-generated answer or extracted field (self-consistency, token log-probs, a verifier model), so the product can abstain or route to a human? [Likely] [18]
114. Labeling budget is limited. How would you use active learning to choose which conversations human reviewers label next for a per-customer model or an LLM judge? [Likely] [16][17]
115. Compliance rules (required disclosures, forbidden promises) must hold on every call. Would you enforce them with a separate guardrail classifier, with rules, or inside the agent prompt? How do you measure the guardrail's miss rate when violations are rare? [Likely] [19][26]
116. Explain mixture-of-experts models. What do they change for serving cost, latency and multi-LoRA adaptation compared with a dense model of similar quality? [Likely] [10]
117. Explain FlashAttention and why attention is memory-bound. How does it change what context lengths are practical for real-time assist on long calls? [Likely]
118. Can acoustic and prosodic signals (tone, pace, interruptions) improve sentiment or escalation prediction over transcript-only models? How would you fuse audio and text features, and prove the gain is worth the extra latency? [Likely] [14][24]
119. You must migrate every customer from one base model to a newer one. How would you plan the migration (re-training adapters, re-calibrating judges and thresholds, staged rollout) and decide which customers move first? [Likely] [10][17]

## References

1. Glassdoor, "Cresta (CA) Interview Questions" (Senior Machine Learning Engineer loop: recruiter call, LeetCode-medium, vibe-coding and system design rounds; "How would you deploy a model in production?"), 2025-2026. https://www.glassdoor.co.in/Interview/Cresta-CA-Interview-Questions-E3180909.htm
2. Glassdoor, "Cresta (CA) Interview Experience and Questions" (FDE technical screen: RAG and hallucination), 2026. https://www.glassdoor.com/Interview/Cresta-CA-Interview-Questions-E3180909.htm
3. Blind, "Anyone interview with Cresta lately?" (recruiter email: "key concepts and fundamentals around LLMs, GenAI, RAG, and AI Agents"), 2025. https://www.teamblind.com/post/anyone-interview-with-cresta-lately-zvj0uhq2
4. PracHub, "Cresta Interview Questions" (company page; "Design sequence decoding with greedy and beam search"), 2025. https://prachub.com/companies/cresta
5. PracHub, "Cresta Machine Learning Interview Questions" ("Implement greedy and beam decoding"). https://prachub.com/companies/cresta/categories/machine-learning
6. 1point3acres, "Cresta Machine Learning Engineer Tech Phone Interview Experience" (MLE, AI Agent direction; greedy and beam-search decoding), 2025. https://www.1point3acres.com/interview/thread/1140481
7. PracHub, "Cresta ML System Design Interview Questions" ("Design an Evaluation Platform for AI Agents Across Prompt and Model Changes"), 2026. https://prachub.com/companies/cresta/categories/ml-system-design
8. Cresta Blog, "How Fine-Tuned LLMs Power Knowledge Assist, Summarization and Chat Suggestions", 2024. https://cresta.com/blog/how-fine-tuned-llms-power-knowledge-assist-summarization-and-chat-suggestions
9. Fireworks AI Blog, "Story: Cresta Knowledge Assist", 2024. https://www.fireworks.ai/blog/story-cresta-knowledge-assist
10. Fireworks AI Blog, "Multi-LoRA", 2024. https://fireworks.ai/blog/multi-lora
11. Cresta Blog, "Introducing Ocean-1: World's First Contact Center Foundation Model", 2023. https://cresta.com/blog/introducing-ocean-1-worlds-first-contact-center-foundation-model
12. Cresta Blog, "AI-Ready Knowledge for Contact Centers: Closing the Gap Between KBs and AI", 2025. https://cresta.com/blog/ai-ready-knowledge-for-contact-centers-closing-the-gap-between-kbs-and-ai
13. Cresta Blog, "Engineering for Real-Time Voice Agent Latency". https://cresta.com/blog/engineering-for-real-time-voice-agent-latency
14. Cresta Blog, "Understanding Cresta's Voice Platform: ML Services, Inference Graphs and Real-Time Intelligence", 2025. https://cresta.com/blog/understanding-crestas-voice-platform-ml-services-inference-graphs-and-real-time-intelligence
15. Cresta Blog, "Understanding Cresta's Voice Platform: The Voice Stack", 2025. https://cresta.com/blog/understanding-crestas-voice-platform-the-voice-stack
16. Cresta Blog, "How Cresta Grades Its Agents: Evaluators in an LLM World". https://cresta.com/blog/how-cresta-grades-its-agents-evaluators-in-an-llm-world
17. Cresta Blog, "Cresta Launches Automated AI Agent Testing: Confidence in Every Conversation". https://cresta.com/blog/cresta-launches-automated-ai-agent-testing-confidence-in-every-conversation
18. Cresta Blog, "Grounding Reality: How Cresta Tackles LLM Hallucinations in Enterprise AI". https://cresta.com/blog/grounding-reality---how-cresta-tackles-llm-hallucinations-in-enterprise-ai
19. Cresta Blog, "Why AI Agent Evaluations Fail, and How the Swiss-Cheese Model Prevails". https://cresta.com/blog/why-ai-agent-evaluations-fail----and-how-the-swiss-cheese-model-prevails
20. Cresta Blog, "Cresta Conductor: The Agent for AI Agent Development". https://cresta.com/blog/cresta-conductor-the-agent-for-ai-agent-development
21. Cresta Blog, "Introducing the Next Generation of Cresta Insights: A Complete System for Enterprises to Understand Their CX", 2026. https://cresta.com/blog/introducing-the-next-generation-of-cresta-insights-a-complete-system-for-enterprises-to-understand-their-cx
22. Cresta on Greenhouse, "Senior Machine Learning Engineer". https://job-boards.greenhouse.io/cresta/jobs/4249943008
23. Cresta on Greenhouse, "Staff Machine Learning Engineer". https://job-boards.greenhouse.io/cresta/jobs/4741283008
24. Cresta on Greenhouse, "Senior Machine Learning Engineer, Voice Experience". https://job-boards.greenhouse.io/cresta/jobs/5155675008
25. Cresta on Greenhouse, "Senior Machine Learning Engineer (ASR)". https://job-boards.greenhouse.io/cresta/jobs/5101011008
26. Built In Vancouver, "Senior Machine Learning Engineer, Cresta" (Agentic Assist track). https://builtinvancouver.org/job/senior-machine-learning-engineer/8619914
27. CX Today, "Cresta CEO Ping Wu: Many Contact Centers Misunderstand the Value of LLMs". https://www.cxtoday.com/contact-center/cresta-ceo-ping-wu-many-contact-centers-misunderstand-the-value-of-llms/
28. PracHub, "Cresta Software Engineer Interview Experience — System Design Round: Build an Eval Platform for AI Agents" (September 2026 onsite; measure how prompt or model changes affect agent performance, including tool usage and response speed), 2026. https://prachub.com/interview-experiences/cresta-software-engineer-interview-experience-system-design-round-build-an-eval-platform-for-ai-agents
29. 1point3acres, "Cresta Software Engineer Tech Phone Screen Overview" (implementing Wikipedia API calls and discussing AI trends; candidate advanced to the virtual onsite). https://www.1point3acres.com/interview/thread/1173094

*Removed or replaced in review:* the Blind Cresta AI SWE AMA thread (it no longer supported any question here). The Blind company-posts page (its "know GenAI/RAG stuff" quote could not be confirmed, and chapter 01 had already dropped it). The Scoutify guide (a generic template, not candidate reports). The startup.jobs and Built In Bristol reposts, replaced by the original Greenhouse postings (refs 24 and 25). The opaque `cresta.com/?p=30805` link, replaced by the canonical voice-stack post (ref 15). Generic filler was removed: "implement logistic regression or k-means from scratch" (a coding item covered in chapter 02), "design an ML pipeline with a feature store" (too generic), and "walk through a project you led" (covered in chapter 05). The bias-variance question was rewritten into a senior-level overfitting and leakage question (Q19).
