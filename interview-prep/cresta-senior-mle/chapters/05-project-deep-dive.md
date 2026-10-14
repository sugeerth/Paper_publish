# 05. Project Deep Dive and Hiring Manager Round: Defending the Architecture of ML Systems You Shipped

> **Scope.** You asked for ML system design content, so this chapter treats the deep dive as an ML system design round about your own past work. Expect to draw your shipped system end to end, covering data, training, serving, evaluation and monitoring, and then defend each box and arrow. Hiring-manager questions are kept only where they probe how you designed, scoped or owned ML systems.

## Format notes for this round

- **What candidates report.**
  - Cresta software-engineering loops have included a **project presentation** round [2].
  - A Senior FDE loop included a separate **leadership round** [2].
  - A Senior/Staff backend loop is described as one hour of coding, one hour of system design and a **hiring-manager chat** [4].
  - A mid-level SWE loop reportedly had **three behavioral rounds** alongside the coding rounds [3].
  - The reported Senior MLE loop was a recruiter call, a LeetCode-medium round, a "vibe coding" round with an LLM tool, and a system design round ("How would you deploy a model in production?") [1]. A recruiter email to another candidate said interviews would cover "key concepts and fundamentals around LLMs, GenAI, RAG, and AI Agents" [5].
  - An MLE (AI Agent) phone screen covered greedy and beam-search decoding [25]. PracHub lists "Design an Evaluation Platform for AI Agents Across Prompt and Model Changes" as a Cresta ML system design question [22][26]. An FDE technical screen covered RAG and hallucination [31].
- **What is not confirmed.** No source reports a standalone project deep dive for the Senior MLE role. Assume it may be folded into the hiring-manager chat or the system design round, so prepare one project you can draw as an architecture diagram in 5 minutes.
- **Duration and interviewer (inferred).** Expect 45-60 minutes with an engineering manager or a senior or staff MLE from the target track: Agentic Assist, Agent and System Quality, Voice/ASR, Chat Agent, or Insights [7][8][9][10][11][30].
- **Senior bar (inferred from postings).**
  - You owned an ML system end to end, from problem framing to production.
  - You can diagnose LLM failure modes: hallucination, retrieval errors, tool misuse, context drift, prompt brittleness, and multi-step reasoning breakdowns [7][11].
  - You evaluate with offline benchmarks, online experiments and LLM-as-a-judge [7].
  - You make deliberate latency and cost trade-offs [17][18].
  - You can turn vague customer asks into measurable ML requirements. Cresta runs forward-deployed engineering loops, so expect MLEs to work alongside customer-facing teams [6].
- **Culture signals.** A third-party summary cites these principles: "Mission is the boss", "Do too much", "Have the courage to be disliked", and "patient in strategy, impatient in execution" [21]. Hiring-manager questions below are framed around them.
- **Research caveat.** During this review, every cited page failed to load (DNS errors for Glassdoor, Blind, Greenhouse, PracHub, 1point3acres, cresta.com and fireworks.ai), and the web-search budget for the run was used up. So no page was re-read and no new candidate reports could be found. Each source is kept only where search-result excerpts gathered for this bank (chapters 01-07) support it. [Reported] marks only questions a source attributes to Cresta. Everything else is [Likely], inferred from the role, Cresta's products and its engineering blog. All added questions are [Likely].

## A. Opening: draw the system

1. Present a significant project you built end to end. [Reported format for SWE: project presentation round] [2]
   - What did you personally own, and what did others build?
   - What was the baseline, and how did you prove the improvement?
2. Draw the architecture of the most complex ML system you shipped, from raw data to the user-facing output. [Likely]
   - Which box on this diagram was hardest to build, and which failed most in production?
   - Which component would you delete if you rebuilt it today?
3. How would you deploy the model from your project in production? Walk through the path from checkpoint to live traffic. [Reported, as a system design prompt for Senior MLE] [1]
   - How did you roll out (shadow, canary, A/B), and what triggered a rollback?
   - How did you version models, prompts and configs together?
4. Walk through one request through your system with timings at each hop. [Likely]
   - Where did p99 latency come from, and what did you do about it?
5. What was the one-sentence problem statement, and who was the user? [Likely]
   - How did the business metric map to your model metric?
6. Was this an LLM system, a classical ML system, or a hybrid? Why that split? [Likely]
   - Which parts did you leave as rules or heuristics on purpose?
7. Redraw the diagram as it looked at v1, then as it looks now. What forced each change? [Likely]
8. Which non-functional requirement (latency, availability, cost, data residency, compliance) shaped the architecture most, and which box exists only because of it? [Likely] [7]

## B. Data and labeling in your project

9. Where did your training and evaluation data come from, and how was it labeled? [Likely]
   - How did you measure label quality and annotator agreement?
10. How did you handle PII or sensitive data in the pipeline? [Likely] [7]
    - Would your approach work for contact-center transcripts with card numbers and health data?
11. Could a customer opt out of training or ask for its data to be deleted? How would you remove that data's influence from a model already in production? [Likely] [23]
12. How did you detect data drift or a shift in the input distribution after launch? [Likely]
13. Did you use synthetic data or LLM-generated labels? How did you validate them? [Likely]
    - What biases did the generator add?
14. How did you split train, validation and test data to avoid leakage (for example by customer, by time, by conversation)? [Likely]
15. How did you version datasets so that any production model could be traced to the exact data and code that trained it? [Likely]
16. If your inputs came from ASR or OCR, how did upstream errors affect labels and model quality, and what did you do about it? [Likely] [9]
17. If you had to build the same dataset for a new enterprise customer in two weeks, what would you do? [Likely] [12]

## C. Modeling decisions

18. Why did you choose this model family or base model over the alternatives? [Likely]
    - What did you try that did not work?
19. Did you fine-tune, prompt, or use retrieval? What evidence drove the choice? [Likely] [12][13]
    - What would make you switch to another approach?
20. If you fine-tuned, walk through the setup: data size, method (full, LoRA, adapters), hyperparameters, and compute. [Likely] [13]
    - How did you stop catastrophic forgetting or regressions on general ability?
21. Cresta reports that fine-tuned smaller models matched GPT-4-level quality on Knowledge Assist at lower cost and latency. Did you make a similar trade in your project? Show the numbers. [Likely] [12][13]
22. How did you handle per-customer customization: one model, per-customer adapters, or per-customer prompts? [Likely] [13][27]
    - How would your design scale to thousands of customers?
23. Describe a decision where you chose a simpler model over a more sophisticated one. [Likely]
24. How did you choose decoding settings (greedy, beam, sampling, temperature) for your generation task, and what did each do to quality, diversity and latency? [Likely; decoding was tested in an MLE phone screen] [25]
25. Did you use structured outputs or constrained decoding for extraction or tool calls? How did you handle schema violations at runtime? [Likely]
26. How did you handle inputs longer than the context window, such as hour-long calls or long chat histories? [Likely]
27. When the base model was upgraded or the API provider changed versions, what broke in your prompts, and how did you regression-test the upgrade? [Likely]
28. If you used RAG, draw the retrieval pipeline: chunking, embedding, index, reranking and grounding. [Likely] [7][31]
    - How did you measure retrieval quality apart from generation quality?
    - What did you do when the knowledge base was stale or contradictory?
29. If your system was an agent with tools, how did you define the tool schemas and control the loop? [Likely] [11]
    - How did you stop runaway loops or wrong tool calls?

## D. Evaluation and quality

30. Describe the evaluation you built. What did your metrics miss? [Likely]
31. Tell me about a model that worked offline but failed online. How did you find the gap, and how did you change the offline evaluation afterwards? [Likely]
32. Did you use LLM-as-a-judge? How did you calibrate it against humans? [Likely] [15]
    - How did you catch judge bias such as position, length or self-preference?
33. Who wrote the rubrics for human raters, and how did you measure and improve inter-rater agreement? [Likely] [15]
34. How did you gate releases? Which tests had to pass before a prompt or model change shipped? [Likely] [20][22]
    - How fast did that gate run, and who could override it?
35. Redraw your project's evaluation stack as a platform for AI agents where prompt and model changes ship every week: datasets, judges, statistical gates and reporting. [Reported, as a Cresta ML system design prompt] [22][26]
    - How would it run across many customers whose agents use different prompts?
36. How large were your eval sets, and how did you decide that a metric difference was statistically significant rather than noise? [Likely]
37. How did you keep eval sets from going stale, or from leaking into training data and prompts? [Likely]
38. Cresta describes layering evaluations like a "Swiss-cheese model". How many independent layers of checks did your project have? [Likely] [16]
    - Which layer caught the most real issues?
39. How did you measure and reduce hallucinations? [Likely] [19][31]
    - What grounding or citation checks did you add?
40. How did you evaluate multi-turn conversations, not just single responses? [Likely] [15][20]
    - Did you simulate users? How realistic were the simulations?
41. What was your error analysis process? Show one taxonomy of failures you built. [Likely]
42. How did you test rare but high-stakes behaviors, such as compliance or legal disclosures? [Likely]

## E. Serving, latency and cost

43. Draw your serving stack: hardware, inference server, batching, caching and autoscaling. [Likely] [13]
44. What was your latency budget, and how was it split across components? [Likely] [17]
    - For a real-time voice agent (ASR, turn detection, LLM, TTS), where would your system's budget go? [Likely] [17][18]
45. What did one request or conversation cost, and how did you bring it down? [Likely]
46. Did you use streaming, speculative decoding, quantization or distillation? What did each buy you? [Likely]
    - What quality did you lose, and how did you measure it?
47. How did your system behave under a traffic spike or a provider outage? [Likely]
    - What was your fallback path?
48. How did you handle provider rate limits and model deprecations when you depended on third-party APIs? [Likely]
49. At what request volume did self-hosting an open model beat a hosted API on cost for your project? How did you calculate the break-even? [Likely] [13][23]
50. How did you serve multiple models or adapters on shared hardware? [Likely] [13][27]
    - With thousands of per-customer LoRA adapters, how would you handle adapter loading, cold starts and GPU memory?
51. If your inference ran as a graph of ML services, how did you order and parallelize the nodes? [Likely] [18]

## F. Production, monitoring and incidents

52. What did you monitor after launch: system metrics, model metrics and business metrics? [Likely]
    - Which alert fired most often, and was it useful?
53. Describe the hardest production incident on this system, and what the postmortem changed. [Likely]
54. How did you detect silent model degradation without labels? [Likely]
55. How did you collect feedback from users or agents and feed it back into training? [Likely] [12]
    - How did you keep the feedback loop from reinforcing bias?
56. How did you run A/B tests in a B2B setting with a small number of large customers? [Likely]
    - How did you handle customers who refused experiments?
57. How did you roll a change out customer by customer, and how did you roll back one customer without affecting the rest? [Likely]
58. What audit trail did you keep so you could explain, weeks later, why the system produced a specific output for a specific customer? [Likely]
59. What would break first if traffic grew 10x? [Likely]

## G. Impact and ownership

60. What was the measurable impact: revenue, handle time, conversion, CSAT, or cost? How confident are you in that number? [Likely]
    - How did you separate your system's effect from seasonality or other launches?
61. What did you personally decide, and where did you defer to others? [Likely]
62. What would you do differently if you started this project again? [Likely]
63. What technical debt did you knowingly take on, and how did you pay it back? [Likely]
64. How did you scope the project to deliver value in weeks rather than quarters? [Likely]
65. If we gave you this problem in the contact-center domain, what would change in your diagram, and which parts would you reuse as they are? [Likely]

## H. Transfer to Cresta's systems (whiteboard follow-ups)

66. Map your project onto real-time agent assist: what would the live-transcript-to-suggestion path look like? [Likely] [12][18]
    - How would your model consume partial, changing ASR transcripts rather than final text? [Likely] [29]
67. How would you adapt your evaluation setup to grade AI agents on thousands of customer conversations? [Likely] [15][22]
68. How would your data pipeline change for post-call analytics and topic discovery at enterprise scale? [Likely] [24]
69. Cresta built Ocean-1 as a contact-center foundation model. If you were extending a domain model like that, how would your project experience apply? [Likely] [14]
70. How would you design guardrails for an autonomous AI agent talking to end customers? [Likely] [19]
71. When should an AI agent hand off to a human, and what context should it pass along? [Likely]
72. Using your RAG experience, how would you make a customer's messy knowledge base "AI-ready": deduplication, conflict resolution and freshness? [Likely] [28]
73. Cresta Conductor builds AI agents from natural-language instructions. How would you test and version an agent built that way before it reaches customers? [Likely] [20][32]
74. For a chat agent track: how would you evaluate an agent that completes multi-step tasks over a long chat, such as a refund that needs identity checks? [Likely] [30]
75. For a voice experience track: how would you reduce time to first audio in your architecture? [Likely] [10][17]
76. For an ASR track: how would you adapt ASR to customer-specific vocabulary such as product names and account terms? [Likely] [9]
77. Which metrics would you put on a dashboard for a customer's AI-agent deployment? [Likely]

## I. Hiring manager: ML ownership and judgment

78. Walk through your background, focusing on ML systems you shipped. Why Cresta and contact-center AI? [Likely]
79. Which track appeals to you: Agentic Assist, Agent and System Quality, Voice/ASR, Chat Agent, or Insights? [Likely] [7][8][30]
    - Where would you have the most impact in your first six months?
80. Tell me about a time you "did too much" to get an ML system shipped. [Likely] [21]
81. Tell me about pushing back on a PM, customer or leader with data ("courage to be disliked"). [Likely] [21]
82. Tell me about a time you were patient on strategy but impatient on execution. [Likely] [21]
83. Describe working directly with customers or forward-deployed teams to define what an ML feature should do. [Likely] [6]
    - How did you turn a vague ask into measurable requirements?
84. A customer gives you an ambiguous request and a one-week deadline. How do you define success for the model yourself? [Likely]
85. Describe a disagreement over quality versus shipping speed for a model launch. [Likely]
86. How do you explain a model regression to an enterprise customer? [Likely]
87. Tell me about a technical bet you got wrong. How did you notice, and how fast did you change course? [Likely]
88. How do you decide which new LLM research or model release is worth adopting? [Likely]
89. How do you split your time between research exploration and shipping in a startup? [Likely]
90. When would you build a capability in-house rather than use a third-party model API? [Likely] [23]
91. How do you raise the ML bar of a team: design reviews, eval standards, or mentoring? [Likely]
92. Tell me about mentoring an engineer through their first production ML launch. [Likely]
93. Tell me about changing another team's technical direction without authority over them. [Likely]
94. If evaluation for a key model turned out to be weak, what would you do in your first 90 days? [Likely]
95. Tell me about a project you killed or pivoted. How did you decide? [Likely]
96. How do you prioritize between customer escalations and roadmap ML work? [Likely]
97. What separates a senior MLE from a staff MLE, in your view? Where are you? [Likely] [8]
98. How do you use AI coding tools in your own ML engineering work? [Likely] [1]
    - How do you check generated code before it reaches production?

## J. Questions to ask the hiring manager

99. Which team would I join, and what ML systems does it own end to end? [Likely]
100. How does the team gate prompt and model changes before they reach customers? [Likely]
101. What is the split between self-hosted models and third-party APIs, and where is it heading? [Likely]
102. How much direct customer contact do MLEs have? [Likely]
103. What is the on-call expectation for MLEs, and who owns model incidents? [Likely]
104. What does success look like for this role at 6 and 12 months? [Likely]

## References

1. Glassdoor, "Cresta (CA) Interview Questions" (includes Senior Machine Learning Engineer loop with vibe-coding and system design rounds; "How would you deploy a model in production?"), 2025-2026. https://www.glassdoor.co.in/Interview/Cresta-CA-Interview-Questions-E3180909.htm
2. Glassdoor, "Cresta (CA) Interview Questions" (SWE project presentation; Senior FDE leadership round), 2025-2026. https://www.glassdoor.ca/Interview/Cresta-CA-Interview-Questions-E3180909.htm
3. Blind, "Cresta interview process", 2024-2025. https://www.teamblind.com/post/cresta-interview-process-jmfnzjoa
4. Blind, "Cresta senior backend SWE interview", 2025. https://www.teamblind.com/post/cresta-senior-backend-swe-interview-6v1tmjld
5. Blind, "Anyone interview with Cresta lately?" (recruiter email on LLMs, GenAI, RAG and AI Agents), 2025. https://www.teamblind.com/post/anyone-interview-with-cresta-lately-zvj0uhq2
6. Blind, "Interview for forward deployment engineer at Cresta", 2025. https://www.teamblind.com/post/interview-for-forward-deployment-engineer-at-cresta-ngswbj44
7. Cresta on Greenhouse, "Senior Machine Learning Engineer". https://job-boards.greenhouse.io/cresta/jobs/4249943008
8. Cresta on Greenhouse, "Staff Machine Learning Engineer". https://job-boards.greenhouse.io/cresta/jobs/4741283008
9. Cresta on Greenhouse, "Senior Machine Learning Engineer (ASR)". https://job-boards.greenhouse.io/cresta/jobs/5101011008
10. Cresta on Greenhouse, "Senior Machine Learning Engineer, Voice Experience". https://job-boards.greenhouse.io/cresta/jobs/5155675008
11. Built In Vancouver, "Senior Machine Learning Engineer, Cresta" (Agentic Assist track). https://builtinvancouver.org/job/senior-machine-learning-engineer/8619914
12. Cresta Blog, "How Fine-Tuned LLMs Power Knowledge Assist, Summarization and Chat Suggestions", 2024. https://cresta.com/blog/how-fine-tuned-llms-power-knowledge-assist-summarization-and-chat-suggestions
13. Fireworks AI Blog, "Story: Cresta Knowledge Assist", 2024. https://www.fireworks.ai/blog/story-cresta-knowledge-assist
14. Cresta Blog, "Introducing Ocean-1: First Contact Center Foundation Model", 2023. https://cresta.com/blog/introducing-ocean-1-worlds-first-contact-center-foundation-model
15. Cresta Blog, "How Cresta Grades Its Agents: Evaluators in an LLM World". https://cresta.com/blog/how-cresta-grades-its-agents-evaluators-in-an-llm-world
16. Cresta Blog, "Why AI Agent Evaluations Fail, and How the Swiss-Cheese Model Prevails". https://cresta.com/blog/why-ai-agent-evaluations-fail----and-how-the-swiss-cheese-model-prevails
17. Cresta Blog, "Engineering for Real-Time Voice Agent Latency". https://cresta.com/blog/engineering-for-real-time-voice-agent-latency
18. Cresta Blog, "Understanding Cresta's Voice Platform: ML Services, Inference Graphs, and Real-Time Intelligence", 2025. https://cresta.com/blog/understanding-crestas-voice-platform-ml-services-inference-graphs-and-real-time-intelligence
19. Cresta Blog, "Grounding Reality: How Cresta Tackles LLM Hallucinations in Enterprise AI". https://cresta.com/blog/grounding-reality---how-cresta-tackles-llm-hallucinations-in-enterprise-ai
20. Cresta Blog, "Cresta Launches Automated AI Agent Testing: Confidence in Every Conversation". https://cresta.com/blog/cresta-launches-automated-ai-agent-testing-confidence-in-every-conversation
21. JobsByCulture, "Cresta: Frontend Engineer" (culture and operating-principles summary; third-party). https://jobsbyculture.com/jobs-at/cresta/frontend-engineer
22. PracHub, "Cresta ML System Design Interview Questions" ("Design an Evaluation Platform for AI Agents Across Prompt and Model Changes"), 2026. https://prachub.com/companies/cresta/categories/ml-system-design
23. CX Today, "Cresta CEO Ping Wu: Many Contact Centers Misunderstand the Value of LLMs". https://www.cxtoday.com/contact-center/cresta-ceo-ping-wu-many-contact-centers-misunderstand-the-value-of-llms/
24. Cresta Blog, "Introducing the Next Generation of Cresta Insights", 2026. https://cresta.com/blog/introducing-the-next-generation-of-cresta-insights-a-complete-system-for-enterprises-to-understand-their-cx
25. 1point3acres, "Cresta Machine Learning Engineer Tech Phone Interview Experience" (MLE, AI Agent direction; greedy and beam-search decoding), 2025. https://www.1point3acres.com/interview/thread/1140481
26. PracHub, "Cresta Software Engineer Interview Experience: System Design Round: Build an Eval Platform for AI Agents". https://prachub.com/interview-experiences/cresta-software-engineer-interview-experience-system-design-round-build-an-eval-platform-for-ai-agents
27. Fireworks AI Blog, "Multi-LoRA", 2024. https://fireworks.ai/blog/multi-lora
28. Cresta Blog, "AI-Ready Knowledge for Contact Centers: Closing the Gap Between KBs and AI", 2025. https://cresta.com/blog/ai-ready-knowledge-for-contact-centers-closing-the-gap-between-kbs-and-ai
29. Cresta Blog, "Understanding Cresta's Voice Platform: The Voice Stack", 2025. https://cresta.com/blog/understanding-crestas-voice-platform-the-voice-stack
30. Cresta on Lever, "Senior Machine Learning Engineer (Chat Agent)". https://jobs.lever.co/cresta/172fe276-9415-4bae-9b0b-af72cf7cf014
31. Glassdoor, "Cresta (CA) Interview Experience and Questions" (FDE technical screen: RAG and hallucination), 2026. https://www.glassdoor.com/Interview/Cresta-CA-Interview-Questions-E3180909.htm
32. Cresta Blog, "Cresta Conductor: The Agent for AI Agent Development". https://cresta.com/blog/cresta-conductor-the-agent-for-ai-agent-development

*Removed or changed in review:* the Blind "Cresta AI SWE AMA" thread (chapter 03's review found it no longer supported any question; the customer-facing claim now rests on the FDE thread). The Scoutify guide (a generic template, not candidate reports). Two duplicates were merged: the "offline vs online correlation" question went into "worked offline but failed online" (Q31), and "which part could be reused at Cresta" went into the contact-center transfer question (Q65). Q1 was retagged from "[Reported for SWE]" to "[Reported format for SWE]", because the source reports a project presentation round, not this exact wording. Eight sources already cited elsewhere in this bank (chapters 01-04 and 07) were added as refs 25-32. Their pages were not re-opened in this review.
