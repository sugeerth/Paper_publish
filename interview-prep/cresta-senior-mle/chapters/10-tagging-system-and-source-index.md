# 10. Tagging System and Source Index: What Each Tag Means, Every Reported ML System Design Prompt by Portal, and Where Each Likely Prompt Comes From

> **Why this chapter exists.** You asked for details on the tagging system. Chapters 01-09 (there is no chapter 06) each repeat the [Reported]/[Likely] definitions in their header, keep their own reference lists, and scatter research caveats. This chapter puts all of it in one place.
> - Part 1 defines each tag and the evidence needed to earn it.
> - Part 2 lists every [Reported] system design prompt with its portal, date, role, round and evidence strength.
> - Part 3 maps each [Likely] design prompt to the job posting, blog post or product it is inferred from.
> - Part 4 lists questions to confirm the loop with your recruiter.
>
> Back-pointers such as "ch04 Q5" mean question 5 in chapter 04. Reference numbers in this chapter use the single consolidated list at the end, not the per-chapter lists.

## Research limits (stated once, applies to the whole bank)

- **No source page was ever opened.** During research and every review pass, the egress proxy refused, or DNS failed for, every cited domain: Glassdoor, Blind (teamblind.com), 1point3acres, PracHub, Greenhouse, Lever, Built In, CoderPad, cresta.com, fireworks.ai and CX Today. While this chapter was being written, a fetch of the PracHub ML system design page failed again (DNS lookup `ENOTFOUND prachub.com`). In the adversarial review of this chapter, 12 more cited URLs were tried, and every one failed DNS lookup (`ENOTFOUND`): PracHub [10], 1point3acres [14], Glassdoor [1], Blind [6], Greenhouse [19], Lever [24], Built In [23], CoderPad [17], a16z Jobs [18], cresta.com [27], fireworks.ai [35] and CX Today [39]. A failed lookup means the page could not be checked. It does not show that the URL is dead or invented, so no reference was removed for that reason.
- **Live web search was exhausted.** The shared web-search budget for this run was already used up. A test search while writing this chapter was refused, and so was a search for Cresta MLE system design reports during the review. No new candidate reports could be found, and no new [Reported] prompts were added. Everything here comes from re-reading chapters 01-09 and the search-result excerpts they recorded, not from new searches.
- **What a [Reported] tag therefore rests on.** It rests on a search-result excerpt (title plus snippet) gathered while the bank was first researched, never on a full read of the page. Dates, roles and wording come from those excerpts and may be incomplete.
- **What you should do.** Treat the loop description as provisional. Before the onsite, ask your recruiter to confirm the round list, which round is ML system design, how long it lasts, who conducts it, and whether the AI-assisted coding round is in your loop (see Part 4).

## Format notes for this round

- **ML system design slot.** About 60 minutes. One 2025-26 Senior MLE report lists it after a recruiter call, a LeetCode-medium round and a "vibe coding" round [1]. Senior/Staff SWE reports describe one hour of system design next to one hour of coding and a hiring-manager chat [6].
- **Interviewer (inferred).** Most likely a senior or staff MLE, or an engineering manager, from the track you are matched to: Agentic Assist, Agent and System Quality, Voice/ASR, Chat Agent or Insights [19][20][21][22][23][24].
- **Senior bar (from postings, not candidate reports).**
  - End-to-end ownership from research to production [23].
  - Evaluation of non-deterministic systems with offline benchmarks, online experiments and LLM-as-a-judge [19][20].
  - Diagnosis of hallucination, retrieval errors, tool misuse, context drift and prompt brittleness [19][20].
  - Design for latency, cost, scale and security, including real-time streaming for voice [21].
- **Process length.** About 25 days on average across Cresta engineering roles [2].

## Part 1. The tags and their evidence thresholds

### 1.1 Tag definitions

| Tag | Meaning | Minimum evidence to earn it | What it does not mean |
|---|---|---|---|
| **[Reported] [n]** | A candidate report or a question bank attributes this question, or a close variant, to Cresta. | A search-result excerpt from an interview portal (Glassdoor, Blind, 1point3acres, PracHub) that names Cresta and states the question or its title. The citation [n] must point to that excerpt's page. | It does not mean the question was asked in the Senior MLE system design round. Where the report is for another role (SWE, FDE) or round (phone screen, coding), the tag says so. |
| **[Reported format]** | A source reports that a round of this type exists at Cresta, not the exact wording. | An excerpt that describes the round (for example "project presentation"), with no question text. | It does not mean the wording in this bank was reported. |
| **[Likely]** | A well-grounded inference from Cresta's products, engineering blog, partner write-ups and ML job postings. No candidate has reported it. | A specific Cresta posting, blog post or product that makes the topic central to the role. A citation on a [Likely] item points to that source, not to a candidate report. | It never means someone was asked it. |
| **[Likely scale-up]** | A design version of a question reported for a Cresta coding round (rate limiter, worker pool, job dispatcher, decoder, API call). The citation points to the coding report. | The coding question must itself be [Reported] (table 2.2 or R4). | The design version was not reported. |
| **[Self-check]** | A study prompt about the evidence behind a reported item, used only in section 3.9. | None. The citation names the reported item it examines. | It is not an interview question and was not reported. |

### 1.2 Evidence-strength scale used in Part 2

- **Strong:** the role (MLE) and the round (system design) both match, and the excerpt contains the question text.
- **Moderate:** the excerpt contains the question text, but the role or round differs, or the source is an aggregator listing with no role or date detail.
- **Weak:** only a listing title, a topic list (for example a recruiter's prep email), or a description of the round format.
- **Verified:** the full page was read and the question confirmed. **No item in this bank reaches this level**, because no page could be opened.

### 1.3 Source-handling rules applied across chapters

- **Glassdoor regional mirrors are one source.** The .co.in, .ca, .com and .sg pages (employer ID E3180909) show the same review pool. Two mirrors citing one review do not count as two reports.
- **PracHub is an aggregator.** Its pages are user-submitted question banks. A PracHub title counts as [Reported], but it is weaker than a first-person account.
- **Removed sources.** The Scoutify guide (a generic template, not candidate reports), the Blind "Cresta AI SWE AMA" thread (no excerpt supported any question) and the second 1point3acres thread 1173094 (no excerpt supported it) were removed.
- **Process claims from non-portal sources are [Likely].** A vendor case study (CoderPad [17]) or a job posting that describes the hiring process (a16z Jobs [18]) is not a candidate report, so a claim resting only on them is [Likely], not [Reported].
- **Restricted source.** The Blind "Cresta senior backend SWE interview" thread [9] supports only an online assessment, so it must not be cited for the system design round.
- **Open clean-up items.** Chapter 02's reference 12 still lists Scoutify. Chapter 01 has two questions numbered 34 and none numbered 39. Chapters 01, 02 and 05 still cite [9] for loop shape. Chapters 01 and 05 cite a JobsByCulture culture summary, and chapter 02 cites a ZeroG Talent repost of the Senior MLE posting [19]. Neither is an interview portal or supports a design prompt, so neither is indexed here.

## Part 2. Every [Reported] prompt, by portal

### 2.1 Index table: reported system design and design-adjacent prompts

| # | Prompt | Portal | Date | Role | Round | Strength |
|---|---|---|---|---|---|---|
| R1 | How would you deploy a model in production? | Glassdoor [1] | 2025-26 | Senior MLE | Systems design | Strong (one review, excerpt only) |
| R2 | Design an evaluation platform for AI agents across prompt and model changes | PracHub [10][11] | 2026 (listing); experience undated | SWE (experience post); listed under ML system design | System design | Moderate (aggregator; role is SWE) |
| R3 | What is RAG, and how do you prevent hallucination? | Glassdoor [3] | 2026 | Forward Deployed Engineer | Technical screen (conceptual) | Moderate for content, weak as a design prompt |
| R4 | Design sequence decoding with greedy and beam search | PracHub [12][13]; 1point3acres [14] | 2025 (Aug 2025 thread) | MLE, AI Agent direction | Phone screen (coding) | Moderate (role matches, round differs) |
| R5 | Walk through how an AI agent generates a sequence step by step | 1point3acres [14] | 2025 | MLE, AI Agent direction | Phone screen follow-up | Moderate |
| R6 | Key concepts of LLMs, GenAI, RAG and AI agents | Blind [7] | 2025 | Not stated (engineering) | Recruiter prep email (topic list) | Weak |
| R7 | One hour of system design in the loop | Blind [6] | 2024-25 | Senior/Staff SWE | Format only | Weak (format) |

### 2.2 Index table: reported coding prompts that design rounds may scale up

| # | Prompt | Portal | Date | Role | Round | Strength |
|---|---|---|---|---|---|---|
| C1 | Implement a rate limiter | Glassdoor [1] | 2024-26 | Not stated | Coding | Moderate |
| C2 | Worker pool plus orchestrator with round-robin assignment | Glassdoor [1] | 2024-26 | FDE, AI Agent | Coding (OOP) | Moderate |
| C3 | Job dispatcher/executor system | 1point3acres [15] | 2026 | Full-time SWE | Phone screen | Weak (title seen in a listing; text not read) |
| C4 | Call a public API (Wikipedia) and process results | 1point3acres [16] | 2024-26 | SWE | Phone screen | Moderate |
| C5 | Project presentation | Glassdoor [2] | 2025-26 | SWE | Project presentation | Weak (format only) |
| C6 | 1.5-hour practical with AI tools allowed; "vibe coding" with an LLM tool | Blind [8]; Glassdoor [1] | 2025; 2025-26 | FDE; Senior MLE | AI-assisted practical | Moderate |

### 2.3 Reported prompts as practice questions

1. How would you deploy a model in production? [Reported] [1] (Senior MLE systems design round, Glassdoor 2025-26; ch04 Q2, ch08 Diagram 4)
   - What path does the model take from registry through shadow, canary and full rollout, and where is the rollback path?
   - What would you change for a real-time assist model compared with a batch post-call QA model?
   - Which signals in the first 24 hours trigger an automatic rollback, and which only page a human?
   - How do you pin one tenant to an old version while the others upgrade?
2. Design an evaluation platform for AI agents that catches regressions when prompts or models change. [Reported] [10][11] (PracHub; experience post labeled SWE system design round; ch04 Q1, ch08 Diagram 3)
   - What must a run pin (prompt, model ID, tools, knowledge snapshot, test set) so it can be reproduced a month later?
   - Which checks are deterministic, and which need an LLM judge?
   - How many reruns per scenario, and which statistical test declares a regression?
   - What does a full run cost across hundreds of customer-specific agents?
3. What is RAG, and how do you prevent hallucination? [Reported] [3] (FDE technical screen, Glassdoor 2026; conceptual, not a design prompt; ch04 Q3, ch08 Q67)
   - If the interviewer asks you to draw it, where in the pipeline does each type of hallucination start, and which check catches it?
   - What do you do when the groundedness check fails in the middle of a live voice turn?
4. Design sequence decoding with greedy and beam search as a reusable component. [Reported] [12][14] (PracHub title; the matching 1point3acres thread is a 2025 MLE AI Agent phone screen; ch04 Q4, ch02 Q3)
   - What interface lets greedy, beam and sampling share code?
   - How does beam width trade quality against latency and memory inside a real-time service?
5. Walk through how an AI agent generates a sequence step by step. Where can the decoding strategy change the agent's behavior? [Reported] [14] (MLE AI Agent phone-screen follow-up; ch03 Q2, ch07 Q24)
   - How would constrained decoding change what the agent can emit as a tool call?
6. Explain the key concepts and fundamentals of LLMs, GenAI, RAG and AI agents. [Reported] [7] (recruiter prep email; a topic list, not a question; ch01 Q57, ch03 Q3)
   - Can you draw how the four fit together inside one Cresta product?

## Part 3. [Likely] design prompts mapped to their sources

Each prompt below is [Likely]: no candidate has reported it. The "Inferred from" line names the posting, blog post or product behind it. The citation is context only.

### 3.1 Inferred from the voice platform posts and the Voice Experience and ASR postings

7. Design real-time Agent Assist that listens to a live call and shows the human agent hints and suggested responses. [Likely] [29][28] (ch04 Q5, ch08 Diagram 1)
   - Inferred from: the voice-stack post (partial and final transcripts about once per second) and the inference-graph post.
   - What is your end-to-end latency budget from spoken words to on-screen hint?
   - Do you trigger on partial transcripts or wait for finals?
8. Design an inference graph that runs many models per conversation (ASR, redaction, intent, sentiment, summarization) with per-customer configuration. [Likely] [28] (ch04 Q9)
   - Inferred from: "Understanding Cresta's Voice Platform: ML Services, Inference Graphs and Real-Time Intelligence".
   - How do you stop one slow node from blocking real-time outputs?
9. Design a voice AI agent that answers inbound calls with responses in under a second. [Likely] [27][21] (ch04 Q19, ch08 Diagram 2)
   - Inferred from: "Engineering for Real-Time Voice Agent Latency" and the Voice Experience posting.
   - Which per-hop metrics do you track (ASR final latency, LLM time to first token, TTS time to first byte)?
   - How do you detect end of turn and handle barge-in?
10. Design a streaming ASR service for thousands of concurrent contact-center calls. [Likely] [22][29] (ch04 Q78)
    - Inferred from: the Senior MLE (ASR) posting and the voice-stack post.
    - How do you balance chunk size and lookahead against accuracy?
11. Design customer-specific vocabulary adaptation for ASR. [Likely] [22] (ch04 Q79, ch01 Q15)
    - Inferred from: the ASR posting.
    - Contextual biasing, fine-tuning or LLM post-correction: which, and why?
12. Design an ASR quality program across accents, noise and domains. [Likely] [22] (ch04 Q80)
    - How does WER relate to downstream task quality?
13. Design a test harness that measures how robust the LLM layer is to ASR errors. [Likely] [22] (ch04 Q41)
    - How do you inject realistic ASR noise on entities and spelled data?
14. Design real-time translation for a live call. [Likely] [29] (ch04 Q18)
    - Where does translation sit relative to ASR, hints and the agent's reply?
15. Design a real-time sentiment and escalation-risk detector. [Likely] [28] (ch04 Q10)
    - How do you combine acoustic and lexical signals?
16. Keep a voice agent's latency stable during peak traffic. [Likely] [27] (ch04 Q25)
    - What degrades first, and what is your graceful-degradation plan?
17. Design PII redaction for audio and transcripts before storage and training. [Likely] [28] (ch04 Q81)
    - What recall target does the cost of a missed card number set?

### 3.2 Inferred from the agent-evaluation posts and the Agent and System Quality track

18. Design a customer simulator that generates synthetic callers to test an AI agent before launch. [Likely] [37] (ch04 Q33)
    - Inferred from: "Cresta Launches Automated AI Agent Testing".
    - How do you stop the simulator and the agent from sharing blind spots?
19. Extend the simulator to voice: accents, background noise, interruptions and silence. [Likely] [37] (ch04 Q40)
    - Which failures appear only in audio?
20. Design a layered ("Swiss-cheese") reliability program for a production AI agent. [Likely] [26] (ch04 Q34)
    - Inferred from: "Why AI Agent Evaluations Fail, and How the Swiss-Cheese Model Prevails".
    - What does each layer catch, and what does it miss?
21. Design an LLM-as-a-judge evaluator service used for both testing and production scoring. [Likely] [25][19][20] (ch04 Q35)
    - Inferred from: "How Cresta Grades Its Agents: Evaluators in an LLM World" and the LLM-as-a-judge requirement in the Senior and Staff postings.
    - How do you calibrate judges against expert labels and detect judge drift?
22. Design production monitoring for AI agents across hundreds of enterprise deployments. [Likely] [19][20] (ch04 Q36)
    - Which label-free signals indicate a quality problem for one customer or one intent?
23. Measure whether an AI agent resolved a call when there is no survey or disposition code. [Likely] [32] (ch04 Q37)
    - How do you validate an inferred-outcome model without ground truth?
24. Design a shadow-mode launch in which a new AI agent runs silently beside human agents. [Likely] [26] (ch04 Q39)
    - Which disagreements are agent errors, and which are policy differences?
25. Design the curation pipeline for golden evaluation sets drawn from production conversations. [Likely] [37] (ch04 Q42)
    - How do you keep the set from going stale or leaking into training data?
26. Design an audit and replay system that reproduces why an agent did something three months ago. [Likely] [19] (ch04 Q43)
    - What must you log, and at what storage cost?
27. Design the release process for a prompt change that ships to 50 customer agents at once. [Likely] [26] (ch04 Q38)
    - How do you stage the rollout and roll back when a customer has customized the prompt?

### 3.3 Inferred from the AI Agent product, Conductor and the grounding post

28. Design the tool-use layer for an AI agent that looks up orders and issues refunds. [Likely] [38][19] (ch04 Q20)
    - Inferred from: Cresta Conductor (flight-rebooking example) and "tool misuse" in the Senior MLE posting.
    - How do you stop the agent from making unauthorized commitments?
29. Design the handoff from an AI agent to a human agent. [Likely] [19] (ch04 Q21)
    - What context do you pass to the human, and in what format?
30. Design an agent-building system that turns a customer's SOPs into a working AI agent. [Likely] [38] (ch04 Q22)
    - Inferred from: "Cresta Conductor: The Agent for AI Agent Development".
    - Structured flow, free-form LLM with tools, or a hybrid?
31. Design tool onboarding from a customer's API spec, with a sandbox and tests. [Likely] [38] (ch04 Q89)
    - How do you detect tool misuse after launch?
32. Design guardrails against prompt injection, PII leakage and policy violations. [Likely] [36] (ch04 Q23)
    - Inferred from: "Grounding Reality: How Cresta Tackles LLM Hallucinations in Enterprise AI".
    - Where do guardrails run, and what latency does each add?
33. Design a grounding checker that flags responses not supported by the knowledge base or tool outputs. [Likely] [36] (ch01 Q10)
    - Claim extraction plus entailment, or a single judge call?
34. Design a multi-agent architecture with a router and specialized sub-agents. [Likely] [23] (ch04 Q24)
    - When does it beat one agent with many tools?
35. Design context management for a 30-60 minute call with many tool calls. [Likely] [19][20] (ch04 Q27)
    - Inferred from: "context drift" in the Senior and Staff postings.
    - What do you keep verbatim, summarize, or store as structured state?
36. Design identity verification inside a voice agent before sensitive actions. [Likely] [21] (ch04 Q28)
    - How do you enforce "verify first" in code rather than in the prompt?
37. Design an outbound voice agent with answering-machine and IVR detection. [Likely] [21] (ch04 Q30)
    - How do consent records and calling-window rules enter the design?
38. Design an AI agent for asynchronous messaging (SMS, WhatsApp) where one conversation spans days. [Likely] [24] (ch04 Q32)
    - Inferred from: the Chat Agent posting on Lever.
    - What changes in latency targets and evaluation compared with voice?

### 3.4 Inferred from Knowledge Assist, the fine-tuning post and the Fireworks write-ups

39. Design Knowledge Assist: surface answers from a customer's knowledge base during a live conversation. [Likely] [33][34][31] (ch04 Q58, ch08 Diagram 5; extends reported R3)
    - Inferred from: "How Fine-Tuned LLMs Power Knowledge Assist, Summarization and Chat Suggestions", Fireworks "Story: Cresta Knowledge Assist" and "AI-Ready Knowledge for Contact Centers".
    - How do you turn implicit conversation context into a retrieval query?
40. Serve Knowledge Assist with a fine-tuned model at low latency and cost. [Likely] [33][34] (ch04 Q59)
    - When does fine-tuning beat prompting a frontier model?
41. Design the ingestion pipeline that makes enterprise knowledge "AI-ready" (duplicates, stale and contradictory articles). [Likely] [31] (ch04 Q60)
    - How do you surface conflicting articles to the content owner?
42. Design retrieval evaluation with no labeled query-document pairs. [Likely] [31] (ch04 Q62)
    - How far can you trust LLM-bootstrapped labels?
43. Plan an embedding-model upgrade that re-embeds billions of vectors without downtime. [Likely] [34] (ch04 Q64)
    - How do you cut over per tenant?
44. Design suggested-response generation for chat agents. [Likely] [33] (ch04 Q7)
    - Inferred from: the fine-tuning post (chat suggestions).
    - Which offline metric best predicts acceptance rate?
45. Design live summarization that is ready when the call ends and writes to the customer's CRM fields. [Likely] [33] (ch04 Q8, ch01 Q23)
    - How do you measure factual consistency at scale?
46. Design multi-tenant LLM serving: one base model with per-customer LoRA adapters. [Likely] [35][34] (ch04 Q65, ch08 Diagram 7)
    - Inferred from: Fireworks "Multi-LoRA" and the Knowledge Assist story.
    - How do you prevent cross-tenant leakage in caches, logs and training data?
47. Design per-customer fine-tuning as a product feature, with evaluation and rollback per tenant. [Likely] [35] (ch04 Q70)
    - How do you stop a tenant's adapter from regressing silently after a base-model upgrade?

### 3.5 Inferred from Ocean-1, the CEO interview and the serving requirements in postings

48. Design the domain adaptation of a base LLM for contact centers, in the spirit of Ocean-1. [Likely] [30] (ch04 Q68, ch01 Q24)
    - Inferred from: "Introducing Ocean-1: World's First Contact Center Foundation Model".
    - How do you show it beats a frontier API on quality, latency and cost?
49. Plan the migration of one high-volume feature from a frontier API to a distilled, self-hosted model. [Likely] [30][39] (ch04 Q69)
    - Inferred from: Ocean-1 and the CX Today interview with Cresta's CEO.
    - What is the break-even volume?
50. Design a model router between a self-hosted small model and a frontier API. [Likely] [39] (ch04 Q67)
    - How do you prove the router does not hurt quality?
51. Design an LLM serving stack that meets p99 latency targets for real-time features. [Likely] [19][21] (ch04 Q66)
    - Inferred from: latency, cost and model-optimization requirements in the postings.
    - What signal would you autoscale GPUs on?
52. Design a distillation pipeline in which a frontier LLM labels data for small real-time classifiers. [Likely] [30] (ch04 Q76)
    - How do you refresh the student when the customer's taxonomy changes?
53. Design scheduling that shares a GPU fleet between real-time inference and batch work. [Likely] [21] (ch04 Q75)
    - How do you pre-empt batch work without losing progress?
54. Design cost accounting for LLM usage per customer and per feature. [Likely] [19] (ch04 Q71)
    - How do you attribute GPU cost on shared, batched infrastructure?
55. Design for data residency and isolation on a shared LLM platform. [Likely] [19] (ch04 Q72)
    - What changes in serving, storage and training?

### 3.6 Inferred from the Insights launch and QA and coaching products

56. Design topic discovery over millions of unlabeled conversations, with an editable taxonomy. [Likely] [32] (ch04 Q44, ch08 Diagram 6)
    - Inferred from: "Introducing the Next Generation of Cresta Insights" (2026).
    - How do you keep assignments stable when admins merge or rename topics?
57. Design an "AI analyst" that answers open-ended business questions over a conversation corpus. [Likely] [32] (ch04 Q45)
    - How do you stop it from presenting a weak correlation as a finding?
58. Design a driver-analysis system that finds which agent behaviors correlate with good outcomes. [Likely] [32] (ch04 Q46)
    - How do you control for call type and customer mood?
59. Design an emerging-issue detector that flags spikes within minutes. [Likely] [32] (ch04 Q47)
    - How do you separate a true spike from seasonality?
60. Design automated QA that scores 100% of calls against each customer's rubric. [Likely] [25] (ch04 Q51)
    - Inferred from: Cresta's QA and evaluator products (the evaluators post).
    - How do you turn a free-text rubric item into an auditable judgment?
61. Design a system that lets a customer add a QA criterion and have it live within an hour. [Likely] [25] (ch04 Q53)
    - What does a backfill over a year of calls cost?
62. Design compliance monitoring with near-zero miss tolerance. [Likely] [25] (ch04 Q56)
    - Which checks run in real time, and which post-call?
63. Design storage and indexing for searching billions of utterances by keyword, meaning and metadata. [Likely] [32] (ch04 Q49)
    - How do you enforce per-tenant isolation in the index?

### 3.7 Scale-ups of reported coding questions

These are design prompts tagged [Likely scale-up]. The coding question underneath each one is [Reported] (see table 2.2), and the citation points to that coding report. The design version itself was never reported.

64. Scale the reported rate limiter into per-tenant quota enforcement in front of shared LLM serving. [Likely scale-up] [1] (ch04 Q74; built on C1)
    - Token-based or request-based limits?
    - How do you protect real-time traffic from a tenant's batch backfill?
65. Scale the reported worker pool and job dispatcher into an orchestrator for agent and evaluation jobs. [Likely scale-up] [1][15] (ch04 Q88; built on C2 and C3)
    - How do you handle stuck jobs, retries and idempotency?
    - How do you give each tenant a fair share?
66. Scale the reported decoder into a structured-output layer that guarantees valid tool-call JSON. [Likely scale-up] [12][14] (ch04 Q77, ch07 Q25; built on R4)
    - Constrained decoding, or validate-and-retry? What does each cost in latency?
67. Scale the reported "call a public API" exercise into a resilient connector layer for customer CRMs and knowledge bases. [Likely scale-up] [16] (built on C4)
    - How do you handle pagination, rate limits, schema drift and outages?
68. Redraw a project you shipped as an evaluation platform for weekly prompt and model changes. [Likely scale-up] [2][10] (ch05 Q35; combines C5 with R2)
    - Which parts of your old evaluation stack would survive, and which would you rebuild?

### 3.8 Cross-cutting probes inferred from the senior bar in postings

69. Which non-functional requirement dominates your design: latency, cost, quality or isolation? [Likely] [19][20] (ch04 Q90)
70. Give a back-of-envelope estimate: concurrent calls, tokens per day, GPUs needed. [Likely] [21] (ch04 Q91)
71. What is your offline evaluation, and how do you know it predicts online results? [Likely] [19][20] (ch04 Q93)
72. What happens when the LLM provider has an outage or rate-limits you mid-call? [Likely] [27] (ch04 Q95)
73. Where could this system hallucinate or mislead a human agent, and how do you contain the blast radius? [Likely] [36] (ch04 Q97)
74. How does the design change for chat versus voice, and for 10 customers versus 1,000? [Likely] [24][21] (ch04 Q96)
75. What would make you choose a classic ML model over an LLM for one component? [Likely] [23] (ch04 Q99)
    - Inferred from: the Agentic Assist track's call for strong ML foundations from before LLMs.

### 3.9 Testing yourself on the tags

These three are study prompts about the evidence, not interview questions, so they carry the [Self-check] tag (see table 1.1). The citation names the reported item each one examines.

76. For the prompt "How would you deploy a model in production?", what exactly did the source report, and which follow-ups in this bank are your own extrapolation? [Self-check] [1]
    - Can you prepare an answer that works whether the interviewer means an LLM feature or a classic model?
77. The agent-evaluation prompt is reported for an SWE system design round. How would an MLE interviewer likely push deeper than an SWE interviewer on the same prompt? [Self-check] [10][11]
    - Would they spend more time on judge calibration and statistical gates than on storage and APIs?
78. The RAG question was a conceptual FDE screen question. If it becomes a design prompt in your loop, which diagram would you draw first? [Self-check] [3]

## Part 4. Questions to confirm the loop with your recruiter

The tag on each question is the tag of the claim it checks.

79. Does my onsite include a dedicated ML system design round, and is it about 60 minutes? [Reported] [1][6]
    - The round's existence is reported for a Senior MLE [1]. The one-hour length is reported only for Senior/Staff SWE loops [6].
    - Is it ML system design specifically, or general backend system design?
80. Is the round more likely to centre on deploying and operating a model, or on LLM agents and evaluation? [Reported] [1][10]
81. Will there be an AI-assisted ("vibe coding") round, and which tools may I use? [Reported] [1][8]
    - Is it scored on the final product, or on how I direct and verify the tool?
82. Is the technical screen a LeetCode-style round, an ML coding problem such as decoding, or both? [Reported] [1][14]
83. Will an asynchronous CoderPad assessment come before the live rounds? [Likely] [17][18]
    - Retagged from [Reported]: [17] is a vendor case study and [18] is a job posting. Neither is a candidate report on an interview portal, and neither is for an MLE role.
84. Should I expect questions on LLM, GenAI, RAG and AI-agent fundamentals, as other candidates were told? [Reported] [7]
85. Is there a project presentation or deep-dive round for the MLE loop, as there is for some SWE loops? [Reported format] [2]
86. Which team or track is the system design interviewer from: Agentic Assist, Agent and System Quality, Voice/ASR, Chat Agent or Insights? [Likely] [19][21][22][23][24]
87. Will the design round focus on real-time voice, chat or batch analytics? [Likely] [21][28]
88. Is there a separate hiring-manager or behavioral round, and how long is it? [Reported] [6]
89. What is the expected timeline from onsite to decision? [Reported] [2]

## References

1. Glassdoor, "Cresta (CA) Interview Questions" (Senior MLE loop: recruiter call, LeetCode-medium, vibe-coding and systems design rounds; "How would you deploy a model in production?"; CoderPad and Zoom Python round; rate limiter; FDE AI Agent worker-pool question), 2024-2026. https://www.glassdoor.co.in/Interview/Cresta-CA-Interview-Questions-E3180909.htm
2. Glassdoor, "Cresta (CA) Interview Questions" (about 25-day average; SWE project presentation; Senior FDE loop), 2025-2026. https://www.glassdoor.ca/Interview/Cresta-CA-Interview-Questions-E3180909.htm
3. Glassdoor, "Cresta (CA) Interview Experience and Questions" (FDE technical screen: "What is RAG, and how do you prevent hallucination?"), 2026. https://www.glassdoor.com/Interview/Cresta-CA-Interview-Questions-E3180909.htm
4. Glassdoor, "Cresta (CA) Senior Software Engineer Interview Questions" (three LeetCode-medium questions in 70 minutes), 2026. https://www.glassdoor.ca/Interview/Cresta-CA-Senior-Software-Engineer-Interview-Questions-EI_IE3180909.0,9_KO10,34.htm
5. Glassdoor, "Cresta (CA) Software Engineer Interview Questions" (autocomplete, list-to-tree, grid puzzle, SQL joins), 2024-2026. https://www.glassdoor.sg/Interview/Cresta-CA-Software-Engineer-Interview-Questions-EI_IE3180909.0,9_KO10,27.htm
6. Blind, "Cresta interview process", 2024-2025. https://www.teamblind.com/post/cresta-interview-process-jmfnzjoa
7. Blind, "Anyone interview with Cresta lately?" (recruiter email: "key concepts and fundamentals around LLMs, GenAI, RAG, and AI Agents"), 2025. https://www.teamblind.com/post/anyone-interview-with-cresta-lately-zvj0uhq2
8. Blind, "Interview for forward deployment engineer at Cresta" (1.5-hour practical with AI allowed), 2025. https://www.teamblind.com/post/interview-for-forward-deployment-engineer-at-cresta-ngswbj44
9. Blind, "Cresta senior backend SWE interview" (supports an online assessment only; not to be cited for the design round), 2025. https://www.teamblind.com/post/cresta-senior-backend-swe-interview-6v1tmjld
10. PracHub, "Cresta ML System Design Interview Questions" ("Design an Evaluation Platform for AI Agents Across Prompt and Model Changes"), 2026. https://prachub.com/companies/cresta/categories/ml-system-design
11. PracHub, "Cresta Software Engineer Interview Experience: System Design Round: Build an Eval Platform for AI Agents". https://prachub.com/interview-experiences/cresta-software-engineer-interview-experience-system-design-round-build-an-eval-platform-for-ai-agents
12. PracHub, "Cresta Interview Questions" (company page; "Design sequence decoding with greedy and beam search"), 2025. https://prachub.com/companies/cresta
13. PracHub, "Cresta Machine Learning Interview Questions" ("Implement greedy and beam decoding"). https://prachub.com/companies/cresta/categories/machine-learning
14. 1point3acres, "Cresta Machine Learning Engineer Tech Phone Interview Experience" (MLE, AI Agent direction; greedy and beam-search decoding; how an agent generates a sequence), 2025. https://www.1point3acres.com/interview/thread/1140481
15. 1point3acres, "Cresta Fulltime Software Engineer Tech Phone Screen: Job Dispatcher-Executor System Implementation" (title seen in a listing; full text not read), 2026. https://www.1point3acres.com/interview/thread/1172433
16. 1point3acres, "cresta interview questions" (company page; SWE/MLE phone screens and online tests, including the API phone screen), 2024-2026. https://www.1point3acres.com/interview/company/cresta
17. CoderPad, "How Cresta consolidated all technical hiring on CoderPad to scale eng hiring" (case study). https://coderpad.io/case-study/how-cresta-consolidated-all-technical-hiring-on-coderpad-to-scale-eng-hiring/
18. a16z Jobs, "Cresta: Senior Product Engineer" (posting describing the self-paced CoderPad assessment). https://jobs.a16z.com/jobs/cresta/5288477008--senior-product-engineer
19. Cresta on Greenhouse, "Senior Machine Learning Engineer". https://job-boards.greenhouse.io/cresta/jobs/4249943008
20. Cresta on Greenhouse, "Staff Machine Learning Engineer". https://job-boards.greenhouse.io/cresta/jobs/4741283008
21. Cresta on Greenhouse, "Senior Machine Learning Engineer, Voice Experience". https://job-boards.greenhouse.io/cresta/jobs/5155675008
22. Cresta on Greenhouse, "Senior Machine Learning Engineer (ASR)". https://job-boards.greenhouse.io/cresta/jobs/5101011008
23. Built In Vancouver, "Senior Machine Learning Engineer, Cresta" (Agentic Assist track). https://builtinvancouver.org/job/senior-machine-learning-engineer/8619914
24. Cresta on Lever, "Senior Machine Learning Engineer (Chat Agent)". https://jobs.lever.co/cresta/172fe276-9415-4bae-9b0b-af72cf7cf014
25. Cresta Blog, "How Cresta Grades Its Agents: Evaluators in an LLM World". https://cresta.com/blog/how-cresta-grades-its-agents-evaluators-in-an-llm-world
26. Cresta Blog, "Why AI Agent Evaluations Fail, and How the Swiss-Cheese Model Prevails". https://cresta.com/blog/why-ai-agent-evaluations-fail----and-how-the-swiss-cheese-model-prevails
27. Cresta Blog, "Engineering for Real-Time Voice Agent Latency". https://cresta.com/blog/engineering-for-real-time-voice-agent-latency
28. Cresta Blog, "Understanding Cresta's Voice Platform: ML Services, Inference Graphs and Real-Time Intelligence", 2025. https://cresta.com/blog/understanding-crestas-voice-platform-ml-services-inference-graphs-and-real-time-intelligence
29. Cresta Blog, "Understanding Cresta's Voice Platform: The Voice Stack", 2025. https://cresta.com/blog/understanding-crestas-voice-platform-the-voice-stack
30. Cresta Blog, "Introducing Ocean-1: World's First Contact Center Foundation Model", 2023. https://cresta.com/blog/introducing-ocean-1-worlds-first-contact-center-foundation-model
31. Cresta Blog, "AI-Ready Knowledge for Contact Centers: Closing the Gap Between KBs and AI", 2025. https://cresta.com/blog/ai-ready-knowledge-for-contact-centers-closing-the-gap-between-kbs-and-ai
32. Cresta Blog, "Introducing the Next Generation of Cresta Insights: A Complete System for Enterprises to Understand Their CX", 2026. https://cresta.com/blog/introducing-the-next-generation-of-cresta-insights-a-complete-system-for-enterprises-to-understand-their-cx
33. Cresta Blog, "How Fine-Tuned LLMs Power Knowledge Assist, Summarization and Chat Suggestions", 2024. https://cresta.com/blog/how-fine-tuned-llms-power-knowledge-assist-summarization-and-chat-suggestions
34. Fireworks AI Blog, "Story: Cresta Knowledge Assist", 2024. https://www.fireworks.ai/blog/story-cresta-knowledge-assist
35. Fireworks AI Blog, "Multi-LoRA", 2024. https://fireworks.ai/blog/multi-lora
36. Cresta Blog, "Grounding Reality: How Cresta Tackles LLM Hallucinations in Enterprise AI". https://cresta.com/blog/grounding-reality---how-cresta-tackles-llm-hallucinations-in-enterprise-ai
37. Cresta Blog, "Cresta Launches Automated AI Agent Testing: Confidence in Every Conversation". https://cresta.com/blog/cresta-launches-automated-ai-agent-testing-confidence-in-every-conversation
38. Cresta Blog, "Cresta Conductor: The Agent for AI Agent Development". https://cresta.com/blog/cresta-conductor-the-agent-for-ai-agent-development
39. CX Today, "Cresta CEO Ping Wu: Many Contact Centers Misunderstand the Value of LLMs". https://www.cxtoday.com/contact-center/cresta-ceo-ping-wu-many-contact-centers-misunderstand-the-value-of-llms/

*Source note:* All 39 references are carried over from the reviewed reference lists in chapters 01-08, with the same attributions. None was opened for this chapter: a fetch of prachub.com failed DNS lookup, and the web-search budget was exhausted. References 4, 5 and 9 support coding or online-assessment claims in other chapters. They are listed here so this index covers every interview-portal source in the bank, but no system design prompt rests on them.
