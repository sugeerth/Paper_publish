# 14. End-to-End GenAI System Design: Whiteboard Diagrams and Probe Questions for RAG, Agents, Evaluation and Guardrails, and Fine-Tune-to-Serve

> Scope: chapter 07 covers GenAI concepts. This chapter covers GenAI architectures in the format of the design round. It has four whiteboard skeletons and the probe questions an interviewer is likely to ask about each block:
>
> 1. a RAG shopping assistant in the style of Rufus
> 2. an LLM agent that uses tools
> 3. an LLM evaluation and guardrail pipeline
> 4. a fine-tuning-to-serving platform on Bedrock and Nova
>
> Cross-cutting sections cover caching, cost and latency budgets, hallucination monitoring, and the online A/B test plus human-evaluation loop. The diagrams are empty frames for you to fill in out loud. They are not reference answers, and every box is keyed to the question numbers below. Any latency or cost figures you write on them are budgets to defend, not Amazon numbers. The question sections hold questions only, with no answers.
>
> Research note (please read): web search was unavailable for this chapter, both in the first run and in the October 2026 review pass, because the shared search budget was already used up. Direct fetches worked only for GitHub-hosted pages. Fetches to amazon.jobs, aboutamazon.com, aws.amazon.com, 1point3acres, Blind, Glassdoor, Medium, IGotAnOffer, Dataford, OA VO Service and IEEE Spectrum all failed DNS resolution in this environment.
>
> Verification status (October 2026 review pass):
>
> - **Re-fetched and checked line by line:** the GitHub references [5]-[10] and [17]-[20]. Every [Reported] tag that cites them was checked against the page text. Wording that went beyond the source was corrected: question 5 now uses the source title, question 97 no longer calls video ads a GenAI feature, and the interviewer note in the format notes was rewritten. A sub-probe under question 18 duplicated question 36 and was removed.
> - **Newly fetched and added:** references [28]-[35], all on GitHub. They support the new Part 7 (questions 108-134).
> - **Not re-checked:** the non-GitHub references [1]-[4], [11]-[16] and [21]-[27] could not be opened. They are carried over from search extracts gathered for chapters 06, 07, 11 and 12. They are kept and not confirmed dead, but they are not confirmed live either. Re-verify them before relying on any single [Reported] item that cites only them (questions 2, 8, 15, 22, 34, 35 and 48).

## How the [Reported] / [Likely] tag system works

Every question in this guide, including this chapter, carries one of two tags.

- **[Reported] [n]**: a cited source attributes this prompt, or a close paraphrase, to an Amazon interview. The source can be a candidate post on Glassdoor, Blind, 1point3acres or Medium, or a question aggregator that compiles candidate reports. The aggregator pages [5]-[7] say they are built from candidate-reported interviews. They cover Amazon Applied Scientist, Research Scientist and MLE loops together, so the round or role is noted where the source gives it. A follow-up probe under a [Reported] question is not itself reported unless it has its own tag.
- **[Likely]**: a grounded inference from Amazon's GenAI products (Rufus or Alexa for Shopping, Bedrock, Nova), from job postings and Amazon Science or AWS open-source work, or from the standard way GenAI design rounds are probed. Nobody has been seen asking it verbatim. A "(see [n])" after [Likely] points to the product or repository the inference is based on, not to an interview report.
- **"[Reported for the topic]"** means the source names the theme (for example, "alert thresholds and rollback") but not the exact wording.
- Reference [8] marks every one of its questions as synthesized ("not leaked"), so it only ever supports [Likely] items.

What the tags do not mean:

- [Reported] does not mean the question is common.
- [Likely] does not mean it is rare. Block-level probes are rarely written down by candidates.
- No tag was upgraded without a citation.

## Format notes for this round

- **Slot and length.** The round is one hour-long interview in a loop of about three to six, each about 55-60 minutes, and each includes Leadership Principle questions [1][5][6][8]. The aggregators report three to five rounds for scientists [5] and four or five for ML engineers [6]. One 2025-2026 Applied Scientist II loop listed "1 Sys Design (Team-specific)" next to breadth, depth, coding and Bar Raiser rounds [2]. Plan for about 40-45 minutes of design.
- **Name of the round.** For scientists it often appears as the *Science Application* round. It is an open-ended product problem, often paired with Think Big or Invent and Simplify [5]. On ML-engineering-flavoured loops it is called *ML system design*, and the prompt comes from the team's domain: ranking, a chatbot, or an evaluation framework [6].
- **Interviewer and prompt.** Hiring is team by team, so the prompt follows the hiring team's domain: search and ranking, ads, generative-AI post-training or training infrastructure [5][6]. One Glassdoor loop listed the design round as "team-specific" [2]. Generative-AI recruiters ask specifically about post-training and agent experience before the loop [5][6].
- **What is graded.** Reported criteria [5]:
  - framing
  - an evaluation design that combines automatic metrics with a human protocol and reconciles the two
  - retrieval, agents or fine-tuning used with judgment, with their failure modes named
  - launch realism: latency, cost, monitoring, and what ships first

  Strong answers cover the lifecycle through monitoring. One candidate spent much of the hour on alert thresholds and rollback [6]. "There is no single right architecture" [5].
- **L5 bar.** L5 (Applied Scientist II) means "owning a science problem from framing to launch" [5]. Candidates can be rejected when their project scope falls short of the target level, even with strong technical answers [5]. On generative-AI teams, LLM depth is the most common technical reason for rejection [6]. A Blind poster reported being rejected from Amazon Ads after not preparing transformers and LLMs [27].
- **Other loop reports.** Amazon AGI loops for ML roles report an onsite with "ML application/design", and the Nova team reportedly follows the standard Applied Scientist process [9]. IGotAnOffer describes the design round as proposing data sources, an annotation strategy, a modeling technique, evaluation metrics, and trade-offs and risks [12]. Blind threads describe a design or application round as typical for L5 loops [15][16]. A vendor-published debrief says L5 follow-ups go deeper than L4 ones [26]. A Medium write-up describes the Applied Scientist II Science Depth round as resume plus GenAI fundamentals plus business impact [14].
- **Practice advice from the aggregator.** "Practice an agent, a generation-quality evaluation and a ranking system end to end in 40 minutes each, starting with the metric" [5].

---

## Part 0. The universal GenAI whiteboard skeleton

Draw this frame first for any GenAI prompt, then specialize it with Parts 1-4. The numbers in brackets are question numbers in this chapter.

```
+------------------------------------------------------------------------------------+
| FRAME: customer | job-to-be-done | north-star metric | guardrail metrics | budgets  |
|        (p50/p99 latency, cost per 1k requests, max hallucination rate)  [1-4]       |
+------------------------------------------------------------------------------------+

 ONLINE PATH (write a latency budget on every arrow)
 [Client] -> [Input guard] -> [Router / intent] -> [Context builder] -> [LLM] -> [Output guard] -> [Stream to UI]
                 [61]            [9,40]           (RAG / tools / memory)  [79-86]     [62-66]          [87]
                                     |                   |                    ^
                                     v                   v                    |
                               [Cache layer]       [Retriever + reranker] [Prompt / model registry]
                                [88-91]                [10-20]              [76,78]

 OFFLINE / CONTROL PLANE
 [Data + labels] -> [Fine-tune / distill] -> [Offline eval gate] -> [Shadow / canary] -> [Online A/B]
    [72-75]              [73-77]                [51-60]                [69]               [96-100]
          ^                                                                                  |
          +------------------- [Logs, human eval, hallucination monitor] <-------------------+
                                        [67-71, 92-95]
```

1. Before drawing anything, what clarifying questions do you ask about the customer, the surface (app, voice, web), and traffic? [Likely]
   - Which single north-star metric would you commit to, and which two or three guardrail metrics? [Likely]
2. Pick an agentic or GenAI product we could build. Who is the customer, and how do you measure success? [Reported] [2]
   - Why is this a GenAI problem at all, and what would the non-LLM baseline be? [Likely]
3. Propose agentic AI solutions for a hypothetical project and defend their viability. [Reported] (Science Application round, Amazon RS/AS aggregate) [5][6]
   - What would ship first as the smallest launchable version, and what would wait for v2? [Likely]
4. Write your latency and cost budget on the board before choosing a model. How do you split p99 latency across guard, retrieval, rerank, generation and streaming? [Likely]
   - How would the budget change for a voice surface (Alexa) compared with a text surface? [Likely]
5. Measure product-market fit for Alexa Shopping ("Measure PMF for Alexa Shopping"). [Reported] (question title in Amazon question list, undated; role not stated) [7]
   - How would the PMF plan change if Alexa Shopping is an LLM-driven conversational assistant rather than voice ordering? [Likely]
   - Which leading indicators would you trust in the first two weeks, before retention data matures? [Likely]
6. Design ChatGPT. [Reported] (Amazon question list, May 2026; role not stated) [7]
   - What changes if the same system must answer only from Amazon's catalog and reviews? [Likely]

---

## Part 1. RAG shopping assistant (Rufus / Alexa for Shopping style)

Context for [Likely] items: Rufus is Amazon's generative shopping assistant. It answers from catalog, review and community Q&A data, and Amazon describes it as built on several LLMs, including Amazon Nova, on Bedrock [21][22][23]. An earlier chapter's search extract says it was renamed Alexa for Shopping in May 2026 (see chapter 11). Confirm the current name with your recruiter.

### 1.1 Skeleton to draw

```
 Customer query ("What do I need for cold-weather golf?") + session context + locale
        |
        v
 [Query understanding]  intent: product Q&A | discovery | comparison | order status | off-topic   [7-9]
        |  rewrite / decompose
        v
 +--------------------------- RETRIEVAL ----------------------------+
 | [Lexical index]   [Dense ANN / vector store]   [Structured APIs] |    [10-14]
 |  (catalog text)   (catalog, reviews, Q&A)      (price, stock,    |
 |                                                 order, Prime)    |
 +------------------------------+-----------------------------------+
                                | top-K candidates (K = ?)
                                v
                     [Re-ranker (cross-encoder / LTR)]                         [15-17]
                                | top-k passages + product cards
                                v
 [Prompt assembly: system prompt | context | citations | personalization]       [18-20]
                                v
 [LLM generation (model router: small vs large)]  -> streaming                 [21-24]
                                v
 [Grounding / citation check] -> [Answer + product cards + follow-up chips]     [25-28]
                                v
 Logs -> clicks, add-to-cart, thumbs, abandonment -> eval + retraining          [29-33]
```

### 1.2 Framing and scope

7. Design an LLM shopping assistant that answers "What do I need for cold-weather golf?" from the catalog, reviews and community Q&A. [Likely] (see [21][22])
   - Which query types should the assistant decline or hand to classic search instead? [Likely]
   - How do you separate the assistant's success from search's success when both are on the same page? [Likely]
8. Design an enterprise RAG service. [Reported] (Amazon ML onsite) [4]
   - How would per-user permissions (ACLs) be enforced at retrieval time instead of after generation? [Likely] (see [8])
9. Design an AI chatbot architecture covering intent detection, dialogue state, retrieval, generation, fallback and continuous learning ("Design a Reliable AI Chatbot Architecture"). [Reported] (Science Application / ML design, Amazon RS, MLE aggregates; title sighted Aug 2026) [5][6][7]
   - What is the fallback when retrieval returns nothing relevant, and how is that decided? [Reported for the topic: fallback] [5]
   - How does the system learn continuously from production conversations without learning from its own mistakes? [Reported for the topic: continuous learning] [5][6]

### 1.3 Retrieval and vector store

10. What sources would you index (titles, bullets, descriptions, reviews, Q&A, images), and how would you chunk each? [Likely] (see [24])
    - How do you keep the index fresh when price, stock and reviews change every minute? [Likely]
11. Lexical, dense or hybrid retrieval for product questions: when does each fail? [Likely]
    - How would you fuse the scores, and how would you tune the fusion weights? [Likely]
12. Compare a small and a large model from the same text-embedding family for this retriever. What do you gain, and what do you pay? [Reported] (Amazon RS aggregate, breadth round) [5]
    - What are the index memory and re-embedding costs when you upgrade the embedding model? [Likely]
13. How would you choose the ANN index type and its recall/latency settings for hundreds of millions of product vectors? [Likely]
    - How do you measure ANN recall loss against exact search in production? [Likely]
14. How would you fine-tune the embedding model on Amazon shopping data (query-product pairs, hard negatives)? [Likely] (see the ESCI shopping-queries dataset [24])
    - How do you mine hard negatives without pulling in false negatives? [Likely]

### 1.4 Re-ranking

15. What is the difference between a cross-encoder and a bi-encoder, and where does each sit in this pipeline? [Reported] [3]
    - How many candidates can the cross-encoder afford to score within the latency budget? [Likely]
16. Design a search retrieval and ranking pipeline with online experiments and position-bias correction. [Reported] (Amazon MLE/AS aggregate) [6]; a faceted product-search variant appears as "Design Faceted Product Search at Large Scale" [Reported] (Amazon question list, Aug 2026; role not stated) [7]
    - What position bias enters the logged training data when the LLM, rather than a ranked list, presents the products? [Likely]
17. Should the re-ranker optimize relevance to the question, purchase likelihood, or both? How do you avoid an assistant that just pushes sponsored items? [Likely]

### 1.5 Prompt assembly and generation

18. How do you budget the context window across the system prompt, conversation history, retrieved passages and product attributes? [Likely]
    - When do you summarize, retrieve or truncate conversation history? [Likely] (the reported multi-turn search agent prompt is question 36)
19. How do you add personalization (purchase history, preferences) to the prompt without privacy leaks or filter bubbles? [Likely]
20. How do you make the answer cite specific products and reviews so that each claim can be traced? [Likely]
21. How would you route each query between a small and a large model, and what signal triggers escalation? [Likely] (see [21])
    - How do you check offline that the router does not send hard queries to the small model? [Likely]
22. Which decoding settings (greedy, beam, temperature, top-p) would you use for a customer-facing shopping answer, and why? [Reported] (Applied Scientist II, LLM team) [13]
23. The model must answer in 10+ marketplaces and languages. Do you translate the query, retrieve multilingually, or use per-locale models? [Likely]
24. How would you stream the answer while product cards and prices are still being fetched? [Likely]

### 1.6 Grounding, hallucination and failure handling

25. What would you do if the assistant states a wrong price, wrong compatibility or a wrong return policy? [Likely]
    - Which facts must never come from the LLM's parametric memory, and how do you enforce that? [Likely]
26. How would you detect, at serve time, an answer that is not supported by the retrieved context? [Likely]
    - What is the latency cost of a separate groundedness checker, and when would you run it asynchronously instead? [Likely]
27. When should the assistant say "I don't know" or ask a clarifying question instead of answering? [Likely]
28. A seller stuffs reviews or listings with text written to manipulate the assistant (indirect prompt injection). How do you defend against it? [Likely]

### 1.7 Metrics for the RAG assistant

29. Which offline metrics would you track for the retriever, the re-ranker and the generator separately? [Likely] (see Amazon's RAGChecker, which splits retriever and generator diagnostics [17])
    - How do you build the labelled evaluation set, and how do you stop it from going stale? [Likely]
30. How would you evaluate a RAG system automatically on a new domain with no labelled data, for example with generated exams? [Likely] (see Amazon's auto-rag-eval, ICML 2024 [18])
31. Which online metrics show that the assistant helped shopping, rather than just being used? [Likely]
    - How do you attribute a purchase that happened two sessions after an assistant conversation? [Likely]
32. Offline metrics improved but the A/B test fell short. How do you find out why? [Reported] (Amazon RS/MLE aggregates, depth round) [5][6]
33. How would you measure and report the assistant's hallucination rate to leadership every week? [Likely]

---

## Part 2. LLM agent with tool use

### 2.1 Skeleton to draw

```
 User goal ("reorder my usual coffee, but cheaper if possible")
        |
        v
 [Planner LLM] <---- [Memory: session state | user profile | episodic summaries]     [36-38]
        |  plan / next action (structured tool call, JSON schema)
        v
 [Policy + permission gate] -- irreversible? --> [Human confirmation]                [44-46]
        |
        v
 +---------------------- TOOL LAYER ---------------------------+
 | search() | get_product() | compare() | cart.add() | order() |                     [39-42]
 | (timeouts, retries, idempotency keys, rate limits)          |
 +------------------------------+------------------------------+
                                | observation (untrusted text)
                                v
 [Verifier / critic] -- fail --> replan (step budget N, token budget T)              [43,47]
        | pass
        v
 [Final response + action summary]  -> trajectory logs -> agent eval harness         [48-50]
```

### 2.2 Prompts

34. Design an agentic recommendation system for Amazon, covering the component architecture, model choices, and quality and safety evaluation. [Reported] [11]
    - Where in the architecture does the agent add value over the existing recommender, and where does it only add latency? [Likely]
35. Design a GitHub Copilot-style coding agent. [Reported] (Senior AS loop) [3]
    - How would you scope it down for an L5 answer in 40 minutes? [Likely]
36. Design a multi-turn search agent and explain how it handles long conversation context. [Reported] (Science Application, Amazon RS aggregate) [5]
    - What goes into short-term memory, what into long-term memory, and how is stale memory expired? [Likely]
37. Design a browser-operating agent (in the spirit of Nova Act) that completes multi-step web tasks reliably. [Likely] (see [8], synthesized; chapter 11)
38. Should the agent be one LLM with many tools, or a router with specialist sub-agents? What evidence would decide it? [Likely]

### 2.3 Tools, planning and reliability

39. How do you define tool schemas so the model calls them correctly, and how do you validate arguments before execution? [Likely]
40. With 200 available tools, how does the agent pick the right one without stuffing every schema into the prompt? [Likely]
41. A tool times out or returns an error halfway through a multi-step plan. What happens next? [Likely]
    - How do you make tool calls idempotent so that a retry does not place two orders? [Likely] (see [8], synthesized)
42. How would you detect and suppress loops or repeated tool calls within one run? [Likely]
43. How do you set and enforce step, token and wall-clock budgets per task, and what does the user see when the budget runs out? [Likely]
44. Which actions require explicit human confirmation, and how is that decided? [Likely] (see [8], synthesized)
45. Tool outputs can carry prompt-injection text (a product page that says "ignore previous instructions"). How do you isolate untrusted observations? [Likely]
46. How do you enforce that the agent acts only with the user's own permissions, for example on payments and addresses? [Likely]
47. Would you add a verifier or critic model, and how do you stop the verifier from sharing the planner's blind spots? [Likely]

### 2.4 Agent evaluation

48. Evaluate an LLM agent with metrics beyond accuracy: tool-use correctness, hallucination and calibration. [Reported] [3]
    - How do you score a trajectory that reached the right outcome through an unsafe step? [Likely]
49. Agents are stochastic. How do you report reliability (pass@k, pass^k, variance across seeds), and how many runs per task do you need? [Likely]
    - How would you run multi-turn, LLM-driven simulated-user tests in CI before every agent release? [Likely] (see AWS Labs agent-evaluation [19])
50. How would you build a failure taxonomy from production trajectories and turn it into regression tests? [Likely]

---

## Part 3. LLM evaluation and guardrail pipeline

### 3.1 Skeleton to draw

```
                 OFFLINE (release gate)                                ONLINE (every request)
 +-------------------------------------------------+     +------------------------------------------------+
 | Golden set (stratified by intent, locale, risk) |     | [Input guard] PII, jailbreak, toxicity,        |
 |   [52-53]                                       |     |   off-topic, prompt injection         [61]     |
 | Auto metrics: groundedness, correctness,        |     |          |                                     |
 |   citation precision, format, safety  [54-56]   |     |          v                                     |
 | LLM-as-judge (calibrated vs humans)   [57-58]   |     | [Model]                                        |
 | Human eval: SxS preference, rubric    [59-60]   |     |          |                                     |
 | Red-team / adversarial set            [64]      |     |          v                                     |
 |        |                                        |     | [Output guard] grounding, policy, brand        |
 |        v                                        |     |   safety, regulated claims, PII     [62-66]    |
 | GATE: ship / block (thresholds per metric) [51] |     |          |  block / rewrite / fall back        |
 +--------+----------------------------------------+     +----------+-------------------------------------+
          |                                                         |
          v                                                         v
  [Shadow -> canary -> A/B]  [69]          [Sampled logs -> judge + human audit -> dashboards/alerts] [67-71]
```

### 3.2 Evaluation pipeline prompts

51. Design an LLM quality validation system. [Reported] (Amazon RS/MLE aggregates; title in Amazon question list) [5][6][7]
    - What do you measure offline, and which online guardrails do you set? [Reported for the topic] [6]
    - How do automated checks and human review divide the work for generated output? [Reported for the topic] [6]
52. How would you build and maintain a golden evaluation set for a shopping assistant across intents, locales and risk levels? [Likely]
    - How do you stop the golden set from leaking into fine-tuning data? [Likely]
53. Design an evaluation framework that checks whether a text-to-image system spells a brand name correctly and places it naturally. [Reported] (Science Application; Amazon RS/MLE aggregates) [5][6]
    - How would OCR accuracy, embedding-based semantic checks and a human review protocol fit together? [Reported for the topic] [5]
54. How would you measure groundedness (faithfulness to retrieved context) at the claim level? [Likely] (see [17])
55. Which metrics catch a model that is fluent and confident but wrong? [Likely]
56. How do you evaluate format and instruction-following, for example JSON for product cards, length limits and tone? [Likely]
57. Would you use an LLM as a judge? How do you calibrate it against human labels and detect its biases (position, verbosity, self-preference)? [Likely]
    - How often must the judge be re-validated when the judged model or the judge model changes? [Likely]
58. When is perplexity misleading for comparing checkpoints, and how do pairwise preference evaluations (win rate, Elo or Bradley-Terry) complement it? [Reported for the topic] (Amazon AGI candidate summaries; see chapter 07, [8]); exact wording [Likely]
59. Design the human-evaluation protocol: rubric, side-by-side preference, number of raters, agreement and cost. [Likely]
    - How do you size the human-eval sample to detect a 2-point win-rate difference? [Likely]
60. Offline judge scores say model B is better, but human raters prefer model A. What do you do? [Likely]

### 3.3 Guardrails

61. What does the input guard check (PII, jailbreak, toxicity, off-topic, injection), and should it be a classifier, rules, or an LLM? [Likely] (see [20], the responsible-AI and securing-RAG examples)
    - How do you trade false blocks (lost customers) against misses (safety incidents), and who sets the thresholds? [Likely]
62. What does the output guard check before an answer reaches a customer: grounding, policy, regulated claims (health, safety), competitor or brand issues? [Likely]
63. When the output guard trips, do you block, rewrite, regenerate or fall back to a template? What does each cost in latency? [Likely]
64. How would you red-team the assistant before launch, and how do red-team findings become regression tests? [Likely]
65. How do you run guardrails on a streamed response, where tokens reach the user before the full answer exists? [Likely]
66. Should guardrails be configurable per tenant or per marketplace (as in a managed service such as Bedrock), and how do you test each configuration? [Likely] (see [8], synthesized; [25])

### 3.4 Hallucination monitoring in production

67. Design a production hallucination monitor: what do you sample, who or what labels it, and what triggers an alert? [Likely]
    - How would you set alert thresholds and a rollback rule for a quality metric that is noisy day to day? [Reported for the topic: alert thresholds and rollback] [6]
68. Which cheap online proxies (thumbs-down, rephrase, abandonment, returns) correlate with hallucination, and how would you validate them? [Likely]
69. Walk through shadow mode, canary and full A/B for a new model version. What has to be true to move from each stage to the next? [Likely] (see [8], synthesized)
70. Catalog data changed, and the assistant now confidently cites a discontinued product. How would monitoring catch it? [Likely]
71. How do you detect drift in the query mix (new seasonal intents, new product categories) that makes the offline evaluation set unrepresentative? [Likely]

---

## Part 4. Fine-tuning to serving infrastructure (Bedrock / Nova)

### 4.1 Skeleton to draw

```
 DATA                         TRAIN                              GATE                SERVE
 [Logs, human labels,] -> [SFT / PEFT (LoRA)] -> [Pref. tuning] -> [Offline eval] -> [Model registry]
 [synthetic data (LLM)]    on Nova / open base     (DPO / RLHF /       + safety          |
   [72-74]                   [75-77]                 GRPO) [77]         [51-60]          v
                                                                              [Distill / quantize] [83]
                                                                                         |
                                                                                         v
 [Request] -> [Gateway: auth, quota, routing] -> [Inference fleet: continuous batching,    ] -> [Response]
                    [80, 84]                       [paged KV cache, prefix cache, spec.   ]
                                                   [decoding; GPU / Inferentia]  [79-82]
                                                          |
                                         [Autoscaler on queue depth / tokens/s]  [84-86]
```

### 4.2 Data and training

72. For a shopping assistant, when do you fine-tune, when do you use RAG, and when is prompting enough? What evidence would decide? [Likely]
73. How would you build the SFT dataset: which sources, how labelled, how deduplicated, and how checked for quality? [Likely]
    - Design a distributed pipeline that ingests, tokenizes, deduplicates and filters training data. [Reported] (Amazon MLE aggregate, staff loop) [6]
74. How would you use a large model to generate or label training data for a smaller one, and how would you control label noise? [Likely]
75. Explain the main parameter-efficient fine-tuning techniques and when each beats full fine-tuning. [Reported] (online assessment, Amazon RS/MLE aggregates) [5][6]
    - When would you serve many LoRA adapters on one base model, and what does that do to batching? [Likely]
76. How would you customize a Nova model on Bedrock compared with self-hosting an open-weights model on SageMaker? What drives the choice? [Likely] (see [20][25])
77. Why does GRPO remove the critic, what does that cost, and how would you design the reward for the assistant? [Reported] (phone screen, Amazon RS aggregate) [5]
    - How would you parallelize GRPO training? [Reported] (ML depth round, Amazon MLE aggregate) [6]
    - How do you design a preference dataset or reward from customer feedback without rewarding sycophancy or upselling? [Likely]
78. How do you version prompts, adapters, retrieval indexes and guardrail configs together so that a rollback is consistent? [Likely]

### 4.3 Serving, latency and cost

79. Describe how an inference engine such as vLLM raises serving throughput for large language models. [Reported] (online assessment, Amazon RS/MLE aggregates) [5][6]
    - How do continuous batching and a paged KV cache interact with p99 time-to-first-token? [Likely]
80. For a residual MLP block in BF16, is it compute-bound or memory-bound at batch size 1 and at batch size 256? Do the analysis from bandwidth and peak throughput. [Reported] (manager screen, Amazon RS/MLE aggregates; title sighted Aug 2026) [5][6][7]
    - What does this imply for the batch size you would serve the assistant at? [Likely]
81. Break the end-to-end latency into queue time, time-to-first-token and per-token time. Which would you optimize first for a chat surface, and which for voice? [Likely] (see [8], synthesized)
82. When would you use speculative or parallel decoding, and how do you check that it does not change output quality? [Likely]
83. You must cut serving cost by 50% with no measurable quality loss. Walk through distillation, quantization, routing, caching and batch inference in the order you would try them. [Likely] (see [8], synthesized; chapter 11)
    - How do you prove that quality did not regress after each cost change? [Likely]
84. Prime Day brings 10x traffic. How do you plan capacity (provisioned versus on-demand), and what degrades gracefully first? [Likely]
85. Design a multi-tenant inference platform: control plane versus data plane, per-tenant throttling, and noisy-neighbor isolation. [Likely] (see [8], synthesized)
86. The p99 latency doubled after a deploy. How do you triage it? [Likely] (see [8], synthesized)
87. How do you stream tokens to the client, and how do you stop upstream generation quickly when the user leaves? [Likely]

### 4.4 Caching

88. Where would you cache in this system: query-to-answer, retrieval results, embeddings, prompt prefixes, KV prefixes? What does each save? [Likely]
89. How would you design a semantic cache for shopping questions, and how do you stop it from serving a stale price or another user's personalized answer? [Likely] (see [8], synthesized)
    - How do you measure the cache's hit rate and its error rate? [Likely]
90. How does prefix caching relate to the KV cache, and how would you lay out the prompt to maximize prefix reuse? [Likely] (see [8], synthesized)
91. What is the cross-tenant leakage risk of prompt caching in a shared service, and how do you prevent it? [Likely] (see [8], synthesized)

---

## Part 5. Online A/B and human-evaluation loop (closing the system)

```
 [New model / prompt / index] -> [Offline gate] -> [Human SxS eval] -> [Shadow] -> [A/B: treatment vs control]
                                                                                     |
     [Decision memo: ship / iterate / kill]  <- [Guardrail check] <- [Metric readout] <-+
                    |
                    +-> [Failure mining -> new golden items + training data]  (loop back)
```

92. Which logged signals (explicit feedback, rephrases, add-to-cart, returns) would you collect for evaluation and retraining, and how do you de-bias them? [Likely]
93. How do you sample production conversations for human review: uniform, risk-weighted or uncertainty-weighted? [Likely]
94. How do you keep human raters consistent across marketplaces and languages, and how do you audit rater quality? [Likely]
95. How do you stop the feedback loop from training the model on its own hallucinations? [Likely]
96. Design the A/B test for a new assistant model: unit of randomization, primary metric, guardrails, duration and power. [Likely]
    - Conversations span sessions and users share devices. How do you handle interference and carry-over effects? [Likely]
97. How would you evaluate adding video ads? [Reported] (Amazon question list, Nov 2025; role not stated) [7]
    - How would the evaluation change if the feature added to the page were an LLM-generated element, such as an AI review summary, instead of video ads? [Likely]
98. The treatment lifts engagement with the assistant but lowers conversion on the page. Ship or not? [Likely]
99. How would you use interleaving or counterfactual (off-policy) evaluation to compare rankers or answer strategies faster than an A/B test? [Likely]
100. Write the one-paragraph launch recommendation you would send to a director after the test. Which numbers must it contain? [Likely]

---

## Part 6. L5-level cross-cutting probes (any of the four designs)

101. Which part of this design is the riskiest science bet, and how would you de-risk it in the first month? [Likely]
102. What would you build if you had one-tenth of the GPU budget? [Likely]
103. Which component would you buy (Bedrock managed features) and which would you build, and why? [Likely] (see [25])
104. Which component fails silently, and how would you find out? [Likely]
105. How would this design change for Alexa voice, where there is no screen for product cards or citations? [Likely]
106. Design Amazon's recommendation system, then explain where an LLM would and would not improve it. [Reported for the base prompt] (Amazon system-design list, SDE-leaning aggregate) [10]; the LLM extension is [Likely]
107. Walk me through the GenAI system you built that is closest to this one. What would you change now? [Likely]
    - Explain an AI-agent project you built and how it was adopted. [Reported] (Amazon MLE aggregate, project deep dive) [6]

---

## Part 7. Additional probes (October 2026 review pass)

These questions were added in the review pass. They are grouped by the part of the design they extend, and numbering continues from Part 6 so that the question numbers on the diagrams stay valid. A [Reported] tag here cites a question title or sample question on a fetched GitHub aggregator page. Where only the base prompt is reported, the GenAI extension is tagged [Likely] separately.

### 7.1 Extends Part 1 (RAG shopping assistant)

108. Design a RAG system. [Reported] (Amazon question list, page 2; System Design; undated; role not stated) [29]
     - Walk the board from ingestion and chunking through retrieval, re-ranking, generation and evaluation in 40 minutes. Which block would you spend the most time on for a shopping use case? [Likely]
109. Compare approaches to building joint image-text embeddings. [Reported] (online assessment, Amazon RS aggregate) [5]
     - A customer uploads a photo of a broken part and asks "what replaces this?" How would you add image queries to the retriever in Part 1? [Likely]
110. Compare a dual-encoder model with a vision-language model that uses a resampler. [Reported] (science breadth round, Amazon RS aggregate) [5]
     - Which of the two would you use for multimodal retrieval, and which for answering a question about a product image? [Likely]
111. Using the Exact, Substitute, Complement and Irrelevant labels from the Shopping Queries dataset, how would you get the assistant to recommend a substitute when the exact item is out of stock, and how would you evaluate those recommendations? [Likely] (see [24])
112. Design typeahead suggestions. [Reported] (multi-company list that includes Amazon, April 2026; role not stated) [30]
     - How would you generate the assistant's follow-up question chips with an LLM within the same latency budget as typeahead? [Likely]
113. When would a long-context model replace retrieval for a product-comparison query, and how would you test whether facts in the middle of a long context are lost? [Likely]
114. Online review content moderation: flag reviews that should not be published. [Reported] (title "Online Review Content Moderation", Amazon question list, page 2; listed as an AI-coding task, not a design round; undated) [29]; a related design title, "Design an end-to-end spam detection system", appears on page 1 of the list [Reported] [7]
     - How would you extend the moderation system to catch reviews written to manipulate an LLM assistant that retrieves them (see question 28)? [Likely]

### 7.2 Extends Part 2 (LLM agent with tool use)

115. An intent classifier sends each turn to a specialist agent (product Q&A, orders, returns). How do you keep conversation context consistent across agents, and how do you measure routing errors? [Likely] (see the classifier-routed orchestrator in [33])
116. A model-driven agent loop or an explicit workflow graph: when do you let the model plan freely, and when do you hard-code the steps? [Likely] (see [35])
117. Which agent concerns would you hand to a managed runtime (memory, identity, a tool gateway, policy, tracing), and which would you keep in your own code? [Likely] (see Bedrock AgentCore [34])
118. You expose existing services, such as order history and returns, to the agent as tools through a protocol such as MCP. How do you version tool contracts so that a schema change does not break agents already in production? [Likely] (see [34])
119. What would you record in an agent trace (spans for each LLM call and tool call) to debug a failed trajectory, and how would you sample traces at Amazon scale? [Likely] (see [34])
120. Walk through a concrete AI-assisted engineering workflow: what you delegated, what context you protected, and how you verified the output. [Reported] (new-grad SDE screens, Aug 2026; not an Applied Scientist report) [10]
     - If you were designing the Copilot-style coding agent from question 35, how would you measure whether its users verify its output? [Likely]

### 7.3 Extends Part 3 (evaluation, guardrails and hallucination monitoring)

121. Would a hallucination checker extract claims as sentences, sub-sentences or knowledge triplets? How does the granularity change precision, recall and cost? [Likely] (see Amazon's RefChecker [31])
     - How does checking differ when the reference is accurate retrieved context, noisy context, or no context at all? [Likely] (see [31])
122. How would you test semantic robustness, that is, whether the answer changes when the query has typos, a paraphrase or different casing? [Likely] (see [32])
123. How would you measure stereotyping or toxicity across demographic groups in shopping answers before launch, and which result would block the launch? [Likely] (see [32])
124. How would you run evaluators online on sampled production agent sessions, and how would you keep the evaluators' cost within budget? [Likely] (see the online evaluation in [34])
125. Design a metrics system. [Reported] (Amazon question list, page 2; System Design, Hard; undated; role not stated) [29]
     - Which per-request fields (model version, prompt version, retrieved document IDs, guard verdicts, tokens, latency) must the metrics system store so that a hallucination spike can be traced to one change? [Likely]

### 7.4 Extends Part 4 (fine-tuning to serving)

126. Explain parallelism and the collective operations used in training. [Reported] (Amazon question list; System Design, Hard; undated) [7]
     - Which parallelism would you use to fine-tune a 70B-parameter model for the assistant on a fixed GPU budget, and what does each choice cost in communication? [Likely]
127. Calculate the memory size for a given model structure. [Reported] (title "Memory Size Calculation for DDN Model Structure", Amazon question list, page 2; AI Coding) [29]
     - Estimate the serving memory for the assistant: weights, KV cache per request at your context length, and activations. How many concurrent requests fit on one accelerator? [Likely] (the aggregator advises being able to explain "the memory arithmetic behind LoRA, quantization and KV caching" [6])
128. The base model you fine-tuned is deprecated or its roadmap narrows. How would you design for model portability across adapters, prompts and evaluation suites? [Likely] (see the reported Nova roadmap note in [8])

### 7.5 Extends Parts 5 and 6 (A/B loop and cross-cutting probes)

129. Design a causal study for the impact of reminders. [Reported] (Amazon question list; System Design, Hard; undated) [7]
     - The assistant sends a proactive message ("the price on your saved item dropped"). How would you estimate its causal effect on purchases when randomization is limited? [Likely]
130. Predict click probability for music recommendations. [Reported] (Amazon question list, Sep 2026; role not stated) [7]
     - How would you add a conversational LLM layer on top of this recommender, and how would you evaluate whether it beats the click model alone? [Likely]
131. Diagnose and fix an underperforming ML model. [Reported] (title linked from the Amazon RS and MLE aggregates) [5][6]
     - The assistant's weekly groundedness score fell three points with no deploy. Walk through your diagnosis across data, retrieval index, model and guardrails. [Likely]
132. Tell me about a time generative AI gave you a wrong answer. How did you catch it, and what did you change? [Reported] (described as close to standard across Amazon loops since 2025) [28]
133. Walk me through your post-training or agent experience. [Reported for the topic] (recruiter screens for generative-AI teams) [5][6]
     - Which of those projects would you present as an L5 example of owning a science problem from framing to launch? [Likely]
134. Your hiring team owns a classic ranking or ads system. Where would you add an LLM first, and what evidence would make you remove it? [Likely]

---

## References

Entries marked "fetched" were read in this chapter's first run and re-fetched and checked in the October 2026 review pass. Entries marked "fetched (review pass)" were first read in that pass. Entries marked "carried over" come from earlier chapters of this guide. They were not re-read, because their hosts failed DNS resolution in this environment, so they are unverified rather than confirmed dead.

1. (carried over) Applied Scientist Interview Prep, amazon.jobs (Amazon, official). https://amazon.jobs/content/en/how-we-hire/applied-scientist-interview-prep
2. (carried over) Amazon Applied Scientist II Interview Experience & Questions, Glassdoor, 2025-2026 (loop with "1 Sys Design (Team-specific)"; agentic-product prompt). https://www.glassdoor.com/Interview/Amazon-Applied-Scientist-II-Interview-Questions-EI_IE6036.0,6_KO7,27.htm
3. (carried over) Amazon Applied Scientist Interview Experience & Questions, Glassdoor, 2025-2026 (bi-encoder vs cross-encoder; Copilot-style agent in a Senior AS loop; LLM-agent evaluation beyond accuracy). https://www.glassdoor.com/Interview/Amazon-Applied-Scientist-Interview-Questions-EI_IE6036.0,6_KO7,24.htm
4. (carried over) "Amazon Onsite Interview Experience for Fulltime Machine Learning Role" (enterprise RAG design), 1point3acres. https://www.1point3acres.com/interview/thread/1157086
5. (fetched) "Amazon Research Scientist Interview Process", kevin-2023-code/Tech-Interview-Questions, GitHub. An aggregator adapted from trueinterview.io. It covers RS and AS loops and is based on candidate-reported interviews. Its Science Application section lists the agentic proposal, multi-turn search agent, AI chatbot and brand-name evaluation prompts, plus the grading criteria and the L5 "framing to launch" definition. Fetched October 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/amazon/interview-process-research-scientist.md
6. (fetched) "Amazon Machine Learning Engineer Interview Process", kevin-2023-code/Tech-Interview-Questions, GitHub. An aggregator. It covers the ML design round (ranking with position bias, chatbot with fallback and continuous learning, evaluation frameworks), the alert-threshold and rollback anecdote, the vLLM and PEFT online-assessment items, the staff data-pipeline prompt, and "LLM depth is the most common technical reason for rejection on generative-AI teams". Fetched October 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/amazon/interview-process-machine-learning-engineer.md
7. (fetched) "Amazon interview process, OA & interview questions" (company question list with formats and sighting dates). Titles include "Design a Reliable AI Chatbot Architecture" (Aug 2026), "Design ChatGPT" (May 2026), "Design Faceted Product Search at Large Scale" (Aug 2026), "MLP Compute-vs-Memory Bound Analysis" (Aug 2026), "How would you evaluate adding video ads?" (Nov 2025), "Design an LLM quality validation system" and "Measure PMF for Alexa Shopping". kevin-2023-code/Tech-Interview-Questions, GitHub. Fetched October 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/amazon.md
8. (fetched) "amazon.md", ombharatiya/AI-Engineer-Interview-Questions, GitHub. Its representative questions are explicitly synthesized from public focus areas, "not leaked questions", so they are used only for [Likely] items. Its loop notes summarize candidate reports: 4-6 rounds of about 60 minutes, and AGI-org questions on evaluation "including perplexity and preference-based evals". Its context notes report that Amazon Q Business closed to new customers at the end of July 2026 in favour of Amazon Quick, and that the Nova model roadmap narrowed in 2026. Fetched October 2026. https://github.com/ombharatiya/AI-Engineer-Interview-Questions/blob/main/14-company-interview-questions/amazon.md
9. (fetched) "AI Companies Interview Questions" (Amazon AGI section: reported loop with "ML application/design" in the onsite; Nova team follows the standard AS process), ombharatiya/FAANG-Coding-Interview-Questions, GitHub. Fetched October 2026. https://github.com/ombharatiya/FAANG-Coding-Interview-Questions/blob/main/AI-Companies-Interview-Questions.md
10. (fetched) "FAANG Recent Questions" (Amazon section: system-design list including "Design Amazon's recommendation system"; 2026 AI-assisted OA trend; August 2026 new-grad screens that asked candidates to walk through an AI-assisted engineering workflow), ombharatiya/FAANG-Coding-Interview-Questions, GitHub. Fetched October 2026. https://github.com/ombharatiya/FAANG-Coding-Interview-Questions/blob/main/FAANG-Recent-Questions.md
11. (carried over) "Amazon Applied Scientist Interview Questions & Guide 2026", Dataford (prep vendor; agentic recommendation system design), 2026. https://dataford.io/interview-guides/amazon/applied-scientist
12. (carried over) "Amazon Applied Scientist Interview", IGotAnOffer (prep vendor; design round covers data sources, annotation strategy, modeling, metrics, trade-offs and risks), 2025-2026. https://igotanoffer.com/en/advice/amazon-applied-scientist-interview
13. (carried over) Amazon interview review RVW96068026 (Applied Scientist II, LLM team, Santa Clara; decoding choices for an Amazon use case), Glassdoor, 2025. https://www.glassdoor.com.hk/Interview/Amazon-Interview-E6036-RVW96068026.htm
14. (carried over) Chaitanya Mundle, "My interview experience for the Applied Scientist II role at Amazon", Medium, circa 2024. https://medium.com/@mundle1997.cm/my-interview-experience-for-the-applied-scientist-ii-role-at-amazon-738aba21f611
15. (carried over) "Amazon Applied Scientist (L5) Virtual Onsite", Blind, circa 2025. https://www.teamblind.com/post/amazon-applied-scientist-l5-virtual-onsite-navwesml
16. (carried over) "Amazon applied scientist interview loops", Blind (design or application round typical at L5). https://www.teamblind.com/post/amazon-applied-scientist-interview-loops-lucdk0uq
17. (fetched) RAGChecker: a fine-grained framework for diagnosing retrieval-augmented generation (claim-level retriever and generator metrics), amazon-science, GitHub; paper arXiv:2408.08067, NeurIPS Datasets and Benchmarks, 2024. Fetched October 2026. https://github.com/amazon-science/RAGChecker
18. (fetched) auto-rag-eval: "Automated Evaluation of Retrieval-Augmented Language Models with Task-Specific Exam Generation", amazon-science, GitHub; ICML 2024. Fetched October 2026. https://github.com/amazon-science/auto-rag-eval
19. (fetched) Agent Evaluation (LLM-driven multi-turn testing of virtual agents; targets include Amazon Bedrock, Amazon Q Business and SageMaker), awslabs, GitHub. Fetched October 2026. https://github.com/awslabs/agent-evaluation
20. (fetched) Amazon Bedrock samples (agents and function calling, RAG and knowledge bases, evaluation and observability, custom models, model-latency benchmarking, responsible AI, securing RAG apps, cost reporting), aws-samples, GitHub. Fetched October 2026. https://github.com/aws-samples/amazon-bedrock-samples
21. (carried over) "Amazon Rufus AI assistant personalized shopping features", About Amazon (LLMs including Amazon Nova on Bedrock). https://aboutamazon.com/news/retail/amazon-rufus-ai-assistant-personalized-shopping-features
22. (carried over) "What do I need for cold weather golf?" (Amazon Rufus), IEEE Spectrum. https://spectrum.ieee.org/amazon-rufus
23. (carried over) Applied Scientist, Rufus Experiences Science (Job ID 2855826), amazon.jobs (posting may have expired). https://amazon.jobs/en/jobs/2855826/applied-scientist-rufus-experiences-science
24. (carried over) Shopping Queries Dataset (ESCI), amazon-science, GitHub. https://github.com/amazon-science/esci-data
25. (carried over) Amazon Bedrock (product page), AWS. Cited only to name the managed service, not as an interview report. https://aws.amazon.com/bedrock/
26. (carried over) "Amazon Applied Scientist Interview Debrief: Five-Round Loop + LP", OA VO Service (vendor-published candidate debrief, promotional; L5 vs L4 follow-up depth). https://oavoservice.com/en/articles/amazon-applied-scientist-vo-loop-lp-ml-coding
27. (carried over) "Applied scientist interview at Amazon Ads", Blind (rejected after not preparing attention, transformers and LLMs). https://www.teamblind.com/post/applied-scientist-interview-at-amazon-ads-zoerec5t
28. (fetched, review pass) "Amazon interview process & questions", kevin-2023-code/Tech-Interview-Questions, GitHub (aggregator adapted from trueinterview.io, CC BY 4.0; "based on 137 candidate-reported interview experiences"). It says that since 2025 "a question about a time generative AI gave you a wrong answer has become close to standard", and that ML and science tracks get ML design or science depth rounds. Fetched October 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/amazon/README.md
29. (fetched, review pass) "Amazon interview process, OA & interview questions", page 2, kevin-2023-code/Tech-Interview-Questions, GitHub (aggregator). Titles include "Design a RAG System" (System Design, Easy), "Online Review Content Moderation" (AI Coding), "Memory Size Calculation for DDN Model Structure" (AI Coding) and "Design Metrics System" (System Design, Hard); none is dated. Fetched October 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/amazon-2.md
30. (fetched, review pass) "Reported in Apr 2026", kevin-2023-code/Tech-Interview-Questions, GitHub (aggregator). It lists "Design Typehead Suggestion" (sic) under LinkedIn / Amazon / Google / Meta / Microsoft AI / Perplexity / Pinterest, and "Search / Ranking / Experimentation" under Amazon / Pinterest. Fetched October 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/by-month/2026-04.md
31. (fetched, review pass) RefChecker: fine-grained hallucination detection (claim extractor, checker and aggregation rules; claims as knowledge triplets; zero-context, noisy-context and accurate-context settings; human-annotated benchmark), amazon-science, GitHub. Fetched October 2026. https://github.com/amazon-science/RefChecker
32. (fetched, review pass) fmeval: Foundation Model Evaluations Library (accuracy, toxicity, semantic robustness and prompt-stereotyping evaluations across open-ended generation, summarization, question answering and classification), aws, GitHub. Fetched October 2026. https://github.com/aws/fmeval
33. (fetched, review pass) Agent Squad (formerly Multi-Agent Orchestrator): a classifier routes each query to a specialist agent and keeps conversation context across agents, awslabs, GitHub. Both the old and the new repository paths resolve to the same README. Fetched October 2026. https://github.com/awslabs/agent-squad
34. (fetched, review pass) Amazon Bedrock AgentCore samples (Runtime, Gateway that turns APIs into MCP-compatible tools, Identity, Memory, Observability with OpenTelemetry, built-in and custom Evaluation for on-demand and online evaluation, Cedar-based Policy), awslabs, GitHub. Fetched October 2026. https://github.com/awslabs/amazon-bedrock-agentcore-samples
35. (fetched, review pass) Strands Agents SDK: "a model-driven approach to building AI agents", strands-agents, GitHub. Fetched October 2026. https://github.com/strands-agents/sdk-python
