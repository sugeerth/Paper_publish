# 07. LLM and GenAI Questions: Transformers, Fine-Tuning, RLHF/DPO, RAG, Evaluation, Hallucination, Efficiency, and Amazon Product Context (Bedrock, Nova, Rufus, Alexa+)

> Scope: the LLM and GenAI questions an L5 Applied Scientist (Applied Scientist II) candidate should expect across the 2025-2026 onsite loop. This chapter is questions only, with no answers.
>
> Research note (please read): this chapter draws on three kinds of evidence:
> - **Candidate-report findings gathered for chapters 01-06 of this guide.** They came from search-result extracts of the cited pages, not full reads. Each [Reported] tag cites the original page.
> - **GitHub-hosted interview aggregators, fetched and read directly** [22][23][24][25][26][35][36]. They compile Amazon candidate reports. Most cite trueinterview.io (CC BY 4.0); some do not name their sources. They are labeled as aggregators wherever used.
> - **Amazon job postings and product pages,** used only to ground [Likely] questions.
>
> Verification status (October 2026 review pass). The shared web-search budget was used up, and the egress proxy refused connections (HTTP 403) to every non-GitHub host this pass tried: Glassdoor, Blind, LeetCode, 1point3acres, Medium, amazon.jobs, amazon.science, aboutamazon.com, IEEE Spectrum, IGotAnOffer, Dataford, Interview Kickstart, Enginebogie and OA VO Service.
> - **Re-fetched and checked line by line:** the seven GitHub aggregator pages [22][23][24][25][26][35][36].
> - **Tags fixed after that check:** questions whose tags those pages did not support were retagged [Likely] or relabeled. That covers transformers replacing RNNs, mixture-of-experts specifics, pretraining vs fine-tuning, LLM production trade-offs, the content-moderation attribution, and several round labels.
> - **Not re-checked:** the non-GitHub references [1]-[21] and [27]-[34] could not be re-opened. They are kept as captured from earlier search extracts, not confirmed dead. Re-verify them before relying on any single item.

## Format notes for this round

- **There is usually no separate "LLM round."** LLM and GenAI questions come up inside the standard loop seats: ML/Science Breadth, Science Depth (your own project), Science Application / ML System Design, and sometimes ML Coding.
  - Amazon's official page describes a loop of about four 55-minute interviews with members of the science community. Each interviewer is assigned technical competencies and Leadership Principles and typically asks 2-3 behavioral questions [1].
  - Candidate reports for AS II more often describe five rounds of 55-60 minutes, for example "1 Science ML Breadth, 1 Science Depth, 1 Sys Design (Team-specific), 1 coding round and 1 Bar Raiser" [2].
  - An aggregator built from about 20 Amazon science reports (mostly 2026) says generative-AI teams ask about LoRA, RLHF, GRPO, KV caching and distributed training. It also describes an optional online assessment covering PEFT, vLLM serving and image-text embeddings [22].
  - On generative-AI teams, "LLM depth is the most common technical reason for rejection" [23].
- **Transformers are now baseline breadth, even off-LLM teams.** An L5 virtual-onsite thread on Blind treats transformers as core breadth [13]. An Amazon Ads candidate reported being rejected after not preparing for attention, transformers and LLMs [14]. 2025-2026 reports cover:
  - decoder-only inference and KV caching;
  - Longformer-style long-context attention;
  - bi-encoders vs cross-encoders;
  - LoRA;
  - BERT vs GPT;
  - DPO vs RLHF;
  - beam search;
  - "modern LLM training techniques" [3][4][7][8][10][11].
- **Team-dependent depth.**
  - AGI/Nova-team loops are reported to lean hard on modern LLM training and ML coding. One aggregator reports a "debug a broken Transformer implementation" coding problem, and candidates compare the ML-coding style to OpenAI's [10][15][17][25]. Another aggregator says AGI candidate reports cover training techniques, evaluation (including perplexity and preference-based evals) and data quality [26].
  - LLM-team and Rufus/Alexa loops tie decoding, training and evaluation to an Amazon use case [8][27][28].
  - Science-application prompts in 2025-2026 include agentic products, agentic recommenders, enterprise RAG, LLM-agent evaluation and LLM quality validation [2][3][16][21][22].
- **Interviewer type.** Usually Applied Scientists, Senior Applied Scientists or Principal Applied Scientists. The application round is often run by the hiring manager [1][2].
- **The L5 bar.** L5 answers are expected to hold up through two or three layers of follow-up, against one layer at L4 [19]. Blind commenters say L5 candidates are vetted harder on whether they "raise the bar" [34]. For LLM topics that means:
  - deriving the mechanics (attention, the DPO loss, KV-cache memory);
  - naming failure modes and trade-offs;
  - choosing between prompting, RAG, fine-tuning and preference tuning for a concrete Amazon problem;
  - defining how you would evaluate it offline and online, including the cost and latency budget.
- **Leadership Principles still apply.** Expect 1-2 LP questions inside each technical round [1][19]. A GenAI-flavored behavioral prompt is now reported in Amazon screens: how you use AI tools, and a time one gave you a wrong answer [24].

**Tag legend.**
- **[Reported]** means a cited source attributes the question, or a close paraphrase, to an Amazon science interview. That is Applied Scientist unless noted. Items from the Research Scientist or MLE aggregators are marked "(Amazon RS aggregate)" or "(Amazon MLE aggregate)".
- **[Likely]** means a grounded inference from the role, Amazon's products or job postings, or adjacent reported questions. It has not been seen verbatim.

---

## A. Transformer fundamentals

### A1. Architecture and attention

1. Explain the Transformer architecture in detail: encoder, decoder, and what each component does. [Reported] [4][9][11]
   - Why scale the dot product by sqrt(d_k)? What happens to the softmax if you don't?
   - Why multiple heads instead of one wide head?
   - Where are the residual connections and layer norms, and why does pre-norm vs post-norm matter for training stability?
2. Explain the attention mechanism "at root level," including scaled dot-product attention, multi-head attention and positional encoding. [Reported] [5][7]
   - Write the attention equation with shapes for Q, K and V.
   - How does the causal mask enter the equation?
3. What happens if you set Wq = Wk = Wv in a Transformer? [Reported] [6]
   - What does that do to the attention matrix's symmetry, and what expressiveness do you lose?
4. Explain the difference between LSTMs and Transformers mathematically. [Reported] [12][37]
   - How does each handle long-range dependencies, and what are the parallelism and compute trade-offs?
   - Why did transformers replace RNNs for language modeling? [Likely] (an Amazon-focused question list that says it is synthesized, not reported [26])
5. Why do Transformers need positional information? Compare sinusoidal, learned, RoPE and ALiBi encodings. [Reported] (Amazon RS aggregate: RoPE and ALiBi in a generative-AI phone screen) [22]
   - Write the RoPE formula and show how it makes attention depend only on relative position. [Reported] (Amazon RS and MLE aggregates) [22][23]
   - How does RoPE behave when you extrapolate past the training context length? [Likely]
6. Compare multi-head and grouped-query attention, and extend the comparison to multi-query attention. [Reported] (Amazon MLE aggregate, Ads-team phone screen) [23]
   - How does each change KV-cache memory and quality?
7. Transformer internals: how does the feed-forward block contribute compared with attention, and where do most of the parameters and FLOPs go? [Reported] (Amazon RS aggregate, generative-AI breadth hour) [22]
   - Where do the parameter count and FLOPs come from as d_model, the number of layers and the sequence length change?
8. Contrast Adam and AdamW. Why does AdamW decouple weight decay, and why does that matter for training transformers? [Reported] (Amazon RS and MLE aggregates) [22][23]
9. You are handed a Transformer implementation that trains but performs badly. Debug it. [Reported] (Amazon AGI, aggregator) [25]
   - What bugs would you check for first (mask direction, softmax axis, scaling, positional-encoding offsets, layer-norm placement)?
10. What is latent attention (low-rank compression of keys and values), and how does it cut KV-cache memory compared with GQA? [Reported] (Amazon MLE aggregate: "latent attention" listed among the most specialized training-at-scale screens) [23]
    - What does it cost in compute or quality, and how does it interact with RoPE? [Likely]

### A2. Model families

11. How do BERT and GPT differ in objectives, architecture and use cases? [Reported] [7][37]
    - How is BERT trained (MLM and NSP), and why did RoBERTa drop NSP?
    - When would an encoder-only model still beat a decoder-only LLM for an Amazon task, such as product classification?
12. What is an encoder-decoder model such as T5, and how would you optimize it for inference? [Reported] [5]
13. Compare LLM architectures: encoder-only vs decoder-only vs encoder-decoder, and dense vs mixture-of-experts. [Reported for the architecture discussion] (Amazon MLE aggregate: a rejected candidate described "a full round on architectures, scaling laws and RLHF trade-offs") [23]; MoE specifics [Likely]
    - How do you load-balance experts in MoE, and what goes wrong without it? [Likely]
14. What do scaling laws say about how to split a fixed compute budget between parameters and tokens? [Reported] (Amazon MLE aggregate, LLM depth round) [23]
    - How would you use a scaling-law fit to decide whether to train a larger Nova-sized model or serve a smaller one? [Likely]
15. Compare a small and a large model from the same text-embedding family. What do you gain, and what do you pay? [Reported] (Amazon RS aggregate, breadth: a weak answer here was felt to hurt the round) [22]
    - Trace the history of embedding models (word2vec, then BERT pooling, then contrastive sentence encoders, then LLM-based embedders). What changed at each step? [Likely]

### A3. LLM components written by hand

16. Hand-write sigmoid, softmax and scaled dot-product attention, numerically stable, in NumPy or PyTorch. [Reported] (Amazon RS and MLE aggregates; sighting dated July 2026 in the Amazon question list) [22][23][36]
    - Where do you subtract the max, and how do you apply a causal mask without producing NaNs?

## B. Inference, decoding and long context

17. How does inference work for decoder-only models? [Reported] [3]
    - Walk through prefill vs decode. Which phase is compute-bound and which is memory-bandwidth-bound?
18. What is the KV cache, and how does it speed up autoregressive generation? [Reported] [3][22][23]
    - Estimate the KV-cache memory for a given model (layers, heads, head dim, sequence length, batch). [Reported] (Amazon MLE aggregate: "memory arithmetic behind LoRA, quantization and KV caching") [23]
    - How do GQA/MQA, paged attention and cache quantization reduce it? [Likely]
19. What makes standard self-attention expensive for long contexts? [Reported] [3]
    - What are the time and memory complexity in sequence length, and what does FlashAttention change (and not change)? [Reported] (Amazon MLE aggregate: flash attention in the depth round) [23]
20. Explain Longformer and how its attention scales to long sequences. [Reported] [3]
    - Compare full attention, sparse or local attention, and other long-context approaches. [Reported] [3]
    - For a multi-turn shopping or search agent, when would you summarize or retrieve rather than extend the context window? [Reported] (Amazon RS aggregate: multi-turn search agent with long-context handling) [22]
21. How does beam search decoding work for LLMs? [Reported] [8]
    - When does beam search hurt open-ended generation, and why?
    - Compare greedy, beam, temperature, top-k and top-p sampling for a customer-facing assistant. [Reported] (LLM team: decoding choices for an Amazon use case) [8]
22. How would you add constrained or structured decoding so an assistant always returns valid JSON for a tool call? [Likely]
    - What does it cost in latency and quality?
23. What is speculative decoding, and when does it give a real speed-up? [Likely]
    - How do you pick the draft model, and how does the acceptance rate depend on the domain?
24. How does prefix (prompt) caching relate to the KV cache? When does it cut cost and latency for a Bedrock-style workload? [Likely] (follow-up in an Amazon-focused synthesized list [26])
    - What cross-tenant leakage risk does a shared prompt cache create, and how do you prevent it?
25. When is extra test-time compute (reasoning traces, self-consistency, best-of-n with a verifier) worth its latency and cost for a shopping or voice assistant? [Likely]
    - How would you decide per request whether to "think longer"?
26. How do you evaluate whether a model actually uses a long context, beyond needle-in-a-haystack tests? [Likely]
    - How would you test for "lost in the middle" effects on real multi-document tasks?

## C. Pretraining, fine-tuning and PEFT

27. Which modern LLM training techniques do you know (pretraining, SFT, preference optimization)? [Reported] [10]
    - For an AGI-team role, expect several follow-ups in a row on each stage. [Reported] [10]
28. Pretraining vs fine-tuning: what changes in data, objective and learning rate? [Likely]
    - What is catastrophic forgetting, and how do you detect and limit it?
29. Walk me through an SFT training pipeline and its loss. [Reported] [19]
    - Why mask the prompt tokens in the loss?
    - How do you build and clean an instruction dataset for a domain like shopping or Alexa intents?
    - How do you mix general and domain data, and how would you choose the ratio?
30. Explain the main parameter-efficient fine-tuning techniques, which parameters each trains, and when each beats full fine-tuning. [Reported] (Amazon RS and MLE aggregates, online assessment) [22][23]; LoRA specifically [Reported] [4]
    - Explain LoRA. Which matrices do you adapt, and how do you choose the rank and alpha?
    - What is the memory cost of an adapter compared with full fine-tuning, counting optimizer states? [Reported] (Amazon MLE aggregate) [23]
    - Compare LoRA variants (QLoRA, DoRA, adapters, prefix tuning). [Reported] (Amazon RS aggregate) [22]
31. You need a domain-specialized model with limited compute. Do you use prompting, RAG, LoRA, full fine-tuning or continued pretraining? [Reported for RAG vs fine-tuning] (Seattle Applied Scientist report, September 2025: "why RAG architecture is used instead of fine-tuning LLM models") [37]; the full option set is [Likely]
    - What evidence would make you move from one option to the next?
    - How does the answer change if the base model is only reachable as a hosted endpoint, for example through Bedrock? [Likely]
32. If you used knowledge distillation from an LLM, what exactly was distilled, and why? [Reported] [20]
    - Why distill instead of quantizing or pruning?
    - How would you distill an LLM into a small classifier using unlabeled data? [Likely] (see Amazon Science work on LLM distillation [31])
33. How would you generate synthetic training data with an LLM without collapsing diversity or leaking evaluation data? [Likely]
34. Your fine-tuning run's loss spikes at step 40k. How do you debug it? [Likely]
    - Which checks would you run on the data, the learning-rate schedule, mixed precision and gradient norms?
35. How does tokenizer choice (BPE vocabulary size, byte fallback) affect cost, latency and quality across Alexa's locales and scripts? [Likely] [27][32]
    - How would you extend a tokenizer for a new language without retraining from scratch?

## D. Alignment: RLHF, DPO and beyond

36. What is the difference between DPO and RLHF? [Reported] [7]
    - Derive how DPO removes the explicit reward model. What role does the reference model play?
    - What does DPO's beta control, and what happens at very small or very large beta?
37. Explain RLHF and PPO, including the clipped objective. [Reported] (Amazon RS and MLE aggregates) [22][23]
    - Why include a KL penalty to the SFT policy, and what failure mode does it prevent?
38. Compare PPO, GRPO and GSPO. Why does GRPO drop the critic? [Reported] (Amazon RS and MLE aggregates) [22][23]
    - What does removing the critic cost, and how would you parallelize GRPO training? [Reported] (generative-AI phone screen; specialized screens) [22][23]
    - When would you prefer verifiable rewards (RLVR) over a learned reward model? [Likely]
39. If you have run GRPO-style training yourself: which instabilities did you hit (entropy collapse, reward spikes, length blow-up), and how did you fix them? [Reported] (Amazon RS aggregate: post-training teams look for "evidence you have run GRPO-style training and fixed its instabilities") [22]
40. RLHF in practice: what goes wrong at scale? [Reported] (Amazon MLE aggregate: "RLHF in practice" and "RLHF trade-offs" in depth rounds) [23]
    - Walk through reward hacking, length bias, sycophancy and mode collapse. [Likely]
    - How would you detect reward hacking from logs? [Likely]
41. How do you design a reward signal or preference dataset for a shopping assistant? [Reported] (Amazon RS aggregate: "how would you design the reward") [22]
    - Who labels, how do you measure agreement, and how do you handle annotator disagreement?
    - Where could RLAIF replace human labels, and where can it not? [Likely] (Alexa postings list SFT, DPO, RLHF and RLAIF [27])
42. Your DPO-tuned model gives longer but less accurate answers. Diagnose it. [Likely]
    - How do you control for length in the preference data and in evaluation?
43. How would you align a multilingual Alexa model when preference data is scarce in some locales? [Likely] (Alexa International Tech posting [27]; the Alexa International loop guide [32])

## E. Retrieval-augmented generation (RAG)

44. Design an enterprise RAG service. [Reported] [16]
    - How do you chunk, embed and index heterogeneous documents?
    - How do you enforce per-user, document-level access control at query time, when permissions change between crawls?
    - How do you evaluate retrieval quality separately from generation quality?
45. What is the difference between a cross-encoder and a bi-encoder for retrieval? [Reported] [3]
    - When would you use each in a retrieve-then-rerank pipeline, and with what latency budget?
    - When would you distill a cross-encoder into a bi-encoder? [Likely] [31]
46. Explain RAG, and where it helps and does not help with hallucination. [Reported] (Amazon RS aggregate: LoRA, RAG and LLM agents in breadth) [22]
    - What failure modes remain once the right passage is retrieved (ignoring it, misreading it, or mixing it with parametric knowledge)?
47. Compare dense, sparse (BM25) and hybrid retrieval for product search queries. [Likely]
    - How do you handle exact attributes such as model numbers, sizes and brand names?
48. Contrastive embedding training: write a numerically stable, symmetric InfoNCE loss for paired image and text embeddings. [Reported] (Amazon RS and MLE aggregates, coding round) [22][23]
    - How do hard negatives change what the encoder learns, and how do you mine them without false negatives?
49. Rufus-style question: build retrieval-augmented answers to "What do I need for cold weather golf?" over the catalog, reviews and community Q&A. [Likely] [28][29][30]
    - How do you ground claims in product facts and avoid made-up specifications?
    - How do you handle stale or conflicting sources (listing vs reviews)?
    - How do you show citations, and how do you check that a citation supports the claim?
50. How would you decide between a longer context window and better retrieval for an assistant? [Likely]
51. How would you evaluate a RAG system offline (context recall and precision, faithfulness, answer relevance)? [Likely]
    - How do you build the golden set, and how often do you refresh it?
52. Much of the catalog is structured (attributes, prices, availability). When would you answer through a structured lookup or text-to-query tool instead of text retrieval, and how would the LLM decide? [Likely]
53. Use an LLM to rewrite or expand product-search queries. How do you keep it inside a search latency budget (offline caching for head queries, a distilled model for the tail)? How do you measure the effect on relevance? [Likely]
54. Would generative retrieval or semantic IDs (an LLM that decodes item identifiers) work for Amazon-scale product recommendation? What breaks as the catalog changes daily? [Likely]

## F. Agents and tool use

55. Pick an agentic product we could build. Who is the customer, and how do you measure success? [Reported] [2]
56. Propose agentic AI solutions for a hypothetical project, and defend their viability. [Reported] (Amazon RS aggregate, science application) [22]
57. Design an agentic recommendation system for Amazon, covering the component architecture, model choices, and quality and safety evaluation. [Reported] [21]
58. Design a GitHub Copilot-style coding agent. [Reported] (Senior AS loop) [3]
    - What context does the agent retrieve, and how do you rank it?
    - How do you measure whether it actually helps (acceptance rate vs downstream defects)?
59. Evaluate an LLM agent with metrics beyond accuracy: tool-use correctness, hallucination and calibration. [Reported] [3]
    - How do you build an offline evaluation set for multi-step agent tasks?
    - How do you assign credit when a five-step trajectory fails at step four?
60. Design a multi-turn search agent that handles long conversation context. [Reported] (Amazon RS and MLE aggregates) [22][23]
61. Architect an AI chatbot with intent detection, dialogue state, retrieval, generation and fallback. [Reported] (Amazon RS aggregate, science application; Amazon MLE aggregate, ML system design) [22][23]
    - Which layers would you keep as classic NLU or rules, and which would you hand to the LLM?
    - How does the system learn continuously from production conversations? [Reported] [23]
62. Alexa+-style question: an LLM-based assistant has to call real device and service APIs (timers, smart home, ordering). How do you make tool calls reliable and safe? [Likely]
    - How do you handle ambiguous requests: ask a clarifying question, or act?
    - Which actions require confirmation, and how do you decide?
63. Design a browser-operating agent (in the spirit of Nova Act) that completes multi-step web tasks. How do you make it reliable enough to ship? [Likely] (Nova Act is named as an active Amazon AGI product [25]; the prompt also appears in a synthesized Amazon list [26])
64. How do you defend an agent against prompt injection from retrieved web pages, product listings or reviews? [Likely]
65. Design long-term memory for an assistant like Alexa+: what to store about a household's preferences, how to retrieve it, and how to let users see and delete it. [Likely]
    - How do you stop a wrong memory from repeatedly degrading answers?
66. When would you split an assistant into several specialized agents or "experts" behind an orchestrator, rather than one model with many tools? [Likely]
    - How do you evaluate and debug the routing between them?

## G. Evaluation

67. Design an LLM quality validation system. [Reported] (Amazon RS and MLE aggregates; the MLE aggregate says strong answers combine automated checks with human review for generated output) [22][23][36]
    - What do you measure offline, and which online guardrails do you set? [Reported] [23]
    - How do you measure hallucination, faithfulness and helpfulness separately? [Likely] (follows from [3][16])
    - When would you trust an LLM-as-judge, and how would you calibrate it against human labels? [Likely]
    - What biases do LLM judges have (position, length, self-preference), and how do you control for them? [Likely]
68. Design an evaluation framework that checks whether a text-to-image system spells a brand name correctly and places it naturally (OCR accuracy, embedding checks, human review protocol). [Reported] (Amazon RS and MLE aggregates) [22][23]
69. How would you decide an LLM-powered assistant is ready to launch to millions of customers? [Likely] (synthesized Amazon list [26])
    - Which launch gates would you use for quality, safety, latency and cost?
70. Offline metrics improved but the A/B test fell short. How do you find out why? [Reported] (Amazon RS and MLE aggregates, depth round, general ML) [22][23]
    - How do you design an online experiment for a conversational assistant, where sessions interact and the outcome (purchase) is delayed? [Likely]
71. How do you detect benchmark contamination when evaluating a new foundation model? [Likely]
72. How would you measure an assistant's calibration and teach it to say "I don't know"? [Likely] (calibration reported as an agent-evaluation metric [3])
73. How would you run a red-teaming and safety evaluation for a shopping or voice assistant (toxicity, unsafe advice, PII leakage, jailbreaks)? [Likely]
74. How would you evaluate a multilingual assistant fairly across locales when you have far fewer human raters in some languages? [Likely] [27][32]
75. When is perplexity a misleading metric for comparing foundation-model checkpoints? How do preference-based evaluations (pairwise win rate, Elo or Bradley-Terry) complement it? [Reported for the topic] (Amazon AGI candidate reports summarized as covering "evaluation (including perplexity and preference-based evals)" [26]); exact wording [Likely]
76. How would you measure product-market fit for a GenAI shopping experience on Alexa? [Reported] (question title in an Amazon aggregator list, "Measure PMF for Alexa Shopping") [36]
    - Which engagement, retention and purchase metrics would you trust, and how would you separate novelty effects from lasting value?
77. How would you use implicit feedback (thumbs, reformulations, abandonment, purchases) from an assistant to improve it? How do you avoid feedback loops and selection bias? [Likely]

## H. Hallucination and grounding

78. How would you reduce hallucination in an LLM-powered assistant? [Likely] (Amazon's shopping assistant runs on LLMs including Amazon Nova, built on Bedrock [29])
    - Which causes are data-side, which are decoding-side, and which are retrieval-side?
    - How do you trade off abstaining against helpfulness?
79. An assistant states a wrong battery life for a product. Walk through your root-cause analysis. [Likely] [28][30]
    - How do you tell whether the cause was retrieval, a stale catalog attribute, or the model?
80. How would you build an automatic, claim-level factuality checker for generated answers? [Likely]
    - How do you validate the checker itself?
81. How would you fine-tune an LLM to be a shopping assistant that cites its sources? [Likely]
    - How would you reward grounded claims and penalize unsupported ones in preference tuning?
82. How would you summarize thousands of customer reviews into a short "Customers say" blurb? [Likely]
    - How do you measure faithfulness and coverage of minority opinions?
83. Design a guardrail layer that sits before and after the model (input filters, output filters, a grounding check, PII redaction). How do you set its thresholds and measure the false-block rate? [Likely] (guardrails as a composable layer in a synthesized Bedrock-shaped design [26])

## I. Efficiency, serving and cost

84. What are the trade-offs of putting LLMs into production (cost per token, latency, reliability, model right-sizing)? [Likely] (Frugality applied to GPUs is listed as an Amazon design axis [26])
85. Explain the memory arithmetic of quantization. Compare int8 and int4, and weight-only vs activation quantization (GPTQ, AWQ). What is the quality cost? [Reported for the memory arithmetic] (Amazon MLE aggregate) [23]; method comparison [Likely]
    - Which layers or tasks are most sensitive?
86. A customer's Bedrock-hosted workload costs too much. How would you cut inference cost sharply without unacceptable quality loss? [Likely] (Bedrock-shaped prompt, synthesized in an Amazon prep list [26])
    - Compare model routing, caching, distillation, quantization, batching and prompt compression.
    - How do you prove the quality loss is acceptable?
87. Your LLM endpoint's p99 latency doubled after a deploy, but the model weights did not change. Walk me through debugging it. [Likely] [26]
88. How does an inference engine such as vLLM raise serving throughput for large language models? [Reported] (Amazon RS and MLE aggregates, online assessment) [22][23]
    - Explain paged KV memory and continuous batching, and what each buys you.
89. Compare data, tensor and pipeline parallelism for training a large model across hundreds of accelerators. [Reported for pipeline-parallel schedules] (Amazon MLE aggregate, staff-level screens) [23]; the full comparison also appears in a synthesized list [26]
    - Where do the communication bottlenecks sit for each, and how do pipeline schedules reduce the bubble? [Reported] [23]
    - How does ZeRO/FSDP sharding change memory per device? [Likely]
90. Explain the collective operations used in distributed training (all-reduce, all-gather, reduce-scatter) and which parallelism strategy relies on each. [Reported] (question title in an Amazon aggregator list, "Explain parallelism and collectives in training") [36]
    - How does inter-node bandwidth change your choice of strategy? [Reported] [23]
91. For a residual MLP of width 4096 in BF16, which of batch size 1 and batch size 256 is compute-bound? Do the analysis from bandwidth and peak throughput. [Reported] (Amazon RS and MLE aggregates, manager-level screen) [22][23]
92. You have one latency budget for Rufus answers (say, first token in under 1 second). How do you split it across retrieval, reranking, generation and safety filters? [Likely] [28]
93. When would you route a request to a small model and when to a large one, and how would you learn the router? [Likely]
94. Design a distributed pipeline that ingests, tokenizes, deduplicates and filters LLM pretraining data. [Reported] (Amazon MLE aggregate, staff loop) [23]
    - How do you detect near-duplicates at web scale?
    - How do you measure the downstream effect of a quality filter?
95. Split an ordered list of sequences into contiguous batches. Each batch costs a fixed overhead plus padding up to its longest item. Minimize the total cost: give the state, the recurrence and a worked example. [Reported] (Amazon RS and MLE aggregates, Applied Scientist phone screen) [22][23][36]
    - How does this relate to length bucketing when serving or training LLMs?
96. Design a multi-tenant inference platform that serves many foundation models to thousands of customers (Bedrock-shaped). [Likely] (synthesized Amazon list [26])
    - How do you handle noisy neighbors, quotas and a 10x traffic spike on one model?
97. You must serve hundreds of customer-specific LoRA adapters on one shared base model. How do you batch requests across adapters, and what does it cost in latency compared with merged weights? [Likely]

## J. Multimodal and speech (team-dependent)

98. Compare a dual-encoder (CLIP-style) model with a vision-language model that uses a resampler or adapter. [Reported] (Amazon RS aggregate) [22]
    - Which would you use for product image search, and which for visual question answering on a listing?
99. How does the Segment Anything Model work? [Reported] [3][10]
100. How do diffusion noise schedules affect sample quality and training? [Reported] (Amazon RS aggregate) [22]
101. What causes codebook collapse in VQ-VAE, and how do you fix it? [Reported] (Amazon RS aggregate) [22]
102. Compare approaches to building joint image-text embeddings. [Reported] (Amazon RS aggregate, online assessment) [22]
    - How is a dual-encoder embedding trained, and what does batch size do to it?
103. For a speech-to-speech assistant (in the spirit of Nova Sonic or Alexa+), what changes when you move from an ASR, then LLM, then TTS cascade to an end-to-end speech model? [Likely] [25]
    - How do you evaluate turn-taking, barge-in and latency?
104. How would you improve Alexa speech recognition for non-native speakers and users with strong regional accents, and could an LLM rescoring stage help? [Reported for the base question] [32]; LLM follow-up [Likely]

## K. Applying GenAI to Amazon problems (science-application flavor)

105. Here is a high-level problem from our team. Would you solve it with an LLM at all? Propose a solution: data sources, annotation strategy, modeling, evaluation, trade-offs and risks. [Reported for the frame] [33]; the LLM-or-not framing is [Likely]
     - What is the cheapest baseline that might be good enough?
106. Use an LLM to categorize millions of products into a large taxonomy. How do you combine LLM labeling, a distilled classifier and human audit? [Likely] (product categorization is reported in classical form in earlier chapters; LLM distillation context [31])
107. Use an LLM to generate or enrich product listings (titles, bullet points, attributes) from seller input. How do you prevent made-up attributes? [Likely]
108. Use LLMs as relevance judges for product search. How do you validate them against human judges and behavioral signals? [Likely]
109. How would you personalize Rufus answers with a customer's purchase history while respecting privacy? [Likely] [29]
     - How would you measure whether personalization helps rather than creeps people out?
110. Sponsored products in a generative answer: how do you blend ads into an LLM response while keeping trust, and how do you measure the impact? [Likely]
111. Design an LLM-based system to detect policy-violating content or fraud in reviews or ads. [Likely] (Advertising Trust context [18])
     - How do you deal with adversaries who adapt to the classifier?
112. Design a system that turns customer-support transcripts into structured issue reports. How do you evaluate extraction quality? [Likely]
113. A team wants to replace an existing BERT classifier with an LLM prompt. What do you need to see before you agree? [Likely]
     - How do you compare cost per 1,000 requests, latency and accuracy on the long tail?
114. How would you detect LLM-generated fake reviews at Amazon scale, and what false-positive rate on genuine reviewers is acceptable? [Likely]

## L. Depth probes on your own LLM project (depth-round style)

115. Walk me through your most significant LLM or NLP project, including the GenAI fundamentals behind it and its business impact. [Reported] [18]
116. Why did you choose this base model and size? What would you do with more compute? [Reported] (Amazon RS aggregate: backbone choice, use of extra resources) [22]
117. How did you build the evaluation set, and how do you know it reflects real customer traffic? [Likely]
118. What did your error analysis show, and which fix had the biggest effect? [Likely]
     - What did you try that did not work?
119. If you fine-tuned, how did you rule out leakage between training data and evaluation prompts? [Likely]
120. Where would your approach break if deployed to a different marketplace or language? [Likely]
121. How did you decide the project was "done" and ready to ship (Deliver Results, Insist on the Highest Standards)? [Likely]
122. Explain an AI-agent project you built, and how it was adopted. [Reported] (Amazon MLE aggregate, project deep dive) [23]
     - What share of the target users actually used it, and what stopped the rest?
123. Paper-read round: discuss a recent LLM paper's method, assumptions and weaknesses. [Reported for the format] (Amazon RS aggregate: paper-read round on FAR and "premium" loops) [22]; the choice of an LLM paper is [Likely]

## M. GenAI behavioral and LP crossovers

124. How do you use AI tools in your work, and when did one give you a wrong answer? How did you catch it? [Reported] (Amazon general aggregate, phone screen and Bar Raiser) [24]
125. What hands-on experience do you have with post-training or agents? [Reported] (Amazon recruiter screens for generative-AI teams) [22][24]
126. Walk me through a concrete AI-assisted engineering workflow: what you delegated, what context you protected, and how you verified the output. [Reported] (Amazon new-grad screens, August 2026, SDE track; aggregator) [35]
127. You get a small repository, some failing tests and a guard-railed AI assistant. How do you work, and how do you know the fix is right? [Reported] (Amazon online assessment, AI-assisted debugging task, SDE and MLE pipelines) [23][24]
128. Tell me about a time you pushed back on using an LLM where a simpler model was better (Frugality, Invent and Simplify). [Likely]
129. Tell me about a time an LLM feature you shipped behaved badly in production. What did you do in the first 24 hours (Ownership, Dive Deep)? [Likely]
130. Tell me about a time you had to explain LLM risks (hallucination, safety) to a non-scientist stakeholder (Earn Trust). [Likely]

---

## Deepening pass (October 2026): newly reported questions

> Source note: this pass ran five web searches. Only Glassdoor review pages came back (no Blind, Reddit, LeetCode Discuss, 1point3acres, Prepfully, interviewing.io or IGotAnOffer threads), and the pages themselves were not opened. Tags below rest on search-result extracts only. Existing questions 4, 11 and 31 were updated in place with reference [37].

### Newly reported

1. Why would you use a RAG architecture instead of fine-tuning an LLM? [Reported] (Seattle Applied Scientist report, September 2025) [37]
   - What kinds of knowledge (fresh, proprietary, fast-changing) push you toward retrieval, and what kinds (style, format, task behavior) push you toward fine-tuning?
   - When would you combine both, for example fine-tuning the generator to use retrieved context?
2. Which evaluation metrics would you use for an LLM, and what does each miss? [Reported] (same September 2025 report: "evaluation metrics for LLMs") [37]
   - How do reference-based metrics (BLEU, ROUGE, exact match) compare with model-based and human pairwise evaluation for open-ended generation?
   - Which metric would you put on a launch dashboard, and which only in offline analysis?
3. Design an end-to-end pipeline that translates a set of multimodal objects (for example, listings with text and images) into another language. [Reported] (Hyderabad Applied Scientist report, system-design segment; mapping of this snippet to reference [38] comes from a search summary and should be re-verified) [38]
   - Which parts would you hand to an LLM or vision-language model, and which to a dedicated MT model?
   - How do you evaluate translation quality for text embedded in images, and how do you keep product attributes (units, sizes, brand names) unchanged?

### Gap-closing questions [Likely]

4. How would you build an LLM-based query-understanding or intent classifier for Alexa that must answer in under 100 ms? Compare prompting a large model, distilling into a small encoder, and fine-tuning a small decoder. [Likely]
5. Derive the memory needed to fully fine-tune a 7B-parameter model with AdamW in mixed precision (weights, gradients, optimizer states, activations). Which techniques (gradient checkpointing, ZeRO, LoRA) cut which term? [Likely]
6. Explain how in-context learning differs from fine-tuning. Why does few-shot performance depend on example order and format, and how would you choose demonstrations automatically? [Likely] (Alexa postings name In-Context Learning alongside SFT [27])
7. How do you write and version prompts as production artifacts? How would you regression-test a prompt change before rolling it out to a high-traffic assistant? [Likely]
8. Your LLM is fine-tuned on Amazon-specific data and now refuses or degrades on general questions. How do you measure and recover lost general capability (replay data, model merging, regularization toward the base model)? [Likely]
9. How would you detect and handle data drift for a production LLM feature, such as new product categories, seasonal queries or newly launched devices? What triggers re-training versus re-indexing the retrieval store? [Likely]
10. A reward model scores 90% pairwise accuracy on held-out preferences, but the policy trained against it gets worse in human evaluation. What explains the gap, and what would you check? [Likely]
11. How do you choose an embedding dimension and index type (HNSW, IVF-PQ) for a billion-item product catalog, and how do you measure the recall lost to approximate search? [Likely]
12. Compare autoregressive LLM decoding with a non-generative approach (extractive QA, classification with a fixed label set) for an Amazon task. When does the generative model's flexibility justify its cost and hallucination risk? [Likely]
13. How would you protect customer privacy when fine-tuning on conversation logs (PII scrubbing, differential privacy, memorization tests)? How would you test whether the model can regurgitate a customer's data? [Likely]

## Coverage notes for prep

- **Highest-yield set across reports:**
  - transformer and attention mechanics (questions 1-2, 16);
  - BERT vs GPT (11);
  - KV cache, decoder inference and long context (17-21);
  - training, SFT and PEFT/LoRA (27, 29-30);
  - DPO vs RLHF, and PPO/GRPO (36-38);
  - bi-encoder vs cross-encoder, and RAG design (44-45);
  - agents and LLM evaluation (55-61, 67);
  - vLLM serving (88).

  Sources: [3][4][7][8][10][16][19][22][23].
- **Team-specific tilt:**
  - AGI/Nova teams: training, RL, parallelism, evaluation and ML coding [15][17][22][25][26].
  - Rufus/Stores: RAG, grounding, ranking and evaluation [28][29][30].
  - Alexa: multilingual alignment, tool use and speech [27][32][36].
  - AWS/Bedrock-adjacent science: PEFT, serving cost and customization [4][26].
- **Aggregator caveat:** questions marked "(Amazon RS aggregate)" or "(Amazon MLE aggregate)" come from compilations of Amazon Research Scientist and ML Engineer reports [22][23]. Those loops overlap heavily with AS loops but are not identical. The aggregator pages regenerate often, so a quoted item may later move or disappear.

## References

1. Applied Scientist Interview Prep, amazon.jobs (Amazon, official), accessed 2026. https://amazon.jobs/content/en/how-we-hire/applied-scientist-interview-prep
2. Amazon Applied Scientist II Interview Experience & Questions, Glassdoor, 2025-2026 (loop "1 Science ML Breadth, 1 Science Depth, 1 Sys Design (Team-specific), 1 coding round and 1 Bar Raiser"; agentic-product prompt). https://www.glassdoor.com/Interview/Amazon-Applied-Scientist-II-Interview-Questions-EI_IE6036.0,6_KO7,27.htm
3. Amazon Applied Scientist Interview Experience & Questions, Glassdoor, 2025-2026 (decoder-only inference, KV caching, Longformer, bi-encoder vs cross-encoder, SAM, Copilot-style agent, LLM-agent evaluation). https://www.glassdoor.com/Interview/Amazon-Applied-Scientist-Interview-Questions-EI_IE6036.0,6_KO7,24.htm
4. Amazon Web Services Applied Scientist Interview Experience & Questions, Glassdoor (explain transformers; LoRA). https://www.glassdoor.com/Interview/Amazon-Web-Services-Applied-Scientist-Interview-Questions-EI_IE7470741.0,19_KO20,37.htm
5. Amazon Applied Scientist Intern Interview Experience & Questions, Glassdoor, 2025 (transformers "at root level"; T5 and optimizing it). https://www.glassdoor.com/Interview/Amazon-Applied-Scientist-Intern-Interview-Questions-EI_IE6036.0,6_KO7,31.htm
6. Amazon Applied Scientist Interview Questions, page 3, Glassdoor, circa 2024 (Wq = Wk = Wv; transformer basics). https://www.glassdoor.com/Interview/Amazon-Applied-Scientist-Interview-Questions-EI_IE6036.0,6_KO7,24_IP3.htm
7. Amazon interview review RVW96903377 (Applied Scientist II, Seattle), Glassdoor, 2026 (attention at root level; BERT vs GPT; DPO vs RLHF). https://www.glassdoor.fr/Entretien/Amazon-Entretien-E6036-RVW96903377.htm
8. Amazon interview review RVW96068026 (Applied Scientist II, LLM team, Santa Clara), Glassdoor, 2025 (beam search; LLM decoding and training in an Amazon use case). https://www.glassdoor.com.hk/Interview/Amazon-Interview-E6036-RVW96068026.htm
9. Amazon interview review RVW102987402 (Applied Scientist), Glassdoor, 2026 (transformer architecture in detail). https://www.glassdoor.co.uk/Interview/Amazon-Interview-E6036-RVW102987402.htm
10. Glassdoor individual Amazon interview reviews, 2026. One reports many "modern LLM training technique" questions for an AGI-team role; one reports a Segment Anything question. Which review holds which question could not be verified. https://fr.glassdoor.ch/Entretien/Amazon-Entretien-E6036-RVW104288349.htm ; https://fr.glassdoor.ch/Entretien/Amazon-Entretien-E6036-RVW104864395.htm ; https://www.glassdoor.fr/Entretien/Amazon-Entretien-E6036-RVW101665167.htm
11. "Amazon online 1st round, Bengaluru, Applied Scientist", LeetCode Discuss, May 2025 (LLM theory, transformer internals). https://leetcode.com/discuss/post/6871782/amazon-online-1st-round-bengaluru-applie-uiwk/
12. "Amazon Applied Scientist Phone Interview", LeetCode Discuss (LSTM vs Transformers mathematically). https://leetcode.com/discuss/post/543684/amazon-applied-scientist-phone-interview/
13. "Amazon Applied Scientist (L5) Virtual Onsite", Blind, circa 2025. https://www.teamblind.com/post/amazon-applied-scientist-l5-virtual-onsite-navwesml
14. "Applied scientist interview at Amazon Ads", Blind (rejected after not preparing attention, transformers and LLMs). https://www.teamblind.com/post/applied-scientist-interview-at-amazon-ads-zoerec5t
15. "Amazon (AGI team) Applied Scientist II interview", Blind. https://www.teamblind.com/post/Amazon-AGI-team-Applied-Scientist-II-interview-2eqBXGtT
16. "Amazon Onsite Interview Experience for Fulltime Machine Learning Role", 1point3acres (enterprise RAG design). https://www.1point3acres.com/interview/thread/1157086
17. "Amazon AGI Applied Scientist Interview Experience and Insights", 1point3acres. https://www.1point3acres.com/interview/thread/1130091
18. Chaitanya Mundle, "My interview experience for the Applied Scientist II role at Amazon", Medium, circa 2024 (Advertising Trust; Science Depth = resume plus GenAI fundamentals plus business impact). https://medium.com/@mundle1997.cm/my-interview-experience-for-the-applied-scientist-ii-role-at-amazon-738aba21f611
19. "Amazon Applied Scientist Interview Debrief: Five-Round Loop + LP", OA VO Service (vendor-published candidate debrief, promotional; SFT pipeline; L5 vs L4 follow-up depth). https://oavoservice.com/en/articles/amazon-applied-scientist-vo-loop-lp-ml-coding
20. "Amazon Interview Experience for Applied Scientist 2", Enginebogie, circa 2025 (knowledge distillation; limited labels). https://enginebogie.com/interview/experience/amazon-applied-scientist-2/1004
21. "Amazon Applied Scientist Interview Questions & Guide 2026", Dataford (prep vendor; agentic recommendation system design), 2026. https://dataford.io/interview-guides/amazon/applied-scientist
22. "Amazon Research Scientist Interview Process", kevin-2023-code/Tech-Interview-Questions, GitHub (aggregator adapted from trueinterview.io, CC BY 4.0; "based on 20 candidate-reported interview experiences, primarily from 2026 Q1 through 2026 Q3"; covers RS and AS loops). Re-fetched and checked October 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/amazon/interview-process-research-scientist.md
23. "Amazon Machine Learning Engineer Interview Process", kevin-2023-code/Tech-Interview-Questions, GitHub (aggregator adapted from trueinterview.io, CC BY 4.0). Re-fetched and checked October 2026. The current version does not mention mixture-of-experts, transformers replacing RNNs, or pretraining vs fine-tuning; items that relied on it for those were retagged. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/amazon/interview-process-machine-learning-engineer.md
24. "Amazon Interview Process", kevin-2023-code/Tech-Interview-Questions, GitHub (aggregator; GenAI-usage behavioral question, AI-assisted debugging assessment, recruiter questions on post-training and agents). Re-fetched October 2026; it does not mention content moderation. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/amazon/interview-process.md
25. "AI Companies Interview Questions" (Amazon AGI section: reported 2025 loop, "Debug a broken Transformer implementation", Nova products and the 2026 org note), ombharatiya/FAANG-Coding-Interview-Questions, GitHub (aggregator; underlying reports not individually named), 2026. Re-fetched October 2026. https://github.com/ombharatiya/FAANG-Coding-Interview-Questions/blob/main/AI-Companies-Interview-Questions.md
26. "amazon.md", ombharatiya/AI-Engineer-Interview-Questions, GitHub. Its representative questions are explicitly "synthesised from this company's publicly known focus areas ... not leaked questions"; they are used only for [Likely] items. Its loop and emphasis notes summarize candidate reports, such as AGI loops covering evaluation including perplexity and preference-based evals. Re-fetched October 2026. https://github.com/ombharatiya/AI-Engineer-Interview-Questions/blob/main/14-company-interview-questions/amazon.md
27. Applied Scientist, Alexa International Tech (Job ID 10490052), amazon.jobs (SFT, DPO, RLHF, RLAIF). https://www.amazon.jobs/en/jobs/10490052/applied-scientist-alexa-international-tech
28. Applied Scientist, Rufus Experiences Science (Job ID 2855826), amazon.jobs. https://amazon.jobs/en/jobs/2855826/applied-scientist-rufus-experiences-science
29. "Amazon Rufus AI assistant personalized shopping features", About Amazon (LLMs including Amazon Nova on Bedrock; an earlier chapter's search extract says it was renamed Alexa for Shopping in May 2026). https://aboutamazon.com/news/retail/amazon-rufus-ai-assistant-personalized-shopping-features
30. "What do I need for cold weather golf?" (Amazon Rufus), IEEE Spectrum. https://spectrum.ieee.org/amazon-rufus
31. "Learning with less: Knowledge distillation from large language models via unlabeled data", Amazon Science. https://www.amazon.science/publications/learning-with-less-knowledge-distillation-from-large-language-models-via-unlabeled-data
32. Two Medium articles, 2025, both on accented-speech ASR:
    - "Amazon Applied Scientist Interview Guide: Alexa International Tech", Medium (Interview Preparation), Apr 2025. https://medium.com/interview-preparation/amazon-applied-scientist-interview-guide-alexa-international-tech-022d3a056feb
    - The companion "Science application: Business problems for Amazon Applied Scientist (Alexa International Tech)", Medium, 2025. https://medium.com/@pragyashukla2580/science-application-business-problems-for-amazon-applied-scientist-alexa-international-tech-9353a52ca556
33. "Amazon Applied Scientist Interview (process, questions, prep)", IGotAnOffer, 2025-2026. https://igotanoffer.com/en/advice/amazon-applied-scientist-interview
34. Blind, "Amazon AS L4 vs L5" and "Amazon Applied Scientist L4 vs Applied Scientist II L5" (level-bar discussion). https://www.teamblind.com/post/amazon-as-l4-vs-l5-n0zunqgt ; https://www.teamblind.com/post/amazon-applied-scientist-l4-vs-applied-scientist-ii-l5-aasmppdi
35. "FAANG Recent Questions" (Amazon section, 2025-2026 trends: AI-assisted OA repository task; August 2026 new-grad screens asking candidates to walk through an AI-assisted engineering workflow), ombharatiya/FAANG-Coding-Interview-Questions, GitHub (aggregator). Fetched October 2026. https://github.com/ombharatiya/FAANG-Coding-Interview-Questions/blob/main/FAANG-Recent-Questions.md
36. "Amazon interview process, OA & interview questions" (company question list of about 320 reported questions with types and sighting dates, including "Handwrite Sigmoid, Softmax, and Attention", "Bucket Batching DP (Applied Scientist Phone Screen)", "Design an LLM quality validation system", "Explain parallelism and collectives in training" and "Measure PMF for Alexa Shopping"), kevin-2023-code/Tech-Interview-Questions, GitHub (aggregator adapted from trueinterview.io). Fetched October 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/amazon.md
37. Amazon interview review RVW100110012 (Applied Scientist, Seattle, September 2025), Glassdoor (LLM evaluation metrics; why RAG instead of fine-tuning; LSTM vs Transformer; BERT vs GPT). Captured from search-result extracts, October 2026; page not opened. https://www.glassdoor.fr/Entretien/Amazon-Entretien-E6036-RVW100110012.htm
38. Amazon interview review RVW101867612 (Applied Scientist, Hyderabad), Glassdoor (general data-science questions plus a system design to "model a pipeline end-to-end for translating one set of multimodal objects to another language"). Attribution of the snippet to this review comes from a search summary and was not confirmed by opening the page, October 2026. https://www.glassdoor.it/Colloquio/Amazon-Colloquio-E6036-RVW101867612.htm
