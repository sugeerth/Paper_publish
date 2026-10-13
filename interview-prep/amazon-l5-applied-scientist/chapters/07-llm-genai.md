# 07. LLM and GenAI Questions: Transformers, Fine-Tuning, RLHF/DPO, RAG, Evaluation, Hallucination, Efficiency, and Amazon Product Context (Bedrock, Nova, Rufus, Alexa+)

> Scope: the LLM and GenAI questions an L5 Applied Scientist (Applied Scientist II) candidate should expect across the 2025-2026 onsite loop. This chapter is questions only, with no answers.
>
> Research note (please read): the shared web-search budget for this run was used up before this chapter started, and direct page fetches to Glassdoor, Blind, LeetCode, 1point3acres, Medium, amazon.jobs, amazon.science, IGotAnOffer and similar sites failed in this environment (DNS refused by the egress proxy). This chapter therefore draws on three kinds of evidence:
> - Candidate-report findings already gathered for chapters 01-06 of this guide. Those came from search-result extracts of the cited pages, not full reads. Each [Reported] tag cites the original page.
> - Three GitHub-hosted interview aggregators that were fetched and read directly for this chapter [22][23][24][25]. They compile Amazon candidate reports from sources they cite (mostly trueinterview.io, CC BY 4.0) or do not name. They are labeled as aggregators wherever used.
> - Amazon job postings and product pages, used only to ground [Likely] questions.
>
> A follow-up session with a fresh search budget could add more first-person reports, especially from Blind and 1point3acres threads about Rufus, AGI/Nova and Alexa teams.

## Format notes for this round

- **There is usually no separate "LLM round."** LLM and GenAI questions come up inside the standard loop seats: ML/Science Breadth, Science Depth (your own project), Science Application / ML System Design, and sometimes ML Coding. Amazon's official page describes a loop of about four 55-minute interviews with members of the science community. Each interviewer is assigned technical competencies and Leadership Principles and typically asks 2-3 behavioral questions [1]. Candidate reports for AS II more often describe five rounds of 55-60 minutes, for example "1 Science ML Breadth, 1 Science Depth, 1 Sys Design (Team-specific), 1 coding round and 1 Bar Raiser" [2]. Aggregated Amazon science reports also describe a "modern deep learning" round on some loops, and one or two ML-theory rounds on GenAI teams [22][23].
- **Transformers are now baseline breadth, even off-LLM teams.** An L5 virtual-onsite thread on Blind treats transformers as core breadth [13]. An Amazon Ads candidate reported being rejected after not preparing for attention, transformers and LLMs [14]. 2025-2026 reports cover decoder-only inference, KV caching, Longformer-style long-context attention, bi-encoders vs cross-encoders, LoRA, BERT vs GPT, DPO vs RLHF, beam search, and "modern LLM training techniques" [3][4][7][8][10][11].
- **Team-dependent depth.** AGI/Nova-team loops are reported to lean hard on modern LLM training and ML coding. One aggregator reports a "debug a broken Transformer implementation" coding problem, and says candidates compare the ML-coding style to OpenAI's [10][15][17][25]. LLM-team and Rufus/Alexa loops tie decoding, training and evaluation to an Amazon use case [8][27][28]. Science-application prompts in 2025-2026 include agentic products, agentic recommenders, enterprise RAG and LLM-agent evaluation [2][3][16][21].
- **Interviewer type.** Usually Applied Scientists, Senior Applied Scientists or Principal Applied Scientists. The application round is often run by the hiring manager [1][2].
- **The L5 bar.** L5 answers are expected to hold up through two or three layers of follow-up, against one layer at L4 [19]. Blind commenters say L5 candidates are vetted harder on whether they "raise the bar" [34]. For LLM topics that means:
  - deriving the mechanics (attention, the DPO loss, KV-cache memory);
  - naming failure modes and trade-offs;
  - choosing between prompting, RAG, fine-tuning and preference tuning for a concrete Amazon problem;
  - defining how you would evaluate it offline and online, including the cost and latency budget.
- **Leadership Principles still apply.** Expect 1-2 LP questions inside each technical round [1][19]. A GenAI-flavored behavioral prompt is now reported in Amazon screens: how you use AI tools, and a time one gave you a wrong answer [24].

**Tag legend.** [Reported] means a cited source attributes the question, or a close paraphrase, to an Amazon science interview. That is Applied Scientist unless noted. Items from the Research Scientist or MLE aggregators are marked "(Amazon RS aggregate)" or "(Amazon MLE aggregate)". [Likely] means a grounded inference from the role, Amazon's products or job postings, or adjacent reported questions. It has not been seen verbatim.

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
4. Explain the difference between LSTMs and Transformers mathematically. [Reported] [12]
   - How does each handle long-range dependencies, and what are the parallelism and compute trade-offs?
   - Why did transformers replace RNNs for language modeling? [Reported] (Amazon MLE aggregate) [23]
5. Why do Transformers need positional information? Compare sinusoidal, learned, RoPE and ALiBi encodings. [Reported] (Amazon RS aggregate: RoPE formula, RoPE and ALiBi) [22]
   - How does RoPE behave when you extrapolate past the training context length?
6. Compare multi-head, multi-query and grouped-query attention. [Reported] (Amazon MLE aggregate, phone screen) [23]
   - How does each change KV-cache memory and quality?
7. Transformer internals: how does the feed-forward block contribute compared with attention, and where do most of the parameters and FLOPs go? [Reported] (Amazon RS aggregate, "modern deep learning" round) [22]
   - Where do the parameter count and FLOPs come from as d_model, the number of layers and the sequence length change?
8. Why does AdamW decouple weight decay, and why does that matter for training transformers? [Reported] (Amazon RS aggregate) [22]
9. You are handed a Transformer implementation that trains but performs badly. Debug it. [Reported] (Amazon AGI, aggregator) [25]
   - What bugs would you check for first (mask direction, softmax axis, scaling, positional-encoding offsets, layer-norm placement)?

### A2. Model families

10. How do BERT and GPT differ in objectives, architecture and use cases? [Reported] [7]
    - How is BERT trained (MLM and NSP), and why did RoBERTa drop NSP?
    - When would an encoder-only model still beat a decoder-only LLM for an Amazon task, such as product classification?
11. What is an encoder-decoder model such as T5, and how would you optimize it for inference? [Reported] [5]
12. Compare architectures: dense vs mixture-of-experts, and encoder-only vs decoder-only vs encoder-decoder. [Reported] (Amazon MLE aggregate, "LLM depth" onsite) [23]
    - How do you load-balance experts in MoE, and what goes wrong without it?
13. What do scaling laws say about how to split a fixed compute budget between parameters and tokens? [Reported] (Amazon MLE aggregate) [23]
    - How would you use a scaling-law fit to decide whether to train a larger Nova-sized model or serve a smaller one? [Likely]
14. Explain the history of embedding models (word2vec, then BERT pooling, then contrastive sentence encoders, then LLM-based embedders). How would you compare two sizes of the same embedding family? [Reported] (Amazon RS aggregate, breadth) [22]

## B. Inference, decoding and long context

15. How does inference work for decoder-only models? [Reported] [3]
    - Walk through prefill vs decode. Which phase is compute-bound and which is memory-bandwidth-bound?
16. What is the KV cache, and how does it speed up autoregressive generation? [Reported] [3][22][23]
    - Estimate the KV-cache memory for a given model (layers, heads, head dim, sequence length, batch).
    - How do GQA/MQA, paged attention and cache quantization reduce it? [Likely]
17. What makes standard self-attention expensive for long contexts? [Reported] [3]
    - What are the time and memory complexity in sequence length, and what does FlashAttention change (and not change)? [Likely]
18. Explain Longformer and how its attention scales to long sequences. [Reported] [3]
    - Compare full attention, sparse or local attention, and other long-context approaches. [Reported] [3]
    - For a multi-turn shopping or search agent, when would you summarize or retrieve rather than extend the context window? [Reported] (Amazon RS aggregate: multi-turn search agent with long-context handling) [22]
19. How does beam search decoding work for LLMs? [Reported] [8]
    - When does beam search hurt open-ended generation, and why?
    - Compare greedy, beam, temperature, top-k and top-p sampling for a customer-facing assistant. [Reported] (LLM team: decoding choices for an Amazon use case) [8]
20. How would you add constrained or structured decoding so an assistant always returns valid JSON for a tool call? [Likely]
    - What does it cost in latency and quality?
21. What is speculative decoding, and when does it give a real speed-up? [Likely]
    - How do you pick the draft model, and how does the acceptance rate depend on the domain?

## C. Pretraining, fine-tuning and PEFT

22. Which modern LLM training techniques do you know (pretraining, SFT, preference optimization)? [Reported] [10]
    - For an AGI-team role, expect several follow-ups in a row on each stage. [Reported] [10]
23. Pretraining vs fine-tuning: what changes in data, objective and learning rate? [Reported] (Amazon MLE aggregate) [23]
    - What is catastrophic forgetting, and how do you detect and limit it?
24. Walk me through an SFT training pipeline and its loss. [Reported] [19]
    - Why mask the prompt tokens in the loss?
    - How do you build and clean an instruction dataset for a domain like shopping or Alexa intents?
    - How do you mix general and domain data, and how would you choose the ratio?
25. Explain LoRA. [Reported] [4][22][23]
    - Which matrices do you adapt, and how do you choose the rank and alpha?
    - What is the memory cost of an adapter compared with full fine-tuning, counting optimizer states? [Reported] (Amazon MLE aggregate) [23]
    - Compare LoRA variants (QLoRA, DoRA, adapters, prefix tuning). [Reported] (Amazon RS aggregate) [22]
26. You need a domain-specialized model with limited compute. Do you use prompting, RAG, LoRA, full fine-tuning or continued pretraining? [Likely]
    - What evidence would make you move from one option to the next?
    - How does the answer change if the base model is only reachable as a hosted endpoint, for example through Bedrock? [Likely]
27. If you used knowledge distillation from an LLM, what exactly was distilled, and why? [Reported] [20]
    - Why distill instead of quantizing or pruning?
    - How would you distill an LLM into a small classifier using unlabeled data? [Likely] (see Amazon Science work on LLM distillation [31])
28. How would you generate synthetic training data with an LLM without collapsing diversity or leaking evaluation data? [Likely]
29. Your fine-tuning run's loss spikes at step 40k. How do you debug it? [Likely]
    - Which checks would you run on the data, the learning-rate schedule, mixed precision and gradient norms?

## D. Alignment: RLHF, DPO and beyond

30. What is the difference between DPO and RLHF? [Reported] [7]
    - Derive how DPO removes the explicit reward model. What role does the reference model play?
    - What does DPO's beta control, and what happens at very small or very large beta?
31. Explain RLHF and PPO, including the clipped objective. [Reported] (Amazon RS aggregate) [22]
    - Why include a KL penalty to the SFT policy, and what failure mode does it prevent?
32. Compare PPO, GRPO and GSPO. Why does GRPO drop the critic? [Reported] (Amazon RS and MLE aggregates) [22][23]
    - When would you prefer verifiable rewards (RLVR) over a learned reward model?
33. RLHF in practice: what goes wrong at scale (reward hacking, length bias, sycophancy, mode collapse)? [Reported] (Amazon MLE aggregate) [23]
    - How would you detect reward hacking from logs?
34. How do you design a reward signal or preference dataset for a shopping assistant? [Reported] (Amazon RS aggregate: reward design) [22]
    - Who labels, how do you measure agreement, and how do you handle annotator disagreement?
    - Where could RLAIF replace human labels, and where can it not? [Likely] (Alexa postings list SFT, DPO, RLHF and RLAIF [27])
35. Your DPO-tuned model gives longer but less accurate answers. Diagnose it. [Likely]
    - How do you control for length in the preference data and in evaluation?
36. How would you align a multilingual Alexa model when preference data is scarce in some locales? [Likely] (Alexa International Tech posting [27]; the Alexa International loop guide [32])

## E. Retrieval-augmented generation (RAG)

37. Design an enterprise RAG service. [Reported] [16]
    - How do you chunk, embed and index heterogeneous documents?
    - How do you enforce per-user, document-level access control at retrieval time?
    - How do you evaluate retrieval quality separately from generation quality?
38. What is the difference between a cross-encoder and a bi-encoder for retrieval? [Reported] [3]
    - When would you use each in a retrieve-then-rerank pipeline, and with what latency budget?
    - When would you distill a cross-encoder into a bi-encoder? [Likely] [31]
39. Explain RAG, and where it helps and does not help with hallucination. [Reported] (Amazon RS aggregate: LoRA, RAG and agents in breadth) [22]
    - What failure modes remain once the right passage is retrieved (ignoring it, misreading it, or mixing it with parametric knowledge)?
40. Compare dense, sparse (BM25) and hybrid retrieval for product search queries. [Likely]
    - How do you handle exact attributes such as model numbers, sizes and brand names?
41. Contrastive embedding training: write out the InfoNCE loss and explain how to keep it numerically stable. [Reported] (Amazon RS aggregate, coding round) [22]
    - How do hard negatives change what the encoder learns, and how do you mine them without false negatives?
42. Rufus-style question: build retrieval-augmented answers to "What do I need for cold weather golf?" over the catalog, reviews and community Q&A. [Likely] [28][29][30]
    - How do you ground claims in product facts and avoid made-up specifications?
    - How do you handle stale or conflicting sources (listing vs reviews)?
    - How do you show citations, and how do you check that a citation supports the claim?
43. How would you decide between a longer context window and better retrieval for an assistant? [Likely]
44. How would you evaluate a RAG system offline (context recall and precision, faithfulness, answer relevance)? [Likely]
    - How do you build the golden set, and how often do you refresh it?

## F. Agents and tool use

45. Pick an agentic product we could build. Who is the customer, and how do you measure success? [Reported] [2]
46. Design an agentic AI solution for a hypothetical product. [Reported] (Amazon RS aggregate, science application) [22]
47. Design an agentic recommendation system for Amazon, covering the component architecture, model choices, and quality and safety evaluation. [Reported] [21]
48. Design a GitHub Copilot-style coding agent. [Reported] (Senior AS loop) [3]
    - What context does the agent retrieve, and how do you rank it?
    - How do you measure whether it actually helps (acceptance rate vs downstream defects)?
49. Evaluate an LLM agent with metrics beyond accuracy: tool-use correctness, hallucination and calibration. [Reported] [3]
    - How do you build an offline evaluation set for multi-step agent tasks?
    - How do you assign credit when a five-step trajectory fails at step four?
50. Design a multi-turn search agent that handles long context. [Reported] (Amazon RS aggregate) [22]
51. Design an AI chatbot with intent, dialogue, retrieval and generation layers. [Reported] (Amazon MLE aggregate, phone-screen design) [23]
    - Which layers would you keep as classic NLU or rules, and which would you hand to the LLM?
52. Alexa+-style question: an LLM-based assistant has to call real device and service APIs (timers, smart home, ordering). How do you make tool calls reliable and safe? [Likely]
    - How do you handle ambiguous requests: ask a clarifying question, or act?
    - Which actions require confirmation, and how do you decide?
53. Design a browser-operating agent (in the spirit of Nova Act) that completes multi-step web tasks. How do you make it reliable enough to ship? [Likely] (Nova Act named as an active Amazon AGI product [25])
54. How do you defend an agent against prompt injection from retrieved web pages, product listings or reviews? [Likely]

## G. Evaluation

55. How would you evaluate an LLM or LLM-based system? [Reported] (Amazon MLE aggregate, "project plus foundations" onsite) [23]
    - What do you measure offline, and which online guardrails do you set? [Reported] (Amazon MLE aggregate) [23]
56. How do you evaluate an LLM-based system for hallucination, faithfulness and helpfulness? [Likely] (follows from [3][16])
    - When would you trust an LLM-as-judge, and how would you calibrate it against human labels?
    - What biases do LLM judges have (position, length, self-preference), and how do you control for them?
57. Design an evaluation framework that checks whether brand names are spelled correctly in generated images (OCR, embedding checks, human protocol). [Reported] (Amazon RS aggregate) [22]
58. How would you decide an LLM-powered assistant is ready to launch to millions of customers? [Likely]
    - Which launch gates would you use for quality, safety, latency and cost?
59. Offline metrics improved but the A/B test is flat. What happened? [Likely]
    - How do you design an online experiment for a conversational assistant, where sessions interact and the outcome (purchase) is delayed?
60. How do you detect benchmark contamination when evaluating a new foundation model? [Likely]
61. How would you measure an assistant's calibration and teach it to say "I don't know"? [Likely] (calibration reported as an agent-evaluation metric [3])
62. How would you run a red-teaming and safety evaluation for a shopping or voice assistant (toxicity, unsafe advice, PII leakage, jailbreaks)? [Likely]
63. How would you evaluate a multilingual assistant fairly across locales when you have far fewer human raters in some languages? [Likely] [27][32]

## H. Hallucination and grounding

64. How would you reduce hallucination in an LLM-powered assistant? [Likely] (Amazon's shopping assistant runs on LLMs including Amazon Nova, built on Bedrock [29])
    - Which causes are data-side, which are decoding-side, and which are retrieval-side?
    - How do you trade off abstaining against helpfulness?
65. An assistant states a wrong battery life for a product. Walk through your root-cause analysis. [Likely] [28][30]
    - How do you tell whether the cause was retrieval, a stale catalog attribute, or the model?
66. How would you build an automatic, claim-level factuality checker for generated answers? [Likely]
    - How do you validate the checker itself?
67. How would you fine-tune an LLM to be a shopping assistant that cites its sources? [Likely]
    - How would you reward grounded claims and penalize unsupported ones in preference tuning?
68. How would you summarize thousands of customer reviews into a short "Customers say" blurb? [Likely]
    - How do you measure faithfulness and coverage of minority opinions?

## I. Efficiency, serving and cost

69. What are the trade-offs of putting LLMs into production? [Reported] (Amazon MLE aggregate, "LLM depth" onsite) [23]
70. Quantization: compare int8 and int4, and weight-only vs activation quantization (GPTQ, AWQ). What is the quality cost? [Likely]
    - Which layers or tasks are most sensitive?
71. A customer's Bedrock-hosted workload costs too much. How would you cut inference cost sharply without unacceptable quality loss? [Likely] (Bedrock-shaped prompt, synthesized in an Amazon prep list [26])
    - Compare model routing, caching, distillation, quantization, batching and prompt compression.
    - How do you prove the quality loss is acceptable?
72. Your LLM endpoint's p99 latency doubled after a deploy, but the model weights did not change. Walk me through debugging it. [Likely] [26]
73. Explain continuous batching and why it improves throughput for LLM serving. [Likely]
74. Compare data, tensor and pipeline parallelism for training a large model across hundreds of accelerators. [Reported] (Amazon RS and MLE aggregates: pipeline parallelism) [22][23]
    - Where do the communication bottlenecks sit for each?
    - How does ZeRO/FSDP sharding change memory per device? [Likely]
75. Is an MLP layer compute-bound or memory-bound at a given batch size? Do the roofline analysis. [Reported] (Amazon MLE aggregate, infrastructure loop) [23]
76. You have one latency budget for Rufus answers (say, first token in under 1 second). How do you split it across retrieval, reranking, generation and safety filters? [Likely] [28]
77. When would you route a request to a small model and when to a large one, and how would you learn the router? [Likely]
78. Design a training-data pipeline for LLM pretraining: tokenization, deduplication and quality filtering. [Reported] (Amazon MLE aggregate) [23]
    - How do you detect near-duplicates at web scale?
    - How do you measure the downstream effect of a quality filter?

## J. Multimodal and speech (team-dependent)

79. Compare a dual-encoder (CLIP-style) model with a vision-language model that uses a resampler or adapter. [Reported] (Amazon RS aggregate) [22]
    - Which would you use for product image search, and which for visual question answering on a listing?
80. How does the Segment Anything Model work? [Reported] [3][10]
81. How do diffusion noise schedules affect sample quality and training? [Reported] (Amazon RS aggregate) [22]
82. What causes codebook collapse in VQ-VAE, and how do you fix it? [Reported] (Amazon RS aggregate) [22]
83. Image-text embeddings and LLM inference engines (online-assessment topics on some science pipelines). [Reported] (Amazon RS aggregate) [22]
84. For a speech-to-speech assistant (in the spirit of Nova Sonic or Alexa+), what changes when you move from an ASR, then LLM, then TTS cascade to an end-to-end speech model? [Likely] [25]
    - How do you evaluate turn-taking, barge-in and latency?
85. How would you improve Alexa speech recognition for non-native speakers and users with strong regional accents, and could an LLM rescoring stage help? [Reported for the base question] [32]; LLM follow-up [Likely]

## K. Applying GenAI to Amazon problems (science-application flavor)

86. On an LLM team, discuss LLM decoding and training choices in the context of an Amazon use case. [Reported] [8]
87. Here is a high-level problem from our team. Would you solve it with an LLM at all? Propose a solution: data sources, annotation strategy, modeling, evaluation, trade-offs and risks. [Reported for the frame] [33]; the LLM-or-not framing is [Likely]
    - What is the cheapest baseline that might be good enough?
88. Use an LLM to categorize millions of products into a large taxonomy. How do you combine LLM labeling, a distilled classifier and human audit? [Likely] (product categorization is reported in classical form in earlier chapters; LLM distillation context [31])
89. Use an LLM to generate or enrich product listings (titles, bullet points, attributes) from seller input. How do you prevent made-up attributes? [Likely]
90. Use LLMs as relevance judges for product search. How do you validate them against human judges and behavioral signals? [Likely]
91. How would you personalize Rufus answers with a customer's purchase history while respecting privacy? [Likely] [29]
    - How would you measure whether personalization helps rather than creeps people out?
92. Sponsored products in a generative answer: how do you blend ads into an LLM response while keeping trust, and how do you measure the impact? [Likely]
93. Design an LLM-based system to detect policy-violating content or fraud in reviews or ads. [Likely] (Advertising Trust context [18]; content moderation listed as an Amazon AI-assisted task [24])
    - How do you deal with adversaries who adapt to the classifier?
94. Design a system that turns customer-support transcripts into structured issue reports. How do you evaluate extraction quality? [Likely]
95. A team wants to replace an existing BERT classifier with an LLM prompt. What do you need to see before you agree? [Likely]
    - How do you compare cost per 1,000 requests, latency and accuracy on the long tail?

## L. Depth probes on your own LLM project (depth-round style)

96. Walk me through your most significant LLM or NLP project, including the GenAI fundamentals behind it and its business impact. [Reported] [18]
97. Why did you choose this base model and size? What would you do with 10x more compute? [Reported] (Amazon RS aggregate: backbone choice, use of extra compute) [22]
98. How did you build the evaluation set, and how do you know it reflects real customer traffic? [Likely]
99. What did your error analysis show, and which fix had the biggest effect? [Likely]
    - What did you try that did not work?
100. If you fine-tuned, how did you rule out leakage between training data and evaluation prompts? [Likely]
101. Where would your approach break if deployed to a different marketplace or language? [Likely]
102. How did you decide the project was "done" and ready to ship (Deliver Results, Insist on the Highest Standards)? [Likely]

## M. GenAI behavioral and LP crossovers

103. How do you use AI tools in your work, and when did one give you a wrong answer? How did you catch it? [Reported] (Amazon general aggregate, phone screen) [24]
104. Tell me about a time you pushed back on using an LLM where a simpler model was better (Frugality, Invent and Simplify). [Likely]
105. Tell me about a time an LLM feature you shipped behaved badly in production. What did you do in the first 24 hours (Ownership, Dive Deep)? [Likely]
106. Tell me about a time you had to explain LLM risks (hallucination, safety) to a non-scientist stakeholder (Earn Trust). [Likely]

---

## Coverage notes for prep

- **Highest-yield set across reports:** transformer and attention mechanics; KV cache and decoder inference; long context; BERT vs GPT; LoRA; SFT and DPO vs RLHF; beam search and sampling; bi-encoder vs cross-encoder; RAG design; and agent and LLM evaluation (questions 1-2, 10, 15-19, 22-25, 30, 37-38, 45-49, 55) [3][4][7][8][10][16][19].
- **Team-specific tilt:**
  - AGI/Nova teams: training, RL, parallelism and ML coding [15][17][22][25].
  - Rufus/Stores: RAG, grounding, ranking and evaluation [28][29][30].
  - Alexa: multilingual alignment, tool use and speech [27][32].
  - AWS/Bedrock-adjacent science: PEFT, serving cost and customization [4][26].
- **Aggregator caveat:** questions marked "(Amazon RS aggregate)" or "(Amazon MLE aggregate)" come from compilations of Amazon Research Scientist and ML Engineer reports [22][23]. Those loops overlap heavily with AS loops but are not identical.

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
10. Glassdoor individual Amazon interview reviews, 2026 (one reports many "modern LLM training technique" questions for an AGI-team role, and one reports a Segment Anything question; which review holds which question could not be verified). https://fr.glassdoor.ch/Entretien/Amazon-Entretien-E6036-RVW104288349.htm ; https://fr.glassdoor.ch/Entretien/Amazon-Entretien-E6036-RVW104864395.htm ; https://www.glassdoor.fr/Entretien/Amazon-Entretien-E6036-RVW101665167.htm
11. "Amazon online 1st round, Bengaluru, Applied Scientist", LeetCode Discuss, May 2025 (LLM theory, transformer internals). https://leetcode.com/discuss/post/6871782/amazon-online-1st-round-bengaluru-applie-uiwk/
12. "Amazon Applied Scientist Phone Interview", LeetCode Discuss (LSTM vs Transformers mathematically). https://leetcode.com/discuss/post/543684/amazon-applied-scientist-phone-interview/
13. "Amazon Applied Scientist (L5) Virtual Onsite", Blind, circa 2025. https://www.teamblind.com/post/amazon-applied-scientist-l5-virtual-onsite-navwesml
14. "Applied scientist interview at Amazon Ads", Blind (rejected after not preparing attention, transformers and LLMs). https://www.teamblind.com/post/applied-scientist-interview-at-amazon-ads-zoerec5t
15. "Amazon (AGI team) Applied Scientist II interview", Blind. https://www.teamblind.com/post/Amazon-AGI-team-Applied-Scientist-II-interview-2eqBXGtT
16. "Amazon Onsite Interview Experience for Fulltime Machine Learning Role", 1point3acres (enterprise RAG design). https://www.1point3acres.com/interview/thread/1157086
17. "Amazon AGI Applied Scientist Interview Experience and Insights", 1point3acres. https://www.1point3acres.com/interview/thread/1130091
18. Chaitanya Mundle, "My interview experience for the Applied Scientist II role at Amazon" (Advertising Trust; Science Depth = resume plus GenAI fundamentals plus business impact), Medium, circa 2024. https://medium.com/@mundle1997.cm/my-interview-experience-for-the-applied-scientist-ii-role-at-amazon-738aba21f611
19. "Amazon Applied Scientist Interview Debrief: Five-Round Loop + LP", OA VO Service (vendor-published candidate debrief, promotional; SFT pipeline; L5 vs L4 follow-up depth). https://oavoservice.com/en/articles/amazon-applied-scientist-vo-loop-lp-ml-coding
20. "Amazon Interview Experience for Applied Scientist 2", Enginebogie, circa 2025 (knowledge distillation; limited labels). https://enginebogie.com/interview/experience/amazon-applied-scientist-2/1004
21. "Amazon Applied Scientist Interview Questions & Guide 2026", Dataford (prep vendor; agentic recommendation system design), 2026. https://dataford.io/interview-guides/amazon/applied-scientist
22. "Amazon Research Scientist Interview Process", kevin-2023-code/Tech-Interview-Questions, GitHub (aggregator adapted from trueinterview.io, CC BY 4.0), 2026; fetched and read for this chapter. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/amazon/interview-process-research-scientist.md
23. "Amazon Machine Learning Engineer Interview Process", kevin-2023-code/Tech-Interview-Questions, GitHub (aggregator adapted from trueinterview.io), 2026; fetched and read for this chapter. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/amazon/interview-process-machine-learning-engineer.md
24. "Amazon Interview Process", kevin-2023-code/Tech-Interview-Questions, GitHub (aggregator; GenAI-usage behavioral question; AI-assisted coding tasks), 2026; fetched and read for this chapter. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/amazon/interview-process.md
25. "AI Companies Interview Questions" (Amazon AGI section: reported 2025 loop, "debug a broken Transformer implementation", Nova products and the 2026 org note), ombharatiya/FAANG-Coding-Interview-Questions, GitHub (aggregator; underlying reports not individually named), 2026; fetched and read for this chapter. https://github.com/ombharatiya/FAANG-Coding-Interview-Questions/blob/main/AI-Companies-Interview-Questions.md
26. "amazon.md", ombharatiya/AI-Engineer-Interview-Questions, GitHub (explicitly synthesized from Amazon focus areas, not leaked questions; used only for [Likely] Bedrock-shaped items), 2026; fetched and read for this chapter. https://github.com/ombharatiya/AI-Engineer-Interview-Questions/blob/main/14-company-interview-questions/amazon.md
27. Applied Scientist, Alexa International Tech (Job ID 10490052), amazon.jobs (SFT, DPO, RLHF, RLAIF). https://www.amazon.jobs/en/jobs/10490052/applied-scientist-alexa-international-tech
28. Applied Scientist, Rufus Experiences Science (Job ID 2855826), amazon.jobs. https://amazon.jobs/en/jobs/2855826/applied-scientist-rufus-experiences-science
29. "Amazon Rufus AI assistant personalized shopping features", About Amazon (LLMs including Amazon Nova on Bedrock; per an earlier chapter's search extract, renamed Alexa for Shopping in May 2026). https://aboutamazon.com/news/retail/amazon-rufus-ai-assistant-personalized-shopping-features
30. "What do I need for cold weather golf?" (Amazon Rufus), IEEE Spectrum. https://spectrum.ieee.org/amazon-rufus
31. "Learning with less: Knowledge distillation from large language models via unlabeled data", Amazon Science. https://www.amazon.science/publications/learning-with-less-knowledge-distillation-from-large-language-models-via-unlabeled-data
32. "Amazon Applied Scientist Interview Guide: Alexa International Tech", Medium (Interview Preparation), Apr 2025; plus the companion "Science application: Business problems for Amazon Applied Scientist (Alexa International Tech)", Medium, 2025 (accented-speech ASR). https://medium.com/interview-preparation/amazon-applied-scientist-interview-guide-alexa-international-tech-022d3a056feb ; https://medium.com/@pragyashukla2580/science-application-business-problems-for-amazon-applied-scientist-alexa-international-tech-9353a52ca556
33. "Amazon Applied Scientist Interview (process, questions, prep)", IGotAnOffer, 2025-2026. https://igotanoffer.com/en/advice/amazon-applied-scientist-interview
34. Blind, "Amazon AS L4 vs L5" and "Amazon Applied Scientist L4 vs Applied Scientist II L5" (level-bar discussion). https://www.teamblind.com/post/amazon-as-l4-vs-l5-n0zunqgt ; https://www.teamblind.com/post/amazon-applied-scientist-l4-vs-applied-scientist-ii-l5-aasmppdi
