# 11. Amazon Science Team Context: What Each Hiring Org Asks, and the Recent Amazon Science Work to Know

> Scope: the team-specific part of the L5 Applied Scientist (Applied Scientist II) onsite. That means the hiring-manager / Science Application round, the "why this team" probes, and the "have you seen our work?" questions. It covers the orgs that most often hire L5 AS: Search (Stores Foundational AI, formerly M5), Ads, Alexa AI / AGI (Nova), AWS AI, Rufus, Personalization and recommendations, SCOT (supply chain), Pricing, and Prime Video. It also lists recent Amazon Science publications and open-source releases, framed as questions an interviewer might ask. This chapter is questions only, with no answers.
>
> Research note (please read): in both the first pass and the October 2026 review pass, the shared web-search budget was used up and only github.com could be reached. Glassdoor, Blind, Medium, LeetCode, 1point3acres, amazon.jobs, amazon.science, aboutamazon.com, IEEE Spectrum and the prep-vendor sites did not resolve from this environment. The evidence comes from two places:
> - Candidate-report findings already gathered and cited in chapters 01-09 of this guide. Those were built from search-result extracts of the cited pages, not full reads. Each [Reported] tag here cites the original page, and I tag a question [Reported] only when one of those chapters attributes it to an Amazon science interview. These non-GitHub citations could not be re-fetched in the review pass, so treat them as unverified.
> - Pages fetched and read directly: the amazon-science GitHub organization and more than 20 of its repositories [33]-[49], [60]-[69], and four pages from two GitHub interview aggregators [19][20][21][59]. The review pass re-read the aggregators. As a result, it retagged two items that the aggregators list only as round topics, not as asked questions (RoPE/ALiBi/KV caching, and the fraud-metrics item), and it fixed the DeepMTL2R method count.
>
> **Team-specific first-person reports for Prime Video and Pricing were not found, so every question in those sections is [Likely].** Before your loop, ask your recruiter for the team's name and re-check its newest postings and papers. A follow-up session with a fresh search budget could add Blind and 1point3acres threads for each org.

## Format notes for this round

- **Where team context shows up.** Every L5 loop includes a team-tailored round, usually called "Science Application" or "ML System Design (team-specific)", and it is often run by the hiring manager or a senior scientist from the hiring team. One 2025-2026 AS II loop was reported as "1 Science ML Breadth, 1 Science Depth, 1 Sys Design (Team-specific), 1 coding round and 1 Bar Raiser" [2]. Amazon's own prep page describes about four 55-minute interviews with members of the science community. It says each interviewer is assigned competencies and asks 2-3 behavioral (Leadership Principle) questions [1]. Blind commenters say L5 loops usually include the application/design round and L4 loops may not [53].
- **Duration and interviewer.** Rounds run 55-60 minutes. Expect about 40 minutes on the team problem and 10-15 minutes on LPs [1][52]. The HM also judges team fit and, with the bar raiser, makes the final call [53].
- **What the L5 bar looks like here.** You take a vague problem from the team's domain and frame it end to end: data and labels, model, offline and online metrics, launch, and monitoring. You make the trade-off calls yourself [17][52]. Depth on the team's own stack, and awareness of what the team has published, separate a "strong hire" from a "hire" in this round. That point is an inference, not a published rubric.
- **The team can change after the loop.** Some candidates pass the loop and are then team-matched or "recycled" to another org's opening [54]. Prepare a 2-minute pitch on how your work maps to at least two orgs.
- **GenAI everywhere.** Even on non-LLM teams, 2025-2026 loops treat transformers and LLMs as core breadth. An Ads candidate reported being rejected after not preparing for them [5][50].
- **Org churn (treat as unverified).** One GitHub aggregator reports that Amazon's AGI org was reorganized in 2025-2026 and that Nova 2 Lite, Nova 2 Sonic, Nova Forge and Nova Act remain active products [21]. An earlier chapter's search extract says Rufus was renamed "Alexa for Shopping" in May 2026 (see [27]). Confirm the current org and product names with your recruiter.

Tag legend: **[Reported]** means a cited source attributes the question, or a close paraphrase, to an Amazon science or ML interview. Items from aggregators are marked "(aggregator)", and items from Research Scientist or MLE loops are marked "(RS loop)" or "(MLE loop)". **[Likely]** means a grounded inference from the team's products, job postings or Amazon Science publications. It has not been seen verbatim.

---

## A. Cross-team "why this team" and team-fit questions

1. Why Amazon, why Applied Science, and why this particular team? [Likely]
   - Which of our team's products have you used as a customer, and what would you change first?
   - Which of our team's papers or launches have you read?
2. How would you apply your research or project to a problem at Amazon? [Reported] [55]
   - Which team's metric would your work move, and by how much, roughly?
3. (Recruiter or HM screen for a GenAI team.) What hands-on experience do you have with LLM post-training or with building agents? [Reported] (aggregator) [59]
   - Which part did you personally own: data, reward design, training infrastructure or evaluation?
4. Here is a high-level ML problem from our team. Propose data sources, an annotation strategy, a modeling technique, evaluation metrics, and the trade-offs and risks. [Reported] [17]
   - What is the version you would ship in one quarter, and what would v2 add?
5. Answer an ML application question about one of our Amazon products. [Reported] [18]
6. Pick an agentic product we could build. Who is the customer, and how do you measure success? [Reported] (Seattle, 2026) [2]
   - Write the press-release headline (Working Backwards).
7. If the loop goes well but this team's headcount closes, which other Amazon science org would you want, and why? [Likely] [54]
8. What science problem in our org do you think is unsolved, and how would you start on it? [Likely]
   - How would you write the 6-page document proposing it?

## B. Search (Stores Foundational AI / Search Science, formerly M5)

9. You have a large collection of user click data for many search queries and product documents. Is click data alone sufficient? How would you train a retrieval or IR model with the data at hand? [Reported] [13]
   - What position, presentation and trust biases contaminate the clicks?
   - How would you estimate relevance from click logs while accounting for query intent? [Likely] (Amazon Science work on intent-based relevance from click logs [29])
10. Walk through how you would design, train, validate and deploy a ranking model for product search. [Reported] [56]
    - Pointwise, pairwise or listwise learning-to-rank: which, and why?
11. Walk through a ranking project you shipped: the customer pain point, why that model, which metrics you used, and how you diagnosed a disappointing A/B result. [Reported] (Amazon MLE loop project deep dive, aggregator) [20]
12. Compare a small and a large model from the same text-embedding family. What do you gain and what do you pay, for first-stage product retrieval? [Reported] (RS breadth round, aggregator) [19]
13. Compare approaches to building joint image-text embeddings, for example for visual search or for matching product photos to queries. [Reported] (RS online assessment, aggregator) [19]
    - Implement a numerically stable InfoNCE loss that is symmetric across both directions (image to text and text to image) for a batch of paired embeddings. [Reported] (science coding round, aggregator) [19][20][59]
    - How does batch size interact with in-batch negatives, and how would you mine hard negatives from the catalog?
14. Answer a search-relevance application question for Amazon search. [Reported] [3]
15. Compare bi-encoders and cross-encoders. Where does each sit in a product-search stack? [Reported] [3]
    - How would you distill the cross-encoder into something that meets a retrieval latency budget?
16. You have 10k product listings in 100 categories, each with a title, description and photo. How would you categorize them? [Reported] [13]
    - How do you handle a long tail of categories with very few examples?
17. Design an end-to-end retrieval and ranking pipeline with candidate generation, serving, offline metrics versus online guardrails, A/B testing and position bias. [Reported] (Amazon search/recs/ads MLE loop, aggregator) [20][59]
18. Design a multi-turn search agent that handles long context. [Reported] (RS loop, aggregator) [19]
    - When would you summarize earlier turns, and when would you retrieve them instead?
19. The ESCI Shopping Queries dataset labels query-product pairs as Exact, Substitute, Complement or Irrelevant. Why is a 4-way label more useful than binary relevance for Amazon search? [Likely] [34]
    - How would you use Substitute and Complement labels differently in ranking and in recommendations?
    - The dataset covers English, Japanese and Spanish. How would you transfer a relevance model to a new marketplace?
20. How would you rank with several objectives at once (relevance, conversion, margin, delivery speed)? [Likely]
    - Would you use multi-task learning-to-rank, and how would you weight the tasks (for example uncertainty weighting, PCGrad or Nash-MTL)? [Likely] (Amazon's DeepMTL2R library lists 21 multi-task weighting methods plus a single-task baseline [45])
21. (Search/IR postings list learning-to-rank and recommender work [26].) How would you select a set of results that is both relevant and diverse from millions of candidates, faster than MMR? [Likely] (MUSS, UAI 2026 [39])
    - When does diversity hurt a high-intent query like a model number?
22. How would you use an LLM as a relevance judge for product search, and how would you validate it against human annotators? [Likely]
23. How would you detect and fix wrong attribute values (size, color, material) in catalog listings that hurt search? [Likely]
    - SynthAVE labels (attribute, value) pairs as Correct, Incorrect or Unknown, using majority vote over 21 LLM-judge configurations (7 model families with 3 prompts each). When would you trust an LLM arena as a labeler? [Likely] (EMNLP 2026 Industry [38])
    - In SynthAVE about 39% of pairs are Unknown (the text cannot decide). How would you treat "not determinable" in training and in your metric? [Likely] [38]
24. Design a system that detects duplicate or variant listings of the same product from different sellers (title, attributes and images), so search does not show the same item five times. [Likely]
    - How do you set the merge threshold when a false merge is costlier than a missed one?

## C. Advertising (Sponsored Products, DSP, Ads Trust, Ads measurement)

25. Explain how transformers and attention work. How does a decoder-only LLM generate text at inference time? (An Ads candidate reported being rejected after not preparing this.) [Reported] [5]
26. Design a system that recommends trustworthy ads to Amazon users while maintaining brand safety and compliance. [Reported] (Advertising Trust) [8]
    - Walk through the full data pipeline from collection to deployment. [Reported] [8]
    - How do you get labels for rare policy violations?
27. Science Depth on an Ads loop: your resume, plus GenAI fundamentals, plus the business impact of your work. [Reported] [8]
28. Amazon Ads generates image creatives for advertisers. Design an evaluation framework that checks whether a generated image spells the brand name correctly and places it naturally. [Reported] (RS and MLE science-application / ML design round, aggregator) [19][20][59]
    - How would you combine OCR, a learned "naturalness" judge and human review, and how would you calibrate the judge?
29. Design a click counter over a sliding k-minute window, then an advertiser-metrics service with campaign and event entities, windowed pre-aggregation and idempotency for duplicate events. [Reported] (Ads MLE loop, aggregator) [20]
30. Design click-through-rate or conversion-rate prediction for Sponsored Products. [Likely]
    - Why does calibration matter for auction pricing?
    - How do you handle delayed conversions?
31. How would you learn a bidding policy offline from logged auction data? [Likely] (AuctionGym, KDD 2023 ADS track [37])
    - Why is offline evaluation of learning-to-bid hard (unobserved confounding, costly experiments)?
    - How does optimal bidding differ between second-price and first-price auctions, and what is bid shading?
32. How would you measure the incremental lift of an ad campaign for an advertiser? [Likely]
    - Ghost ads, geo experiments or observational causal methods: what are the trade-offs?
33. How would you blend sponsored products into a generative shopping answer without eroding trust, and how would you measure the effect? [Likely]
34. Design a budget-pacing algorithm for advertisers. How would you frame it as constrained optimization or as a bandit? [Likely]

## D. Alexa AI, AGI and Amazon Nova (foundation models)

35. Expect several "modern LLM training technique" questions, with follow-ups on each stage (pretraining, SFT, preference tuning). [Reported] (AGI-team role) [6][7] (see chapter 07)
36. You are handed a Transformer implementation that trains but performs badly. Debug it. [Reported] (Amazon AGI, aggregator) [21]
    - Which bugs in masking, scaling, positional encoding or normalization produce silent quality loss?
37. Build and train a small classifier live. [Reported] (Amazon AGI, aggregator) [21]
38. Reinforcement-learning fundamentals plus coding. [Reported] (Amazon AGI, aggregator) [21]
    - Why does GRPO remove the critic, what does that cost, and how would you design the reward? [Reported] (RS phone screen, aggregator) [19]
    - How would you parallelize GRPO training? [Reported] (MLE loop, aggregator) [20]
    - Explain PPO's clipped objective. [Likely] [19]
39. Write the rotary position embedding (RoPE) formula and explain how it keeps relative position inside attention. [Reported] (MLE loop, aggregator) [20][59]
    - Compare RoPE with ALiBi, and explain KV caching and pipeline parallelism. [Likely] (listed as round topics, not as asked questions, in the RS aggregator [19])
40. For a residual MLP of width 4096 in BF16, is batch size 1 or 256 compute-bound, and what does that imply for serving an assistant model? [Reported] (MLE breadth round, aggregator) [20]
41. Architect an AI chatbot, covering intent detection, dialogue state, generation, fallback behavior, evaluation and continuous learning. [Reported] (RS and MLE ML-design rounds, aggregator) [19][20]
    - Where does a classic NLU intent model still beat an LLM for an Alexa-style assistant?
42. (Stretch, reported from a staff-level loop, so above the L5 bar.) Design a distributed pipeline that ingests, tokenizes, deduplicates and filters pretraining data for a foundation model. [Reported] (aggregator) [20]
    - How would you measure the effect of a quality filter without a full pretraining run?
43. Discuss LLM decoding and training choices in the context of an Amazon use case. [Reported] (AS II, LLM team, Santa Clara) [12]
    - How does beam search work, and when does it hurt open-ended generation? [Reported] [12]
44. How would you improve Alexa speech recognition for non-native speakers and users with strong regional accents? [Reported] (Alexa International Tech) [10]
    - Where would you get diverse accent data, and how would you handle phonetic variation? [Reported] [10]
45. The Alexa International loop is breadth, depth, science application, coding and a bar raiser, with LPs throughout. What would you expect the science-application problem to look like for a multilingual assistant? [Reported for the format] [9]; the follow-up is [Likely]
46. How would you align a multilingual assistant with SFT, DPO, RLHF or RLAIF when preference data is scarce in some locales? [Likely] (the Alexa International posting lists these methods [23])
47. Instruction-tuned LLMs are trained on text preferences, but Alexa speaks its answers. What makes a response "speechworthy", and how would you collect preferences for it? [Likely] (Speechworthy Instruction-tuned LMs, EMNLP 2024 [42])
48. AlexaTM 20B was a multilingual seq2seq model pretrained with denoising plus causal LM. When would an encoder-decoder beat a decoder-only model for an Alexa task? [Likely] [40]
49. Training LLMs with MXFP4: why can the backward pass run in 4-bit with stochastic rounding and a random Hadamard transform and still give unbiased gradients? [Likely] (AISTATS 2025 [41])
    - What breaks if rounding is biased?
50. Moving from an ASR, then LLM, then TTS cascade to an end-to-end speech model (in the spirit of Nova Sonic): what changes in latency, evaluation and failure modes? [Likely] [21]
51. Design a browser-operating agent (in the spirit of Nova Act) that completes multi-step web tasks reliably. [Likely] [21]
    - How do you build its evaluation set and measure step-level versus task-level success?
52. An Alexa+ style assistant calls real device and service APIs (timers, smart home, ordering). How do you make tool calls reliable and safe? [Likely]
53. How would you make an on-device wake-word or NLU model 10x smaller with minimal quality loss? [Likely]

## E. AWS AI (Bedrock, SageMaker, Q / coding agents, AI research)

54. Explain transformers, then explain LoRA. [Reported] (AWS AS loop) [4]
    - Where are the LoRA adapters inserted, and what does the rank control?
    - Explain the main parameter-efficient fine-tuning techniques, which parameters each trains, and when each beats full fine-tuning. [Reported] (RS and MLE online assessment, aggregator) [19][20]
55. Explain how an inference engine such as vLLM raises LLM serving throughput (paged KV cache, continuous batching). [Reported] (MLE online assessment, aggregator) [20]
    - Amazon Science's ammo uses multiple agents to optimize vLLM GPU kernels for one deployment. How would you check that an auto-generated kernel is both correct and faster? [Likely] [49]
56. Design an LLM quality-validation system that gates model or prompt updates before they reach customers. [Reported] (title listed for the RS science-application round, aggregator) [19]
57. Design an enterprise RAG service. [Reported] [51]
    - How do you enforce document-level access control in retrieval?
58. Design a GitHub Copilot-style coding agent. [Reported] (Senior AS, Boston) [3]
    - SWE-PolyBench scores agents on repository-level tasks in Java, JavaScript, TypeScript and Python with pass rate and file- and node-retrieval metrics. Why measure retrieval separately from pass rate? [Likely] (arXiv 2025 [46])
59. Evaluate an LLM agent with metrics beyond accuracy: tool use, hallucination and calibration. [Reported] [3]
60. RAGChecker separates retriever metrics (claim recall, context precision) from generator metrics (faithfulness, noise sensitivity, hallucination, context utilization). How would you use that split to debug a customer's RAG system? [Likely] (NeurIPS 2024 Datasets and Benchmarks [35])
61. RefChecker checks LLM outputs at the level of (subject, predicate, object) claim triplets in zero, noisy and accurate context settings. Why triplets rather than sentences? [Likely] (2024 [36])
    - Where does triplet extraction itself fail?
62. A customer's Bedrock-hosted workload costs too much. How would you cut inference cost sharply without unacceptable quality loss? [Likely] [21] (see chapter 07)
63. Chronos turns a time series into tokens by scaling and quantization, then trains a language model on them. Why does that work, and what does Chronos-Bolt change (patches, direct multi-step quantile output)? [Likely] (TMLR 2024; Chronos-Bolt Nov 2024 [33])
    - Chronos-2 (Oct 2025) adds multivariate and covariate-informed zero-shot forecasting. How would you evaluate it fairly against a tuned, dataset-specific model?
64. Design an agent that migrates a whole Java 8 repository to Java 17. How do you decide that the migration succeeded? [Likely] (MigrationBench, KDD 2026 [60])
    - MigrationBench approximates functional equivalence: the build passes, compiled class versions match the target, test methods are unchanged and the test count does not fall. What can still go wrong under these checks?
65. Coding agents run inside customer workspaces. How would you red-team one for jailbreaks that hide malicious intent in a half-finished multi-file project rather than in the prompt? [Likely] (JAWS-Bench, arXiv 2510.01359 [61])
66. A customer asks a Bedrock-hosted assistant about AWS services whose behavior changed after the model's training cutoff. How would you detect and handle stale knowledge? [Likely] (evolveQA, EMNLP 2026 [62])
67. Design a hallucination detector for RAG answers over very long contexts (around 90K characters), with sentence-level labels (Supported, Contradicted, Not Mentioned, Supplementary). [Likely] (TRIVIA+, ACL 2026 [63])
    - Why would you inject controlled label noise into the benchmark?
68. Synthesize realistic tabular data, or detect anomalies in tabular data, with generative models. What are the evaluation pitfalls? [Likely] (amazon-science TabSyn and AnoLLM repositories [49])

## F. Rufus (generative shopping assistant)

69. Design Rufus: answer "What do I need for cold weather golf?" using the catalog, reviews and community Q&A. [Likely] [22][28]
    - How do you ground answers in product facts and avoid hallucinated specs?
    - How do you evaluate offline before an A/B test, and what is the online success metric?
70. Design an agentic recommendation system for Amazon, covering components, model choices, and quality and safety evaluation. [Reported] [16]
71. Rufus runs on several LLMs, including Amazon Nova, on Bedrock. How would you route each query between a small and a large model? [Likely] [27]
72. How would you personalize answers with a customer's purchase and browse history while respecting privacy? [Likely] [27]
73. You have one latency budget for an assistant answer (say, first token in under one second). How do you split it across retrieval, reranking, generation and safety filters? [Likely] [22]
74. How would you summarize thousands of reviews into a short "Customers say" blurb, and measure faithfulness and coverage of minority opinions? [Reported] (Science Application round, Luxembourg, reported as "summarize product reviews into key concepts") [70]
75. Personalized autocomplete for a chat interface: how would you rank candidate completions, and would you report saved-keystrokes@k or acceptance-rate@k? [Likely] (amazon-science personalized chat autocomplete experiments [49])
76. How would you red-team a shopping assistant for unsafe, biased or policy-violating answers? [Likely]

## G. Personalization and recommendations (homepage, Amazon Music, explanations)

77. Design a pipeline to improve personalized product recommendations on the Amazon homepage. [Reported] [16]
    - What is the north-star metric, and what are the guardrails?
78. A user goes from viewing to clicking to purchasing. Model the probability of each action given a user and a product. [Reported] [14]
    - Three independent models, or a joint model? [Reported] [14]
    - How would you build the dataset and down-sample (for example with reservoir sampling)? [Reported] [14]
79. Design a recommender system end to end. [Reported] (reported as the design round in an AS loop) [57]
    - How do you handle cold start for new users and new ASINs?
80. You have logged rankings from the Amazon Music home page with logging-policy propensities. How would you estimate a new ranking policy's value offline? [Likely] (Music off-policy evaluation benchmark [44])
    - Compare IPS, self-normalized IPS and doubly-robust estimators for slate rankings.
    - How do you check whether your OPE estimate matches the online A/B result?
81. You have a pool of LLM-generated explanations for a recommendation. Would you choose among them with RL fine-tuning (PPO, GRPO, DPO) or with a pairwise ranker? [Likely] (RecSys 2026 finding that LambdaRank beat single-action RL [47])
82. How would you avoid feedback loops that recommend only already-popular items? [Likely]
83. How would you personalize email or push notifications as uplift modeling rather than response modeling? [Likely]

## H. SCOT (Supply Chain Optimization Technologies), forecasting and inventory

84. What is the newsvendor problem, and how would you use it to set inventory levels? [Reported] (Amazon supply-chain interviews) [15]
    - How does a quantile forecast map to the critical ratio?
85. Model an optimization problem from your own work: objective, decision variables and constraints. [Reported] [52]
    - When would you use a heuristic instead of an exact solver?
86. How would you forecast demand for millions of products, including new ones with no history? [Likely] [30]
    - Why do buying decisions need probabilistic or quantile forecasts rather than point forecasts?
    - Would you use a global model or per-item models?
87. Design a system that decides how much of each product to buy, from which vendor, and when. [Likely] (SCOT Optimal Sourcing / Buying Science posting [24])
88. Design inventory placement across fulfillment centers. How do forecast errors propagate into transportation cost? [Likely] [31]
89. How do you forecast through Prime Day when history is sparse and demand was censored by stock-outs? [Likely]
90. Could a pretrained foundation model like Chronos replace your team's per-domain forecaster? What evidence would convince you? [Likely] [33]
91. Design a delivery-promise prediction (the date shown on the detail page). What loss would you use if late costs more than early? [Likely] (Last Mile candidate context [11])
92. How would you run an experiment on a buying policy when products share warehouses and trucks (interference)? [Likely]

## I. Pricing and causal inference

93. Design a detector that catches pricing errors (for example a $500 item listed at $5) before they go live, across millions of ASINs and sellers. [Likely]
    - How do you trade off blocking a real deal against letting a costly error through?
94. Design a system that recommends competitive prices for millions of products. [Likely]
    - How do you estimate price elasticity when your own algorithm set the prices (endogeneity)?
    - What guardrails prevent price spirals and loss of customer trust?
95. How would you measure the effect of a pricing algorithm when prices interact across products? [Likely]
    - Switchback, cluster-randomized or synthetic-control designs: when would you use each?
96. How would you validate a causal model before trusting it? [Likely] (Amazon's causal-validation package simulates data with a known effect and supports placebo tests [48])
    - Design a placebo test for a synthetic-control estimate of a price change.
97. How would you design an A/B test for a model launch, and what would you do if it is underpowered? [Reported] [58]
98. How would you decide whether to discount a product for a deal event, accounting for cannibalization and pull-forward demand? [Likely]
99. When would you use double ML, instrumental variables or difference-in-differences for a promotion's effect? [Likely]

## J. Prime Video (and other Amazon content and devices teams)

No first-person Prime Video AS reports were found in this run. Everything below is [Likely].

100. Design recommendations for Prime Video, where viewing is sequential and feedback is implicit. [Likely]
    - Is a show watched for 3 minutes a positive or a negative label?
101. How would you choose artwork or thumbnails per user, and evaluate the policy offline? [Likely]
102. How would you detect video-quality problems (buffering, artifacts, audio-video sync) at scale from playback telemetry? [Likely]
103. Design ad insertion for ad-supported streaming: where to place breaks, and how to measure viewer drop-off. [Likely]
104. How would you forecast live-sports streaming demand for capacity planning? [Likely]
105. How would you detect and skip intros and recaps, or generate scene-level metadata, with video models? [Likely]

## K. Payments, trust, fraud and abuse

106. Design a model to predict payment-method failure and pick the best fallback for the customer. [Likely] (Amazon Payment Products L5 posting [25])
107. Train and evaluate a fraud model under extreme class imbalance, delayed chargeback labels and adaptive adversaries. [Likely]
    - Amazon's Fraud Dataset Benchmark groups public datasets by fraud type (card-not-present, bots, malicious traffic, content moderation, credit risk). How would you check that a method generalizes across them? [Likely] [43]
108. Report precision, recall and AUC for imbalanced fraud data, and say which you would optimize. [Likely] (the RS aggregator lists imbalanced metrics as a round topic, not as an asked question [19])
109. Use computer vision to detect defective products in fulfillment-center warehouses. [Reported] [18]

## L. "Have you read our work?" Recent Amazon Science publications as interview questions

These are [Likely] prompts. HMs often ask "What do you think of X?" about their own team's work. Read the abstract and the evaluation section of anything in your target org.

110. Chronos (TMLR 2024), Chronos-Bolt (Nov 2024) and Chronos-2 (arXiv, Oct 2025; released to SageMaker JumpStart in Dec 2025 and AutoGluon-Cloud in Jun 2026): what is the weakest part of the "language of time series" argument? [Likely] [33]
111. Shopping Queries Dataset (ESCI, arXiv 2022, used in a KDD Cup): what does ESCI miss that real Amazon search needs (personalization, freshness, session context)? [Likely] [34]
112. RAGChecker (NeurIPS 2024 D&B) and RefChecker (2024): how would you meta-evaluate an automatic hallucination checker against human judgment? [Likely] [35][36]
113. AuctionGym (KDD 2023 ADS; AdKDD 2022 best paper): what does a simulator buy you that logged data cannot, and what does it cost you? [Likely] [37]
114. SynthAVE (EMNLP 2026 Industry): what are the risks of evaluating on synthetically generated products with anonymized brands? [Likely] [38]
115. MUSS (UAI 2026): MMR is a special case of MUSS with one partition. What does the constant-factor approximation guarantee buy you in practice? [Likely] [39]
116. Training LLMs with MXFP4 (AISTATS 2025): why target the backward pass first? [Likely] [41]
117. Speechworthy instruction-tuned LMs (EMNLP 2024): how would you extend a listening-based preference dataset to multi-turn voice? [Likely] [42]
118. Multimodal chain-of-thought (mm-cot) and Auto-CoT: are these still relevant once reasoning models are trained with RL? [Likely] [49]
119. Pairwise ranking beats single-action RL for offline explanation selection (RecSys 2026): when would you expect that result to flip? [Likely] [47]
120. Amazon Science's 2026 open-source releases include agent self-improvement recipes (Ratchet, Double Ratchet), RL-training kernels (DualKV shared-prompt FlashAttention), multi-agent vLLM kernel optimization (ammo) and long-context hallucination benchmarks (ACL 2026). Pick one relevant to this team and critique it. [Likely] [49]
121. Select, Label, Evaluate: Active Testing in NLP (EMNLP 2026) reports performance estimates within 1% of the full test set using up to 95% fewer labels. With a fixed annotation budget, how would you choose which examples to label to estimate a new model's accuracy? [Likely] [64]
122. What Fits (Into Few Tokens) Doesn't Overfit (arXiv 2026) argues that strategies which compress into a short description overfit a reused validation set less. Your team has tuned against the same offline benchmark for two years. How would you check whether it is overfit? [Likely] [65]
123. ReSkill (arXiv 2026) treats creating reusable skills as part of the agentic RL loop rather than a separate step. What could go wrong when skills and the policy evolve together, and how would you evaluate it? [Likely] [66]
124. Adaptively looped diffusion language models (ALoDLM, arXiv 2610.04198) give hard tokens more refinement passes than easy ones. Would you consider a diffusion LM for a latency-bound assistant, and what would you measure first? [Likely] [67]
125. WebIE (ACL 2023) targets faithful information extraction from web text. How would you extract product attributes from noisy seller descriptions without inventing facts that are not in the text? [Likely] [68]
126. CaML (WWW 2023) maps retail products to emission-factor categories with zero-shot text similarity. How would you evaluate a zero-shot product-to-taxonomy matcher when you have only a few hundred human labels? [Likely] [69]
127. Earlier LLM distillation work, "Learning with less: knowledge distillation from LLMs via unlabeled data": when is distilling an LLM into a small classifier the right call for a production team? [Likely] [32]

## M. L5-bar probes that recur across all teams

128. Your model's online A/B result fell short of its offline gain. How do you find out why? [Reported] (RS depth round, aggregator) [19]
    - Separate the causes: offline metric mismatch, train-serve skew, position or presentation effects, novelty, and an underpowered test.
129. Of everything you proposed for our problem, what is the riskiest assumption, and how would you de-risk it in two weeks? [Likely]
130. Your design improves our team metric but hurts a sister team's metric (for example, conversion up and returns up). How do you decide? [Likely]
131. Which parts would you build, and which would you reuse from existing Amazon services or other science teams? [Likely]
132. How would you hand this system to engineering for long-term ownership, with monitoring and retraining triggers? [Likely]
133. Tell me about a time you influenced a team outside your own with your science. How would that look here? [Likely] (L5 scope: influence on peers and sister teams; see chapter 01)

## Deepening pass (October 2026): newly reported questions

Method: five web searches (English and Chinese phrasing; Blind, LeetCode Discuss and 1point3acres filters). Every useful hit was an anonymous Glassdoor review, read only as a search-result extract and not opened. The Rufus/reviews item (74) was retagged [Reported] in place from [70]. No Prime Video, Pricing, SCOT or Ads-specific reports turned up, so those sections stay [Likely].

134. (Science Application round, AGI Data Services, Bellevue, April 2025.) Devise three metrics for picking high-quality data out of crowdsourced submissions. [Reported] [71]
    - Which metric needs gold labels, and which can be computed without them?
    - How would you detect annotators who are fast but careless, or who collude?
    - How would you prove that filtering on your metrics improves the downstream model rather than only the metric?
135. (System design round, Hyderabad, April 2026.) Outline an end-to-end pipeline that translates multimodal inputs (text, speech and images) into another language. [Reported] [72]
    - Would you use a cascade (ASR/OCR, then MT, then TTS) or an end-to-end multimodal model, and why?
    - How would you evaluate translation quality per modality when references are scarce?
    - Where do errors compound in the cascade, and how would you localize them?
136. (Same Hyderabad loop, a dedicated round.) Go into depth on your own projects. [Reported for the format] [72]
    - Which decision in that project would you reverse with what you know now?

### Added [Likely] questions closing gaps for the L5 team round

137. Our team's data comes with noisy, weak labels (implicit feedback, LLM labels, vendor annotations). How would you estimate label noise and decide whether to invest in cleaner labels or a noise-robust loss? [Likely]
138. Walk me through the 6-pager or design doc you would write in your first 90 days on this team. What is the customer problem, the baseline, and the decision you need from leadership? [Likely]
139. Our production model was trained two years ago and has drifted. How would you decide between retraining, fine-tuning, or replacing it with an LLM-based approach? [Likely]
    - What evidence would you collect before proposing a rewrite?
140. You have one marketplace with rich data and a new marketplace with almost none. How would you launch the team's model in the new marketplace? [Likely]
141. How would you size the business impact of your proposed model before building it (expected lift, traffic, cost to serve)? [Likely]
    - What would make you kill the project after the first A/B test?
142. An LLM-based component in our pipeline costs more per request than the margin it creates. How would you cut cost (distillation, caching, routing, batching) while protecting quality? [Likely]
143. How would you set up an offline evaluation set for our team's problem that stays trustworthy as the product changes (versioning, leakage checks, refresh cadence)? [Likely]
144. A senior engineer on the team disagrees with your modeling choice and wants a simpler heuristic. How would you settle it with data? [Likely] (Have Backbone; Disagree and Commit)
145. Our model affects sellers or advertisers as well as customers. How would you check for unfair or unintended impact on a subgroup (small sellers, new brands, a locale)? [Likely]
146. Which of our team's metrics do you think is the wrong north star, and what would you replace it with? [Likely]

---

## References

1. Applied Scientist Interview Prep, amazon.jobs (Amazon, official), accessed 2026. https://amazon.jobs/content/en/how-we-hire/applied-scientist-interview-prep
2. Amazon Applied Scientist II Interview Experience & Questions, Glassdoor, 2025-2026. https://www.glassdoor.com/Interview/Amazon-Applied-Scientist-II-Interview-Questions-EI_IE6036.0,6_KO7,27.htm
3. Amazon Applied Scientist Interview Experience & Questions, Glassdoor, 2025-2026. https://www.glassdoor.com/Interview/Amazon-Applied-Scientist-Interview-Questions-EI_IE6036.0,6_KO7,24.htm
4. Amazon Web Services Applied Scientist Interview Experience & Questions, Glassdoor. https://www.glassdoor.com/Interview/Amazon-Web-Services-Applied-Scientist-Interview-Questions-EI_IE7470741.0,19_KO20,37.htm
5. "Applied scientist interview at Amazon Ads", Blind. https://www.teamblind.com/post/applied-scientist-interview-at-amazon-ads-zoerec5t
6. "Amazon (AGI team) Applied Scientist II interview", Blind. https://www.teamblind.com/post/Amazon-AGI-team-Applied-Scientist-II-interview-2eqBXGtT
7. "Amazon AGI Applied Scientist Interview Experience and Insights", 1point3acres. https://www.1point3acres.com/interview/thread/1130091
8. Chaitanya Mundle, "My interview experience for the Applied Scientist II role at Amazon" (Advertising Trust), Medium, circa 2024. https://medium.com/@mundle1997.cm/my-interview-experience-for-the-applied-scientist-ii-role-at-amazon-738aba21f611
9. "Amazon Applied Scientist Interview Guide: Alexa International Tech", Medium (Interview Preparation), Apr 2025. https://medium.com/interview-preparation/amazon-applied-scientist-interview-guide-alexa-international-tech-022d3a056feb
10. "Science application: Business problems for Amazon Applied Scientist (Alexa International Tech)", Medium, 2025. https://medium.com/@pragyashukla2580/science-application-business-problems-for-amazon-applied-scientist-alexa-international-tech-9353a52ca556
11. Ajinkya Pahinkar, "My Amazon Applied Science Interview Journey" (Last Mile), Medium, Jan 2024. https://medium.com/@ajinkya.pahinkar98/my-amazon-applied-science-interview-journey-f3eb1ed748a6
12. Amazon interview review RVW96068026 (Applied Scientist II, LLM team, Santa Clara), Glassdoor, 2025. https://www.glassdoor.com.hk/Interview/Amazon-Interview-E6036-RVW96068026.htm
13. "Amazon Applied Scientist | L6 Luxembourg | Passed, freeze | 2022", LeetCode Discuss, 2022. https://leetcode.com/discuss/interview-experience/4421594/
14. Akanksha Paul, "Interview Experience for Applied Scientist at Amazon", Medium, circa 2020. https://akanksha-paul.medium.com/interview-experience-for-applied-scientist-at-amazon-5ff5003375dd
15. Amazon Supply Chain Interview Experience & Questions (newsvendor problem), Glassdoor. https://www.glassdoor.com/Interview/Amazon-Supply-Chain-Interview-Questions-EI_IE6036.0,6_KO7,19.htm
16. Amazon Applied Scientist Interview Questions & Guide 2026, Dataford (prep vendor), 2026. https://dataford.io/interview-guides/amazon/applied-scientist
17. Amazon Applied Scientist Interview, IGotAnOffer (prep vendor), 2025-2026. https://igotanoffer.com/en/advice/amazon-applied-scientist-interview
18. The 2026 Amazon Applied Scientist Interview Guide, Interview Kickstart (prep vendor), 2026. https://interviewkickstart.com/interview-questions/companies/amazon/applied-scientist
19. "Amazon Research Scientist Interview Process", kevin-2023-code/Tech-Interview-Questions, GitHub (aggregator adapted from trueinterview.io), 2026; fetched and read for this chapter. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/amazon/interview-process-research-scientist.md
20. "Amazon Machine Learning Engineer Interview Process", kevin-2023-code/Tech-Interview-Questions, GitHub (aggregator), 2026; fetched and read for this chapter. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/amazon/interview-process-machine-learning-engineer.md
21. "AI Companies Interview Questions" (Amazon AGI section: loop, reported problems, 2026 org notes, active Nova products), ombharatiya/FAANG-Coding-Interview-Questions, GitHub (aggregator; underlying reports not individually named), 2026; fetched and read for this chapter. https://github.com/ombharatiya/FAANG-Coding-Interview-Questions/blob/main/AI-Companies-Interview-Questions.md
22. Applied Scientist, Rufus Experiences Science (Job ID 2855826), amazon.jobs. https://amazon.jobs/en/jobs/2855826/applied-scientist-rufus-experiences-science
23. Applied Scientist, Alexa International Tech (Job ID 10490052), amazon.jobs. https://www.amazon.jobs/en/jobs/10490052/applied-scientist-alexa-international-tech
24. Applied Scientist II, SCOT Optimal Sourcing Systems, Buying Science Team (Job ID 10468056), amazon.jobs. https://www.amazon.jobs/en/jobs/10468056/applied-scientist-ii-scot-optimal-sourcing-systems-buying-science-team
25. Applied Scientist II, Amazon Payment Products (L5) (Job ID 3168758), amazon.jobs. https://www.amazon.jobs/en/jobs/3168758/applied-scientist-ii-amazon-payment-products-l5
26. 2027 Applied Science Internship: Recommender Systems / Information Retrieval (Job ID 10564598), amazon.jobs. https://www.amazon.jobs/en/jobs/10564598/2027-applied-science-internship-recommender-systems-information-retrieval-machine-learning-united-states-phd-student-science-recruiting
27. "Amazon Rufus AI assistant personalized shopping features", About Amazon (LLMs including Amazon Nova on Bedrock). https://aboutamazon.com/news/retail/amazon-rufus-ai-assistant-personalized-shopping-features
28. "What do I need for cold weather golf?" (Amazon Rufus), IEEE Spectrum. https://spectrum.ieee.org/amazon-rufus
29. "Intent based relevance estimation from click logs", Amazon Science. https://www.amazon.science/publications/intent-based-relevance-estimation-from-click-logs
30. "Predicting what Amazon's customers will need tomorrow, next week, and beyond", Amazon Science. https://www.amazon.science/working-at-amazon/predicting-what-amazons-customers-will-need-tomorrow-next-week-and-beyond
31. "reMARS revisited: Amazon's supply chain optimization", Amazon Science. https://www.amazon.science/latest-news/remars-revisited-amazons-supply-chain-optimization
32. "Learning with less: Knowledge distillation from large language models via unlabeled data", Amazon Science. https://www.amazon.science/publications/learning-with-less-knowledge-distillation-from-large-language-models-via-unlabeled-data
33. amazon-science/chronos-forecasting (Chronos, TMLR 2024; Chronos-Bolt, Nov 2024; Chronos-2, arXiv 2510.15821, Oct 2025), GitHub; fetched and read. https://github.com/amazon-science/chronos-forecasting
34. amazon-science/esci-data, "Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product Search" (Reddy et al., arXiv 2206.06588, 2022), GitHub; fetched and read. https://github.com/amazon-science/esci-data
35. amazon-science/RAGChecker, "RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation" (NeurIPS 2024 Datasets and Benchmarks), GitHub; fetched and read. https://github.com/amazon-science/RAGChecker
36. amazon-science/RefChecker, "RefChecker: Reference-based Fine-grained Hallucination Checker and Benchmark for LLMs" (arXiv 2405.14486, 2024), GitHub; fetched and read. https://github.com/amazon-science/RefChecker
37. amazon-science/auction-gym, "Off-Policy Learning-to-Bid with AuctionGym" (KDD 2023 Applied Data Science; AdKDD 2022 Best Paper), GitHub; fetched and read. https://github.com/amazon-science/auction-gym
38. amazon-science/SynthAVE, "SynthAVE: Scalable Synthetic Labeling for E-Commerce with LLM-Arena Validation" (EMNLP 2026 Industry Track), GitHub; fetched and read. https://github.com/amazon-science/SynthAVE
39. amazon-science/muss, "MUSS: Multilevel Subset Selection for Relevance and Diversity" (UAI 2026), GitHub; fetched and read. https://github.com/amazon-science/muss
40. amazon-science/alexa-teacher-models, "AlexaTM 20B: Few-Shot Learning Using a Large-Scale Multilingual Seq2seq Model" (2022), GitHub; fetched and read. https://github.com/amazon-science/alexa-teacher-models
41. amazon-science/mxfp4-llm, "Training LLMs with MXFP4" (AISTATS 2025), GitHub; fetched and read. https://github.com/amazon-science/mxfp4-llm
42. amazon-science/SpeechworthyLLMs, "Speechworthy Instruction-tuned Language Models" (EMNLP 2024), GitHub; fetched and read. https://github.com/amazon-science/SpeechworthyLLMs
43. amazon-science/fraud-dataset-benchmark, "Fraud Dataset Benchmark and Applications" (arXiv 2208.14417), GitHub; fetched and read. https://github.com/amazon-science/fraud-dataset-benchmark
44. amazon-science/music-off-policy-evaluation-benchmark (Amazon Music home-page logged rankings for OPE; paper under review, 2026), GitHub; fetched and read. https://github.com/amazon-science/music-off-policy-evaluation-benchmark
45. amazon-science/DeepMTL2R, "DeepMTL2R: A Library for Deep Multi-task Learning to Rank" (arXiv 2602.14519, 2026), GitHub; fetched and read. https://github.com/amazon-science/DeepMTL2R
46. amazon-science/SWE-PolyBench, "SWE-PolyBench: A multi-language benchmark for repository level evaluation of coding agents" (arXiv 2504.08703, 2025), GitHub; fetched and read. https://github.com/amazon-science/SWE-PolyBench
47. amazon-science/XplainRecommendation, "Pairwise Ranking Outperforms Single-Action RL for Offline Explanation Selection: A Practical Lesson" (RecSys 2026), GitHub; fetched and read. https://github.com/amazon-science/XplainRecommendation
48. amazon-science/causal-validation ("Validate your causal models!"), GitHub; fetched and read. https://github.com/amazon-science/causal-validation
49. amazon-science organization repository listing (pages 1-2, sorted by stars and by recent activity, Oct 2026; includes mm-cot, auto-cot, tabsyn, AnoLLM, Personalized-chat-interaction-autocomplete, Self-Evolving-Agents-Ratchet, dualkv-flash-attn-for-rl, ammo, hallucination-benchmark-trivialplus), GitHub; fetched and read. https://github.com/orgs/amazon-science/repositories
50. "Amazon Applied Scientist (L5) Virtual Onsite", Blind, circa 2025. https://www.teamblind.com/post/amazon-applied-scientist-l5-virtual-onsite-navwesml
51. "Amazon Onsite Interview Experience for Fulltime Machine Learning Role" (enterprise RAG design), 1point3acres. https://www.1point3acres.com/interview/thread/1157086
52. "Amazon Applied Scientist Interview Debrief: Five-Round Loop + LP", OA VO Service (vendor-published candidate debrief; promotional). https://oavoservice.com/en/articles/amazon-applied-scientist-vo-loop-lp-ml-coding
53. "Amazon applied scientist interview loops", Blind. https://www.teamblind.com/post/amazon-applied-scientist-interview-loops-lucdk0uq
54. "Amazon post-interview team match experience", Blind. https://www.teamblind.com/post/amazon-post-interview-team-match-experience-cqdnnskg
55. "Science Breadth Interview @Amazon" (thread and comments), Blind. https://www.teamblind.com/post/science-breadth-interview-amazon-hfwqznrb
56. "What are some Amazon Research Scientist interview questions?", Quora. https://www.quora.com/What-are-some-Amazon-Research-Scientist-interview-questions
57. "Interview Experience Amazon Applied Scientist" (recommender design round), Blind. https://www.teamblind.com/post/Interview-Experience-Amazon-Applied-Scientist-hWgaGsmO
58. "Amazon applied scientist interview-onsite", Blind, circa 2024. https://www.teamblind.com/post/amazon-applied-scientist-interview-onsite-t01hpnc4
59. "Amazon Interview Process", kevin-2023-code/Tech-Interview-Questions, GitHub (aggregator), 2026; fetched and read. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/amazon/interview-process.md
60. amazon-science/MigrationBench, "MigrationBench: Repository-Level Code Migration Benchmark from Java 8" (KDD 2026; arXiv 2505.09569), GitHub; fetched and read. https://github.com/amazon-science/MigrationBench
61. amazon-science/JAWS-Bench, "Breaking the Code: Security Assessment of AI Code Agents Through Systematic Jailbreaking Attacks" (arXiv 2510.01359), GitHub; fetched and read. https://github.com/amazon-science/JAWS-Bench
62. amazon-science/evolveQA, "When Facts Change: Probing LLMs on Evolving Knowledge with evolveQA" (EMNLP 2026; arXiv 2510.19172), GitHub; fetched and read. https://github.com/amazon-science/evolveQA
63. amazon-science/hallucination-benchmark-trivialplus, "Rethinking Evaluation for LLM Hallucination Detection: A Desiderata, A New RAG-based Benchmark, New Insights" (TRIVIA+, ACL 2026), GitHub; fetched and read. https://github.com/amazon-science/hallucination-benchmark-trivialplus
64. amazon-science/NLPActiveTesting, "Select, Label, Evaluate: Active Testing in NLP" (EMNLP 2026; arXiv 2603.21840), GitHub; fetched and read. https://github.com/amazon-science/NLPActiveTesting
65. amazon-science/whatfits, "What Fits (Into Few Tokens) Doesn't Overfit" (arXiv 2606.11045, 2026), GitHub; fetched and read. https://github.com/amazon-science/whatfits
66. amazon-science/reskill, "ReSkill: Reconciling Skill Creation with Policy Optimization in Agentic RL" (arXiv 2606.01619, 2026), GitHub; fetched and read. https://github.com/amazon-science/reskill
67. amazon-science/ALoDLM, "Adaptively Looped Diffusion Language Models" (arXiv 2610.04198, 2026), GitHub; fetched and read. https://github.com/amazon-science/ALoDLM
68. amazon-science/webie, "WebIE: Faithful and Robust Information Extraction on the Web" (ACL 2023), GitHub; fetched and read. https://github.com/amazon-science/webie
69. amazon-science/carbon-assessment-with-ml, "CaML: Carbon Footprinting of Household Products with Zero-Shot Semantic Text Similarity" (WWW 2023 Web4Good), GitHub; fetched and read. https://github.com/amazon-science/carbon-assessment-with-ml
70. Amazon interview review RVW99133644 (Applied Scientist, Luxembourg; Science Application round: summarize product reviews into key concepts), Glassdoor, circa 2023-2024; search-result extract only. https://www.glassdoor.sg/Interview/Amazon-Interview-E6036-RVW99133644.htm
71. Amazon interview review RVW103552813 (Applied Scientist, AGI Data Services, Bellevue, April 2025; Science Application round: three metrics for selecting high-quality crowdsourced data), Glassdoor; search-result extract only. https://www.glassdoor.de/Vorstellungsgespräch/Amazon-Vorstellungsgespräch-E6036-RVW103552813.htm
72. Amazon interview review RVW96903377 (Applied Scientist, Hyderabad, April 2026; system design: end-to-end pipeline translating multimodal inputs into another language; project deep-dive round), Glassdoor; search-result extract only. https://www.glassdoor.com.ar/Entrevista/Amazon-Entrevista-E6036-RVW96903377.htm
