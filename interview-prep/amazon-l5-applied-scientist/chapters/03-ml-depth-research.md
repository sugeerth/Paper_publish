# 03. ML Depth and Science Application: Research and Project Deep Dives, and Applying Science to Amazon Problems

## Format notes

- **Official frame.** Amazon's Applied Scientist interview-prep page describes one or two 60-minute technical phone screens with a senior leader, then a loop of four 55-minute interviews with members of the science community. Each interview covers technical competencies and Leadership Principles (LPs), and the behavioral portion focuses heavily on your past work [1].
- **What "Science Depth" means in practice.** Candidates consistently describe the depth round as a deep dive into one or two projects or papers from the resume, usually led by a Senior Applied Scientist.
  - An AS II (L5) Advertising Trust candidate described "Science Depth" as a resume deep dive plus GenAI fundamentals (Transformers) plus the business impact of past projects [3].
  - Other AS2 reports describe a "deep dive into the specifics of my resume and past projects" [22], "modeling details in my project" [11], and the questions "why went with a particular choice of architecture?" [12].
  - IGotAnOffer summarizes the round as walking through a research project, with follow-ups on technical decisions, the modeling trade-offs considered, and outcomes [2].
  - A 2026 aggregate of about 20 candidate reports (2026 Q1 to Q3) describes the depth round as a full hour on your most complex project or your publications. The interviewer reads the resume in advance and chooses the topics: problem framing, data cleaning, baselines, class imbalance, why that backbone, metrics, deployment, and what you would do with more compute. Interns had each publication taken in turn [41].
  - Some loops replace or supplement this round with a 30-40 minute research talk or job talk followed by Q&A [10][16][26]; senior loops more often include presentations [39]. Blind commenters say the recruiter or coordinator should send a brief naming the topics and LPs for each round, so ask for it [19]. The same 2026 aggregate warns that the recruiter's preview of the rounds is often inaccurate [41].
- **Phone screens.** 2026 reports describe screens of 30-60 minutes that combine a two-project resume dive (the data, the training problems and why the design was chosen), a theory block and a coding problem [41]. Machine Learning Engineer loops report a separate "project deep dive" round of 10-15 minutes per project followed by layered questions [42].
- **What "Science Application" means.** This is the second half of the chapter, and it is reported in two flavors:
  - (a) An open-ended Amazon business or product problem. You propose data sources, labeling, modeling, metrics and risks [2][5][6]. Examples include modeling view, click and purchase probabilities [7]; categorizing 10k products into 100 categories [9]; training retrieval from click logs [9]; trustworthy ad recommendation [3]; and Alexa ASR for accented speech [6]. 2026 prompts include agentic AI proposals, a multi-turn search agent, a chatbot architecture and an evaluation framework for brand names in generated images [41].
  - (b) Discussion "based on past project experience," per a Blind comment [18].
  - An L5 candidate (Nov 2024) listed "ML application" as a distinct onsite round alongside depth, breadth, DSA and Bar Raiser [4]. An Alexa guide lists "science application" as one of five loop rounds [5].
- **Interviewer type.** Usually Senior or Principal Applied Scientists, and sometimes the hiring manager (Applied Science Manager), who may run a "tech talk" style mix of design and behavioral questions [3][8].
  - LPs are probed in every round. Older reports say about 5-15 minutes of each interview goes to one or two LP questions [5][27]. 2026 reports describe much more: often about half the hour, so the technical part may not start until minute 25 or later [41][43].
  - The depth round is often paired with Dive Deep, Deliver Results or Earn Trust [41]; Ownership and Customer Obsession also fit it naturally [Likely].
- **L5-specific bar.** At L5 (Applied Scientist II), the depth interviewer is checking several things:
  - whether you were "the architect or just a passenger" [23][25], with your decisions separated from your collaborators' and your advisor's [41];
  - whether you can defend every choice (architecture, loss, metric, data split) against alternatives [23][24][41];
  - whether you understand leakage and "how models cheat" [23][24];
  - whether you can connect the science to measurable customer or business impact [3][41][42]. Level is judged here: one candidate passed every round on content but was told the projects lacked the impact expected for the target level. The 2026 aggregate's advice is to lead with the project that has the largest measured business effect [41][42].
  - A Blind commenter notes that a fresh PhD is typically placed at L5 [21]. That means the bar includes publication-level rigor plus evidence that you can ship [Likely].
  - Interviewers often want "plausible and scientific sounding ideas on the why," even when they do not know the answer themselves [20].
- **Recent trend (2025-2026).** Depth rounds increasingly drift from your project into the fundamentals behind it. Reported topics include:
  - transformer internals, KV caching, long-context attention (Longformer) and bi-encoder vs. cross-encoder retrieval [12];
  - BERT vs. GPT and DPO vs. RLHF [13];
  - beam search decoding [17];
  - knowledge distillation and learning with limited labels [8];
  - how Segment Anything works [12];
  - for generative-AI and post-training teams, LoRA, RLHF, GRPO, KV caching and distributed training, while ads and seller teams lean on statistics, experimentation and ranking [41].
  - AGI-team loops are reported to lean heavily on modern LLM training and to weight ML depth above behavioral questions [28].
- **Method note (2026-10-14 review).** The original pass came from about 38 web searches, using search-result extracts rather than full page reads.
  - In this review, the egress proxy refused every cited host except raw GitHub content (HTTP 403 on Glassdoor, Blind, LeetCode, Medium, Quora, amazon.jobs, amazon.science, IGotAnOffer, Interview Kickstart, Dataford, OA VO Service and others), and the web-search budget was already used up. Those references could not be re-checked. "Unreachable" does not mean dead, so they were kept.
  - Five GitHub-hosted sources were fetched and read in full: [28], [41], [42], [43] and [44]. [28] supports the claims it is cited for. [41]-[44] are new and back the added 2026 questions.
  - Retagging: [Reported] items that cite only a prep-vendor guide ([24], [27]) or a non-science Amazon loop ([38]) were retagged [Likely], in line with the policy below. Two generic "answer a question about our product" items and one near-duplicate were removed.
  - Several sources are prep-vendor guides (Interview Kickstart, Dataford, OA VO Service). When a question appears only in a vendor guide, it is tagged [Likely] with "see [n]". [41]-[44] are TrueInterview pages mirrored on GitHub. They aggregate candidate reports and name the round, but not individual candidates.

**Tag legend:** [Reported] means a candidate or source reports it for an Amazon Applied Scientist or closely related Amazon science loop (cited). Items reported at Amazon with no role stated are marked "(role not stated)", and items from Research Scientist or MLE loops are marked as such. [Likely] means a well-grounded inference from the role, level, team or Amazon products.

---

## A. Opening the deep dive: framing, scope and ownership

1. Walk me through the project or paper on your resume that you are most proud of, end to end. [Reported] [2][3][22][41]
   - Describe your recent work and projects. Which would you choose for a 45-minute deep dive, and why that one? [Reported] [2]
   - What was the problem statement, and who was the customer? What was the business motivation? [Reported] [41][42]
   - Which parts did you personally design, implement and evaluate, and which were done by others? [Reported] [41][42]
   - If I asked your collaborator or manager, how would they describe your contribution?
2. Pick two projects from your resume. For each one, what was the data, what training problems did you hit, and why was the design chosen? [Reported] (phone screen, 2026) [41]
3. Describe one of your proudest projects. What's the most innovative thing you did in it? [Reported] [2]
   - What made it novel compared with the prior art or the existing baseline?
   - Was the novelty necessary, or would a simpler baseline have got most of the gain?
4. Tell me about a project where you were the single-threaded owner of the science. [Likely]
   - What decisions did you make without being told to?
   - What would have happened to the project without you?
5. What was the state of the art or the production baseline when you started, and how far did you move it? [Likely] (see [26])
   - How did you choose the baselines, and were they tuned as carefully as your method? [Reported] [41]
6. What was the hardest technical obstacle in this project, and how did you get past it? [Likely]
   - What did you try that failed?
   - How long did you spend on dead ends before pivoting? (Bias for Action and Dive Deep)

## B. Problem formulation and data

7. How did you turn the business or research question into an ML problem (classification, ranking, regression, generation, RL)? [Likely] (see [41])
   - Could it have been framed differently, and why did you reject the alternative framings?
8. How large was the training data, and where did it come from? How did you clean it? [Reported] [41] (see also the Glassdoor depth-round reports summarized at [12])
   - How did you sample it, and is the sample representative of production traffic?
   - How did you down-sample or construct the dataset? [Reported] [7]
9. How did you obtain labels, and how noisy were they? [Likely] (see [2])
   - Did you measure inter-annotator agreement?
   - How did label noise affect your model, and did you correct for it?
10. How would you build this with very little labeled data? [Reported] [8]
    - Would you use weak supervision, semi-supervised learning, active learning, or LLM-generated labels?
    - How would you validate synthetic or LLM-generated labels before training on them?
11. How did you handle class imbalance in your project? [Reported] [27][41][42]
    - Why did you choose re-weighting, re-sampling or threshold tuning over the alternatives?
    - How did the imbalance change your choice of evaluation metric? [Reported] [27][41]
12. Which features mattered most, and how do you know? [Likely] (see [24])
    - Would you trust impurity-based or permutation importance here, and why? [Reported] (phone-screen theory block) [41]
    - How did you guard against features that are not available at inference time?
13. How did you handle missing data, outliers and distribution differences between data sources? [Likely]
14. Would you standardize your inputs? How did you treat sparse numeric features, and what did you do about correlated features? [Reported] (role not stated) [44]

## C. Modeling choices and alternatives

15. Why did you choose this particular architecture or backbone? [Reported] [12][41][42]
    - Why X over Y? What concrete evidence made you choose? [Likely] (see [23])
    - What would a simpler model (logistic regression or GBDT) have achieved?
16. What alternatives did you consider, and why did you rule them out? [Reported] [41] (see also the Glassdoor depth-round reports summarized at [12])
    - Did you run the alternatives, or reason about them?
17. Why did you use one gradient-boosting library or framework over another? [Likely] (see [23])
18. If you used a pretrained model, why that backbone, and how did you adapt it (full fine-tuning, adapters, LoRA, prompting)? [Likely] (see [28][41])
    - How did you choose the size of the model given latency and cost limits?
19. Derive or write out the core equations of your method on the whiteboard. [Likely] (see [25][41])
    - Which assumptions does the derivation rely on, and which of them does your data violate?
20. What are the main hyperparameters in your model, and how sensitive are the results to each? [Likely]
    - How did you tune them, and on which split?
21. If you used knowledge distillation, what exactly was distilled and why? [Reported] [8]
    - What were the teacher-student gap, the temperature and the loss mix?
    - Why distill instead of quantize or prune?

## D. Loss, optimization and training dynamics

22. Why did you choose that specific loss function? [Likely] (see [23])
    - How does it align with the metric you actually care about?
    - What happens if you swap it for a ranking loss, focal loss or a calibrated probabilistic loss?
23. Tell me about loss functions and optimizers you used, and why. [Reported] (see the Glassdoor depth-round reports summarized at [12])
    - Why Adam or AdamW over SGD with momentum? [Reported] (phone-screen theory block) [41]
    - What learning-rate schedule did you use, and why?
24. Did you see overfitting or training instability, and how did you diagnose and fix it? [Likely]
    - Training accuracy is high but test accuracy is low: what do you do? [Reported] [2]
    - Your model is underperforming: diagnose and fix it. [Reported] (listed as a depth-round question) [41][42]
25. How was training distributed and how long did it take? What was the compute budget? [Likely]
    - Explain the parallelism strategies and collective operations your training used. [Reported] (role not stated) [44]
    - What would you do with more compute or resources? [Reported] [41][42]
    - What would you change if you had 10x less compute?

## E. Evaluation and scientific rigor

26. How did you evaluate the model, and why did you pick those metrics? [Reported] [41][42] (see also the Glassdoor depth-round reports at [12] and [25])
    - Choose between precision, recall, F1 and ROC-AUC for this problem, and justify your choice. [Reported] [27][41]
    - How does the offline metric relate to the online business metric? [Reported] [41][42]
    - Can you write code that computes AUC from scores and labels? [Reported] [14]
27. How did you confirm the model wasn't just memorizing training data? [Likely] (see [23])
28. How did you handle data leakage in that specific pipeline? [Likely] (see [23][42])
    - How did you split train, validation and test (random, temporal, by user, by product)?
    - Give an example of leakage you found or almost missed.
29. Are your improvements statistically significant? [Likely]
    - How many seeds or runs did you use, and what is the variance?
    - What confidence interval or test did you use?
30. What ablations did you run, and which component contributed most? [Likely] (see [26])
31. Where does your model fail? Show me the error analysis. [Likely]
    - Which slices (new users, long-tail products, low-resource locales) are worst?
    - What would you do next to fix the worst slice?
32. Was the model A/B tested online? How did you design the test? [Reported] [41][42]
    - What was the treatment, the primary metric and the guardrails, and how long did the test run?
    - Your online A/B result fell short of the offline gain. How do you find out why? [Reported] [41][42]

## F. Deployment, scale and post-launch

33. How was the model deployed, and what were the latency, throughput and cost constraints? [Likely] (see [24][41])
    - What did you trade away to meet latency?
34. How did you monitor the model after launch? [Likely] (see [23])
    - How would you detect data drift vs. concept drift? [Reported] [21]
    - What triggers retraining, and how do you roll back?
35. What happened when the data distribution shifted (seasonality, Prime Day, new marketplace)? [Likely]
36. If you had to scale this 100x (more traffic, more locales, more products), what breaks first? [Likely]

## G. Impact, business value and reflection

37. What was the business impact of this project, in numbers? [Reported] [3][41][42]
    - How did you attribute the impact to your model rather than to other launches?
    - Who used the output, and how did it change their decisions?
38. What would you do differently if you started this project again today? [Likely] (see [25][42])
    - Which limitations did you not mention in the paper or launch doc? [Likely] (see [41])
    - What was a real mistake in this project, how was it caught, and what changed afterward? [Likely] (see [42])
39. Tell me about a time your results disagreed with a stakeholder's intuition. How did you resolve it? [Likely] (see [1][27])
40. How did you decide this project was worth doing over other options? [Likely]
41. Tell me about a time you simplified a scientific solution to ship faster. What did you give up? [Likely] (Invent and Simplify)

## H. Research talk, job talk or paper round (when the loop includes one)

42. Present your PhD or research work in about 30 minutes, then take ML-focused Q&A. [Reported] [10]
    - For each publication, how did you choose the baselines, and are they the strongest available? [Reported] [41]
    - For each publication, what was your specific contribution vs. your co-authors'? [Reported] [41]
43. Present a project of your choice (a "tech talk"). [Reported] [16][12][28]
    - How would this research transfer to an Amazon product?
44. Why did you make this specific choice in your presentation? (Reported as an interviewer picking one detail and asking "why" repeatedly.) [Reported] [20]
    - What would you expect to happen if this assumption were removed?
    - Can you propose a plausible explanation for this unexpected result?
45. Paper-read round: discuss a paper's method, assumptions and weaknesses. [Reported] (listed among depth-round questions for FAR and "premium" loops; the exact prompt was not captured) [41]
46. What is the next paper you would write on this line of work? [Likely]
47. Which reviewer criticism of your paper was most valid, and how did you address it? [Likely]

## I. Depth in your specialty: NLP and LLMs (common for 2025-2026 L5 loops)

48. Explain the transformer architecture in detail. [Reported] [15]
    - Explain the attention mechanism at "root level," including scaled dot-product, multi-head attention and positional encoding. [Reported] [13][15]
    - Walk me through the mathematical formulation of self-attention. [Likely] (see [23])
49. How do BERT and GPT differ (objectives, architecture, use cases)? [Reported] [13]
    - How is BERT trained? [Reported] [9]
    - What are the differences between BERT and RoBERTa? [Reported] (see the Glassdoor depth-round reports summarized at [12])
50. How does decoder-only inference work, and what is the KV cache? [Reported] [12]
    - Why does standard self-attention get expensive on long inputs? [Reported] [12]
    - How does Longformer, or sparse or local attention, handle long context? [Reported] [12]
51. Compare bi-encoders and cross-encoders for retrieval and re-ranking. [Reported] [12]
    - When would you distill a cross-encoder into a bi-encoder? [Likely] (see [34])
52. How does beam search decoding work for LLMs? [Reported] [17]
    - What are the trade-offs vs. sampling (temperature, top-k, top-p)?
53. What is the difference between DPO and RLHF? [Reported] [13]
    - Walk through an SFT training pipeline and its loss. [Likely] (see [27])
    - What does DPO's beta control? [Likely]
54. If you did post-training: why does GRPO remove the critic, what does that cost, and how would you design the reward? [Reported] (phone screen, generative-AI team) [41]
    - Which instabilities did you hit in GRPO-style training, and how did you fix them? [Likely] (see [41])
55. Explain the main parameter-efficient fine-tuning techniques, which parameters each trains, and when each beats full fine-tuning. [Reported] (online assessment, resume-based) [41][42]
    - How do you choose the LoRA rank and which layers to adapt? [Likely] (see [25][28])
56. How does an inference engine such as vLLM raise serving throughput for large language models? [Reported] (online assessment, MLE loop) [42]
57. How do you evaluate an LLM-based system (hallucination, faithfulness, helpfulness)? [Likely] (see [28][32])
    - When would you trust an LLM-as-judge, and how would you calibrate it?
58. Explain an AI-agent project you built, and how it was adopted. [Reported] (MLE project deep dive) [42]
59. Why do LSTMs beat vanilla RNNs, and why did transformers replace both? [Reported] (see the Glassdoor reports summarized at [12])

## J. Depth in your specialty: search, ranking and recommendations

60. You have a large collection of user click data for many search queries and product documents. Is click data alone sufficient? How would you train a retrieval or IR model with the data at hand? [Reported] [9]
    - How do you correct for position bias in clicks? [Reported] [42] (see also [29][40])
    - What other biases (query intent, presentation, trust) contaminate click labels? [Likely] (see [29])
61. Walk through how you would design, train, validate and deploy a ranking model for product search. [Reported] [40][42]
    - How would you estimate true relevance from clickstream logs with position bias? [Reported] [40]
62. Walk me through a search-ranking project you led: the pain point, why this model, the offline and online metrics, and how you diagnosed a disappointing A/B result. [Reported] [41][42]
63. Why did you choose pointwise, pairwise or listwise learning-to-rank in your project? [Likely]
    - How do NDCG and MRR relate to online engagement?
64. How do you handle cold start for new users or products in your recommender? [Likely]
65. How does matrix factorization for recommendation work, and when would you use it over a two-tower model? [Likely] (see [25])
66. Predict the click probability for music recommendations. [Reported] (role not stated; phone screen, Sep 2026) [44]

## K. Depth in your specialty: vision and multimodal

67. Describe how SAM (Segment Anything Model) works. [Reported] [12]
    - How would you adapt it to a product-image task with few labels?
68. You have 10k product listings across 100 categories, each with a title, description and photo. How would you categorize them? [Reported] [9]
    - How do you fuse text and image signals?
    - How do you handle a long tail of categories with very few examples?
    - How do you evaluate when the category taxonomy itself is noisy?
69. For a detection system you built, explain the backbone choice, how you handled class imbalance, and what more resources would change. [Reported] [42]
70. Compare approaches to building joint image-text embeddings (CLIP-style contrastive pretraining and others). Where would they help Amazon? [Reported] (online assessment) [41]

## L. Depth in your specialty: forecasting, optimization and causal inference

71. Explain your forecasting model, and how you produce and evaluate uncertainty (quantiles, calibration). [Likely] (see [30][31])
    - Why probabilistic forecasts rather than point forecasts for inventory decisions?
72. What is the newsvendor problem, and what is its solution? [Likely] (see [38]; reported in Amazon supply-chain interviews, not a science loop)
    - How does a quantile forecast connect to the newsvendor critical ratio?
73. How would you model an optimization problem from your work: objective, decision variables and constraints? [Likely] (see [27])
    - When would you use a heuristic instead of an exact solver?
74. Build a package-allocation model for couriers. [Reported] (role not stated) [44]
75. Prove that a new allocation method outperforms the manual baseline. [Reported] (role not stated) [44]
76. Design an end-to-end regression model for energy demand. [Reported] (role not stated) [44]
77. Design a causal study to measure the impact of reminders on customer behavior. [Reported] (role not stated) [44]
78. How would you estimate the causal effect of a price or promotion change without a clean experiment? [Likely] (see [31])
    - When would you use double ML, instrumental variables, or difference-in-differences?
    - How do you validate a causal estimate you can never observe directly?
79. In your project, how did you separate correlation from causation in the impact claim? [Likely]

---

## M. Science application: reported Amazon problems

80. A user moves from viewing to clicking to purchasing a product. Model the probability of each action given a user and a product. [Reported] [7]
    - Would you build three independent models, or model view, click and purchase jointly? [Reported] [7]
    - How would you construct the dataset, choose features, and down-sample (for example with reservoir sampling)? [Reported] [7]
    - How would you handle the selection bias that purchases are only observed after clicks?
81. Design a system that recommends trustworthy ads to Amazon users while maintaining brand safety and compliance. [Reported] [3]
    - Walk through the full data pipeline, from collection to deployment. [Reported] [3]
    - How do you trade off ad revenue against trust and safety errors?
82. How would you improve Alexa speech recognition for non-native speakers and users with strong regional accents? [Reported] [6]
    - Where would you get diverse accent training data, and how would you handle phonetic variation? [Reported] [6]
    - How would you measure improvement per accent group without hurting the majority?
83. You are given a high-level ML problem. Propose a solution, including data sources, annotation strategy, modeling techniques, evaluation metrics, and trade-offs and risks. [Reported] [2]
84. Propose agentic AI solutions for a hypothetical project, and defend their viability. [Reported] (2026) [41]
85. Design a multi-turn search agent, and explain how it handles long conversation context. [Reported] (2026) [41]
86. Architect an AI chatbot: intent detection, dialogue management, generation and a knowledge base. [Reported] [41][42]
    - How do fallback behavior, evaluation and continuous learning work? [Reported] [41][42]
87. Design an evaluation framework that checks whether a text-to-image system spells a requested brand name correctly and places it naturally. [Reported] [41][42]
    - How would you combine OCR accuracy, embedding-based semantic checks and a human review protocol? [Reported] [41]
88. Design an LLM quality validation system. [Reported] [41][44]
89. Design a distributed training-data pipeline (reported for a FAR loop). [Reported] [42][44]
90. How would you evaluate adding video ads? [Reported] (role not stated, Nov 2025) [44]
91. How would you measure product-market fit for Alexa Shopping? [Reported] (role not stated) [44]
92. Design an end-to-end spam detection system. [Reported] (role not stated) [44]
93. Discuss LLM decoding and training choices in the context of an Amazon use case (LLM team). [Reported] [17][28]
94. Design and evaluate an ML pipeline to improve personalized product recommendations on the Amazon homepage. [Likely] (see [25])
    - What is the north-star metric, and what are the guardrails?
    - How would you trade off customer engagement against sales? [Likely] (see [25])

## N. Science application: likely Amazon problems by org (L5, 2025-2026)

95. Rufus (shopping assistant): how would you build retrieval-augmented answers to "What do I need for cold weather golf?" over the catalog and reviews? [Likely] (see [32][33])
    - How do you ground answers in product facts and avoid hallucinated specs?
    - How do you evaluate it offline before an A/B test?
96. How would you summarize thousands of customer reviews into a short "customers say" blurb? [Reported] (Science Application round, framed as "summarize product reviews with key concepts"; Luxembourg, Nov 2023) [47]
    - How do you measure faithfulness and coverage of minority opinions?
97. Fraud and abuse: how would you detect fake reviews or abusive sellers with heavily imbalanced and adversarial labels? [Likely]
    - How do you handle adversaries adapting to your model?
98. Supply chain (SCOT): how would you forecast demand for millions of products, including new ones with no history? [Likely] (see [30][31][37])
    - How would forecast errors propagate into buying and placement decisions?
99. Pricing: how would you design an experiment to measure the effect of a pricing algorithm when prices interact across products? [Likely] (see [31])
    - What about interference between treated and control products?
100. Last mile and geospatial: how would you predict delivery time or improve address geocoding accuracy? [Likely] (see [35])
     - What loss would you use if late deliveries cost more than early ones?
101. Ads: how would you predict click-through rate for sponsored products, and how does calibration interact with the auction? [Likely] (see [3][41])
     - Revenue is flat, CPC is up and conversions are down after a pacing change. How would you diagnose it?
102. Product categorization and catalog quality: how would you detect and fix wrong attributes (size, color, brand) on listings at catalog scale? [Likely]
103. Alexa or devices: how would you make an on-device model 10x smaller with minimal quality loss? [Likely] (see [34])
104. AWS AI services: how would you evaluate whether a new foundation-model fine-tuning method is worth exposing to customers? [Likely]
105. How would you build a query-understanding model (intent, product type, attributes) for a new marketplace and language with little data? [Likely] (see [36])

## O. Applying your own research to Amazon

106. How would you apply your research or project to a problem at Amazon? [Reported] [18]
     - Which team would benefit most, and what metric would move?
     - What would have to change in your method to work at Amazon's scale and latency?
107. If you joined this team, what would you work on in your first six months, and how would you measure success? [Likely]
108. Take your method and tell me the first experiment you would run with our data. [Likely]
     - What result would make you abandon the approach?
109. How would you explain your project's value to a non-scientist product manager in two minutes? [Likely] (Earn Trust)
110. Write the one-paragraph "PR/FAQ" headline for a product built on your research. [Likely]

## P. LP probes woven into the depth round

111. Tell me about a time you went deep into the data and found something that changed the direction of the project. [Likely] (Dive Deep; see [1][5])
112. Describe a decision you made with incomplete information, and how you later validated it. [Reported] [41] (Bias for Action)
113. Describe a calculated risk you took: what you knew at the time, and how it turned out. [Reported] [42]
114. Tell me about a time a client or decision maker proposed a plan that was far too aggressive. What did you do? [Reported] [41][42]
115. Tell me about a time a model you shipped failed or underperformed in production. What did you do? [Likely] (Ownership)
116. Tell me about a time you disagreed with a senior scientist's technical approach, for example about an evaluation metric, and then committed. [Likely] (see [41][42]; Have Backbone; Disagree and Commit)
117. Tell me about a time you raised the scientific bar on your team (reviews, standards, reproducibility). [Likely] (Insist on the Highest Standards)
118. Tell me about a time you learned a new area quickly to solve a problem. [Likely] (Learn and Be Curious)
119. Tell me about the most ambiguous problem you worked on, and how you scoped it. [Likely]
120. Tell me about a time you traded scientific purity for customer impact, or refused to. [Likely] (Customer Obsession)
121. How do you use AI tools in your work? Tell me about a time an AI tool gave you a wrong answer, and how you caught it. [Reported] (Amazon behavioral, 2025-2026) [43]

## Deepening pass (October 2026): newly reported questions

Five web searches (Blind, Glassdoor, LeetCode Discuss and Chinese-language "面经" phrasing) mostly returned Glassdoor reviews already summarized above. Only snippet text was available. Format confirmations: a Blind L5 virtual onsite listed five rounds (science breadth, science depth, "Science Application: propose a model and implementation to a business problem", coding, and a hiring-manager fit round) [45]. An L5 Blind post says depth is "usually" built on one of your publications or resume projects, and that the motivation behind each choice matters most [46]. A Luxembourg loop paired one tech talk with coding, depth, breadth and application rounds, each with 10-15 minutes of LPs [47]. Item 96 above was retagged [Reported] on the strength of [47]. A 2025 Vancouver snippet described a depth round that unexpectedly focused on statistics, and a Hyderabad AS II snippet described a design prompt about multimodal translation. Neither snippet could be tied to a specific URL, so the related items below are tagged [Likely].

122. In the context of your own project, explain the bias-variance trade-off. Where did your model sit, and how did you know? [Reported] (Applied Scientist II depth round, Seattle) [13]
     - Which change you made reduced variance, and which reduced bias?
123. What is the difference between bagging and boosting? Which suits your project's data, and why? [Reported] (Applied Scientist II depth round, Seattle) [13]
     - Why does boosting tend to overfit noisy labels more than bagging does?
124. For one of your publications, what was the problem, which choices did you make, and why does the solution make sense? [Reported] (L5 depth round) [46]
     - What motivated each design choice over the obvious alternative?
125. What research directions would you pursue to improve a CLIP-style model? [Reported] (Applied Scientist review, Italy) [48]
     - Which of CLIP's weaknesses (compositionality, fine-grained attributes, text in images) matters most for Amazon product data?

### Gap-closing [Likely] questions for L5

126. Statistics-heavy depth: how did you quantify uncertainty in your project's main result, and what assumptions does that test make? [Likely] (a 2025 snippet reports a statistics-focused depth round; no URL could be tied to it)
     - When would you use MLE vs. MAP estimation for a parameter in your model?
     - How would you correct for multiple comparisons across the many ablations or metrics you report?
127. Design an end-to-end pipeline that translates multimodal content (for example, a product listing's text and images) into another language. [Likely] (a Hyderabad AS II snippet reports a similar prompt; no URL could be tied to it)
     - How would you evaluate translation quality when the image and the text must stay consistent?
128. Were your model's predicted probabilities calibrated? How did you check, and did a downstream decision depend on it? [Likely]
129. Could another scientist reproduce your headline result from your code and data? What would they most likely get wrong? [Likely] (Insist on the Highest Standards)
130. How did performance scale with more data or a larger model? Did you plot learning curves, and what did they tell you about where to invest next? [Likely]
131. How did you make sure your offline evaluation set matched what the model would see in production? [Likely]
132. Did you check performance across sensitive or under-served groups (locales, device types, new sellers)? What did you find? [Likely] (responsible AI)
133. Would a prompted large language model have solved your problem? How would you compare it with your model on quality, latency and cost? [Likely]
134. Pick a recent paper outside your area that you think matters for Amazon. What is its key idea, and what is its weakest assumption? [Likely]
135. Suppose your project had to run in a new marketplace with a different language and different customer behavior. Which parts of your method transfer, and which would you have to rebuild? [Likely]

---

## References

1. Applied Scientist Interview Prep, amazon.jobs (Amazon official), accessed 2026. https://amazon.jobs/content/en/how-we-hire/applied-scientist-interview-prep
2. Amazon Applied Scientist Interview (process, questions, prep), IGotAnOffer, 2025. https://igotanoffer.com/en/advice/amazon-applied-scientist-interview
3. My interview experience for the Applied Scientist II role at Amazon (Advertising Trust), Medium, about 2024. https://medium.com/@mundle1997.cm/my-interview-experience-for-the-applied-scientist-ii-role-at-amazon-738aba21f611
4. Applied Scientist at Amazon: Interview Insights (L5), Medium, Nov 2024. https://medium.com/@rahulnkumar/applied-scientist-at-amazon-interview-insights-25265e1405f5
5. Amazon Applied Scientist Interview Guide — Alexa International Tech, Medium, Apr 2025. https://medium.com/interview-preparation/amazon-applied-scientist-interview-guide-alexa-international-tech-022d3a056feb
6. Science application — Business problems for Amazon Applied Scientist (Alexa International Tech), Medium, 2025. https://medium.com/@pragyashukla2580/science-application-business-problems-for-amazon-applied-scientist-alexa-international-tech-9353a52ca556
7. Interview Experience for Applied Scientist at Amazon (Akanksha Paul), Medium, about 2020. https://akanksha-paul.medium.com/interview-experience-for-applied-scientist-at-amazon-5ff5003375dd
8. Amazon Interview Experience for Applied Scientist 2, Enginebogie, about 2025. https://enginebogie.com/interview/experience/amazon-applied-scientist-2/1004
9. Amazon Applied Scientist | L6 Luxembourg | Passed, freeze | 2022, LeetCode Discuss, 2022. https://leetcode.com/discuss/interview-experience/4421594/
10. Amazon | Applied/Research Scientist | Cambridge UK | May 2020 [Reject], LeetCode Discuss, 2020. https://leetcode.com/discuss/interview-experience/658771/amazon-appliedresearch-scientist-cambridge-uk-may-2020-reject
11. Amazon Applied Scientist II Interview Experience & Questions, Glassdoor, 2025-2026. https://www.glassdoor.com/Interview/Amazon-Applied-Scientist-II-Interview-Questions-EI_IE6036.0,6_KO7,27.htm
12. Amazon Applied Scientist Interview Experience & Questions, Glassdoor (aggregated reviews, including April 2026 reports), 2026. https://www.glassdoor.com/Interview/Amazon-Applied-Scientist-Interview-Questions-EI_IE6036.0,6_KO7,24.htm
13. Amazon interview review RVW96903377 (Applied Scientist II, Seattle), Glassdoor, 2026. https://www.glassdoor.fr/Entretien/Amazon-Entretien-E6036-RVW96903377.htm
14. Amazon interview review RVW95575263 (Applied Scientist II), Glassdoor, 2026. https://www.glassdoor.fr/Entretien/Amazon-Entretien-E6036-RVW95575263.htm
15. Amazon interview review RVW102987402 (Applied Scientist), Glassdoor, 2026. https://www.glassdoor.co.uk/Interview/Amazon-Interview-E6036-RVW102987402.htm
16. Amazon interview review RVW104768247 (Senior Applied Scientist, Boston), Glassdoor, 2025. https://www.glassdoor.com/Interview/Amazon-Interview-E6036-RVW104768247.htm
17. Amazon interview review RVW96068026 (Applied Scientist II, LLM team, Santa Clara), Glassdoor, 2025. https://www.glassdoor.com.hk/Interview/Amazon-Interview-E6036-RVW96068026.htm
18. Science Breadth Interview @Amazon (thread and comments), Blind. https://www.teamblind.com/post/science-breadth-interview-amazon-hfwqznrb
19. What can I expect from the science breadth round at Amazon, Blind. https://www.teamblind.com/post/what-can-i-expect-from-the-science-breadth-round-at-amazon-cbtdq86j
20. Amazon applied scientist interview-onsite, Blind. https://www.teamblind.com/post/amazon-applied-scientist-interview-onsite-t01hpnc4
21. Amazon Applied Scientist II interview, Blind (2022-2025 thread cluster). https://www.teamblind.com/post/Amazon-Applied-Scientist-II-interview-WqazQMSC
22. Applied scientist interview Amazon, Blind. https://www.teamblind.com/post/applied-scientist-interview-amazon-4udsobcp
23. The 2026 Amazon Senior Applied Scientist Interview Guide, Interview Kickstart (prep vendor), 2026. https://interviewkickstart.com/interview-questions/companies/amazon/senior-applied-scientist
24. The 2026 Amazon Applied Scientist Interview Guide, Interview Kickstart (prep vendor), 2026. https://interviewkickstart.com/interview-questions/companies/amazon/applied-scientist
25. Amazon Applied Scientist Interview Questions & Guide 2026, Dataford (prep vendor), 2026. https://dataford.io/interview-guides/amazon/applied-scientist
26. Amazon Research Scientist Interview Questions & Guide 2026, Dataford (prep vendor), 2026. https://dataford.io/interview-guides/amazon/research-scientist
27. Amazon Applied Scientist Interview Debrief: Five-Round Loop + LP, OA VO Service (prep vendor). https://oavoservice.com/en/articles/amazon-applied-scientist-vo-loop-lp-ml-coding
28. AI-Engineer-Interview-Questions: amazon.md, GitHub (ombharatiya), last reviewed Oct 2026; fetched and read 2026-10-14 (loop table with "Science application / tech talk" and ML depth rounds; AGI loops heavy on modern LLM training and evaluation). https://github.com/ombharatiya/AI-Engineer-Interview-Questions/blob/main/14-company-interview-questions/amazon.md
29. Intent based relevance estimation from click logs, Amazon Science. https://www.amazon.science/publications/intent-based-relevance-estimation-from-click-logs
30. Predicting what Amazon's customers will need tomorrow, next week, and beyond, Amazon Science. https://www.amazon.science/working-at-amazon/predicting-what-amazons-customers-will-need-tomorrow-next-week-and-beyond
31. Applied Scientist II, SCOT-Optimal Sourcing Systems, Buying Science Team (Job ID 10468056), amazon.jobs. https://www.amazon.jobs/en/jobs/10468056/applied-scientist-ii-scot-optimal-sourcing-systems-buying-science-team
32. Applied Scientist, Rufus Experiences Science (Job ID 2855826), amazon.jobs. https://amazon.jobs/en/jobs/2855826/applied-scientist-rufus-experiences-science
33. "What do I need for cold weather golf?" (Amazon Rufus), IEEE Spectrum. https://spectrum.ieee.org/amazon-rufus
34. Learning with less: Knowledge distillation from large language models via unlabeled data, Amazon Science. https://www.amazon.science/publications/learning-with-less-knowledge-distillation-from-large-language-models-via-unlabeled-data
35. My Amazon Applied Science Interview Journey (Last Mile Geospatial), Medium, Jan 2024. https://medium.com/@ajinkya.pahinkar98/my-amazon-applied-science-interview-journey-f3eb1ed748a6
36. Applied Scientist (L4), AEE Science and AI (Job ID 2823771), amazon.jobs. https://amazon.jobs/en/jobs/2823771/applied-scientist-l4-aee-science-and-ai
37. reMARS revisited: Amazon's supply chain optimization, Amazon Science. https://www.amazon.science/latest-news/remars-revisited-amazons-supply-chain-optimization
38. Amazon Supply Chain Interview Experience & Questions, Glassdoor (non-science roles). https://www.glassdoor.com/Interview/Amazon-Supply-Chain-Interview-Questions-EI_IE6036.0,6_KO7,19.htm
39. Amazon Senior Applied Scientist Interview Questions, Glassdoor. https://www.glassdoor.com/Interview/Amazon-Senior-Applied-Scientist-Interview-Questions-EI_IE6036.0,6_KO7,31.htm
40. What are some Amazon Research Scientist interview questions?, Quora. https://www.quora.com/What-are-some-Amazon-Research-Scientist-interview-questions
41. "Amazon Research Scientist Interview Process" (covers Research and Applied Scientist loops; based on about 20 candidate reports, mainly 2026 Q1 to Q3), TrueInterview, mirrored in kevin-2023-code/Tech-Interview-Questions, GitHub, 2026; fetched and read 2026-10-14. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/amazon/interview-process-research-scientist.md
42. "Amazon Machine Learning Engineer Interview Process" (includes the project deep-dive and ML system design rounds), TrueInterview, mirrored in kevin-2023-code/Tech-Interview-Questions, GitHub, 2026; fetched and read 2026-10-14. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/amazon/interview-process-machine-learning-engineer.md
43. "Amazon Interview Process" (all tracks; GenAI-usage behavioral question; time split between LPs and technical work), TrueInterview, mirrored in kevin-2023-code/Tech-Interview-Questions, GitHub, 2026; fetched and read 2026-10-14. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/amazon/interview-process.md
44. "Amazon interview process, OA & interview questions" (320 candidate-reported questions with format, round and sighting date; role not given per question), TrueInterview, mirrored in kevin-2023-code/Tech-Interview-Questions, GitHub, 2026; fetched and read 2026-10-14. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/amazon.md
45. amazon as2 interview loop, Blind (five-round AS virtual onsite: science breadth, science depth, science application, coding, hiring-manager fit). https://www.teamblind.com/post/amazon-as2-interview-loop-ry5amdu4
46. amazon applied scientist l5 virtual onsite, Blind (depth round built around one publication or resume project). https://www.teamblind.com/post/amazon-applied-scientist-l5-virtual-onsite-navwesml
47. Amazon interview review RVW99133644 (Applied Scientist, Luxembourg, Nov 2023; Science Application round: summarize product reviews with key concepts), Glassdoor. https://www.glassdoor.sg/Interview/Amazon-Interview-E6036-RVW99133644.htm
48. Amazon interview review RVW101056902 (Applied Scientist; research directions for improving CLIP), Glassdoor Italy, 2026. https://www.glassdoor.it/Colloquio/Amazon-Colloquio-E6036-RVW101056902.htm
