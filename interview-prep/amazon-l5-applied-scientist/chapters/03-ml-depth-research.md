# 03. ML Depth and Science Application: Research and Project Deep Dives, and Applying Science to Amazon Problems

## Format notes

- **Official frame.** Amazon's Applied Scientist interview-prep page describes one or two 60-minute technical phone screens with a senior leader, then a loop of four 55-minute interviews with members of the science community. Each interview covers technical competencies and Leadership Principles (LPs), and the behavioral portion focuses heavily on your past work [1].
- **What "Science Depth" means in practice.** Candidates consistently describe the depth round as a deep dive into one or two projects or papers from the resume, usually led by a Senior Applied Scientist.
  - An AS II (L5) Advertising Trust candidate described "Science Depth" as a resume deep dive plus GenAI fundamentals (Transformers) plus the business impact of past projects [3].
  - Other AS2 reports describe a "deep dive into the specifics of my resume and past projects" [22], "modeling details in my project" [11], and the questions "why went with a particular choice of architecture?" [12].
  - IGotAnOffer summarizes the round as walking through a research project, with follow-ups on technical decisions, the modeling trade-offs considered, and outcomes [2].
  - Some loops replace or supplement this round with a 30-40 minute research talk or job talk followed by Q&A [10][16][26]; senior loops more often include presentations [39]. Blind commenters say the recruiter or coordinator should send a brief naming the topics and LPs for each round, so ask for it [19].
- **What "Science Application" means.** This is the second half of the chapter, and it is reported in two flavors:
  - (a) An open-ended Amazon business or product problem. You propose data sources, labeling, modeling, metrics and risks [2][5][6]. Examples include modeling view, click and purchase probabilities [7]; categorizing 10k products into 100 categories [9]; training retrieval from click logs [9]; trustworthy ad recommendation [3]; and Alexa ASR for accented speech [6].
  - (b) Discussion "based on past project experience," per a Blind comment [18].
  - An L5 candidate (Nov 2024) listed "ML application" as a distinct onsite round alongside depth, breadth, DSA and Bar Raiser [4]. An Alexa guide lists "science application" as one of five loop rounds [5].
- **Interviewer type.** Usually Senior or Principal Applied Scientists, and sometimes the hiring manager (Applied Science Manager), who may run a "tech talk" style mix of design and behavioral questions [3][8].
  - LPs are probed in every round. Reports say about 5-15 minutes of each interview goes to one or two LP questions [5][27].
  - Dive Deep, Ownership and Customer Obsession map most naturally onto the depth round [Likely].
- **L5-specific bar (inferred from sources).** At L5 (Applied Scientist II), the depth interviewer is checking several things:
  - whether you were "the architect or just a passenger" [23][25];
  - whether you can defend every choice (architecture, loss, metric, data split) against alternatives [23][24];
  - whether you understand leakage and "how models cheat" [23][24];
  - whether you can connect the science to measurable customer or business impact [3].
  - A Blind commenter notes that a fresh PhD is typically placed at L5 [21]. That means the bar includes publication-level rigor plus evidence that you can ship [Likely].
  - Interviewers often want "plausible and scientific sounding ideas on the why," even when they do not know the answer themselves [20].
- **Recent trend (2025-2026).** Depth rounds increasingly drift from your project into the fundamentals behind it. Reported topics include:
  - transformer internals, KV caching, long-context attention (Longformer) and bi-encoder vs. cross-encoder retrieval [12];
  - BERT vs. GPT and DPO vs. RLHF [13];
  - beam search decoding [17];
  - knowledge distillation and learning with limited labels [8];
  - how Segment Anything works [12].
  - AGI-team loops are reported to lean heavily on modern LLM training [28].
- **Method note.** This chapter comes from about 38 web searches. Direct page fetches (Glassdoor, Blind, LeetCode, Medium, amazon.jobs, and others) were blocked by this environment's egress policy, and the search budget ran out before the planned 45 queries.
  - Every [Reported] tag rests on search-result extracts of the cited page, not a full read.
  - Several sources are prep-vendor guides (Interview Kickstart, Dataford, OA VO Service). When a question appears only in a vendor guide, it is tagged [Likely] with "see [n]".

**Tag legend:** [Reported] means a candidate or source reports it for an Amazon Applied Scientist or closely related Amazon science loop (cited). [Likely] means a well-grounded inference from the role, level, team or Amazon products.

---

## A. Opening the deep dive: framing, scope and ownership

1. Walk me through the project or paper on your resume that you are most proud of, end to end. [Reported] [2][3][22]
   - What was the problem statement, and who was the customer? [Reported] (see the Glassdoor depth-round reports summarized at [12])
   - Which parts did you personally design, implement and evaluate, and which were done by others? [Likely] (see [23][25])
   - If I asked your collaborator or manager, how would they describe your contribution?
2. Describe your recent work and projects. [Reported] [2]
   - Which of these would you choose for a 45-minute deep dive, and why that one?
3. Describe one of your proudest projects. What's the most innovative thing you did in it? [Reported] [2]
   - What made it novel compared with the prior art or the existing baseline?
   - Was the novelty necessary, or would a simpler baseline have got most of the gain?
4. Tell me about a project where you were the single-threaded owner of the science. [Likely]
   - What decisions did you make without being told to?
   - What would have happened to the project without you?
5. What was the state of the art or the production baseline when you started, and how far did you move it? [Likely] (see [26])
   - How did you pick baselines, and were they tuned as carefully as your method? [Likely] (see [26])
6. What was the hardest technical obstacle in this project, and how did you get past it? [Likely]
   - What did you try that failed?
   - How long did you spend on dead ends before pivoting? (Bias for Action and Dive Deep)

## B. Problem formulation and data

7. How did you turn the business or research question into an ML problem (classification, ranking, regression, generation, RL)? [Likely]
   - Could it have been framed differently, and why did you reject the alternative framings?
8. How large was the training data, and where did it come from? [Reported] (see the Glassdoor depth-round reports summarized at [12])
   - How did you sample it, and is the sample representative of production traffic?
   - How did you down-sample or construct the dataset? [Reported] [7]
9. How did you obtain labels, and how noisy were they? [Likely] (see [2])
   - Did you measure inter-annotator agreement?
   - How did label noise affect your model, and did you correct for it?
10. How would you build this with very little labeled data? [Reported] [8]
    - Would you use weak supervision, semi-supervised learning, active learning, or LLM-generated labels?
    - How would you validate synthetic or LLM-generated labels before training on them?
11. How did you handle class imbalance in your project? [Reported] [27]
    - Why did you choose re-weighting, re-sampling or threshold tuning over the alternatives?
    - How did the imbalance change your choice of evaluation metric? [Reported] [27]
12. Which features mattered most, and how do you know? [Likely] (see [24])
    - How did you guard against features that are not available at inference time?
13. How did you handle missing data, outliers and distribution differences between data sources? [Likely]

## C. Modeling choices and alternatives

14. Why did you choose this particular architecture? [Reported] [12]
    - Why X over Y? What concrete evidence made you choose? [Likely] (see [23])
    - What would a simpler model (logistic regression or GBDT) have achieved?
15. What alternatives did you consider, and why did you rule them out? [Reported] (see the Glassdoor depth-round reports summarized at [12])
    - Did you run the alternatives, or reason about them?
16. Why did you use one gradient-boosting library or framework over another? [Likely] (see [23])
17. If you used a pretrained model, why that backbone, and how did you adapt it (full fine-tuning, adapters, LoRA, prompting)? [Likely] (see [28])
    - How did you choose the size of the model given latency and cost limits?
18. Derive or write out the core equations of your method on the whiteboard. [Likely] (see [25])
    - Which assumptions does the derivation rely on, and which of them does your data violate?
19. What are the main hyperparameters in your model, and how sensitive are the results to each? [Likely]
    - How did you tune them, and on which split?
20. If you used knowledge distillation, what exactly was distilled and why? [Reported] [8]
    - What were the teacher-student gap, the temperature and the loss mix?
    - Why distill instead of quantize or prune?

## D. Loss, optimization and training dynamics

21. Why did you choose that specific loss function? [Likely] (see [23])
    - How does it align with the metric you actually care about?
    - What happens if you swap it for a ranking loss, focal loss or a calibrated probabilistic loss?
22. Tell me about loss functions and optimizers you used, and why. [Reported] (see the Glassdoor depth-round reports summarized at [12])
    - Why Adam or AdamW over SGD with momentum?
    - What learning-rate schedule did you use, and why?
23. Did you see overfitting or training instability, and how did you diagnose and fix it? [Likely]
    - Training accuracy is high but test accuracy is low: what do you do? [Reported] [2]
24. How was training distributed and how long did it take? What was the compute budget? [Likely]
    - What would you change if you had 10x less compute?

## E. Evaluation and scientific rigor

25. How did you evaluate the model, and why did you pick those metrics? [Reported] (see the Glassdoor depth-round reports summarized at [12]; see also [25])
    - Choose between precision, recall, F1 and ROC-AUC for this problem, and justify your choice. [Reported] [27]
    - How does the offline metric relate to the online business metric?
    - Can you write code that computes AUC from scores and labels? [Reported] [14]
26. How did you confirm the model wasn't just memorizing training data? [Likely] (see [23])
27. How did you handle data leakage in that specific pipeline? [Likely] (see [23])
    - How did you split train, validation and test (random, temporal, by user, by product)?
    - Give an example of leakage you found or almost missed.
28. Are your improvements statistically significant? [Likely]
    - How many seeds or runs did you use, and what is the variance?
    - What confidence interval or test did you use?
29. What ablations did you run, and which component contributed most? [Likely] (see [26])
30. Where does your model fail? Show me the error analysis. [Likely]
    - Which slices (new users, long-tail products, low-resource locales) are worst?
    - What would you do next to fix the worst slice?
31. Was the model A/B tested online? [Likely] (see [36])
    - What was the treatment, the primary metric and the guardrails, and how long did the test run?
    - What would you do if offline gains did not translate online?

## F. Deployment, scale and post-launch

32. How was the model deployed, and what were the latency, throughput and cost constraints? [Likely] (see [24])
    - What did you trade away to meet latency?
33. How did you monitor the model after launch? [Likely] (see [23])
    - How would you detect data drift vs. concept drift? [Reported] [21]
    - What triggers retraining, and how do you roll back?
34. What happened when the data distribution shifted (seasonality, Prime Day, new marketplace)? [Likely]
35. If you had to scale this 100x (more traffic, more locales, more products), what breaks first? [Likely]

## G. Impact, business value and reflection

36. What was the business impact of this project, in numbers? [Reported] [3]
    - How did you attribute the impact to your model rather than to other launches?
    - Who used the output, and how did it change their decisions?
37. What would you do differently if you started this project again today? [Likely] (see [25])
    - Which limitations did you not mention in the paper or launch doc?
38. Tell me about a time your results disagreed with a stakeholder's intuition. How did you resolve it? [Likely] (see [1][27])
39. How did you decide this project was worth doing over other options? [Likely]
40. Tell me about a time you simplified a scientific solution to ship faster. What did you give up? [Likely] (Invent and Simplify)

## H. Research talk or job talk Q&A (when the loop includes a presentation)

41. Present your PhD or research work in about 30 minutes, then take ML-focused Q&A. [Reported] [10]
    - Why these baselines, and are they the strongest available? [Likely] (see [26])
    - What is your individual contribution vs. your co-authors'? [Likely] (see [26])
42. Present a project of your choice (a "tech talk"). [Reported] [16][12]
    - How would this research transfer to an Amazon product?
43. The interviewer picks a detail from your presentation and asks "why" repeatedly. [Reported] [20]
    - What would you expect to happen if this assumption were removed?
    - Can you propose a plausible explanation for this unexpected result?
44. What is the next paper you would write on this line of work? [Likely]
45. Which reviewer criticism of your paper was most valid, and how did you address it? [Likely]

## I. Depth in your specialty: NLP and LLMs (common for 2025-2026 L5 loops)

46. Explain the transformer architecture in detail. [Reported] [15]
    - Explain the attention mechanism at "root level," including scaled dot-product, multi-head attention and positional encoding. [Reported] [13][15]
    - Walk me through the mathematical formulation of self-attention. [Likely] (see [23])
47. How do BERT and GPT differ (objectives, architecture, use cases)? [Reported] [13]
    - How is BERT trained? [Reported] [9]
    - What are the differences between BERT and RoBERTa? [Reported] (see the Glassdoor depth-round reports summarized at [12])
48. How does decoder-only inference work, and what is the KV cache? [Reported] [12]
    - Why does standard self-attention get expensive on long inputs? [Reported] [12]
    - How does Longformer, or sparse or local attention, handle long context? [Reported] [12]
49. Compare bi-encoders and cross-encoders for retrieval and re-ranking. [Reported] [12]
    - When would you distill a cross-encoder into a bi-encoder? [Likely] (see [34])
50. How does beam search decoding work for LLMs? [Reported] [17]
    - What are the trade-offs vs. sampling (temperature, top-k, top-p)?
51. What is the difference between DPO and RLHF? [Reported] [13]
    - Walk through an SFT training pipeline and its loss. [Reported] [27]
    - What does DPO's beta control? [Likely]
52. How would you fine-tune an LLM for a domain task with limited compute (LoRA or PEFT)? [Likely] (see [25][28])
    - How do you choose the LoRA rank and which layers to adapt?
53. How do you evaluate an LLM-based system (hallucination, faithfulness, helpfulness)? [Likely] (see [32])
    - When would you trust an LLM-as-judge, and how would you calibrate it?
54. Why do LSTMs beat vanilla RNNs, and why did transformers replace both? [Reported] (see the Glassdoor reports summarized at [12])

## J. Depth in your specialty: search, ranking and recommendations

55. You have a large collection of user click data for many search queries and product documents. Is click data alone sufficient? How would you train a retrieval or IR model with the data at hand? [Reported] [9]
    - How do you correct for position bias in clicks? [Likely] (see [29][40])
    - What other biases (query intent, presentation, trust) contaminate click labels? [Likely] (see [29])
56. Walk through how you would design, train, validate and deploy a ranking model for product search. [Reported] [40]
    - How would you estimate true relevance from clickstream logs with position bias? [Reported] [40]
57. Why did you choose pointwise, pairwise or listwise learning-to-rank in your project? [Likely]
    - How do NDCG and MRR relate to online engagement?
58. How do you handle cold start for new users or products in your recommender? [Likely]
59. How does matrix factorization for recommendation work, and when would you use it over a two-tower model? [Likely] (see [25])

## K. Depth in your specialty: vision and multimodal

60. Describe how SAM (Segment Anything Model) works. [Reported] [12]
    - How would you adapt it to a product-image task with few labels?
61. You have 10k product listings across 100 categories, each with a title, description and photo. How would you categorize them? [Reported] [9]
    - How do you fuse text and image signals?
    - How do you handle a long tail of categories with very few examples?
    - How do you evaluate when the category taxonomy itself is noisy?
62. How does contrastive image-text pretraining (CLIP-style) work, and where would it help Amazon? [Likely] (see [25])

## L. Depth in your specialty: forecasting, optimization and causal inference

63. Explain your forecasting model, and how you produce and evaluate uncertainty (quantiles, calibration). [Likely] (see [30][31])
    - Why probabilistic forecasts rather than point forecasts for inventory decisions?
64. What is the newsvendor problem, and what is its solution? [Reported] [38]
    - How does a quantile forecast connect to the newsvendor critical ratio?
65. How would you model an optimization problem from your work: objective, decision variables and constraints? [Reported] [27]
    - When would you use a heuristic instead of an exact solver?
66. How would you estimate the causal effect of a price or promotion change without a clean experiment? [Likely] (see [31])
    - When would you use double ML, instrumental variables, or difference-in-differences?
    - How do you validate a causal estimate you can never observe directly?
67. In your project, how did you separate correlation from causation in the impact claim? [Likely]

---

## M. Science application: reported Amazon problems

68. A user moves from viewing to clicking to purchasing a product. Model the probability of each action given a user and a product. [Reported] [7]
    - Would you build three independent models, or model view, click and purchase jointly? [Reported] [7]
    - How would you construct the dataset, choose features, and down-sample (for example with reservoir sampling)? [Reported] [7]
    - How would you handle the selection bias that purchases are only observed after clicks?
69. Design a system that recommends trustworthy ads to Amazon users while maintaining brand safety and compliance. [Reported] [3]
    - Walk through the full data pipeline, from collection to deployment. [Reported] [3]
    - How do you trade off ad revenue against trust and safety errors?
70. How would you improve Alexa speech recognition for non-native speakers and users with strong regional accents? [Reported] [6]
    - Where would you get diverse accent training data, and how would you handle phonetic variation? [Reported] [6]
    - How would you measure improvement per accent group without hurting the majority?
71. You are given a high-level ML problem. Propose a solution, including data sources, annotation strategy, modeling techniques, evaluation metrics, and trade-offs and risks. [Reported] [2]
72. Answer an ML application question about an Amazon product (team-specific). [Reported] (see the Glassdoor reports summarized at [24])
73. Design and evaluate an ML pipeline to improve personalized product recommendations on the Amazon homepage. [Likely] (see [25])
    - What is the north-star metric, and what are the guardrails?
74. How would you optimize Amazon's recommendation system to improve customer engagement and sales? [Likely] (see [25])
75. Answer a search-results application question (search relevance at Amazon). [Reported] (see the Glassdoor reports summarized at [12])
76. The phone screen or depth round on an LLM team covers LLM decoding and training in the context of an Amazon use case. [Reported] [17][28]

## N. Science application: likely Amazon problems by org (L5, 2025-2026)

77. Rufus (shopping assistant): how would you build retrieval-augmented answers to "What do I need for cold weather golf?" over the catalog and reviews? [Likely] (see [32][33])
    - How do you ground answers in product facts and avoid hallucinated specs?
    - How do you evaluate it offline before an A/B test?
78. How would you summarize thousands of customer reviews into a short "customers say" blurb? [Likely]
    - How do you measure faithfulness and coverage of minority opinions?
79. Fraud and abuse: how would you detect fake reviews or abusive sellers with heavily imbalanced and adversarial labels? [Likely]
    - How do you handle adversaries adapting to your model?
80. Supply chain (SCOT): how would you forecast demand for millions of products, including new ones with no history? [Likely] (see [30][31][37])
    - How would forecast errors propagate into buying and placement decisions?
81. Pricing: how would you design an experiment to measure the effect of a pricing algorithm when prices interact across products? [Likely] (see [31])
    - What about interference between treated and control products?
82. Last mile and geospatial: how would you predict delivery time or improve address geocoding accuracy? [Likely] (see [35])
    - What loss would you use if late deliveries cost more than early ones?
83. Ads: how would you predict click-through rate for sponsored products, and how does calibration interact with the auction? [Likely] (see [3])
    - Revenue is flat, CPC is up and conversions are down after a pacing change. How would you diagnose it?
84. Product categorization and catalog quality: how would you detect and fix wrong attributes (size, color, brand) on listings at catalog scale? [Likely]
85. Alexa or devices: how would you make an on-device model 10x smaller with minimal quality loss? [Likely] (see [34])
86. AWS AI services: how would you evaluate whether a new foundation-model fine-tuning method is worth exposing to customers? [Likely]
87. How would you build a query-understanding model (intent, product type, attributes) for a new marketplace and language with little data? [Likely] (see [36])

## O. Applying your own research to Amazon

88. How would you apply your research or project to a problem at Amazon? [Reported] [18]
    - Which team would benefit most, and what metric would move?
    - What would have to change in your method to work at Amazon's scale and latency?
89. If you joined this team, what would you work on in your first six months, and how would you measure success? [Likely]
90. Take your method and tell me the first experiment you would run with our data. [Likely]
    - What result would make you abandon the approach?
91. How would you explain your project's value to a non-scientist product manager in two minutes? [Likely] (Earn Trust)
92. Write the one-paragraph "PR/FAQ" headline for a product built on your research. [Likely]

## P. LP probes woven into the depth round

93. Tell me about a time you went deep into the data and found something that changed the direction of the project. [Likely] (Dive Deep; see [1][5])
94. Tell me about a time you had to make a decision with incomplete data. [Likely] (Bias for Action)
95. Tell me about a time a model you shipped failed or underperformed in production. What did you do? [Likely] (Ownership)
96. Tell me about a time you disagreed with a senior scientist's technical approach. [Likely] (Have Backbone; Disagree and Commit)
97. Tell me about a time you raised the scientific bar on your team (reviews, standards, reproducibility). [Likely] (Insist on the Highest Standards)
98. Tell me about a time you learned a new area quickly to solve a problem. [Likely] (Learn and Be Curious)
99. Tell me about the most ambiguous problem you worked on, and how you scoped it. [Likely]
100. Tell me about a time you traded scientific purity for customer impact, or refused to. [Likely] (Customer Obsession)

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
23. The 2026 Amazon Senior Applied Scientist Interview Guide, Interview Kickstart, 2026. https://interviewkickstart.com/interview-questions/companies/amazon/senior-applied-scientist
24. The 2026 Amazon Applied Scientist Interview Guide, Interview Kickstart, 2026. https://interviewkickstart.com/interview-questions/companies/amazon/applied-scientist
25. Amazon Applied Scientist Interview Questions & Guide 2026, Dataford, 2026. https://dataford.io/interview-guides/amazon/applied-scientist
26. Amazon Research Scientist Interview Questions & Guide 2026, Dataford, 2026. https://dataford.io/interview-guides/amazon/research-scientist
27. Amazon Applied Scientist Interview Debrief: Five-Round Loop + LP, OA VO Service. https://oavoservice.com/en/articles/amazon-applied-scientist-vo-loop-lp-ml-coding
28. AI-Engineer-Interview-Questions: amazon.md, GitHub (ombharatiya). https://github.com/ombharatiya/AI-Engineer-Interview-Questions/blob/main/14-company-interview-questions/amazon.md
29. Intent based relevance estimation from click logs, Amazon Science. https://www.amazon.science/publications/intent-based-relevance-estimation-from-click-logs
30. Predicting what Amazon's customers will need tomorrow, next week, and beyond, Amazon Science. https://www.amazon.science/working-at-amazon/predicting-what-amazons-customers-will-need-tomorrow-next-week-and-beyond
31. Applied Scientist II, SCOT-Optimal Sourcing Systems, Buying Science Team (Job ID 10468056), amazon.jobs. https://www.amazon.jobs/en/jobs/10468056/applied-scientist-ii-scot-optimal-sourcing-systems-buying-science-team
32. Applied Scientist, Rufus Experiences Science (Job ID 2855826), amazon.jobs. https://amazon.jobs/en/jobs/2855826/applied-scientist-rufus-experiences-science
33. "What do I need for cold weather golf?" (Amazon Rufus), IEEE Spectrum. https://spectrum.ieee.org/amazon-rufus
34. Learning with less: Knowledge distillation from large language models via unlabeled data, Amazon Science. https://www.amazon.science/publications/learning-with-less-knowledge-distillation-from-large-language-models-via-unlabeled-data
35. My Amazon Applied Science Interview Journey (Last Mile Geospatial), Medium, Jan 2024. https://medium.com/@ajinkya.pahinkar98/my-amazon-applied-science-interview-journey-f3eb1ed748a6
36. Applied Scientist (L4), AEE Science and AI (Job ID 2823771), amazon.jobs. https://amazon.jobs/en/jobs/2823771/applied-scientist-l4-aee-science-and-ai
37. reMARS revisited: Amazon's supply chain optimization, Amazon Science. https://www.amazon.science/latest-news/remars-revisited-amazons-supply-chain-optimization
38. Amazon Supply Chain Interview Experience & Questions, Glassdoor. https://www.glassdoor.com/Interview/Amazon-Supply-Chain-Interview-Questions-EI_IE6036.0,6_KO7,19.htm
39. Amazon Senior Applied Scientist Interview Questions, Glassdoor. https://www.glassdoor.com/Interview/Amazon-Senior-Applied-Scientist-Interview-Questions-EI_IE6036.0,6_KO7,31.htm
40. What are some Amazon Research Scientist interview questions?, Quora. https://www.quora.com/What-are-some-Amazon-Research-Scientist-interview-questions
