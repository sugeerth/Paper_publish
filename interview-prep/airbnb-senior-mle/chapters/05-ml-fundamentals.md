# 05. ML Fundamentals and Modeling Depth: Ranking, GBDT vs. NN, Embeddings, Calibration, Features, Offline/Online Gap

> **Research note (read first).** Adversarially reviewed 2026-10-14. The shared web-search budget was used up, so no new search queries could run. The proxy could not resolve arxiv.org, dl.acm.org, medium.com, prachub.com, teamblind.com or trueinterview.io. GitHub (pages and raw files) was the only reachable host. **Fetched and read on review:** the TrueInterview-derived Airbnb MLE guide (3 candidate reports, mainly 2026 Q1-Q2) [1]; the dated Airbnb question index [2]; the Airbnb company interview-process page [3]; and about 100 posts in a GitHub mirror of 1point3acres Airbnb interview posts. Only two of those posts contain ML rounds [4][5]. The other posts were checked and contain no ML content.
>
> **How references were checked.** arXiv, ACM and Medium could not be opened directly. Airbnb paper and blog URLs marked "(URL confirmed)" were checked against the curated applied-ML index [30], which lists the same titles and URLs. URLs marked "(not verified)" are cited from prior knowledge or from Chapter 09. Check those before relying on them. The PracHub report [29] could not be opened. Its topics are taken from Chapter 09's summary and are tagged [Likely], not [Reported].
>
> **Tags.** [Reported] means a fetched source names the prompt, the follow-up or the topic for an Airbnb MLE interview. [Likely] means an inference from the Senior MLE role, the reported round structure and Airbnb's published systems. A citation next to a [Likely] tag points to the paper or post behind the topic. It does **not** mean anyone was asked the question.

## Format notes

- **There is no reported standalone "ML theory" round.** Reported MLE loops have a phone screen, one coding round, one or two ML system design rounds (sometimes an extra one), an experience deep dive, core values and a hiring-manager conversation [1][3]. Fundamentals come up as depth probes inside the design rounds and the deep dive. The design rounds are "conversations without code", and "the first minutes of scoping are graded" [1].
- **The design rounds decide the loop.** Rejections in the reports were attributed to design depth, not to coding. A borderline signal led to an extra design round instead of a pass [1]. One ML rejection was blamed on "jumping straight to model ideas"; the bar is "problem and label before model" [3].
- **Expected skeleton:** decision supported → label → data → features → model → offline and online metrics → serving → experiment [1]. Interviewers assess problem framing, label definition and time window, concrete marketplace features (reviews, host responsiveness, booking completion, photos, location encodings), the link from offline to online metrics, and production behaviour (candidate generation, latency, freshness, cold start for new listings, monitoring) [1].
- **Interviewers steer hard.** Reported examples: a lifetime-value round defined the label and then moved from regression to bucketized classification; a relisting round went into geohashing; search-ranking rounds pushed on latency, freshness and experimentation [1]. Treat a redirection as a hint and restate the narrower scope.
- **Reported prompts these probes hang off:** search ranking end to end, ranking without a text query, personalized search ranking, user-embedding semantic search, listing LTV, relisting detection, cover-photo selection, high-quality listings, recommendations, and GenAI support retrieval and ranking [1][2]. 1point3acres loops add customer LTV and host pricing [4], plus listing recommendation and price prediction [5].
- **Senior (G9) bar:** design carries more weight at Senior and above. Scope is scored separately from modeling difficulty, and a deep project with narrow reach "reads weaker" than a simpler one with wide impact [1]. Expect a "why" on every choice: label, GBDT vs. NN, metric, and what should move online. Practise each answer in 2-4 minutes, tied to an Airbnb surface.
- **Role context:** Airbnb MLE postings stress training-serving skew, A/B testing, and feature and model selection. Shared infrastructure includes Chronon, Airbnb's open-source feature platform for consistent online and offline features [1][25].
- **Public Airbnb context interviewers know well:** search moved from GBDT to deep NNs [6][7]; in-session listing embeddings [8]; Experiences ranking [9]; diversity in ranking [10][21]; multi-task journey ranking [11]; a customized pricing regression loss [12]; interleaving [13]; embedding-based retrieval [14].

## Questions

### A. Problem framing, labels and targets

1. You are asked to estimate the lifetime value of a listing (property). Would you frame it as regression or as bucketized classification, and why? [Reported] [1][2]
   - The interviewer pushes you toward bucketized classification. What do you gain and lose? [Reported] [1]
   - How do you choose bucket boundaries, and how do you turn bucket probabilities back into an expected value? Would you use an ordinal loss or a plain softmax?
   - How do you handle the heavy right tail and listings that churn off the platform (censoring)?
2. Before modeling LTV, agree on what "lifetime value" means here. Over what time window is the label defined, and how does that choice trade label delay against usefulness? [Reported] [1]
   - Which business decision does the LTV model support (host acquisition spend, host coaching, ranking), and how does that change the target and the metric? [Reported] [1]
3. Design a guest (customer) lifetime-value model. How do the unit of prediction, the label and the features differ from listing LTV? [Reported] [4]
   - How would you validate an LTV model whose label only matures a year after prediction? [Likely] [18]
4. How would you identify "high-quality listings"? What is your label, given that no ground truth exists? [Reported] [1]
   - Which proxies (reviews, rebook rate, cancellations, CS tickets, host responsiveness) would you combine? How do you avoid circularity with the ranking model's own exposure?
   - How would you use human raters to validate the proxy label, and how many ratings do you need? [Likely] [17]
5. Choose a listing's cover photo to lift bookings. Is this a supervised problem, a ranking problem or a bandit problem? [Reported] [1]
   - What label would you train on, given that the cover photo you showed also caused the clicks?
   - How do you handle cold start for a new listing with no engagement on any photo? [Reported] [1]
   - When would you pick a contextual bandit over a supervised model plus an A/B test? [Likely]
6. Detect relisting (a removed or banned listing reappears, possibly under a new account). What features and model family would you use? [Reported] [1][2]
   - Is this classification, similarity search or entity resolution? Defend the framing.
   - How does geohashing help you generate candidate pairs, and how does geohash precision trade recall against pair volume? [Reported] [1]
   - How would the model's outputs show up in online metrics, given that confirmed relistings arrive late and only for flagged cases? [Reported] [1]
   - How do you choose an operating threshold when false positives hurt legitimate hosts?
7. For search ranking, what is the training label: click, long click, wishlist, booking request, accepted booking or completed stay without cancellation? [Likely] [6][11]
   - How do you weight these in a single objective vs. a multi-task model?
   - How do you deal with label delay (booking days after search)?
8. Rank home search results when the user gives no text query (only location and dates, or nothing). What does "relevance" mean here, and what is the label? [Reported] [2]
   - With no query, popular listings dominate the logs. How do you avoid a popularity feedback loop and still give new listings exposure?
9. A home page combines search, calendar availability and ranking. How do availability and date constraints enter the model: as hard filters, as features, or through the label (unavailable listings as negatives)? [Reported] [2]
10. How would you define a negative example in Airbnb search: an impressed and not clicked listing, a clicked and not booked listing, or a random listing? What bias does each choice introduce? [Likely] [6][8]
11. A booking requires host acceptance. How does that change your label definition and your model? [Likely] [6]
12. You have far fewer bookings than clicks. How does label sparsity drive your choice of objective, model capacity and evaluation? [Likely] [6]

### B. Learning to rank

13. Compare pointwise, pairwise and listwise learning-to-rank objectives. Which would you start with for Airbnb search, and why? [Likely] [6]
    - Airbnb's search team reported success with a pairwise formulation. Why might pairwise suit a booking-or-not label inside a search?
14. How do you build training pairs or lists from search logs (the booked listing vs. the other listings impressed in the same search)? [Likely] [6]
    - How many negatives per positive, and does sampling change the metric you are optimizing?
15. Explain how LambdaRank / LambdaMART connects a pairwise loss to NDCG. Why can't you optimize NDCG directly with gradient descent? [Likely]
16. Which offline metric would you use for search ranking (NDCG on the booked listing, MRR, recall@k, AUC), and why? Why might AUC improve while NDCG is flat? [Likely] [6]
17. How do you handle position bias in logged search data? [Likely] [7]
    - Compare position as a feature (with dropout, or set to a constant at inference), inverse propensity weighting and randomization-based estimates.
    - How would you estimate position propensities at Airbnb without hurting users?
18. Your ranker learns that cheap listings get booked. Why does pointwise use of price fail, and how would you encode price relative to the user's query and market? [Likely] [7]
19. A ranker trained on bookings learns the majority guest's preference and pushes one listing segment to the top for everyone. Why does that happen, and how do you correct it? [Likely] [21]
20. How do you rank diversely (avoid ten near-identical listings) without destroying relevance? [Likely] [10][21]
    - Is diversity a re-ranking step, a loss term or a list-aware model? What are the trade-offs?
    - How would you measure whether diversity helped bookings?
21. How would you design a two-stage system (retrieval + ranking) for Airbnb search, and what does each stage optimize? [Reported] [1]
    - How do you stop the ranker from never seeing listings that retrieval filters out (selection bias)? [Likely] [14]
22. Airbnb search is a multi-step journey (search → click → listing page → request → booking). How would you model it with multi-task learning? [Likely] [11]
    - How do you combine the task heads into a single ranking score?
    - How do you handle tasks with very different base rates and loss scales?
23. You have several ranking objectives (bookings, price competitiveness, host quality). Compare a weighted multi-objective loss with distilling the blended objective into one model. [Likely] [27]
24. How would you incorporate host-side objectives (acceptance likelihood, host preferences, fair exposure) into the ranking score? [Likely] [6]
25. Personalized search ranking: which user signals would you use, and how do you avoid overfitting to a user's last click? [Reported] [2]
26. How would you rank Airbnb Experiences differently from Stays, given much less data per item? [Likely] [9]
27. Listings are also shown as pins on a map, where there is no clear top-to-bottom order. How do your ranking metric and position-bias assumptions change? [Likely] [26]

### C. GBDT vs. neural networks

28. When would you choose GBDT over a deep NN for a tabular ranking problem, and vice versa? [Likely] [6]
    - Airbnb moved search from GBDT to a NN and initially saw little gain. Why does that happen, and what made the NN eventually win?
29. Compare a factorization machine with an embedding + DNN ranker for an Airbnb ranking problem. Which would you ship, and why? [Likely] [29]
30. A NN on the same features as your GBDT performs worse. What do you check first: normalization, distribution shape, learning rate, capacity or label noise? How do you tell which one it is? [Likely] [6]
31. GBDTs are invariant to monotone feature transforms; NNs are not. What normalization would you apply to price, review count and distance before a NN, and why might a log or Gaussian-rank transform matter for the NN but not the trees? [Likely] [6]
32. How would you use a GBDT and a NN together (GBDT leaves as NN features, stacking or distillation)? [Likely]
33. Why might listing-ID embeddings overfit in Airbnb search, when the same idea works for e-commerce item IDs? What about the booking-per-listing distribution makes it hard? [Likely] [6]
34. Deep models are harder to interpret. How do you explain to a PM or a host why a listing ranked lower? [Likely] [6]
    - Compare SHAP, partial dependence plots and ablation-based importance. Which are trustworthy for a NN?
35. Your NN is more accurate offline but has 3x the inference latency, and the interviewer pushes on the latency budget. How do you decide, and what closes the gap (distillation, caching, fewer candidates, smaller model)? [Reported] [1]
36. How would you trade off precision, recall, latency and infrastructure cost when choosing a model for production ranking? [Likely] [29]
37. How would you enforce a monotonicity requirement (for example, all else equal, a higher price must not raise the score)? [Likely] [7]
38. You need to classify listings using text, photos and tabular attributes together. How would you design a multimodal model, and when is it worth it over a GBDT on hand-built features? [Likely] [22]

### D. Embeddings and representation learning

39. How would you learn listing embeddings from user click sessions? [Likely] [8]
    - Why treat a session as a "sentence" and use skip-gram with negative sampling?
    - What is the effect of using the booked listing as global context?
    - Why sample negatives from the same market, and what goes wrong if you sample uniformly?
40. How do you produce an embedding for a brand-new listing with no sessions (cold start)? What are the failure modes of averaging nearby similar listings? [Likely] [8]
41. Bookings are too sparse to learn per-user embeddings. How would you learn user-type and listing-type embeddings in the same space? [Likely] [8]
42. How do you evaluate listing embeddings offline before they reach production? Which intrinsic checks (geography, price, style) and which extrinsic checks would you run? [Likely] [8]
43. Design listing recommendations (similar listings, or listings for a guest). How would you evaluate an embedding-based recommender offline and online? [Reported] [1][5]
44. Use embeddings for real-time personalization: how do you turn a user's in-session clicks and skips into ranking features, and how do you weight recent vs. older signals? [Likely] [8]
45. Design user embeddings for semantic search (free-text queries such as "cabin with a hot tub near a lake"). [Reported] [2]
    - How would you align queries and listings in one embedding space?
    - Which training pairs and hard negatives would you use?
46. How would you build embedding-based retrieval for Airbnb search and combine it with hard filters (dates, price, map bounds, guests)? Pre-filter vs. post-filter with ANN: what are the recall and latency trade-offs? [Likely] [14]
47. Two-tower vs. cross-encoder: where in the stack would you use each, and why? [Likely] [14]
48. Prices, photos and availability change daily. How do you keep embeddings and features fresh, and how do you measure what staleness costs you? [Reported] [1]
    - What breaks downstream when you retrain the embedding model and the space rotates? [Likely]
49. Spatial features: compare geohash buckets, learned location embeddings and raw latitude/longitude. Why can raw lat/long be a poor NN input, and how does geohash granularity affect generalization? [Reported] [1][3]
50. How would you use image embeddings from listing photos (room type, amenity detection, photo quality) as ranking or classification features? [Likely] [15][16]
51. When does a pre-trained LLM or foundation-model embedding beat a domain-trained listing embedding, and when does it not? [Likely]

### E. Calibration and probability quality

52. What is calibration, and why does it matter if you only use the score to rank? Name Airbnb consumers that need a calibrated probability (combining multi-task heads, expected-value LTV, trust thresholds). [Likely]
53. How do you measure calibration (reliability diagram, ECE, log loss vs. Brier score), and how do Platt scaling, isotonic regression and temperature scaling each fail? [Likely]
54. You down-sample negatives 10:1 for training. How do you correct the predicted probabilities? [Likely] [6]
55. Your model is well calibrated overall but miscalibrated for new listings and small markets. How do you detect and fix that? [Likely]
56. You combine pClick x pBook|click x pAccept into one ranking score. What happens if one head is miscalibrated? [Likely] [11]
57. Calibration drifts after a ranking change, because the model now trains on its own exposure. How do you monitor and recalibrate? [Likely]
58. In the bucketized LTV model, how do you check that the bucket probabilities are calibrated enough to compute an expected value? [Likely] [1]

### F. Feature engineering

59. Name the most important feature families for Airbnb search ranking and justify each. Which are query-dependent (dates, guest count, location) and which are listing-only? [Likely] [6]
60. Which concrete marketplace signals (reviews, host responsiveness, booking completion, photos) would you turn into features for listing quality or LTV, and how? [Reported] [1]
61. How would you build a feature for "price relative to the market for these dates and this listing type"? [Likely] [7][12]
62. How do you create features from reviews (count, average, recency, text) and historical CTR or booking rate without leaking future information? How do you guarantee point-in-time correctness in the training pipeline? [Likely] [25]
63. How would you compute per-guest cumulative booking counts and first and last booking times as point-in-time-correct model features? [Likely] [2]
64. How do you handle new listings whose historical features are missing: imputation, a missingness indicator or a separate model? How would you estimate their engagement? [Likely] [7]
65. How do you handle high-cardinality categorical features (city, host ID, amenity sets) in a GBDT vs. a NN? [Likely]
66. Which features would you compute in real time (session clicks, current filters), and which in batch? What does each choice cost? [Likely] [8]
67. How do you detect and prevent training-serving skew (a feature computed differently online than offline)? What does a shared feature platform such as Chronon solve, and what does it not? [Likely] [1][25]
68. How do you decide whether a new feature is worth its engineering and latency cost? What ablation and online test would you run? [Likely]
69. Seasonality and lead time strongly affect demand. How would you encode check-in date, lead time and length of stay? [Likely] [12]
70. How do you handle features that the ranking model itself influences (for example, a listing's CTR depends on how highly it was ranked)? [Likely] [7]

### G. Offline evaluation and the offline/online gap

71. How do you translate an offline metric (NDCG, AUC) into an expected online metric (bookings, revenue), and which A/B test would confirm it? [Reported] [1]
    - Have you validated the correlation between your offline metric and online lift across past launches?
72. Offline NDCG improved, but the online A/B test is flat or negative. List the hypotheses you would investigate, in order: training-serving skew, logging bias, metric mismatch, novelty, low power, interference between listings. How do you rule each out? [Likely] [6][29]
73. Why can a model trained on logs from the old ranker look good offline but fail online? How would off-policy evaluation (IPS, doubly robust) help, and what assumptions does it need? [Likely] [7][28]
74. When would you use interleaving instead of an A/B test to compare two rankers? [Likely] [13]
    - What does interleaving measure well, and what can it not tell you (revenue, long-term effects)?
    - How do you validate that interleaving outcomes agree with A/B outcomes?
75. How would you combine counterfactual (logged-data) evaluation with interleaving to filter ranking candidates before a full A/B test? [Likely] [28]
76. How do you build an offline evaluation set that represents the online traffic mix (markets, devices, new vs. returning users)? [Likely]
77. Your model wins in aggregate but loses in some markets or segments. How do you decide whether to launch? [Likely]
78. How do you guard against overfitting to the offline evaluation set after many iterations? [Likely]
79. Which primary metric should a ranking launch be judged on, and which guardrail metrics would block it? How do you make both robust to noise and gaming? [Reported] [2]
80. How would you set up a long-term holdout for ranking changes, and what does it show that a two-week A/B test cannot? [Likely]
81. What metric would you use when the goal is "a great stay", not just a booking (cancellations, reviews, rebookings)? [Likely] [11]
82. Airbnb converts short-term actions into a predicted future value so that teams can compare trade-offs. How would you build such a model, and how would you check it before using it as an experiment metric? [Likely] [19]
83. Once a model ships, which signals would you monitor to catch degradation before bookings drop? [Reported] [1]

### H. Training, generalization and debugging

84. How would you detect and fix overfitting in a deep ranking model trained on one year of search logs? [Likely] [6]
85. How do you split data for time-dependent problems (random vs. temporal vs. by user/listing), and why does it matter at Airbnb? [Likely]
86. How do you handle class imbalance (bookings much rarer than impressions): re-weighting, sampling or focal loss? Which of these break calibration? [Likely]
87. How often would you retrain the ranker, and how do you decide (fixed schedule vs. drift-triggered)? [Likely]
88. Model quality dropped after a holiday season. How do you tell concept drift from a data-pipeline bug? [Likely]
89. How do you make training reproducible enough that a 0.5% offline gain is real and not noise? How many seeds or bootstrap resamples would you run? [Likely]
90. How do you choose the loss for a pricing or demand model where under- and over-prediction have asymmetric costs? [Likely] [12]

### I. Regression, forecasting and pricing-flavoured modeling

91. Design a model that helps hosts set a price for each future date, using market demand. What exactly is the target: the booked price, the listed price, or a demand curve? [Reported] [4][5]
92. Predict the nightly price for a brand-new listing with no booking history. What features and model would you use, and how do you evaluate it? [Reported] [5]
93. Airbnb proposed a customized regression loss for price suggestions because true "optimal price" labels do not exist. How would you learn a price when you only see booked or not booked at a set price? [Likely] [12]
94. How would you estimate demand elasticity when price is confounded with listing quality and seasonality? [Likely] [12]
95. How would you evaluate a price-suggestion model offline when hosts can ignore the suggestion? [Likely] [12]
96. How would you forecast occupancy or demand for a listing 90 days out, and which uncertainty estimates would you provide? When would you use quantile regression instead of mean regression for host-facing estimates? [Likely]
97. Airbnb built an ML pipeline to predict the value of homes. How would you choose among candidate models for such a value model, and how would you check that it is fair across markets? [Likely] [18]

### J. Classification, trust and quality models

98. Design the label and model for detecting fake or fraudulent listings, when positives are rare and arrive late. How do you handle labels that only exist for cases an earlier model or rule flagged (selection bias)? [Likely]
99. Predict party risk at reservation time. The label (a reported party) is rare and noisy. How would you build it, and which features would you refuse to use for fairness reasons? [Likely] [1]
100. A property-damage risk score feeds a protection or screening decision. How do asymmetric error costs shape the threshold, and why does calibration matter here? [Likely] [1]
101. Build a classifier that flags "family friendly" listings. Where do labels come from (weak supervision, rules, raters), and how do you measure precision on a long tail of listing types? [Likely] [29]
102. How do you evaluate a classifier when precision matters far more than recall (for example, removing a host's listing)? [Likely]
103. How would you use text (descriptions, messages) in a classifier: TF-IDF + GBDT, a fine-tuned transformer, or LLM embeddings? [Likely] [22]
104. You must classify guest-host messages by intent, but no taxonomy exists yet. How would you discover the intent classes and then train a supervised classifier? [Likely] [23]
105. How would you build an active-learning or human-in-the-loop process to label listing categories or quality? [Likely] [17]
106. How would you check a ranking or trust model for unfair outcomes across host or guest groups? [Likely]

### K. Retrieval and LLM modeling for support (recent senior loops)

107. Recent senior loops asked ranking and retrieval questions about generative-AI support tools. How would you evaluate a retriever that finds the right policy passages for a support case (recall@k, nDCG on labeled pairs, LLM-as-judge)? [Reported] [1]
108. Generating customer-support instructions from company policies: when would you fine-tune an LLM, and when would retrieval-augmented generation be enough? How would you measure hallucination and policy compliance? [Reported] [1][2]
109. How would you rerank retrieved support documents, and what labels would you train the reranker on? [Likely] [1]

### L. Experimentation-adjacent modeling (role focus)

110. Your new ranker is evaluated by a user-randomized A/B test. Why can supply competition between listings bias the estimate, and how does that affect how you read a model's lift? [Likely] [13]
111. How would you use a model-based surrogate metric (predicted booking value) to read experiments faster? What validation does the surrogate need? [Likely] [19]
112. How would you use pre-experiment covariates (CUPED or regression adjustment) to reduce variance, and which model features make good covariates? [Likely]
113. You cannot randomize a host-facing model (for example, a pricing tool hosts opt into). How would you estimate its causal effect, and how could an ML-based counterfactual help? [Reported] [2]
    - What are the failure modes of ML-predicted counterfactuals compared with a randomized control? [Likely] [20]
114. Heterogeneous treatment effects: how would you model which users or markets benefit from a ranking change (uplift models, causal forests)? [Likely]
115. If you can run only one experiment this quarter, how do you choose among three model improvements using offline evidence? [Likely]
116. Which guardrail metrics would you set for launching a new ranking model, and how do you avoid guardrails blocking every launch through false alarms? [Likely] [24]

### M. Senior-level depth probes (deep dive and design rounds)

117. In the ML project you are walking us through, why did you choose this model, feature set or serving path over the alternatives? [Reported] [1]
118. In your most impactful model, which single modeling decision mattered most, and how do you know? [Likely] [1]
119. Tell me about a time a simpler model (heuristic, logistic regression, GBDT) beat a more complex one in production. Why did it win? [Likely]
120. What did you choose *not* to model, and why? [Likely]
121. How did you choose your offline metric, and did it ever mislead you? [Likely] [1]
122. Given twice the data, twice the features or twice the model capacity, which would you take for Airbnb search, and why? [Likely]
123. The interviewer redirects you mid-design (for example, from regression to classification). How do you restate the narrower scope and adapt your modeling plan? [Reported] [1]
124. How do you explain a ranking model's trade-offs to a PM who only cares about bookings? [Likely]

## References

[1]-[5], [25] and [30] were fetched and read on 2026-10-14. URLs marked "(URL confirmed)" match the title and URL listed in [30]. They could not be opened directly because arxiv.org and medium.com were unreachable. URLs marked "(not verified)" could not be checked in this session.

1. "Airbnb Machine Learning Engineer Interview Process" (TrueInterview guide, CC BY 4.0, built from 3 candidate reports, mainly 2026 Q1-Q2: LTV regression to bucketized steering, relisting and geohashing, cover photo with cold start, high-quality listings, latency/freshness pushback, offline-to-online link, GenAI support retrieval and ranking, monitoring, Chronon and training-serving skew in postings). kevin-2023-code/Tech-Interview-Questions, GitHub, 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/airbnb/interview-process-machine-learning-engineer.md
2. "Airbnb" question index with report dates ("Listing Lifetime Value — Estimation", May 2026; "Design User Embedding Semantic Search", Apr 2026; "Rank Home Search Results Without a Text Query", Sep 2026; "Home Page — Search + Availability + Ranking", Mar 2026; "Model Cumulative Booking Metrics and First and Last Booking Times", Sep 2026; "Design A Personalized Search Ranking System"; "Design Airbnb Relisting Detection"; "Design robust primary and guardrail metrics"; "Design an A/B test with causal inference"; "Generate Customer-Support Instructions from Company Policies"). kevin-2023-code/Tech-Interview-Questions, GitHub, 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/airbnb.md
3. "Airbnb interview process" (MLE candidates get one or two ML design rounds on search ranking, property LTV, recommendations, or relisting detection with geohashing; "Problem and label before model"). kevin-2023-code/Tech-Interview-Questions, GitHub, 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/airbnb/interview-process.md
4. 1point3acres post #40 (English mirror): project deep dive; coding LC 1298; ML design rounds on customer lifetime value and on helping hosts set prices. FlowSight/coding, GitHub. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/40.md
5. 1point3acres post #79 (English mirror): ML design rounds on listing recommendation and price prediction; ConnectN coding; an ML experience round that turned into an unplanned ML design question. FlowSight/coding, GitHub. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/79.md
6. Haldar et al., "Applying Deep Learning To Airbnb Search," arXiv 1810.09591 (KDD 2019). (URL confirmed) https://arxiv.org/abs/1810.09591
7. Haldar et al., "Improving Deep Learning For Airbnb Search," arXiv 2002.05515 (KDD 2020). (URL confirmed) https://arxiv.org/abs/2002.05515
8. Grbovic and Cheng, "Real-time Personalization using Embeddings for Search Ranking at Airbnb," KDD 2018, and the companion blog post "Listing Embeddings for Similar Listing Recommendations and Real-time Personalization in Search." (URLs confirmed) https://www.kdd.org/kdd2018/accepted-papers/view/real-time-personalization-using-embeddings-for-search-ranking-at-airbnb ; https://medium.com/airbnb-engineering/listing-embeddings-for-similar-listing-recommendations-and-real-time-personalization-in-search-601172f7603e
9. Grbovic, "Machine Learning-Powered Search Ranking of Airbnb Experiences," Airbnb Tech Blog, 2019. (URL confirmed) https://medium.com/airbnb-engineering/machine-learning-powered-search-ranking-of-airbnb-experiences-110b4b1a0789
10. "Learning To Rank Diversely," Airbnb Tech Blog, 2022. (URL confirmed) https://medium.com/airbnb-engineering/learning-to-rank-diversely-add6b1929621 . Paper version: Abdool, Haldar et al., "Learning To Rank Diversely At Airbnb," arXiv 2210.07774. (not verified) https://arxiv.org/abs/2210.07774
11. Tan et al., "Optimizing Airbnb Search Journey with Multi-task Learning," arXiv 2305.18431 (KDD 2023). (not verified) https://arxiv.org/abs/2305.18431
12. Ye et al., "Customized Regression Model for Airbnb Dynamic Pricing," KDD 2018. (not verified) https://dl.acm.org/doi/10.1145/3219819.3219830
13. "Beyond A/B Test: Speeding up Airbnb Search Ranking Experimentation through Interleaving," Airbnb Tech Blog, 2022. (URL confirmed) https://medium.com/airbnb-engineering/beyond-a-b-test-speeding-up-airbnb-search-ranking-experimentation-through-interleaving-7087afa09c8e
14. "Embedding-Based Retrieval for Airbnb Search," Airbnb Tech Blog, and "Applying Embedding-Based Retrieval to Airbnb Search," arXiv 2601.06873, 2026. (not verified; taken from Chapter 09) https://airbnb.tech/ai-ml/embedding-based-retrieval-for-airbnb-search/ ; https://arxiv.org/abs/2601.06873
15. "Categorizing Listing Photos at Airbnb," Airbnb Tech Blog, 2018. (URL confirmed) https://medium.com/airbnb-engineering/categorizing-listing-photos-at-airbnb-f9483f3ab7e3
16. "Amenity Detection and Beyond — New Frontiers of Computer Vision at Airbnb," Airbnb Tech Blog, 2019. (URL confirmed) https://medium.com/airbnb-engineering/amenity-detection-and-beyond-new-frontiers-of-computer-vision-at-airbnb-144a4441b72e
17. "Building Airbnb Categories with ML and Human-in-the-Loop," Airbnb Tech Blog, 2022. (URL confirmed) https://medium.com/airbnb-engineering/building-airbnb-categories-with-ml-and-human-in-the-loop-e97988e70ebb
18. "Using Machine Learning to Predict Value of Homes On Airbnb," Airbnb Tech Blog, 2017. (URL confirmed) https://medium.com/airbnb-engineering/using-machine-learning-to-predict-value-of-homes-on-airbnb-9272d3d4739d
19. "How Airbnb Measures Future Value to Standardize Tradeoffs," Airbnb Tech Blog, 2021. (URL confirmed) https://medium.com/airbnb-engineering/how-airbnb-measures-future-value-to-standardize-tradeoffs-3aa99a941ba5
20. "Artificial Counterfactual Estimation (ACE): Machine Learning-Based Causal Inference at Airbnb," Airbnb Tech Blog, 2022. (URL confirmed) https://medium.com/airbnb-engineering/artificial-counterfactual-estimation-ace-machine-learning-based-causal-inference-at-airbnb-ee32ee4d0512
21. Abdool et al., "Managing Diversity in Airbnb Search," arXiv 2004.02621, 2020. (URL confirmed) https://arxiv.org/abs/2004.02621
22. "WIDeText: A Multimodal Deep Learning Framework," Airbnb Tech Blog, 2020. (URL confirmed) https://medium.com/airbnb-engineering/widetext-a-multimodal-deep-learning-framework-31ce2565880c
23. "Discovering and Classifying In-app Message Intent at Airbnb," Airbnb Tech Blog, 2019. (URL confirmed) https://medium.com/airbnb-engineering/discovering-and-classifying-in-app-message-intent-at-airbnb-6a55f5400a0c
24. "Designing Experimentation Guardrails," Airbnb Tech Blog, 2021. (URL confirmed) https://medium.com/airbnb-engineering/designing-experimentation-guardrails-ed6a976ec669
25. "Chronon: a data platform for AI/ML" (Airbnb's open-source feature platform), airbnb/chronon, GitHub. Fetched 2026-10-14. https://github.com/airbnb/chronon
26. Haldar et al., "Learning to Rank for Maps at Airbnb," arXiv 2407.00091 (KDD 2024). (not verified; taken from Chapter 09) https://arxiv.org/abs/2407.00091
27. Tang, Gao, He and Katariya, "Multi-objective Learning to Rank by Model Distillation," arXiv 2407.07181 (KDD 2024). (not verified; taken from Chapter 09) https://arxiv.org/abs/2407.07181
28. Zhang et al., "Harnessing the Power of Interleaving and Counterfactual Evaluation for Airbnb Search Ranking," arXiv 2508.00751, 2025. (not verified; taken from Chapter 09) https://arxiv.org/abs/2508.00751
29. "Airbnb Senior Machine Learning Engineer Interview Experience — Four ML System Design Rounds, Rejected on Calibration," PracHub, 2026. Not reachable in this session. The topics cited here (FM vs. embedding + DNN; precision/recall/latency/cost trade-offs; offline NDCG up but A/B flat; a "family friendly" classifier) come from Chapter 09's summary. "Calibration" in the title may refer to level calibration, not probability calibration. https://prachub.com/interview-experiences/airbnb-senior-machine-learning-engineer-interview-experience-four-ml-system-design-rounds-rejected-on-calibration
30. Yan, "applied-ml: Papers & tech blogs by companies sharing their work on data science & machine learning in production" (used to confirm Airbnb paper and blog titles and URLs). eugeneyan/applied-ml, GitHub. Fetched 2026-10-14. https://github.com/eugeneyan/applied-ml
