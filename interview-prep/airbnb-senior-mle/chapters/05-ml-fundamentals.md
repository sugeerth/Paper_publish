# 05. ML Fundamentals and Modeling Depth: Ranking, GBDT vs. NN, Embeddings, Calibration, Features, Offline/Online Gap

> **Research note (read first).** The shared web-search budget for this run was already used up before this chapter began, so no new searches were possible. Direct fetches failed with DNS errors for every site tried except github.com: arxiv.org, medium.com, teamblind.com, igotanoffer.com, interviewquery.com and trueinterview.io. The only candidate-report material that could be read live is a GitHub-hosted compilation of TrueInterview data: an Airbnb MLE process write-up built from 3 candidate experiences, plus an Airbnb question index [1][2]. Two general format pages were also read [3][4]. The Airbnb papers and blog posts in References [5]-[14] are cited from prior knowledge as the public source of the systems that the [Likely] questions probe. They were **not re-fetched**. Confirm every link before relying on it.
>
> **Tags.** [Reported] means a fetched source attributes the prompt or topic to an Airbnb MLE interview. The citation says which source. [Likely] means an inference from the role, the Senior level and the systems Airbnb has published. A citation next to a [Likely] tag points to the paper behind the topic. It does **not** mean anyone was asked the question.

## Format notes

- **There is no standalone "ML fundamentals / ML theory" round in reported Airbnb MLE loops.** The fetched candidate-report summary says no reported round asked for derivations, loss-function proofs or architecture comparisons. "Paper-level theory" and naming a fashionable model without a metric are listed as *not tested* [1].
- **Where fundamentals actually show up:** as depth probes inside the **ML system design** round (one or two rounds, 45-60 min each, run by senior/staff MLEs, described as the core and decisive round of the track). They also show up in the **experience / project deep dive** (45-60 min, a single past project). ML design prompts are sometimes folded into the 45-60 min technical phone screen too [1].
- **Expected structure in those rounds:** framing → label definition → data sources → features → model choice → offline evaluation → serving, latency, freshness, cold start for new listings → experimentation. Model choice took little time in the reported rounds. **Jumping straight to the model is the most-cited failure** [1].
- **Recurring reported follow-up themes:** turning an offline metric into an online one, spatial features such as geohashing, and framing a target as regression vs. bucketized classification. In the lifetime-value prompt the interviewer pushed toward bucketized classification [1].
- **Senior (L5) bar:** the problem scope you owned matters more than model sophistication [1]. Expect "why" questions on every choice: why this label, why GBDT or NN, why this metric, and what you would expect to move online. Questions in this chapter are written as those depth probes. Practise answering each one in 2-4 minutes, tied to an Airbnb surface.
- **Airbnb context that interviewers know well:** search moved from GBDT to deep NNs [5][6]. In-session listing embeddings [7][8], multi-objective journey ranking [10], learning to rank diversely [9], a customized pricing regression loss [11] and interleaving for ranking experiments [12] are all public. Expect probes that check whether you can reason about the same trade-offs.

## Questions

### A. Problem framing, labels and targets

1. You are asked to estimate the lifetime value of a listing (or property). Would you frame it as regression or as bucketized classification, and why? [Reported] [1][2]
   - The interviewer pushes you toward bucketized classification. What do you gain and lose?
   - How do you choose bucket boundaries, and how do you turn bucket probabilities back into an expected value?
   - How do you handle the heavy right tail and listings that churn off the platform (censoring)?
2. How would you identify "high-quality listings"? What is your label, given that no ground truth exists? [Reported] [1]
   - Which proxies (reviews, rebook rate, cancellations, CS tickets) would you combine, and how do you avoid circularity with the ranking model's own exposure?
3. Choose a listing's cover photo to lift bookings. Is this a supervised problem, a ranking problem or a bandit problem? [Reported] [1]
   - What label would you train on, given that the cover photo you showed also caused the clicks?
   - How do you handle listings with only a few photos or a few impressions?
4. Detect relisting (a host or listing removed for policy reasons reappears under a new account). What features and model family would you use? [Reported] [1][2]
   - Is this classification, similarity search or entity resolution? Defend the framing.
   - How do you choose an operating threshold when false positives hurt legitimate hosts?
5. For search ranking, what is the training label: click, long click, wishlist, booking request, accepted booking or completed stay without cancellation? [Likely] [5][10]
   - How do you weight these in a single objective vs. a multi-task model?
   - How do you deal with label delay (booking days after search)?
6. Rank home search results when the user gives no text query (only location and dates, or none at all). What does "relevance" mean here, and what is the label? [Reported] [2]
7. How would you define a negative example in Airbnb search: an impressed and not clicked listing, a clicked and not booked listing, or a random listing? [Likely] [5][7]
   - What bias does each choice introduce?
8. A booking requires host acceptance. How does that change your label definition and your model? [Likely] [5]
9. When would you predict a calibrated probability rather than a ranking score, and which downstream consumers in Airbnb need which? [Likely]
10. You have far fewer bookings than clicks. How does label sparsity drive your choice of objective, model capacity and evaluation? [Likely] [5]

### B. Learning to rank

11. Compare pointwise, pairwise and listwise learning-to-rank objectives. Which would you start with for Airbnb search, and why? [Likely] [5]
    - Airbnb's search team reported success with a pairwise formulation. Why might pairwise suit a booking-or-not label inside a search?
12. How do you build training pairs or lists from search logs (booked listing vs. the other listings impressed in the same search)? [Likely] [5]
    - How many negatives per positive, and does sampling them change the metric you are optimizing?
13. Explain how LambdaRank / LambdaMART connects a pairwise loss to NDCG. [Likely]
    - Why can't you optimize NDCG directly with gradient descent?
14. What offline metric would you use for search ranking (NDCG on the booked listing, MRR, recall@k, AUC)? Why? [Likely] [5]
    - Why might AUC improve while NDCG is flat?
15. How do you handle position bias in logged search data? [Likely] [6]
    - Compare including position as a feature (with dropout or set to a constant at inference), inverse propensity weighting and randomization-based estimates.
    - How would you estimate position propensities at Airbnb without hurting users?
16. Your ranker learns that cheap listings get booked. Why does pointwise use of price fail, and how would you encode price relative to the user's query and market? [Likely] [6]
17. How do you rank diversely (avoid ten near-identical listings) without destroying relevance? [Likely] [9]
    - Is diversity a re-ranking step, a loss term or a list-aware model? Trade-offs?
    - How would you measure whether diversity helped bookings?
18. How would you design a two-stage system (retrieval + ranking) for Airbnb search, and what does each stage optimize? [Likely] [13]
    - How do you avoid the ranker never seeing listings that retrieval filters out (selection bias)?
19. Airbnb search is a multi-step journey (search → click → listing page → request → booking). How would you model it with multi-task learning? [Likely] [10]
    - How do you combine task heads into a single ranking score?
    - How do you handle tasks with very different base rates and loss scales?
20. How would you incorporate host-side objectives (acceptance likelihood, host preferences, fairness of exposure) into the ranking score? [Likely] [5]
21. Personalized search ranking: which user signals would you use, and how do you avoid overfitting to a user's last click? [Reported] [2]
22. How would you rank Airbnb Experiences differently from Stays, given much less data per item? [Likely] [8]

### C. GBDT vs. neural networks

23. When would you choose GBDT over a deep neural network for a tabular ranking problem, and vice versa? [Likely] [5]
    - Airbnb moved search from GBDT to a NN and initially saw little gain. Why does that happen, and what made the NN eventually win?
24. A NN on the same features as your GBDT performs worse. What do you check first? [Likely] [5]
    - Feature normalization, distribution shape, learning rate, capacity, or label noise: how do you tell which?
25. GBDTs are invariant to monotone feature transforms; NNs are not. What normalization would you apply to price, review count and distance before a NN? [Likely] [5]
    - Why might a log transform or a Gaussian rank transform matter for the NN but not for the trees?
26. How would you use a GBDT and a NN together (GBDT leaves as NN features, stacking, or distillation)? [Likely]
27. Why might adding listing-ID embeddings to the NN overfit at Airbnb, while the same idea works for e-commerce item IDs? [Likely] [5]
    - What about the booking-per-listing distribution makes it hard?
28. Deep models are harder to interpret. How do you explain to a PM or a host why a listing ranked lower? [Likely] [5]
    - Compare SHAP, partial dependence plots and ablation-based feature importance. Which are trustworthy for a NN?
29. Your NN is more accurate offline but has 3x the inference latency. How do you decide, and what can you do to close the gap? [Likely]
30. How do you set the main GBDT hyperparameters (depth, learning rate, number of trees, min child weight), and which ones matter most for overfitting? [Likely]
31. What kinds of feature interactions does a GBDT capture well, and which ones does it capture badly? [Likely]
32. How would you handle a monotonicity requirement (for example, ranking must not reward higher price, all else equal)? [Likely] [6]

### D. Embeddings and representation learning

33. How would you learn listing embeddings from user click sessions? [Likely] [7]
    - Why treat a session as a "sentence" and use skip-gram with negative sampling?
    - What is the effect of using the booked listing as global context?
    - Why sample negatives from the same market, and what goes wrong if you sample uniformly?
34. How do you produce embeddings for a brand-new listing with no sessions (cold start)? [Likely] [7]
    - Averaging embeddings of nearby, similar listings: what are the failure modes?
35. Bookings are too sparse to learn per-user embeddings. How would you learn user-type and listing-type embeddings in the same space? [Likely] [7]
36. How do you evaluate listing embeddings offline before they reach production? [Likely] [7]
    - What intrinsic checks (geography, price, architecture style) and which extrinsic checks would you run?
37. Use embeddings for real-time personalization: how do you turn a user's in-session clicks and skips into ranking features? [Likely] [7]
    - How do you weight recent clicks vs. older ones, and how do you use negative signals (skipped listings)?
38. Design user embeddings for semantic search (free-text queries such as "cabin with a hot tub near a lake"). [Reported] [2]
    - How would you align text queries and listings in the same embedding space?
    - Which training pairs and hard negatives would you use?
39. How would you build embedding-based retrieval for Airbnb search, and how do you combine it with hard filters (dates, price, map bounds, guests)? [Likely] [13]
    - Pre-filter vs. post-filter with ANN: what are the recall and latency trade-offs?
40. Two-tower vs. cross-encoder: where in the stack would you use each, and why? [Likely] [13]
41. How do you keep embeddings fresh when listings change (price, photos, availability) daily? [Likely]
    - What breaks downstream when you retrain the embedding model and the space rotates?
42. How do you handle spatial information: geohashing, learned location embeddings or raw latitude/longitude? [Reported] [1]
    - Why can raw lat/long be a poor NN input, and how do geohash granularity choices affect generalization?
43. How would you use image embeddings from listing photos (amenity detection, room type, photo quality) as ranking features? [Likely] [14]
44. When does a pre-trained LLM / foundation-model embedding beat a domain-trained embedding for listings, and when does it not? [Likely]

### E. Calibration and probability quality

45. What is calibration, and why does it matter if you only use the score to rank? [Likely]
    - Name two Airbnb consumers where a calibrated probability is required (for example combining multi-task heads, expected-value pricing, fraud thresholds).
46. How do you measure calibration (reliability diagram, ECE, log loss vs. Brier score)? [Likely]
47. Compare Platt scaling, isotonic regression and temperature scaling. When would each fail? [Likely]
48. You down-sample negatives 10:1 for training. How do you correct the predicted probabilities? [Likely] [5]
49. Your model is well calibrated overall but miscalibrated for new listings and small markets. How do you detect and fix that? [Likely]
50. You combine pClick x pBook|click x pAccept into one ranking score. What happens if one head is miscalibrated? [Likely] [10]
51. Calibration drifts after a ranking change, because the model now trains on its own exposure. How do you monitor and recalibrate? [Likely]
52. For a fraud or trust model, how do you choose the decision threshold, and how do you communicate the precision-recall trade-off? [Likely]

### F. Feature engineering

53. Name the 10 most important feature families for Airbnb search ranking, and justify each. [Likely] [5]
    - Which ones are query-dependent (dates, guest count, location) and which are listing-only?
54. How would you build a feature for "price relative to the market for these dates and this listing type"? [Likely] [6][11]
55. How do you engineer location features: distance to the searched point, neighbourhood popularity, geohash-level booking rates? [Reported] [1]
56. How do you create features from reviews (count, average, recency, text) without leaking future information? [Likely]
57. How do you avoid target leakage when computing historical listing CTR or booking rate as a feature? [Likely]
    - Point-in-time correctness: how do you guarantee it in your training data pipeline?
58. How do you handle new listings whose historical features are missing? Imputation, a missingness indicator or a separate model? [Likely] [6]
    - Airbnb has written about new listings being disadvantaged in search. How would you estimate their engagement?
59. How do you handle high-cardinality categorical features (city, host ID, amenity sets) in a GBDT vs. a NN? [Likely]
60. Which features would you compute in real time (session clicks, current search filters), and which in batch? What does each choice cost? [Likely] [7]
61. How do you detect that a feature is computed differently at serving time than in training (training-serving skew)? [Likely]
62. How do you decide whether a new feature is worth its engineering and latency cost? [Likely]
    - What ablation and online test would you run?
63. Seasonality and lead time strongly affect demand. How would you encode check-in date, lead time and length of stay? [Likely] [11]
64. How would you build features capturing host quality and host responsiveness? [Likely] [5]
65. How do you handle features that the ranking model itself influences (for example, a listing's CTR depends on how highly it was ranked)? [Likely] [6]

### G. Offline evaluation and the offline/online gap

66. How do you translate an offline metric (NDCG, AUC) into an expected online metric (bookings, revenue)? [Reported] [1]
    - Have you ever validated the correlation between your offline metric and online lift across past launches?
67. Offline NDCG improved by 2%, but the online A/B test showed no booking lift. List the hypotheses you would investigate, in order. [Likely] [5]
    - Training-serving skew, logging bias, metric mismatch, a novelty effect, the experiment being underpowered, or interference between listings: how do you rule each out?
68. Why can a model trained on logs from the old ranker look good offline but fail online (feedback loops, counterfactual evaluation)? [Likely] [6]
    - How would off-policy evaluation (IPS, doubly robust) help, and what assumptions does it need?
69. How do you build an offline evaluation set that represents the online traffic mix (markets, devices, new vs. returning users)? [Likely]
70. What is a "replay" or "re-rank the logged results" evaluation, and what are its blind spots? [Likely]
71. When would you use interleaving instead of an A/B test to compare two rankers? [Likely] [12]
    - What does interleaving measure well, and what can it not tell you (for example, revenue or long-term effects)?
    - How do you validate that interleaving outcomes agree with A/B outcomes?
72. Your model wins in aggregate but loses in some markets or segments. How do you decide whether to launch? [Likely]
73. How do you guard against overfitting to the offline evaluation set after many iterations? [Likely]
74. How do you estimate the online impact of a model before running an A/B test, to prioritize which experiment to run? [Likely]
75. How would you set up a long-term holdout for ranking changes, and what would you learn from it that an A/B test cannot show? [Likely]
76. What metric would you use when the goal is "a great stay", not just a booking (cancellations, reviews, rebookings)? [Likely] [10]

### H. Training, generalization and debugging

77. Walk through how you would detect and fix overfitting in a deep ranking model trained on one year of search logs. [Likely] [5]
78. How do you split data for time-dependent problems (random vs. temporal vs. by user/listing), and why does it matter at Airbnb? [Likely]
79. Your training loss decreases but validation NDCG oscillates. What could cause it? [Likely]
80. How do you handle class imbalance (bookings much rarer than impressions): re-weighting, sampling or focal loss? [Likely]
    - Which of these break calibration?
81. How often would you retrain the ranker, and how do you decide (fixed schedule vs. drift-triggered)? [Likely]
82. Model quality dropped after a holiday season. How do you diagnose concept drift vs. data pipeline bugs? [Likely]
83. How do you make training reproducible enough that a 0.5% offline gain is real and not noise? [Likely]
    - How many seeds or bootstrap resamples would you run before trusting a gain?
84. How do you choose the loss for a pricing or demand model where under- and over-prediction have asymmetric costs? [Likely] [11]
85. How would you use dropout, early stopping and weight decay together, and how do you choose among them? [Likely]

### I. Regression, forecasting and pricing-flavoured modeling

86. Airbnb proposed a customized regression loss for price suggestions, because true "optimal price" labels do not exist. How would you frame learning a price when you only observe booked or not booked at a set price? [Likely] [11]
87. How would you estimate demand elasticity when price is confounded with listing quality and seasonality? [Likely] [11]
88. How would you evaluate a price-suggestion model offline when hosts can ignore the suggestion? [Likely] [11]
89. How would you forecast occupancy or demand for a listing 90 days out, and which uncertainty estimates would you provide? [Likely]
90. Quantile regression vs. mean regression: when would you use each for host-facing estimates (earnings, nightly price)? [Likely]

### J. Classification, trust and quality models

91. Design the label and model for detecting fake or fraudulent listings, given that positives are rare and arrive with delay. [Likely]
    - How do you handle labels that only exist for cases that an earlier model or rule flagged (selection bias)?
92. How do you evaluate a classifier when precision matters far more than recall (for example, removing a host's listing)? [Likely]
93. How would you use text (listing description, messages) in a classifier: TF-IDF + GBDT, fine-tuned transformer, or LLM embeddings? Trade-offs? [Likely]
94. How do you build an active-learning or human-in-the-loop loop to label listing categories or quality? [Likely] [14]
95. How would you check a ranking or trust model for unfair outcomes across host or guest groups? [Likely]

### K. Experimentation-adjacent modeling (role focus)

96. Your new ranker is evaluated by user-randomized A/B. Why can supply competition between listings bias the estimate, and how does that affect how you interpret a model's lift? [Likely] [12]
97. How would you use a model-based surrogate metric (predicted booking value) to read experiments faster? What validation does the surrogate need? [Likely]
98. How would you use pre-experiment covariates (CUPED or regression adjustment) to reduce variance, and which model features make good covariates? [Likely]
99. Heterogeneous treatment effects: how would you model which users or markets benefit from a ranking change (uplift models, causal forests)? [Likely]
100. How do you decide between a contextual bandit and a supervised model plus A/B test for something like cover-photo selection? [Likely] [1]
101. If you can only run one experiment this quarter, how do you choose among three model improvements using offline evidence? [Likely]

### L. Senior-level depth probes (from the deep dive and design rounds)

102. In your most impactful model, which single modeling decision mattered most, and how do you know? [Likely] [1]
103. Tell me about a time a simpler model (heuristic, logistic regression, GBDT) beat a more complex one in production. Why did it win? [Likely] [1]
104. What did you choose *not* to model, and why? [Likely]
105. How did you choose your offline metric, and did it ever mislead you? [Likely] [1]
106. If I gave you twice the data vs. twice the features vs. twice the model capacity, which would you take for Airbnb search, and why? [Likely]
107. How do you explain a ranking model's trade-offs to a PM who only cares about bookings? [Likely]

## References

[1]-[4] were fetched live for this chapter. [5]-[14] are cited from prior knowledge and were **not** fetched, because DNS resolution failed for arxiv.org, medium.com and other non-GitHub hosts. Verify those links before relying on them.

1. "Airbnb Machine Learning Engineer Interview Process" (TrueInterview data, built from 3 candidate-reported experiences, CC BY 4.0), GitHub: kevin-2023-code/Tech-Interview-Questions, 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/airbnb/interview-process-machine-learning-engineer.md
2. "Airbnb" company question index (lists reported prompts incl. "Listing Lifetime Value — Estimation", "Design User Embedding Semantic Search", "Rank Home Search Results Without a Text Query", "Design A Personalized Search Ranking System", "Design Airbnb Relisting Detection"), GitHub: kevin-2023-code/Tech-Interview-Questions, 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/airbnb.md
3. "Interview formats of top companies" (Airbnb section), Tech Interview Handbook, GitHub: yangshun/tech-interview-handbook, ongoing. https://github.com/yangshun/tech-interview-handbook/blob/main/apps/website/contents/interview-formats-top-companies.md
4. "Airbnb" company prep page (confirms no ML-specific section; general SWE loop), GitHub: jugaldb/faang-interview-prep, 2026. https://github.com/jugaldb/faang-interview-prep/blob/main/docs/companies/airbnb.md
5. Haldar et al., "Applying Deep Learning to Airbnb Search", arXiv / KDD 2019. https://arxiv.org/abs/1810.09591
6. Haldar et al., "Improving Deep Learning for Airbnb Search", arXiv / KDD 2020. https://arxiv.org/abs/2002.05515
7. Grbovic & Cheng, "Real-time Personalization using Embeddings for Search Ranking at Airbnb", KDD 2018. https://dl.acm.org/doi/10.1145/3219819.3219885
8. Grbovic, "Machine Learning-Powered Search Ranking of Airbnb Experiences", Airbnb Tech Blog (Medium), 2019. https://medium.com/airbnb-engineering/machine-learning-powered-search-ranking-of-airbnb-experiences-110b4b1a0789
9. Abdool, Haldar et al., "Learning To Rank Diversely At Airbnb", arXiv, 2022. https://arxiv.org/abs/2210.07774
10. Tan et al., "Optimizing Airbnb Search Journey with Multi-task Learning", arXiv / KDD 2023. https://arxiv.org/abs/2305.18431
11. Ye et al., "Customized Regression Model for Airbnb Dynamic Pricing", KDD 2018. https://dl.acm.org/doi/10.1145/3219819.3219830
12. "Beyond A/B Test: Speeding up Airbnb Search Ranking Experimentation through Interleaving", Airbnb Tech Blog (Medium), 2022. https://medium.com/airbnb-engineering/beyond-a-b-test-speeding-up-airbnb-search-ranking-experimentation-through-interleaving-7087afa09c8e
13. "Embedding-Based Retrieval for Airbnb Search", Airbnb Tech Blog, 2025. The article URL was not verified. Search for the title via https://medium.com/airbnb-engineering
14. Airbnb Tech Blog posts on listing-photo amenity detection / room classification (2018-2019) and on building Airbnb Categories with ML and human-in-the-loop (2022-2023). The article URLs were not verified. Search for the titles via https://medium.com/airbnb-engineering
