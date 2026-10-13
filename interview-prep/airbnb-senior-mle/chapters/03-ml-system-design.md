# 03. ML System Design: Search Ranking, Embeddings, Pricing, Trust, Recommendations and More

> Research note (read first). Live web research was not possible when this chapter was written. The shared web-search budget for the run was already used up. The egress proxy also refused connections (DNS failure or HTTP 403) to glassdoor.com, interviewquery.com, prepfully.com, arxiv.org and medium.com, so no page could be fetched or re-read. The chapter is therefore built from what was already known about Airbnb's published ML systems and its widely described interview loop.
>
> Tagging policy, applied strictly: **no question in this chapter is tagged [Reported]**, because no candidate report could be verified for this run. Every question is tagged **[Likely]**. That means it is an inference from the role, the senior level and the systems Airbnb has described publicly. A citation next to a [Likely] question points to the Airbnb paper or blog post that describes the underlying system. It does **not** mean that anyone was asked the question. References marked "URL not verified" give the title and the blog index rather than a guessed article URL. Before the onsite, cross-check this list against Glassdoor, Blind and your recruiter's prep material [13][14][15].

## Format notes

- **Slot and length.** Airbnb's loop is widely described as including one ML/ML-system-design round for MLE candidates. It runs about 45-60 minutes and is held over video, with a shared doc or whiteboard [13][14]. Chapter 01 of this guide covers the full loop.
- **Interviewer.** Usually a senior or staff MLE from an ML-heavy area such as Search/Relevance, Pricing, Trust, Experiences/Recommendations or a platform team. Expect the interviewer to steer toward their own domain [Likely].
- **Shape of the round.** Typically one open-ended Airbnb product prompt ("design search ranking", "design a fraud model"). You are expected to drive it: clarify the business goal, pick the objective and labels, design data and features, choose a model, plan offline evaluation and online experimentation, cover serving and latency, and plan monitoring and iteration [Likely; 1][2].
- **Senior bar (L5).** Signals that the round assesses:
  - You frame the problem as a marketplace problem: guest and host, supply and demand.
  - You choose labels and objectives with care, including delayed or censored outcomes.
  - You propose a pragmatic baseline before deep models.
  - You reason about position bias, feedback loops and interference.
  - You design the **launch experiment** yourself. This matters especially for an experimentation-flavored role.
  - You discuss trade-offs without heavy prompting.

  Airbnb's own search papers emphasize simple baselines, failed ideas and lessons learned. Expect interviewers to reward the same humility and iteration [Likely; 1][2].
- **Experimentation weight.** For this role, assume every design will end with "how would you launch and measure it?" Expect probes on interleaving, guardrails, two-sided interference and long-horizon metrics [Likely; 8][9][17].

## Questions

### A. Search ranking for stays (core prompt)

1. Design the search ranking system for Airbnb stays: a guest enters a location, dates and guest count, and you return an ordered list of listings. [Likely] [1][2]
   - What is the business objective: bookings, booking value, guest satisfaction or long-term host and guest retention? How do you encode it?
   - What are the stages (candidate retrieval, filtering, first-pass ranker, second-pass re-ranker), and what is each stage's latency budget?
2. What label would you train on, given that bookings are rare and many searches end with no booking? [Likely] [1]
   - How would you use clicks, listing-page views, wishlist saves, contacts to host and bookings as weaker labels or auxiliary tasks?
   - How do you treat a booking request the host later rejected or cancelled?
3. How would you frame ranking: pointwise, pairwise or listwise? Why? [Likely] [1][2]
   - How would you build pairs from one search session with a single booked listing?
4. How do you correct for position bias in logged search data? [Likely] [2]
   - How would you model position as a feature during training, and what do you set it to at inference time?
   - How would you validate that your debiasing helped, rather than just shifting the distribution?
5. Which feature families would you build for listings, queries, guests and query-listing pairs? [Likely] [1]
   - How do you normalize price so that the model generalizes across markets and currencies?
   - How do you represent location relevance (distance to map center, the neighborhood the guest wanted)?
6. Your first model was a GBDT. When would you move to a neural network, and what would you expect to break? [Likely] [1]
   - How do you handle feature scaling, sparse categorical features and very wide value ranges in a neural ranker?
7. The ranker strongly prefers cheaper listings. Is that a bug or correct behavior? How would you investigate? [Likely] [1][2]
   - How would you make the model price-aware without hard-coding a monotonic price penalty?
8. How do you handle cold-start listings that have no booking or review history? [Likely] [2]
   - How do you estimate engagement for a new listing, and how do you give it controlled exposure without hurting conversion?
9. How would you add diversity to the results so the top of the page is not ten near-identical listings? [Likely] [5]
   - How would you measure diversity, and how would you trade it off against relevance in the loss?
10. How would you rank listings shown as map pins, where there is no clear top-to-bottom position? [Likely] [2]
    - How does user attention on a map differ from attention on a list, and how does that change your discounting?
11. How would you model the whole guest search journey (many searches across days before one booking) rather than a single search? [Likely] [6]
    - What would a multi-task model with heads for clicks, long views, wishlist and booking look like? How do you weight the heads?
12. How do you handle flexible-date or flexible-location searches ("I'm flexible", category browsing)? [Likely] [11]
13. How would you incorporate host-side outcomes (acceptance, cancellations, host earnings) into guest-facing ranking? [Likely] [1]
14. Offline NDCG went up but the online A/B test showed flat bookings. Walk me through your debugging plan. [Likely] [1][2]
    - How would you check for training-serving skew, feature leakage and a mismatch between offline and online candidate sets?
15. What offline evaluation set would you build, and how do you avoid leakage from future data or from repeat sessions by the same guest? [Likely] [1]

### B. Candidate retrieval and serving infrastructure for search

16. Design candidate retrieval that cuts millions of listings to a few thousand within strict latency, respecting date availability, guest count and map bounds. [Likely] [12]
    - Where would embedding-based retrieval fit next to hard filters, and how do you combine them?
17. How would you train a two-tower model for embedding-based retrieval of listings? [Likely] [12]
    - What is your negative-sampling strategy (in-batch, same-market, hard negatives)?
    - How do you handle availability, which changes daily and cannot easily live inside the embedding?
18. How would you serve a neural ranker at p99 latency under 200 ms for thousands of candidates per query? [Likely]
    - Which features are precomputed offline, which are computed in real time, and how do you keep them consistent?
19. How do you prevent training-serving skew in features such as "listing price for these dates" or "number of recent bookings"? [Likely] [1]
20. How often would you retrain the ranking model, and how would you detect that it has gone stale? [Likely]
21. What would you log at serving time so you can retrain, debias and debug later? [Likely] [2]

### C. Listing embeddings and real-time personalization

22. Design listing embeddings trained from guest click sessions, and use them for similar-listing recommendations and in-session personalization. [Likely] [3][10]
    - How do you define a session and its context window?
    - What negatives would you use? Why might random negatives make embeddings that only learn geography?
23. How would you change the embedding objective so that the booked listing influences every listing in the session? [Likely] [3]
24. How would you build user-type and listing-type embeddings to capture long-term preferences from sparse booking histories? [Likely] [3]
25. How would you create an embedding for a brand-new listing with no sessions? [Likely] [3]
26. How would you evaluate embeddings offline before using them in production? [Likely] [3]
    - What would a "rank of the eventually booked listing among the clicked listings" evaluation look like?
27. How would you turn in-session clicks and skips into real-time ranking features? [Likely] [3]
    - What streaming infrastructure is needed, and what is the freshness requirement?
28. When would you replace skip-gram embeddings with a sequence model or transformer over user actions? What would you expect to gain? [Likely]

### D. Pricing (host price suggestions / Smart Pricing)

29. Design a system that suggests a nightly price to hosts for each future date. [Likely] [4][16]
    - What is the objective: booking probability, expected host revenue, or marketplace occupancy?
    - Why can't you train directly on "the optimal price" as a label?
30. How would you predict booking probability as a function of price, lead time and listing quality? [Likely] [4]
    - How do you handle price being confounded with demand, because hosts raise prices when they expect demand?
31. How would you estimate price elasticity per listing or per market from observational data? [Likely] [4]
    - When would you need randomized price experiments, and what ethical and host-trust constraints apply?
32. Hosts can accept, modify or ignore your suggestion. How do you evaluate the model offline and online? [Likely] [4]
    - What is your unit of randomization, and how do you handle spillover between treated and control hosts in the same market?
33. How would you forecast market demand for a city on a given date (events, holidays, seasonality)? [Likely] [16]
34. How would you prevent the pricing model from producing extreme or volatile suggestions that destroy host trust? [Likely]
35. How would you design total-price display or fee-related ranking changes, and how would you measure whether guests trust the price more? [Likely]
36. A pricing-model launch raised bookings but cut average host revenue. How do you decide? [Likely] [17]

### E. Fraud, risk and trust and safety

37. Design a model that scores every booking for payment-fraud risk in real time. [Likely] [18]
    - Chargeback labels arrive weeks later. How do you train and evaluate with delayed labels?
    - How do you handle label censoring, where blocked transactions never reveal whether they were fraud?
38. Rather than block or allow, how would you design graduated friction (ID verification, extra payment checks) driven by risk scores? [Likely] [18]
    - How do you choose thresholds per action, given the cost of false positives (lost good guests) and false negatives?
39. Design an account-takeover detection system for guest and host logins. [Likely] [18]
40. Design a system that predicts and prevents unauthorized parties at listings. [Likely] [19]
    - What features are legitimate and fair to use, and which risk proxy discrimination (for example by age)?
    - How do you measure the system's impact when the outcome (a party happened) is rarely observed?
41. Design a fake or scam listing detector at listing-creation time. [Likely] [19]
    - How do you combine text, image, host-history and graph signals (shared devices, payment instruments)?
42. How would you use graph-based features or graph neural networks to detect fraud rings? [Likely]
43. Fraudsters adapt to your model within weeks. How do you design for adversarial drift? [Likely]
    - How do you keep a clean holdout to measure true fraud rates when the model blocks most of it?
44. How do you A/B test a fraud model when exposing the control group to fraud has real costs? [Likely] [18]
45. How do you build fairness checks into a trust model that affects who can book? [Likely] [19]

### F. Recommendations (Experiences, similar listings, discovery surfaces)

46. Design search ranking for Airbnb Experiences, which has far less data than stays. [Likely] [7]
    - How would you stage the system as data grows (heuristics, then GBDT, then personalization, then online scoring)?
47. Design "similar listings" recommendations on the listing detail page. [Likely] [3]
    - What is the right objective: clicks, bookings, or rescuing guests whose first choice is unavailable?
48. Design homepage recommendations for a logged-in guest with no active search ("where should I go next?"). [Likely]
49. Design personalized marketing or email recommendations (destinations, wishlisted listings with price drops). [Likely]
    - How do you measure incremental lift rather than attribution from people who would have booked anyway?
50. Design recommendations for cross-selling Experiences or Services to guests with an upcoming stay. [Likely]
51. How would you recommend wishlist listings or destinations for group trips where several people collaborate? [Likely]
52. How do you avoid a popularity feedback loop in which top listings get more exposure, more bookings and more exposure? [Likely] [5]

### G. Host acceptance, response time and other time-to-event predictions

53. Design a model that predicts the probability a host will accept a booking request, and use it in ranking. [Likely] [1]
    - How do you avoid unfairly penalizing a host for a single rejection?
    - How do you use host preferences (trip length, lead time, gaps between bookings) as features?
54. Design a model that predicts host response time to a guest message or booking request. [Likely]
    - How do you handle censoring for messages that never get a reply?
55. Predict the probability that a confirmed booking will be cancelled by the host or by the guest. [Likely]
    - How would you use this in ranking, in host coaching, or in guest-protection policy?
56. Predict booking lead time and length of stay for a search, and explain how they would change ranking. [Likely] [1]
57. Design a model that estimates customer-support wait time or resolution time, and routes cases to the right agent queue. [Likely] [20]
58. Which model families would you consider for time-to-event problems (survival models, hazard models, classification at fixed horizons)? Why? [Likely]

### H. Categories, content understanding and listing quality

59. Design the system behind Airbnb Categories (for example "Beachfront", "Treehouses"), which assigns listings to themed browse categories. [Likely] [11]
    - How do you combine ML candidates with human-in-the-loop review, and how do you budget human review?
    - How do you rank listings within a category, and how do you rank the categories themselves for a user?
60. Design an image-understanding pipeline that detects amenities and room types from listing photos. [Likely] [21]
    - How would you get labels cheaply, and how do you measure precision when amenities drive search filters?
61. How would you choose and order a listing's cover photo to maximize engagement without misleading guests? [Likely]
62. Design a listing-quality score used in ranking and host coaching (photos, description, completeness, reviews). [Likely]
    - How do you keep the score from simply rewarding listings that are already popular?
63. Design a system that detects inaccurate or misleading listing descriptions by comparing text, photos and reviews. [Likely]
64. How would you use LLMs or multimodal models for listing understanding at Airbnb's scale while controlling cost? [Likely]
65. How would you translate or localize listing content and decide when machine translation is good enough to show? [Likely]

### I. Reviews: quality, ranking and trust

66. Design a system that ranks reviews on a listing page so guests see the most helpful reviews first. [Likely]
    - How do you define "helpful" without explicit helpfulness votes?
67. Design a fake or retaliatory review detector. [Likely] [19]
68. Design a system that extracts aspects (cleanliness, location, accuracy) from review text and feeds them into search or listing quality. [Likely]
69. Design LLM-generated review summaries or highlights for a listing. [Likely]
    - How do you evaluate faithfulness, so the summary claims nothing the reviews do not support?
    - How do you A/B test whether summaries change booking or cancellation behavior?
70. Ratings are highly skewed toward 5 stars. How do you build a reliable quality signal from them? [Likely]
    - How would you apply Bayesian shrinkage for listings with few reviews?
71. How would you predict a guest's post-stay satisfaction before the stay, and how would you use it in ranking? [Likely] [1]

### J. Messaging, customer support and LLM systems

72. Design an ML system that classifies the intent of guest-host messages to route or auto-suggest replies. [Likely] [20]
73. Design an LLM-based customer-support assistant that resolves common issues (cancellations, refunds) and escalates the rest. [Likely] [20]
    - How do you evaluate it offline, monitor hallucinations and policy errors, and measure deflection versus satisfaction online?
    - What guardrails are needed for actions involving money or safety?
74. Design suggested replies for hosts. How do you avoid generating replies that commit the host to things they did not intend? [Likely]
75. How would you build retrieval-augmented generation over help-center articles and booking context? [Likely]
76. How would you decide whether to fine-tune, prompt or distill a model for a high-volume support classifier? [Likely]

### K. Experimentation and measurement inside the ML design round

77. For your ranking model, what is the primary metric of the launch A/B test, and what are the guardrails? [Likely] [8][17]
    - Why might "bookings" be a better primary metric than "booking requests" or "clicks"?
78. What randomization unit would you use for a search ranking test (guest, session, search, listing, market)? Why? [Likely] [8]
79. Explain why a guest-randomized test of a ranking change can be biased in a two-sided marketplace with limited inventory. [Likely] [8]
    - How could market-level, cluster or switchback designs help? What do they cost in statistical power?
80. When would you use interleaving instead of a classic A/B test for ranking? [Likely] [9]
    - How do you attribute a booking to one ranker in an interleaved result list?
    - How do you validate that interleaving and A/B testing agree on direction?
81. Bookings happen days after search, and stays happen weeks later. How do you get a sensitive, timely decision metric? [Likely] [8][17]
    - How would you build and validate a surrogate or predicted long-term value metric?
82. How would you reduce variance in a booking-conversion experiment (CUPED, stratification, trimming)? [Likely] [8]
83. How do you detect and handle sample ratio mismatch, bot traffic or logging bugs in a ranking experiment? [Likely] [8]
84. How do you handle novelty and primacy effects, and seasonality around holidays, in experiment readouts? [Likely]
85. Your model wins overall but loses badly in one market or one guest segment. What do you do? [Likely]
86. How would you set up a long-term holdout to measure the cumulative impact of many ranking launches? [Likely]
87. How would you measure the impact of a pricing or fraud model where the treatment changes host or fraudster behavior over time? [Likely] [4][18]
88. How would you use counterfactual or off-policy evaluation from logged ranking data before running an online test? [Likely] [2]
    - What propensities do you need to log, and how much randomization would you inject?
89. How do you trade off guest-side and host-side metrics in a single ship decision? [Likely] [17]

### L. Monitoring, reliability and iteration (senior depth)

90. What would you monitor once your ranking or fraud model is live (data drift, feature nulls, score distributions, business KPIs)? [Likely]
    - Which alerts would page an on-call engineer, and which go to a dashboard?
91. How would you roll out a new model safely (shadow scoring, canary, ramp, automatic rollback)? [Likely]
92. A feature pipeline broke silently and the model has served on stale features for three days. How do you detect it, mitigate it and prevent it from happening again? [Likely]
93. How do you decide what to build next after launch: more features, a bigger model, or better labels? [Likely] [1][2]
94. How would you reduce serving cost by 50% without measurable loss in bookings? [Likely]
    - What is your approach to distillation, caching and candidate-set pruning?
95. How would you design the system so that other teams (Experiences, Services, Long-term stays) can reuse your ranking stack? [Likely]
96. How would you explain to a product manager why a model change that "looks worse" in qualitative review should still ship? [Likely]
97. Which parts of this design would you not use ML for at all, and why? [Likely] [1]
98. What are the main fairness and discrimination risks in Airbnb ranking and trust systems, and how would you audit for them? [Likely] [19]

### M. Variations and curveball prompts

99. Design ranking for long-term stays (28+ nights). What changes from nightly stays? [Likely]
100. Design ranking for Airbnb Services or new verticals with almost no historical data. [Likely] [7]
101. Design autocomplete and query understanding for the search box (locations, landmarks, categories). [Likely]
102. Design a model that predicts which markets will face supply shortages for an upcoming event, and recommend actions to recruit hosts. [Likely]
103. Design a host-onboarding recommendation system that tells new hosts what to improve first. [Likely]
104. Design a system that detects duplicate listings (the same property listed several times or across accounts). [Likely]
105. Design a "price drop" or "rare find" badge. How would you verify that badges are accurate and actually help guests? [Likely]

## References

None of the sources below could be opened for this chapter: the egress proxy blocked arxiv.org, medium.com, glassdoor.com and the others. Each is listed from prior knowledge. Entries marked "URL not verified" give the title and the Airbnb tech blog index rather than a guessed article URL. Search the title there. Confirm every link before relying on it.

1. Haldar et al., "Applying Deep Learning to Airbnb Search", arXiv / KDD 2019. https://arxiv.org/abs/1810.09591
2. Haldar et al., "Improving Deep Learning for Airbnb Search", arXiv / KDD 2020. https://arxiv.org/abs/2002.05515
3. Grbovic & Cheng, "Real-time Personalization using Embeddings for Search Ranking at Airbnb", KDD 2018. https://dl.acm.org/doi/10.1145/3219819.3219885
4. Ye et al., "Customized Regression Model for Airbnb Dynamic Pricing", KDD 2018. https://dl.acm.org/doi/10.1145/3219819.3219830
5. Abdool, Haldar et al., "Learning To Rank Diversely At Airbnb", arXiv, 2022 (CIKM 2023). https://arxiv.org/abs/2210.07774
6. Tan et al., "Optimizing Airbnb Search Journey with Multi-task Learning", arXiv / KDD 2023. https://arxiv.org/abs/2305.18431
7. Grbovic, "Machine Learning-Powered Search Ranking of Airbnb Experiences", Airbnb Tech Blog (Medium), 2019. https://medium.com/airbnb-engineering/machine-learning-powered-search-ranking-of-airbnb-experiences-110b4b1a0789
8. "Experiments at Airbnb", Airbnb Engineering & Data Science (Medium), 2014. https://medium.com/airbnb-engineering/experiments-at-airbnb-e2db3abf39e7
9. "Beyond A/B Test: Speeding up Airbnb Search Ranking Experimentation through Interleaving", Airbnb Tech Blog (Medium), 2022. https://medium.com/airbnb-engineering/beyond-a-b-test-speeding-up-airbnb-search-ranking-experimentation-through-interleaving-7087afa09c8e
10. Grbovic, "Listing Embeddings in Search Ranking", Airbnb Tech Blog (Medium), 2018. https://medium.com/airbnb-engineering/listing-embeddings-for-similar-listing-recommendations-and-real-time-personalization-in-search-601172f7603e
11. "Building Airbnb Categories with ML and Human-in-the-Loop", Airbnb Tech Blog, 2022-2023. URL not verified. Search via https://medium.com/airbnb-engineering
12. "Embedding-Based Retrieval for Airbnb Search", Airbnb Tech Blog, 2025. URL not verified. Search via https://medium.com/airbnb-engineering
13. "Airbnb Machine Learning Engineer Interview Questions", Glassdoor (candidate reports), ongoing. https://www.glassdoor.com/Interview/Airbnb-Machine-Learning-Engineer-Interview-Questions-EI_IE391850.0,6_KO7,32.htm
14. "Airbnb Interview Process & Questions", IGotAnOffer, 2023-2024. https://www.igotanoffer.com/blogs/tech/airbnb-interview-process
15. "Airbnb Interview Discussions", LeetCode Discuss (search "Airbnb"), ongoing. https://leetcode.com/discuss/interview-question?currentPage=1&orderBy=hot&query=airbnb
16. "Learning Market Dynamics for Optimal Pricing", Airbnb Tech Blog, 2018. URL not verified. Search via https://medium.com/airbnb-engineering
17. "How Airbnb Measures Future Value to Standardize Tradeoffs", Airbnb Tech Blog, 2022. URL not verified. Search via https://medium.com/airbnb-engineering
18. "Fighting Financial Fraud with Targeted Friction", Airbnb Tech Blog, 2018. URL not verified. Search via https://medium.com/airbnb-engineering
19. Airbnb Trust & Safety and anti-discrimination posts (party prevention, Project Lighthouse), Airbnb Tech Blog and Airbnb Newsroom, 2020-2024. URL not verified. Search via https://medium.com/airbnb-engineering
20. Airbnb posts on message-intent classification and customer-support ML / LLM systems, Airbnb Tech Blog, 2019-2025. URL not verified. Search via https://medium.com/airbnb-engineering
21. Airbnb posts on amenity detection and room-type classification from listing photos, Airbnb Tech Blog, 2018-2019. URL not verified. Search via https://medium.com/airbnb-engineering
