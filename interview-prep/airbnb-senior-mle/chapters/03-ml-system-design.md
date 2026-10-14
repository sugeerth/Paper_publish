# 03. ML System Design: Search Ranking, Embeddings, Pricing, Trust, Recommendations and More

> Research note (read first). This chapter was reviewed again on 2026-10-14. The shared web-search budget was used up, and the network proxy could not resolve arxiv.org, dl.acm.org, medium.com, airbnb.tech, news.airbnb.com, kdd.org, glassdoor.com, teamblind.com, igotanoffer.com, interviewquery.com or trueinterview.io. Reddit refused the fetch. **GitHub was the only reachable source.** Four candidate-report pages were fetched and read there:
>
> - a TrueInterview-derived summary of the Airbnb MLE loop, built from 3 candidate reports (mostly 2026 Q1-Q2) [22];
> - a TrueInterview-derived Airbnb question index with report dates [23];
> - two translated 1point3acres (一亩三分地) Airbnb onsite posts that list the ML design rounds [24][25].
>
> A general ML-system-design question list was also read [36].
>
> Tagging policy:
>
> - **[Reported]** means one of those fetched pages names the prompt, or a close variant of it, for Airbnb. The TrueInterview question pages [26]-[35] are cited only as the links those pages give. They could not be opened directly.
> - **[Reported, role unspecified]** means the prompt is in the Airbnb index [23], but the page does not say whether it came from an MLE or a data-science loop.
> - **[Likely]** means the question is inferred from the role, the senior level and Airbnb's published systems. A paper or blog citation next to [Likely] points to the system the question probes. It does **not** mean anyone was asked the question.
>
> The Airbnb papers and blog posts [1]-[12] and [16]-[21], [37] and [38] could not be re-fetched. They are listed from prior knowledge. Before the onsite, cross-check this list against Glassdoor, IGotAnOffer and LeetCode Discuss [13][14][15]. Those sites were unreachable for this review.

## Format notes

- **Slot and length.** The MLE-track summary says the loop is "filtered by design depth, not by code". It describes **1-2 ML system design rounds** as the decisive rounds; the level guidance applies from Senior (G9) up [22].
  - One 1point3acres loop had two ML design rounds: guest/customer LTV and host pricing [24].
  - Another loop also had two: listing recommendation and price prediction. In the same loop, the "ML experience" round unexpectedly turned into a further ML design question [25].
  - Rounds run about 45-60 minutes over video, with a shared doc or whiteboard [22].
- **Interviewer.** Usually a senior or staff MLE from an ML-heavy area such as Search/Relevance, Pricing, Trust, Experiences/Recommendations, Customer Support/GenAI or a platform team. Expect the interviewer to steer toward their own domain [Likely]. Recent senior loops reportedly added retrieval and ranking prompts around **generative-AI customer-support tools** [22].
- **Shape of the round.** Typically one open-ended Airbnb product prompt, such as "design search ranking", "estimate a listing's lifetime value" or "detect relisted properties". You are expected to drive it:
  - Clarify the business goal and **agree a definition** of the target. In one LTV round the interviewer first made the candidate define "lifetime value" [22].
  - Pick the labels, design data and features, and choose a model.
  - Plan offline evaluation and online experimentation.
  - Cover serving and latency, then monitoring and iteration [22][1][2].

  Interviewers reportedly steer mid-round, for example from regression to bucketized classification in the LTV round, or into geohashing in the relisting round [22].
- **Senior bar.** The round looks for these signals:
  - You frame the problem as a marketplace with two sides: guest and host, supply and demand.
  - You choose labels and objectives with care, including delayed or censored outcomes.
  - You propose a pragmatic baseline before reaching for deep models.
  - You reason about position bias, feedback loops and interference.
  - You design the **launch experiment** yourself.
  - You discuss trade-offs without heavy prompting.

  Airbnb's own search papers stress simple baselines, failed ideas and lessons learned [Likely; 1][2].
- **Experimentation weight.** For this role, assume every design ends with "how would you launch and measure it?" Expect probes on interleaving, guardrails, two-sided interference and long-horizon metrics [Likely; 8][9][17]. Experiment-design prompts also appear in the Airbnb question index [23].

## Questions

### A. Search ranking for stays (core prompt)

1. Design Airbnb search ranking, from candidate generation through offline and online evaluation and deployment: a guest enters a location, dates and guest count, and you return an ordered list of listings. [Reported] [22][26]
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
10. How would you rank listings shown as map pins, where there is no clear top-to-bottom position? [Likely] [2][38]
    - How does user attention on a map differ from attention on a list, and how does that change your discounting?
11. How would you model the whole guest search journey (many searches across days before one booking) rather than a single search? [Likely] [6]
    - What would a multi-task model with heads for clicks, long views, wishlist and booking look like? How do you weight the heads?
12. How do you handle flexible-date or flexible-location searches ("I'm flexible", category browsing)? [Likely] [11]
13. How would you incorporate host-side outcomes (acceptance, cancellations, host earnings) into guest-facing ranking? [Likely] [1]
14. Offline NDCG went up but the online A/B test showed flat bookings. Walk me through your debugging plan. [Likely] [1][2]
    - How would you check for training-serving skew, feature leakage and a mismatch between offline and online candidate sets?
15. What offline evaluation set would you build, and how do you avoid leakage from future data or from repeat sessions by the same guest? [Likely] [1]
16. Rank home-page search results when the guest has typed no text query, for example on app open or a logged-in home feed. [Reported] [23][29]
    - What signals replace the query: past searches, wishlists, location, season, booking history?
    - How do you handle a brand-new guest with no history?
    - How does this ranking differ from query-driven ranking in objective and evaluation?
17. A guest types a natural-language query ("cabin near a lake with a hot tub for six next weekend"). How would you parse it into hard filters, soft preferences and a semantic retrieval query? [Likely] [12]
    - How do you evaluate parsing errors that silently drop a hard constraint?
    - Where would you use an LLM, and how would you bound its latency and cost on the search path?

### B. Candidate retrieval and serving infrastructure for search

18. Design candidate retrieval that cuts millions of listings to a few thousand within strict latency, respecting date availability, guest count and map bounds. [Likely] [12]
    - Where would embedding-based retrieval fit next to hard filters, and how do you combine them?
19. How would you train a two-tower model for embedding-based retrieval of listings? [Likely] [12]
    - What is your negative-sampling strategy (in-batch, same-market, hard negatives)?
    - How do you handle availability, which changes daily and cannot easily live inside the embedding?
20. Design semantic search driven by user embeddings: represent each guest by an embedding and retrieve listings or destinations that match their intent. [Reported] [23][31]
    - How do you build and refresh the user embedding from sparse, long-gap travel histories?
    - How do you serve approximate nearest-neighbor search under geography and availability filters?
    - How do you evaluate retrieval recall separately from end-to-end ranking quality?
21. How would you serve a neural ranker at p99 latency under 200 ms for thousands of candidates per query? [Likely]
    - Which features are precomputed offline, which are computed in real time, and how do you keep them consistent?
22. How do you prevent training-serving skew in features such as "listing price for these dates" or "number of recent bookings"? [Likely] [1]
23. How often would you retrain the ranking model, and how would you detect that it has gone stale? [Likely]
24. What would you log at serving time so you can retrain, debias and debug later? [Likely] [2]
25. Listings with stale calendars cause booking requests that hosts then decline. How would you predict that a listing's availability is unreliable, and use that in retrieval or ranking? [Likely] [1]

### C. Listing embeddings and real-time personalization

26. Design listing embeddings trained from guest click sessions, and use them for similar-listing recommendations and in-session personalization. [Likely] [3][10]
    - How do you define a session and its context window?
    - What negatives would you use? Why might random negatives make embeddings that only learn geography?
27. How would you change the embedding objective so that the booked listing influences every listing in the session? [Likely] [3]
28. How would you build user-type and listing-type embeddings to capture long-term preferences from sparse booking histories? [Likely] [3]
29. How would you create an embedding for a brand-new listing with no sessions? [Likely] [3]
30. How would you evaluate embeddings offline before using them in production? [Likely] [3]
    - What would a "rank of the eventually booked listing among the clicked listings" evaluation look like?
31. How would you turn in-session clicks and skips into real-time ranking features? [Likely] [3]
    - What streaming infrastructure is needed, and what is the freshness requirement?
32. When would you replace skip-gram embeddings with a sequence model or transformer over user actions? What would you expect to gain? [Likely]

### D. Pricing, price prediction and lifetime value

33. Design a system that helps hosts set prices for each future date, taking market demand into account. [Reported] [24][25]
    - What is the objective: booking probability, expected host revenue, or marketplace occupancy?
    - Why can't you train directly on "the optimal price" as a label?
34. Predict the nightly price a listing should command (price prediction), including for a brand-new listing with no booking history. [Reported] [25]
    - Which comparable-listing, location, amenity and seasonality features would you use?
    - How do you avoid learning hosts' mispricing as the target?
35. How would you predict booking probability as a function of price, lead time and listing quality? [Likely] [4]
    - How do you handle price being confounded with demand, because hosts raise prices when they expect demand?
36. How would you estimate price elasticity per listing or per market from observational data? [Likely] [4]
    - When would you need randomized price experiments, and what ethical and host-trust constraints apply?
37. Hosts can accept, modify or ignore your suggestion. How do you evaluate the model offline and online? [Likely] [4]
    - What is your unit of randomization, and how do you handle spillover between treated and control hosts in the same market?
38. How would you forecast market demand for a city on a given date (events, holidays, seasonality)? [Likely] [16]
39. How would you prevent the pricing model from producing extreme or volatile suggestions that destroy host trust? [Likely]
40. How would you design total-price display or fee-related ranking changes, and how would you measure whether guests trust the price more? [Likely]
41. A pricing-model launch raised bookings but cut average host revenue. How do you decide? [Likely] [17]
42. Estimate the lifetime value of a property (listing). First agree with the interviewer on what lifetime value means here. [Reported] [22][27]
    - What horizon, which revenue (gross booking value or Airbnb take), and how do you treat churned or delisted hosts?
    - The interviewer pushes you from regression to bucketized classification. Why might value buckets work better for heavy-tailed, censored LTV, and what do you lose? [Reported] [22]
    - How would the LTV estimate be used, for example in host acquisition, support prioritization or ranking?
43. Design a system that computes a guest's (customer's) lifetime value on the platform. [Reported] [24]
    - How do you handle guests who book once every two years, so that the data is censored and very sparse?
    - How would you validate an LTV model whose labels take years to mature?
44. Which hosts should be prompted to turn on Smart Pricing or adopt a suggested price? Design an uplift (heterogeneous treatment-effect) model and the experiment that validates it. [Likely] [4][17]

### E. Fraud, risk and trust and safety

45. Design a model that scores every booking for payment-fraud risk in real time. [Likely] [18]
    - Chargeback labels arrive weeks later. How do you train and evaluate with delayed labels?
    - How do you handle label censoring, where blocked transactions never reveal whether they were fraud?
46. Rather than block or allow, how would you design graduated friction (ID verification, extra payment checks) driven by risk scores? [Likely] [18]
    - How do you choose thresholds per action, given the cost of false positives (lost good guests) and false negatives?
47. Design an account-takeover detection system for guest and host logins. [Likely] [18]
48. Design a system that predicts and prevents unauthorized parties at listings. [Likely] [19]
    - What features are legitimate and fair to use, and which risk proxy discrimination (for example by age)?
    - How do you measure the system's impact when the outcome (a party happened) is rarely observed?
49. Design a fake or scam listing detector at listing-creation time. [Likely] [19]
    - How do you combine text, image, host-history and graph signals (shared devices, payment instruments)?
50. Design a relisting-detection system: find a property that was removed or banned and then relisted, possibly under a new account. [Reported] [22][28]
    - How does geohashing help you generate candidate pairs efficiently? [Reported] [22]
    - How would the outputs show up in online metrics, and what would you watch for false-positive harm to legitimate hosts? [Reported] [22]
    - How would you separate relisting from legitimate cases such as a new host buying the property, or a co-host managing it?
51. Design a system that detects duplicate listings (the same property listed several times at the same time). [Likely]
52. Design a model that detects attempts to move payment or communication off the platform in guest-host messages. [Likely] [19][20]
53. How would you use graph-based features or graph neural networks to detect fraud rings? [Likely]
54. Fraudsters adapt to your model within weeks. How do you design for adversarial drift? [Likely]
    - How do you keep a clean holdout to measure true fraud rates when the model blocks most of it?
55. How do you A/B test a fraud model when exposing the control group to fraud has real costs? [Likely] [18]
56. How do you build fairness checks into a trust model that affects who can book? [Likely] [19]

### F. Recommendations (Experiences, similar listings, discovery surfaces)

57. Design Airbnb listing recommendation for a guest. [Reported] [25][36]
    - What are the candidate sources (embedding neighbors, popular-in-destination, wishlist-similar), and how do you blend them?
    - How does it differ from search ranking when the guest has no dates or destination yet?
58. Design search ranking for Airbnb Experiences, which has far less data than stays. [Likely] [7]
    - How would you stage the system as data grows (heuristics, then GBDT, then personalization, then online scoring)?
59. Design "similar listings" recommendations on the listing detail page. [Likely] [3]
    - What is the right objective: clicks, bookings, or rescuing guests whose first choice is unavailable?
60. Design personalized marketing or email recommendations (destinations, wishlisted listings with price drops). [Likely]
    - How do you measure incremental lift rather than attribution from people who would have booked anyway?
61. Design recommendations for cross-selling Experiences or Services to guests with an upcoming stay. [Likely]
62. Homes, Experiences and Services now share discovery surfaces. How would you rank a blended feed whose item types have different conversion rates, prices and data volumes? [Likely] [7][37]
    - How do you calibrate scores across verticals so that one vertical does not crowd out the others?
63. How would you recommend wishlist listings or destinations for group trips where several people collaborate? [Likely]
64. Design a system that matches hosts with co-hosts or local service providers. [Likely] [37]
    - What does a successful match mean, and how do you handle two-sided preferences and capacity limits?
65. How do you avoid a popularity feedback loop in which top listings get more exposure, more bookings and more exposure? [Likely] [5]

### G. Host acceptance, response time and other time-to-event predictions

66. Design a model that predicts the probability a host will accept a booking request, and use it in ranking. [Likely] [1]
    - How do you avoid unfairly penalizing a host for a single rejection?
    - How do you use host preferences (trip length, lead time, gaps between bookings) as features?
67. Design a model that predicts host response time to a guest message or booking request. [Likely]
    - How do you handle censoring for messages that never get a reply?
    - Would you use survival or hazard models, or classification at fixed horizons? Why?
68. Predict the probability that a confirmed booking will be cancelled by the host or by the guest. [Likely]
    - How would you use this in ranking, in host coaching, or in guest-protection policy?
69. Predict booking lead time and length of stay for a search, and explain how they would change ranking. [Likely] [1]
70. Design a model that estimates customer-support wait time or resolution time, and routes cases to the right agent queue. [Likely] [20]

### H. Categories, content understanding and listing quality

71. Design the system behind Airbnb Categories (for example "Beachfront", "Treehouses"), which assigns listings to themed browse categories. [Likely] [11]
    - How do you combine ML candidates with human-in-the-loop review, and how do you budget human review?
    - How do you rank listings within a category, and how do you rank the categories themselves for a user?
72. Design an image-understanding pipeline that detects amenities and room types from listing photos. [Likely] [21]
    - How would you get labels cheaply, and how do you measure precision when amenities drive search filters?
73. Choose a listing's cover photo to raise its booking rate. [Reported] [22]
    - How do you handle cold start for a new listing with no engagement on any photo? [Reported] [22]
    - How do you avoid choosing a photo that raises clicks but misleads guests, and so raises cancellations or bad reviews?
    - Would you treat this as a bandit, a learned photo-quality model, or both?
74. Design a model that organizes a listing's photos into a room-by-room tour and orders them. [Likely] [21][37]
75. Design a system for finding high-quality listings, used in ranking and host coaching. [Reported] [22]
    - How do you define quality without simply rewarding listings that are already popular?
    - Which signals would you use: photos, description, completeness, reviews, cancellations, response rate?
76. Design the selection of a "most-loved" or "Guest Favorite" designation from ratings, reviews and reliability. [Likely] [37]
    - How do you keep the badge stable over time and hard to game, and how do you measure its effect on bookings?
77. Design a system that detects inaccurate or misleading listing descriptions by comparing text, photos and reviews. [Likely]
78. How would you use LLMs or multimodal models for listing understanding at Airbnb's scale while controlling cost? [Likely]
79. How would you translate or localize listing content and decide when machine translation is good enough to show? [Likely]

### I. Reviews: quality, ranking and trust

80. Design a system that ranks reviews on a listing page so guests see the most helpful reviews first. [Likely]
    - How do you define "helpful" without explicit helpfulness votes?
81. Design a fake or retaliatory review detector. [Likely] [19]
82. Design a system that extracts aspects (cleanliness, location, accuracy) from review text and feeds them into search or listing quality. [Likely]
83. Design LLM-generated review summaries or highlights for a listing. [Likely]
    - How do you evaluate faithfulness, so the summary claims nothing the reviews do not support?
    - How do you A/B test whether summaries change booking or cancellation behavior?
84. Ratings are highly skewed toward 5 stars. How do you build a reliable quality signal from them? [Likely]
    - How would you apply Bayesian shrinkage for listings with few reviews?
85. How would you predict a guest's post-stay satisfaction before the stay, and how would you use it in ranking? [Likely] [1]

### J. Messaging, customer support and LLM systems

86. Generate customer-support instructions (step-by-step agent guidance) from Airbnb's company policies. [Reported] [22][30]
    - How do you keep generated instructions faithful to the source policy when policies change?
    - How do you evaluate correctness before agents or an automated assistant act on them?
87. Design the retrieval and ranking layer for a generative-AI customer-support tool: given a case, retrieve the right policies, past resolutions and booking context. [Reported] [22]
    - What are the candidate sources, and what is the relevance label for training a ranker?
    - How do you measure whether better retrieval actually reduces resolution time or escalations?
88. Design an ML system that classifies the intent of guest-host messages to route or auto-suggest replies. [Likely] [20]
89. Design an LLM-based customer-support assistant that resolves common issues (cancellations, refunds) and escalates the rest. [Likely] [20]
    - How do you evaluate it offline, monitor hallucinations and policy errors, and measure deflection versus satisfaction online?
    - What guardrails are needed for actions involving money or safety?
90. Design suggested replies for hosts. How do you avoid generating replies that commit the host to things they did not intend? [Likely]
91. How would you build an LLM-as-judge pipeline to produce relevance or quality labels for search and support, and how would you validate it against human raters? [Likely] [20]

### K. Experimentation and measurement inside the ML design round

92. For your ranking model, what is the primary metric of the launch A/B test, and what are the guardrails? [Likely] [8][17]
    - Why might "bookings" be a better primary metric than "booking requests" or "clicks"?
93. Design robust primary and guardrail metrics for a product or model change, so the metrics resist gaming and noise. [Reported, role unspecified] [23][33]
94. What randomization unit would you use for a search ranking test (guest, session, search, listing, market)? Why? [Likely] [8]
95. Explain why a guest-randomized test of a ranking change can be biased in a two-sided marketplace with limited inventory. [Likely] [8]
    - How could market-level, cluster or switchback designs help? What do they cost in statistical power?
96. Design an experiment for a listing badge (for example a "fast Wi-Fi" badge) when guests and listings interact across a network and treatment can spill over. [Reported, role unspecified] [23][35]
97. Design an A/B test with causal inference for a case where clean randomization is hard (hosts opt in, or the change is market-wide). [Reported, role unspecified] [23][34]
    - Which quasi-experimental methods would you use (difference-in-differences, synthetic control, regression discontinuity), and how do you check their assumptions?
98. When would you use interleaving instead of a classic A/B test for ranking? [Likely] [9]
    - How do you attribute a booking to one ranker in an interleaved result list?
    - How do you validate that interleaving and A/B testing agree on direction?
99. Bookings happen days after search, and stays happen weeks later. How do you get a sensitive, timely decision metric? [Likely] [8][17]
    - How would you build and validate a surrogate or predicted long-term value metric?
100. How would you reduce variance in a booking-conversion experiment (CUPED, stratification, trimming)? [Likely] [8]
101. How do you detect and handle sample ratio mismatch, bot traffic or logging bugs in a ranking experiment? [Likely] [8]
102. How do you handle novelty and primacy effects, and seasonality around holidays, in experiment readouts? [Likely]
103. Your model wins overall but loses badly in one market or one guest segment. What do you do? [Likely]
104. How would you set up a long-term holdout to measure the cumulative impact of many ranking launches? [Likely]
105. How would you measure the impact of a pricing or fraud model where the treatment changes host or fraudster behavior over time? [Likely] [4][18]
106. How would you use counterfactual or off-policy evaluation from logged ranking data before running an online test? [Likely] [2]
    - What propensities do you need to log, and how much randomization would you inject?
107. How do you trade off guest-side and host-side metrics in a single ship decision? [Likely] [17]
108. Design the experimentation platform that your ML launches run on: assignment, exposure logging, metric computation and readout. [Reported, role unspecified] [23][32]
    - How would it support marketplace designs (market-level or switchback) and interleaving, not only user-level tests?
109. The ranking team has more candidate models than traffic to test them. How would you prioritize, run sequential or adaptive tests, and control false discoveries across many launches? [Likely] [9]

### L. Monitoring, reliability and iteration (senior depth)

110. What would you monitor once your ranking or fraud model is live (data drift, feature nulls, score distributions, business KPIs)? [Likely]
    - Which alerts would page an on-call engineer, and which go to a dashboard?
111. How would you roll out a new model safely (shadow scoring, canary, ramp, automatic rollback)? [Likely]
112. A feature pipeline broke silently and the model has served on stale features for three days. How do you detect it, mitigate it and prevent it from happening again? [Likely]
113. How do you decide what to build next after launch: more features, a bigger model, or better labels? [Likely] [1][2]
114. How would you reduce serving cost by 50% without measurable loss in bookings? [Likely]
    - What is your approach to distillation, caching and candidate-set pruning?
115. How would you design the system so that other teams (Experiences, Services, Long-term stays) can reuse your ranking stack? [Likely]
116. Which parts of this design would you not use ML for at all, and why? [Likely] [1]
117. What are the main fairness and discrimination risks in Airbnb ranking and trust systems, and how would you audit for them? [Likely] [19]

### M. Variations and curveball prompts

118. Design ranking for long-term stays (28+ nights). What changes from nightly stays? [Likely]
119. Design ranking for Airbnb Services or new verticals with almost no historical data. [Likely] [7][37]
120. Design autocomplete and query understanding for the search box (locations, landmarks, categories). [Likely]
121. Design a model that predicts which markets will face supply shortages for an upcoming event, and recommend actions to recruit hosts. [Likely]
122. Design a host-onboarding recommendation system that tells new hosts what to improve first. [Likely]
123. Design a "price drop" or "rare find" badge. How would you verify that badges are accurate and actually help guests? [Likely]
124. The interviewer scheduled a project deep dive but instead gives you an unfamiliar ML design prompt. How do you scope it in the first five minutes? [Reported] [25]

## References

Sources [22]-[25] and [36] were fetched and read on GitHub on 2026-10-14. Sources [26]-[35] are TrueInterview question pages linked from [22] and [23]; their host could not be reached directly. The remaining sources could not be reached from this environment, because the proxy could not resolve arxiv.org, dl.acm.org, medium.com, glassdoor.com or igotanoffer.com. They are listed from prior knowledge. Entries marked "URL not verified" give a title and an index page rather than a guessed article URL. Confirm every link before relying on it.

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
13. "Airbnb Machine Learning Engineer Interview Questions", Glassdoor (candidate reports), ongoing. Host unreachable from this environment. https://www.glassdoor.com/Interview/Airbnb-Machine-Learning-Engineer-Interview-Questions-EI_IE391850.0,6_KO7,32.htm
14. "Airbnb Interview Process & Questions", IGotAnOffer, 2023-2024. Host unreachable from this environment. https://www.igotanoffer.com/blogs/tech/airbnb-interview-process
15. "Airbnb Interview Discussions", LeetCode Discuss (search "Airbnb"), ongoing. https://leetcode.com/discuss/interview-question?currentPage=1&orderBy=hot&query=airbnb
16. "Learning Market Dynamics for Optimal Pricing", Airbnb Tech Blog, 2018. URL not verified. Search via https://medium.com/airbnb-engineering
17. "How Airbnb Measures Future Value to Standardize Tradeoffs", Airbnb Tech Blog, 2022. URL not verified. Search via https://medium.com/airbnb-engineering
18. "Fighting Financial Fraud with Targeted Friction", Airbnb Tech Blog, 2018. URL not verified. Search via https://medium.com/airbnb-engineering
19. Airbnb Trust & Safety and anti-discrimination posts (party prevention, Project Lighthouse), Airbnb Tech Blog and Airbnb Newsroom, 2020-2024. URL not verified. Search via https://medium.com/airbnb-engineering
20. Airbnb posts on message-intent classification and customer-support ML / LLM systems, Airbnb Tech Blog, 2019-2025. URL not verified. Search via https://medium.com/airbnb-engineering
21. Airbnb posts on amenity detection and room-type classification from listing photos, Airbnb Tech Blog, 2018-2019. URL not verified. Search via https://medium.com/airbnb-engineering
22. "Airbnb Machine Learning Engineer interview process" (TrueInterview-derived summary of 3 candidate reports, mostly 2026 Q1-Q2: ML design prompts, LTV and relisting steering, cover photo, high-quality listings, GenAI support retrieval and ranking). kevin-2023-code/Tech-Interview-Questions, GitHub. Fetched 2026-10-14. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/airbnb/interview-process-machine-learning-engineer.md
23. "Airbnb" question index with report dates (user-embedding semantic search, Apr 2026; A/B testing platform, Apr 2026; rank home results without a text query, Sep 2026; listing LTV, May 2026; experiment-design prompts). kevin-2023-code/Tech-Interview-Questions, GitHub. Fetched 2026-10-14. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/airbnb.md
24. 1point3acres post #40 (English mirror): ML design rounds "design a system to calculate customer lifetime value" and "design a system to help landlords set prices", plus coding LC 1298. FlowSight/coding, GitHub. Fetched 2026-10-14. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/40.md
25. 1point3acres post #79 (English mirror): two ML design rounds (Airbnb listing recommendation; price prediction); the ML experience round turned into an unexpected ML design question; ConnectN coding. FlowSight/coding, GitHub. Fetched 2026-10-14. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/79.md
26. TrueInterview, "Design A Personalized Search Ranking System" (tagged Airbnb, among others). Linked from [22]. https://trueinterview.io/questions/d6c8316c-c7a9-4074-bec9-f19e4252f21d
27. TrueInterview, "Listing Lifetime Value — Estimation" (reported May 2026). Linked from [22][23]. https://trueinterview.io/questions/listing-lifetime-value-ml-design
28. TrueInterview, "Design Airbnb Relisting Detection". Linked from [22]. https://trueinterview.io/questions/design-airbnb-relisting-detection
29. TrueInterview, "Rank Home Search Results Without a Text Query" (reported Sep 2026). Linked from [22][23]. https://trueinterview.io/questions/rank-home-search-results-without-a-text-query
30. TrueInterview, "Generate Customer-Support Instructions from Company Policies". Linked from [22]. https://trueinterview.io/questions/generate-customer-support-instructions-from-company-policies
31. TrueInterview, "Design User Embedding Semantic Search" (reported Apr 2026). Linked from [23]. https://trueinterview.io/questions/design-user-embedding-semantic-search
32. TrueInterview, "Design an A/B Testing Platform" (reported Apr 2026). Linked from [23]. https://trueinterview.io/questions/design-an-a-b-testing-platform-3
33. TrueInterview, "Design robust primary and guardrail metrics". Linked from [23]. https://trueinterview.io/questions/design-robust-primary-and-guardrail-metrics
34. TrueInterview, "Design an A/B test with causal inference". Linked from [23]. https://trueinterview.io/questions/design-an-a-b-test-with-causal-inference
35. TrueInterview, "Design a network-aware Wi-Fi badge experiment". Linked from [23]. https://trueinterview.io/questions/design-a-network-aware-wifi-badge-experiment
36. A. Dirafzoon, "ML System Design" sample questions (lists "Rental recommendation (Airbnb)" among common MLE design prompts). alirezadir/AIMLInterviews, GitHub. Fetched 2026-10-14. https://github.com/alirezadir/AIMLInterviews/blob/main/src/MLSD/ml-system-design.md
37. Airbnb product release announcements: Guest Favorites (2023), Photo Tour (2023), Co-Host Network (2024), and the Services and Experiences relaunch (2025). Airbnb Newsroom. URL not verified. Search via https://news.airbnb.com
38. "Learning to Rank for Maps at Airbnb", Airbnb search team, KDD 2024. URL not verified. Search the title on arXiv or https://medium.com/airbnb-engineering
