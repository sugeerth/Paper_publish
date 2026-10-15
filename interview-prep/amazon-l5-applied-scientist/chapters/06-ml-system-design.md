# 06. ML and Science System Design: Recommendations, Search, Ads, Fraud, Forecasting, Alexa/Rufus LLMs, Pricing and Supply Chain

> Scope: the "Science Application" / "ML System Design (team-specific)" round in the Amazon L5 Applied Scientist (Applied Scientist II) onsite loop, 2025-2026. This chapter is questions only, with no answers.
>
> Research note (please read): no web searches or page fetches worked for this chapter. The run's shared web-search budget was already used up, and direct fetches to Glassdoor, Blind, Medium, Dataford, IGotAnOffer, amazon.jobs and similar sites were refused by this environment's egress proxy. Every [Reported] tag below therefore comes from the sourced findings already gathered for chapters 01-04 of this guide. Those findings were themselves built from search-result extracts of the cited pages, not full reads. Each [Reported] question cites the original page. A question is tagged [Reported] only if one of those chapters attributes it to an Amazon science interview. Everything else is [Likely]. A follow-up session with a fresh search budget could add more first-person design-round reports, especially from Blind and 1point3acres.
>
> Review pass (2026-10-14): an adversarial review tried to re-verify the references and found none reachable. WebFetch could not resolve medium.com, leetcode.com, dataford.io or igotanoffer.com. The proxy returned 403 for amazon.jobs, amazon.science, spectrum.ieee.org, Glassdoor and Blind. The web-search budget was also used up. No reference could be confirmed live or confirmed dead, so none were removed. The three amazon.jobs job IDs [28][29][30] are especially unverified, because postings expire. The review removed duplicate and generic items and added a few [Likely] questions. It added no new [Reported] items, because none could be sourced.

## Format notes for this round

- **Where it sits in the loop.** L5 AS loops are usually 4-6 interviews of 55-60 minutes each. One 2025-2026 Applied Scientist II report gave the loop as "1 Science ML Breadth, 1 Science Depth, 1 Sys Design (Team-specific), 1 coding round and 1 Bar Raiser" [2]. Amazon's official prep page describes the loop as about four 55-minute interviews with members of the science community. It says interviewers are assigned competencies based on the Leadership Principles and typically ask 2-3 behavioral questions each [1]. Other candidates and guides list "ML application" or "science application" as a distinct onsite round [17][18].
- **L5 versus L4.** Blind commenters say L5 loops usually include the design or application round and L4 loops may not [5]. The L5 expectation is end-to-end ownership: framing a vague business problem, choosing data, labels and models, defining offline and online metrics, and launching and monitoring the system, with sound trade-off calls along the way [5][16].
- **Who runs it.** It is often the hiring manager or a senior scientist from the hiring team, and the problem is drawn from that team's domain [2][8]. Examples include Ads trust [9], Alexa speech [10], product search and recommendations [3][12][13], and LLM agents and RAG [3][14].
- **What is assessed.** IGotAnOffer describes the round as being given a high-level ML problem and proposing the data sources, annotation strategy, modeling technique, evaluation metrics, and trade-offs and risks [7]. In candidate reports, interviewers drill into the data pipeline "from collection to deployment" [9], dataset construction and sampling [11], and bias in implicit feedback such as click position bias [12][13]. In 2025-2026, GenAI design prompts appear more often: agentic recommenders, RAG and LLM-agent evaluation [3][6][14].
- **Leadership Principles.** Expect 1-2 LP questions inside this round, often Customer Obsession, Dive Deep, Invent and Simplify, or Think Big. Reports put LPs at roughly 5-15 minutes of each technical interview [1][16].
- **How it differs from SDE system design.** The weight is on the science (problem framing, labels, modeling, metrics, experimentation) more than on distributed-systems boxes. Candidates still report questions about scale, latency and deployment [9][22]. One campus loop had a separate "Scalable ML" round [21].

Tag legend: **[Reported]** means a cited source attributes the question, or a close paraphrase, to an Amazon science interview (AS unless noted). **[Likely]** means a grounded inference from the team's products, job postings or Amazon Science publications. It has not been seen verbatim.

---

## A. Generic framing (the questions every design round opens or pivots with)

1. Here is a high-level ML problem from our team. Propose a solution, covering data sources, annotation strategy, modeling techniques, evaluation metrics, and trade-offs and risks. [Reported] [7]
   - Which parts of the problem actually need ML, and which could be rules or heuristics?
   - What is your v1 that could ship in one quarter, and what is v2?
2. Walk me through the full data pipeline for your design, from collection to deployment. [Reported] [9]
   - Where do labels come from, and how noisy are they?
   - How do you keep training and serving features consistent?
3. Pick an agentic product we could build. Who is the customer, and how do you measure success? (Seattle 2026 report) [Reported] [2]
   - What would the press release headline say (Working Backwards)?
   - Which single metric would you defend to a VP?
4. How would you take this from a prototype to Amazon-scale traffic? [Likely]
   - Where do latency and cost hit first: feature fetch, model inference, or candidate generation?
   - What do you precompute offline, and what must be computed in real time?
5. What offline metric and online metric would you pair, and what do you do when they disagree? [Likely]
   - How do you design the A/B test, including the unit of randomization, duration, power and guardrails?
6. What would make you kill or pivot this project after launch? [Likely]

## B. Recommendations and personalization

7. Design a pipeline to improve personalized product recommendations on the Amazon homepage. [Reported] [6]
   - What is the north-star metric, and what are the guardrails?
   - How do you split the work into candidate generation, ranking and re-ranking?
   - How do you trade relevance against diversity and novelty on one page of widgets?
   - How do you avoid a feedback loop that only recommends already-popular products?
8. Design an agentic recommendation system for Amazon, covering component architecture, model choices, and quality and safety evaluation. [Reported] [6]
   - Which decisions should the LLM agent make, and which belong to a classical ranker?
   - How do you evaluate multi-turn recommendation quality before an A/B test?
   - How do you stop the agent from recommending unsafe, unavailable or irrelevant products?
9. Design a recommender system (reported as the design round in an AS loop). [Reported] [15]
    - Two-tower retrieval, matrix factorization, or sequence models: which, and why?
    - How do you handle cold start for new users and new ASINs?
10. A user goes from viewing to clicking to purchasing a product. Model the probability of each action given a user and a product. [Reported] [11]
    - Would you build three independent models, or model view, click and purchase jointly? [Reported] [11]
    - How would you build the dataset, choose features, and down-sample (for example with reservoir sampling)? [Reported] [11]
    - How do you handle the selection bias that purchases are only observed after clicks?
11. Design "Frequently bought together" / "Customers who bought this also bought". [Likely]
    - How do you separate substitutes from complements?
    - How do you deal with spurious co-purchases from promotions or bundles?
12. Design recommendations for Prime Video or Amazon Music, where consumption is sequential and the feedback is implicit. [Likely]
    - How do you define a positive label for a show that was watched for only 3 minutes?
13. How would you personalize email or push notifications without fatiguing customers? [Likely]
    - How would you frame this as uplift modeling rather than response modeling?
14. Design a size and fit recommendation system for apparel and shoes, so customers order the right size the first time. [Likely]
    - How do you learn from return reasons such as "too small" or "too large" as weak labels?
    - How do you handle brands whose sizing differs from the size printed on the label?
15. A product is out of stock. Design a system that recommends substitutes on the detail page and in the cart. [Likely]
    - How do you tell a true substitute from a merely similar item, and how do you evaluate substitutability offline?
16. How would you detect and fix popularity bias or exposure unfairness toward small sellers in recommendations? [Likely]

## C. Product search and ranking

17. You have a large collection of user click data for many search queries and product documents. Is click data alone sufficient? How would you train a retrieval or IR model with the data at hand? [Reported] [12]
    - How do you correct for position bias in clicks? [Likely] (see [23])
    - What other biases (presentation, trust, query intent) contaminate click labels?
18. Walk through how you would design, train, validate and deploy a ranking model for product search. [Reported] [13]
    - How would you estimate true relevance from clickstream logs with position bias? [Reported] [13]
    - Pointwise, pairwise or listwise learning-to-rank: which would you choose, and why?
    - How do you define "relevant" for a query like "running shoes" versus "nike pegasus 40 size 10"? (search-relevance application questions are reported in [3])
19. Compare bi-encoders and cross-encoders for retrieval and re-ranking. Where would each sit in a product-search stack? [Reported] [3]
    - What is your latency budget per stage, and how do you distill the cross-encoder?
20. Design query understanding for Amazon search (intent, product type and attribute extraction). [Likely] (see [27])
    - How would you launch it in a new marketplace and language with little labeled data?
21. Design a system to detect and reduce zero-result or low-quality-result searches. [Likely]
    - When should you relax the query, and when should you show a "did you mean"?
22. How would you combine relevance with business objectives (conversion, margin, sponsored placement) in the final ranking? [Likely]
    - How do you stop business boosts from eroding relevance over time?
23. Design an evaluation framework for search relevance using human annotation and LLM judges. [Likely]
    - How do you measure annotator agreement and calibrate an LLM judge against humans?
24. How would you build semantic or vector search over a catalog of hundreds of millions of products? [Likely]
    - How do you handle index freshness for new listings and price or stock changes?

## D. Advertising (Sponsored Products, DSP, Ads Trust)

25. Design a system that recommends trustworthy ads to Amazon users while maintaining brand safety and compliance. [Reported] [9]
    - Walk through the full data pipeline from collection to deployment. [Reported] [9]
    - How do you trade ad revenue against trust and safety errors?
    - How do you get labels for rare policy violations?
26. Design click-through-rate prediction for Sponsored Products ads. [Likely] (see [20])
    - Why does calibration matter in a second-price or first-price auction?
    - How do you handle delayed conversions in conversion-rate modeling?
27. Revenue is flat, cost-per-click is up and conversions are down after a budget-pacing change. How would you diagnose it? [Likely]
28. Design an ad-relevance or ad-moderation classifier for images and text in sponsored creatives. [Likely] (see [9])
    - How do you handle adversarial advertisers who iterate on rejected creatives?
29. How would you measure the incremental impact (lift) of an ad campaign for an advertiser? [Likely]
    - Ghost ads, geo experiments or observational methods: what are the trade-offs?
30. Design a bidding or budget-pacing algorithm for advertisers on Amazon DSP. [Likely]
    - How would you frame it as constrained optimization, or as a bandit problem?
31. Design keyword or audience targeting recommendations for advertisers. [Likely]
32. Design the ad-load and ad-placement policy for streaming video ads (Prime Video). How do you trade ad revenue against viewer engagement and churn? [Likely]
    - Which long-term metric would you use as a guardrail, and how do you measure it within an experiment's duration?

## E. Fraud, abuse, trust and payments

33. Design a fraud detection system for Amazon payments transactions. [Likely] (see [28])
    - How do you deal with extreme class imbalance and delayed chargeback labels?
    - How do you pick the operating threshold given the costs of false positives and false negatives?
    - How do you handle adversaries adapting to your model?
34. How would you detect fake reviews or abusive sellers with heavily imbalanced and adversarial labels? [Likely]
    - Would you use graph-based signals across accounts, devices and payment instruments?
    - How do you evaluate when your labels come from your own past enforcement?
35. Design a system to detect counterfeit products or listings. [Likely]
    - How do you combine image, text, seller history and brand-registry signals?
36. Design account-takeover detection at login time under a strict latency budget. [Likely]
    - What do you do with the "uncertain" band: friction, step-up auth, or human review?
37. Predict, at checkout time, the probability that an order will be returned. How would this be used, and what are the risks? [Likely]
    - How do you keep the model from penalizing protected groups or legitimate high-return categories?
38. Design returns-abuse detection. How do you avoid punishing legitimate customers? [Likely]
39. How would you prioritize a human-investigator queue when you can review only 1% of flagged cases? [Likely]
    - How do the investigators' decisions feed back into training without biasing it?
40. Your fraud model's precision drops sharply over two weeks. Walk me through the investigation. [Likely]

## F. Computer vision and catalog quality

41. Use computer vision to detect defective products in fulfillment-center warehouses. [Reported] [8]
    - How do you collect and label defect images when defects are rare?
    - Edge or cloud inference, and what is the throughput requirement per station?
42. You have 10k product listings across 100 categories, each with a title, description and photo. How would you categorize them? [Reported] [12]
    - How do you fuse text and image signals?
    - How do you handle a long tail of categories with very few examples?
    - How do you evaluate when the taxonomy itself is noisy?
43. How would you detect and fix wrong attributes (size, color, brand) on listings at catalog scale? [Likely]
    - How would you use an LLM for attribute extraction, and how do you verify its outputs?
44. Design duplicate-listing detection (the same product listed under multiple ASINs). [Likely]
    - How do you scale pairwise matching to hundreds of millions of items?

## G. Forecasting, supply chain and inventory (SCOT)

45. How would you forecast demand for millions of products, including new ones with no history? [Likely] (see [24][25][29])
    - Why use probabilistic or quantile forecasts rather than point forecasts for buying decisions?
    - Would you use a global model or per-item models, and why?
46. What is the newsvendor problem, and how would you use it to set inventory levels? (Amazon supply chain interviews) [Reported] [26]
    - How does a quantile forecast map to the newsvendor critical ratio?
47. How would you model an optimization problem from your work, including objective, decision variables and constraints? [Reported] [16]
    - When would you use a heuristic instead of an exact solver?
48. Design an inventory placement system that decides which fulfillment centers should stock each product. [Likely] (see [25])
    - How do forecast errors propagate into placement and transportation costs?
49. How would you forecast demand during Prime Day or other promotions, when history is sparse and distorted? [Likely]
    - How do you correct for censored demand when items were out of stock?
50. How do you evaluate a forecasting system against a business outcome (in-stock rate, holding cost) rather than MAPE alone? [Likely]
    - How would you run an experiment on a buying policy when products share warehouses and trucks?
51. Design a system to predict delivery time (the promise date shown on the detail page). [Likely] (see [19])
    - What loss would you use if a late delivery costs more than an early one?
    - How do you calibrate promises across regions and carriers?
52. How would you improve address geocoding or delivery-location accuracy for last-mile delivery? [Likely] (see [19])
53. Design a capacity or labor-planning forecast for fulfillment centers. [Likely]
54. For grocery delivery (Amazon Fresh or Whole Foods), design the system that picks a replacement when an ordered item is unavailable at picking time. [Likely]
    - How do you learn each customer's tolerance for substitutions, and what is the success metric?

## H. Pricing, causal inference and experimentation

55. Design a system that recommends competitive prices for millions of products. [Likely] (see [29])
    - How do you estimate price elasticity when prices are set by your own algorithm (endogeneity)?
    - What guardrails stop the system from causing price spirals or customer-trust problems?
56. How would you design an experiment to measure the effect of a pricing algorithm when prices interact across products? [Likely]
    - How do you handle interference between treated and control products, for example with switchback or cluster designs?
57. How would you estimate the causal effect of a promotion without a clean experiment? [Likely]
    - When would you use double ML, instrumental variables, or difference-in-differences?
58. How would you design an A/B test for a model launch? [Reported] [4]
    - What do you do if the test is underpowered or the metric is heavy-tailed?
    - How would you use variance reduction such as CUPED?
59. A new ranking model wins offline but is neutral online. What could explain it, and what do you do next? [Likely]
60. Design the system that chooses which seller's offer wins the featured offer (Buy Box) on a detail page. [Likely]
    - Which signals (price, delivery speed, seller performance) do you use, and how do you avoid being gamed by sellers?
    - How do you evaluate the effect on customers versus on seller fairness?
61. How would you decide whether to discount a product, and by how much, for a deal event? [Likely]
    - How do you account for cannibalization and pull-forward demand?

## I. Alexa, Rufus and LLM-powered products

62. How would you improve Alexa speech recognition for non-native speakers and users with strong regional accents? [Reported] [10]
    - Where would you get diverse accent training data, and how would you handle phonetic variation? [Reported] [10]
    - How would you measure improvement per accent group without regressing the majority?
63. Design an enterprise RAG (retrieval-augmented generation) service. [Reported] [14]
    - How do you chunk, embed and index heterogeneous documents?
    - How do you enforce document-level access control in retrieval?
    - How do you evaluate retrieval quality separately from generation quality?
64. Evaluate an LLM agent with metrics beyond accuracy, such as tool-use correctness, hallucination and calibration. [Reported] [3]
    - How do you build an offline evaluation set for multi-step agent tasks?
    - When would you trust an LLM-as-judge?
65. Design a GitHub Copilot-style coding agent (reported in a Senior AS loop, Boston). [Reported] [3]
    - What context does the agent retrieve, and how do you rank it?
    - How do you measure whether it helps developers (acceptance rate versus downstream bugs)?
66. On an LLM team, discuss LLM decoding and training choices in the context of an Amazon use case. [Reported] [22]
    - Greedy, beam or sampling decoding for a customer-facing assistant: which would you use, and why?
67. Design Rufus: answer "What do I need for cold weather golf?" using the catalog, reviews and Q&A. [Likely] (see [27][31][32])
    - How do you ground answers in product facts and avoid hallucinated specs?
    - How do you evaluate it offline before an A/B test, and what is the online success metric?
    - How do you mix organic and sponsored products in a generative answer?
68. How would you summarize thousands of customer reviews into a short "Customers say" blurb? [Likely]
    - How do you measure faithfulness and coverage of minority opinions?
    - How do you stop manipulated reviews from steering the summary?
69. Design an Alexa skill or intent router that maps an utterance to the right skill or API (natural-language understanding). [Likely]
    - How do you handle new skills with no training data?
    - What is your latency budget end to end, from wake word to response?
70. How would you serve an LLM-based assistant to millions of users at acceptable cost and latency? [Likely]
    - Would you use distillation, quantization, caching, speculative decoding, or routing between small and large models?
71. How would you fine-tune and align an assistant for a new language or marketplace (SFT, DPO, RLHF or RLAIF)? [Likely] (see [27])
    - How do you get preference data, and how do you detect reward hacking?
72. How would you make an on-device model 10x smaller with minimal quality loss (Echo or Kindle devices)? [Likely]
73. How would you detect and mitigate unsafe, biased or policy-violating responses from a shopping assistant? [Likely]
    - Which red-teaming process and which guardrail models would you use?
74. Design proactive suggestions for Alexa ("hunches" or follow-up prompts) that are helpful without being intrusive. [Likely]
    - How do you set the interruption threshold, and how do you measure negative feedback that customers never state explicitly?
75. Design a conversational customer-service agent that can issue refunds or replacements. [Likely]
    - Which actions require human-in-the-loop, and how do you decide?
    - How do you evaluate resolution rate without incentivizing over-refunding?

## J. Scaling, MLOps and lifecycle (probes across any design)

76. How do you monitor this model in production, and how do you detect data or concept drift? [Likely]
    - What triggers a retrain, and how do you validate it before rollout?
77. How would you run inference at scale, in real time versus batch, and on what infrastructure (for example, SageMaker endpoints)? [Likely]
78. How do you handle feature-store consistency and training/serving skew? [Likely]
79. How would you roll back safely if the new model misbehaves on a subset of traffic? [Likely]
80. Your labels arrive weeks late (returns, chargebacks, long-term retention). How do you train and evaluate under label delay? [Likely]
81. How would you reduce the cost of this system by 50% with minimal quality loss? [Likely] (Frugality)
82. Which parts of your design would you build, and which would you reuse from existing Amazon services or other teams? [Likely]
83. How would you handle personally identifiable information and privacy constraints in the training data? [Likely]

## K. L5-bar probes and LP crossovers inside the design round

84. Of everything you proposed, what is the single riskiest assumption, and how would you de-risk it in two weeks? [Likely]
85. A product manager wants to launch in a month, and your offline evaluation is incomplete. What do you do? [Likely] (Bias for Action versus Insist on the Highest Standards)
86. Your design improves conversion but increases returns. How do you decide whether to ship it? [Likely]
87. Which metric would you refuse to optimize directly, and why? [Likely]
88. How would you explain the trade-offs in your design to a non-scientist stakeholder? [Likely] (Earn Trust)
89. If you had 10x the data, or one-tenth of it, how would the design change? [Likely]
90. How would you hand this system over to an engineering team for long-term ownership? [Likely]

## L. Team-specific "warm-up" prompts to rehearse (inferred from postings and Amazon Science)

91. (SCOT Buying Science) Design a system that decides how much of each product to buy from which vendor and when. [Likely] (see [29])
92. (Payments) Design a model to predict payment-method failure and pick the best fallback for the customer. [Likely] (see [28])
93. (Last mile) Design a model that predicts whether a package will be delivered successfully on the first attempt. [Likely] (see [19])
94. (Alexa International) Design a multilingual natural-language-understanding model for a marketplace with code-switching (for example Hinglish). [Likely] (see [10][17])
95. (Ads Trust) Design a policy-violation detection system for ads with an annotation budget of 5,000 labels per month. [Likely] (see [9])

---

## References

1. Applied Scientist Interview Prep, amazon.jobs (Amazon, official), accessed 2026. https://amazon.jobs/content/en/how-we-hire/applied-scientist-interview-prep
2. Amazon Applied Scientist II Interview Experience & Questions, Glassdoor, 2025-2026 (includes the "1 Sys Design (Team-specific)" loop report and the agentic-product prompt). https://www.glassdoor.com/Interview/Amazon-Applied-Scientist-II-Interview-Questions-EI_IE6036.0,6_KO7,27.htm
3. Amazon Applied Scientist Interview Experience & Questions, Glassdoor, 2025-2026 (Copilot-style agent, LLM agent evaluation, search relevance, bi-encoder vs cross-encoder). https://www.glassdoor.com/Interview/Amazon-Applied-Scientist-Interview-Questions-EI_IE6036.0,6_KO7,24.htm
4. Amazon applied scientist interview-onsite, Blind, circa 2024. https://www.teamblind.com/post/amazon-applied-scientist-interview-onsite-t01hpnc4
5. Amazon applied scientist interview loops, Blind. https://www.teamblind.com/post/amazon-applied-scientist-interview-loops-lucdk0uq
6. Amazon Applied Scientist Interview Questions & Guide 2026, Dataford (prep vendor), 2026. https://dataford.io/interview-guides/amazon/applied-scientist
7. Amazon Applied Scientist Interview (process, questions, prep), IGotAnOffer (prep vendor), 2025-2026. https://igotanoffer.com/en/advice/amazon-applied-scientist-interview
8. The 2026 Amazon Applied Scientist Interview Guide, Interview Kickstart (prep vendor), 2026. https://interviewkickstart.com/interview-questions/companies/amazon/applied-scientist
9. "My interview experience for the Applied Scientist II role at Amazon" (Advertising Trust), Chaitanya Mundle, Medium, circa 2024. https://medium.com/@mundle1997.cm/my-interview-experience-for-the-applied-scientist-ii-role-at-amazon-738aba21f611
10. "Science application: Business problems for Amazon Applied Scientist (Alexa International Tech)", Medium, 2025. https://medium.com/@pragyashukla2580/science-application-business-problems-for-amazon-applied-scientist-alexa-international-tech-9353a52ca556
11. "Interview Experience for Applied Scientist at Amazon", Akanksha Paul, Medium, circa 2020. https://akanksha-paul.medium.com/interview-experience-for-applied-scientist-at-amazon-5ff5003375dd
12. "Amazon Applied Scientist | L6 Luxembourg | Passed, freeze | 2022", LeetCode Discuss, 2022. https://leetcode.com/discuss/interview-experience/4421594/
13. "What are some Amazon Research Scientist interview questions?", Quora. https://www.quora.com/What-are-some-Amazon-Research-Scientist-interview-questions
14. "Amazon Onsite Interview Experience for Fulltime Machine Learning Role", 1point3acres. https://www.1point3acres.com/interview/thread/1157086
15. "Interview Experience Amazon Applied Scientist" (recommender design round), Blind. https://www.teamblind.com/post/Interview-Experience-Amazon-Applied-Scientist-hWgaGsmO
16. Amazon Applied Scientist Interview Debrief: Five-Round Loop + LP, OA VO Service (vendor-published candidate debrief; promotional). https://oavoservice.com/en/articles/amazon-applied-scientist-vo-loop-lp-ml-coding
17. "Amazon Applied Scientist Interview Guide: Alexa International Tech", Medium (Interview Preparation), Apr 2025. https://medium.com/interview-preparation/amazon-applied-scientist-interview-guide-alexa-international-tech-022d3a056feb
18. "Applied Scientist at Amazon: Interview Insights" (L5, lists an ML application round), Rahul Kumar, Medium, Nov 2024. https://medium.com/@rahulnkumar/applied-scientist-at-amazon-interview-insights-25265e1405f5
19. "My Amazon Applied Science Interview Journey" (Last Mile Geospatial), Ajinkya Pahinkar, Medium, Jan 2024. https://medium.com/@ajinkya.pahinkar98/my-amazon-applied-science-interview-journey-f3eb1ed748a6
20. "Applied scientist interview at Amazon Ads", Blind. https://www.teamblind.com/post/applied-scientist-interview-at-amazon-ads-zoerec5t
21. "Amazon Interview Experience for Applied Scientist" (campus, Oct 2023; lists a "Scalable ML" round), GeeksforGeeks. https://www.geeksforgeeks.org/interview-experiences/amazon-interview-experience-for-applied-scientist/
22. Amazon interview review RVW96068026 (Applied Scientist II, LLM team, Santa Clara), Glassdoor, 2025. https://www.glassdoor.com.hk/Interview/Amazon-Interview-E6036-RVW96068026.htm
23. "Intent based relevance estimation from click logs", Amazon Science. https://www.amazon.science/publications/intent-based-relevance-estimation-from-click-logs
24. "Predicting what Amazon's customers will need tomorrow, next week, and beyond", Amazon Science. https://www.amazon.science/working-at-amazon/predicting-what-amazons-customers-will-need-tomorrow-next-week-and-beyond
25. "reMARS revisited: Amazon's supply chain optimization", Amazon Science. https://www.amazon.science/latest-news/remars-revisited-amazons-supply-chain-optimization
26. Amazon Supply Chain Interview Experience & Questions (newsvendor problem), Glassdoor. https://www.glassdoor.com/Interview/Amazon-Supply-Chain-Interview-Questions-EI_IE6036.0,6_KO7,19.htm
27. Applied Scientist, Rufus Experiences Science (Job ID 2855826), amazon.jobs. https://amazon.jobs/en/jobs/2855826/applied-scientist-rufus-experiences-science
28. Applied Scientist II, Amazon Payment Products (L5) (Job ID 3168758), amazon.jobs. https://www.amazon.jobs/en/jobs/3168758/applied-scientist-ii-amazon-payment-products-l5
29. Applied Scientist II, SCOT Optimal Sourcing Systems, Buying Science Team (Job ID 10468056), amazon.jobs. https://www.amazon.jobs/en/jobs/10468056/applied-scientist-ii-scot-optimal-sourcing-systems-buying-science-team
30. 2027 Applied Science Internship: Recommender Systems / Information Retrieval (Job ID 10564598), amazon.jobs. https://www.amazon.jobs/en/jobs/10564598/2027-applied-science-internship-recommender-systems-information-retrieval-machine-learning-united-states-phd-student-science-recruiting
31. "What do I need for cold weather golf?" (Amazon Rufus), IEEE Spectrum. https://spectrum.ieee.org/amazon-rufus
32. "Amazon Rufus AI assistant personalized shopping features", About Amazon. https://aboutamazon.com/news/retail/amazon-rufus-ai-assistant-personalized-shopping-features
