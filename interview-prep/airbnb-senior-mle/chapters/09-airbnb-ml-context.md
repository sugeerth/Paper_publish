# 09. Airbnb ML Context: Papers, Platforms and Recent AI Products as Interview Questions

This chapter turns Airbnb's public ML record into questions you could be asked. That record covers the KDD/CIKM papers, the Bighead / Zipline / Chronon platform line, Minerva and ERF, interleaving, and the 2024-2026 LLM customer-support and AI-search launches. The chapter has no round of its own. Its questions come up inside the ML system design rounds, the experience deep dive (past-project round) and the experimentation discussion. They work best as design prompts ("design X at Airbnb"), so the system-design framing is kept throughout.

> **Research note (read first).** This chapter went through an adversarial review on 2026-10-14. The shared web-search budget was used up, so no new search queries could run. The proxy could not resolve prachub.com, teamblind.com, 1point3acres.com, interviewquery.com, trueinterview.io, arxiv.org, medium.com, airbnb.tech, infoq.com, cxtoday.com, zenml.io, altexsoft.com or syncedreview.com. Twelve of the previously cited URLs were tried and all failed to resolve. GitHub was the only reachable host.
>
> - **Fetched and read on 2026-10-14:** [1]-[7]. These are the TrueInterview-derived Airbnb MLE guide (3 candidate reports, mostly 2026 Q1-Q2), the dated Airbnb question index (entries to Sep 2026), the general Airbnb process page, two 1point3acres MLE reports mirrored on GitHub, the Chronon repository, and the applied-ml index.
> - **URL confirmed:** [8]-[27]. Title, URL and year match the applied-ml index [7], but the page itself was not opened.
> - **Unverified:** [28]-[54]. These are carried over from the previous version of this chapter and could not be opened. Check them before you rely on them.
>
> **Tags.** **[Reported]** means a fetched source [1]-[5] says a candidate was asked this at Airbnb, or something very close to it. **[Likely]** means an inference from Airbnb's published work, the role and the senior bar. A citation on a [Likely] question only gives context.
>
> **Retagging in this review.** Six questions were moved from [Reported] to [Likely] because their only sources (PracHub, Blind, CleverPrep) could not be opened: FM vs. DNN, offline-NDCG-up-but-A/B-flat, precision/recall/latency/cost, host fairness, the "family friendly" classifier and the ML-platform code review. The topics are plausible but unconfirmed. In the October 2026 deepening pass, PracHub search snippets supported three of them (FM vs. DNN, offline-up-but-A/B-flat, precision/recall/latency/cost), and those three are now [Reported] again on snippet evidence [52].

## Format notes (how this material surfaces in the loop)

- **Where it shows up:** Airbnb MLE loops have one or two ML system design rounds, sometimes more. The prompts are open-ended marketplace problems with no code: search ranking, a property's lifetime value, recommendations, relisting detection, cover photos [1][3]. The loop is decided mainly in these rounds, and a borderline design signal can earn an extra design round [1].
- **Experience round can turn into design:** One MLE candidate expected a past-project deep dive in the "ML experience" round, but the interviewer asked an unplanned ML design question instead [5]. Prepare to whiteboard an Airbnb problem in any ML round.
- **Senior bar:** At Senior (G9) and above, design carries more weight. In the reports, rejection feedback pointed to design depth or overall match, never to coding [1]. One ML rejection cited "jumping straight to model ideas"; the guide's advice is "Problem and label before model" [3].
- **Recent senior loops:** These loops added ranking and retrieval prompts around generative-AI customer-support tools and a document-redaction exercise [1]. In design rounds, interviewers push on candidate generation, latency, freshness, cold start for new listings and monitoring [1].
- **Interviewer steering:** Treat a redirection as a hint and restate the narrower scope before going on [1].
- **Unverified prep advice:** This advice comes from search-index excerpts of Blind and PracHub [52][53] and could not be re-checked: "be very familiar with Airbnb business and product," because questions often reflect a real problem the team is working on.
- **How to use this chapter:** You will rarely be asked to recite a paper. You are more likely to get the paper's problem as an open design prompt, then a probe on whether you know how Airbnb approached it.

---

## 1. Search ranking at Airbnb (KDD 2019 / KDD 2020 deep learning line)

### 1.1 Search-ranking prompts

1. [Reported] Rank home search results when the guest gives no text query, only location, dates and guest count. Reported Sep 2026. [1][2][3]
   - What is the label: click, wishlist, booking request, or an uncancelled booking that the host accepted?
   - How do you handle the fact that most of a guest's searches never end in a booking?
   - What retrieval stage feeds the ranker, and how large is the candidate set?
   - Walk through candidate generation, offline evaluation, online evaluation and deployment. [1]
2. [Reported] Design a personalized search ranking system for Airbnb. [1][2]
   - [Likely] How does ranking Experiences differ from ranking Homes (smaller inventory, time-slot availability, cold start)? [11]
   - How would you surface new or high-quality Experiences that have little engagement history?
3. [Reported] In your ranking design, the interviewer pushes on latency, feature freshness, cold start for new listings and monitoring. Which of these sets your architecture, and what would you give up first? [1]
4. [Likely] In Airbnb search ranking, how do you balance relevance, personalization and host fairness?
   - How would you measure fairness or exposure for hosts, and which guardrail would you set on it?
5. [Reported] Compare a factorization machine with an embedding + DNN ranker for this problem. Which would you ship, and why? Retagged in the October 2026 deepening pass: the PracHub search snippet (Senior MLE, Apr 2026) lists FM vs. embedding + DNN among the discussed topics. [52]
   - Explain the tradeoffs in interaction modeling, serving latency and debuggability.
6. [Reported] Your new ranker improves offline NDCG, but the online A/B test is flat or negative. What do you do? Retagged in the October 2026 deepening pass: the PracHub search snippet (Senior MLE, Apr 2026) says the interviewer asked what to do when offline and online metrics disagree. [1][52]
   - Possible causes include position bias in training logs, label leakage, distribution shift, a metric mismatch with bookings, and set-level effects.
7. [Reported] How do you trade off precision, recall, latency and infrastructure cost in a production ranking system? Retagged in the October 2026 deepening pass: the PracHub search snippet (Senior MLE, Apr 2026) quotes this trade-off as a past-project probe. [52]

### 1.2 Moving from GBDT to neural ranking ("Applying Deep Learning to Airbnb Search")

8. [Likely] Airbnb's GBDT search ranker plateaued, and the team moved to neural networks [8]. When is that a good decision, and how would you de-risk it?
   - Why start with a single-hidden-layer network to validate the pipeline before trying deeper models?
   - Which feature-engineering habits from GBDTs (raw counts, unnormalized prices) hurt neural nets, and how do you fix them?
9. [Likely] Airbnb reported that listing-ID embeddings overfit badly in search ranking [8]. Why would listing IDs overfit in a marketplace where each listing books only a few times a year?
   - What features would you use in place of high-cardinality IDs?
10. [Likely] A listing can be booked by only one guest per night. How would you build a "booking probability" label from search logs under that constraint?
    - How do you deal with listings that were unavailable, or that the host rejected?
11. [Likely] How do you check whether a neural ranker learned the expected relationship with price, for example a preference for cheaper listings when everything else is equal?
    - What would you do if partial-dependence plots showed the model preferring more expensive listings?
12. [Likely] Airbnb search is a two-sided marketplace. How would you include host preferences (acceptance, rejection, Instant Book) in the ranking objective?

### 1.3 Position bias, cold start and architecture ("Improving Deep Learning for Airbnb Search")

13. [Likely] Explain how you would correct for position bias in training data from search logs. [9]
    - How does a shallow position "tower" whose logit is added to the relevance tower work, and how do you serve without it?
    - What assumption about factorizing click probability does this rely on?
    - When would you use dropout on the position feature instead?
14. [Likely] New Airbnb listings have no engagement history. How do you rank them fairly without flooding results with untested inventory? [9]
    - How would you predict engagement features for a new listing from its attributes and nearby listings?
    - How do you A/B test a cold-start change when its effects show up on the supply side?
15. [Likely] Airbnb found that a deeper or more complex network did not automatically win [9]. How do you decide when to stop increasing model capacity?
16. [Likely] How would you design a two-tower "query/guest" vs "listing" architecture for ranking, and what is lost compared with a full cross-feature network?

### 1.4 Diversity and set-level ranking

17. [Likely] Ranking each listing by its own booking probability ignores how similar the listings are to each other. How would you diversify Airbnb search results? [12][13]
    - How do you define similarity: price, location, room type, or a learned embedding?
    - Why might diversity optimization break interleaving-based evaluation? [18][37]
18. [Likely] How does a set-level (listwise or contextual) ranker differ from a pointwise one, and how would you serve it within latency budgets?

### 1.5 Multi-task and multi-objective ranking (Journey Ranker, KDD 2023; distillation, KDD 2024)

19. [Likely] Guests research for days or weeks before booking, and hosts can reject or cancel. How would you design a ranker that uses intermediate "milestones" such as clicks, wishlists, contact host and booking request? [28]
    - How do you treat negative milestones such as host rejection or guest cancellation?
    - How would you use guest state (early vs late in the journey) and query context as inputs?
    - How would you make the architecture modular so that other surfaces (email, Experiences, recommendations) can reuse it?
20. [Likely] Search ranking must maximize bookings while keeping cancellations, low reviews and customer-support contacts down. How would you combine these objectives? [29]
    - What are the drawbacks of a hand-tuned weighted sum of task heads?
    - How would distilling several single-objective "teacher" models into one "student" ranker work?
    - How do you add a non-differentiable business objective?
    - How do you show that the secondary-objective constraints still hold after launch?
21. [Likely] How would you choose the primary metric for a multi-task ranker: uncancelled bookings, booking value, or something else?

### 1.6 Maps, home page and split stays

22. [Likely] On a map there is no ranked list, and attention does not decay by rank position. How would you evaluate and train ranking for map pins? [30]
    - Why is NDCG a poor fit for maps, and what attention model (uniform, tiered pins, spatial decay) would you use?
    - How would you decide which listings get full price pins and which get mini-pins?
23. [Likely] How should list ranking and map results stay consistent when a guest pans or zooms?
24. [Reported] Design the Airbnb home page: search, availability and ranking working together. Reported Mar 2026. [2]
    - Where does availability filtering happen relative to retrieval and ranking, and how stale can availability be?
    - How do you rank home-page modules or carousels against each other, not only listings within a module?
25. [Likely] When no single listing is free for all of a guest's dates, Airbnb can suggest a split stay across two listings. How would you rank listing pairs for that? A split-stay combination problem appears in the coding pool [2].
    - How do you score a pair (distance between the listings, combined price, quality of each) and keep the candidate space tractable?

---

## 2. Retrieval, location and semantic search

26. [Likely] Design the first embedding-based retrieval (EBR) stage for Airbnb search, which narrows eligible homes before the expensive ranker runs. [31]
    - How do you build training pairs from search sessions, and which negatives do you use (random, same-market, or hard negatives from booked-vs-viewed)?
    - Why compute the listing tower offline daily and the query tower online?
    - How would you choose an ANN index (IVF vs HNSW) given that availability and price filters change constantly?
    - How do you combine EBR with hard filters such as dates, guest count and price?
27. [Likely] Airbnb reported that EBR increased bookings and also cut compute [31]. How could a retrieval change save compute and improve conversion at the same time?
28. [Reported] Design semantic search over user embeddings. Reported Apr 2026. [2]
    - What does a user embedding encode (stays, searches, wishlists, reviews), and how is it refreshed?
    - What is the query: a free-text description, another user, or a listing? How does that change the index design?
    - How do you evaluate semantic relevance when there is no clean labeled set?
29. [Likely] For a query like "Lake Tahoe," which geographic areas should search retrieve from? How would you move from heuristic bounding boxes to a learned approach? [32]
    - How do you evaluate location recall, meaning whether the area included the listing the guest eventually booked?
    - Why might you use reinforcement learning or bandit exploration for location retrieval?
30. [Likely] Suppose you represent the world as discrete location cells for retrieval. How would you do it, and what tradeoff do you face between cell size, precision and recall? [32]
31. [Likely] How would you support flexible-date or "anywhere" searches, where there is no fixed location or date?

---

## 3. Embeddings, content understanding and recommendations

32. [Likely] Walk through how you would learn listing embeddings from click sessions with a skip-gram style model. [10][33]
    - Why use the booked listing as a global context in every session window?
    - Why sample negatives from the same market as well as at random?
    - How would you check embedding quality offline (same-market neighbors, similar price, similar listing type)?
33. [Likely] Bookings are too sparse per user and per listing to learn ID embeddings directly. How did Airbnb's user-type and listing-type embeddings address this, and how would you build them? [10]
    - How do you put host rejections into the embedding space?
34. [Reported] Design Airbnb listing recommendation. [5]
    - [Likely] Make it concrete as the "similar listings" carousel on a listing page [10][33]. What offline metric and what online A/B metric would you use?
    - How would you handle a new listing with no clicks?
35. [Likely] How would you use short-term (in-session) and long-term (past bookings) interest embeddings as real-time ranking features?
    - What latency and freshness requirements apply, and where do the session features come from?
36. [Likely] Airbnb trained a two-tower model on guest engagement to produce listing embeddings for host-side and similar-listing surfaces [33]. How do you correct sampling bias in in-batch negatives?
37. [Likely] How would you build one multimodal embedding per listing from many photos and text fields, without just averaging them? [34]
    - How would you use LLMs or CLIP-style models to align text and images?
    - How would you compress the embedding (PCA, quantization) to fit a serving memory budget?
    - How does this embedding help with zero-shot free-text search and cold-start listings?
38. [Likely] Design a room-type classifier for listing photos (bedroom, kitchen, bathroom) and its labeling pipeline. Airbnb's Photo Tour depends on this kind of classification. [20]
    - How would multi-task learning, ensembling or distillation improve accuracy within latency limits?
39. [Reported] Choose a listing's cover photo so that it raises bookings, including for new listings with no engagement data. [1]
    - Do you predict a photo-quality score, run a bandit over candidate photos, or rank photos by expected click-through? What is the label?
    - How do you avoid optimizing for click-bait photos that lead to worse reviews?
    - How would you A/B test this when the treatment changes what other guests see in search?
40. [Likely] Design amenity detection from listing photos, and use it to check that listings really have the amenities they claim. [21]
    - How do you label at scale, and what precision do you need before showing a detected amenity to guests?
41. [Likely] Design Airbnb Categories (collections such as "Treehouses" or "Amazing pools"), where ML proposes listings and humans review them. [24]
    - How do you choose which candidates go to human review, and how do reviewer decisions retrain the model?
    - How do you rank inside a category versus across categories?
42. [Likely] How would you build a knowledge graph of places, Experiences and listing attributes, and use it in search or recommendations? [23]

---

## 4. Listing quality, pricing, value and trust models

43. [Reported] Design a system that finds high-quality listings (asked at the phone screen in one MLE loop). [1]
    - How do you define "quality" without circular reliance on bookings: reviews, host behavior, photos, cancellations?
    - How would the score feed search, badges or curated collections?
44. [Reported] Design a model that predicts or recommends the nightly price for an Airbnb listing, helping hosts set prices from market demand. [4][5]
    - Listings are not identical products, so how do you estimate demand without a clean demand curve? [35]
    - How would a "booking probability, then price strategy" two-stage design work, and why might you use a custom loss with an asymmetric penalty instead of MSE? [35]
    - How do you evaluate price suggestions when you never see what would have happened at a different price?
    - How do you handle hosts who do not accept the suggestion (selection bias in adoption)?
45. [Reported] Estimate a property's lifetime value. First agree with the interviewer on what lifetime value means here. Reported May 2026. [1][2][3]
    - Would you frame it as regression or as bucketized classification, and why might bucketizing be the better choice for steering decisions? [1]
    - What horizon and discounting do you use, and how do you handle censoring for new listings? [25]
    - Which teams would consume the estimate (host acquisition, onboarding, support priority), and how would you check that it changes their decisions?
46. [Reported] Design a system that calculates customer (guest) lifetime value and builds its ML pipeline. [4]
    - How does guest LTV differ from listing LTV in labels, horizon and use?
    - How would you use it to pick users for a promotional campaign and measure incremental impact? [32]
47. [Reported] Design relisting detection: catch a host who deletes a listing and relists the same property, for example to shed bad reviews. Show how the outputs show up in online metrics. [1][2][3]
    - How would geohashing help generate candidate pairs, and how do you handle nearby but distinct units in the same building? [1]
    - Which signals would you combine (photo embeddings, text similarity, host/device/payment links)?
    - What action follows a match (block, merge reviews, manual review), and how do you measure false positives?
48. [Likely] Design a "family friendly" listing classifier. [52]
    - Where do the labels come from (amenities, reviews, host text, photos), and how noisy are they?
    - How would you use an LLM or vision-language model to bootstrap labels, and how do you audit it?
49. [Likely] Design a model that predicts guest or host cancellations at booking time.
    - How would that prediction be used: in ranking, in host tools, or in support?
50. [Likely] Design a real-time booking-time risk model for offline risks such as property damage or unauthorized parties. [36]
    - How do you set the decision threshold when false positives block legitimate guests?
    - How do you get labels when most incidents are never reported?
    - How do you measure the impact of a risk model in an A/B test without letting fraud through in control?
51. [Likely] Design a system that detects fake listings or scam messages using graph signals, embeddings and device IDs. [36]
52. [Likely] How would a guest-host match quality model feed into both search ranking and Instant Book eligibility?
53. [Likely] Forecast vacant nights or occupancy for each listing so that host tools can suggest price or availability changes. "Average Vacant Days" appears as an Airbnb SQL prompt [2].
    - How do you handle listings the host blocks on purpose, which look the same as nights with no demand?

---

## 5. ML platform: Bighead, Zipline, Chronon

### 5.1 Bighead (end-to-end ML platform)

54. [Likely] Airbnb built Bighead because shipping a model took weeks to months [38]. If you joined a team without a unified ML platform, which component would you build first, and why?
    - How would you standardize the path from notebook to production?
    - How do you make training reproducible: environment pinning with Docker, data snapshots, versioned features?
55. [Likely] How would you design model serving for a search ranker that scores hundreds of listings per request under a strict p99 latency budget?
    - What do you batch, cache or precompute?
    - How do you safely roll out a new model version (shadow, canary, A/B)?
56. [Likely] What should a production ML monitoring stack watch for an Airbnb ranker (feature drift, prediction drift, calibration, business KPIs)? [1]
57. [Likely] How would you review an ML pipeline pull request for correctness and training/serving consistency? A multi-PR code-review round is reported in Airbnb SWE loops (Jun 2026) [2], and one unverified post describes it for an ML-platform role [54].

### 5.2 Zipline and Chronon (feature platform)

58. [Likely] What is training/serving skew, and how does a "define once, compute offline and online" feature platform such as Chronon prevent it? Airbnb postings stress training/serving skew [1]. [6][19]
    - What does point-in-time correctness mean for a feature like "number of views of this listing in the last 7 days"?
    - What goes wrong if you join today's snapshot to last month's labels?
59. [Likely] Explain how a Chronon `GroupBy` (windowed aggregations such as SUM, COUNT, AVERAGE keyed by an entity) and a `Join` (a left event source plus several GroupBys) produce a point-in-time training set. [6]
    - How do you backfill a new feature over a year of history efficiently, without snapshotting every timestamp? [6][19]
    - How do batch and streaming paths combine to serve a fresh value online?
60. [Likely] Chronon measures online/offline consistency by logging online fetches, replaying them as a backfill and comparing the values [6]. How would you set alert thresholds on that metric, and what would you do when it drifts?
61. [Likely] Design the real-time feature pipeline that supplies "listings this guest viewed in the current session" to the search ranker within a few hundred milliseconds.
    - Which store (key-value store, in-memory cache) do you use, and what happens on a cache miss?
62. [Likely] Chronon is open source, and its repository lists other companies such as Stripe among its adopters [6]. What tradeoffs would you weigh between an in-house feature platform and an off-the-shelf feature store?
63. [Likely] How would you reuse a feature platform for LLM applications, for example to fetch real-time user and reservation context for a support agent? [39]

---

## 6. Metrics and experimentation context: Minerva, ERF, interleaving

64. [Likely] Airbnb built Minerva so that a metric is defined once and used for analytics, experimentation and reporting [40][41]. Why does that matter for ML launch decisions?
    - What goes wrong when Data Science and Finance compute "bookings" with different logic?
    - How would you add a new metric for your model launch without breaking downstream consumers?
65. [Reported] Design the data model behind cumulative booking metrics, including each user's first and last booking times. Reported Sep 2026. [2]
    - [Likely] How would this metric layer serve both experiment readouts and dashboards consistently (the Minerva problem)? [40]
66. [Reported] Design an A/B testing platform. Reported Apr 2026. [2]
    - [Likely] Using Airbnb's ERF as the reference [14], how do you join assignment logs with metric events and compute results for thousands of metric/experiment pairs?
    - At what unit do you randomize for a search ranker: user, device or search session? What happens with logged-out users?
    - How do you deal with users who switch devices between searching and booking?
67. [Reported] Design robust primary and guardrail metrics for an experiment. [2]
    - [Likely] Airbnb escalates a launch when it moves another team's metrics, for example a test that raised bookings but lowered review ratings [15]. Which guardrails would you set for a ranking launch: cancellations, review ratings, CS contacts, host earnings, latency?
    - How do you set an escalation threshold, and what is a "power guardrail"? [15]
68. [Reported] Design an A/B test that uses causal inference. [2]
    - [Likely] When you cannot randomize (a policy change in one market, a host-side launch), how would an ML-based counterfactual estimator such as Airbnb's ACE work, and how do you check it for bias? [17]
69. [Reported] Design an experiment for a listing badge (for example a "fast Wi-Fi" badge) that accounts for network effects. [2]
    - Why does a listing-level badge violate SUTVA when guests compare badged and unbadged listings in the same search?
    - Would you randomize by listing, by guest, by market, or with a two-sided (bipartite) design? What bias does each leave?
70. [Likely] Bookings have high variance and long latency. How would you make ranking experiments more sensitive? [42]
    - How could a model of leading indicators (a surrogate or value function) give credit for the eventual booking to earlier steps?
    - What are the risks of optimizing a surrogate metric instead of bookings?
71. [Likely] Airbnb puts a monetary "future value" on actions so that teams can compare tradeoffs on a common scale [16]. How would you estimate the long-term value of a cancellation, a bad review or a support contact, and use it in a launch decision?
72. [Likely] Airbnb uses interleaving to screen search-ranking candidates on a small share of traffic before A/B testing [18][37]. Explain team-draft interleaving and how you would attribute bookings to one ranker.
    - When is interleaving invalid (set-level diversity, results shared with the map)?
    - What is online counterfactual evaluation, and when would you use it instead?
    - How do you show that interleaving outcomes predict A/B outcomes?
73. [Likely] In a two-sided marketplace, treatment-group and control-group guests compete for the same listing-nights, so a ranking test can move bookings from control to treatment. How do you detect and correct this interference?
    - When would you use market-level or time-split (switchback) randomization?
74. [Likely] A model launch helps guests but lowers host earnings in some markets. How do you decide whether to ship?

---

## 7. 2024-2026 Airbnb AI: customer-service LLMs

75. [Reported] Design retrieval and ranking for a generative-AI tool that helps customer-support staff answer guest and host issues (recent senior loops). [1]
    - What does the knowledge base contain (policies, help articles, past resolved tickets), and how do you chunk and index it?
    - How do you evaluate retrieval quality and answer faithfulness before agents see the answers?
    - How do you stop the system from citing an outdated or region-specific policy?
76. [Reported] Generate customer-support instructions from company policies. Reported Sep 2026 in senior loops. [1][2]
    - [Likely] How would you turn policies and workflows into a format an LLM can follow reliably, such as an intent-context-action structure? [46]
    - How would you generate synthetic fine-tuning or evaluation data from those instructions?
    - How do you detect when a policy change makes generated instructions stale?
77. [Reported] Redact sensitive information (PII, contact details, payment data) from documents or messages. In recent senior loops this was reported as a coding exercise; here it is asked as a design question. [1]
    - How do you trade off recall (missing PII) against precision (over-redacting useful content)?
    - Would you use regex, NER, an LLM, or a cascade, and why? How does Airbnb-scale automated data classification inform this? [27]
78. [Reported] Design a customer-support ticketing and routing system. Reported Jun 2026 as a system design prompt. [2]
    - [Likely] Where does ML enter: contact-reason classification, priority and urgency scoring, agent-skill matching, deflection to self-serve? [22]
    - How do you measure whether a routing model shortened resolution time and did not just move work between queues?
79. [Likely] Airbnb's support agent was reported to be built on many models (13, per the CEO) and to resolve a large share of issues without a human [43][44]. How would you architect a multi-model support agent?
    - Which components would you expect: intent/contact-reason classifier, retriever, policy-reasoning LLM, tool/action executor, guardrail model, escalation model?
    - Which model do you route each sub-task to, and how do you balance cost, latency and quality?
80. [Likely] Airbnb's automation platform evolved from predefined conversational workflows [26] to a version that combines LLM reasoning (chain of thought, tool selection) with those workflows plus a guardrails layer [39]. When should an action (refunds, cancellations) be handled by a deterministic workflow, and when by an LLM?
    - How do you design guardrails against off-topic, unsafe or policy-violating responses?
    - How do you handle hallucination and latency?
81. [Likely] What is the right primary metric for an AI support agent: self-resolution rate, CSAT, reduction in contacts to humans, repeat-contact rate, or refund accuracy? How would you A/B test it?
    - Why are "share of tickets handled by AI" and "reduction in people needing a human" not the same metric? [43][44]
82. [Likely] Design the voice support pipeline: speech-to-text, contact-reason detection, self-serve article retrieval and escalation-intent detection. [45]
    - What latency budget applies to intent detection on a live call?
    - How do you deal with transcription errors before classification?
83. [Likely] Airbnb's earlier work fine-tuned text-generation models (for example MT5) for help-center content recommendation, agent QA assistance and chatbot paraphrasing [47]. How would you redo that system with today's LLMs, and what would you keep?
84. [Likely] How would you fine-tune a support model (SFT, preference optimization, distillation) on millions of past agent conversations without learning bad agent behavior? [48]
    - How do you filter or weight the transcripts, and how do you protect PII in the training data?
85. [Likely] How would you build an offline evaluation harness for a support agent, with LLM-as-judge, policy-compliance checks and human review sampling? [1]
    - How do you calibrate the LLM judge against expert agents?
86. [Likely] A guest says the listing was not as described and asks for a refund. Walk through how the AI agent should gather evidence, apply policy and decide whether to escalate.
87. [Likely] Design intent classification for guest-host messages (check-in questions, pet requests, cancellations, off-platform payment attempts). [22]
    - How do you discover new intents without labels, and how do you route a high-risk intent to trust and safety?

---

## 8. 2025-2026 Airbnb AI: natural-language and AI search

88. [Likely] Airbnb has rolled out natural-language / AI-assisted search across Homes, Experiences and Services, which accepts text or voice and creates filters on the fly [49][50]. Design this system end to end.
    - How do you parse a query like "quiet cabin near a lake for 6 with a hot tub, under $300" into structured filters and soft preferences?
    - Which constraints are hard filters and which are ranking signals?
    - How do you keep classic filters and AI-generated filters consistent?
89. [Likely] How would you combine LLM query understanding with the existing EBR and Journey Ranker stack, rather than replace it?
    - Where does free-text-to-listing matching happen: retrieval (multimodal embeddings), reranking, or both? [34]
90. [Likely] AI search usage is opt-in and selection-biased. How would you A/B test it against classic search?
    - Which metrics separate "more engagement" from "more bookings"?
91. [Likely] How do you prevent an LLM-powered search from claiming that listings have amenities they lack? [21]
92. [Likely] Chesky has described an app that "doesn't just search for you; it understands you" [50]. How would you use a guest's past stays and reviews for personalization in AI search while respecting privacy?
93. [Likely] How would you handle LLM latency and cost for search at Airbnb's query volume (caching, small models, distillation, query-class routing)?
94. [Likely] Airbnb is reportedly building an AI agent to keep travel discovery on its own platform rather than in third-party AI assistants [51]. What ML and product risks follow from that strategy?

---

## 9. Cross-cutting senior-level probes using Airbnb context

95. [Reported] Walk through one ML project end to end: the problem, your decisions, the trade-offs, the scope and the measured impact. Then the interviewer asks about milestones and a scope conflict. [1][4]
    - [Likely] Be ready for the interviewer to drop the project and ask an ML design question instead [5]. How would you link your project to an Airbnb problem?
96. [Likely] Which Airbnb ML paper or blog post do you know best? What would you have done differently, and what would you try next?
97. [Likely] Pick one Airbnb surface (search, similar listings, pricing, support). In your view, what is the current bottleneck in its ML system, and what is your 6-month roadmap?
98. [Likely] Airbnb's published ranking gains are often small, such as about 0.3% booking conversion [29][31]. How do you decide whether an improvement that small is real and worth the complexity?
    - How do you account for long-term effects and novelty?
99. [Likely] Airbnb inventory is seasonal, and travel demand changed sharply in 2020-2021. How would you make ranking and pricing models robust to large distribution shifts?
100. [Likely] How do you prioritize between model improvements, feature-platform investment and experimentation-speed investment such as interleaving?
101. [Likely] How would you design an ML system that serves Homes, Experiences and Services, which have very different inventory and booking patterns? Do you share models or separate them?
102. [Likely] Airbnb's audience-expansion work uses extreme classification for marketing in a two-sided marketplace [32]. How would you choose which users to target for a promotional email, and how would you measure incremental impact?
103. [Likely] Describe a time your offline evaluation and online results disagreed. How did you diagnose it, and what did you change in your process? This is the project-deep-dive version of question 6. [1]
104. [Likely] An Airbnb ranker underperforms in a single market, such as a new country. How do you debug it?
    - What do you check first: feature coverage, label sparsity, currency normalization, or local inventory differences?
105. [Likely] A host claims their listing "disappeared" from search. How would you explain the ranking change to them?
    - What tooling (ranking dashboards, explanations) would you build for internal teams? [11]

## Deepening pass (October 2026): newly reported questions

> Five web searches were run on 2026-10-16. Pages themselves could not be opened, so every [Reported] tag below rests on the search-result title or snippet only. The 1point3acres threads are login-gated: only their titles and summaries are visible. [58] comes from data-science / analytics loops, not MLE loops, and is included because this role covers experimentation.

### Newly reported

106. [Reported] Design an ML system for property damage claims at Airbnb, for example triaging or pricing a host's damage claim after a stay. Reported in a full-time MLE onsite alongside a pricing question. [55]
    - Is the target fraud likelihood, claim validity, payout amount, or routing priority? Can one model serve all of them?
    - What evidence does the model see (photos, messages, check-in/out timing, guest and host history), and how do you label claims that were settled without investigation?
    - How does this claims model relate to a booking-time risk model (question 50), and should they share features or labels?
107. [Reported] In an ML design round on price prediction, compare tree-based models with neural networks. Which would you choose, and why? Reported in a 2022 PhD MLE loop and again in onsite overviews listing pricing prediction among ML design topics. [56][59]
    - How does your answer change with feature types (categorical location, sparse amenities, photos and text)?
    - How would you check that the model's price response is monotonic where it should be?
108. [Reported] How would you choose the decision threshold for a production classifier, and when would you settle it with an A/B test instead of offline curves? [52]
    - What business cost do you assign to false positives and false negatives, and who owns that number?
109. [Reported] Design a system for real-time content personalization at Airbnb. Listed among candidate-submitted Airbnb MLE questions. [57]
    - What does "real-time" mean here: in-session re-ranking, a refreshed home feed, or push notifications?
    - Which signals must be fresh within seconds, and which can be computed daily?
110. [Reported] Design an accommodation booking system for Airbnb. Listed among candidate-submitted Airbnb MLE questions. [57]
    - How do you prevent double bookings for the same listing-night under concurrent requests?
    - Where would ML plug into the booking flow (risk check, Instant Book eligibility, price display)?
111. [Reported] Which loss functions have you used, and why did you choose each one? Listed among candidate-submitted Airbnb MLE questions. [57]
    - When would you choose a ranking loss (pairwise, listwise) over a pointwise loss for search?
    - When is an asymmetric or custom loss justified, as in Airbnb's pricing work? [35]
112. [Reported] How would you control for seasonality in an A/B test? Reported in Airbnb data-science / analytics loops (Nov 2023, Nov 2024). [58]
    - What is your experience with experimentation, and which test of yours had the most surprising result?
    - Would you extend the test, use a pre-period covariate, or use a holdback, and why?

### Gap-closing questions (inferred for the senior bar)

113. [Likely] How would you use CUPED or another pre-experiment covariate to reduce variance in a booking-conversion test, and which covariate would you choose for guests who have never booked?
114. [Likely] Product managers check experiment dashboards daily. How do you prevent peeking from inflating false positives (sequential tests, always-valid p-values, fixed horizons)?
115. [Likely] Your ranking experiment shows a sample ratio mismatch between treatment and control. How do you find the cause, and can any of the results be trusted?
116. [Likely] The average treatment effect of a ranker is flat, but you suspect it helps new guests and hurts repeat guests. How do you estimate heterogeneous treatment effects without p-hacking across segments?
117. [Likely] Bookings arrive days or weeks after the search that caused them. How do you handle label delay when training a ranker daily, and when reading out an experiment that ran for two weeks?
118. [Likely] A ranker trained on its own logged results learns a feedback loop in which top listings stay on top. How do you detect this, and what exploration or counterfactual correction would you use?
119. [Likely] Your model's predicted booking probabilities feed pricing and ads decisions downstream. How do you check and keep calibration after each retrain?
120. [Likely] How would you evaluate AI-generated filters or answers in natural-language search before launch, given that there is no ground-truth label for most free-text queries?
121. [Likely] A newly launched ranker causes a drop in bookings in one region over a weekend. Walk through the incident: detection, rollback decision, root cause and the process change you would make.
122. [Likely] As a senior engineer, how would you settle a disagreement with data science or product over whether a launch with mixed guardrail results should ship?

---

## References

Status key: **verified** = fetched and read on 2026-10-14; **URL confirmed** = title, URL and year match the applied-ml index [7], but the page was not opened; **unverified** = the host could not be reached in this review, and the entry is carried over from the previous version of this chapter.

1. "Airbnb Machine Learning Engineer Interview Process" (TrueInterview-derived guide, 3 candidate reports, mostly 2026 Q1-Q2: search ranking, property lifetime value, relisting with geohashing, cover photo with cold start, high-quality listings, generative-AI support retrieval and ranking, document redaction, policy-to-instructions). kevin-2023-code/Tech-Interview-Questions, GitHub, 2026. Verified. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/airbnb/interview-process-machine-learning-engineer.md
2. "Airbnb" question index with report dates (Rank Home Search Results Without a Text Query, Sep 2026; Generate Customer-Support Instructions from Company Policies, Sep 2026; Model Cumulative Booking Metrics, Sep 2026; Design Customer Support Ticketing System, Jun 2026; Code Review Multi-PR, Jun 2026; Listing Lifetime Value, May 2026; Design User Embedding Semantic Search, Apr 2026; Design an A/B Testing Platform, Apr 2026; Home Page Search + Availability + Ranking, Mar 2026; undated: personalized search ranking, relisting detection, primary and guardrail metrics, A/B test with causal inference, network-aware Wi-Fi badge experiment). kevin-2023-code/Tech-Interview-Questions, GitHub, 2026. Verified. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/airbnb.md
3. "Airbnb interview process" (MLE candidates get one or two ML system design rounds on search ranking, a property's lifetime value, recommendations, or relisting detection with geohashing; "Problem and label before model"). kevin-2023-code/Tech-Interview-Questions, GitHub, 2026. Verified. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/airbnb/interview-process.md
4. 1point3acres post #40, English mirror (MLE loop: project deep dive; ML design on customer lifetime value and on helping hosts set prices; coding LC 1298). FlowSight/coding, GitHub. Verified. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/40.md
5. 1point3acres post #79, English mirror (MLE loop: two ML design rounds on listing recommendation and price prediction; an "ML experience" round that turned into an unplanned ML design question; ConnectN coding). FlowSight/coding, GitHub. Verified. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/79.md
6. "Chronon: a data platform for AI/ML" (GroupBy and Join, point-in-time backfills, online/offline consistency measurement, adopters list). airbnb/chronon, GitHub. Verified. https://github.com/airbnb/chronon
7. Yan, "applied-ml: Papers & tech blogs by companies sharing their work on data science & machine learning in production." eugeneyan/applied-ml, GitHub. Verified; used to confirm [8]-[27]. https://github.com/eugeneyan/applied-ml
8. Haldar et al., "Applying Deep Learning To Airbnb Search," arXiv 1810.09591 (KDD 2019). URL confirmed. https://arxiv.org/abs/1810.09591
9. Haldar et al., "Improving Deep Learning For Airbnb Search," arXiv 2002.05515 (KDD 2020). URL confirmed. https://arxiv.org/abs/2002.05515
10. Grbovic, "Listing Embeddings in Search Ranking," Airbnb Tech Blog, 2018 (companion to Grbovic and Cheng, "Real-time Personalization using Embeddings for Search Ranking at Airbnb," KDD 2018). URL confirmed. https://medium.com/airbnb-engineering/listing-embeddings-for-similar-listing-recommendations-and-real-time-personalization-in-search-601172f7603e
11. Grbovic, "Machine Learning-Powered Search Ranking of Airbnb Experiences," Airbnb Tech Blog, 2019. URL confirmed. https://medium.com/airbnb-engineering/machine-learning-powered-search-ranking-of-airbnb-experiences-110b4b1a0789
12. Abdool et al., "Managing Diversity in Airbnb Search," arXiv 2004.02621, 2020. URL confirmed. https://arxiv.org/abs/2004.02621
13. "Learning To Rank Diversely," Airbnb Tech Blog, 2022. URL confirmed. https://medium.com/airbnb-engineering/learning-to-rank-diversely-add6b1929621 . Paper version, arXiv 2210.07774, unverified: https://arxiv.org/abs/2210.07774
14. "Scaling Airbnb's Experimentation Platform" (ERF), Airbnb Tech Blog, 2017. URL confirmed. https://medium.com/airbnb-engineering/https-medium-com-jonathan-parks-scaling-erf-23fd17c91166
15. "Designing Experimentation Guardrails," Airbnb Tech Blog, 2021. URL confirmed. https://medium.com/airbnb-engineering/designing-experimentation-guardrails-ed6a976ec669
16. "How Airbnb Measures Future Value to Standardize Tradeoffs," Airbnb Tech Blog, 2021. URL confirmed. https://medium.com/airbnb-engineering/how-airbnb-measures-future-value-to-standardize-tradeoffs-3aa99a941ba5
17. "Artificial Counterfactual Estimation (ACE): Machine Learning-Based Causal Inference at Airbnb," Airbnb Tech Blog, 2022. URL confirmed. https://medium.com/airbnb-engineering/artificial-counterfactual-estimation-ace-machine-learning-based-causal-inference-at-airbnb-ee32ee4d0512
18. "Beyond A/B Test: Speeding up Airbnb Search Ranking Experimentation through Interleaving," Airbnb Tech Blog, 2022. URL confirmed. https://medium.com/airbnb-engineering/beyond-a-b-test-speeding-up-airbnb-search-ranking-experimentation-through-interleaving-7087afa09c8e
19. Simha, "Zipline: Airbnb's Machine Learning Data Management Platform" (2018) and "Zipline - A Declarative Feature Engineering Framework" (2020), conference talks. URL confirmed. https://www.youtube.com/watch?v=Tg5VEMEsC-0 ; https://www.youtube.com/watch?v=LjcKCm0G_OY
20. "Categorizing Listing Photos at Airbnb," Airbnb Tech Blog, 2018. URL confirmed. https://medium.com/airbnb-engineering/categorizing-listing-photos-at-airbnb-f9483f3ab7e3
21. "Amenity Detection and Beyond — New Frontiers of Computer Vision at Airbnb," Airbnb Tech Blog, 2019. URL confirmed. https://medium.com/airbnb-engineering/amenity-detection-and-beyond-new-frontiers-of-computer-vision-at-airbnb-144a4441b72e
22. "Discovering and Classifying In-app Message Intent at Airbnb," Airbnb Tech Blog, 2019. URL confirmed. https://medium.com/airbnb-engineering/discovering-and-classifying-in-app-message-intent-at-airbnb-6a55f5400a0c
23. "Contextualizing Airbnb by Building Knowledge Graph," Airbnb Tech Blog, 2019. URL confirmed. https://medium.com/airbnb-engineering/contextualizing-airbnb-by-building-knowledge-graph-b7077e268d5a
24. "Building Airbnb Categories with ML and Human-in-the-Loop," Airbnb Tech Blog, 2022. URL confirmed. https://medium.com/airbnb-engineering/building-airbnb-categories-with-ml-and-human-in-the-loop-e97988e70ebb
25. "Using Machine Learning to Predict Value of Homes On Airbnb," Airbnb Tech Blog, 2017. URL confirmed. https://medium.com/airbnb-engineering/using-machine-learning-to-predict-value-of-homes-on-airbnb-9272d3d4739d
26. "Intelligent Automation Platform: Empowering Conversational AI and Beyond at Airbnb," Airbnb Tech Blog, 2022. URL confirmed. https://medium.com/airbnb-engineering/intelligent-automation-platform-empowering-conversational-ai-and-beyond-at-airbnb-869c44833ff2
27. "Automating Data Protection at Scale, Part 1," Airbnb Tech Blog, 2021. URL confirmed. https://medium.com/airbnb-engineering/automating-data-protection-at-scale-part-1-c74909328e08
28. Tan et al., "Optimizing Airbnb Search Journey with Multi-task Learning" (Journey Ranker), arXiv 2305.18431 (KDD 2023). Unverified. https://arxiv.org/abs/2305.18431
29. Tang, Gao, He and Katariya, "Multi-objective Learning to Rank by Model Distillation," arXiv 2407.07181 (KDD 2024). Unverified. https://arxiv.org/abs/2407.07181
30. Haldar et al., "Learning to Rank for Maps at Airbnb," arXiv 2407.00091 (KDD 2024). Unverified. https://arxiv.org/abs/2407.00091
31. "Embedding-Based Retrieval for Airbnb Search," Airbnb Tech Blog, and "Applying Embedding-Based Retrieval to Airbnb Search," arXiv 2601.06873, 2026. Unverified. https://airbnb.tech/ai-ml/embedding-based-retrieval-for-airbnb-search/ ; https://arxiv.org/abs/2601.06873
32. "Academic Publications: Airbnb Tech 2025 Year in Review" (location retrieval, audience expansion, interleaving), Airbnb Tech Blog, 2025, and "Transforming Location Retrieval at Airbnb," arXiv 2408.13399. Unverified. https://airbnb.tech/infrastructure/academic-publications-airbnb-tech-2025-year-in-review/ ; https://arxiv.org/abs/2408.13399
33. "Learning and Applying Airbnb Listing Embeddings" (KDD 2024 workshop paper), Airbnb Tech. Unverified. https://airbnb.tech/wp-content/uploads/sites/19/2024/12/Learning-and-Applying-Airbnb-Listing-Embeddings-.pdf
34. Guy, Grbovic, Tan and Zhao, "BiListing: Modality Alignment for Listings," arXiv 2508.20396 (CIKM 2025). Unverified. https://arxiv.org/abs/2508.20396
35. Ye et al., "Customized Regression Model for Airbnb Dynamic Pricing," KDD 2018. Unverified. https://dl.acm.org/doi/10.1145/3219819.3219830
36. "Senior Machine Learning Engineer, Trust & Safety," Airbnb Careers (job posting; may have closed). Unverified. https://careers.airbnb.com/positions/7185448/
37. Zhang et al., "Harnessing the Power of Interleaving and Counterfactual Evaluation for Airbnb Search Ranking," arXiv 2508.00751, 2025. Unverified. https://arxiv.org/abs/2508.00751
38. "Bighead: Airbnb's End-to-End Machine Learning Platform," Data Council talk. Unverified. https://datacouncil.ai/talks/bighead-airbnbs-end-to-end-machine-learning-platform
39. "Automation Platform v2: hybrid LLM conversational platform" (third-party summary of a 2024 Airbnb Tech Blog post), ZenML LLMOps database. Unverified. https://www.zenml.io/mlops-database/airbnb-chronon-internal-dataai-app-platform-conversational-ai-platform-automation-platform-v2-hybrid-llm-conversational
40. "Ep 1: Robert Chang on Scaling Metrics @ Airbnb" (Minerva), dbt Roundup. Unverified. https://roundup.getdbt.com/p/ep-1-robert-chang-on-scaling-metrics
41. "How did Airbnb achieve metric consistency at scale" (Minerva), Analytics India Magazine. Unverified. https://analyticsindiamag.com/ai-features/how-did-airbnb-achieve-model-metric-consistency-at-scale
42. Deng, Du and Matlin, "Continuous Attribution of Episodical Outcomes for More Efficient and Targeted Online Measurement," arXiv 2210.16373. Unverified. https://arxiv.org/abs/2210.16373
43. "AI Agents Handle 30% of Airbnb Customer Support Tickets as Company Expands Automation," CX Today, 2026. Unverified. https://www.cxtoday.com/?p=212503
44. "Airbnb Claims 'Best AI Self-Solve in All of Travel' After AI Resolves 40% of Customer Issues," CXM World, 2026. Unverified. https://cxm.world/customer-experience/airbnb-claims-best-ai-self-solve-in-all-of-travel-after-ai-resolves-40-of-customer-issues/
45. "Listening, Learning, and Helping at Scale: How Machine Learning Transforms Airbnb's Voice Support Experience," Airbnb Tech Blog. Unverified. https://airbnb.tech/ai-ml/listening-learning-and-helping-at-scale-how-machine-learning-transforms-airbnbs-voice-support-experience/
46. "LLM-Friendly Knowledge Representation for Customer Support," arXiv 2510.10331, 2025. Unverified; the venue claimed in the previous version (COLING 2025) did not fit the October 2025 arXiv ID and has been dropped. https://arxiv.org/abs/2510.10331
47. "LLM integration for customer support automation and enhancement" (third-party summary of Airbnb's "How AI Text Generation Models Are Reshaping Customer Support"), ZenML LLMOps database. Unverified. https://zenml.io/llmops-database/llm-integration-for-customer-support-automation-and-enhancement
48. "Principal Machine Learning Engineer - LLM Fine-tuning and Optimization," Airbnb Careers (job posting; may have closed). Unverified. https://careers.airbnb.com/positions/7955579/
49. "Airbnb Adds AI-Powered Search and Social Features," Let's Data Science, 2026. Unverified. https://letsdatascience.com/news/airbnb-adds-ai-powered-search-and-social-features-dd4ce7ea
50. "Airbnb's AI Strategy for 2026 Is Already Taking Shape," Rental Scale-Up, 2026. Unverified. https://www.rentalscaleup.com/airbnbs-ai-strategy-2026/
51. "Airbnb Is Building an AI Agent to Preserve Customer Traffic," AltexSoft, 2025. Unverified. https://www.altexsoft.com/travel-industry-news/airbnb-is-building-an-ai-agent-to-preserve-customer-traffic
52. "Airbnb Senior Machine Learning Engineer Interview Experience — Four ML System Design Rounds, Rejected on Calibration," PracHub, Apr 2026. Page still unreachable, but search-result snippets seen on 2026-10-16 (October 2026 deepening pass) confirm four ML system / ranking / marketplace rounds, past-project probes on precision vs. recall vs. latency vs. infra cost, offline/online metric disagreement, threshold tuning via A/B test, and FM vs. embedding + DNN; questions 5-7 are retagged [Reported] on that basis. https://prachub.com/interview-experiences/airbnb-senior-machine-learning-engineer-interview-experience-four-ml-system-design-rounds-rejected-on-calibration
53. "Airbnb senior MLE interview prep," Blind. Unverified (host unreachable). https://www.teamblind.com/post/airbnb-senior-mle-interview-prep-n1mdsd3b
54. "Interviewing for Airbnb ML platform role," Blind, 2025. Unverified (host unreachable). https://www.teamblind.com/post/interviewing-for-airbnb-ml-platform-role-8in7czso
55. "Airbnb Machine Learning Fulltime Onsite Interview Experience and Insights" (title and summary only: coding, project discussions, and a system design on property damage claims and pricing; details login-gated). 1point3acres, 2025. Search snippet only, 2026-10-16. https://www.1point3acres.com/interview/thread/1162455
56. "Airbnb Fulltime Machine Learning Engineering Onsite Interview Overview" (title and summary only: ML design, pricing prediction, coding, project discussion; details login-gated). 1point3acres, 2025. Search snippet only, 2026-10-16. https://www.1point3acres.com/interview/thread/1114733
57. "Airbnb Machine Learning Engineer interview questions" (candidate-submitted list: design an accommodation booking system, real-time content personalization, loss functions you have used). Prepfully. Search snippet only, 2026-10-16. https://prepfully.com/interview-questions/airbnb/machine-learning-engineer?page=4
58. Airbnb interview reviews (analytics / data-science loops, Nov 2023 and Nov 2024: experimentation case study, "What is your experience with experimentation?", controlling seasonality in an A/B test). Glassdoor UK. Search snippet only, 2026-10-16. https://www.glassdoor.co.uk/Interview/Airbnb-Interview-E391850-RVW84022306.htm ; https://www.glassdoor.co.uk/Interview/Airbnb-Interview-E391850-RVW96180524.htm
59. "2022(7-9月) MachineLearningEng 博士 全职@airbnb" (PhD full-time MLE loop: price prediction system design, comparing tree models with neural networks). 1point3acres, 2022. Search snippet only, 2026-10-16. https://www.1point3acres.com/bbs/thread-934565-1-1.html
