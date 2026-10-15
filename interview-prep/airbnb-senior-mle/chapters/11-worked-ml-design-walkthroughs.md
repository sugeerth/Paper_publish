# 11. Timed ML Design Walkthroughs: Diagram Templates, Timing Checklist and Stage-by-Stage Questions for Search Ranking, Host Pricing, Guest LTV, Listing Recommendation, Relisting Detection and Support Retrieval

> **Research note (read first).** Reviewed and expanded on 2026-10-14. The shared web-search budget for this run was already used up, so **no new search queries could run** during this review. The proxy could not resolve medium.com, arxiv.org, dl.acm.org, kdd.org or trueinterview.io, so the Airbnb Tech Blog posts and papers, and TrueInterview's own question pages, could not be opened. Glassdoor, Blind, LeetCode Discuss, Reddit, interviewing.io, Prepfully, IGotAnOffer, Exponent and InterviewQuery could not be searched or opened either.
>
> What **was** fetched and read on 2026-10-14:
>
> - the TrueInterview-derived Airbnb MLE loop guide, built from 3 candidate reports, mostly 2026 Q1-Q2 [1], read in full, including its sample ML design prompts and the interviewer steering it reports;
> - the dated Airbnb question index (78 questions) [2], the general Airbnb process page [3] and the Airbnb software-engineer process page [23];
> - every English 1point3acres Airbnb mirror post in the FlowSight archive that could be downloaded (about 100 posts) [4][5]. Only posts #40 and #79 contain ML design rounds; the rest are coding, standard system design or experience rounds;
> - the general nine-step ML design template [6];
> - two paper indexes, used only to confirm the titles and URLs of the Airbnb papers and posts [7][21];
> - the README of Airbnb's open-source feature platform, Chronon [22].
>
> The Airbnb papers and blog posts [8]-[20] and [24]-[31] were **not** opened. Before the onsite, cross-check this chapter against Glassdoor, Blind and LeetCode Discuss.
>
> **Scope, as you asked.** You asked for ML system design diagrams and for details on the tag system, as a PDF. So this chapter gives you:
>
> 1. a reusable whiteboard diagram template and six annotated prompt diagrams (four core walkthroughs plus two newly reported prompts);
> 2. a minute-by-minute timing checklist;
> 3. the tagged questions and follow-up probes an interviewer is likely to ask at each stage of the round.
>
> It does **not** contain written model answers. The diagrams are **study scaffolds**: each one shows where the boxes go and what to label. They are **not** Airbnb's production architecture, and no source says interviewers hand out diagrams. Chapter 03 lists the prompts. Chapter 10 has deeper per-system diagrams. This chapter is the timed, end-to-end practice layer that sits on top of both.

## How the tags work

Every numbered question carries exactly one tag. Chapters 01-10 use the same scheme, so the whole pack reads the same way in the PDF.

- **[Reported] [n]**: a fetched source [n] says this prompt, or a close variant of it, came up in an Airbnb interview. Sources [1], [3], [4] and [5] are MLE loops. The question index [2] does not record the role for each question, so a [2]-only item may come from a non-ML loop; the question says so where it matters.
- **[Reported: criterion] [n]**: the source does not quote this exact question. It does name the topic as something interviewers **assess** or **push on** in the Airbnb ML design round, for example "label definition", "cold start" or "freshness" [1][3]. Expect to be pushed on it, but the wording is ours.
- **[Likely]**: a well-grounded inference from the Senior MLE role, the reported round structure and Airbnb's published systems. A citation next to [Likely] points to the paper, post or platform behind the topic. It does **not** mean a candidate reported the question.
- **Follow-up probes** are the indented bullets under a question. They carry no tag of their own. They are probes a senior interviewer would plausibly use, unless the bullet itself names a source.
- **Never inferred:** no question is tagged [Reported] because it is common at other companies. The general template [6] mentions "Rental recommendation (Airbnb)" only as a practice prompt, so it is not counted as a report.

## Format notes for this round

- **How many rounds, and why they matter.** MLE candidates get one or two ML system design rounds instead of a general system design round, sometimes more [1][3]. The loop is decided mainly by design depth and past scope. In the reports, rejection feedback pointed to design depth or overall match, never coding [1].
- **Reported prompt pairs.** In one 1point3acres loop, the two design rounds were customer LTV and helping hosts set prices [4]. In another, they were listing recommendation and price prediction, and the "ML experience" round also turned into an unplanned design question [5]. A referred candidate got two ML design prompts (cover photo, high-quality listings) inside the phone screen [1]. So prepare at least four end-to-end walkthroughs, not two, and a two-minute opener for each.
- **Length and medium.** All reported onsites were over video [1]. Airbnb's standard design rounds are reported as about an hour of discussion without code [23]; the MLE sources give no per-round length [1][3]. Plan for 45-60 minutes with a shared doc or whiteboard.
- **What the interviewer scores** [1]:
  - problem framing and the success metric, before any model is mentioned;
  - label definition, including the time window;
  - data and features (reviews, host responsiveness, booking completion, photos, location encodings);
  - the offline-to-online link through A/B testing;
  - production concerns: candidate generation, latency, freshness, cold start and monitoring.
- **Level bar.** A Senior (G9) candidate is expected to own a modelling area from framing to launch and experiments. Staff (G10) is about technical direction across several teams [1]. At G9, you should drive the agenda yourself, make trade-offs out loud and defend a launch decision.
- **Interviewer behaviour.** Interviewers steer firmly. One LTV round fixed the label first, then moved from regression to bucketized classification; a relisting round went into geohashing; search ranking rounds pushed on latency, freshness and experimentation [1]. Treat a redirection as a hint: restate the narrower scope and carry on [1]. Two failure modes are reported:
  - jumping to model ideas before the problem and label are defined ("Problem and label before model") [1][3];
  - chasing perfect detail instead of agreeing on trade-offs [3].
- **Extra round.** A borderline design signal can trigger an added design round. Treat that as a warning sign, not as a neutral step [1][3].

## Reusable diagram template (draw this in the first 8-10 minutes)

The same nine boxes fit every Airbnb prompt in this chapter. Draw the skeleton first, then label each box for the prompt in front of you. Adapted from the step structure in [6] and the design skeleton and scoring criteria in [1].

```
 [1] OBJECTIVE & METRICS (two-sided)
     guest metric | host metric | marketplace/guardrail metric | north star (e.g. uncancelled bookings, GBV)
                    |
                    v
 [2] DATA & LABELS -----------------------------------------+
     event logs (search, impressions, clicks, contacts,     |
     booking requests, bookings, cancellations, reviews)    |
     label = ? | horizon = ? | delayed / censored? | bias?  |
                    |                                       |
                    v                                       |
 [3] FEATURES (point-in-time correct, one definition         |
     shared by offline backfill and online serving)          |
     guest | listing | host | query/context | market | real-time
                    |                                       |
         +----------+-----------+                           |
         v                      v                           |
 [4a] CANDIDATE GENERATION   [4b] MODEL / RANKER            |
      (retrieval, filters,        (objective, loss,         |
       availability)               multi-task heads)        |
         |                      |                           |
         +----------+-----------+                           |
                    v                                       |
 [5] POLICY / BUSINESS LAYER                                |
     diversity | host fairness | price rules | trust filters  |
                    |                                       |
                    v                                       |
 [6] OFFLINE EVAL  ---> [7] ONLINE TEST ---> [8] SERVING     |
     NDCG / AUC /         A/B, interleaving,   latency budget, |
     calibration /        two-sided            caching,        |
     error by segment     interference         fallbacks       |
                                                   |         |
                                                   v         |
 [9] MONITORING & FEEDBACK  -------------------------------+
     feature drift | prediction drift | label delay | alerts | retrain cadence
```

Label every arrow with **what flows** (for example, "logged impressions with position") and every box with **one owner decision** (for example, "label = booking within 7 days of search"). The scoring criteria are about these decisions (label, metric, offline-to-online link), not about how many boxes you draw [1][3].

## Timing checklist (45-60 minute round)

| Minute (60-min / 45-min) | Stage | Must-say before moving on |
|---|---|---|
| 0-5 / 0-4 | Clarify | Who is the user: guest, host or both? What decision does the model drive? Constraints: scale, latency, cold start |
| 5-10 / 4-8 | Objective & metrics | One primary metric, one guest guardrail, one host guardrail. The offline proxy for each |
| 10-17 / 8-13 | Data & labels | Exact label, horizon, censoring, position or selection bias, leakage risk |
| 17-24 / 13-18 | Features | Feature families by entity, real-time vs batch, point-in-time correctness |
| 24-32 / 18-25 | Model & diagram | Baseline first, then the main model; draw the template boxes |
| 32-38 / 25-30 | Offline eval | Metric, split (time-based), segment cuts, the gap between offline and online |
| 38-46 / 30-37 | Online test | Randomisation unit, interference handling, duration, decision rule |
| 46-53 / 37-42 | Serving & monitoring | Latency budget, fallbacks, freshness, drift alerts, retraining |
| 53-60 / 42-45 | Trade-offs & wrap | Two senior trade-offs, what you would ship first, what you would cut |

Checkpoint habits:

- At each stage boundary, ask the interviewer: "Want me to go deeper here or move on?"
- If you are behind at minute 30 (or 22), compress the features stage to entity families and protect online testing. The A/B link is an explicit scoring item [1].
- If design comes up inside a phone screen, the source says you are expected to name a target variable and a cold-start plan within minutes [1]. Use rows 1-3 of the table as a two-minute opener.

## Walkthrough A: Search ranking (home search, with and without a text query)

```
 Guest request (dates, guests, map viewport / destination, or NO query)
        |
        v
 [Retrieval] geo + availability + hard filters --> EBR / embedding recall --> ~1-5k candidates
        |                                                        ^
        v                                                        | listing & guest embeddings (batch)
 [First-pass ranker] light model --> top ~200                    |
        |                                                        |
        v                                                        |
 [Main ranker] DNN / multi-task heads: P(click), P(contact), P(book), P(no cancel)
        |   features: listing quality, price vs market, guest history, query/context, real-time session
        v
 [Re-rank policy] diversity (location, price), new-listing boost, trust filters
        |
        v
 Results page / map pins --> impression + position logs --> labels for next training run
```

Box labels to defend: the label (booking, contact or click, and which), position-bias correction, the "no query" context [2], feature freshness [1], and the latency budget per stage.

### Clarify and objective

1. Design search ranking for Airbnb homes, end to end, from candidate generation through offline and online evaluation and deployment. [Reported] [1][3]
    - Before any model: what is the label, and why that one rather than a click? [3]
    - What is the single business metric you would make the launch decision on?
2. Design a personalized search ranking system. [Reported] [2]
    - How much of the gain do you expect from personalization rather than from listing quality?
    - What do you show a logged-out first-time guest?
3. Rank home search results when the guest gives no text query. [Reported] [1][2]
    - What replaces the query as context: dates, map viewport, past searches, location?
    - How do flexible dates or a flexible destination change retrieval?
4. Design the home page so that search, availability and ranking work together. [Reported] [2]
    - The index lists this as a hard system design question (Mar 2026) without saying whether it came from an ML loop. In an ML round, where does the availability check sit relative to the ranker, and what do you cache per map viewport?
5. Which guest-side and host-side metrics would you track, and which is the guardrail? [Likely]
    - How would you detect that a ranking gain for guests concentrates bookings on a few hosts?
6. Should the ranker optimise booking, uncancelled booking, or booking plus a good review? [Likely] [17][21]
    - How do you handle labels that arrive weeks after the search?

### Data, labels and features

7. How do you correct for position bias in the training logs? [Likely] [10]
    - Would you use randomised slots, inverse propensity weighting or a position feature? What does each cost?
8. A guest searches over several sessions before booking. How do you assign credit across the journey? [Likely] [17][21]
    - Would intermediate milestones (wishlist, contact, booking request) become extra training heads?
    - How would you weight those heads, and what do you do when two heads pull in opposite directions?
9. Which listing features matter most, and which of them leak future information? [Reported: criterion] [1]
    - How do you keep review score and price features point-in-time correct?
10. How would you encode location for ranking: geohash cells, map-viewport buckets, distance to the searched point? What resolution would you choose? [Reported: criterion] [1]
11. How do you represent price: absolute, relative to the market, or relative to what this guest usually pays? [Likely] [8]
12. How would you use listing embeddings built from session co-clicks in ranking? [Likely] [11][24]
    - How do you get an embedding for a listing published yesterday?
13. How would you use the listings a guest clicked or skipped earlier in the same session to re-rank results in real time? [Likely] [24]
    - Which signals must arrive within seconds, and which can be batch?
14. How do you make sure the features the ranker trains on match what it sees at serving time, and how would you measure the gap? [Likely] [1][22]
    - Airbnb MLE postings stress training-serving skew, and Airbnb open-sourced a feature platform built for point-in-time backfills [1][22]. What would you check before blaming the model for an offline-online gap?

### Model, retrieval and diagram

15. Walk me through the stages from candidate retrieval to the final page, with a latency budget for each. [Reported: criterion] [1]
    - Where does availability filtering happen, and what breaks if it happens too late?
16. Design user-embedding semantic search. [Reported] [2]
    - How are user and listing towers trained, and what are the negatives?
    - How fresh must the user embedding be within a session?
17. Why move from GBDT to a neural ranker, and what would make you stay with GBDT? [Likely] [8][9]
18. How would you add diversity across location and price without retraining the ranker? [Likely] [12][13]
    - How would you measure whether diversity helped bookings or only looked better?
19. How do new listings get impressions without hurting guest experience? [Reported: criterion] [1]
    - How long does a new-listing boost last, and how do you decide?
20. How would ranking differ for map pins compared with the list view? [Likely]
21. How would you rank Airbnb Experiences, where there is far less booking data per item than for homes? [Likely] [10]

### Offline evaluation

22. Which offline metric would you use, and why might NDCG improve while bookings stay flat? [Likely]
23. How would you build a time-based holdout that mimics launch conditions? [Likely]
24. Which segments would you cut the evaluation by before trusting a win? [Likely]
    - New guests vs repeat guests, cities vs rural markets, short vs long stays?

### Online test and interference

25. How would you run the A/B test for a new ranker, and what is the randomisation unit? [Reported: criterion] [1]
    - Guest, search or market: what bias does each introduce?
26. Treatment guests book the best listings, so control guests see less inventory. How do you handle this interference? [Likely] [14]
    - Would you use listing-side or market-level randomisation, or a budget-split design?
27. When would you use interleaving instead of an A/B test, and what can interleaving not tell you? [Likely] [14]
    - How do you validate that interleaving agrees with A/B results?
28. What is your launch decision rule if bookings rise but cancellations also rise? [Likely] [20]
    - Which guardrail would block the launch? (Compare the reported prompt "Design robust primary and guardrail metrics" [2].)

### Serving, monitoring and trade-offs

29. What would you cache, what must be computed per request, and what is the fallback when the ranker times out? [Reported: criterion] [1]
30. Which features must be fresh within minutes (availability, price changes, in-session clicks), and how do you serve them without blowing the latency budget? [Reported: criterion] [1][22]
31. Which drift alerts would you set up, and what would trigger a retrain? [Reported: criterion] [1]
32. How do you stop the model training on its own feedback loop (ranking what it already ranked)? [Likely]
33. Which senior trade-off would you surface first: model complexity vs latency, personalization vs fairness, or short-term bookings vs long-term trust? [Likely]

## Walkthrough B: Host dynamic pricing (helping hosts set prices)

```
 Listing x night (date) request from Host Pricing UI / Smart Pricing
        |
        v
 [Demand model]  P(book | listing, night, price, lead time)  <-- seasonality, events, market supply, search demand
        |
        v
 [Price optimiser]  pick price maximising expected revenue (or host objective) within host min/max
        |                       ^
        v                       | price-elasticity estimate (needs exploration / causal design)
 [Guardrails] host min/max, market sanity bounds, change-rate limits
        |
        v
 Suggested nightly price --> host accepts / overrides --> booked or unbooked night --> labels
```

Box labels to defend:

- **The label.** A booked night is observed only at the price that was shown. An unbooked night is censored.
- **The objective.** Say whether you are optimising host revenue, booking probability or marketplace health.
- **Host adoption.** Hosts can override the suggested price.

### Clarify and objective

34. Design a system that helps hosts set prices from market demand and other factors. [Reported] [4]
    - Is the goal to maximise host revenue, occupancy or something else, and who decides?
35. Design a price-prediction model for Airbnb listings. [Reported] [5]
    - Are you predicting the "right" price, the price that will book, or the market price of similar listings?
36. What are the host-side, guest-side and marketplace metrics for a pricing product? [Likely] [15]
    - How would you notice that suggestions raise host revenue but lower guest conversion?
37. Is this a regression problem, a classification problem or a decision problem? [Likely] [15]

### Data and labels

38. You only see a booking at the price that was shown. How do you learn demand at prices you never showed? [Likely] [15]
    - What exploration would hosts accept?
39. How do you treat nights that stay unbooked: as negatives, as censored, or by lead time? [Likely]
40. Would you model the probability that a night ever books, or the probability that it books before a given lead time? How does lead time enter the model? [Likely] [15]
41. How do you separate price effects from listing-quality effects in the logs? [Likely]
42. How do host overrides of the suggested price enter the training data? [Likely]

### Features and model

43. Which demand signals would you use: search volume, events, seasonality, competitor supply? [Likely] [15]
    - Which of them are available in real time and which are batch?
44. How do you price a brand-new listing with no booking history? [Reported: criterion] [1]
45. How would you build a set of comparable listings to anchor a suggested price, and how do you define "comparable"? [Likely] [19]
46. Would you use one global model or market-level models, and how would you share strength across markets? [Likely]
47. How would you estimate price elasticity, and why is a plain regression on logged prices biased? [Likely]
48. How do you turn a predicted booking probability into a recommended price? [Likely] [15]
    - How do you keep recommendations stable from one day to the next?

### Evaluation and online test

49. Which offline metric fits a demand model, and why is price-prediction RMSE not enough? [Likely] [15]
50. Before any live test, how would you estimate the effect of a new pricing policy from logged data (off-policy or counterfactual estimation)? [Likely] [18]
51. How do you A/B test a pricing feature when treatment and control listings compete for the same guests? [Likely]
    - Would you randomise by listing, host or market, and what does each cost in power?
    - When would a market-level or switchback design be worth the loss of power?
52. How do you measure host adoption, and how do you separate the effect of the feature from selection bias among the hosts who opt in? [Likely]
53. Hosts who see suggestions may also change their calendar, minimum stay or discounts. How do you measure those second-order effects? [Likely]
54. If the experiment shows more bookings but less revenue per night, what would you ship? [Likely]

### Serving, monitoring and trade-offs

55. How often do suggested prices refresh, and what triggers an out-of-cycle update (for example, an event announced)? [Likely]
56. What monitoring would catch a price spike caused by a bad feature? [Reported: criterion] [1]
57. How do you handle fairness, regulation or price-gouging concerns during emergencies? [Likely]
58. How do you explain a suggested price to a host so that they trust it? [Likely]

## Walkthrough C: Guest (customer) and listing lifetime value

```
 Guest or listing at time t (signup / listing creation, first booking, or any later day)
        |
        v
 [Feature snapshot @ t] trips, nights, GBV, cancellations, reviews, channel, region, tenure, engagement
        |                (listing: reviews, host responsiveness, booking completion, photos, location)
        v
 [LTV model]  E[value over horizon H | features @ t]
        |       options: two-part (P(active) x spend), survival / BTYD baseline,
        |                GBDT/DNN regression, or bucketized classification over value bands
        v
 [Calibration + uncertainty] by cohort, region, acquisition channel
        |
        v
 Consumers: marketing bid caps | CRM targeting | ranking tie-breaks | finance forecasts
        |
        v
 Realised value after H (delayed label) --> back-test & recalibrate
```

Box labels to defend:

- **The definition.** Is value revenue, GBV or contribution margin, and over which horizon H? Agree it before modelling [1].
- **The censored label.** A recent guest or listing has not lived through the full horizon yet.
- **The consumer.** Who will act on the score?

### Clarify and objective

59. Design a system that calculates customer lifetime value. [Reported] [4]
    - Who will use the score, and how does that change the design?
    - The interviewer may not have time for deployment and serving. Which parts would you protect? [4]
60. Estimate a listing's (property's) lifetime value; start by agreeing on what lifetime value means here. [Reported] [1][2][3]
    - How does listing LTV differ from guest LTV in its label and its horizon?
61. Value, measured how: revenue, GBV, contribution margin, or incremental value? [Likely] [16]
62. What horizon would you choose, and how do you trade forecast accuracy against usefulness? [Likely] [16]
63. How would you turn an LTV estimate into a common currency that different teams can use to compare trade-offs? [Likely] [16]

### Data and labels

64. How do you build a training label when recent guests have not completed the horizon? [Likely]
65. How do you stop the training snapshot from leaking behaviour that happens after time t? [Reported: criterion] [1]
66. How do you handle the heavy tail of very high-value guests and the many zeros? [Likely]
67. Should cancellations, refunds and support costs reduce the label? [Likely]

### Features and model

68. Which features would you build at signup, when you have almost no behaviour? [Reported: criterion] [1]
69. For listing LTV, which marketplace signals would you use (reviews, host responsiveness, booking completion, photos, location), and which exist on the day the listing is created? [Reported: criterion] [1]
70. The interviewer moves you from regression to bucketized classification of LTV. Why might that be the better framing, and how would you choose the buckets and evaluate the model? [Reported] [1]
71. How would you estimate the value of a home before it is listed, for supply acquisition? [Likely] [19]
72. Why might a two-part model (probability of being active multiplied by spend) beat a single regression? [Likely]
73. What simple baseline (for example, a cohort average or a BTYD model) would you start with, and how would you know when to replace it? [Likely]
74. How would you add uncertainty to the estimate, and who needs it? [Likely]

### Evaluation, online use and causal questions

75. Which offline metrics would you use: calibration by decile, ranking quality (Gini or AUC on top deciles), or error on aggregate value? [Likely]
76. You can't A/B test an LTV score directly. How do you test the decisions it drives? [Likely] [16]
    - For example: a marketing-spend test with LTV-based bids against rule-based bids.
77. How do you separate predicted LTV from incremental LTV caused by an action? [Likely] [18]
    - Would you use uplift models, holdouts or counterfactual estimation?
78. Design robust primary and guardrail metrics for a launch that uses the LTV score. [Reported] [2]

### Serving, monitoring and trade-offs

79. How often do you rescore guests: batch daily, or on events? [Likely]
80. How do you monitor for drift after a large shift such as travel seasonality or a new market launch? [Reported: criterion] [1]
81. What would you trade off between interpretability for finance and accuracy for marketing? [Likely]
82. What privacy and fairness constraints apply when value scores affect who gets incentives? [Likely]

## Walkthrough D: Listing recommendation (similar listings, homepage, email, cover photo)

```
 Context: anchor listing (similar-listings carousel) OR guest home feed OR email
        |
        v
 [Candidate generation]  listing-embedding kNN (session co-click / booking skip-gram)
        |                + same market + available for searched dates + guest-intent embedding
        v
 [Ranker]  P(click) x P(book | click), with features: similarity, price delta, quality, distance, availability
        |
        v
 [Policy]  dedupe, diversity, exclude unavailable / low-trust, new-listing exploration
        |
        v
 Carousel / feed / email --> impressions, clicks, bookings --> retrain embeddings & ranker
```

Box labels to defend: the surface (each surface needs its own objective), the embedding training signal, availability at the guest's dates, and cold start for new listings.

### Clarify and objective

83. Design an Airbnb listing recommendation system. [Reported] [1][3][5]
    - Which surface: similar listings, homepage or email? Does the answer change by surface?
    - What is the success metric for a recommendation carousel that is not the main search?
84. Choose a listing's cover photo to increase bookings, including cold start for new listings. [Reported] [1]
    - How do you avoid a photo that drives clicks but disappoints guests at booking?
85. Find high-quality listings to recommend or highlight. [Reported] [1]
    - What does "quality" mean, over what time window, and which label would you trust: reviews, rebookings or human ratings? [1]

### Data, labels and features

86. How would you train listing embeddings, and what defines a positive pair? [Likely] [11][24]
    - Why might you use the booked listing as a global context in each session?
87. How do you handle the fact that most listings are booked rarely, so booking labels are sparse? [Likely] [11][24]
88. How do you represent a guest with no history, using only their current session? [Reported: criterion] [1]
89. Which features capture "similar" for a guest: price, location, room type, style, capacity? [Likely] [19]
90. For cover-photo selection, how would you use photo room-type classification and multimodal (image plus text) models as features? [Likely] [25][26]
91. How would you build themed listing categories (for example, treehouses or lakefront) with ML plus human review, and then recommend them? [Likely] [27]

### Model, evaluation and online test

92. Why use two stages (candidate generation, then ranking) rather than one model? [Likely]
93. Which offline metrics would you use: recall@k for retrieval, NDCG for ranking, coverage and diversity? [Likely]
94. How would you A/B test a new carousel when it may take bookings from the main search page? [Likely]
    - How would you measure net incremental bookings rather than bookings moved between surfaces?
95. A cover photo appears in search, recommendations and emails at once. How would you design the online test for a cover-photo change, and what is the randomisation unit? [Likely]
96. How would you give new listings exposure without hurting click-through? [Reported: criterion] [1]

### Serving, monitoring and trade-offs

97. How often do you refresh embeddings, and how do you keep old and new embedding versions compatible? [Likely]
98. What is your fallback when the nearest neighbours are unavailable on the guest's dates? [Likely]
99. How would you monitor for popularity bias, where the same listings dominate every carousel? [Likely]

## Walkthrough E: Relisting detection (newly reported prompt)

The MLE guide lists relisting detection as a reported prompt and says the round went into geohashing and into how the outputs show up in online metrics [1][3]. The question index also lists "Design Airbnb Relisting Detection" [2]. None of the sources defines "relisting", so clarifying it is part of the round.

```
 New or edited listing (photos, text, amenities, lat/long, host account)
        |
        v
 [Candidate pairs]  same / neighbouring geohash cells  +  near-duplicate photo hash  +  host-account links
        |
        v
 [Pair model]  P(same home or banned host returning | pair features)
        |        features: photo & text similarity, amenity overlap, geo distance, account / device / payment links
        v
 [Decision policy]  threshold by action: auto-merge | human review queue | allow
        |
        v
 Actions --> reviewer verdicts + appeals --> labels;  online metrics: guest issues, removed-listing reappearance
```

Box labels to defend: what counts as a relisting, the geohash cell size, the cost of a false positive on a legitimate host, and the online metric the system moves.

100. Design Airbnb relisting detection. [Reported] [1][2][3]
    - What counts as a relisting: the same physical home listed twice, a removed listing re-created, or a removed host returning under a new account?
101. How does geohashing help you generate candidate pairs, and what cell size would you use in a dense city versus a rural area? [Reported] [1][3]
    - What breaks at cell borders, and how do you fix it?
102. How do the outputs of a relisting system show up in online metrics, and which metric would you report? [Reported] [1]
103. Where do labels come from (trust-team investigations, reviewer verdicts, appeals), and how do you handle so few positives? [Likely]
104. Which pair features would you use: photo and text similarity, amenity overlap, geo distance, account and device links? [Likely] [25][31]
105. How would you set the threshold and size the human review queue, given the cost of wrongly blocking a legitimate host? [Likely]
106. Relisters change photos and shift the map pin to evade detection. How do you monitor for this and retrain? [Likely]

## Walkthrough F: Retrieval and ranking for generative-AI support (newly reported prompt)

The MLE guide says recent senior loops added ranking and retrieval prompts around generative-AI support tools, and it lists "Generate Customer-Support Instructions from Company Policies" (hard, Sep 2026) [1][2]. Chapter 10, Diagram 5 has a deeper support-routing diagram; this is the timed version.

```
 Support ticket / guest or host message
        |
        v
 [Intent + routing]  intent classifier --> topic, urgency, language
        |
        v
 [Retrieval]  policy & help-article index (hybrid: keyword + embedding) --> top-k passages
        |
        v
 [Re-ranker]  passage relevance x policy freshness x region applicability
        |
        v
 [Generator + guardrails]  instructions grounded in cited passages; refuse if no policy matches
        |
        v
 Agent or guest --> resolution, escalation, CSAT, agent edits --> labels for retriever & re-ranker
```

Box labels to defend: the retrieval label (which passage was actually correct), policy versioning by region, the grounding check, and the online metric (resolution without escalation).

107. Generate customer-support instructions from company policies. [Reported] [1][2]
    - How would you retrieve the right policy passages for a ticket, and how would you evaluate retrieval separately from generation?
108. Design the ranking of help articles or policy passages shown to a support agent or a support assistant. [Reported] [1]
    - What is the label: agent click, agent use in the reply, or ticket resolved?
109. How would you classify message intent and route a ticket before retrieval? [Likely] [28][29]
110. What guardrails stop the assistant from stating a policy that does not exist or does not apply in the guest's region, and how would you measure that offline and online? [Likely]
111. How would you A/B test a support assistant: unit of randomisation, primary metric (resolution, escalation, handle time, CSAT) and guardrails? [Likely] [29]

## Cross-cutting experimentation probes (asked in any walkthrough)

112. Design an A/B testing platform. [Reported] [2]
    - How does the platform handle assignment, logging, metric computation and guardrails?
    - How would you keep search and pricing experiments that run at the same time from colliding? [30]
113. Design an A/B test with causal inference. [Reported] [2]
    - What would you do when randomisation is impossible?
114. Design a network-aware experiment (the reported prompt was a Wi-Fi badge experiment). [Reported] [2]
    - How does a two-sided marketplace create spillover between treatment and control?
115. Compute control and treatment metrics from bookings and visits data, and say which denominator you would use. [Reported] [2]
    - The index lists this as a SQL question; in a design round, expect it as "show me how you would compute the metric".
116. Build a panel dataset and run a causal regression to estimate a treatment effect. [Reported] [2]
    - Which fixed effects would you include, and what assumption must hold for the estimate to be causal?
117. In a two-sided marketplace, when would you randomise by guest, by listing, by market or by time (switchback)? [Likely] [14]
118. How long would you run the test, and how do you handle novelty effects and weekly seasonality? [Likely]
119. How do you choose guardrails so that a ranking change can't quietly hurt hosts? [Likely] [20]
120. How do you reduce variance (for example, CUPED) so that a booking test reaches power sooner? [Likely]
121. How do you decide when an offline gain is large enough to justify an online test? [Reported: criterion] [1]

## Senior-level trade-off probes (final 5-7 minutes)

122. What would you ship in the first month, and what would you defer? [Likely]
123. What did you trade off to stay inside the latency budget? [Reported: criterion] [1]
124. Where would you push back on the business metric you were given? [Likely]
125. Which part of your design is most likely to fail in production, and how would you know? [Reported: criterion] [1]
126. How would you split this work across two engineers and a data scientist, and what would you own personally? [Reported: criterion] [1]
    - At G9 you are expected to own the area from framing to launch [1]. Show that.
127. If the interviewer narrows scope halfway through, how do you restate the plan and keep going? [Reported: criterion] [1]
128. Your past-project round turns into a design question: redesign the system you built for Airbnb's marketplace. What changes? [Reported] [5]
129. What would make you decide not to use ML here at all? [Likely]

## References

Status: **fetched** = read on 2026-10-14; **title confirmed** = title and URL match a fetched index ([7] or [21]), but the page itself was not opened (medium.com, arxiv.org and kdd.org could not be resolved); **unverified** = host unreachable and not found in a fetched index, listed from prior knowledge. No web searches could run during this review.

1. "Airbnb Machine Learning Engineer interview process" (TrueInterview-derived guide built from 3 candidate reports, mostly 2026 Q1-Q2; sample prompts, interviewer steering, scoring criteria, levels). kevin-2023-code/Tech-Interview-Questions, GitHub, 2026. Fetched. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/airbnb/interview-process-machine-learning-engineer.md
2. "Airbnb" question index, 78 questions with report dates (Rank Home Search Results Without a Text Query, Sep 2026; Generate Customer-Support Instructions from Company Policies, Sep 2026; Listing Lifetime Value — Estimation, May 2026; Design User Embedding Semantic Search, Apr 2026; Design an A/B Testing Platform, Apr 2026; Home Page — Search + Availability + Ranking, Mar 2026; undated: personalized search ranking, relisting detection, primary and guardrail metrics, A/B test with causal inference, network-aware Wi-Fi badge experiment, C/T metrics from bookings and visits, panel and causal regression). kevin-2023-code/Tech-Interview-Questions, GitHub, 2026. Fetched. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/airbnb.md
3. "Airbnb interview process" ("Problem and label before model"; one or two ML design rounds for MLEs; search ranking, property LTV, recommendations, relisting detection with geohashing). kevin-2023-code/Tech-Interview-Questions, GitHub, 2026. Fetched. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/airbnb/interview-process.md
4. 1point3acres post #40, English mirror (ML design: customer LTV; helping hosts set prices; deployment may be skipped for time). FlowSight/coding, GitHub. Fetched. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/40.md
5. 1point3acres post #79, English mirror (ML design: listing recommendation; price prediction; ML experience round turned into ML design). FlowSight/coding, GitHub. Fetched. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/79.md
6. Dirafzoon, "ML System Design" (nine-step template; lists "Rental recommendation (Airbnb)" as a sample prompt). alirezadir/AIMLInterviews, GitHub. Fetched. https://github.com/alirezadir/AIMLInterviews/blob/main/src/MLSD/ml-system-design.md
7. Yan, "applied-ml: Papers & tech blogs by companies sharing their work on data science & machine learning in production." eugeneyan/applied-ml, GitHub. Fetched. https://github.com/eugeneyan/applied-ml
8. Haldar et al., "Applying Deep Learning To Airbnb Search," arXiv, 2018. Title confirmed [7][21]. https://arxiv.org/abs/1810.09591
9. Haldar et al., "Improving Deep Learning For Airbnb Search," arXiv, 2020. Title confirmed [7]. https://arxiv.org/abs/2002.05515
10. "Machine Learning-Powered Search Ranking of Airbnb Experiences," Airbnb Tech Blog, 2019. Title confirmed [7]. https://medium.com/airbnb-engineering/machine-learning-powered-search-ranking-of-airbnb-experiences-110b4b1a0789
11. "Listing Embeddings in Search Ranking," Airbnb Tech Blog, 2018. Title confirmed [7]. https://medium.com/airbnb-engineering/listing-embeddings-for-similar-listing-recommendations-and-real-time-personalization-in-search-601172f7603e
12. Abdool et al., "Managing Diversity in Airbnb Search," arXiv, 2020. Title confirmed [7][21]. https://arxiv.org/abs/2004.02621
13. "Learning To Rank Diversely," Airbnb Tech Blog, 2022. Title confirmed [7]. https://medium.com/airbnb-engineering/learning-to-rank-diversely-add6b1929621
14. "Beyond A/B Test: Speeding up Airbnb Search Ranking Experimentation through Interleaving," Airbnb Tech Blog, 2022. Title confirmed [7]. https://medium.com/airbnb-engineering/beyond-a-b-test-speeding-up-airbnb-search-ranking-experimentation-through-interleaving-7087afa09c8e
15. Ye et al., "Customized Regression Model for Airbnb Dynamic Pricing," KDD 2018. Unverified. https://dl.acm.org/doi/10.1145/3219819.3219830
16. "How Airbnb Measures Future Value to Standardize Tradeoffs," Airbnb Tech Blog, 2021. Title confirmed [7]. https://medium.com/airbnb-engineering/how-airbnb-measures-future-value-to-standardize-tradeoffs-3aa99a941ba5
17. Tan et al., "Optimizing Airbnb Search Journey with Multi-task Learning," arXiv, 2023. Title confirmed [21]; arXiv URL unverified. https://arxiv.org/abs/2305.18431
18. "Artificial Counterfactual Estimation: Machine Learning-Based Causal Inference at Airbnb," Airbnb Tech Blog, 2022. Title confirmed [7]. https://medium.com/airbnb-engineering/artificial-counterfactual-estimation-ace-machine-learning-based-causal-inference-at-airbnb-ee32ee4d0512
19. "Using Machine Learning to Predict Value of Homes On Airbnb," Airbnb Tech Blog, 2017. Title confirmed [7]. https://medium.com/airbnb-engineering/using-machine-learning-to-predict-value-of-homes-on-airbnb-9272d3d4739d
20. "Designing Experimentation Guardrails," Airbnb Tech Blog, 2021. Title confirmed [7]. https://medium.com/airbnb-engineering/designing-experimentation-guardrails-ed6a976ec669
21. Wang, "Reco-papers: classic papers and resources on recommendation" (lists the Airbnb search, embedding, diversity and multi-task search-journey papers). wzhe06/Reco-papers, GitHub. Fetched. https://github.com/wzhe06/Reco-papers
22. "Chronon: A Data Platform for AI/ML" (point-in-time accurate backfills; online/offline consistency monitoring). airbnb/chronon, GitHub. Fetched. https://github.com/airbnb/chronon
23. "Airbnb Software Engineer interview process" (design rounds about an hour each without code; search, pricing and community-support teams). kevin-2023-code/Tech-Interview-Questions, GitHub, 2026. Fetched. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/airbnb/interview-process-software-engineer.md
24. Grbovic and Cheng, "Real-time Personalization using Embeddings for Search Ranking at Airbnb," KDD 2018. Title confirmed [7][21]. https://www.kdd.org/kdd2018/accepted-papers/view/real-time-personalization-using-embeddings-for-search-ranking-at-airbnb
25. "Categorizing Listing Photos at Airbnb," Airbnb Tech Blog, 2018. Title confirmed [7]. https://medium.com/airbnb-engineering/categorizing-listing-photos-at-airbnb-f9483f3ab7e3
26. "WIDeText: A Multimodal Deep Learning Framework," Airbnb Tech Blog, 2020. Title confirmed [7]. https://medium.com/airbnb-engineering/widetext-a-multimodal-deep-learning-framework-31ce2565880c
27. "Building Airbnb Categories with ML and Human-in-the-Loop," Airbnb Tech Blog, 2022. Title confirmed [7]. https://medium.com/airbnb-engineering/building-airbnb-categories-with-ml-and-human-in-the-loop-e97988e70ebb
28. "Discovering and Classifying In-app Message Intent at Airbnb," Airbnb Tech Blog, 2019. Title confirmed [7]. https://medium.com/airbnb-engineering/discovering-and-classifying-in-app-message-intent-at-airbnb-6a55f5400a0c
29. "Intelligent Automation Platform: Empowering Conversational AI and Beyond at Airbnb," Airbnb Tech Blog, 2022. Title confirmed [7]. https://medium.com/airbnb-engineering/intelligent-automation-platform-empowering-conversational-ai-and-beyond-at-airbnb-869c44833ff2
30. "Scaling Airbnb's Experimentation Platform," Airbnb Tech Blog, 2017. Title confirmed [7]. https://medium.com/airbnb-engineering/https-medium-com-jonathan-parks-scaling-erf-23fd17c91166
31. "Amenity Detection and Beyond — New Frontiers of Computer Vision at Airbnb," Airbnb Tech Blog, 2019. Title confirmed [7]. https://medium.com/airbnb-engineering/amenity-detection-and-beyond-new-frontiers-of-computer-vision-at-airbnb-144a4441b72e
