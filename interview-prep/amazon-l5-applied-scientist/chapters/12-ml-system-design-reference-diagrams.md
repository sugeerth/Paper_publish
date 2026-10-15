# 12. ML System Design Reference Diagrams: Six Canonical Amazon Prompts and a 45-Minute Whiteboard Template

> Scope: this chapter adds the diagrams that chapter 06 leaves out. It gives one labelled block diagram for each of six canonical Amazon Applied Scientist (L5) design prompts, a whiteboard template for drawing and narrating any of them in 45 minutes, and the questions an interviewer is likely to ask about each block. The diagrams are skeletons to draw from memory. The latency figures in them are illustrative budgets to defend or revise, not Amazon numbers. The question sections contain questions only, with no answers.
>
> Research note (please read): no new web research was possible for this chapter. The run's shared web-search budget was already used up before this chapter started, so every planned search was refused. Test fetches to aws.amazon.com and docs.aws.amazon.com failed DNS resolution in this environment. Every [Reported] tag below therefore comes from the sourced findings already recorded in chapter 06 (and through it, chapters 01-04) of this guide, and each one cites the original page. Those findings were built from search-result extracts, not full page reads. The AWS service references [23]-[29] are official product or documentation pages. They are cited only to name the components, were not re-fetched in this run, and are not interview reports. A follow-up session with a fresh search budget could add first-person design-round reports, especially from Blind, 1point3acres and Prepfully.
>
> Review pass (2026-10-14): an adversarial review tried to re-fetch about 20 of the cited URLs. Only GitHub-hosted pages could be reached. [20] (Chronos) and [21] (ESCI) were confirmed live, and their content matches how this chapter uses them. Every other host returned a DNS failure or a proxy 403. That includes amazon.jobs, amazon.science, aws.amazon.com, docs.aws.amazon.com, Glassdoor, Blind, Medium, LeetCode, Quora, 1point3acres, Dataford, IGotAnOffer, OA VO Service, IEEE Spectrum and About Amazon. Those references could be confirmed neither live nor dead, so they were kept as recorded in chapter 06. The web-search budget was also used up. The review made three kinds of change. (1) It added four new verified sources [37]-[40], all read in full from GitHub: two TrueInterview round guides that summarise 2026 candidate reports, the TrueInterview Amazon question index, and a compiled set of candidate posts. It also added about 25 new questions sourced from those pages or grounded in the diagrams. (2) It retagged the newsvendor question as [Likely], because its source is a supply-chain interview and not a science or ML one. It upgraded the offline-wins-online-neutral probe to [Reported]. (3) It removed four duplicate or generic items. Questions that the index [39] attributes to Amazon without naming the role are tagged [Likely] with a "(see [39])" pointer, not [Reported].

## How the [Reported] / [Likely] tag system works

Every question in this guide carries one of two tags:

- **[Reported] [n]**: a cited source (a candidate post on Glassdoor, Blind, LeetCode Discuss, Medium or 1point3acres, a prep-vendor write-up such as IGotAnOffer, Dataford or Interview Kickstart, or a round guide compiled from candidate reports such as TrueInterview) attributes this question, or a close paraphrase, to an Amazon science interview. An Amazon Machine Learning Engineer loop also counts, and the question is then marked "(MLE loop)". A question the source places above L5 (for example in a staff loop) is marked as such. The number points to the References list at the end of the chapter. A follow-up probe under a [Reported] question is not itself reported unless it carries its own [Reported] tag.
- **[Likely]**: a grounded inference from the team's products, from Amazon job postings or Amazon Science publications, or from the standard way ML system design rounds are probed. Nobody has been seen asking it verbatim. A "(see [n])" after a [Likely] tag points to the product or posting the inference is based on, not to an interview report.

What the tags do not mean:

- [Reported] does not mean the question is common. One report is enough for the tag.
- [Likely] does not mean the question is rare. Most block-level probes ("how do you detect drift?") are [Likely] because candidates rarely write them down, not because interviewers skip them.
- No tag was upgraded to [Reported] without a citation. Where this chapter could not trace a question to a source, it is [Likely].

## Format notes for this round

- **Duration and slot.** The design round is one of about four to six interviews of 55-60 minutes each in the L5 Applied Scientist loop. One 2025-2026 Applied Scientist II report lists "1 Sys Design (Team-specific)" next to breadth, depth, coding and Bar Raiser rounds [2]. Amazon's official prep page describes interviews of about 55 minutes with members of the science community, each also covering Leadership Principles [1]. Plan for roughly 45 minutes of design and 10-15 minutes of LP questions [1][16].
- **Interviewer.** Usually the hiring manager or a senior scientist on the hiring team, with a prompt from that team's domain [2][8]: Ads trust [9], Alexa speech [10], search and recommendations [3][12][13], or LLM agents and RAG [3][14].
- **L5 bar.** Blind commenters say the design or application round is typical for L5 and may be missing from L4 loops [5]. At L5 you are expected to own the system end to end: framing a vague business problem, choosing data, labels and models, picking offline and online metrics, and planning launch and monitoring, with sound trade-off calls along the way [5][16]. IGotAnOffer describes the round as proposing data sources, annotation strategy, modeling technique, evaluation metrics, and trade-offs and risks [7]. Candidates report being asked to walk the data pipeline "from collection to deployment" [9].
- **2026 reports.** A TrueInterview round guide built from 20 candidate reports (mostly 2026 Q1-Q3) describes the science application round as "an open-ended product problem" graded on framing, evaluation design and launch realism. It adds that "Ads and search teams ask for ranking pipelines instead" [37]. The matching MLE guide says prompts "follow the team's domain" and lists train-serve skew, drift, duplicate events and rollback paths as the failure modes candidates are expected to raise [38]. Several 2026 reports say Leadership Principle questions ate much of the hour, so rehearse a compressed version of the template [37].
- **Science over boxes.** The weight is on framing, labels, modeling and metrics rather than on distributed-systems detail. Candidates still report questions about scale, latency and deployment [9][22]. So the diagram exists to anchor the science discussion, not to replace it.

---

## Part 1. The whiteboard template (draw this first, for any prompt)

### 1.1 The universal skeleton

Draw the same seven zones every time, then fill them for the specific prompt. Left to right is the online path; the bottom band is offline; the arrow back from the right closes the loop.

```
 +--------------------------------------------------------------------------------+
 | (1) PROBLEM FRAME:  customer | decision being made | north-star | guardrails   |
 +--------------------------------------------------------------------------------+

  ONLINE PATH  (write the latency budget above each arrow, total p99 in the corner)
  ------------------------------------------------------------------------------
  [Client/Request] --> [(4a) Feature fetch] --> [(4b) Stage 1: recall / score]
                                                  --> [(4c) Stage 2: rank / decide]
                                                  --> [(4d) Policy / business rules]
                                                  --> [Response]  ----+
                                                                      |
  [(6) LOGGING: impressions, actions, model version, features served] <+
            |
  ----------|-------------------------------------------------------------------
  OFFLINE   v
  [(2) DATA SOURCES] --> [(3a) Labels + sampling] --> [(3b) Training + offline eval]
          |                                                  |
          v                                                  v
  [(3c) FEATURE STORE  offline <--same definitions--> online]   [Model registry]
                                                             |
                                                             v
                                         [(5) Deploy: shadow -> canary -> A/B]

  [(7) MONITORING: data drift | prediction drift | business KPI | latency/cost]
        --> retraining trigger --> back to (3b)
```

Zone checklist (say each heading out loud as you draw it):

1. Problem frame: customer, decision, north-star metric, guardrails, constraints (latency, cost, scale).
2. Data sources: logs, catalog, transactions, third-party or human labels.
3. Offline: label definition, sampling, training, offline evaluation, feature store, model registry.
4. Online serving: stages with a latency budget per stage.
5. Launch: shadow mode, canary, A/B test with guardrails.
6. Feedback and logging loop: what is logged, at what granularity, with which model version.
7. Monitoring and retraining triggers.

### 1.2 The 45-minute narration clock

```
 min 0-5    Clarify: who is the customer, what decision, what scale, what latency, what "good" means
 min 5-8    State north-star + 2 guardrails + offline proxy metric; write them in zone (1)
 min 8-12   Draw the empty skeleton (all 7 zones) before going deep anywhere
 min 12-20  Data + labels + sampling (zone 2-3a)          <- science depth expected here
 min 20-30  Models per stage, features, offline eval (3b, 4b-4c)
 min 30-35  Serving path with latency budget, AWS components (4, 3c)
 min 35-40  Launch, A/B design, feedback loop, monitoring + retrain triggers (5-7)
 min 40-45  Risks, v1 vs v2, what you would cut, questions for the interviewer
```

### 1.3 AWS component cheat-sheet (where each one goes on the board)

```
 Zone                 Component an interviewer expects you to be able to name
 -------------------  -------------------------------------------------------------
 Event ingestion      Amazon Kinesis Data Streams (clickstream, transactions) [25]
 Stream features      Kinesis -> Managed Service for Apache Flink / Lambda -> online store
 Data lake            Amazon S3 + AWS Glue catalog; Athena / EMR for batch features
 Feature store        SageMaker Feature Store (online + offline store) [24]
 Low-latency lookup   Amazon DynamoDB (user state, counters, precomputed candidates) [26]
 Vector / text index  Amazon OpenSearch Service (BM25 + k-NN vector search) [27]
 Training             SageMaker Training jobs, SageMaker Pipelines, Model Registry [23]
 Serving              SageMaker real-time endpoints / batch transform [23]
 Monitoring           SageMaker Model Monitor + CloudWatch alarms [23]
 LLM / RAG            Amazon Bedrock (foundation models, Knowledge Bases, Guardrails) [28]
 Managed recsys/fcst  Amazon Personalize [29]; Chronos open-source forecasting models [20]
```

### Questions on the template itself

1. Here is a high-level ML problem from our team. Propose a solution, covering data sources, annotation strategy, modeling techniques, evaluation metrics, and trade-offs and risks. [Reported] [7]
   - Which parts of the problem actually need ML, and which could be rules?
   - What is your v1 that could ship in one quarter?
2. Walk me through the full data pipeline for your design, from collection to deployment. [Reported] [9]
   - At which arrow on your diagram can training and serving features diverge?
3. Before you draw anything: what decision does this system make, and how often? [Likely]
   - Is it a batch decision (nightly) or a per-request decision, and how does that change the diagram?
4. What is the end-to-end p99 latency budget, and how have you split it across the stages you drew? [Likely]
   - Which stage would you cut first if the budget were halved?
5. Which boxes on your diagram are precomputed offline, and which must run per request? [Likely]
   - What runs on SageMaker real-time endpoints, and what runs as batch transform?
6. Which boxes would you build, and which would you reuse from existing AWS services or other Amazon teams? [Likely]
   - Why SageMaker Feature Store rather than DynamoDB alone for your features?
7. Point to where you log. What exactly is in a log record, and why do you need the served feature values rather than recomputing them? [Likely]
8. Here is a problem from a domain you know but are not expert in. Propose a product that solves it, then explain how you would collect data, which models you would compare, and how you would evaluate it with both engineering and business metrics. (AWS AI Applied Scientist loop) [Reported] [40]
9. Leadership Principle questions took the first 25 minutes, and you have about 30 minutes for the design. Which zones of the template do you compress or skip, and which must you still cover? [Likely] (see [37])
10. You are drawing the streaming feature path (Kinesis to a stream processor to the SageMaker Feature Store online store). How do you make sure the training set reads each feature exactly as it was at prediction time, with no future leakage? [Likely] (see [24][25])

---

## Part 2. Recommendations: candidate generation -> ranking -> re-ranking

### Diagram

```
 FRAME: homepage / detail-page widgets | north-star: downstream purchases (or GMS) per session
        guardrails: diversity, returns, latency p99 ~150-200 ms (illustrative)

 ONLINE
 [Request: customer_id, page, device]
     | ~10 ms
     v
 [Feature fetch]  DynamoDB / SageMaker Feature Store online:
     |            recent views, purchases, category affinities, session events
     | ~30 ms
     v
 [CANDIDATE GENERATION  (thousands, high recall)]
     |  - two-tower ANN (user emb -> item emb index, OpenSearch k-NN)
     |  - item-to-item co-view / co-purchase lists (precomputed, DynamoDB)
     |  - trending / popular in category (cold-start fallback)
     | ~50 ms
     v
 [RANKING  (hundreds -> ~50)]  GBDT or deep multi-task model
     |  heads: p(click), p(add-to-cart), p(purchase)   on SageMaker endpoint
     | ~30 ms
     v
 [RE-RANKING / POLICY]  diversity, dedupe variants, out-of-stock filter,
     |                  business rules, exploration slots
     v
 [Widget response] ---> impressions + clicks + purchases ---> Kinesis
                                                               |
 OFFLINE                                                       v
 [S3 data lake: impressions, actions, catalog, orders]  <-------+
     |
     +--> labels (view->click->purchase), negative sampling, position-bias correction
     +--> train retrieval towers (weekly) + ranker (daily)  [SageMaker Training]
     +--> rebuild item embedding index (OpenSearch) + co-purchase tables (DynamoDB)
     +--> offline eval: recall@K (retrieval), NDCG / AUC (ranker), replay/off-policy est.
     v
 [Model registry] -> shadow -> A/B (unit: customer)

 MONITOR: CTR / purchase rate by widget, coverage of catalog, embedding drift,
          popularity concentration -> retrain or re-index triggers
```

### Questions

11. Design a pipeline to improve personalized product recommendations on the Amazon homepage. [Reported] [6]
    - Draw candidate generation, ranking and re-ranking. Why three stages and not one model?
    - What is the north-star metric, and what are the guardrails?
12. Design a recommender system. [Reported] [15]
    - Two-tower retrieval, matrix factorization, or a sequence model for candidate generation: which, and why?
    - How do you handle cold start for new customers and new ASINs, and where on the diagram does that path sit?
13. A user goes from viewing to clicking to purchasing a product. Model the probability of each action given a user and a product. [Reported] [11]
    - Would you build three independent models, or model view, click and purchase jointly? [Reported] [11]
    - How would you build the dataset, choose features, and down-sample (for example with reservoir sampling)? [Reported] [11]
    - Purchases are only observed after clicks. How do you correct for that selection bias?
14. Design an agentic recommendation system for Amazon, covering component architecture, model choices, and quality and safety evaluation. [Reported] [6]
    - Which box in the classic three-stage diagram does the LLM agent replace, and which does it call as a tool?
15. How do you build the negatives for the two-tower model: random, in-batch, or hard negatives? [Likely]
    - How does the choice show up in recall@K offline versus in the ranking stage online?
16. How often do you rebuild the item embedding index, and how do new ASINs get into it between rebuilds? [Likely]
17. Which ranking features must be real-time (session events), and how do they reach the online feature store within seconds? [Likely]
    - What happens to ranking quality if the streaming path lags by 10 minutes?
18. What does the re-ranking stage optimize that the ranker cannot? [Likely]
    - How do you trade relevance against diversity and novelty on one page of widgets?
19. How do you evaluate a new ranker offline when your logs were generated by the old ranker? [Likely]
    - When would you use inverse propensity scoring or a replay estimator, and what logging does each require?
20. How do you avoid a feedback loop in which the system only recommends already-popular products? [Likely]
    - Where do exploration slots sit on your diagram, and what do they cost?
21. Design "Frequently bought together". Where does it sit in the diagram: candidate generator, ranker feature, or separate widget? [Likely]
22. Which monitoring signal would tell you the embedding model has gone stale before CTR drops? [Likely]
23. Predict the click probability for music recommendations. [Likely] (see [39]: reported at Amazon in a September 2026 phone screen, role not stated)
    - Which features and labels carry over from the product-widget diagram, and what changes for streaming media (skips, play duration, repeat plays)?
24. Design a "Popular products" or module-ranking layer for the homepage that decides which widgets appear and in what order. [Likely]
    - How do you compare a widget that drives clicks with one that drives purchases on a common scale?
25. When would you use a managed recommender such as Amazon Personalize instead of building the three-stage stack yourself? [Likely] (see [29])
    - What would you lose in control over the re-ranking and exploration stages?
26. The ESCI labels separate exact matches, substitutes and complements. How would you use the substitute-versus-complement distinction differently in a "similar items" widget and in a "frequently bought together" widget? [Likely] (see [21])

---

## Part 3. Product search: retrieval -> ranking

### Diagram

```
 FRAME: query -> ranked product list | north-star: purchases per search / search conversion
        guardrails: zero-result rate, relevance (human-judged), latency p99 ~200-300 ms (illustrative)

 ONLINE
 [Query "running shoes men size 10", customer, locale]
     | ~15 ms
     v
 [QUERY UNDERSTANDING]  spell-correct, product-type + attribute extraction,
     |                  intent classifier, query embedding
     | ~40 ms
     v
 [RETRIEVAL (hybrid, thousands)]   OpenSearch:
     |   lexical BM25 on title/attributes  +  k-NN on bi-encoder embeddings
     |   filters: locale, in-stock, eligibility
     | ~60 ms
     v
 [RANKING L1]  GBDT / light neural LTR on query-product features
     | ~50 ms
     v
 [RANKING L2]  cross-encoder or distilled transformer on top ~100
     |
     v
 [BLENDING]  organic results + sponsored slots + widgets; business rules
     v
 [SERP] ---> impressions (with position), clicks, adds, purchases, reformulations ---> Kinesis

 OFFLINE
 [S3: search logs, catalog, human relevance judgments (ESCI-style labels [21])]
     +--> click model / position-bias correction -> training labels
     +--> train bi-encoder (retrieval), LTR ranker, cross-encoder teacher -> distill
     +--> re-embed catalog -> OpenSearch index refresh; incremental updates for new listings
     +--> offline eval: recall@K, NDCG on judged sets, LLM-judge vs human agreement
     v
 [Registry] -> interleaving / A/B (unit: customer or query-session)

 MONITOR: zero-result rate, null-click rate, reformulation rate, index freshness lag,
          per-locale metrics -> retrain / re-index triggers
```

### Questions

27. You have a large collection of click data for many queries and product documents. Is click data alone sufficient? How would you train a retrieval model with the data at hand? [Reported] [12]
    - How do you correct for position bias in clicks? (see [19])
    - What other biases (presentation, trust, query intent) contaminate click labels?
28. Walk through how you would design, train, validate and deploy a ranking model for product search. [Reported] [13]
    - How would you estimate true relevance from clickstream logs with position bias? [Reported] [13]
    - Pointwise, pairwise or listwise learning-to-rank: which, and why?
    - Design the search retrieval and ranking pipeline including the online experiments and the position-bias correction, with candidate generation drawn explicitly. [Reported] [37][38]
29. Today the Amazon search bar only supports general retail products. How would you design it to also support Amazon Fresh grocery search? [Reported] [40]
    - What changes in retrieval and ranking when availability depends on the delivery zone and time slot?
    - Would you train one ranker across both catalogs or a separate one for Fresh, and how would you decide?
30. Design faceted product search at large scale, with brand, price and attribute filters and accurate facet counts. [Likely] (see [39]: reported at Amazon in an August 2026 onsite, role not stated)
    - Where do the attribute-extraction models that fill the facets sit on your diagram, and how do their errors show up to customers?
31. How would you turn the four ESCI labels (Exact, Substitute, Complement, Irrelevant) into training targets for the ranker? [Likely] (see [21])
    - Should a substitute rank above or below an exact match that is out of stock?
32. Compare bi-encoders and cross-encoders. Where would each sit in a product-search stack? [Reported] [3]
    - What is your latency budget per stage, and how do you distill the cross-encoder?
33. Why hybrid lexical plus vector retrieval rather than vector alone? [Likely]
    - Which queries does each one fail on (part numbers, brand names, vague intent)?
34. How do you keep the OpenSearch index fresh for new listings and for price and stock changes? [Likely]
    - Which updates can wait for a batch rebuild, and which must be incremental?
35. Draw the query-understanding block. How would you launch it in a new marketplace and language with little labeled data? [Likely]
36. How would you detect and reduce zero-result or low-quality-result searches? [Likely]
    - When should you relax the query, and when should you show "did you mean"?
37. How do you combine relevance with conversion, margin and sponsored placement in the blending box? [Likely]
    - How do you stop business boosts from eroding relevance over time?
38. Design an evaluation framework for search relevance using human annotation and LLM judges. [Likely]
    - How do you measure annotator agreement and calibrate an LLM judge against humans?
39. Interleaving or A/B test for a new ranker: when would you use each? [Likely]
40. How would you build semantic search over hundreds of millions of products within the latency budget you drew? [Likely]
    - What ANN index parameters trade recall against latency, and how do you pick them?

---

## Part 4. Sponsored ads: CTR prediction + auction

### Diagram

```
 FRAME: which ads to show, in which slot, at what price | north-star: long-run advertiser
        + shopper value (revenue subject to relevance/trust); guardrails: shopper CTR on organic,
        ad relevance, advertiser ROAS; latency p99 ~100 ms for the ads call (illustrative)

 ONLINE
 [Search / page context + shopper]
     | ~10 ms
     v
 [AD CANDIDATE RETRIEVAL]  keyword / product targeting match, semantic match (OpenSearch k-NN)
     |                     budget + eligibility filters (budget state in DynamoDB)
     | ~20 ms
     v
 [AD TRUST / POLICY FILTER]  creative moderation results (precomputed), brand safety
     | ~30 ms
     v
 [pCTR x pCVR MODELS]  calibrated, on SageMaker endpoint; features from online store
     |
     v
 [AUCTION]  rank by bid x pCTR (x quality); reserve price; pricing rule (GSP / first-price)
     |       pacing multiplier from budget-pacing service
     v
 [Ad slots in SERP / detail page] ---> impressions, clicks, spend events ---> Kinesis
                                                                              |
     +--> real-time spend aggregation -> DynamoDB budget state (pacing)  <----+
     +--> S3 logs -> conversions joined later (attribution window, delayed labels)

 OFFLINE
 [S3: impressions, clicks, delayed conversions, advertiser metadata, creatives]
     +--> label join with attribution window; delayed-feedback correction
     +--> train pCTR / pCVR (daily or incremental); calibration layer
     +--> offline eval: log-loss, AUC, calibration (predicted vs observed CTR by bucket)
     +--> auction simulator / counterfactual replay for reserve and pricing changes
     v
 [Registry] -> A/B (unit: shopper; budget-split or advertiser-side designs for marketplace effects)

 MONITOR: calibration drift, spend pacing error, advertiser ROAS, policy-violation rate
          -> recalibrate / retrain / pacing alerts
```

### Questions

41. Design a system that recommends trustworthy ads to Amazon users while keeping brand safety and compliance. [Reported] [9]
    - Walk through the full data pipeline from collection to deployment. [Reported] [9]
    - Where does the trust filter sit relative to pCTR scoring, and why there?
    - How do you get labels for rare policy violations?
42. Design click-through-rate prediction for Sponsored Products ads. [Likely] (see [18])
    - Why does calibration matter in the auction box, and how do you measure it?
    - How do you handle delayed conversions in pCVR training?
43. What breaks in your pCTR model when a new advertiser or ASIN has no history? [Likely]
44. Generalized second price or first price: how does the choice change what the model must output? [Likely]
45. Where does budget pacing sit in the diagram, and what happens if the spend stream lags? [Likely]
    - How would you frame pacing as constrained optimization or as a bandit?
46. Why is a plain shopper-level A/B test biased for auction changes, and what design fixes it? [Likely]
47. Revenue is flat, cost-per-click is up and conversions are down after a pacing change. Use your diagram to diagnose it. [Likely]
48. Design an ad-moderation classifier for images and text in creatives. Is it online or offline in your diagram? [Likely] (see [9])
    - How do you handle advertisers who iterate on rejected creatives?
49. How would you measure the incremental lift of an ad campaign for an advertiser? [Likely]
50. How do you set the number of ad slots on a page without hurting organic search quality? [Likely]
51. Design an evaluation framework that measures whether a text-to-image system spells a requested brand name correctly and places it naturally in a generated ad image. [Reported] [37][38]
    - Which checks can be automated, and which need a human review protocol?
    - How do you reconcile automatic scores with human ratings when they disagree?
52. Design an advertiser platform that records impressions, clicks and conversions and returns rates over custom time windows. (MLE loop) [Reported] [38]
    - Which of these aggregates would you also reuse as features for the pCTR model, and how do you keep the two definitions identical?
53. How would you evaluate adding video ads to the search results page? [Likely] (see [39]: reported at Amazon in a November 2025 phone screen, role not stated)
    - Which metrics guard the shopper experience, and how would you design the experiment when advertisers are also affected?

---

## Part 5. Fraud detection with streaming features

### Diagram

```
 FRAME: approve / challenge / block a transaction or account action | north-star: fraud loss
        prevented at a fixed customer-friction budget; guardrails: false-positive rate on good
        customers, review-queue capacity; latency p99 ~50-100 ms in the checkout path (illustrative)

 ONLINE
 [Checkout / login event]
     |
     +--> Kinesis Data Streams ---> stream processor (Flink / Lambda)
     |                               velocity counters: orders per card per 1h,
     |                               new-device-per-account per 24h, address changes
     |                               ---> online feature store / DynamoDB (seconds fresh)
     | ~10 ms
     v
 [Feature fetch]  streaming counters + batch features (account age, history)
     |            + graph features (shared device / card / address links, precomputed)
     | ~20 ms
     v
 [RISK MODEL]  GBDT on SageMaker endpoint (+ rules engine for known patterns)
     |
     v
 [DECISION POLICY]  thresholds by cost: approve | step-up auth | hold for review | block
     v
 [Outcome] ---> decision + score + features logged ---> Kinesis / S3
                     |
                     v
 [HUMAN REVIEW QUEUE]  prioritized by expected loss; investigator labels --> label store

 OFFLINE
 [S3: transactions, chargebacks (weeks late), investigator labels, account graph]
     +--> label maturity window; handle labels only seen for blocked/reviewed cases
     +--> train risk model (frequent), graph embeddings (batch)
     +--> offline eval: precision/recall at operating point, $ loss curves, by segment
     v
 [Registry] -> shadow scoring -> canary -> champion/challenger

 MONITOR: score distribution drift, approve/decline rates, chargeback rate (lagging),
          review-queue backlog, new-pattern alerts -> retrain / rule hot-fix triggers
```

### Questions

54. Design a fraud detection system for Amazon payments transactions. [Likely] (see [17])
    - How do you deal with extreme class imbalance and delayed chargeback labels?
    - Which of precision, recall and AUC would you report on imbalanced fraud data, and why? [Reported] [37]
    - How do you pick the operating threshold given the costs of false positives and false negatives?
55. Which features have to be streaming, and why can't they be batch? [Likely]
    - How do you compute a "1-hour velocity" counter identically in the stream and in the training data?
56. Given a stream of clicks with an ad ID and a timestamp, return a given ad's clicks in the last k minutes. (MLE loop, phone screen) [Reported] [38]
    - Now move that counter onto your diagram. Where does it run, how does it handle late or duplicate events, and how is the same window rebuilt for training data?
57. Design an end-to-end spam detection system. [Likely] (see [39]: reported at Amazon in a phone screen, role not stated)
    - Which boxes from the fraud diagram carry over unchanged, and which change when the "transaction" is a review, message or listing?
58. Where in your diagram can training/serving skew appear, and how would you detect it? [Likely]
59. You only observe true labels for transactions you approved. How does that bias training, and what would you do about it? [Likely]
    - Would you let a small random slice through for unbiased labels, and how would you justify the cost?
60. What do you do with the uncertain band: friction, step-up auth, or human review? [Likely]
61. How would you prioritize a human-investigator queue when you can review only 1% of flagged cases? [Likely]
    - How do investigator decisions feed back into training without biasing it?
62. How would you add graph signals (shared devices, cards, addresses) within the latency budget? [Likely]
63. How do you handle adversaries adapting to your model, and which monitor catches it first? [Likely]
64. Your fraud model's precision drops sharply over two weeks. Walk me through the investigation on your diagram. [Likely]
65. How would you detect fake reviews or abusive sellers with the same architecture? What changes? [Likely]
66. Design account-takeover detection at login time under a strict latency budget. [Likely]

---

## Part 6. SCOT demand forecasting

### Diagram

```
 FRAME: how many units of each product to buy, where, and when | north-star: in-stock
        at minimum holding + shortage cost; guardrails: forecast bias, excess inventory;
        batch system: daily/weekly run, SLA in hours, not milliseconds

 DATA SOURCES
 [Sales history (censored by stockouts)] [Prices, promotions, Prime Day calendar]
 [Catalog attributes / embeddings]        [Glance views / search demand signals]
 [Holidays, weather, macro]               [Lead times, vendor constraints]
     |
     v
 [S3 data lake + Glue] --> [FEATURE PIPELINE (EMR / Spark, SageMaker Processing)]
     |                      demand de-censoring, hierarchy (item -> category -> region)
     v
 [TRAINING]  global probabilistic model across millions of series
     |       (quantile / distribution outputs; deep global models, Chronos-style
     |        pretrained models [20]); cold-start from attributes of similar items
     v
 [BACKTEST]  rolling-origin evaluation: quantile loss (P50/P90), bias, by segment
     v
 [BATCH INFERENCE]  SageMaker batch transform / Processing: quantile forecasts per
     |               item x location x week, horizon of weeks to months
     v
 [FORECAST STORE (S3 / DynamoDB for lookup)]
     |
     v
 [DOWNSTREAM OPTIMIZATION]  buying (newsvendor-style quantile choice), placement
     |                      across fulfillment centers, transportation
     v
 [Purchase orders]  ---> realized sales, stockouts, overrides ---> back to data sources

 MONITOR: forecast error vs realized demand by segment, bias drift, in-stock rate,
          holding cost, manual overrides -> retrain / recalibrate triggers
          (A/B is hard: shared warehouses and trucks create interference)
```

### Questions

67. How would you forecast demand for millions of products, including new ones with no history? [Likely] (see [32][33][34])
    - Why use probabilistic or quantile forecasts rather than point forecasts for buying decisions?
    - Would you use a global model or per-item models, and why?
68. What is the newsvendor problem, and how would you use it to set inventory levels? [Likely] (see [36]: reported in Amazon supply-chain interviews, not a science loop)
    - How does a quantile forecast map to the newsvendor critical ratio?
69. How would you model an optimization problem from your work, including objective, decision variables and constraints? [Reported] [16]
    - When would you use a heuristic instead of an exact solver?
70. Sales are censored when items are out of stock. Where in your pipeline do you correct for that? [Likely]
71. How would you forecast demand during Prime Day or other promotions, when history is sparse and distorted? [Likely]
72. How do you design the backtest so that it does not leak future information? [Likely]
73. How do forecast errors propagate into placement and transportation costs? [Likely] (see [33])
74. How do you evaluate the forecasting system against in-stock rate and holding cost rather than MAPE alone? [Likely]
    - How would you run an experiment on a buying policy when products share warehouses and trucks?
75. What triggers a retrain in a batch forecasting system, and how do you guard against a bad model reaching purchase orders? [Likely]
76. Buyers override forecasts by hand. How do you use the overrides, and how do you measure whether they help? [Likely]
77. Design a system that decides how much of each product to buy from which vendor and when. [Likely] (see [34])
78. Chronos-style pretrained models forecast zero-shot, and newer versions accept covariates. When would you use one as the cold-start or fallback forecaster in your diagram, and when would you train a global model on Amazon data instead? [Likely] (see [20])
79. Design an end-to-end regression system for demand in a domain with strong weather and calendar effects (energy demand was the reported example). Then map it onto the SCOT diagram. [Likely] (see [39]: reported at Amazon in an onsite, role not stated)
80. Build a package-allocation model for couriers, then prove that the new allocation beats the current manual baseline. [Likely] (see [39]: both reported at Amazon in onsites, role not stated)
    - Why is a simple A/B split of packages biased here, and which design (switchback, geographic or time clusters) would you use?
81. Inventory counts reach the forecasting system from a distribution-center inventory update service. What happens to your forecasts and buying decisions when those updates arrive late or are duplicated? [Likely] (see [39])

---

## Part 7. Rufus / Alexa-style RAG assistant

### Diagram

```
 FRAME: answer shopping questions grounded in catalog, reviews, Q&A | north-star: task
        success / downstream purchase without regret (returns); guardrails: factuality,
        safety, latency (time-to-first-token ~1 s, illustrative), cost per query

 ONLINE
 [User turn "What do I need for cold weather golf?" + conversation state + customer context]
     | ~50 ms
     v
 [INTENT / ROUTER]  small classifier or LLM: product search | product Q&A | comparison |
     |              order status (tool) | out-of-scope
     | ~100-200 ms
     v
 [RETRIEVAL]  query rewrite -> hybrid search in OpenSearch (BM25 + k-NN embeddings)
     |        over catalog attributes, review snippets, Q&A; product search API as a tool
     |        re-rank top-k with cross-encoder
     | ~500 ms+ (streaming)
     v
 [GENERATION]  LLM via Amazon Bedrock or a self-hosted fine-tuned model on SageMaker;
     |         prompt with retrieved evidence + citations to ASINs
     v
 [GUARDRAILS]  grounding check vs evidence, safety / policy filter (Bedrock Guardrails),
     |         price / availability facts taken from live systems, not the model
     v
 [Streamed answer + product cards] ---> thumbs, clicks, adds, purchases, follow-up turns
                                         ---> logs (with retrieved doc IDs + prompt version)

 OFFLINE
 [Catalog + reviews + Q&A] --> chunk + embed --> OpenSearch vector index (incremental refresh)
 [Logs + human annotations + LLM-judge labels]
     +--> eval sets: retrieval recall, answer faithfulness, helpfulness, safety red-team
     +--> fine-tuning (SFT, preference tuning) / distillation to a smaller serving model
     v
 [Registry: model + prompt + retriever versions] -> offline eval gate -> A/B

 MONITOR: hallucination rate (sampled judge), refusal rate, latency / cost per turn,
          retrieval hit rate, safety incidents -> prompt fix / re-index / retrain triggers
```

### Questions

82. Design an enterprise RAG service. [Reported] [14]
    - How do you chunk, embed and index heterogeneous documents?
    - How do you enforce document-level access control in retrieval?
    - How do you evaluate retrieval quality separately from generation quality?
83. Evaluate an LLM agent with metrics beyond accuracy, such as tool-use correctness, hallucination and calibration. [Reported] [3]
    - How do you build an offline evaluation set for multi-step agent tasks?
    - When would you trust an LLM-as-judge?
84. Pick an agentic product we could build. Who is the customer, and how do you measure success? [Reported] [2]
    - Propose agentic AI solutions for a hypothetical project and defend their viability. [Reported] [37]
85. Architect an AI chatbot, covering intent detection, dialogue state, retrieval from a knowledge base, generation, fallback behaviour, evaluation and continuous learning. [Reported] [37][38]
    - Where in the Rufus diagram does dialogue state live, and what does the router do when it is uncertain?
    - What is the fallback when retrieval returns nothing relevant: refuse, ask a clarifying question, or hand off?
86. Design a multi-turn search agent, and explain how it handles long conversation context. [Reported] [37][38]
    - What do you keep verbatim, what do you summarize, and what do you re-retrieve on each turn?
    - How would you evaluate a whole multi-turn session rather than single answers?
87. Design an LLM quality validation system that checks assistant answers before and after launch. [Likely] (see [39]: reported at Amazon in an onsite, role not stated)
88. How would you measure product-market fit for a voice or chat shopping assistant such as Alexa Shopping? [Likely] (see [39]: reported at Amazon in an onsite, role not stated)
    - Which behaviour separates retained, satisfied users from novelty use?
89. Bedrock Knowledge Bases or your own chunking, embedding and OpenSearch retrieval pipeline: what would push you to build rather than reuse? [Likely] (see [27][28])
90. Design a distributed pipeline that ingests, tokenizes, deduplicates and filters the training data for the assistant's fine-tuning. (staff-level loop; above the L5 bar, so treat it as a stretch) [Reported] [37][38]
91. On an LLM team, discuss LLM decoding and training choices in the context of an Amazon use case. [Reported] [22]
    - Greedy, beam or sampling decoding for a customer-facing assistant: which, and why?
92. How would you improve Alexa speech recognition for non-native speakers and users with strong regional accents? [Reported] [10]
    - Where would you get diverse accent training data? [Reported] [10]
    - Where does ASR sit in front of the RAG diagram, and how do its errors propagate into retrieval?
93. Design Rufus: answer "What do I need for cold weather golf?" from the catalog, reviews and Q&A. [Likely] (see [35][30][31])
    - How do you ground answers in product facts and avoid hallucinated specs?
    - How do you mix organic and sponsored products in a generative answer?
94. Why does the router exist? What would break if every turn went to the large model? [Likely]
95. Which facts must come from live systems (price, stock, delivery date) rather than from the index or the model? [Likely]
96. How do you keep the vector index fresh as reviews and listings change? [Likely]
97. Bedrock-hosted model or a fine-tuned model on SageMaker: how do you decide? [Likely]
    - How would you serve the assistant to millions of users at acceptable cost: distillation, quantization, caching, speculative decoding, or model routing?
98. What is your time-to-first-token budget, and which stage consumes most of it? [Likely]
99. How would you summarize thousands of reviews into a short "Customers say" blurb, and where does that job sit in the diagram (online or offline)? [Likely]
    - How do you stop manipulated reviews from steering the summary?
100. How would you detect and mitigate unsafe, biased or policy-violating responses? [Likely]
     - Which red-teaming process and which guardrail models would you use?
101. What does your feedback loop log, and how do you turn thumbs-down and follow-up turns into training data? [Likely]
102. How do you version prompts, retrievers and models together so a regression can be rolled back? [Likely]

---

## Part 8. Cross-cutting probes on any diagram (feedback, monitoring, retraining)

103. How do you monitor this model in production, and how do you detect data or concept drift? [Likely]
     - Where on the diagram does the retraining trigger fire, and what threshold fires it?
     - How do you validate the retrained model before rollout?
104. How would you roll back safely if the new model misbehaves on a subset of traffic? [Likely]
105. Your labels arrive weeks late (returns, chargebacks, retention). How do you train and evaluate under label delay? [Likely]
106. How would you design an A/B test for a model launch? [Reported] [4]
     - What do you do if the test is underpowered or the metric is heavy-tailed?
107. Your model's online A/B result fell short of the offline gain. How do you find out why? [Reported] [37]
     - Which arrows on your diagram would you inspect first?
108. How would you cut the cost of this system by 50% with minimal quality loss? [Likely] (Frugality)
109. Where does personally identifiable information flow in your diagram, and how do you minimize it? [Likely]
110. Design a causal study to measure the impact of a customer reminder or notification that your model triggers, when you cannot randomize everyone. [Likely] (see [39]: reported at Amazon in a phone screen, role not stated)
111. Which monitoring alarm would page you at 3 a.m., and which would only go on a weekly dashboard? [Likely]

## Part 9. L5-bar probes inside the design round

112. Of everything on your diagram, what is the single riskiest assumption, and how would you de-risk it in two weeks? [Likely]
113. A product manager wants to launch in a month and your offline evaluation is incomplete. What do you do? [Likely] (Bias for Action versus Insist on the Highest Standards)
114. Your design improves conversion but increases returns. How do you decide whether to ship? [Likely]
115. Which metric would you refuse to optimize directly, and why? [Likely]
116. If you had 10x the data, or one-tenth of it, which boxes on the diagram would change? [Likely]
117. How would you explain this diagram to a non-scientist stakeholder in two minutes? [Likely] (Earn Trust)
118. How would you hand this system over to an engineering team for long-term ownership? [Likely]

---

## References

Interview reports and prep guides [1]-[22] (all previously recorded in chapter 06 of this guide). The 2026-10-14 review tried to re-fetch them. Every host failed DNS or was refused by the egress proxy, so none could be confirmed live or dead. [20] and [21] are on GitHub and were confirmed live.

1. Applied Scientist Interview Prep, amazon.jobs (Amazon, official), accessed 2026. https://amazon.jobs/content/en/how-we-hire/applied-scientist-interview-prep
2. Amazon Applied Scientist II Interview Experience & Questions, Glassdoor, 2025-2026 (includes the "1 Sys Design (Team-specific)" loop report and the agentic-product prompt). https://www.glassdoor.com/Interview/Amazon-Applied-Scientist-II-Interview-Questions-EI_IE6036.0,6_KO7,27.htm
3. Amazon Applied Scientist Interview Experience & Questions, Glassdoor, 2025-2026 (LLM agent evaluation, bi-encoder vs cross-encoder). https://www.glassdoor.com/Interview/Amazon-Applied-Scientist-Interview-Questions-EI_IE6036.0,6_KO7,24.htm
4. Amazon applied scientist interview-onsite, Blind, circa 2024. https://www.teamblind.com/post/amazon-applied-scientist-interview-onsite-t01hpnc4
5. Amazon applied scientist interview loops, Blind. https://www.teamblind.com/post/amazon-applied-scientist-interview-loops-lucdk0uq
6. Amazon Applied Scientist Interview Questions & Guide 2026, Dataford (prep vendor), 2026. https://dataford.io/interview-guides/amazon/applied-scientist
7. Amazon Applied Scientist Interview, IGotAnOffer (prep vendor), 2025-2026. https://igotanoffer.com/en/advice/amazon-applied-scientist-interview
8. The 2026 Amazon Applied Scientist Interview Guide, Interview Kickstart (prep vendor), 2026. https://interviewkickstart.com/interview-questions/companies/amazon/applied-scientist
9. "My interview experience for the Applied Scientist II role at Amazon" (Advertising Trust), Chaitanya Mundle, Medium, circa 2024. https://medium.com/@mundle1997.cm/my-interview-experience-for-the-applied-scientist-ii-role-at-amazon-738aba21f611
10. "Science application: Business problems for Amazon Applied Scientist (Alexa International Tech)", Medium, 2025. https://medium.com/@pragyashukla2580/science-application-business-problems-for-amazon-applied-scientist-alexa-international-tech-9353a52ca556
11. "Interview Experience for Applied Scientist at Amazon", Akanksha Paul, Medium, circa 2020. https://akanksha-paul.medium.com/interview-experience-for-applied-scientist-at-amazon-5ff5003375dd
12. "Amazon Applied Scientist | L6 Luxembourg | Passed, freeze | 2022", LeetCode Discuss, 2022. https://leetcode.com/discuss/interview-experience/4421594/
13. "What are some Amazon Research Scientist interview questions?", Quora. https://www.quora.com/What-are-some-Amazon-Research-Scientist-interview-questions
14. "Amazon Onsite Interview Experience for Fulltime Machine Learning Role" (enterprise RAG design), 1point3acres. https://www.1point3acres.com/interview/thread/1157086
15. "Interview Experience Amazon Applied Scientist" (recommender design round), Blind. https://www.teamblind.com/post/Interview-Experience-Amazon-Applied-Scientist-hWgaGsmO
16. Amazon Applied Scientist Interview Debrief: Five-Round Loop + LP, OA VO Service (vendor-published candidate debrief; promotional). https://oavoservice.com/en/articles/amazon-applied-scientist-vo-loop-lp-ml-coding
17. Applied Scientist II, Amazon Payment Products (L5) (Job ID 3168758), amazon.jobs (posting may have expired). https://www.amazon.jobs/en/jobs/3168758/applied-scientist-ii-amazon-payment-products-l5
18. "Applied scientist interview at Amazon Ads", Blind. https://www.teamblind.com/post/applied-scientist-interview-at-amazon-ads-zoerec5t
19. "Intent based relevance estimation from click logs", Amazon Science. https://www.amazon.science/publications/intent-based-relevance-estimation-from-click-logs
20. Chronos: pretrained time-series forecasting models, amazon-science, GitHub. https://github.com/amazon-science/chronos-forecasting
21. Shopping Queries Dataset (ESCI), amazon-science, GitHub. https://github.com/amazon-science/esci-data
22. Amazon interview review RVW96068026 (Applied Scientist II, LLM team, Santa Clara), Glassdoor, 2025. https://www.glassdoor.com.hk/Interview/Amazon-Interview-E6036-RVW96068026.htm

AWS components (official product and documentation pages, cited only to name components; not interview reports; not re-fetched in this run):

23. Amazon SageMaker (training, endpoints, Pipelines, Model Registry, Model Monitor), AWS. https://aws.amazon.com/sagemaker/
24. Amazon SageMaker Feature Store, AWS documentation. https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store.html
25. Amazon Kinesis Data Streams, AWS. https://aws.amazon.com/kinesis/data-streams/
26. Amazon DynamoDB, AWS. https://aws.amazon.com/dynamodb/
27. Amazon OpenSearch Service, AWS. https://aws.amazon.com/opensearch-service/
28. Amazon Bedrock, AWS. https://aws.amazon.com/bedrock/
29. Amazon Personalize, AWS. https://aws.amazon.com/personalize/

Product, science and team context (basis for [Likely] inferences):

30. "What do I need for cold weather golf?" (Amazon Rufus), IEEE Spectrum. https://spectrum.ieee.org/amazon-rufus
31. "Amazon Rufus AI assistant personalized shopping features", About Amazon. https://aboutamazon.com/news/retail/amazon-rufus-ai-assistant-personalized-shopping-features
32. "Predicting what Amazon's customers will need tomorrow, next week, and beyond", Amazon Science. https://www.amazon.science/working-at-amazon/predicting-what-amazons-customers-will-need-tomorrow-next-week-and-beyond
33. "reMARS revisited: Amazon's supply chain optimization", Amazon Science. https://www.amazon.science/latest-news/remars-revisited-amazons-supply-chain-optimization
34. Applied Scientist II, SCOT Optimal Sourcing Systems, Buying Science Team (Job ID 10468056), amazon.jobs (posting may have expired). https://www.amazon.jobs/en/jobs/10468056/applied-scientist-ii-scot-optimal-sourcing-systems-buying-science-team
35. Applied Scientist, Rufus Experiences Science (Job ID 2855826), amazon.jobs (posting may have expired). https://amazon.jobs/en/jobs/2855826/applied-scientist-rufus-experiences-science
36. Amazon Supply Chain Interview Experience & Questions (newsvendor problem; a supply-chain role, not a science loop), Glassdoor. https://www.glassdoor.com/Interview/Amazon-Supply-Chain-Interview-Questions-EI_IE6036.0,6_KO7,19.htm

Added in the 2026-10-14 review (each read in full from GitHub):

37. "Amazon Research Scientist Interview Process", TrueInterview, mirrored in kevin-2023-code/Tech-Interview-Questions on GitHub (CC BY 4.0). Built from 20 candidate-reported experiences, mostly 2026 Q1-Q3, which the page says come mostly from Applied Scientist loops. Covers science application prompts (agentic proposals, multi-turn search agent, AI chatbot, brand-name image evaluation, ranking pipelines), the online-versus-offline A/B probe, and fraud metrics in ads phone screens. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/amazon/interview-process-research-scientist.md
38. "Amazon Machine Learning Engineer Interview Process", TrueInterview, same repository, 2026. Covers the ML system design round (search retrieval and ranking with position bias, AI chatbot, brand-name image evaluation, staff-level training-data pipeline), the advertiser-metrics platform and the sliding-window ad-click counter. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/amazon/interview-process-machine-learning-engineer.md
39. TrueInterview question index (data/questions.jsonl), same repository, accessed 2026-10-14. Gives the question title, company, round and reported date. It does not give the role, so questions taken from it are tagged [Likely]. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/data/questions.jsonl
40. goldin2008/Note, "interview_ds.md", GitHub. A compiled set of candidate posts, mostly in Chinese and undated. Its "Amazon AS" section includes the Amazon Fresh search application question and an AWS AI Applied Scientist "Machine Learning Design" round report. https://github.com/goldin2008/Note/blob/HEAD/interview_ds.md
