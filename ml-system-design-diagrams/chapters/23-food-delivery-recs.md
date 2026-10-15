# 23. Design Restaurant / Food Recommendations and Store Ranking (DoorDash, Uber Eats)

> Prompt as usually given: "Design the home feed for a food-delivery app. Given a user who
> opens the app at a location and time, rank the restaurants (stores) and dishes to show."
> Variants: rank stores in a cuisine filter, "dishes you may like" carousel, ranking after
> a search query, a carousel-of-carousels homepage, cold start for new users or new stores,
> and grocery item substitutions.
>
> Sourcing note: in both the compile and review passes, web search was used up and only
> GitHub-hosted pages could be read (uber.com, doordash.engineering and arxiv.org were
> blocked). In review, every (list) title, year and URL was re-checked against [10][11];
> [15][16][17] were re-read. Labels used below:
> (fetched) means the page was read for this chapter. (list) means the exact title, year and
> URL were taken from a fetched curated list ([10][11]); those posts could not be opened,
> so the details given for them are the author's summary of the published system.
> (ch-ref) means the source was already verified in an earlier chapter. (known) means a
> canonical paper cited by its arXiv ID or DOI. No portal page that reports this exact
> question at a named company could be opened, so treat the company tags as indicative.

## Company tags

| Company | What is reported / why it is tagged | Source |
|---|---|---|
| DoorDash | Publishes the matching production system: store feed with embeddings, homepage explore/exploit, cuisine filter. Khang Pham's "Food delivery" list links the store-feed post | [4][5][6][15] |
| Uber (Uber Eats) | Publishes the "Food Discovery with Uber Eats" series: marketplace recs, query understanding, graph learning. Khang Pham links the query-understanding and trip-optimization posts | [1][2][3][15][41] |
| Instacart (and DoorDash grocery) | AIMLInterviews lists "Replacement product recommendation" (Instacart) as a question. Contextual-bandit and substitution posts | [16][30][32] |
| Swiggy, Delivery Hero, Grubhub | Publish restaurant learning-to-rank, dish-search ranking, cold-start ranking and query embeddings. Not confirmed as interview prompts | [12][13][14][31][39] |
| Zomato, Deliveroo, Grab, Meituan | Not tagged: same marketplace pattern, but no interview report or ranking post found in [10][11] | none |

**Portals that cover it**

| Portal | Coverage | Link |
|---|---|---|
| Khang Pham (khangich, Educative "ML System Design") | "Food delivery" reading list: Uber Eats query understanding, DoorDash personalized store feed, dispatch | [15] |
| alirezadir AIMLInterviews | 9-step MLSD formula. Food-related prompts are "Food delivery time approximation" and Instacart "Replacement product recommendation"; the recsys template is reused | [16] |
| ByteByteGo (Aminian & Xu) | No food chapter. Closest: "Event Recommendation" (location-aware, LTR) and "Video Recommendation" (multi-stage) | [18][19][20] |
| Educative "Grokking the ML Interview" | Recsys module (candidate gen, then ranker). Same funnel | [21] |
| Hello Interview | Recommendation-systems pattern (retrieve, rank, re-rank) | [22] |
| Exponent | MLSD course. Framework applies | [23] |
| Chip Huyen exercises | No food prompt (re-checked). Ex. 3 (replacement items) is the closest | [17] |
| Curated lists (eugeneyan applied-ml, Engineer1999) | Index the DoorDash, Uber Eats, Swiggy and Delivery Hero posts used here | [10][11] |

## Requirements & scale

| Dimension | Value | Source |
|---|---|---|
| Surfaces | Home feed (vertical store list plus horizontal carousels), cuisine filter, dish carousel, post-search ranking | [5][6][1][7] |
| Functional | Return about 50-200 ranked stores, all open and deliverable to the address. Group them into carousels. Show an ETA and fee on each | assumption |
| Hard constraints | Delivery radius, store open now, the store's kitchen not overloaded, ETA under a cap | [2] |
| Objectives | Conversion (order), GMV, retention. Marketplace balance across eaters, merchants and couriers. Exposure for new stores | [2][5] |
| Users | 10^7-10^8 monthly active. Many low-frequency users. New users need cold-start ranking | assumption, [13] |
| Catalog | 10^5-10^6 stores in total, but only about 10^2-10^4 are eligible at one location | assumption |
| QPS | 10^4-10^5 feed requests/s at the lunch and dinner peaks (sharp time-of-day spikes) | assumption |
| Latency | Under 200 ms p99 for the feed. Ranking under 50 ms | assumption |
| Freshness | Store open/busy status and ETA in real time (seconds). Embeddings daily | assumption |
| Key difference from e-commerce recs | Retrieval is geo-filtering, not ANN over the whole catalog. Supply-side state (prep time, couriers) changes the ranking | [2][5] |

## Diagram 1: End-to-end architecture (online serving path)

**Figure 23.1: Online serving path. Geo-eligibility retrieval, then real-time filters, a
multi-task ranker and a marketplace-aware re-ranker that builds the carousel layout.
Event stream and online store follow DoorDash's Kafka/Flink and Redis posts [33][34][35].**

```
 +-------------+ (1) user_id, lat/lng, time, filters  +---------------------------+
 |  Eater app  |------------------------------------->|  Feed API / BFF           |
 |             |<-------------------------------------|  (orchestrates stages)    |
 +-------------+ (7) carousels + list, ETA, fee       +---------------------------+
        |                                                           | (2) context
        | impressions, clicks, orders                               v
        v                                 +----------------------------------------+
 +-------------+                          | CANDIDATE GENERATION (union ~10^3)     |
 | Event log   |                          |  geo index: H3 cells in radius         |
 | Kafka->Flink|                          |  + reorder / past stores               |
 +-------------+                          |  + embedding sim (store2vec / GNN)     |
        |                                 |  + popular-in-cell + new-store explore |
        | real-time aggregates            +----------------------------------------+
        v                                                           | (3)
 +-----------------+                      +----------------------------------------+
 | Feature store   |                      | FILTERS (real time)                    |
 | (Redis online)  |                      |  open now, deliverable, ETA < cap,     |
 | user/store emb, |                      |  kitchen busy, dietary / active filter |
 | ETA, prep load  |                      +----------------------------------------+
 +-----------------+                                                | (4) ~10^2-10^3
        |              features                                     v
        +-------------------------------->+----------------------------------------+
                                          | RANKER  multi-task DNN / GBDT          |
 +-----------------+  model weights       |  p(click), p(order|click), E[GMV],     |
 | Model registry  |--------------------->|  p(bad experience)                     |
 +-----------------+                      +----------------------------------------+
                                                                    | (5) scores
                                                                    v
                                          +----------------------------------------+
                                          | RE-RANK / POLICY                       |
                                          |  value = w1*pCVR + w2*GMV - w3*ETA     |
                                          |  + UCB exploration bonus (new stores)  |
                                          |  + cuisine diversity, ads/promo slots  |
                                          |  + carousel assembly (rows x slots)    |
                                          +----------------------------------------+
                                                                    | (6) layout
                                                                    +--> Feed API
```

## Diagram 2: Offline training & data pipeline

**Figure 23.2: Offline pipeline. Impression logs are joined with orders and delivery
outcomes to build labels. Graph/session embeddings and the ranker are retrained daily.**

```
 +-------------+   +-------------+   +--------------+   +-------------------+
 | Impressions |   | Clicks /    |   | Orders,      |   | Store state logs  |
 | (position,  |   | store-page  |   | ratings,     |   | ETA, prep time,   |
 |  carousel)  |   | views       |   | cancels      |   | open/close, menus |
 +-------------+   +-------------+   +--------------+   +-------------------+
        |                 |                 |                     |
        +-----------------+--------+--------+---------------------+
                                   |
                                   v
               +------------------------------------------+
               | Point-in-time join on request_id/session |
               +------------------------------------------+
                     |                              |
                     v                              v
       +----------------------+       +---------------------------+
       | Label builder        |       | Graph / session builder   |
       | click, order (<=24h),|       | user-store, user-dish     |
       | GMV, bad-exp flags   |       | bipartite; store sessions |
       +----------------------+       +---------------------------+
                 |                                  |
                 |                                  v
                 |                    +---------------------------+
                 |                    | Embedding training        |
                 |                    | GraphSAGE / store2vec     |
                 |                    +---------------------------+
                 v                                  | daily embeddings
       +----------------------+                     | (also pushed online)
       | Feature backfill     |<--------------------+
       | (offline store,      |
       |  point-in-time)      |
       +----------------------+
                 | training set (neg-sampling, position debias)
                 v
       +----------------------+  +------------------+  +-----------------------+
       | Ranker training      |->| Offline eval     |->| Registry, shadow,     |
       | multi-task DNN/GBDT  |  | AUC, NDCG@k, by  |  | A/B or switchback     |
       +----------------------+  | city / new users |  | rollout               |
                 ^               +------------------+  +-----------------------+
                 |                                                |
                 +------ feedback loop: new impressions + orders <+
                         (explore traffic logged with propensity)
```

## Diagram 3: Model architecture (multi-task store ranker with graph embeddings)

**Figure 23.3: Ranker. Graph-learned user and store embeddings (Uber Eats) and
session-based store2vec similarity (DoorDash) feed a shared-bottom multi-task network.**

```
 USER / CONTEXT                STORE                         USER x STORE
 +--------------------+  +--------------------------+  +-------------------------+
 | user GNN embedding |  | store GNN / store2vec    |  | cos(user_emb,store_emb) |
 | order history seq  |  | cuisine, price band,     |  | past orders at store    |
 | cuisine affinity   |  | rating, popularity       |  | dish-level match        |
 | time, day, weather |  | live ETA, prep load, fee |  | distance, ETA vs habit  |
 | lat/lng (H3 cell)  |  | promo, is_new            |  |                         |
 +--------------------+  +--------------------------+  +-------------------------+
           |                       |                              |
           v                       v                              v
   +-----------------------------------------------------------------------+
   | embeddings (categoricals) + normalized dense --> concat               |
   +-----------------------------------------------------------------------+
                                   |
                                   v
   +-----------------------------------------------------------------------+
   | shared bottom MLP / MMoE experts (target attention over order history)|
   +-----------------------------------------------------------------------+
        |               |                |                  |
        v               v                v                  v
  +-----------+  +--------------+  +-------------+   +----------------+
  | p(click)  |  | p(order|clk) |  | E[GMV]      |   | p(bad exp.):   |
  | sigmoid   |  | sigmoid      |  | regression  |   | late / cancel  |
  +-----------+  +--------------+  +-------------+   +----------------+
        |               |                |                  |
        +---------------+-------+--------+------------------+
                                v
              score = pCTR * pCVR * (a + b*GMV) - c*p(bad)  --> re-ranker
```

## Diagram 4: Real systems (Uber Eats graph learning, DoorDash explore/exploit)

**Figure 23.4: Two published patterns. Left: Uber Eats bipartite graph to GraphSAGE
embeddings to a ranking feature [1]. Right: DoorDash homepage with a UCB-style
exploration bonus on top of the predicted conversion [5]. Both posts were unreachable in
review; the formula and reported lifts are the author's summary.**

```
 UBER EATS (2019) [1]                     | DOORDASH HOMEPAGE (2022) [5]
                                          |
 +---------+  ordered  +---------+        |  +---------------------------+
 |  user   |---------->|  dish / |        |  | candidate stores (geo +   |
 |  nodes  |  weight = | restau- |        |  | knowledge-graph / embed)  |
 +---------+  # orders | rant    |        |  +---------------------------+
      |                +---------+        |               |
      v  sample K-hop neighbors           |               v
 +------------------------------+         |  +---------------------------+
 | GraphSAGE aggregate + MLP    |         |  | ranker: pCVR(user, store) |
 | max-margin loss, hard negs   |         |  +---------------------------+
 +------------------------------+         |               |
      | user & dish/store embeddings      |               v
      v                                   |  +---------------------------+
 +------------------------------+         |  | score = pCVR + c * sqrt(  |
 | cos sim = feature in the     |         |  |  ln N / n_impr(user,store)|
 | downstream personalized      |         |  |  ) (UCB-style bonus)      |
 | ranking model (GBDT/DNN)     |         |  +---------------------------+
 +------------------------------+         |               |
      | reported: offline AUC up,         |               v
      v  online CTR/order lift            |  carousels + vertical list;
   dish & restaurant carousels            |  fewer repeats, new stores seen
```

## Data & labels

| Signal | Label use | Notes |
|---|---|---|
| Impression to click (store page view) | Positive for the CTR head | Heavy position bias. Log position and carousel |
| Click to order | Positive for the CVR head (main business label) | Sparse. Attribution window, e.g. 24 h |
| Order value (GMV), basket size | Regression or weight | Business objective [2] |
| Reorder within 30 days, rating | Long-term satisfaction | Delayed labels |
| Late delivery, cancel, refund, low rating | Negative or "bad experience" head | Penalizes slow or overloaded stores [2] |
| Store page view without order | Weak negative | Possibly closed or high fee |
| Unseen stores | Not negative | Use exploration traffic and propensities [5] |

## Features

| Group | Examples | Freshness |
|---|---|---|
| User | Order history (stores, cuisines, dishes), price sensitivity, dietary tags, user embedding [1][4] | Daily plus session |
| Context | Time of day, weekday, meal slot, weather, location H3 cell [28], device, active filter [6] | Request |
| Store static | Cuisine, price band, rating, menu embedding, photos [8], age (new store) | Daily |
| Store real-time | Open/closed, quoted ETA [29], prep-time load, courier supply, fee, promo | Seconds |
| Cross | cos(user, store) from store2vec [4] or GNN [1], prior orders at store, cuisine affinity x store cuisine [6] | Request |
| Locality | Store popularity in the cell, local cuisine popularity [6] | Hourly |
| Dish / menu tags | Cuisine, dish type, dietary tags on menu items (human-in-the-loop tagging for cold start) [40] | Daily |
| Streaming aggregates | Store orders and views in last N minutes, user session clicks; Kafka + Flink jobs written to a Redis online store [33][34][35] | Seconds |

## Model choices (baseline -> advanced)

| Stage | Option | Tradeoff |
|---|---|---|
| Baseline | Popularity in area x distance, sorted by ETA | No personalization, but a strong cold-start fallback [13] |
| Embeddings v1 | store2vec: word2vec over session store sequences. User = mean of ordered-store vectors [4][25] | Cheap. Ignores graph structure |
| Embeddings v2 | GraphSAGE on user-dish/store bipartite graph [1][24] | Inductive (new dishes), costly to train |
| Ranker v1 | Logistic regression / GBDT pointwise on pCVR | Fast and interpretable |
| Ranker v2 | LambdaMART / pairwise LTR (Swiggy-style) [12] | Optimizes the list. Single objective |
| Ranker v3 | Multi-task DNN / MMoE (CTR, CVR, GMV, bad-exp) [26] | Handles multiple objectives. Needs weight tuning |
| Policy | Value model with marketplace terms + UCB/Thompson exploration [2][5][30] | Fairness to new stores vs short-term CVR |
| Cold-start users | Context-only model (location, time, device), popular-in-cell [13] | No history required |

## Loss functions

| Component | Loss |
|---|---|
| GNN embeddings | Max-margin (hinge) ranking loss with hard negatives [1] |
| store2vec | Skip-gram with negative sampling [4][25] |
| CTR / CVR / bad-exp heads | Binary cross-entropy (CVR trained on clicked rows or ESMM-style over impressions) |
| GMV head | MSE or Huber on log(GMV) |
| LTR variant | LambdaRank / pairwise logistic [12] |
| Total | Weighted sum of the head losses. Weights tuned against online metrics |

## Offline metrics

| Metric | Why |
|---|---|
| AUC per head (CTR, CVR) | Pointwise quality [1] |
| NDCG@k, MRR on ordered store | List quality for the feed |
| Recall@k of candidate generation | The ordered store must survive retrieval |
| Calibration (predicted / observed) | Needed because scores are combined with GMV and ETA |
| Slice metrics | New users, new stores, city, meal slot |
| Coverage / diversity | Share of stores and cuisines exposed |

## Online metrics / A-B

| Metric | Type |
|---|---|
| Conversion rate (sessions with an order), orders per user | Primary |
| GMV per session, take rate | Business |
| CTR on feed and carousels | Engagement (guardrail against clickbait) |
| 7/28-day retention, reorder rate | Long-term |
| Delivery time, lateness, cancel rate | Marketplace guardrail [2] |
| Merchant exposure Gini, new-store first-order rate | Supply-side fairness [5] |
| Experiment design | Geo or time-sliced switchback when supply interference matters. Otherwise user-level A-B. Use CUPAC (model predictions as covariate) to cut variance [37][38] |

## Serving & latency budget (assumption, p99)

| Step | Budget |
|---|---|
| Context + geo candidate lookup (H3 index) | 15 ms |
| Real-time filters (open, ETA service, busy) | 20 ms |
| Feature fetch (online store, batched) | 25 ms |
| Ranker inference (about 500 stores, batched GPU/CPU) | 40 ms |
| Re-rank, exploration, carousel assembly, ads | 20 ms |
| Network and serialization overhead | 30 ms |
| Total | about 150 ms (under 200 ms SLA) |
| Caching | Per-cell popular lists and per-user embeddings cached. Fall back to the cached feed when the ranker times out |

## Monitoring

| What | Signal |
|---|---|
| Feature drift | ETA and prep-load distributions, missing store-state features |
| Prediction drift | Mean pCVR by city and meal slot vs observed |
| Business | Conversion, GMV, cancel rate dashboards by market |
| Supply health | Busy-store exposure, late-delivery rate of top-ranked stores |
| Freshness | Embedding job lag, online/offline feature skew (platform: Michelangelo-style [27]) |
| Exploration | Share of explore impressions, new-store conversion |
| Model accuracy | Logged predictions vs outcomes per model, alert on drift (DoorDash monitoring post) [36] |

## Failure modes

| Failure | Mitigation |
|---|---|
| Recommending closed, overloaded or out-of-radius stores | Hard real-time filters and a busy penalty [2] |
| Rich-get-richer (popular stores dominate) | UCB exploration bonus, new-store boosts [5] |
| Position bias in labels | Log position, position feature at train time and constant at serve time, or IPS |
| Cold-start user | Context-only model, popular-in-cell, onboarding cuisine picks [13] |
| Cold-start store | Content/menu embedding, inductive GNN, explore slots [1] |
| Meal-time shift (breakfast vs dinner) | Time features, per-meal-slot models or slices [6] |
| Feedback loop on ETA (rank fast stores, they get overloaded) | Include live load, cap exposure per store |
| Promotions distort labels | Promo flag feature. Separate ads auction slots |

## Interviewer follow-ups

- How is retrieval different here from YouTube-scale ANN? (Answer: geo-eligibility shrinks the set to about 10^3.)
- How do you combine pCTR, pCVR, GMV and ETA into one score, and how do you set the weights?
- How do you handle a store that is popular but currently backed up in the kitchen?
- Cold start for a new user with no orders, and for a newly onboarded restaurant?
- How would you add exploration without hurting conversion? UCB vs Thompson sampling? [5][30]
- How do you learn store embeddings? Session word2vec vs graph neural network, and why? [1][4]
- How do you rank dishes vs stores, and how do you build carousels (row selection plus in-row ranking)?
- What labels do you use? Click, order, reorder? Which attribution window?
- How do you correct for position bias in a two-dimensional carousel layout?
- How do you A-B test when treatments compete for shared courier supply (switchback, geo experiments)?
- How do sponsored listings (ads) blend with organic ranking?
- How does ranking change for a search query vs the browse feed? [3][9]
- How would you ensure fairness or exposure for small merchants?
- What do you monitor at the dinner peak, and what is the fallback if the ranker times out?
- How would you personalize the cuisine filter order itself? [6]
- How do you rank dishes inside a store or in dish search (query-dish relevance)? [39]
- How do you compute "orders in the last 15 minutes" features and keep online/offline values consistent? [34][35]
- How do you reach significance on small marketplace experiments (variance reduction, CUPAC)? [37]

## Portal comparison

| Source | Framing | Retrieval | Ranker | Distinctive emphasis |
|---|---|---|---|---|
| Uber Eats blog [1][2] | Marketplace recs for eaters, merchants and couriers | Geo + graph embeddings | Personalized ranker with GNN similarity features | Multi-objective marketplace, GraphSAGE |
| DoorDash blog [4][5][6] | Home feed, carousels, cuisine filter | Geo + store2vec / knowledge-graph candidates | pCVR ranker + UCB exploration | Explore/exploit, local popularity |
| Khang Pham [15] | Food-delivery reading list | Points to DoorDash / Uber Eats posts | (Reading-based) | Real-system case studies |
| ByteByteGo [18][19] | Event recs (location-aware), video recs | Multi-source + ANN | Pointwise LTR / multi-task DNN | Location and time features; funnel design |
| alirezadir AIMLInterviews [16] | Generic recsys template | CF, two-tower | Pointwise / pairwise / listwise LTR | 9-step formula; food prompts are ETA and substitutions |
| Swiggy / Delivery Hero [12][13][39] | Restaurant LTR, new-user ranking | Geo-eligible list | LTR / cold-start model | Cold start and dish search |

## References

1. Uber Engineering, "Food Discovery with Uber Eats: Using Graph Learning to Power Recommendations", eng.uber.com, 2019. https://eng.uber.com/uber-eats-graph-learning/ (list; host unreachable)
2. Uber Engineering, "Food Discovery with Uber Eats: Recommending for the Marketplace", eng.uber.com, 2018. https://eng.uber.com/uber-eats-recommending-marketplace/ (list)
3. Uber Engineering, "Food Discovery with Uber Eats: Building a Query Understanding Engine", eng.uber.com, 2018. https://eng.uber.com/uber-eats-query-understanding/ (list, ch-ref)
4. DoorDash Engineering, "Personalized Store Feed with Vector Embeddings", doordash.engineering, 2018. https://doordash.engineering/2018/04/02/personalized-store-feed-with-vector-embeddings/ (list; [15] links an older copy at blog.doordash.com)
5. DoorDash Engineering, "Homepage Recommendation with Exploitation and Exploration", doordash.engineering, 2022. https://doordash.engineering/2022/10/05/homepage-recommendation-with-exploitation-and-exploration/ (list)
6. DoorDash Engineering, "Personalized Cuisine Filter Based on Customer Preference and Local Popularity", doordash.engineering, 2020. https://doordash.engineering/2020/01/27/personalized-cuisine-filter/ (list)
7. DoorDash Engineering, "Powering Search & Recommendations at DoorDash", doordash.engineering, 2017. https://doordash.engineering/2017/07/06/powering-search-recommendations-at-doordash/ (list)
8. DoorDash Engineering, "Selecting the Best Image for Each Merchant Using Exploration and Machine Learning", doordash.engineering, 2023. https://doordash.engineering/2023/01/04/selecting-the-best-image-for-each-merchant-using-exploration-and-machine-learning/ (list)
9. DoorDash Engineering, "Things Not Strings: Understanding Search Intent with Better Recall", doordash.engineering, 2020. https://doordash.engineering/2020/12/15/understanding-search-intent-with-better-recall/ (list)
10. E. Yan, "applied-ml" (curated list), GitHub, 2024. https://github.com/eugeneyan/applied-ml (fetched)
11. Engineer1999, "A Curated List of ML System Design Case Studies", GitHub, 2024. https://github.com/Engineer1999/A-Curated-List-of-ML-System-Design-Case-Studies (fetched)
12. Swiggy Bytes, "Learning To Rank Restaurants", bytes.swiggy.com, 2021. https://bytes.swiggy.com/learning-to-rank-restaurants-c6a69ba4b330 (list)
13. Delivery Hero Tech, "Personalisation @ Delivery Hero: Ranking restaurants for new users", tech.deliveryhero.com, 2023. https://tech.deliveryhero.com/personalisation-delivery-hero-ranking-restaurants-for-new-users/ (list)
14. Delivery Hero Tech, "Personalisation @ Delivery Hero: Understanding Customers", tech.deliveryhero.com, 2023. https://tech.deliveryhero.com/personalisation-at-delivery-hero-understanding-customers/ (list)
15. K. Pham, "machine-learning-interview" (Food delivery section), GitHub, 2021. https://github.com/khangich/machine-learning-interview (fetched)
16. alirezadir, "ML System Design" (9-step formula; question list), AIMLInterviews, GitHub, 2025. https://github.com/alirezadir/AIMLInterviews/blob/main/src/MLSD/ml-system-design.md (fetched)
17. C. Huyen, "Machine Learning Systems Design: Exercises", GitHub, 2019. https://github.com/chiphuyen/machine-learning-systems-design/blob/master/content/exercises.md (fetched)
18. A. Aminian & A. Xu, "Video Recommendation System" (Machine Learning System Design Interview), ByteByteGo, 2023. https://bytebytego.com/courses/machine-learning-system-design-interview/video-recommendation-system (ch-ref)
19. A. Aminian & A. Xu, "Event Recommendation System" (Machine Learning System Design Interview), ByteByteGo, 2023. https://bytebytego.com/courses/machine-learning-system-design-interview/event-recommendation-system (ch-ref)
20. A. Aminian & A. Xu, "Similar Listings on Vacation Rental Platforms", ByteByteGo, 2023. https://bytebytego.com/courses/machine-learning-system-design-interview/similar-listings-on-vacation-rental-platforms (ch-ref)
21. Educative, "Grokking the Machine Learning Interview", educative.io, n.d. https://www.educative.io/courses/grokking-the-machine-learning-interview (ch-ref)
22. Hello Interview, "Recommendation Systems (ML System Design pattern)", hellointerview.com, 2025. https://www.hellointerview.com/learn/ml-system-design/patterns/recommendation-systems (ch-ref)
23. Exponent, "Introduction to the ML System Design Interview", tryexponent.com, 2025. https://www.tryexponent.com/courses/ml-system-design/mlsd-intro (ch-ref)
24. W. Hamilton, R. Ying, J. Leskovec, "Inductive Representation Learning on Large Graphs (GraphSAGE)", arXiv, 2017. https://arxiv.org/abs/1706.02216 (known)
25. T. Mikolov et al., "Efficient Estimation of Word Representations in Vector Space", arXiv, 2013. https://arxiv.org/abs/1301.3781 (known)
26. J. Ma et al., "Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts", KDD, 2018. https://doi.org/10.1145/3219819.3220007 (ch-ref)
27. Uber Engineering, "Meet Michelangelo: Uber's Machine Learning Platform", uber.com, 2017. https://www.uber.com/en-DE/blog/michelangelo-machine-learning-platform/ (ch-ref)
28. Uber Engineering, "H3: Uber's Hexagonal Hierarchical Spatial Index", uber.com, 2018. https://www.uber.com/blog/h3/ (ch-ref)
29. DoorDash Engineering, "Improving ETAs with multi-task models, deep learning, and probabilistic forecasts", careersatdoordash.com, 2024. https://careersatdoordash.com/blog/improving-etas-with-multi-task-models-deep-learning-and-probabilistic-forecasts/ (ch-ref)
30. Instacart Tech, "Using Contextual Bandit models in large action spaces at Instacart", tech.instacart.com, 2023. https://tech.instacart.com/using-contextual-bandit-models-in-large-action-spaces-at-instacart-cb7ab4d8fa4f (list)
31. Grubhub Bytes, "Query2vec: Search query expansion with query embeddings", bytes.grubhub.com, 2019. https://bytes.grubhub.com/search-query-embeddings-using-query2vec-f5931df27d79 (list)
32. DoorDash Engineering, "Evolving DoorDash's Substitution Recommendations Algorithm", doordash.engineering, 2022. https://doordash.engineering/2022/09/08/evolving-doordashs-substitution-recommendations-algorithm/ (list)
33. DoorDash Engineering, "Building a Scalable ML Feature Store with Redis, Binary Serialization, and Compression", doordash.engineering, 2020. https://doordash.engineering/2020/11/19/building-a-gigascale-ml-feature-store-with-redis/ (list)
34. DoorDash Engineering, "Building Riviera: A Declarative Real-Time Feature Engineering Framework", doordash.engineering, 2021. https://doordash.engineering/2021/03/04/building-a-declarative-real-time-feature-engineering-framework/ (list)
35. DoorDash Engineering, "Building Scalable Real Time Event Processing with Kafka and Flink", doordash.engineering, 2022. https://doordash.engineering/2022/08/02/building-scalable-real-time-event-processing-with-kafka-and-flink/ (list)
36. DoorDash Engineering, "Maintaining Machine Learning Model Accuracy Through Monitoring", doordash.engineering, 2021. https://doordash.engineering/2021/05/20/monitor-machine-learning-model-drift/ (list)
37. DoorDash Engineering, "Improving Experimental Power through Control Using Predictions as Covariate", doordash.engineering, 2020. https://doordash.engineering/2020/06/08/improving-experimental-power-through-control-using-predictions-as-covariate-cupac/ (list)
38. DoorDash Engineering, "Meet Dash-AB: The Statistics Engine of Experimentation at DoorDash", doordash.engineering, 2022. https://doordash.engineering/2022/05/24/meet-dash-ab-the-statistics-engine-of-experimentation-at-doordash/ (list)
39. Swiggy Bytes, "Using Deep Learning for Ranking in Dish Search", bytes.swiggy.com, 2021. https://bytes.swiggy.com/using-deep-learning-for-ranking-in-dish-search-4df2772dddce (list)
40. DoorDash Engineering, "Using a Human-in-the-Loop to Overcome the Cold Start Problem in Menu Item Tagging", doordash.engineering, 2020. https://doordash.engineering/2020/08/28/overcome-the-cold-start-problem-in-menu-item-tagging/ (list)
41. Uber Engineering, "How Trip Inferences and Machine Learning Optimize Delivery Times on Uber Eats", eng.uber.com, 2018. https://eng.uber.com/uber-eats-trip-optimization/ (list)
