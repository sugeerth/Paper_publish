# 13. Design Event Recommendation (Eventbrite / Meetup) with Location and Time Constraints

> How the prompt is usually worded: "Design an event recommendation system like Eventbrite.
> Users should see personalized events they are likely to register for" (ByteByteGo ch. 7
> [1][2]). The AIMLInterviews question bank lists it as "Event recommendation system
> (Eventbrite)" [3][4].
> Variants: Meetup group and event suggestions, Facebook Events "Discover", local concerts
> and ticketing (Ticketmaster-style), "things to do this weekend near me".
>
> Sourcing note: re-checked on 2026-10-14. Only GitHub hosts resolved; bytebytego.com,
> uber.com, kaggle.com, arxiv.org, huyenchip.com, hellointerview.com and
> developers.google.com did not. (fetched) = read through GitHub on that date. (ch-ref) =
> verified in an earlier chapter. (known) = canonical page not re-read here. No Eventbrite
> or Meetup engineering post could be verified, so no production system is claimed.

## Company tags

| Company | What is reported | Source |
|---|---|---|
| Eventbrite | AIMLInterviews names the question "Event recommendation system (Eventbrite)" and gives a worked solution | [3][4] |
| Eventbrite-style platforms | ByteByteGo / Aminian & Xu MLSDI ch. 7, "Event Recommendation System", is framed on Eventbrite | [1][2][5] |
| General MLE loops (Meta, Amazon) | Portals list "design a recommender for X" as a standard MLSD round; events is one common instance | [12][13] |

**Portals that cover it**

| Portal | Treatment | Link |
|---|---|---|
| ByteByteGo (Aminian & Xu, MLSDI ch. 7) | Full chapter: pointwise LTR, location/time/social features, LR -> GBDT -> NN, mAP, online learning | [1][2] |
| alirezadir AIMLInterviews | Worked notes "mlsd-event-recom": 100M DAU, 1M new events/month, 200 ms, two-stage funnel | [3][4] |
| junfanz1 Awesome-AI-Review | Community summary of the ByteByteGo chapter (features, models, metrics, serving) | [5] |
| Educative "Grokking the ML Interview" | Generic recommendation and search-ranking modules; no event-specific chapter verified | [9] |
| Hello Interview | Recommendation-systems pattern (retrieval, ranking, re-ranking) | [10] |
| Exponent / Aced | General MLSD course and framework; no event-specific write-up verified | [11] |
| Chip Huyen, Khang Pham | No event exercise. Closest: Airbnb location search (Huyen ex. 9); Airbnb Search Ranking and Estimate Delivery Time cases (Pham) | [6][7] |

## Requirements & scale

| Dimension | Value | Source |
|---|---|---|
| Business goal | Increase ticket sales / registrations | [4] |
| Functional | Personalized list of upcoming events for a user, near the user (or a chosen city), in a usable time window; free and paid events | [1][4] |
| Users | About 100M DAU, worldwide, many languages | [4] |
| Catalog | About 1M new events per month. Live inventory of a few million events at any time | [4] / assumption |
| Latency | About 200 ms end to end (proposed) | [4] |
| QPS | About 2-5k average, 20-50k peak (100M DAU x a few feed loads per day) | assumption |
| Candidates | Millions of events -> hundreds after location/time filtering -> top K (10-50) | [4] |
| Key constraints | Events are short-lived and expire, so every event is cold start. Strict geo and time windows. Capacity limits (sold out). Interaction data is sparse | [1][4] |
| Freshness | Registration counts and "friends attending" are streaming features. Model retrains daily or continually | [1][5] |
| Privacy | Location and friend graph are sensitive; obey consent and coarsen location | [1] |

## Diagram 1: End-to-end architecture (online serving path)

**Figure 13.1: Online serving. Hard location and time filters cut millions of events to a
few hundred candidates. A pointwise ranker scores each user-event pair. A policy layer then
handles diversity, freshness, sold-out events and promoted events.**

```
+---------+  GET /events?lat,lng,t   +-----------------+
| Client  |------------------------->| API Gateway +   |
| app/web |<-------------------------| Rec Service     |
+----+----+   top-K events + scores  +--------+--------+
     |                                        | user_id, geo, now, filters
     | impressions / clicks /                 v
     | registrations          +-------------------------------+   +-----------------+
     |                        | 1. CANDIDATE GENERATION       |<--| Geo-time index  |
     |                        |  - geo: geohash / H3 cells    |   | (cell, day) ->  |
     |                        |    within radius R            |   | event ids       |
     |                        |  - time: start in [now, now+T]|   +-----------------+
     |                        |  - social: friends' events,   |   +-----------------+
     |                        |    followed hosts             |<--| ANN index       |
     |                        |  - ANN on user embedding      |   | (event vectors) |
     |                        +---------------+---------------+   +-----------------+
     |                                        | ~1k event ids
     |                                        v
     |                        +-------------------------------+
     |                        | 2. FILTERING                  |
     |                        |  sold out, cancelled, past,   |
     |                        |  age-restricted, blocked host,|
     |                        |  already registered           |
     |                        +---------------+---------------+
     |                                        | ~300-500 candidates
     |                                        v
     |  +-----------------+  features  +-------------------------------+   +-----------+
     |  | Feature store   |----------->| 3. RANKING                    |<--| Model     |
     |  |  batch: user,   |            |  pointwise P(register | u, e) |   | registry  |
     |  |  event, host    |  +-------->|  GBDT or DNN, batch scoring   |   +-----------+
     |  |  stream: reg    |  |         +---------------+---------------+
     |  |  count, friends |  |                         | scored list
     |  |  attending      |  |                         v
     |  +-----------------+  |         +-------------------------------+   +-----------+
     |  +-----------------+  |         | 4. RE-RANK / POLICY           |<--| Rules:    |
     |  | Online features |--+         |  diversity (category, date),  |   | promoted, |
     |  | distance, travel|            |  freshness, host cap,         |   | A/B config|
     |  | time, time-to-  |            |  sold-out guard, dedupe       |   +-----------+
     |  | start (computed |            +---------------+---------------+
     |  | per request)    |                            | top-K -> Rec Service -> Client
     |  +-----------------+
     v
+------------------------+   Kafka   +------------------+   +--------------------------+
| Event logger           |---------->| Stream processor |-->| Stream features (to the  |
| (impression id, rank,  |           | (Flink / Spark)  |   | feature store) + online  |
| model ver, features)   |           +------------------+   | training data (joins)    |
+------------------------+                                  +--------------------------+
```

## Diagram 2: Offline training & data pipeline

**Figure 13.2: Offline and continual pipeline. Registrations are joined to logged
impressions. The label window closes at event start. Training uses a time-based split,
because events expire and random splits leak future information.**

```
+-------------+  +-------------+  +-------------+  +-------------+  +-------------+
| Impression/ |  | Registration|  | Event DB    |  | User DB +   |  | Friendship  |
| click logs  |  | & ticket txn|  | (time, geo, |  | location    |  | graph,      |
| (Kafka)     |  | (OLTP)      |  | price, host)|  | history     |  | invitations |
+------+------+  +------+------+  +------+------+  +------+------+  +------+------+
       |                |                |                |                |
       +-------+--------+                +-------+--------+----------------+
               | impression_id join              | point-in-time snapshot
               v                                 v
     +-----------------------+         +-----------------------------+
     | LABEL BUILDER         |         | FEATURE PIPELINES           |
     |  y=1: registered after|         |  batch (daily): user, event,|
     |  impression, before   |         |   host, text embeddings     |
     |  event start          |         |  stream: reg count, friends |
     |  y=0: shown, no reg   |         |   attending, recent clicks  |
     |  neg sampling + weight|         |  geo: distance, travel time,|
     +-----------+-----------+         |   walk / transit score      |
                 |                     +--------------+--------------+
                 | (user, event, t, y)                | features as of time t
                 +-------------------+----------------+
                                     v
                      +-----------------------------+
                      | TRAINING DATASET            |<-- offline feature store
                      |  time split: train <= T,    |    (point-in-time correct,
                      |  validate on (T, T+7d]      |     no leakage)
                      +--------------+--------------+
                                     v
                      +-----------------------------+
                      | TRAINER                     |
                      |  GBDT baseline; DNN with    |
                      |  warm start for continual   |
                      |  learning (daily / hourly)  |
                      +--------------+--------------+
                                     v
                      +-----------------------------+   fail
                      | EVALUATOR                   |------> alert, keep old model
                      |  mAP, nDCG@K, recall@K,     |
                      |  calibration, slice by city,|
                      |  new event / new user       |
                      +--------------+--------------+
                                     | pass
                                     v
          +----------------+   +----------------+   +----------------------------+
          | Model registry |-->| Shadow/canary  |-->| A/B test -> full rollout   |
          +----------------+   +----------------+   +-------------+--------------+
                                                                  |
   feedback loop: new impressions and registrations --> logs -----+
```

## Diagram 3: Model architecture

**Figure 13.3: Pointwise ranker. Inputs are grouped as ByteByteGo groups them: user, event,
location, time, social and price. The output is P(register). A two-task head (click and
register) is an optional upgrade.**

```
 USER            EVENT            LOCATION         TIME             SOCIAL / PRICE
+------------+  +-------------+  +-------------+  +-------------+  +----------------+
| age bucket |  | category,   |  | distance    |  | time to     |  | # friends      |
| gender     |  | host id +   |  | (bucketed)  |  | start, diff |  | attending,     |
| language   |  | popularity, |  | travel time |  | from user's |  | # invitations  |
| past regs  |  | reg count,  |  | walk score, |  | avg; day of |  | from friends,  |
| (seq, with |  | text emb of |  | transit     |  | week, hour; |  | host is friend |
| time decay)|  | description |  | score, same |  | duration    |  | price, price - |
|            |  |             |  | city flag   |  |             |  | user avg price |
+-----+------+  +------+------+  +------+------+  +------+------+  +-------+--------+
      | embed / one-hot | embed          | bucketize       | bucketize       | normalize
      +--------+--------+-------+--------+--------+--------+--------+--------+
                                |                          |
                                v                          v
               +-------------------------------+   +-----------------------+
               | user tower . event tower      |   | feature crosses       |
               | (dot product = affinity)      |   | (distance x time,     |
               +---------------+---------------+   |  price x user avg)    |
                               |                   +-----------+-----------+
                               +-------------+-----------------+
                                             v
                            +-----------------------------------+
                            | MLP 512 -> 256 -> 128 (ReLU)      |
                            |  (or GBDT on the same features)   |
                            +-----------------+-----------------+
                                              |
                           +------------------+----------------+
                           v                                   v
                  +----------------+                  +----------------+
                  | head: P(click) |                  | head: P(reg)   |
                  | sigmoid, opt.  |                  | sigmoid        |
                  +--------+-------+                  +--------+-------+
                           |                                   |
                           +-----------------+-----------------+
                                             v
                    score = P(reg)   or   w1 * P(click) + w2 * P(reg)
                    loss: BCE / focal / class-balanced (registrations are rare)
```

## Diagram 4: Variant: the geo-time candidate index

**Figure 13.4: Location and time handled as hard constraints at retrieval. Events are
bucketed by geohash cell and start-time bucket. The query expands to neighbouring cells
until enough candidates are found. Online events skip the geo step.**

```
 request: (lat, lng, now, radius R, horizon T)
            |
            v
 +---------------------+    +------------------------------------------------+
 | geohash(lat,lng)    |--->| cells = center + 8 neighbours (precision p)    |
 | precision from R    |    | if |candidates| < N: lower p (bigger cells)    |
 +---------------------+    +-----------------------+------------------------+
                                                    |
                                                    v
 +----------------------------------------------------------------------------+
 | KV store  key = (cell, day_bucket)  ->  [event_id, start_ts, capacity_left]|
 |   day buckets in [now, now+T]  (e.g. today, this weekend, next 30 days)    |
 +---------+------------------------------------------------------------------+
           |  exact distance <= R,     online events:     friends / followed hosts
           v  start_ts > now           no geo filter       (social graph lookup)
 +--------------------+    +-------------------+    +-------------------------+
 | geo-time pool      |    | online / virtual  |    | social pool (friends    |
 | (~1k)              |    | events pool       |    | going, followed hosts)  |
 +---------+----------+    +---------+---------+    +------------+------------+
           |                         |                           |
           +-------------------------+---------------------------+
                                     v
                         merge + dedupe --> ranking (Figure 13.3)
```

## Data & labels

| Item | Choice | Notes / source |
|---|---|---|
| Training row | (user, event, request time, features, label) | One row per shown pair [4] |
| Positive | Registered or bought a ticket after the impression, before the event starts | [1][4] |
| Negative | Shown, not registered (impressed negatives); optionally random in-radius events | [1] / assumption |
| Weak labels | Click, bookmark/like, share: auxiliary heads or weighted positives | [1][4] |
| Imbalance | Registrations are rare -> downsample negatives, focal or class-balanced loss | [1][4][5] |
| Leakage | Point-in-time features only; "registration count" must be the value at impression time | [2] ref 26 |
| Split | Time-based (train on past weeks, validate on next week) | assumption |
| Public data | Kaggle "Event Recommendation Engine Challenge" (users, events, attendees, friends) | [21] |

## Features

| Group | Examples | Computation |
|---|---|---|
| Location | Distance user-event, same city/state/country, walk score, transit score, gap from user's typical distance | Online at request (distance) + batch (scores) [1][5] |
| Time | Time remaining until start, gap from the user's typical lead time, travel time, day of week, hour | Online [1][5] |
| Social | Number/ratio of friends attending, friend invitations, host is a friend, similarity to friend-hosted events | Streaming [1][4] |
| User | Age (bucketized), gender, language, past registrations with time decay, user embedding | Batch [4][5] |
| Event | Category, price, duration, description text embedding, host popularity, registration count | Batch + streaming [4] |
| Price | Price, gap from the user's average paid price, free flag | Batch [1][5] |
| User-event similarity | Similarity of price, description and registration count to the user's past registered events; time-gap similarity | Batch + online [4] |
| Geo encoding | Lat/lng bucketing, geohash or cluster ids, distance to user's frequent places | Batch [2] ref 9 |
| Feature crosses | Distance x time-to-start, price x user avg price (for LR / wide part) | Batch [2] ref 22 |
| Context | Device, current location vs home location, weather (optional) | Online / assumption |

## Model choices (baseline -> advanced)

| Stage | Model | Pros | Cons |
|---|---|---|---|
| 0 | Rule-based: nearest + soonest + most popular | No training, works on day one | Not personalized |
| 1 | Logistic regression | Fast, interpretable | Misses non-linear interactions (distance x time) [1][5] |
| 2 | GBDT (XGBoost / LightGBM) | Strong on tabular features, little preprocessing [22] | Hard to train continually on streaming data [1][4] |
| 3 | DNN pointwise ranker | Non-linear, uses embeddings and text, fine-tunes incrementally | Needs more data and tuning [1][5] |
| 4 | Two-tower retrieval + multi-task DNN ranker | Personalized candidates beyond the geo filter; click + register heads | More infrastructure; sampling bias in retrieval [17][18] |
| Alt | Pairwise / listwise LTR (RankNet, LambdaMART) | Optimizes ordering directly | Harder to calibrate; ByteByteGo picks pointwise [1][8] |

## Loss functions

| Loss | When |
|---|---|
| Binary cross-entropy | Default pointwise P(register) [1][5] |
| Focal loss | Heavy imbalance; down-weights easy negatives [19] |
| Class-balanced loss | Re-weights by effective number of samples per class [20] |
| Pairwise (RankNet) / LambdaRank | Pairwise LTR alternative [2] refs 2-4 [8] |
| Multi-task sum: a*BCE(click) + b*BCE(register) | Two-head ranker in Figure 13.3 |

## Offline metrics

| Metric | Use | Caveat |
|---|---|---|
| mAP | Primary choice: binary relevance (registered or not) [1][4] | Needs the full ranked list |
| nDCG@K | Graded relevance (register > bookmark > click) | Gain values must be chosen |
| Recall@K / Precision@K | Candidate generation quality | Ignore order [4] |
| MRR | Only first relevant item | Weak when many events fit [4] |
| Calibration / log loss | Needed if scores feed thresholds or ads | |
| Slices | City, new events (< 24 h), new users, online vs in-person | |

## Online metrics / A-B

| Metric | Type |
|---|---|
| Registration / ticket conversion rate per session | Primary [1][4] |
| CTR | Engagement [1][4] |
| Bookmark / like rate | Engagement [1][4] |
| Revenue lift (paid tickets) | Business [1][4] |
| Attendance (check-in) rate, refunds, no-shows | Quality guardrail (assumption) |
| Host-side: share of events with at least 1 registration | Marketplace fairness (assumption; see [16]) |
| Latency p99, empty-result rate | System guardrail |

## Serving & latency budget (assumption, p99, ~200 ms total [4])

| Step | Budget |
|---|---|
| Gateway + auth + context (geo lookup) | 15 ms |
| Candidate generation (geo-time KV + ANN + social) | 30 ms |
| Filtering (sold out, past, blocked) | 10 ms |
| Feature fetch (feature store, batched) | 40 ms |
| Ranking (~500 candidates, batched GBDT/DNN) | 50 ms |
| Re-rank / policy + response | 20 ms |
| Headroom | 35 ms |

## Monitoring

| What | Signal |
|---|---|
| Data | Feature null rate, geo lookup failures, stream lag for registration counts |
| Model | Score distribution drift, calibration by city, mAP on daily holdout |
| Product | CTR, conversion, empty results in low-density areas |
| Freshness | Share of shown events that are sold out or already started (should be ~0) |
| Fairness | Exposure share for small hosts and new events |

## Failure modes

| Failure | Mitigation |
|---|---|
| Cold start for every new event | Content features (text embedding, category, host history), exploration slots [4] |
| New user with no history | Popularity in their city + onboarding interests |
| Sparse areas (rural) | Expand radius / time horizon, add online events |
| Showing sold-out or past events | Streaming capacity feature + hard filter at serve time |
| Label leakage from registration counts | Point-in-time feature joins [1] |
| Popularity feedback loop | Exploration, freshness boost, host exposure caps [14] |
| Wrong user location (VPN, travel) | Use device GPS when allowed; let user pick city |
| Privacy of location / friends | Coarsen location, honour consent [2] ref 24 |

## Interviewer follow-ups

- Why is this a ranking problem and not plain classification? Pointwise vs pairwise vs listwise? [1][5]
- Events expire within days or weeks. How do you handle constant cold start? [4]
- Is location a hard filter or a feature? What happens in low-density areas?
- How do you compute distance and travel time efficiently for hundreds of candidates? [4]
- Which features are batch and which are streaming? How fresh must "registration count" be? [1][5]
- How do you handle extreme class imbalance? Focal loss vs downsampling, and how do you recalibrate? [1][4]
- GBDT or neural network, given continual learning needs? [1][4]
- Why mAP rather than nDCG or MRR here? [1][4]
- How do you avoid data leakage when building training data? [1]
- How would you use the friendship graph and invitations? Any privacy risks? [1][4]
- How do you add diversity (dates, categories) and freshness in re-ranking? [1][14]
- This is a two-sided marketplace. How do you keep exposure fair to new or small hosts? [1][16]
- How often do you retrain, and how would online learning work? [1][15]
- How do you rank online/virtual events that have no location?
- Paid promoted events: how do you blend them with organic ranking?
- RSVP vs paid registration: are they the same label? What about age groups and many languages? [4]
- Which features are batch vs real time, and what does each cost to serve? [2] ref 8
- How do you train without exposing users' locations or friend lists (privacy-preserving ML)? [2] ref 24

## Portal comparison

| Portal | Framing | Candidate step | Model | Metrics | Distinctive point |
|---|---|---|---|---|---|
| ByteByteGo ch. 7 [1][2][5] | Pointwise LTR, P(register) | Event filtering (location, type) | LR -> trees -> GBDT -> NN; picks NN for continual learning | mAP offline; CTR, conversion, bookmark, revenue | Deep location/time/social feature design; batch vs streaming |
| alirezadir AIMLInterviews [3][4] | Same, with explicit scale (100M DAU, 1M events/month, 200 ms) | Rule-based filters, millions -> hundreds | GBDT baseline, then NN | mAP; CTR, conversion, revenue | Lists open questions (RSVP vs register, age group, multilingual) |
| junfanz1 notes [5] | Summary of ByteByteGo | Same | Model comparison table | P/R@K, MRR, nDCG, mAP | Adds walk/transit-score features, decay weighting |
| Hello Interview / Educative [9][10] | Generic recsys funnel | Retrieval (embeddings) + ranking | Two-tower + DNN ranker | Recall@K, nDCG | No event-specific constraints; apply geo/time as filters |
| Chip Huyen / Khang Pham [6][7] | No event case | n/a | n/a | n/a | Nearest analogues: Airbnb location search / search ranking |

## References

1. A. Aminian & A. Xu, "Event Recommendation System" (Machine Learning System Design Interview, ch. 7), ByteByteGo, 2023. https://bytebytego.com/courses/machine-learning-system-design-interview/event-recommendation-system (URL follows ByteByteGo's chapter pattern; host unreachable on 2026-10-14)
2. ByteByteGo, "Reference Materials for ML System Design Interview" (Chapter 7: Event Recommendation System, 27 references; "[2] ref n" in the tables means that list's item n), GitHub, 2023. https://github.com/ByteByteGoHq/ml-bytebytego (fetched)
3. alirezadir, "ML System Design" (sample questions: "Event recommendation system (Eventbrite)"), AIMLInterviews, GitHub, 2025. https://github.com/alirezadir/AIMLInterviews/blob/main/src/MLSD/ml-system-design.md (fetched)
4. alirezadir, "Event recommendation system" (mlsd-event-recom), AIMLInterviews, GitHub, 2025. https://github.com/alirezadir/AIMLInterviews/blob/main/src/MLSD/mlsd-event-recom.md (fetched)
5. junfanz1, "ML System Design Interview" notes (Eventbrite event recommendation section), Awesome-AI-Review, GitHub, n.d. https://github.com/junfanz1/Awesome-AI-Review/blob/main/System%20Design/ML%20System%20Design%20Interview.md (fetched)
6. C. Huyen, "Machine Learning Systems Design: Exercises" (no event exercise; ex. 9 Airbnb location search), GitHub, 2019. https://github.com/chiphuyen/machine-learning-systems-design/blob/master/content/exercises.md (fetched)
7. K. Pham, "machine-learning-interview" (no event case; Airbnb Search Ranking and Estimate Delivery Time cases), GitHub, 2021. https://github.com/khangich/machine-learning-interview (fetched)
8. C. J. C. Burges, "From RankNet to LambdaRank to LambdaMART: An Overview", Microsoft Research TR, 2010. https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf (URL from [2])
9. Educative, "Grokking the Machine Learning Interview", educative.io, n.d. https://www.educative.io/courses/grokking-the-machine-learning-interview (ch-ref)
10. Hello Interview, "Recommendation Systems (ML System Design pattern)", hellointerview.com, 2025. https://www.hellointerview.com/learn/ml-system-design/patterns/recommendation-systems (ch-ref)
11. Exponent, "Introduction to the ML System Design Interview" (course), tryexponent.com, 2025. https://www.tryexponent.com/courses/ml-system-design/mlsd-intro (ch-ref)
12. IGotAnOffer, "Meta Machine Learning Engineer Interview", igotanoffer.com, 2025. https://igotanoffer.com/blogs/tech/facebook-machine-learning-engineer-interview (ch-ref)
13. IGotAnOffer, "Amazon Machine Learning Engineer Interview", igotanoffer.com, 2025. https://igotanoffer.com/blogs/tech/amazon-machine-learning-engineer-interview (ch-ref)
14. Google for Developers, "Re-ranking" (freshness, diversity, fairness), Recommendation Systems course, n.d. https://developers.google.com/machine-learning/recommendation/dnn/re-ranking (URL from [2], ref 23)
15. C. Huyen, "Real-time machine learning: challenges and solutions", huyenchip.com, 2022. https://huyenchip.com/2022/01/02/real-time-machine-learning-challenges-and-solutions.html (URL from [2], ref 27)
16. Uber Engineering, "Food Discovery with Uber Eats: Recommending for the Marketplace", uber.com, 2018. https://www.uber.com/blog/uber-eats-recommending-marketplace/ (URL from [2], ref 25; host unreachable on 2026-10-14)
17. P. Covington, J. Adams, E. Sargin, "Deep Neural Networks for YouTube Recommendations", RecSys / Google Research, 2016. https://research.google/pubs/pub45530/ (ch-ref)
18. X. Yi et al., "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations", RecSys / Google Research, 2019. https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/ (ch-ref)
19. T.-Y. Lin et al., "Focal Loss for Dense Object Detection", arXiv, 2017. https://arxiv.org/abs/1708.02002 (known)
20. Y. Cui et al., "Class-Balanced Loss Based on Effective Number of Samples", arXiv, 2019. https://arxiv.org/abs/1901.05555 (known)
21. Kaggle, "Event Recommendation Engine Challenge", kaggle.com, 2013. https://www.kaggle.com/c/event-recommendation-engine-challenge (known; host unreachable on 2026-10-14)
22. XGBoost developers, "XGBoost Documentation", readthedocs, n.d. https://xgboost.readthedocs.io/en/stable/ (URL from [2])
