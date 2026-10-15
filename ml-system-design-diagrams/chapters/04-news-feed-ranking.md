# 04. Design a Personalized News Feed Ranking System (Facebook / LinkedIn / Twitter Home Timeline)

> Prompt as usually given: "When a user opens the app, rank the unseen posts from friends,
> follows, groups and recommended sources so the feed maximizes meaningful engagement."
> Variants: Twitter/X "For You" timeline, LinkedIn homepage feed, Instagram feed,
> comment ranking under a post, "compare ranking objectives for a feed".

## Company tags

| Company (reported) | Where it was reported | Notes |
|---|---|---|
| Meta | IGotAnOffer Meta MLE guide [25], Prepfully Meta MLE bank [26], PracHub "comment ranking for a news feed" [14], Blind [28] | Feed / ranking is the standard Meta ML-design round; comment-ranking variant balances engagement, spam/toxicity and freshness [14] |
| LinkedIn | PracHub "personalized feed + compare ranking objectives" [13] | Follow-up: pointwise vs. pairwise vs. listwise LTR [13]; Khang Pham uses "LinkedIn Feed Ranking" as a case study [7][6] |
| Twitter / X (case study, not a verified interview tag) | Educative "Feed Based System" uses the Twitter feed as its worked example [3]; Khang Pham repo lists "Twitter timeline ranking" as reading under Adtech, linking [21] [7] | Use as the real-system reference (Figure 4.4), not as evidence that X asks this question |
| Generic FAANG / big tech | Exponent question bank [2], AlgoMaster [10], techinterview.org [11], Analytics Vidhya 2026 [12] | Exponent page is a question listing; company tag details were not retrievable in this run |

**Portals that cover it**

| Portal | Treatment | Link |
|---|---|---|
| ByteByteGo (Aminian & Xu, MLSDI ch. 10 "Personalized News Feed") | Full chapter; ads between posts, engagement goal; click/like/comment/hide/block actions | [1] |
| Educative "Grokking the ML Interview" (Feed Based System, Twitter) | 9 lessons: metrics, architecture, tweet selection, features, training data, ranking, diversity, online experiments | [3] |
| Educative "Grokking the ML System Design Interview" (Social feed ranking) | Framing, guardrails, interleaving, creator equity | [4][5] |
| Educative / Khang Pham "Machine Learning System Design" | Feed Ranking System Design lesson (LinkedIn-style) | [6][7] |
| alirezadir AIMLInterviews (GitHub) | Free full write-up: 2B DAU, 200 ms, multi-task NN, dwell + skip heads | [8][9] |
| Exponent | "Design personalized news ranking system" question | [2] |
| AlgoMaster, techinterview.org, Analytics Vidhya, PracHub | Guides and reported prompts | [10][11][12][13][14] |
| Hello Interview | Recsys pattern page (applies directly) | [27] |

## Requirements & scale

| Dimension | Value | Source |
|---|---|---|
| Surface | Home feed on app open / refresh; no query; user_id + context in | [1][9] |
| Timelines served | X home-mixer serves For You (ranked) plus Following and Lists (reverse-chron) | [18] |
| Business goal | Maximize engagement (ads are interleaved, so engagement -> revenue) | [1][9] |
| Output | Ranked list of unseen posts from friends/follows + recommended sources | [9][3] |
| Users | 5B total, 2B DAU, ~2 refreshes/user/day (AIMLInterviews); Meta "more than 2B" users | [9][15] |
| Avg / peak QPS | 2B x 2 / 86,400 ~ 46k avg; x3 peak ~ 140k QPS | typical assumption from [9] |
| Candidate inventory | Hundreds of millions of posts; LinkedIn retrieves ~2,000 from hundreds of millions | [22] |
| Candidates per stage | ~1e8 pool -> ~1-2k retrieved -> ~500 light-ranked -> ~50 shown per page | [22][33] + typical |
| End-to-end latency | 200 ms (AIMLInterviews); 100-200 ms (PracHub); < 200 ms (techinterview.org) | [9][14][11] |
| Features | ~6,000 features hydrated per candidate in Twitter home-mixer | [18] |
| Freshness | New posts visible within seconds-minutes; engagement counters streamed | [9] + typical |
| Non-functional | Integrity/safety filters, diversity, creator fairness, privacy, cold start | [9][11][5] |

---

## Diagram 1 - Online serving path

**Figure 4.1 - Home-feed request through the multi-stage ranking funnel**

```
 +---------+  GET /feed(user_id, ctx)   +----------------------+
 | Client  |--------------------------->|  Feed API / Mixer    |
 | (app)   |<---------------------------|  auth, A/B bucket,   |
 +---------+   page of ~20-50 posts     |  seen-post cache     |
                                        +----------+-----------+
                                                   | user_id, ctx
                                                   v
 +------------------------------------------------------------------------+
 | STAGE 1  CANDIDATE GENERATION  (~1e8 pool -> ~1-2k)          ~30 ms    |
 | +--------------+ +--------------+ +---------------+ +----------------+ |
 | | In-network   | | Out-of-net   | | Graph walk    | | Trending /     | |
 | | friends,     | | two-tower    | | user-post     | | groups, pages, | |
 | | follows,     | | ANN (embed.) | | graph (e.g.   | | topics, fresh  | |
 | | recency idx  | |              | | GraphJet)     | | posts          | |
 | +------+-------+ +------+-------+ +-------+-------+ +-------+--------+ |
 +--------|----------------|-----------------|-----------------|----------+
          +------+---------+-----------------+-----------------+
                 | union + dedup
                 v
 +-------------------------------+     +----------------------------------+
 | STAGE 2  FILTERING            |<----| Integrity / visibility service   |
 | seen, blocked, muted, NSFW,   |     | (spam, hate, misinfo classifiers)|
 | policy, privacy               |     +----------------------------------+
 +---------------+---------------+
                 | ~1k
                 v
 +-------------------------------+  features  +------------------------------+
 | STAGE 3a LIGHT RANKER         |<-----------| FEATURE STORE                |
 | small DNN / GBDT, ~1k -> ~500 |            | online KV: user, author,     |
 +---------------+---------------+            | post counters, affinities,   |
                 | ~500                       | embeddings (batch + stream)  |
                 v                            +--------------+---------------+
 +-------------------------------+  ~6k feats                |
 | STAGE 3b HEAVY RANKER         |<--------------------------+
 | multi-task DNN on GPU/CPU     |<-----------+------------------------------+
 | P(click) P(like) P(comment)   |  weights   | MODEL STORE / REGISTRY       |
 | P(share) P(hide) E[dwell]     |            | versioned ranker, value-model|
 +---------------+---------------+            | weights, calibration         |
                 | per-task probs             +------------------------------+
                 v
 +-------------------------------+
 | VALUE MODEL                   |  score = sum_i w_i * P(action_i)
 | weighted sum of task probs    |  (negative w for hide/report)
 +---------------+---------------+
                 | scored list
                 v
 +-------------------------------+
 | STAGE 4  RE-RANK / POLICY     |  author diversity, in/out-network mix,
 | diversity, dedup, freshness,  |  feedback fatigue, demote borderline,
 | ads + module insertion        |  insert ads / who-to-follow
 +---------------+---------------+
                 | final page --> Feed API --> Client
                 v
 +-------------------------------+     +----------------------------------+
 | IMPRESSION + ACTION LOGGING   |---->| Kafka / log stream -> warehouse  |
 | request_id, position, feats   |     | (training data, metrics, A/B)    |
 +-------------------------------+     +----------------------------------+
```

Sources: funnel and stages [1][9][15]; filters/heuristics/mixing [18]; value model [15][19].

---

## Diagram 2 - Offline training & data pipeline

**Figure 4.2 - Logging -> labels -> features -> training -> registry -> deploy**

```
 +----------------+   impressions,  +-----------------+   join on request_id,
 | Online logging |   actions,      | Raw event lake  |   post_id, timestamp
 | (Figure 4.1)   |---------------->| (Kafka -> HDFS) |------------------+
 +----------------+   dwell time    +-----------------+                  |
        ^                                                                v
        |                                          +---------------------------+
        |                                          | LABEL GENERATION          |
        |                                          | per task: click, like,    |
        |                                          | comment, share = 1/0;     |
        |                                          | hide/report = negative;   |
        |                                          | dwell (regression), skip  |
        |                                          | attribution window ~1 day |
        |                                          +-------------+-------------+
        |                                                        |
        |   +------------------------+  point-in-time join       v
        |   | Feature store (offline)|-------------->+---------------------------+
        |   | daily batch: profiles, |               | TRAINING SET BUILDER      |
        |   | affinities, embeddings |               | negative downsampling,    |
        |   | streaming: counters    |               | dedup per user, time split|
        |   +------------------------+               +-------------+-------------+
        |                                                        |
        |                                                        v
        |                                          +---------------------------+
        |                                          | TRAINING                  |
        |                                          | multi-task DNN, sum of    |
        |                                          | per-task losses; warm     |
        |                                          | start / incremental daily |
        |                                          +-------------+-------------+
        |                                                        |
        |                                                        v
        |                                          +---------------------------+
        |                                          | OFFLINE EVAL              |
        |                                          | per-task AUC / NE,        |
        |                                          | calibration, nDCG on      |
        |                                          | replay, slice checks      |
        |                                          +-------------+-------------+
        |                                                        | pass gates
        |                                                        v
        |   +------------------------+  promote    +---------------------------+
        |   | Shadow -> canary ->    |<------------| MODEL REGISTRY            |
        +---| A/B test -> ramp 100%  |             | version, metrics, value-  |
            | (online metrics +      |             | model weights, lineage    |
            |  guardrails)           |             +---------------------------+
            +------------------------+
   FEEDBACK LOOP: the ranker decides what is shown -> exposure bias in labels;
   mitigate with small exploration traffic + position features.
```

Sources: labels, downsampling, continual fine-tune "deploy only if metrics improve" [9];
offline learning systems at Meta [15]; Twitter weights retuned over time [19].

---

## Diagram 3 - Model architecture: multi-task heavy ranker + value model

**Figure 4.3 - Shared-bottom multi-task ranker with per-action heads**

```
 INPUTS (per user, post candidate)
 +-------------+ +-------------+ +-------------+ +--------------+ +-------------+
 | User        | | Author      | | Post        | | User-author  | | Context     |
 | id emb,     | | id emb,     | | text emb    | | affinity,    | | device,     |
 | demographics| | follower #, | | (BERT), img | | real-graph   | | time of day,|
 | history agg | | health      | | emb (CLIP), | | P(interact), | | position,   |
 |             | |             | | age, counts | | 2-hop feats  | | network     |
 +------+------+ +------+------+ +------+------+ +------+-------+ +------+------+
        +---------------+---------------+---------------+----------------+
                                        | concat (sparse emb + dense)
                                        v
                     +-----------------------------------------+
                     | FEATURE INTERACTION / SHARED LAYERS     |
                     | DCNv2 / MaskNet / transformer blocks    |
                     +--------------------+--------------------+
                                          | shared representation
     +-------------+------------+---------+---+-------------+-------------+
     v             v            v             v             v             v
 +--------+   +--------+   +---------+   +--------+   +----------+   +---------+
 |P(click)|   |P(like) |   |P(cmnt)  |   |P(share)|   |P(hide /  |   |E[dwell] |
 |sigmoid |   |sigmoid |   |sigmoid  |   |sigmoid |   | report)  |   |regress. |
 +---+----+   +---+----+   +----+----+   +---+----+   +----+-----+   +----+----+
     +------------+-------------+------------+-------------+--------------+
                                          | calibrated probabilities
                                          v
                     +-----------------------------------------+
                     | VALUE MODEL  score = sum_i w_i * p_i    |
                     | w > 0 for engagement, w < 0 for hide,   |
                     | report; weights tuned by A/B, not SGD   |
                     +-----------------------------------------+
   LOSS: L = sum_t lambda_t * BCE_t  +  lambda_d * Huber(dwell)
```

Sources: multi-task heads incl. dwell + skip, BERT/CLIP encoders, summed loss [9]; multitask
NN + embeddings at Meta [15]; MaskNet ranker [19][29]; DCNv2 -> transformer at LinkedIn [22].

---

## Diagram 4 - Real system: Twitter/X "For You" (open-sourced 2023)

**Figure 4.4 - Twitter home-mixer pipeline vs. the textbook funnel**

```
 +-------------------------------- CANDIDATE SOURCES ---------------------------+
 | Earlybird search index | UTEG (GraphJet   | CR-Mixer (SimClusters, | FRS    |
 | in-network, light      | user-tweet graph)| TwHIN embeddings,      | follow |
 | ranker; ~50% of posts  |                  | out-of-network)        | recs   |
 +-----------+------------+--------+---------+-----------+------------+---+----+
             +---------------------+---------------------+----------------+
                                   | ~1,500 candidates [33]
                                   v
                   +-------------------------------+
                   | FEATURE HYDRATION ~6,000 feats|  real-graph, TwHIN 3x200-d,
                   | (aggregates, graph, embeds)   |  engagement aggregates [20]
                   +---------------+---------------+
                                   v
                   +-------------------------------+
                   | HEAVY RANKER (parallel        |  10 heads -> weighted sum
                   | MaskNet, ~48M params [33])    |  reply_engaged_by_author +75
                   +---------------+---------------+  reply +13.5, fav +0.5,
                                   v                  neg_feedback -74, report -369
                   +-------------------------------+
                   | FILTERS + HEURISTICS          |  author diversity, in/out
                   | visibility-filters (T&S)      |  balance, fatigue, dedup,
                   +---------------+---------------+  seen, blocked/muted, NSFW
                                   v
                   +-------------------------------+
                   | MIXING: ads, who-to-follow,   |-----> For You timeline
                   | prompts                       |
                   +-------------------------------+
```

Sources: components [17][18]; heads and weights (Apr 5, 2023 config) [19]; features [20].

**Table 4.1 - X Heavy Ranker heads and value-model weights (Apr 5, 2023) [19]**

| Head (predicted P) | Weight | Head (predicted P) | Weight |
|---|---|---|---|
| reply_engaged_by_author | +75.0 | retweet | +1.0 |
| reply | +13.5 | fav (like) | +0.5 |
| good_profile_click | +12.0 | video_playback50 | +0.005 |
| good_click | +11.0 | negative_feedback_v2 | -74.0 |
| good_click_v2 | +10.0 | report | -369.0 |

In-network candidates (~50% of the feed) come from the Earlybird search index with its own
light ranker [17]; TwHIN supplies three 200-d embeddings and aggregates use 50-day and
short-term windows [20].

---

## Data & labels

| Signal | Type | Label use | Note |
|---|---|---|---|
| Click / expand | Implicit, abundant | P(click) head | Noisy; clickbait risk [1] |
| Like / react | Explicit | P(like) | Weight: share > comment > like > click [9] |
| Comment / reply | Explicit, sparse | P(comment) | Strong "meaningful interaction" signal [15][19] |
| Share / retweet | Explicit, sparse | P(share) | Virality |
| Dwell time | Implicit | Regression head | Covers passive users [9] |
| Skip (dwell < t) | Implicit negative | P(skip) | Passive-user negative [9] |
| Hide / block / report | Explicit negative | Negative-weight heads | Twitter report weight -369 [19] |
| Non-engaged impression | Negative | All heads = 0 | Downsample; imbalance [9] |
| Surveys ("worth your time?") | Explicit | Eval / auxiliary | Satisfaction metric [9] |

## Features

| Group | Examples | Freshness |
|---|---|---|
| User | id embedding, demographics, locale, history aggregates (CTR, likes) | daily batch |
| Post content | text embedding (BERT), image/video embedding (CLIP/SimCLR), hashtags, age, media type | at post time |
| Post engagement | like/comment/share counts and velocity (30 min, 1 day) | streaming |
| Author | follower count, author health/quality, posting rate | daily |
| User-author affinity | connection type, past reactions, real-graph P(interact), 2-hop | batch + stream |
| Graph / embeddings | SimClusters, TwHIN, two-tower embeddings | daily |
| Context | device, time of day, day of week, network, position | request time |

Sources: [9][20][15].

## Model choices (baseline -> advanced)

| Stage | Baseline | Advanced | Tradeoff |
|---|---|---|---|
| Retrieval | Reverse-chron in-network + rules | Two-tower ANN, graph walks, LLM dual encoder (LinkedIn LLaMA-3 retrieval, ~2k from 100s of M) [22] | Recall vs. infra cost |
| Light ranker | Logistic regression | Small DNN / GBDT | Cheap pre-filter of ~1k |
| Heavy ranker | Independent per-action LR / GBDT | Shared-bottom multi-task DNN; DCNv2, MaskNet, transformer (LinkedIn Feed-SR) [19][22] | Accuracy vs. GPU cost |
| Objective | Single P(click) | Multi-task + value model weights | Avoids clickbait; weights become a product lever |
| LTR | Pointwise | Pairwise / listwise (LinkedIn follow-up) [13][8] | Pointwise simplest; listwise better ordering |
| Re-rank | Hard rules | Diversity-aware (MMR/DPP), integrity demotion | Engagement vs. diversity |

## Loss

| Head | Loss | Note |
|---|---|---|
| Binary actions | Binary cross-entropy | Sum over tasks, optional task weights [9] |
| Dwell time | MAE / MSE / Huber | Huber robust to long tails [9] |
| Calibration | Post-hoc isotonic / Platt | Needed because value model sums probabilities |
| Imbalance | Negative downsampling + log-odds correction | Recalibrate after sampling |

## Offline metrics

| Metric | What it checks | Source |
|---|---|---|
| ROC-AUC per task | Discrimination per head | [9] |
| Normalized cross-entropy / log loss | Calibrated probability quality | typical |
| nDCG@10, mAP, MRR | Ranking quality of value-model order | [8][11] |
| Calibration (pred / actual) | Value-model correctness | typical |
| Slice metrics | New users, new creators, regions | [5] |

## Online metrics / A-B

| Type | Metrics | Source |
|---|---|---|
| Primary | Time spent, sessions/DAU, meaningful interactions (comments, shares) | [9][22] |
| Secondary | CTR, like/comment/share rates | [8][9] |
| Counter / guardrail | Hide, report, unfollow rates; ad revenue; latency | [8][5] |
| Satisfaction | In-app surveys | [9] |
| Fast comparison | Interleaving of rankers | [5] |
| Reported wins | LinkedIn Feed-SR +2.10% time spent; LiRank +0.5% member sessions | [22] |

## Serving & latency budget (200 ms p99 target [9])

| Step | Budget | Notes |
|---|---|---|
| API, auth, A/B, seen cache | 10 ms | |
| Candidate generation (parallel sources) | 30 ms | ANN + index lookups fan out |
| Filtering + integrity lookups | 10 ms | precomputed classifier scores |
| Feature hydration | 30 ms | batched KV reads, ~6k features [18] |
| Light ranker (~1k) | 15 ms | CPU |
| Heavy ranker (~500) | 50 ms | GPU batch inference |
| Value model + re-rank + ads mix | 15 ms | |
| Network / slack | 40 ms | |
| Precompute option | - | Batch-precompute scores for inactive users; online for active ones (batch vs. online prediction tradeoff [8]) |
| Fallback | - | Reverse-chron in-network feed if the scored pipeline fails (X Conversation Service fallback [18]) |

## Monitoring

| What | Signal | Action |
|---|---|---|
| Feature drift | PSI/KL on top features, null rates | Alert, fallback defaults |
| Prediction drift | Mean predicted vs. observed per head | Recalibrate / retrain |
| Online KPIs | CTR, time spent, hide rate by slice | Auto-rollback |
| Freshness | Streaming counter lag | Page on-call |
| Integrity | Prevalence of violating content in feed | Tighten filters |
| Latency / errors | p50/p99 per stage, timeouts | Degrade to light ranker |

## Failure modes

| Failure | Symptom | Mitigation |
|---|---|---|
| Clickbait / engagement bait | High CTR, low dwell, more hides | Multi-task + negative-weight heads [19] |
| Popular / prolific posters dominate | Close friends crowded out | Affinity features, author diversity [15][18] |
| Celebrity / viral posts | Fan-out hot spots, repeats | Cache, dedup, fatigue rules [9][18] |
| Cold start (user or post) | Generic feed; new posts never shown | Onboarding topics, exploration slots [9] |
| Position bias | Top slots over-labelled positive | Position feature, set to a fixed value at serving (typical; [9] lists positional bias as a topic only) |
| Feedback loop / filter bubble | Narrowing topics | Exploration traffic, out-of-network quota |
| Data leakage | Offline AUC up, online flat | Point-in-time feature joins [9] |
| Catastrophic forgetting | Drift after incremental training | Replay buffer, periodic full retrain [9] |
| Integrity misses | Harmful content ranks high | Filter + demotion before and after ranking; X visibility-filters use hard filters, visible treatments and coarse down-ranking [17] |
| Ranker timeout / outage | Empty or stale feed | Serve reverse-chron in-network fallback; degrade to light ranker [18] |

## Interviewer follow-ups

- Why multi-task instead of N separate models? How do you set the value-model weights? [9][19]
- How would you handle passive users who never like or comment? [9]
- Pointwise vs. pairwise vs. listwise ranking: which and why? [13][8]
- How do you keep close friends' posts from being drowned by pages and frequent posters? [15]
- How do you rank a celebrity post with millions of followers (fan-out, caching)? [9]
- How do you correct for position bias in training labels?
- How do you calibrate probabilities after negative downsampling?
- Design the comment ranking under a post (spam, toxicity, freshness). [14]
- How do you add out-of-network / recommended content to a follow-based feed? [17][22]
- What guardrail metrics stop you from shipping an engagement-bait winner? [5]
- How fresh must features be, and what goes in batch vs. streaming? [9]
- How do you fit ~500 candidates through a heavy ranker inside 200 ms? [9][18]
- Cold start for new users and new posts? [9]
- How would you measure long-term satisfaction rather than short-term clicks? [23]
- Where do integrity classifiers run, and how do they interact with ranking? [17]
- What does the feed show if the heavy ranker times out? [18]
- Why would a reply the author engages with be weighted 150x a like? [19]

## Portal comparison

| Portal | Objective | Candidate gen | Ranker | Distinctive emphasis |
|---|---|---|---|---|
| ByteByteGo [1] | Engagement (ads interleaved) | Retrieval of unseen posts | Multi-task DNN over reactions | Clarifying-questions dialogue; hide/block as negatives |
| AIMLInterviews [9] | Weighted sum of reactions | Rule-based unseen posts | Multi-task NN + dwell + skip heads | Passive users; BERT/CLIP features; 2B DAU / 200 ms |
| Educative Grokking ML Interview [3] | P(engage) on Twitter feed | Tweet selection stage | Ranking + diversity | Training data generation, online experimentation |
| Educative Grokking MLSD [4][5] | Social feed ranking | Funnel | Multi-stage | Guardrails, interleaving, creator equity |
| Khang Pham [6][7] | LinkedIn-style long-term engagement | Multi-source | Ranking model | Real-company case study framing |
| techinterview.org / AlgoMaster [11][10] | NDCG@10 offline, A/B online | Followed, topics, trending, embeddings | Multi-task weighted score | < 200 ms, fairness to minority creators |

## References

1. A. Aminian & A. Xu, "Personalized News Feed" (Machine Learning System Design Interview, ch. 10), ByteByteGo, 2023. https://bytebytego.com/courses/machine-learning-system-design-interview/personalized-news-feed
2. Exponent, "Design personalized news ranking system" (question), tryexponent.com, n.d. https://www.tryexponent.com/questions/3093/design-personalized-news-ranking-system
3. Educative, "Grokking the Machine Learning Interview" (Feed Based System module), educative.io, n.d. https://www.educative.io/courses/grokking-the-machine-learning-interview
4. Educative, "Social feed ranking: problem framing and requirements" (Grokking the ML System Design Interview), educative.io, 2025. https://www.educative.io/courses/grokking-the-machine-learning-system-design-interview/social-feed-ranking-problem-framing-and-requirements
5. Educative, "Social feed ranking: evaluation and experimentation" (Grokking the ML System Design Interview), educative.io, 2025. https://www.educative.io/courses/grokking-the-machine-learning-system-design-interview/social-feed-ranking-evaluation-and-experimentation
6. Educative / K. Pham, "Feed Ranking System Design" (Machine Learning System Design course), educative.io, n.d. https://www.educative.io/courses/machine-learning-system-design/lta/feed-ranking-system-design
7. K. Pham, "machine-learning-interview" (LinkedIn Feed Ranking listed), GitHub, 2021. https://github.com/khangich/machine-learning-interview
8. alirezadir, "ML System Design" template (ranking, LTR, metrics), GitHub, 2025. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/ml-system-design.md
9. alirezadir, "News Feed System" (mlsd-newsfeed.md), GitHub, 2025. https://github.com/alirezadir/AIMLInterviews/blob/main/src/MLSD/mlsd-newsfeed.md
10. AlgoMaster, "Design Personalized News Feed" (ML system design), algomaster.io, n.d. https://algomaster.io/learn/ml-system-design/design-personalized-news-feed
11. techinterview.org, "Machine Learning System Design Interview: Feed Ranking, Recommendations, and Search", techinterview.org, n.d. https://www.techinterview.org/post/3233460469/machine-learning-system-design-interview/
12. Analytics Vidhya, "System Design for ML Interviews: 10 Real Problems Walked Through", analyticsvidhya.com, 2026. https://www.analyticsvidhya.com/blog/2026/06/system-design-for-ml-interviews-real-problems-solved/
13. PracHub, "Design a Personalized Feed and Compare Ranking Objectives" (LinkedIn), prachub.com, n.d. https://prachub.com/interview-questions/design-a-personalized-feed-and-compare-ranking-objectives
14. PracHub, "Design comment ranking for a news feed" (Meta), prachub.com, n.d. https://prachub.com/interview-questions/design-comment-ranking-for-a-news-feed
15. Meta Engineering, "How machine learning powers Facebook's News Feed ranking algorithm", engineering.fb.com, 2021. https://engineering.fb.com/2021/01/26/ml-applications/news-feed-ranking/
16. Facebook, "Adaptive ranking of news feed in social networking systems" (US patent 8768863), Justia Patents, 2014. https://patents.justia.com/patent/8768863
17. Twitter, "the-algorithm" (For You timeline source), GitHub, 2023. https://github.com/twitter/the-algorithm
18. Twitter, "home-mixer README", GitHub, 2023. https://github.com/twitter/the-algorithm/blob/main/home-mixer/README.md
19. Twitter, "the-algorithm-ml: Heavy Ranker (projects/home/recap)", GitHub, 2023. https://github.com/twitter/the-algorithm-ml/tree/main/projects/home/recap
20. Twitter, "Heavy Ranker FEATURES.md", GitHub, 2023. https://github.com/twitter/the-algorithm-ml/blob/main/projects/home/recap/FEATURES.md
21. Twitter Engineering, "Using deep learning at scale in Twitter's timelines", blog.twitter.com, 2017. https://blog.twitter.com/engineering/en_us/topics/insights/2017/using-deep-learning-at-scale-in-twitters-timelines.html
22. F. Borisyuk et al., "LiRank: Industrial Large Scale Ranking Models at LinkedIn", arXiv, 2024. https://arxiv.org/abs/2402.06859
23. KDnuggets, "How LinkedIn Uses Machine Learning To Rank Your Feed", kdnuggets.com, 2022. https://www.kdnuggets.com/2022/11/linkedin-uses-machine-learning-rank-feed.html
24. Shaped, "Feed ranking systems", shaped.ai, n.d. https://shaped.ai/blog/feed-ranking-systems
25. IGotAnOffer, "Meta Machine Learning Engineer Interview", igotanoffer.com, 2025. https://igotanoffer.com/blogs/tech/facebook-machine-learning-engineer-interview
26. Prepfully, "Meta Machine Learning Engineer interview questions", prepfully.com, 2025. https://prepfully.com/interview-questions/meta/machine-learning-engineer?page=8
27. Hello Interview, "Recommendation Systems (ML System Design pattern)", hellointerview.com, 2025. https://www.hellointerview.com/learn/ml-system-design/patterns/recommendation-systems
28. Blind, "Meta - help for ML design round feedback", teamblind.com, n.d. https://www.teamblind.com/post/meta-help-for-ml-design-round-feedback-nrxobup2
29. Z. Wang, Q. She, J. Zhang, "MaskNet: Introducing Feature-Wise Multiplication to CTR Ranking Models", arXiv, 2021. https://arxiv.org/abs/2102.07619
30. MLconf, "Generating a Billion Personal News Feeds", mlconf.com, n.d. https://mlconf.com/sessions/generating-a-billion-personal-news-feeds-with-exp/
31. P. Halina, "ML Systems Design Interview Guide", patrickhalina.com, 2020. http://patrickhalina.com/posts/ml-systems-design-interview-guide/
32. C. Huyen, "Introduction to Machine Learning Interviews Book", huyenchip.com, 2021. https://huyenchip.com/ml-interviews-book/
33. Twitter Engineering, "Twitter's Recommendation Algorithm", blog.twitter.com, 2023. https://blog.twitter.com/engineering/en_us/topics/open-source/2023/twitter-recommendation-algorithm

> Research note: in the review pass, [7], [8], [9], [17], [18], [19] and [20] were fetched
> and checked against the claims above; non-GitHub hosts were unreachable from the
> sandbox (DNS/proxy refusal) and the shared web-search budget was exhausted, so [1]-[6],
> [10]-[16], [21]-[33] rest on earlier search-result summaries and are unverified here.
> Company tags for Meta and LinkedIn depend on those unverified portal pages. Values
> marked "typical" are assumptions.
