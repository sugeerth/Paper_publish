# 03. Design a Short-Video Feed (TikTok For You / Instagram Reels / YouTube Shorts) with Real-Time Signals

## Company tags

| Company | Evidence | Ref |
|---|---|---|
| Meta (Instagram / Facebook Reels) | interviewing.io mock "Meta ML system design: Instagram Reels"; Hello Interview community report (TikTok-style recommender, goal = maximize comments; senior variant on Reels with CLIP, multi-task training, nDCG); Exponent-reported Meta loop ending in a TikTok-style rec round | [13] [14] [15] |
| TikTok / ByteDance | Codemia TikTok MLE guide ("design the end-to-end ML pipeline for the For You Page"); techinterview.org ByteDance guide (ranking, candidate generation, real-time serving); LeetCode TikTok MLE phone screen (recommender for videos in a user's followed circle, scale to ~1B users, drift) | [16] [19] [20] |
| YouTube / Google (Shorts) | *Indicative only*: YouTube video rec is the canonical prompt in prep repos, and the reference designs are Google papers; no first-hand Shorts interview report found | [6] [7] [21] [22] |

**Portals that cover it**

| Portal | Page | Ref |
|---|---|---|
| interviewing.io | Meta ML system design mock: Instagram Reels | [13] |
| Hello Interview | Community question: Design TikTok Video Recommendation System | [14] |
| Exponent | TikTok MLE interview guide; "Design TikTok" question | [15] |
| Codemia | Design TikTok's For You Page Recommendation Feed | [16] |
| ByteByteGo (Aminian & Xu) | *ML System Design Interview*, ch. "Video Recommendation System" | [30] |
| Educative | Grokking MLSD: video recommendation candidate-retrieval lesson; TikTok SD blog | [23] |
| recsysml / BuildML (Substack) | "Design a short-video platform" series; short-video recommender case study | [17] [18] |
| GitHub | alirezadir/Machine-Learning-Interviews (video recsys); khangich/machine-learning-interview | [21] [22] |
| Jointaro, DesignGurus, techinterview.org, dataford | TikTok rec walkthroughs; Reels question | [19] [24] [25] [26] |

## Requirements & scale

| Category | Requirement | Source / note |
|---|---|---|
| Functional | Infinite vertical feed, one full-screen video at a time, autoplay; next video personalised instantly from in-session actions | [13] [17] |
| Functional | Signals: watch time / completion, replay, like, share, comment, follow, skip, "not interested" | [3] [4] |
| Functional | Diversity: no two consecutive videos with same sound or creator; integrity filtering | [3] [4] |
| Functional | Cold start: new users (onboarding, device/locale priors) and new videos (small-audience exploration pools) | [16] [19] |
| Non-functional | p99 serving latency < 50-100 ms for ranking | [17] [16] (typical) |
| Non-functional | Freshness: model reacts to new interactions in minutes (online training) | [1] |
| Scale | ~1B DAU; feed requests ~millions/s; ~1T views/day | [16] [19] (prep-site assumptions) |
| Scale | Corpus: 100M to billions of videos; tens of millions eligible for retrieval | [8] [21] (typical) |
| Funnel sizes | Retrieval ~1e4 -> light rank ~1e3 -> heavy rank ~100-500 -> serve ~10 per page | [4] [6] (typical) |
| Out of scope | Upload/transcoding/CDN (classic SD), ads auction, monetization | [17] |

## Diagram 1 - Online serving path

**Figure 3.1 - End-to-end online serving: request -> retrieval -> filtering -> ranking -> re-ranking -> response, with real-time feature updates.**

```
 +--------+  1. GET /feed (user, ctx,  +-------------------+
 | Client |------ session events) ---->|   Feed Service    |
 | (app)  |<-- 8. page of ~8-10 vids --|   (orchestrator)  |
 +---+----+                            +----+--------------+
     | impressions, watch_ms,               | 2. user/ctx features
     | skip, like, share (stream)           v
     |                         +------------------------------+
     |                         |      Feature Store           |
     |                         | offline (batch) + online     |
     |                         | (sec-fresh session counters, |
     |                         |  last-N watched seq)         |
     |                         +--------------+---------------+
     |                                        | 3. fan-out
     |          +-----------------+-----------+------+-----------------+
     |          v                 v                  v                 v
     |   +-------------+  +--------------+  +---------------+  +--------------+
     |   | Two-tower   |  | Item-to-item |  | Follow graph  |  | Trending /   |
     |   | ANN (user   |  | (last watched|  | + creator     |  | fresh-upload |
     |   | emb -> HNSW)|  |  -> similar) |  | sources       |  | explore pool |
     |   +------+------+  +------+-------+  +-------+-------+  +------+-------+
     |          +----------------+--------+---------+-----------------+
     |                                    v ~10k candidates (union, dedup)
     |                     +------------------------------+
     |                     | 4. Filter: seen, blocked,    |
     |                     | integrity/policy, geo/lang   |
     |                     +--------------+---------------+
     |                                    v ~5k
     |                     +------------------------------+   +----------------+
     |                     | 5. Light ranker (distilled   |<--| Model Store /  |
     |                     |    two-tower / small MLP)    |   | Serving PS     |
     |                     +--------------+---------------+   | (emb params    |
     |                                    v ~500              |  synced in min)|
     |                     +------------------------------+   +-------+--------+
     |                     | 6. Heavy ranker (multi-task: |<----------+
     |                     | P(finish), E[watch], P(like),|
     |                     | P(share), P(follow), P(skip))|
     |                     +--------------+---------------+
     |                                    v value = sum(w_i * p_i)
     |                     +------------------------------+
     |                     | 7. Re-rank / policy: creator |
     |                     | & sound dedup, diversity,    |
     |                     | explore slots, boosts        |
     |                     +--------------+---------------+
     |                                    v
     |                          back to Feed Service (8)
     v
 +-----------------+  events   +-----------------+  agg   +---------------------+
 | Event Gateway   |---------->|  Kafka (actions,|------->| Stream processor    |
 | (client logger) |           |  impressions)   |        | (Flink): session    |
 +-----------------+           +--------+--------+        | counters -> online  |
                                        |                 | feature store       |
                                        v                 +---------------------+
                              to offline pipeline (Fig 3.2) and online trainer
```

## Diagram 2 - Offline + streaming training pipeline

**Figure 3.2 - Data, labels, training, evaluation and deployment, with the real-time (online-training) loop that defines this problem.**

```
 +-----------------+        +------------------+        +----------------------+
 | Client events   |------->| Kafka: actions   |---+--->| Streaming Joiner     |
 | (watch, skip,   |        +------------------+   |    | (Flink): join action |
 |  like, share)   |        +------------------+   |    | to served features   |
 +-----------------+        | Kafka: features  |---+    | by request_id; wait  |
 +-----------------+  log   | snapshot at serve|        | window for late watch|
 | Feed Service    |------->| time (no skew)   |        +----+------------+----+
 +-----------------+        +------------------+             |            |
                                                  real-time  |            | dump
                                                  examples   v            v
             +----------------------------+     +--------------+   +--------------+
             | Online trainer (workers +  |<----| Neg sampling |   | Data lake    |
             | Training PS); sparse emb   |     | + log-Q corr.|   | (Hive/HDFS)  |
             | rows updated continuously  |     +--------------+   +------+-------+
             +-------------+--------------+                               |
                           | sparse params: minute-level sync             v
                           | dense params: slower (e.g. daily)   +-------------------+
                           v                                     | Batch training    |
             +----------------------------+                      | (daily retrain /  |
             | Serving PS / Model Store   |<---- new base -------| backfill; label   |
             +-------------+--------------+      checkpoint      | engineering, D2Q) |
                           |                                     +---------+---------+
                           |                                               v
                           |                              +-------------------------+
                           |                              | Offline eval: AUC / NE  |
                           |                              | per head, GAUC, nDCG,   |
                           |                              | calibration, slices     |
                           |                              +------------+------------+
                           |                                           v pass gate
                           |                              +-------------------------+
                           |                              | Model Registry ->       |
                           |                              | shadow -> canary -> A/B |
                           |                              +------------+------------+
                           v                                           |
                 +--------------------+   promote model (A/B win)      |
                 | Online serving     |<-------------------------------+
                 | (Fig 3.1)          |------ feedback loop ---> Client events
                 +--------------------+
```

## Diagram 3 - Model architecture

**Figure 3.3 - Two-tower retrieval (left) and multi-task heavy ranker with long-sequence user modelling (right).**

```
      RETRIEVAL: TWO-TOWER                     HEAVY RANKER: MTL (MMoE-style)
 +----------------+ +-----------------+  +-------------------------------------+
 | User tower     | | Video tower     |  | Inputs per (user, video, ctx)       |
 | user_id emb    | | video_id emb    |  | - sparse ids (user, video, creator, |
 | last-50 watched| | creator emb     |  |   sound, hashtag) -> embeddings     |
 | seq (mean/attn)| | multimodal emb  |  | - last-N session seq + lifelong seq |
 | ctx: time, dev,| | (frames, audio, |  |   (GSU search -> target attention)  |
 | locale         | |  ASR text)      |  | - real-time counters (1h, 24h)      |
 |                | | age, duration   |  | - retrieval scores, two-tower dot   |
 +-------+--------+ +--------+--------+  +------------------+------------------+
         |                   |                              v
         v                   v                 +-------------------------+
   +-----------+       +-----------+           | Feature interaction     |
   | MLP -> u  |       | MLP -> v  |           | (DCN / attention)       |
   | (d=64-256)|       | (d=64-256)|           +------------+------------+
   +-----+-----+       +-----+-----+                        v
         |                   |                 +-------------------------+
         +---> dot(u,v) <----+                 | Shared experts E1..Ek   |
                  |                            | + one softmax gate/task |
                  v                            +-------------------------+
   +-------------------------------+             |     |     |     |     |
   | in-batch softmax + log-Q      |             v     v     v     v     v
   | correction; ANN index over v  |          +-----+-----+-----+-----+-----+
   | (refreshed every few minutes) |          | fin |watch|like |share|skip |
   +-------------------------------+          | BCE |D2Q/ | BCE | BCE | BCE |
                                              |     |quant|     |     |     |
                                              +-----+--+--+-----+-----+-----+
                                                       | p_i (all heads)
                                                       v
                                              +-----------------------------+
                                              | Value model:                |
                                              | score = sum_i w_i * p_i     |
                                              | (w tuned by online A/B)     |
                                              +-----------------------------+
   Shallow tower (bias): position, autoplay, device -> logit added in training,
   dropped at serving [7]
```

## Diagram 4 - Real production system: ByteDance Monolith (real-time training)

**Figure 3.4 - Monolith: collisionless embedding table + online training loop that lets TikTok-style feeds learn from feedback in minutes [1] [2].**

```
 +-------------+  user actions  +---------------+
 | Serving     |--------------->| Kafka         |----+
 | (Serving PS |                | (actions)     |    |
 |  + workers) |  served feats  +---------------+    |   +--------------------+
 |             |--------------->| Kafka         |----+-->| Flink Joiner       |
 +------+------+                | (features)    |        | (in-mem cache +    |
        ^                       +---------------+        |  on-disk KV for    |
        |                                                |  late actions)     |
        |                                                +---------+----------+
        |                                                          | training
        |                                                          v examples
        |                                         +---------------------------------+
        |   periodic param sync                   | Training workers + Training PS  |
        +-----------------------------------------| Embedding table = Cuckoo hash   |
          sparse: only rows touched since last    |  map (collisionless ids)        |
          sync, at minute level;                  | + frequency filter (admit id    |
          dense: less often                       |   after N occurrences)          |
                                                  | + expirable embeddings (evict   |
                                                  |   inactive ids)                 |
                                                  +---------------------------------+
                                                  batch stage: train from data lake
                                                  then switch to online stage
```

Other real systems to name-drop: Instagram Explore/Reels 4-stage funnel (retrieval -> first-stage ranking -> second-stage MTML ranking -> final re-ranking) with two-tower + caching [4]; Facebook Reels UTIS survey-alignment model feeding both retrieval and late ranking [5]; Kuaishou TWIN lifelong-sequence ranker [9], D2Q duration-debiased watch time [10], on-device re-ranking [12]; Meta HSTU generative recommender [11].

## Often missed (vs. real systems)

| Item | Why it matters | Ref |
|---|---|---|
| Upload-time content understanding (multimodal emb, near-dup / re-upload detection, safety classifiers) | Makes a new video rankable before it has interactions; blocks harmful items pre-ranking | [3] [27] |
| New-video traffic pool (staged exposure, promote on early engagement) | Item cold start and creator fairness; gated by early finish/like rates | [3] [16] |
| On-device re-ranking of the prefetched page | Reacts to a skip within the same page without a server round trip | [12] |
| Survey / satisfaction model as an extra objective | Counters watch-time-only optimisation; used in retrieval and late ranking | [5] |
| Lifelong user sequence (GSU/ESU, TWIN) | Long-term interests beyond last 50-100 items within latency | [9] |
| Value-model weight tuning by online experiments | Head weights are product decisions, not learned end-to-end | [4] [7] |
| Fault tolerance of online training (PS snapshots, rollback) | Monolith reports daily PS snapshots suffice despite real-time updates | [1] |

## Data & labels

| Label | Definition | Notes |
|---|---|---|
| Watch time | ms watched (capped, incl. replays) | Biased toward long videos -> duration-quantile (D2Q) or watch_ratio [10] |
| Completion / "effective view" | watched >= X% or >= N s | recsysml: count of videos watched > ~10 s ~ watch time for short videos [17] |
| Skip | swiped away < 1-3 s | Strong, dense negative; short-video equivalent of non-click |
| Like / share / comment / follow | binary per impression | Sparse; shares weighted heavily (Reels) [4] |
| Negative feedback | "not interested", report, hide | Counter-metric and hard-negative label [21] |
| Survey | "how well does this match your interests" 1-5 | Debiased; aligns to true interest [5] |
| Retrieval positives | engaged (finish or like) items; negatives = in-batch + random | log-Q correction for popularity [8] |

## Features

| Group | Examples | Freshness |
|---|---|---|
| User static | id emb, locale, language, device, age bucket | daily |
| User sequence | last 50-100 watched ids + watch ratio; lifelong (1e4-1e6) via GSU/ESU search [9] | real-time / daily |
| Session (real-time) | videos this session, skips in last 5, current topic streak, dwell | seconds |
| Video content | multimodal emb (frames, audio, ASR, caption, hashtags, sound id) | at upload |
| Video stats | CTR/finish rate 1h/24h/7d, age since upload, duration | minutes (stream) |
| Creator | creator emb, follower count, user-creator affinity | hourly |
| Context | hour, weekday, network type, position in session | request time |
| Cross | user-topic affinity, user-sound affinity, two-tower score | request time |

## Model choices (baseline -> advanced)

| Stage | Baseline | Advanced | Tradeoff |
|---|---|---|---|
| Retrieval | Popularity + item-item CF | Two-tower ANN (log-Q), multi-source; transformer seq retriever | Recall vs. freshness of item index |
| Light ranking | Logistic regression on counters | Distilled small DNN / two-tower dot | Latency on ~5k items |
| Heavy ranking | GBDT on hand features | MMoE/PLE multi-task DNN + DCN + target attention over long seq [7] [9] | Accuracy vs. GPU cost |
| Watch-time head | Weighted logistic (YouTube 2016) [6] | D2Q quantile / tree-based progressive regression [10] [28] | Duration bias handling |
| Freshness | Daily batch retrain | Online training, minute-level sparse sync [1] | Stability vs. reactivity |
| Next-gen | Separate stages | Generative recommender (HSTU) over action sequence [11] | Compute; infra maturity |

## Loss

| Component | Loss |
|---|---|
| Retrieval | Sampled softmax / in-batch softmax with log-Q correction [8] |
| Binary heads (finish, like, share, follow, skip) | BCE per head, sum_i lambda_i * L_i (task weights tuned / uncertainty-weighted) |
| Watch time | Weighted logistic (watch-time-weighted positives) [6] or regression on duration-conditioned quantile [10] |
| Position / autoplay bias | Shallow bias tower added during training only [7] |
| Survey alignment | Calibrated regression on debiased survey score [5] |

## Offline metrics

| Stage | Metric |
|---|---|
| Retrieval | Recall@K, hit rate@K on next-watched videos |
| Ranking heads | AUC / GAUC (per-user), normalized entropy, calibration (pred/actual) |
| Watch time | MAE / XAUC on watch time, per-duration-bucket |
| List | nDCG@K, diversity (unique creators/topics per 10), coverage of new videos |
| Slices | new users, new videos, low-activity regions, long vs short videos |

## Online metrics / A-B

| Type | Metrics |
|---|---|
| North star | Total watch time / DAU, retention (D1/D7), sessions per user |
| Engagement | Effective views, finish rate, likes/shares/comments per view, follows |
| Session | Session length, session success rate [17] |
| Guardrails | Skip rate, "not interested"/reports, integrity violations, creator diversity, new-creator exposure, latency/QPS |
| Long-term | Holdouts (weeks) for retention; survey-based satisfaction [5] |
| Method | User-level randomisation; watch for network effects on creators; interleaving for ranking sensitivity |

## Serving & latency budget (typical, p99 ~100 ms)

| Step | Budget | Technique |
|---|---|---|
| Feature fetch (user + session) | 10 ms | online KV (Redis-like), co-located cache |
| Retrieval (parallel sources) | 15 ms | HNSW/ScaNN ANN, precomputed item emb |
| Filtering | 5 ms | Bloom filter of seen ids, policy flags |
| Light ranking ~5k | 10 ms | CPU small DNN, precomputed item side |
| Heavy ranking ~500 | 35 ms | GPU batch inference, item-feature cache, fp16/int8 |
| Re-rank + response | 10 ms | rules + diversity (MMR/DPP) |
| Prefetch | n/a | return 8-10 videos; client prefetches next page while user watches |

## Monitoring

| What | Signal |
|---|---|
| Data | Event lag in Kafka, joiner drop rate, feature null rate, train/serve feature skew |
| Model | Per-head calibration, AUC drift on fresh traffic, embedding norm explosions in online training |
| Business | Watch time, skip rate, report rate by region, creator concentration (Gini) |
| System | p50/p99 latency per stage, PS sync delay, ANN index age, GPU utilisation |

## Failure modes

| Failure | Mitigation |
|---|---|
| Filter bubble / feedback loop | Explore slots, diversity re-rank, creator/sound dedup [3], survey alignment [5] |
| Clickbait / long-video bias in watch time | Multi-objective value model, D2Q debiasing [10] |
| Item cold start (new uploads) | Content emb in item tower, fresh-upload pool with traffic boosting, bandits |
| User cold start | Onboarding interests, locale/device priors, popular-in-region |
| Online training instability (bad batch, bot traffic) | Anomaly filters, rollback to last good checkpoint, dense sync slower than sparse [1] |
| Embedding table blow-up | Frequency filtering + expiring embeddings [1] |
| Train/serve skew | Log features at serving time and join (Fig 3.2) |
| Viral spike / hot item | Item-feature cache, real-time counters, rate-limit boosting |
| Harmful content amplification | Integrity filter before ranking + post-rank policy layer [4] |

## Interviewer follow-ups

- How do you define "engagement" for a feed with no clicks (autoplay)? Watch time vs. completion vs. effective views? [17]
- How do you remove duration bias from watch-time labels? [10]
- How would the next video change based on a skip that happened 2 seconds ago (session features vs. online training vs. on-device re-rank)? [1] [12]
- How do you combine 5+ predicted objectives into one score, and how do you tune the weights? [4] [7]
- Walk through two-tower training: negatives, popularity bias, log-Q correction, index refresh cadence. [8] [23]
- How do you handle cold start for new users and for a creator's first video? [16] [19]
- How do you avoid filter bubbles and ensure creator diversity? [3]
- How would you model a user's lifelong history (10^5+ items) within latency? [9]
- What happens if the online trainer ingests bot traffic or a corrupted batch? [1]
- How do you detect and fix training/serving skew? 
- Offline metrics improved but online watch time dropped: what do you check?
- How would you measure long-term satisfaction rather than short-term engagement? [5]
- How do you scale to ~1B users / millions of QPS; what is cached? [16] [20]
- Concept/data drift: how do you monitor and when do you retrain? [20]
- (Meta variant) Optimise for comments instead of watch time: what changes? [14]
- How do you treat re-uploads / near-duplicates and creator originality in ranking? [3]
- Would you re-rank on the device, and what model and features fit there? [12]
- How do you give a new video its first few hundred impressions, and when do you stop? [3] [16]

## Portal comparison

| Portal | Framing | Retrieval | Ranking | Real-time angle | Distinctive point |
|---|---|---|---|---|---|
| ByteByteGo / Aminian & Xu [30] | Generic YouTube-style video rec | Multiple candidate generators incl. two-tower | Ranking model on relevance labels | Light | Clean 7-step framework; MF vs two-tower tradeoffs |
| alirezadir GitHub [21] | Homepage video rec, 100M DAU, 200 ms | CF (MF / two-tower) + popularity + trending | Content-based ranker | None | Sequential hybrid; precision@k, mAP, diversity |
| Educative Grokking MLSD [23] | Video rec retrieval deep-dive | Two-tower + negative sampling + ANN | Separate lessons | Medium | ANN indexing tradeoffs |
| recsysml Substack [17] | Short video explicitly | MVP then personalisation | Session success rate as CTR analogue | p99 < 50 ms MVP | Watch time ~ count of >10 s views for short clips |
| Codemia / techinterview.org [16] [19] | TikTok FYP, 1B+ DAU | Two-tower + follow graph + trending | ML ranker | Real-time feature store, Kafka | Heavy on infra and cold start |
| interviewing.io / Hello Interview [13] [14] | Meta Reels mock / reports | Candidate generation | Heavier multi-task ranker, CLIP content emb | Varies | Real interviewer feedback; problem-solving scored hardest |

## References

1. Liu et al., "Monolith: Real Time Recommendation System With Collisionless Embedding Table", ByteDance, arXiv / ORSUM@RecSys workshop, 2022. https://arxiv.org/abs/2209.07663
2. ByteDance, "monolith" open-source repository (collisionless embeddings, real-time training), GitHub, 2022. https://github.com/bytedance/monolith
3. TikTok, "How TikTok recommends videos #ForYou", TikTok Newsroom, 2020. https://newsroom.tiktok.com/en-us/how-tiktok-recommends-videos-for-you/
4. Meta Engineering, "Scaling the Instagram Explore recommendations system", engineering.fb.com, 2023. https://engineering.fb.com/2023/08/09/ml-applications/scaling-instagram-explore-recommendations-system/
5. Meta Engineering, "Adapting the Facebook Reels RecSys AI model based on user feedback", engineering.fb.com, 2026. https://engineering.fb.com/2026/01/14/ml-applications/adapting-the-facebook-reels-recsys-ai-model-based-on-user-feedback/
6. Covington, Adams, Sargin, "Deep Neural Networks for YouTube Recommendations", Google Research / RecSys, 2016. https://research.google/pubs/pub45530
7. Zhao et al., "Recommending What Video to Watch Next: A Multitask Ranking System", Google Research / RecSys, 2019. https://research.google/pubs/recommending-what-video-to-watch-next-a-multitask-ranking-system/
8. Yi et al., "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations", Google Research / RecSys, 2019. https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/
9. Chang et al., "TWIN: TWo-stage Interest Network for Lifelong User Behavior Modeling in CTR Prediction at Kuaishou", KDD, 2023. https://arxiv.org/abs/2302.02352
10. Zhan et al., "Deconfounding Duration Bias in Watch-time Prediction for Video Recommendation" (D2Q), Kuaishou, KDD, 2022. https://arxiv.org/abs/2206.06003
11. Zhai et al., "Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations" (HSTU), Meta, ICML, 2024. https://arxiv.org/abs/2402.17152
12. Gong et al., "Real-time Short Video Recommendation on Mobile Devices", Kuaishou, CIKM, 2022. https://arxiv.org/abs/2208.09577
13. interviewing.io, "Meta ML system design mock: Instagram Reels", interviewing.io, 2024. https://interviewing.io/mocks/meta-system-design-ml-instagram-reels
14. Hello Interview, "Design TikTok Video Recommendation System" (community question), hellointerview.com, 2026. https://www.hellointerview.com/community/questions/tiktok-video-recommendation/cmgzuubzb02eq08adu99oof84
15. Exponent, "TikTok Machine Learning Engineer Interview Guide" and "Design TikTok" question, tryexponent.com, 2023-2025. https://tryexponent.com/guides/tiktok-machine-learning-engineer-interview , https://www.tryexponent.com/questions/1372/design-tiktok
16. Codemia, "Design TikTok's For You Page Recommendation Feed" and "TikTok Machine Learning Engineer guide", codemia.io, 2025. https://codemia.io/practice/design-tiktok-s-for-you-page-recommendation-feed , https://codemia.io/guides/tiktok-machine-learning-engineer
17. recsysml, "ML Design Interview | Design a short-video platform Part 2", Substack, 2021. https://recsysml.substack.com/p/ml-design-interview-design-a-short
18. BuildML, "Data Science Case Study: Design a Short-Video Recommender System", Substack, 2024. https://buildml.substack.com/p/data-science-case-study-design-a
19. techinterview.org, "System Design: TikTok / Short-Form Video Platform" and "ByteDance / TikTok Interview Guide", techinterview.org, 2025. https://www.techinterview.org/post/3233460374/system-design-tiktok-short-form-video-platform/ , https://www.techinterview.org/companies/bytedance-tiktok-interview-guide/
20. LeetCode Discuss, "Machine Learning Engineer | Tiktok | Phone Screen", leetcode.com, 2022. https://leetcode.com/discuss/interview-experience/2073085/
21. A. Dirafzoon, "Design a video recommendation system", alirezadir/Machine-Learning-Interviews, GitHub, 2023. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/mlsd-video-recom.md
22. K. Pham, "machine-learning-interview" (YouTube recommendation case, TikTok reading list), GitHub, 2022. https://github.com/khangich/machine-learning-interview
23. Educative, "Video recommendation: candidate retrieval architecture" (Grokking the ML System Design Interview) and "TikTok System Design Interview Questions", educative.io, 2025. https://www.educative.io/courses/grokking-the-machine-learning-system-design-interview/video-recommendation-candidate-retrieval-architecture , https://www.educative.io/blog/tiktok-system-design-interview-questions
24. DesignGurus, "What is TikTok system design interview like?", designgurus.io, 2024. https://www.designgurus.io/answers/detail/what-is-tiktok-system-design-interview-like
25. Taro, "System Design Example: TikTok Recommendation System", jointaro.com, 2024. https://www.jointaro.com/course/master-the-machine-learning-interview-as-a-software-engineer/system-design-example-tiktok-recommendation-system/
26. Dataford, "Design Instagram Reels Recommendations", dataford.io, 2025. https://dataford.io/interview-guides/questions/design-instagram-reels-recommendations
27. Meta Transparency Center, "Instagram Explore AI system", transparency.meta.com, 2024. https://transparency.meta.com/features/explaining-ranking/ig-explore/
28. Lin et al., "Tree based Progressive Regression Model for Watch-Time Prediction in Short-video Recommendation", Kuaishou, KDD, 2023. https://arxiv.org/abs/2306.03392
29. Meta AI, "Powered by AI: Instagram's Explore recommender system", ai.facebook.com, 2019. https://ai.facebook.com/blog/powered-by-ai-instagrams-explore-recommender-system
30. A. Aminian & A. Xu, *Machine Learning System Design Interview* (ch. "Video Recommendation System"), ByteByteGo, 2023. https://bytebytego.com

*Research note: portal pages were read via search-result excerpts. During review, live fetches were blocked by the build environment, so URLs were checked against known publication identifiers (arXiv ids, venues) rather than fetched. Company tags from anonymous reports are indicative, not confirmed.*
