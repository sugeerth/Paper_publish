# 02. Design a Video Recommendation System (YouTube Home Feed)

> Prompt as usually given: "Design the ML system that fills the YouTube home page with
> personalized videos for each logged-in user." Variants: "watch next", Reels/Shorts/TikTok
> short-video feed, Netflix-style rows.

## Company tags

| Company (reported) | Where it was reported | Notes |
|---|---|---|
| Google / YouTube | interviewing.io mock [6], Blind [13] | ML SWE loop; follow-ups on RNN serving throughput and monitoring [13] |
| Meta | Blind [12][14], LeetCode Discuss [15], Hello Interview community [8] | Rec/ranking is the standard Meta ML-design round; Reels/TikTok-style variant [8] |
| Coupang | 1point3acres [16] | "YouTube recommendation, from metrics to pipeline" |
| TikTok / ByteDance | Codemia [17], Hello Interview [8], Jointaro [27] | Short-video variant; real-time training (Monolith) [24] |
| Netflix-style variant (practice framing, not a reported loop) | Chip Huyen exercises [29], AIMLInterviews [10] | Rows/titles; "when does a user stop watching a show" |

**Portals that cover it**

| Portal | Treatment | Link |
|---|---|---|
| ByteByteGo (Aminian & Xu, MLSDI ch. 6) | Full chapter, hybrid filtering, multi-source candidate gen; newsletter deep dive | [1][2] |
| Educative "Grokking the ML System Design Interview" | 5-lesson case study (framing, data, retrieval, ranking, serving) | [3][4][5] |
| interviewing.io | Google mock, 2B users / 10B videos / < 1 s | [6] |
| Hello Interview | Recsys pattern page + TikTok community question | [7][8] |
| alirezadir AIMLInterviews (GitHub) | Free full write-up, mirrors ByteByteGo structure | [9][10] |
| khangich machine-learning-interview (Khang Pham) | "YouTube Recommendation" listed as design #1 | [11] |
| Exponent | MLSD course (model selection; related YouTube-ads question) | [28] |
| Codemia, AlgoMaster, Jointaro, techinterview.org | Practice prompts / guides | [17][18][31][27][26] |

## Requirements & scale

| Dimension | Value | Source |
|---|---|---|
| Surface | Home page, logged-in user, no query | [1][10] |
| Business goal | Engagement (watch time, completed views) + satisfaction (likes, surveys) | [1][20][22] |
| Relevance label | Explicit like OR watched >= ~50% of video | [10] |
| Users | 2B users (interviewing.io); 100M DAU (AIMLInterviews) | [6][10] |
| Catalog | 10B videos (interviewing.io); 100M (AIMLInterviews); "tens of millions" in YouTube retrieval paper | [6][10][21] |
| End-to-end latency | ~200 ms (AIMLInterviews); < 1 s (interviewing.io) | [10][6] |
| Retrieval budget | < 10 ms over > 1B videos to a few hundred | [4] |
| Peak QPS | ~2B DAU x ~10 home loads/day / 86,400 x 3 peak ~ 700k QPS | typical assumption |
| Candidates per stage | ~1e9 -> ~1e3 (retrieval) -> ~1e2 (ranking) -> ~20-50 shown | [19] + typical |
| Signals | "> 80 billion signals" (YouTube) | [23] |
| Freshness | New uploads recommendable within minutes-hours | typical assumption |
| Non-functional | Diversity, safety/policy, cold start, multilingual, fairness | [10][7] |

---

## Diagram 1 - Online serving path

**Figure 2.1 - End-to-end home-feed serving (multi-stage funnel)**

```
 +---------+  GET /home(user_id,ctx)  +-------------------+
 | Client  |------------------------->|  Rec API Gateway  |
 | (app)   |<-------------------------|  (auth, AB bucket)|
 +---------+   ranked list (~30)      +---------+---------+
      |                                         | user_id, ctx
      | impressions/clicks/watch                v
      |                      +----------------------------------+
      |                      |  Feature Service (online store)  |
      |                      |  user emb, recent watch seq,     |
      |                      |  counters, context               |
      |                      +----------------+-----------------+
      |                                       | user features
      |                                       v
      |   +--------------------------------------------------------------+
      |   |           STAGE 1: CANDIDATE GENERATION  (~1e9 -> ~1e3)      |
      |   |  +-------------+ +-------------+ +----------+ +-----------+  |
      |   |  | Two-tower   | | Item-to-item| | Trending/| | Subscript.|  |
      |   |  | ANN (HNSW/  | | co-watch    | | popular  | | / fresh   |  |
      |   |  | ScaNN)      | | graph       | | by region| | uploads   |  |
      |   |  +------+------+ +------+------+ +-----+----+ +-----+-----+  |
      |   |         +---------------+------+-------+------------+        |
      |   |                         v  union + dedup                     |
      |   +-------------------------+------------------------------------+
      |                             | ~1-2k video_ids
      |                             v
      |   +--------------------------------------------------------------+
      |   |  FILTERING: watched, blocked, age/region, policy, dup        |
      |   +-------------------------+------------------------------------+
      |                             | ~500-1k
      |                             v
      |   +--------------------------------------------------------------+
      |   |  STAGE 2: LIGHT RANKER (small DNN / distilled)  -> ~200      |
      |   +-------------------------+------------------------------------+
      |                             v
      |   +----------------------------------------------------+  +----------+
      |   |  STAGE 3: HEAVY RANKER (multi-task MMoE)           |<-| Model    |
      |   |  P(click), E[watch], P(like), P(dismiss), P(surv.) |  | Store /  |
      |   |  score = sum_k w_k * f(pred_k)                     |  | Registry |
      |   +-------------------------+--------------------------+  +----------+
      |                             v ~100 scored
      |   +--------------------------------------------------------------+
      |   |  RE-RANK / POLICY: diversity (MMR/DPP), freshness boost,     |
      |   |  creator caps, clickbait/borderline demotion, explore slots  |
      |   +-------------------------+------------------------------------+
      |                             v ~30 videos
      v                             |
 +----------------------+           |   served list + features (logged)
 | Event Logger (Kafka) |<----------+
 +----------------------+  --> offline pipeline (Figure 2.2)
```

**Latency budget (target p99 ~200 ms server-side)** [10]

| Step | Budget | Notes |
|---|---|---|
| Gateway + feature fetch | 15-25 ms | Online KV store, batched multi-get |
| Candidate generation | 10-30 ms | Parallel sources; ANN < 10 ms [4] |
| Filtering | ~5 ms | Bloom filter of watched ids |
| Light ranker | 10-20 ms | ~1k items, CPU |
| Heavy ranker | 40-80 ms | ~200 items, GPU/TPU batched |
| Re-rank + policy | ~10 ms | Greedy diversification |
| Logging | async | Never on critical path |

---

## Diagram 2 - Offline data & training pipeline

**Figure 2.2 - Logging -> labels -> features -> training -> registry -> deployment**

```
 +--------------+   impressions, clicks,     +-------------------+
 | Event Logger |   watch_time, likes,  ---> | Raw Log Lake      |
 | (Kafka)      |   dismiss, survey          | (daily partitions)|
 +------+-------+                            +---------+---------+
        | stream                                       | batch
        v                                              v
 +--------------+                            +-------------------+
 | Stream proc. | -- real-time counters -->  | Label Join        |
 | (Flink)      |    (views/1h, CTR/1h)      | impression x      |
 +------+-------+                            | outcome (24h win) |
        |                                    | pos: like or >50% |
        v                                    | watch; neg: shown |
 +--------------+  point-in-time features    | not clicked       |
 | Feature Store|<-------------------------->+---------+---------+
 | online+offl. |  (avoid train/serve skew)            |
 +------+-------+                                      v
        |                                    +-------------------+
        |                                    | Training Datasets |
        |                                    | - retrieval: (u,  |
        |                                    |   next watch) seq |
        |                                    | - ranking: impr.  |
        |                                    |   w/ multi labels |
        |                                    +---------+---------+
        |                                              v
        |   +----------------------+         +-------------------+
        |   | Content encoders     |  embs   | Trainers (TPU/GPU)|
        +-->| (title BERT, frames, |-------->| two-tower (daily) |
            |  audio) for new vids |         | ranker (hourly    |
            +----------------------+         | warm-start)       |
                                             +---------+---------+
                                                       v
                                             +-------------------+
                                             | Offline Eval      |
                                             | recall@k, nDCG,   |
                                             | AUC/task, slices  |
                                             +---------+---------+
                                          pass gates   v
 +--------------+   item embs -> ANN index   +-------------------+
 | ANN Index    |<---------------------------| Model Registry    |
 | builder      |                            | (versioned)       |
 +------+-------+                            +---------+---------+
        |                                              v
        v                                    +-------------------+
 serving (Fig 2.1) <----- shadow -> canary --| Deployment / A-B  |
        ^                                    +---------+---------+
        |                                              |
        +------ new impressions = new training data <--+
                 (FEEDBACK LOOP: add explore traffic +
                  position/selection-bias correction)
```

---

## Diagram 3 - Model architecture

**Figure 2.3a - Retrieval: two-tower with sampled softmax (logQ correction)** [19][21]

```
        USER / QUERY TOWER                         VIDEO TOWER
 +-----------------------------------+   +-----------------------------------+
 | watch history ids -> emb -> avg   |   | video_id emb                      |
 |   or seq. encoder (last ~50)      |   | channel_id emb, topic/tags emb    |
 | search tokens -> emb -> avg       |   | title text emb (BERT, frozen)     |
 | geo, lang, device, gender, age    |   | duration, upload age, language    |
 | example_age (0 at serve time)     |   | popularity counters (log)         |
 +-----------------+-----------------+   +-----------------+-----------------+
                   v                                       v
          +-----------------+                     +-----------------+
          | MLP ReLU 1024-> |                     | MLP ReLU 1024-> |
          | 512 -> 256      |                     | 512 -> 256      |
          +--------+--------+                     +--------+--------+
                   | u (256-d, L2 norm)                    | v_i (256-d)
                   +------------------+--------------------+
                                      v
                        +----------------------------+
                        | s(u,v_i) = <u, v_i> / tau  |
                        | in-batch softmax, subtract |
                        | log q(i) (sampling prob.)  |
                        +-------------+--------------+
                                      v
             TRAIN: cross-entropy vs next watched video (positives)
             SERVE: v_i precomputed -> ANN index; u computed per request
```

**Figure 2.3b - Ranking: multi-task MMoE + shallow position-bias tower** [20]

```
 inputs: user feats | video feats | user x video cross | context (time,
         device); ids -> embeddings, counts -> log/normalize
                                |
        +---------------+-------+-------+---------------+
        v               v               v               v
 +------------+  +------------+  +------------+  +------------+
 | Expert 1   |  | Expert 2   |  | Expert 3   |  | Expert N   |   shared MLP experts
 +------------+  +------------+  +------------+  +------------+
        |               |               |               |
        +---------------+-------+-------+---------------+
                                |   all expert outputs go to every gate
        +---------------+-------+-------+---------------+
        v               v               v               v
 +------------+  +------------+  +------------+  +------------+
 | gate_click |  | gate_watch |  | gate_like  |  | gate_neg   |   softmax over experts
 +------------+  +------------+  +------------+  +------------+
        |               |               |               |
 +------------+  +------------+  +------------+  +------------+
 | tower+head |  | tower+head |  | tower+head |  | tower+head |   per-task MLP tower
 | P(click)   |  | E[watch]   |  | P(like/shr)|  | P(dismiss) |
 | sigmoid    |  | wLR / reg. |  | sigmoid    |  | sigmoid    |
 +------------+  +------------+  +------------+  +------------+
        |               |               |               |
       (+)<-+          (+)<-+           |               |   engagement heads get
        |   |           |   |           |               |   a position-bias logit
        |   +-----------|---+-----------|---------------|---< SHALLOW TOWER
        |               |               |               |   (position, device);
        |               |               |               |   train only, dropped
        |               |               |               |   at serving
        +---------------+-------+-------+---------------+
                                v
 final score = w1*P(click) + w2*E[watch] + w3*P(like) - w4*P(dismiss)
 (weights w_k tuned by online A/B, not by offline loss)
```

---

## Diagram 4 - Real system: YouTube (2016 -> 2019) vs. ByteByteGo textbook

**Figure 2.4 - How the reference solutions map to YouTube's published system** [19][20][21][25]

```
 YouTube 2016 (Covington et al.) [19]        YouTube 2019 additions [20][21]
 +-----------------------------------+       +-----------------------------------+
 | Candidate gen = extreme multiclass|       | Retrieval = two-tower ("NDR"),    |
 | softmax over millions of videos;  | ----> | streaming item-frequency estimate |
 | serve with nearest-neighbour      |       | -> logQ-corrected in-batch softmax|
 +-----------------+-----------------+       +-----------------+-----------------+
                   v                                           v
 +-----------------------------------+       +-----------------------------------+
 | Ranking = weighted logistic regr. | ----> | Ranking = MMoE multi-task         |
 | positives weighted by watch time  |       | (engagement + satisfaction) +     |
 | -> odds ~ expected watch time     |       | shallow tower for position bias   |
 +-----------------------------------+       +-----------------------------------+

 ByteByteGo / AIMLInterviews textbook answer [1][10]
 +-------------+   +--------------+   +-------------+   +--------------+
 | CandGen x N | ->| Scoring      | ->| Re-ranking  | ->| Home feed    |
 | two-tower,  |   | heavier model|   | region/age, |   | (top-k)      |
 | popular,    |   | (DNN / DCN)  |   | fresh, dedup|   |              |
 | trending    |   | more feats   |   | clickbait   |   |              |
 +-------------+   +--------------+   +-------------+   +--------------+

 Short-video variant (TikTok / Monolith) [24]: same funnel, but ONLINE training
 (minutes) + collisionless hash embedding tables; watch completion dominates.
```

---

## Production details most portals omit

| Detail (YouTube) | Why it matters in the interview | Source |
|---|---|---|
| Train candidate gen on all watches, incl. embedded/off-surface plays, not only rec impressions | Surfaces new content; avoids learning only from the recommender's own exposure | [19] |
| Fixed number of training examples per user | Stops heavy users dominating the loss | [19] |
| Predict the *next* watch from past history (no future context), not a random held-out watch | Avoids label leakage; matches asymmetric co-watch behaviour | [19] |
| `example_age` feature, set to 0 at serving | Lets the model learn the freshness preference instead of stale averages | [19] |
| Streaming item-frequency estimation (hashed arrays) feeding logQ | In-batch negatives over-penalise popular items without it | [21] |
| Ranker trained continuously on streaming logs; shallow tower for selection/position bias | Implicit feedback is biased by what was shown and where | [20] |
| User surveys ("valued watch time"), borderline-content demotion | Watch time alone is not satisfaction; responsibility guardrail | [22] |

## Data & labels

| Label / signal | Definition | Use | Pitfall |
|---|---|---|---|
| Click | Impression -> play | Ranker head | Clickbait; position bias |
| Watch time | Seconds watched (or fraction) | Ranker head; wLR weight [19] | Biased toward long videos |
| Completed / >50% watched | Binary relevance | Retrieval + ranker positive [10] | Short videos inflate |
| Like / share / subscribe | Explicit positive | Satisfaction head | Sparse |
| Dismiss / "not interested" / dislike | Explicit negative | Negative head | Sparse, high value |
| Survey ("was this worth your time?") | Satisfaction | Satisfaction head [22] | Very sparse, needs weighting |
| Implicit negative | Shown, not clicked | Ranker negatives | Not a true negative [9] |
| Random negatives | Sampled from corpus | Retrieval negatives | Needs logQ correction [21] |

## Features

| Group | Examples | Encoding |
|---|---|---|
| User | id, age, gender, geo, language, device | Embeddings / one-hot / bucketize [10] |
| User history | last N watched ids, search tokens, liked channels | Avg pooling or sequence model (Transformer/GRU) |
| Video | id, channel, topic, tags, duration, upload age, language | Embeddings; BERT for title/desc; CBOW for tags [10] |
| Video content | frame / audio embeddings | Pretrained encoders (cold start) |
| Video stats | views, CTR, avg watch % (1h/1d/7d windows) | log-normalize; streaming counters |
| Cross | user watched this channel?, #times this video shown to user, time since last watch of channel | Numeric; strongest ranking features [19] |
| Context | time of day, day of week, device, network | One-hot / embeddings |
| Freshness | example_age (training), 0 at serving | Numeric [19] |

## Model choices (baseline -> advanced)

| Stage | Baseline | Better | Advanced | Tradeoff |
|---|---|---|---|---|
| Retrieval | Popular/trending by region | Matrix factorization (WALS) | Two-tower DNN + ANN [1][10] | MF: fast, no side features, cold start fails; two-tower: features, cold-start users, costlier [10] |
| Retrieval (alt) | Item-item co-watch | Graph / sequence (next-video) | Generative/semantic-ID retrieval | Coverage vs. freshness |
| Ranking | Logistic regression / GBDT | Wide & Deep, DCN [5] | MMoE / PLE multi-task [20] | Capacity vs latency; negative transfer between tasks |
| Re-rank | Rules (dedup, caps) | MMR diversity | Slate/RL optimization, bandit explore | Engagement vs. diversity |
| Cold start | Popular in locale | Content-tower embeddings | Explore budget / bandits [10] | Short-term loss vs. learning |

## Loss functions

| Model | Loss |
|---|---|
| Two-tower retrieval | Sampled / in-batch softmax CE with logQ correction [21]; alternative: BCE with random negatives |
| 2016 candidate gen | Sampled softmax over millions of classes (+ importance weighting) [19] |
| 2016 ranker | Weighted logistic regression (positives weighted by watch time) [19] |
| Multi-task ranker | Sum_k lambda_k * L_k: BCE for click/like/dismiss, regression or wLR for watch time [20] |
| Position bias | Shallow tower logit added in training, removed at serving [20] |

## Offline metrics

| Stage | Metric |
|---|---|
| Retrieval | Recall@k (k=100..1000), hit rate of next watch, coverage |
| Ranking | AUC / log-loss per head, nDCG@k, mAP, Precision@k [9][10] |
| List | Diversity (intra-list distance), novelty, freshness share |
| Slices | New users, new videos, languages, regions, devices |
| Counterfactual | IPS / off-policy estimates on logged random traffic |

## Online metrics / A-B

| Type | Metric |
|---|---|
| Primary | Total watch time per user/day, # completed videos [10] |
| Engagement | CTR, sessions/day, session length |
| Satisfaction | Likes, survey score, dismiss rate, dislikes (guardrail) [20][22] |
| Long-term | D7/D28 retention, return visits (holdouts for weeks) |
| Guardrails | Latency p99, borderline/harmful exposure, creator diversity |
| Method | User-randomized A/B, interleaving for rankers, long-term holdbacks |

## Monitoring

| What | How |
|---|---|
| Feature drift / missing features | Per-feature null rate, PSI vs. training distribution |
| Train/serve skew | Log served features; compare to offline recompute |
| Prediction calibration | Predicted vs. observed CTR per bucket and slice |
| Business KPIs | Watch time, CTR, dismiss rate dashboards with alerting |
| Index freshness | Age of ANN index, % of new uploads retrievable |
| Infra | QPS, p50/p99 latency per stage, timeouts -> fallback to cache/popular |

## Failure modes

| Failure | Symptom | Mitigation |
|---|---|---|
| Clickbait | CTR up, watch time / satisfaction down | Optimize watch + satisfaction, not clicks [1][10] |
| Feedback loop / filter bubble | Narrowing topics, popularity bias | Explore slots, diversity re-rank, logQ correction [21] |
| Position bias | Top slots over-learned | Shallow bias tower [20], randomization data |
| Cold start (new video) | Never surfaced | Content embeddings, fresh-upload candidate source, explore |
| Cold start (new user) | Generic feed | Demographic/context features in user tower, onboarding topics |
| Long-video bias | Watch-time objective favors long videos | Normalize by duration, completion-rate labels [10] |
| Stale model | Trends missed | Hourly warm-start / online training [24] |
| Harmful/borderline content | Policy risk | Classifier demotion in re-rank stage [10][22] |

---

## Interviewer follow-ups

- Why not just optimize CTR? What happens to clickbait, and how do you define "relevant"? [1][10]
- Watch time vs. completion vs. satisfaction: how do you combine heads into one score, and who sets the weights? [20]
- Matrix factorization vs. two-tower: when is MF enough? How does each handle new users? [10]
- How do you choose negatives for the two-tower, and why is in-batch sampling biased? [21]
- Why hold out the *next* watch instead of a random watch, and why cap examples per user? [19]
- How do you serve retrieval in < 10 ms over 1B videos (ANN choice, sharding, index refresh)? [4]
- How do you handle cold start for a video uploaded 5 minutes ago? [10][18]
- How do you correct for position and selection bias in logged training data? [20]
- How would you optimize prediction throughput if the user model is an RNN/Transformer? [13]
- How do you monitor the system and detect a silent quality regression? [13]
- How do you ensure diversity and avoid filter bubbles without hurting watch time?
- Offline AUC went up but online watch time went down - why? (skew, proxy-metric mismatch)
- How fresh does the model need to be? Batch daily vs. hourly vs. online training [24]
- Videos vs. ads on the same surface: how would scope change the design? [12]
- How do you evaluate long-term effects (retention) vs. short-term A/B wins?
- How would you adapt this to Shorts/Reels/TikTok (no click, swipe-based feedback)? [8][30]

## Portal comparison

| Portal | Objective / label | Candidate generation | Ranking | Distinctive emphasis |
|---|---|---|---|---|
| ByteByteGo (Aminian & Xu) [1] | Relevance = like or partial watch; warns against pure clicks | Multiple sources incl. two-tower, popular, trending | Heavier model on fewer candidates | Clean 3-stage funnel, MF vs. two-tower tradeoff |
| AIMLInterviews (alirezadir) [10] | Like or >=50% watched | CF two-tower chosen over MF; popularity + trending | FFN or DCN; multi-task option | Free; detailed feature encoding table, re-rank rules |
| Educative Grokking [3][4][5] | Engagement vs. satisfaction conflict | Two-tower + contrastive negatives + ANN (<10 ms, >1B videos) | Wide & Deep / DCN, re-ranking lesson | Retrieval latency/recall/index-freshness tradeoffs |
| interviewing.io (Google mock) [6] | Maximize engagement | Two-stage, 2B users / 10B videos | Ranking stage within < 1 s | Real interviewer pressure, scale reasoning |
| Hello Interview [7][8] | Watch time vs. satisfaction | Retrieval stage, CF vs. content, embeddings | Two-stage pattern | Pattern-based reuse across rec problems; TikTok variant |
| YouTube papers [19][20][21] | Expected watch time -> multi-objective | Softmax -> two-tower w/ logQ | wLR -> MMoE + bias tower | The production ground truth all portals mirror |

## References

1. "Video Recommendation System" (Machine Learning System Design Interview, ch. 6, A. Aminian & A. Xu), ByteByteGo, 2023. https://bytebytego.com/courses/machine-learning-system-design-interview/video-recommendation-system
2. "How Video Recommendations Work - Part 1", ByteByteGo Newsletter, 2024. https://blog.bytebytego.com/p/how-video-recommendations-work-part
3. "Video Recommendation: Problem Framing and Requirements", Educative (Grokking the ML System Design Interview), 2025. https://www.educative.io/courses/grokking-the-machine-learning-system-design-interview/video-recommendation-problem-framing-and-requirements
4. "Video Recommendation: Candidate Retrieval Architecture", Educative, 2025. https://www.educative.io/courses/grokking-the-machine-learning-system-design-interview/video-recommendation-candidate-retrieval-architecture
5. "Neural Architectures for Ranking and Retrieval", Educative, 2025. https://www.educative.io/courses/grokking-the-machine-learning-system-design-interview/neural-architectures-for-ranking-and-retrieval
6. "Google System Design Mock: YouTube Recommendation System", interviewing.io, n.d. https://interviewing.io/mocks/google-system-design-youtube-recommendation-system
7. "Recommendation Systems (ML System Design pattern)", Hello Interview, 2025. https://www.hellointerview.com/learn/ml-system-design/patterns/recommendation-systems
8. "Design TikTok Video Recommendation System" (community question), Hello Interview, 2025. https://www.hellointerview.com/community/questions/tiktok-video-recommendation/cmgzuubzb02eq08adu99oof84
9. "ML System Design" guide, alirezadir/Machine-Learning-Interviews, GitHub, 2023. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/ml-system-design.md
10. "Video/Movie Recommendation" case study, alirezadir/AIMLInterviews, GitHub, 2023. https://github.com/alirezadir/AIMLInterviews/blob/main/src/MLSD/mlsd-video-recom.md
11. "machine-learning-interview" (Khang Pham), GitHub, 2021. https://github.com/khangich/machine-learning-interview
12. "Got asked how to design YouTube recommendation system", Blind, n.d. https://www.teamblind.com/post/got-asked-how-to-design-youtube-recommendation-system-hdao6bge
13. "Google ML SWE interview - ML design question", Blind, n.d. https://www.teamblind.com/post/google-ml-swe-interview-ml-design-question-tokqydu7
14. "Meta - help for ML design round feedback", Blind, n.d. https://www.teamblind.com/post/meta-help-for-ml-design-round-feedback-nrxobup2
15. "Meta E4 Seattle Dec 2021 [Offer]", LeetCode Discuss, 2021. https://leetcode.com/discuss/interview-experience/1727566/meta-e4-seattle-dec-2021-offer
16. "Coupang Machine Learning Engineer Interview: YouTube Recommendation System Design", 1point3acres, n.d. https://www.1point3acres.com/interview/thread/1087366
17. "Design an ML pipeline for video recommendation", Codemia, 2025-2026. https://codemia.io/practice/design-an-ml-pipeline-for-video-recommendation
18. "Design a YouTube Recommendation System", Codemia, n.d. https://codemia.io/practice/design-a-youtube-recommendation-system
19. P. Covington, J. Adams, E. Sargin, "Deep Neural Networks for YouTube Recommendations", RecSys / Google Research, 2016. https://research.google/pubs/pub45530/ (PDF: https://storage.googleapis.com/pub-tools-public-publication-data/pdf/45530.pdf)
20. Z. Zhao, L. Hong, et al., "Recommending What Video to Watch Next: A Multitask Ranking System", RecSys / Google Research, 2019. https://research.google/pubs/recommending-what-video-to-watch-next-a-multitask-ranking-system/
21. X. Yi, J. Yang, L. Hong, et al., "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations", RecSys / Google Research, 2019. https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/
22. "On YouTube's recommendation system", YouTube Official Blog, 2021. https://blog.youtube/inside-youtube/on-youtubes-recommendation-system/
23. "How YouTube recommendations work", YouTube Help, n.d. https://support.google.com/youtube/answer/16089387?hl=en
24. Z. Liu et al., "Monolith: Real Time Recommendation System With Collisionless Embedding Table" (ByteDance), arXiv, 2022. https://arxiv.org/abs/2209.07663
25. A. Colyer, "Deep neural networks for YouTube recommendations", The Morning Paper, 2016. https://blog.acolyer.org/2016/09/19/deep-neural-networks-for-youtube-recommendations/
26. "How to design a recommendation system in an ML interview", techinterview.org, 2026. https://www.techinterview.org/post/3233476954/how-to-design-a-recommendation-system-ml-interview/
27. "System Design Example: TikTok Recommendation System", Jointaro, n.d. https://www.jointaro.com/course/master-the-machine-learning-interview-as-a-software-engineer/system-design-example-tiktok-recommendation-system/
28. "ML System Design: Model Selection", Exponent, n.d. https://www.tryexponent.com/courses/ml-system-design/mlsd-model-selection
29. C. Huyen, "Machine Learning Systems Design - Exercises", GitHub, 2019. https://github.com/chiphuyen/machine-learning-systems-design/blob/master/content/exercises.md
30. "ML Design Interview | Design a short-video platform Part 2", RecSys ML Substack, n.d. https://recsysml.substack.com/p/ml-design-interview-design-a-short
31. "Design Video Recommendation System", AlgoMaster, n.d. https://algomaster.io/learn/ml-system-design/design-video-recommendation-system
