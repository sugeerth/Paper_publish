# 05. Design People You May Know (PYMK) / Friend Recommendation (Graph-Based)

> Prompt as usually given: "Design the People You May Know feature. Given a user, return a
> ranked list of other users they are likely to connect with."
> Variants: Facebook PYMK, LinkedIn PYMK (connection invites), Twitter/X "Who to Follow"
> (asymmetric follow), Snapchat "Quick Add", Instagram "Suggested for you".

## Company tags

| Company (reported) | Where it was reported | Notes |
|---|---|---|
| Meta / Facebook | alirezadir AIMLInterviews question list: "Friends / follower recommendation: Facebook, Twitter, LinkedIn" [2] | Symmetric friendship; maximize connections formed [1] |
| LinkedIn | Same list [2]; Kao's write-up frames it "similar to LinkedIn" [6] | Production PYMK uses LiGNN embeddings [10]; equity-aware PYMK [13] |
| Twitter / X | Same list [2] | Asymmetric "Who to Follow"; open-sourced FRS [7][8], WTF paper [9] |
| Snap (production, not a reported ask) | Snap Research papers on friend-recommendation EBR [11][12] | Useful "real system" to cite in the interview |

Note: company tags for this problem come mainly from the AIMLInterviews list [2]. Blind, LeetCode
and Prepfully pages could not be reached in this run.

**Portals that cover it**

| Portal | Treatment | Link |
|---|---|---|
| ByteByteGo (Aminian & Xu, MLSDI ch. 11 "People You May Know") | Full chapter. Graph edge prediction with a GNN, FoF candidates, batch vs. online prediction | [3] |
| alirezadir AIMLInterviews (GitHub, free) | Outline: 1B users, pointwise vs. edge-prediction framing, GNN, FoF, batch precompute | [1][2] |
| shafaypro CrackingMachineLearningInterview (GitHub, free) | Deepest free write-up: 5-stage funnel, P(send) x P(accept), leakage, interference, safety | [4] |
| jsheng0901 MLSD notes (GitHub, free) | GCN -> GraphSAGE -> GAT progression, negative resampling, weekly batch refresh | [5] |
| M. C. J. Kao "MLSD Interview Extension: PYMK" (Medium, mirrored on GitHub) | Serving flowchart, Airflow offline pipeline, ego-cluster A/B testing | [6][16] |
| Khang Pham (Educative / repo), Chip Huyen booklet | Checked: no PYMK chapter (LinkedIn feed and generic recsys only) | [17][18] |

## Requirements & scale

| Dimension | Value | Source |
|---|---|---|
| Input / output | user_id (+ surface context) -> ranked list of candidate users | [1][6] |
| Business goal | Maximize connections formed (accepted, not just sent) | [1][4] |
| Edge type | Symmetric friendship (FB/LinkedIn) vs. asymmetric follow (X) | [1][4] |
| Users | ~1B total users | [1][4] |
| Degree | ~10K connections/user (alirezadir) vs. "few hundred median, long tail of hubs" (shafaypro) | [1][4] |
| DAU | 300M DAU (exercise assumption) | [4] |
| Candidate space | 1B -> ~1M FoF (1K x 1K) -> ~1-5K after scoring/quotas | [1][4] |
| Serving latency | ~50 ms p99 (exercise assumption); rules out request-time graph traversal | [4] |
| Peak QPS | 300M DAU x ~3 PYMK views/day / 86,400 ~ 10K avg, ~30K peak | typical assumption |
| Freshness | Minutes for new users / big graph events; daily for everyone else | [4] |
| Batch refresh | Precompute for active users (last 30 days); refresh daily to weekly | [5][6] |
| Non-functional | Privacy (contacts, minors), blocks applied instantly, fallback = hide module | [1][4] |

---

## Diagram 1 - Online serving path

**Figure 5.1 - PYMK request: precomputed list + read-time filters + light re-score**

```
 +---------+  GET /pymk(user_id, surface)   +-------------------------+
 | Client  |------------------------------->|   PYMK Service          |
 | (app)   |<-------------------------------|   A/B bucket (ego-      |
 +---------+   top ~20 cards + "N mutual"   |   cluster), fallback    |
                                            +-----------+-------------+
                                                        | user_id
              +-----------------------------------------+
              v                                         | cache miss / new user
 +---------------------------+                          v
 | Precomputed PYMK store    |          +-------------------------------------+
 | (KV: user -> 100-500      |          |  Online candidate generation        |
 |  scored candidates,       |          | FoF (hub-capped), contacts (opt-in) |
 |  daily batch)             |          | co-membership, GNN EBR (ANN)        |
 +-------------+-------------+          +------------------+------------------+
               | candidates                                | ~1-5K candidates
               +-------------------+-----------------------+
                                   v
                 +-------------------------------------+   +--------------------+
                 | Hard filters (read time)            |<--| Safety KV sets:    |
                 | already friends, pending, blocked,  |   | blocks, removals,  |
                 | "don't suggest", minors/age rules   |   | pending requests   |
                 +------------------+------------------+   +--------------------+
                                    | ~500
                                    v
 +----------------+  features  +-------------------------------------+
 | Feature store  |----------->| Ranker (light GBDT -> heavy MTL)    |
 | online: mutual |            | P(send) x P(accept|send) - P(report)|
 | counts, degree,|            +------------------+------------------+
 | recent activity|                               | ~50 scored
 +----------------+                               v
 +----------------+  weights   +-------------------------------------+
 | Model store    |----------->| Re-rank / policy                    |
 | (registry)     |            | source diversity, recipient request |
 +----------------+            | caps, dedup across surfaces,        |
                               | impression/dismiss cooldowns        |
                               +------------------+------------------+
                                                  | top ~20 + social proof
                                                  v
                               +-------------------------------------+
                               | Response + impression logging       |
                               | (Kafka: impression, send, accept,   |
                               |  dismiss, "don't know", report)     |
                               +-------------------------------------+
```

- Graph events (accept, signup, contact upload, block) trigger incremental refresh of the
  affected users' lists via a stream processor [4].
- If the store is unavailable, hide the module; never show unfiltered candidates [4].

---

## Diagram 2 - Offline training & data pipeline

**Figure 5.2 - Snapshot-at-t / label-at-t+1 pipeline with batch precompute**

```
 +-------------------+   +-------------------+   +-----------------------------+
 | Social graph      |   | Interaction logs  |   | PYMK impression logs        |
 | snapshots (edges, |   | profile views,    |   | impression/send/accept/     |
 | timestamps)       |   | co-comments, msgs |   | dismiss/report (Kafka)      |
 +---------+---------+   +---------+---------+   +--------------+--------------+
           |                       |                            |
           +-----------+-----------+----------------------------+
                       v
 +-----------------------------------------------------------------------+
 | Label builder                                                         |
 | features @ snapshot t --> labels = edges formed in (t, t+window]      |
 | send = impression->request; accept = request accepted within 14 days  |
 | negatives = seen-not-sent impressions (+ hard negatives), NOT random  |
 | drop target edge from feature graph (leakage)                         |
 +----------------------------------+------------------------------------+
                                    v
 +---------------------------------------+   +-------------------------------+
 | Feature pipeline (point-in-time join) |-->| Offline feature store         |
 | CN, Adamic-Adar, Jaccard, PPR, degree,|   | (same code as online path)    |
 | shared school/company, embeddings     |   +-------------------------------+
 +------------------+--------------------+
                    v
 +---------------------------------------+   +-------------------------------+
 | Training (Airflow/Flyte DAG)          |   | GNN embedding job             |
 | GNN edge model + MTL ranker           |<--| GraphSAGE / PyTorch-BigGraph  |
 | temporal split: train < T <= eval     |   | on snapshot t (no future)     |
 +------------------+--------------------+   +---------------+---------------+
                    v                                        |
 +---------------------------------------+                   v
 | Offline eval: recall@k on future      |   +-------------------------------+
 | edges, AUC/logloss per head, slices   |   | ANN index (node embeddings)   |
 | (new users, low-degree, minors)       |   | for EBR candidate source      |
 +------------------+--------------------+   +-------------------------------+
                    v pass
 +---------------------------------------+
 | Model registry (MLflow)               |
 +------------------+--------------------+
                    v
 +---------------------------------------+     +-------------------------------+
 | Batch prediction: active users (30d)  |---->| Precomputed PYMK store (KV)   |
 | FoF + PPR + EBR -> rank -> top 100-500|     | read by online service        |
 +---------------------------------------+     +---------------+---------------+
                    ^                                          |
                    |      new edges / feedback (loop)         v
                    +------------------------------------- user actions
```

- Retrain on a schedule (weekly/monthly) or on graph-drift alarms [5].
- Accepts arrive days late: use a fixed collection window or treat recent rows as censored [4][5].

---

## Diagram 3 - Model architecture

**Figure 5.3 - GNN node encoder + pair scorer with multi-task heads**

```
   viewer u                                         candidate v
 +------------------+                            +------------------+
 | node features:   |                            | node features:   |
 | profile (school, |                            | profile, degree, |
 | company, city),  |                            | account age,     |
 | degree, activity |                            | accept rate      |
 +--------+---------+                            +---------+--------+
          |   sampled k-hop neighbourhood (GraphSAGE)     |
          v                                               v
 +------------------+   shared weights           +------------------+
 | GNN encoder      |<-------------------------->| GNN encoder      |
 | 2-3 layers,      |                            | 2-3 layers,      |
 | mean/attn aggr.  |                            | mean/attn aggr.  |
 +--------+---------+                            +---------+--------+
          | h_u (e.g. 128-d)                               | h_v
          +----------------------+-------------------------+
                                 |
          +----------------------+----------------------+
          |  (A) retrieval: sigmoid(h_u . h_v) -> ANN   |
          |      trained with BCE / sampled softmax     |
          +----------------------+----------------------+
                                 |  (B) ranking path
                                 v
 +-----------------------------------------------------------------------+
 | Pair feature vector = [h_u, h_v, h_u*h_v,  CN, Adamic-Adar, Jaccard,  |
 |  PPR(u->v), PPR(v->u), shortest path, shared school/company/groups,   |
 |  profile views, source id + rank, surface, position]                  |
 +----------------------------------+------------------------------------+
                                    v
                    +-------------------------------+
                    | Shared MLP tower (ReLU, BN,   |
                    | dropout)                      |
                    +---+-----------+-----------+---+
                        |           |           |
                        v           v           v
                 +-----------+ +-----------+ +-----------+
                 | P(send)   | | P(accept  | | P(report /|
                 | on impr.  | |  | send)  | | dont know)|
                 +-----+-----+ +-----+-----+ +-----+-----+
                       |             |             |
                       +------+------+-------------+
                              v
        score = P(send) x P(accept|send) x value(edge) - w x P(report)
```

- (A) alone is the ByteByteGo / AIMLInterviews answer (GNN edge prediction) [1][3].
- (B) adds the reciprocity objective (ESMM-style, like CTR x CVR in ads) [4].

---

## Diagram 4 - Real systems: Twitter/X FRS and Snap / LinkedIn graph embeddings

**Figure 5.4 - Twitter/X Follow Recommendations Service (open-sourced 2023) [7][8][9]**

```
 +------------------+   display location (NUX, home WTF, FutureGraph, ...)
 | Product / flow   |-------------------------------------------+
 +------------------+                                           v
 +-----------------------------------------------------------------------+
 | 1. Candidate sources (common/candidate_sources/)                      |
 |  two_hop_random_walk, salsa, triangular_loops, socialgraph,           |
 |  addressbook, real_graph, stp (strong-tie prediction), sims,          |
 |  sims_expansion, recent_engagement, geo, top_organic_follows, ...     |
 +----------------------------------+------------------------------------+
                                    v
 +-----------------------------------------------------------------------+
 | 2. Filters / predicates: cheap ones pre-rank, heavy ones post-rank    |
 +----------------------------------+------------------------------------+
                                    v
 +-----------------------------------------------------------------------+
 | 3. Ranker: hydrate features -> DataRecord per (user, candidate)       |
 |    score = weighted sum of p(follow | rec) and p(positive eng | follow)|
 +----------------------------------+------------------------------------+
                                    v
 +-----------------------------------------------------------------------+
 | 4. Transforms: dedup, social proof ("followed by ..."), tracking tok. |
 +----------------------------------+------------------------------------+
                                    v
                        5. Truncate --> WTF module
```

- Each display location (post-signup NUX, home WTF, FutureGraph tweets) selects its own flow of
  sources, rankers and filters; candidate-source names above are the folders in the repo [7][20].

**Figure 5.5 - Embedding retrieval added beside FoF (Snap EBR [11][12]; LinkedIn LiGNN [10])**

```
 +---------------+    FoF reaches little for cold-start and is costly past 2 hops
 | Friend graph  |-----------------------------+
 +-------+-------+                             v
         |                       +--------------------------+
         | GNN training          | FoF candidates (2-hop)   |---+
         v                       +--------------------------+   |
 +---------------+  embeddings   +--------------------------+   |  union
 | GNN encoder   |-------------->| ANN index (billions of   |---+-----> ranker
 | (users)       |               | users), query expansion  |   |
 +---------------+               +--------------------------+   |
                                 +--------------------------+   |
                                 | LiGNN feature for PYMK   |---+
                                 | ranker (~+0.1% WAU) [10] |
                                 +--------------------------+
```

- Snap: EBR improved friendships made in both sparse and dense markets [11]; ANN query
  expansion gave ~11% relative gain in reciprocated communication, 800M+ MAU [12].

---

## Data & labels

| Item | Choice | Notes / source |
|---|---|---|
| Positive (edge model) | Edge formed in (t, t+1] that was absent at snapshot t | [1][5][6] |
| Positive (ranker) | send on impression; accept given send; edge used within 28 days | [4] |
| Negatives | Seen-but-not-sent impressions; hard negatives (FoF not connected) | Random pairs teach "rank by degree" [4][5] |
| Explicit negatives | Dismiss, "I don't know this person", ignored/declined, report | Separate heavily-weighted head [4] |
| Resampling | Re-sample negatives each epoch, keep class balance | [5] |
| Label window | Fixed 14-day accept window, or censor recent sends | [4][5] |
| Split | Temporal: train before cutoff, evaluate after | Random edge splits overstate quality [4] |

## Features

| Group | Examples | Source |
|---|---|---|
| Neighbourhood overlap | Common neighbours (time-discounted), Adamic-Adar, Jaccard, resource allocation | [1][4] |
| Paths / walks | Shortest path (cap 3-4), 2/3-hop path counts, PPR both directions | [4][9][14] |
| Degree | Degrees, degree ratio, incoming request volume, clustering coefficient | [4][5] |
| Shared context | Same school + years overlap, employer, industry, city, groups, contact match | [1][5] |
| Interaction (no edge) | Profile views, co-tags, comments on same posts (feature only, never shown) | [4] |
| Embeddings | GNN / node2vec cosine, pair score | [4][10] |
| User | Account age, activity, accept rate of received requests, device, time | [1][5] |
| Source | Which generator produced the candidate + rank within it | [4] |

## Model choices (baseline -> advanced)

| Stage | Option | Pros | Cons |
|---|---|---|---|
| Heuristic | Rank FoF by mutual-friend count / Adamic-Adar | Cheap, explainable ("12 mutual") | Hubs dominate; no cold start |
| Walk baseline | Personalized PageRank / SALSA random walk with restart | Reaches 3+ hops, down-weights hubs | Hard to add side features [9] |
| Learned walk | Supervised random walks (edge weights learned from features) | Combines structure + attributes | Older, costly per-user [14] |
| Pointwise | GBDT / LR on pair features | Strong light ranker, fast batch | Ignores graph beyond hand features [1] |
| GNN | GCN -> GraphSAGE (inductive, sampled) -> GAT (attention edge weights) | Uses structure + features; new users | Training scale, leakage risk [5][15][19] |
| MTL ranker | Shared tower, send / accept / report heads | Models reciprocity, safety | More labels, calibration needed [4] |
| EBR | GNN embeddings in ANN as extra source | Beyond-2-hop recall, cold start | "Similar stranger" risk [4][11] |

## Loss

| Component | Loss | Notes |
|---|---|---|
| GNN edge prediction | BCE on sigmoid(h_u . h_v) with sampled negatives | [1][5] |
| EBR retrieval | Sampled softmax over in-batch / sampled users | Snap friend EBR pattern [11] |
| MTL ranker | sum_k w_k BCE_k (send, accept|send, report) | ESMM or IPW for accept selection bias [4] |
| Regularization | Dropout, BN, residuals, early stopping | [5] |

## Offline metrics

| Metric | Measures | Source |
|---|---|---|
| Recall@k on future edges | Candidate source coverage | [4] |
| ROC-AUC, log loss per head | Edge / head discrimination | [1][4][5] |
| mAP, NDCG@k, precision@k | Ranked list quality | [1][4] |
| Calibration per head | Heads are multiplied, so must be calibrated | [4] |
| Sliced metrics | New users, low-degree, minors, with/without contacts | [4] |

## Online metrics / A-B

| Type | Metric | Source |
|---|---|---|
| Primary | Accepted connections per viewer; requests sent and accepted per window | [1][4][5] |
| Long term | Connections with interaction within 28 days; new-user retention; WAU | [4][10] |
| Funnel | Impressions -> send rate -> acceptance rate | [4] |
| Guardrails | "Don't know"/remove rate, reports, pending backlog, requests to minors | [4] |
| Design | Interference (treated users invite control users): ego-cluster or graph-cluster randomization; long-term holdouts | [4][6][16] |

## Serving & latency budget (~50 ms p99 assumption [4])

| Step | Budget | Notes |
|---|---|---|
| Fetch precomputed list (KV) | ~5 ms | 100-500 candidates per user [4] |
| Read-time hard filters | ~5 ms | KV sets for blocks, pending, friends |
| Online feature fetch | ~10 ms | Mutual counts, recent activity |
| Light re-score (optional) | ~15 ms | GBDT or small MLP on ~500 |
| Re-rank + social proof | ~5 ms | Diversity, caps, explanations |
| Network / slack | ~10 ms | Typical assumption |
| Cache miss / new user | Async | Online FoF + EBR with lightweight ranker, then write back [5][6] |

## Monitoring

| Signal | What to watch |
|---|---|
| Data | Feature distribution drift, degree-distribution change, graph-structure alarms [5] |
| Model | AUC/calibration per head on fresh data, score distributions |
| System | p99 latency, KV hit rate, batch job freshness/SLA, training cost [5] |
| Ecosystem | Incoming-request concentration (rich-get-richer), cross-cluster edge share [4][13] |
| Safety | Reports, "don't know", adult-to-minor suggestion rate (should be 0) [4] |

## Failure modes

| Failure | Mitigation |
|---|---|
| Hub explosion in FoF (one friend with 100K edges) | Sample/cap hubs, Adamic-Adar 1/log(deg) weighting [4] |
| Target / future-edge leakage | Point-in-time joins, drop target edge, temporal split [4][6] |
| Optimizing sends only -> spam to popular users | P(accept|send) term, recipient caps [4][7] |
| Rich-get-richer, homophily | Exposure caps, degree discounts, exploration, equity-aware objective [4][13] |
| Cold start (no edges) | Contacts (consented), onboarding profile via inductive GNN, inviter's friends [4][11] |
| Creepy / privacy-leaking suggestions | No sensitive sources, explanations only from visible signals [4] |
| Self-confirming evaluation | Exploration traffic, long-term holdouts [4] |
| A/B contamination via network | Ego-cluster / graph-cluster randomization [6][16] |

## Interviewer follow-ups

- Symmetric friendship or asymmetric follow? How does the objective change? [1][4]
- How do you generate candidates on a 1B-user graph without request-time traversal? [1][4]
- Why is Adamic-Adar often better than common-neighbour counts? [4]
- Train on send, accept, or edge usage? How do you handle accept selection bias? [4]
- How can link-prediction features leak, and how do you split train/test? [4][6]
- Pointwise classifier vs. GNN edge prediction: what does each miss? [1][5]
- GCN vs. GraphSAGE vs. GAT; how do you train a GNN on billions of nodes? [5][15]
- Batch precompute vs. online prediction; how fresh must new users' lists be? [3][4][5]
- Brand-new user with zero friends: what do you show? [4][11]
- A/B test shows +5% connections; what could be wrong (network interference)? [4][6]
- How do you avoid rich-get-richer and suggesting the same celebrities? [4][13]
- What makes a suggestion "creepy", and how do you stop it? Minors? [4]
- How do accepts arriving days later affect labels and metrics? [1][5]
- How would you add embedding-based retrieval beside FoF? [10][11]
- Which filters run before ranking and which after? (cheap pre-rank, heavy post-rank) [7]
- Should onboarding (NUX) and the home module share one flow? (per-surface flows) [7]
- How do you rank users who are not active enough to have a precomputed list? [5][6]

## Portal comparison

| Portal | Framing | Candidates | Model | Serving | Distinctive |
|---|---|---|---|---|---|
| ByteByteGo [3] | Graph edge prediction | FoF | GNN | Batch vs. online prediction trade-off | Polished diagrams; canonical book answer |
| alirezadir [1] | Edge prediction over pointwise LTR | FoF (1K x 1K = 1M) | GNN embeddings + similarity | Batch precompute to DB | Notes delayed feedback, PRW baseline |
| shafaypro [4] | Two-sided funnel value score | FoF, PPR, EBR, contacts, co-membership | GBDT light + MTL heavy | Batch + stream refresh + read-time filters | Leakage, interference, safety, minors |
| jsheng0901 [5] | Edge prediction | FoF | GCN -> SAGE -> GAT, dot + sigmoid | Weekly batch + online fallback | Negative resampling, Cluster-GCN scaling |
| Kao (Medium) [6] | GNN pair score | FoF, same company/school | GNN | Precomputed DB, scoring on miss | Ego-cluster A/B testing, Airflow DAG |
| Real systems [7]-[12] | Follow / friend rec | Walks, address book, embeddings | p(follow) + p(engage|follow) | Multi-source union | FRS open source; Snap/LinkedIn GNN EBR |

## References

1. alirezadir, "People You May Know (PYMK)" (mlsd-pymk.md), AIMLInterviews, GitHub, 2025. https://github.com/alirezadir/AIMLInterviews/blob/main/src/MLSD/mlsd-pymk.md
2. alirezadir, "ML System Design" question list (ml-system-design.md), AIMLInterviews, GitHub, 2026. https://github.com/alirezadir/AIMLInterviews/blob/main/src/MLSD/ml-system-design.md
3. A. Aminian & A. Xu, "People You May Know" (Machine Learning System Design Interview, ch. 11), ByteByteGo, 2023. https://bytebytego.com/courses/machine-learning-system-design-interview/people-you-may-know
4. shafaypro, "People You May Know" (system_design/people_you_may_know.md), CrackingMachineLearningInterview, GitHub, n.d. https://github.com/shafaypro/CrackingMachineLearningInterview/blob/master/system_design/people_you_may_know.md
5. jsheng0901, "Friends/Follower Recommendation (PYMK)" (mlsd_pymk.md), Machine-Learning-Implementation, GitHub, n.d. https://github.com/jsheng0901/Machine-Learning-Implementation/blob/main/machine_learning_system_design/mlsd_pymk.md
6. M. C. J. Kao, "Machine Learning System Design Interview Extension: People You May Know", Medium (clipped in alexsnow348/learning-notes), GitHub, 2024. https://github.com/alexsnow348/learning-notes/blob/main/learning-notes/machine-learning-and-ai-notes/ml-interview/People%20You%20May%20Know.md
7. Twitter/X, "Follow Recommendations Service" README, the-algorithm, GitHub, 2023. https://github.com/twitter/the-algorithm/blob/main/follow-recommendations-service/README.md
8. Twitter/X, "the-algorithm" repository, GitHub, 2023. https://github.com/twitter/the-algorithm
9. P. Gupta, A. Goel, J. Lin, A. Sharma, D. Wang, R. Zadeh, "WTF: The Who to Follow Service at Twitter", WWW, 2013. https://web.stanford.edu/~rezab/papers/wtf_overview.pdf
10. F. Borisyuk et al., "LiGNN: Graph Neural Networks at LinkedIn", KDD / arXiv, 2024. https://arxiv.org/abs/2402.11139
11. Snap Research, "Embedding Based Retrieval in Friend Recommendation", SIGIR, 2023. https://research.snap.com/publications/embedding-based-retrieval-in-friend-recommendation.html
12. Snap Research, "Improving Embedding-Based Retrieval in Friend Recommendation with ANN Query Expansion", research.snap.com, n.d. https://research.snap.com/publications/improving-embedding-based-retrieval-in-friend-recommendation-with-ann-query-expansion.html
13. LinkedIn Engineering, "Optimizing PYMK for equity in network creation", LinkedIn Engineering Blog, 2021. https://engineering.linkedin.com/blog/2021/optimizing-pymk-for-equity-in-network-creation
14. L. Backstrom & J. Leskovec, "Supervised Random Walks: Predicting and Recommending Links in Social Networks" (Facebook data), WSDM / arXiv, 2011. https://arxiv.org/abs/1011.4071
15. W. L. Hamilton, R. Ying, J. Leskovec, "Inductive Representation Learning on Large Graphs" (GraphSAGE), NeurIPS / arXiv, 2017. https://arxiv.org/abs/1706.02216
16. G. Saint-Jacques et al. (LinkedIn), "Using Ego-Clusters to Measure Network Effects at LinkedIn", arXiv, 2019. https://arxiv.org/abs/1903.08755
17. K. Pham, "machine-learning-interview" (checked: no PYMK chapter), GitHub, 2021. https://github.com/khangich/machine-learning-interview
18. C. Huyen, "machine-learning-systems-design" booklet (checked: no PYMK question), GitHub, 2019. https://github.com/chiphuyen/machine-learning-systems-design
19. A. Lerer et al., "PyTorch-BigGraph: A Large-scale Graph Embedding System", Facebook AI / SysML, 2019. https://github.com/facebookresearch/PyTorch-BigGraph
20. Twitter/X, FRS candidate_sources directory, the-algorithm, GitHub, 2023. https://github.com/twitter/the-algorithm/tree/main/follow-recommendations-service/common/src/main/scala/com/twitter/follow_recommendations/common/candidate_sources

> Research note (review pass): web search was exhausted and non-GitHub hosts were unreachable.
> [1], [2], [4]-[7], [20] were re-read by cloning the repositories (alirezadir: "1 B total users,
> on avg. 10000 connection per user"; Kao: active users "in the last, say, 30 days"; FRS
> candidate-source folders and p(follow)/p(engage) score). [9]-[16], [19] are cross-checked against
> verified chapters of this repository (recsys-system-design ch. 2, 7, 23). [3] could not be
> fetched; its URL follows ByteByteGo's chapter-URL pattern. QPS and latency splits are assumptions.
