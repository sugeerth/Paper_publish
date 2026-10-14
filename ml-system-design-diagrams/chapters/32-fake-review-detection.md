# 32. Design Fake Review Detection and Review Ranking (Amazon, Yelp)

> Prompt as usually given: "Design a system that detects fake / incentivized / spam
> reviews on an e-commerce or local-business platform, and ranks the remaining reviews so
> the most helpful ones appear first." Variants: opinion-spam detection, fake reviewer /
> review-ring detection, "Most helpful" review ordering, Yelp "recommended vs. not
> recommended" reviews, app-store review fraud, LLM-generated review detection.
>
> Sourcing note: non-GitHub hosts (arxiv.org, acm.org, aaai.org, aboutamazon.com,
> yelp.com) could not be reached from the review sandbox and web search was exhausted.
> Tags: (fetched) = page read and claim checked. (repo-link) = URL listed verbatim in a
> fetched GitHub page. (ch-ref) = URL verified in another chapter. (known) = canonical
> page cited from memory, not re-fetched. No portal lists this exact prompt; company
> tags below are evidence of the system, not interview reports. "Typical assumption"
> numbers are interview estimates, not company disclosures.

## Company tags

| Company | Evidence | Notes | Source |
|---|---|---|---|
| Amazon | Mirrors it: Amazon review graph (upu, usu, uvu relations) is a standard fake-reviewer benchmark (CARE-GNN, DGL FraudAmazon) | Product reviews + "Most helpful" ordering; no Amazon production paper verified here | [1][3][28] |
| Yelp | Mirrors it: YelpChi labels come from Yelp's own review filter ("not recommended") | Filter behavior studied externally [4]; embeddings blog is adjacent (not spam) | [4][5][7][28] |
| Alibaba (Xianyu) | Built it: GAS - GCN-based spam review detection deployed on Xianyu second-hand marketplace | Best documented production "review spam" GNN | [8][9] |
| App-store review systems | Academic: FdGars detects fraudsters in an online app review system (WWW 2019 workshop) | No company tag claimed | [10] |
| Meta, LinkedIn, Pinterest | Adjacent only: integrity / spam systems cited by ByteByteGo ch. 5 and case-study lists, not review prompts | Same actor-graph + content-classifier pattern | [12][13][29][30] |

**Portals that cover it**

| Portal | Treatment | Link |
|---|---|---|
| alirezadir Machine-Learning-Interviews | No dedicated review prompt; closest listed: "Harmful content / Spam detection", "Fraud detection", "Sentiment analysis", "Proximity service / Yelp" | [14] |
| ByteByteGo (Aminian & Xu, MLSDI) ch. 5 | Harmful content detection template (multimodal, multi-task, harmful actors as follow-up) is the usual base | [15][16] |
| Educative "Grokking the ML Interview" / Khang Pham MLSD | Classification + ranking templates; reuse for detector + helpfulness ranker | [17][18][19] |
| Chip Huyen MLSD exercises | Closest: fraud classification exercise with labelled records | [20] |
| Exponent, Hello Interview, IGotAnOffer, Patrick Halina | Generic MLSD frameworks only; no review-fraud prompt verified | [21][22][23][24] |
| Engineer1999 case-study list | No review-fraud case; nearest are Pinterest spam clustering, LinkedIn viral spam, Whatnot GenAI trust & safety | [27] |
| safe-graph paper lists (academic) | Canonical graph-based review-spam methods (GAS, CARE-GNN, GraphConsis, FdGars) | [8][10][25] |

## Requirements & scale

| Dimension | Requirement | Source / note |
|---|---|---|
| Functional 1 | Score every new review for fake / spam / incentivized / LLM-generated; actions: publish, hide ("not recommended"), hold for human review, remove + penalize account | Typical assumption |
| Functional 2 | Detect fake reviewers and review rings (groups), retroactively remove reviews | [3][11] |
| Functional 3 | Rank surviving reviews per product/business ("Most helpful", "Recommended") with recency, diversity, personalization | [4][6] |
| Non-functional | Precision first on removal (false removal hurts trust, legal risk); recall on rings; explainable decisions for appeals | Typical assumption |
| Review write QPS | ~50-500 reviews/s peak (tens of millions/year) | Typical assumption |
| Review read QPS | ~50k-200k review-page renders/s (ranking served from cache) | Typical assumption |
| Latency | Sync pre-publish check p99 < 200 ms; deep graph/LLM checks async in minutes; ranking read p99 < 50 ms | Typical assumption |
| Catalog / users | 100M-1B products/businesses, 100M-500M reviewers, 1B+ historical reviews | Typical assumption |
| Label prevalence | Fake rate ~5-20% of submissions, highly skewed; YelpChi ~14.5% spam (commonly reported) | [5][7] (known) |

## Diagram 1 - Online architecture

**Diagram 1: Write path (detect) and read path (rank) for reviews**

```
 WRITE PATH (new review)
 +--------+  POST review  +-------------+  sync features  +--------------------+
 | Client |-------------->| Review API  |---------------->| Feature store      |
 +--------+               | (+ rate lim)|<----------------| online: user/item/ |
                          +------+------+   user, device, | device/IP counters |
                                 |          item velocity +--------------------+
                                 v
                    +--------------------------+   +-------------------------+
                    | Stage 0: rules/blocklist |   | Model store / registry  |
                    | (IP, device, paid-review |   | detector vN, ranker vM, |
                    |  keywords, dup hash)     |   | thresholds per locale   |
                    +------------+-------------+   +-----------+-------------+
                                 | pass                        | load
                                 v                             v
                    +------------------------------------------------+
                    | Stage 1: fast detector (GBDT + text encoder)   |
                    | p(fake) < 200 ms                               |
                    +----+------------------+------------------+-----+
                  low    |          medium  |           high   |
                         v                  v                  v
                  +-------------+   +---------------+   +----------------+
                  | publish     |   | publish-pend. |   | hold / reject  |
                  | (eligible   |   | + human queue |   | + account risk |
                  |  for rank)  |   +-------+-------+   +--------+-------+
                  +------+------+           |                    |
                         |                  v                    v
                         |        +-------------------------------------+
                         +------->| Stage 2 async (min-hours):          |
                                  | review-user-item graph GNN, ring    |
                                  | detection, LLM-text detector,       |
                                  | retroactive takedown                |
                                  +------------------+------------------+
                                                     | status + scores
                                                     v
 READ PATH (product page)                    +---------------+
 +--------+ GET reviews  +-------------+     | Review index  |
 | Client |------------->| Ranking svc |<----| (status,      |
 +--------+<-------------| 1 filter    |     | quality score,|
   top-K reviews +       | 2 rank      |     | helpful votes)|
   star summary          | 3 re-rank   |     +---------------+
                         +------+------+
                                | impressions, helpful/unhelpful votes, reports
                                v
                         +-------------+     +--------------------------+
                         | Event log   |---->| Kafka -> warehouse / OLAP|
                         +-------------+     +--------------------------+
```

- Read path: filter (status = published, not fake, language) -> rank (helpfulness model) -> re-rank (diversity of star ratings, recency, verified purchase boost, media) -> cache per item.
- Star rating aggregate recomputed only from trusted reviews (fake-adjusted average).

## Diagram 2 - Offline training & data pipeline

**Diagram 2: Labels, features, training, deployment and feedback loop**

```
 +---------------+  +----------------+  +----------------+  +-------------------+
 | review events |  | moderator      |  | seller / user  |  | external signals  |
 | text, stars,  |  | decisions,     |  | reports,       |  | paid-review group |
 | user, device  |  | appeals        |  | takedowns      |  | stings, lawsuits  |
 +-------+-------+  +-------+--------+  +-------+--------+  +---------+---------+
         |                  |                   |                     |
         v                  v                   v                     v
 +--------------------------------------------------------------------------+
 | Label builder: hard labels (moderator) + weak labels (rules, ring        |
 | membership, deleted accounts) -> point-in-time join, label delay 7-30d  |
 +-------------------------------------+------------------------------------+
                                       |
          +----------------------------+----------------------------+
          v                            v                            v
 +-----------------+      +--------------------------+     +-------------------+
 | Tabular feats   |      | Graph builder            |     | Text/LLM embeds   |
 | user age, burst |      | review-user-item-IP-     |     | encoder over      |
 | rating deviation|      | device hetero graph      |     | title + body      |
 +--------+--------+      +------------+-------------+     +---------+---------+
          |                            |                             |
          +----------------------------+-----------------------------+
                                       | offline + online feature store
                                       v
 +--------------------------------------------------------------------------+
 | Training: detector (GBDT, text, GNN), ring detector, helpfulness ranker  |
 | time-based split, negative downsampling + recalibration                  |
 +-------------------------------------+------------------------------------+
                                       v
 +--------------------------------------------------------------------------+
 | Evaluation: PR-AUC, precision@removal threshold, ring recall, NDCG for   |
 | ranker, slice checks (locale, new users, category), red-team LLM fakes   |
 +-------------------------------------+------------------------------------+
                                       | pass gates
                                       v
 +----------------+  shadow  +-----------------+  canary/A-B  +---------------+
 | Model registry |--------->| shadow scoring  |------------->| production    |
 +----------------+          +-----------------+              +-------+-------+
        ^                                                             |
        |  retrain weekly; new attack patterns -> new rules/features  |
        +-------------------- feedback: decisions, appeals, votes ----+
```

## Diagram 3 - Model architecture

**Diagram 3: Multi-signal fake review detector + helpfulness ranker**

```
 DETECTOR (per review)
 +----------------+  +----------------+  +--------------------+  +----------------+
 | Review text    |  | Behavior feats |  | Graph neighborhood |  | Item context   |
 | title + body   |  | burst, rating  |  | review-user-item   |  | avg rating,    |
 | (multilingual) |  | dev., acct age |  | + IP/device edges  |  | review volume  |
 +-------+--------+  +-------+--------+  +---------+----------+  +-------+--------+
         v                   v                     v                     v
 +----------------+  +----------------+  +--------------------+  +----------------+
 | Transformer    |  | Dense / GBDT   |  | Hetero GNN (GAS /  |  | Embedding +    |
 | encoder [CLS]  |  | leaf embedding |  | CARE-GNN style,    |  | numeric norm   |
 | + LLM-gen head |  |                |  | relation-aware)    |  |                |
 +-------+--------+  +-------+--------+  +---------+----------+  +-------+--------+
         +-------------------+---------+-----------+---------------------+
                                       v  concat / attention fusion
                             +--------------------+
                             | MLP shared trunk   |
                             +----+----------+----+
                                  |          |
                                  v          v
                     +--------------+  +--------------------+
                     | p(fake)      |  | p(incentivized),   |
                     | review head  |  | p(LLM-generated),  |
                     +--------------+  | p(fake reviewer)   |
                                       +--------------------+

 RANKER (per item, published reviews only)
 +--------------------------------------------------------+
 | inputs: text quality emb, length, media, verified buy, |
 | helpful votes (smoothed), recency, reviewer reputation,|
 | p(fake), star rating, viewer locale / prefs            |
 +---------------------------+----------------------------+
                             v
          +-------------------------------------+
          | LambdaMART / DNN: p(helpful vote)   |
          | trained on vote + dwell labels      |
          +------------------+------------------+
                             v
          +-------------------------------------+
          | re-rank: star-mix diversity, dedupe,|
          | freshness, demote p(fake) > t       |
          +-------------------------------------+
```

## Diagram 4 - Real system: Xianyu GAS and review graphs

**Diagram 4: Alibaba Xianyu GAS (GCN-based Anti-Spam) as reported in [8][9]**

```
 +-----------------+  review edge  +-----------------+
 | User node       |-------------->| Item node       |
 | (aggregates its |  (comment     | (aggregates its |
 |  review edges)  |   text emb)   |  review edges)  |
 +--------+--------+               +--------+--------+
          |  heterogeneous bipartite GCN, edge = review to classify
          v                                  v
 +--------------------------------------------------------+
 | Edge repr = concat(user emb, item emb, review text emb)|
 +---------------------------+----------------------------+
                             |   + homogeneous comment graph
                             |   (KNN over similar comments,
                             v    captures spam campaigns)
                 +-----------------------------+
                 | classifier -> spam / normal |
                 +-----------------------------+

 Benchmark review graphs used in papers (CARE-GNN [1]):
   YelpChi: review nodes; relations rur (same user), rtr (same product+month,
            same star), rsr (same product+star)    label sim: rur 0.91, others ~0.18
   Amazon:  user nodes; relations upu (same product), usu (same star in a week),
            uvu (similar review text)              label sim: upu 0.17, others ~0.05
   (label similarity over fraud nodes only, per CARE-GNN README 06/2021 update)
```

- Lesson: user-linked edges (rur) are highly homophilous; product-linked edges are mostly camouflaged, so relation-aware neighbor filtering (CARE-GNN) matters [1].

## Data & labels

| Source | Label type | Issue | Mitigation |
|---|---|---|---|
| Moderator decisions | Hard, high precision | Expensive, biased to flagged items | Active learning, uncertainty sampling |
| Platform filter output (e.g. Yelp not-recommended) | Proxy | Model learns its own past filter | Hold-out random audits |
| Paid-review stings / leaked review-farm lists | Hard positives, rings | Rare, bursty | Label propagation over graph |
| Account bans, deleted accounts | Weak, delayed | Label leakage, delay | Point-in-time joins |
| Helpful / unhelpful votes | Ranker label | Position bias, rich-get-richer | IPW, Bayesian smoothing, exploration slots |
| Synthetic LLM fakes | Augmentation | Distribution mismatch | Red-team generation per release |

## Features

| Group | Examples |
|---|---|
| Text | Encoder embedding, length, sentiment extremity, duplicate/near-dup hash (MinHash), template phrases, LLM-perplexity / detector score |
| Reviewer behavior | Account age, review count, review burst per day, rating deviation from item mean, fraction 5-star, verified-purchase ratio |
| Item behavior | Review velocity spikes, rating distribution shift, seller-linked accounts |
| Network / device | Shared IP, device fingerprint, payment instrument, geo mismatch with business |
| Graph | GNN embeddings over review-user-item-device graph, co-review group density, ring membership |
| Ranking | Helpfulness votes (smoothed), media present, detail/aspect coverage, recency, reviewer reputation |

## Model choices

| Stage | Baseline | Advanced | Tradeoff |
|---|---|---|---|
| Rules | Blocklists, velocity caps | Auto-generated rules from clusters | Fast, explainable; brittle |
| Review detector | Logistic regression / GBDT on behavior + n-grams | Transformer text + GBDT fusion, multi-task | Text alone weak vs. skilled writers |
| Reviewer / ring | Co-review frequent itemset mining | Hetero GNN (GAS, CARE-GNN, GraphConsis), dense-subgraph detection | Graph catches coordination; costly, async |
| LLM-generated | Perplexity heuristics | Fine-tuned detector + LM-embedding + GNN | Arms race, false positives on non-native writers |
| Actor-level | Per-account aggregates | Temporal interaction embeddings over account action sequences (Facebook TIES) | Catches fake accounts before they review; needs event history [29] |
| Moderation assist | Manual queue triage | LLM classifier/explainer for queue items (Whatnot GenAI T&S pattern) | Faster triage + rationale for appeals; cost, prompt drift [30] |
| Ranker | Sort by helpful votes / Wilson score | LambdaMART / DNN with quality + trust features | Votes biased by position |

## Loss

| Model | Loss |
|---|---|
| Detector | Weighted BCE or focal loss per head; multi-task sum with head weights |
| GNN | Node/edge BCE with relation-aware neighbor sampling; class-balanced sampling |
| Ring | Unsupervised density / clustering score, or contrastive loss on group embeddings |
| Ranker | Pairwise / listwise (LambdaRank) on helpful-vote preferences, debiased with IPW |

## Offline metrics

| Model | Metric |
|---|---|
| Detector | PR-AUC, recall at fixed precision (e.g. 95%), calibration (ECE) |
| Ring | Ring recall, precision of top-K suspicious groups |
| Ranker | NDCG@5/10, MAP on held-out vote data |
| Fairness slices | FPR by locale, language, new vs. tenured user, small business |
| Benchmarks | AUC / recall on YelpChi and Amazon fraud graphs (DGL FraudYelp / FraudAmazon) [1][28] |

## Online metrics / A-B

| Metric | Direction |
|---|---|
| Fake prevalence on served reviews (audited sample) | Down |
| Appeal overturn rate, false takedown rate | Down |
| Moderator queue size / time-to-action | Down |
| Helpful-vote rate, review-section CTR, scroll depth | Up |
| Conversion / return rate (e-commerce), booking rate (local) | Up / neutral |
| Seller / business complaints | Down |

- A-B caveat: adversaries adapt; use holdout audit samples and per-market rollouts rather than user-level split for enforcement.

## Serving & latency budget

| Step | Budget (p99) |
|---|---|
| Rules + rate limit | 10 ms |
| Online feature fetch | 30 ms |
| Text encoder (distilled) | 60 ms |
| GBDT + fusion head | 20 ms |
| Decision + write | 30 ms |
| Total sync write path | < 200 ms |
| GNN / ring / LLM async | minutes-hours |
| Ranked review read (cached) | < 50 ms |

## Monitoring

| Signal | Alert on |
|---|---|
| Score distribution drift per locale/category | PSI shift |
| Sudden review velocity on item/seller | Burst attack |
| Hold/reject rate | Spike or collapse |
| Appeal overturn rate | Precision drop |
| Label delay / moderator agreement | Label quality |
| LLM-text detector hit rate | New generator release |

## Failure modes

| Failure | Mitigation |
|---|---|
| Camouflaged fraudsters (aged accounts, mixed honest reviews) | Relation-aware GNN, behavior over long windows |
| LLM-written fakes evade text models | Behavior + graph signals over text-only; red-team retraining |
| Competitor "review bombing" negative fakes | Item-level anomaly detection, temporary lock |
| Honest bursts (viral event, launch) flagged | Context features (traffic, sales), human review |
| Feedback loop: ranker hides reviews -> no votes | Exploration slots, IPW |
| Over-removal hurts small businesses / non-native writers | Slice metrics, high-precision removal, soft demotion |

## Interviewer follow-ups

- How do you get labels when ground truth is mostly unobservable?
- Fake review vs. fake reviewer vs. review ring: which entity do you model and why?
- How would you detect coordinated review groups that individually look normal?
- How do you handle LLM-generated reviews that are fluent and unique?
- Sync vs. async: what must be decided before the review is visible?
- How do you set thresholds for hide vs. remove vs. human review?
- How do you avoid penalizing non-native speakers or new legitimate users?
- How do you recompute the star average once fakes are removed retroactively?
- How do you debias helpful votes for position and age in the ranker?
- How would you diversify top reviews across positive and negative ratings?
- How do you evaluate a detector when the adversary adapts to it?
- What graph would you build, and how do you scale GNN inference to 1B reviews?
- How do you handle incentivized reviews (free product) vs. outright fake?
- How do you explain decisions for appeals and legal compliance?
- Would you model accounts (action sequences) before they post, not just reviews? [29]
- Where can an LLM help (queue triage, rationale) without becoming the sole judge? [30]

## Portal comparison

| Portal | Framing | Core model | Distinctive emphasis |
|---|---|---|---|
| ByteByteGo ch. 5 (adapted) | Harmful content classification | Multimodal multi-task fusion | Harmful actors, human review queue [15] |
| alirezadir | Spam / fraud detection templates | Classifier + features | Checklist of MLSD steps [14] |
| Educative / Khang Pham | Classification then ranking | GBDT / DNN | Metrics, sampling, serving [17][18][19] |
| Chip Huyen exercises | Fraud classification | Supervised with label delay | Data/label issues [20] |
| Academic (safe-graph, CARE-GNN, GAS) | Graph anomaly detection | Hetero GNN | Camouflage, relation-aware sampling [1][8][25] |
| ByteByteGo ch. 5 references | Integrity at scale | Multimodal + actor embeddings (TIES) | Human review system, trending abuse [12][16][29] |

## References

1. Y. Dou et al., "Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters" (CARE-GNN, code + YelpChi/Amazon data), GitHub, 2020. https://github.com/YingtongDou/CARE-GNN (fetched)
2. Y. Dou et al., "Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters", CIKM 2020 / arXiv. https://arxiv.org/abs/2008.08692 (repo-link; venue confirmed in CARE-GNN README)
3. J. Ni, "Amazon Review Data (2018)", nijianmo.github.io, 2018. https://nijianmo.github.io/amazon/index.html (known)
4. A. Mukherjee et al., "What Yelp Fake Review Filter Might Be Doing?", ICWSM, 2013. https://ojs.aaai.org/index.php/ICWSM/article/view/14389 (known)
5. ODDS, "YelpChi dataset", Stony Brook University. http://odds.cs.stonybrook.edu/yelpchi-dataset/ (repo-link)
6. Yelp Engineering, "Yelp Content As Embeddings", engineeringblog.yelp.com, 2023. https://engineeringblog.yelp.com/2023/04/yelp-content-as-embeddings.html (repo-link)
7. S. Rayana, L. Akoglu, "Collective Opinion Spam Detection: Bridging Review Networks and Metadata", KDD, 2015. https://www.andrew.cmu.edu/user/lakoglu/pubs/15-kdd-collectiveopinionspam.pdf (repo-link)
8. A. Li et al., "Spam Review Detection with Graph Convolutional Networks" (GAS, Xianyu), CIKM 2019 / arXiv. https://arxiv.org/abs/1908.10679 (repo-link)
9. safe-graph, "DGFraud-TF2: GAS implementation", GitHub. https://github.com/safe-graph/DGFraud-TF2/tree/main/algorithms/GAS (repo-link)
10. J. Wang et al., "FdGars: Fraudster Detection via Graph Convolutional Networks in Online App Review System", WWW Companion, 2019. https://dl.acm.org/citation.cfm?id=3316586 (repo-link)
11. "Fake Reviewer Group Detection in Online Review Systems", ICDM Workshop / arXiv, 2021. https://arxiv.org/pdf/2112.06403.pdf (repo-link)
12. A. Halevy et al., "Preserving Integrity in Online Social Networks", arXiv, 2020. https://arxiv.org/pdf/2009.10311.pdf (repo-link: ByteByteGo ch. 5 ref 4)
13. Pinterest Engineering, "Fighting Spam using Clustering and Automated Rule Creation", Medium, 2021. https://medium.com/pinterest-engineering/fighting-spam-using-clustering-and-automated-rule-creation-1c01d8c11a05 (repo-link)
14. alirezadir, "ML System Design" (ml-system-design.md), GitHub, 2025. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/ml-system-design.md (fetched)
15. A. Aminian, A. Xu, "Machine Learning System Design Interview", ch. 5 "Harmful Content Detection", ByteByteGo, 2023. https://bytebytego.com/courses/machine-learning-system-design-interview/harmful-content-detection (ch-ref)
16. ByteByteGo, "ml-bytebytego: references for Machine Learning System Design Interview", GitHub, 2023. https://github.com/ByteByteGoHq/ml-bytebytego (fetched)
17. Educative, "Grokking the Machine Learning Interview", educative.io, 2024. https://www.educative.io/courses/grokking-the-machine-learning-interview (ch-ref)
18. Educative / K. Pham, "Machine Learning System Design", educative.io, 2021. https://educative.io/courses/machine-learning-system-design/introduction (ch-ref)
19. K. Pham, "machine-learning-interview", GitHub. https://github.com/khangich/machine-learning-interview (ch-ref)
20. C. Huyen, "Machine Learning Systems Design - Exercises", GitHub. https://github.com/chiphuyen/machine-learning-systems-design/blob/master/content/exercises.md (ch-ref)
21. Exponent, "Introduction to the ML System Design Interview", tryexponent.com, 2025. https://www.tryexponent.com/courses/ml-system-design/mlsd-intro (ch-ref)
22. Hello Interview, "ML System Design in a Hurry: Delivery Framework", hellointerview.com, 2025. https://www.hellointerview.com/learn/ml-system-design/in-a-hurry/delivery (ch-ref)
23. IGotAnOffer, "Meta Machine Learning Engineer Interview", igotanoffer.com, 2025. https://igotanoffer.com/blogs/tech/facebook-machine-learning-engineer-interview (ch-ref)
24. P. Halina, "ML Systems Design Interview Guide", patrickhalina.com, 2021. http://patrickhalina.com/posts/ml-systems-design-interview-guide/ (ch-ref)
25. safe-graph, "graph-fraud-detection-papers", GitHub. https://github.com/safe-graph/graph-fraud-detection-papers (fetched)
26. safe-graph, "DGFraud: A Deep Graph-based Toolbox for Fraud Detection", GitHub. https://github.com/safe-graph/DGFraud (fetched)
27. Engineer1999, "A Curated List of ML System Design Case Studies", GitHub, 2024. https://github.com/Engineer1999/A-Curated-List-of-ML-System-Design-Case-Studies (fetched)
28. DGL, "Yelp & Amazon Fraud Graph Dataset" (FraudYelp / FraudAmazon), fraud.py, GitHub. https://github.com/dmlc/dgl/blob/master/python/dgl/data/fraud.py (repo-link)
29. Facebook, "TIES: Temporal Interaction Embeddings for Enhancing Social Media Integrity at Facebook", arXiv, 2020. https://arxiv.org/pdf/2002.07917.pdf (repo-link: ByteByteGo ch. 5 ref 21)
30. Whatnot Engineering, "How Whatnot Utilizes Generative AI to Enhance Trust and Safety", Medium, 2023. https://medium.com/whatnot-engineering/how-whatnot-utilizes-generative-ai-to-enhance-trust-and-safety-c7968eb6315e (repo-link)
