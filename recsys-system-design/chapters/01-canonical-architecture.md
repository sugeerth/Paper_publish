# 01. The Canonical Multi-Stage Recommender Architecture

## Overview

Almost every large recommender system that has been publicly described, including YouTube, Instagram, Facebook, X/Twitter, Pinterest, LinkedIn, Taobao, Kuaishou and Netflix, has the same basic shape. It is a funnel of progressively more expensive models, each applied to a smaller candidate set, sitting on top of a split between offline, nearline and online computation. The funnel exists because of one constraint. You cannot run your best model over a corpus of millions to billions of items for every request within a latency budget of tens to a few hundred milliseconds. The layering exists because of a second constraint. Some computation, such as training on months of logs, cannot happen at request time, and some, such as reacting to the click a user made two seconds ago, cannot wait for a nightly batch job.

This chapter documents the canonical architecture as practitioners have drawn it, from Netflix's 2013 "online / nearline / offline" blueprint [1] and YouTube's 2016 two-stage deep network [3] through Instagram's four-stage Explore funnel [7][8][9], X's open-sourced For You pipeline [11][12], Alibaba's pre-ranking work [19][20] and Kuaishou's multi-stage cascade [22][23]. It ends with the 2024-2026 push to collapse the funnel into one generative model (Kuaishou OneRec [25][26], LinkedIn's LLM retrieval and generative ranker [18], Meta's GEM [28], Netflix's foundation model [29]).

The canonical stages, using the vocabulary that most companies converge on:

1. Retrieval / candidate generation / matching / recall: many heterogeneous sources reduce a corpus of 10^6 to 10^10 items to roughly 10^3 to 10^4 candidates.
2. Pre-ranking / first-stage ranking / light ranking / early-stage ranking (ESR) / coarse ranking: a cheap learned model, often a two-tower network or a distilled one, cuts thousands down to hundreds.
3. Ranking / second-stage ranking / heavy ranking / late-stage ranking (LSR) / fine ranking: a large multi-task neural network with rich user-item cross features predicts several engagement probabilities for each surviving candidate.
4. Re-ranking / blending / ordering / final pass: business rules, integrity filters, diversity, freshness, ad insertion, de-duplication and listwise optimization produce the final page.

## 1. Why a funnel at all

### 1.1 The cost argument

Google's public recommendation-systems course gives the canonical three-box version: candidate generation, scoring and re-ranking [4]. Its description is the clearest one-line rationale available. The candidate generator "reduces billions of videos down to hundreds or thousands" for YouTube, so it must be fast. The scoring model then ranks a smaller pool and picks the set to show, "on the order of 10 items," and can afford to be more precise [4]. The course also makes a point that is often forgotten in system-design interviews. When a system has multiple candidate generators, their scores are not comparable, so a separate scoring model is required rather than simply merging retrieval scores [4].

Eugene Yan's widely cited essay "System Design for Recommendations and Search" draws the same structure as a 2x2 grid. One axis is offline versus online, and the other is candidate retrieval versus ranking [5]. Offline, you train models, compute item embeddings and build approximate nearest neighbour (ANN) indices. Online, you embed the user or query, retrieve candidates from the index, then rank the few hundred survivors with a heavier model that can use richer features [5][32].

```
                 RETRIEVAL (recall, fast, coarse)    RANKING (precision, slow, rich)
              +---------------------------------+-----------------------------------+
   OFFLINE    | train embedding / two-tower     | train ranking model on logged     |
   (batch)    | compute item vectors            | impressions + labels              |
              | build ANN / i2i / graph indices | build feature tables / store      |
              +---------------------------------+-----------------------------------+
   ONLINE     | embed user/query, ANN lookup,   | hydrate features for ~10^2-10^3   |
   (request)  | graph walks, rules, merge       | candidates, score, re-rank, serve |
              +---------------------------------+-----------------------------------+
                        (after Yan, 2021 [5])
```

NVIDIA's Merlin team, citing Even Oldridge's article "Recommender Systems, Not Just Recommender Models," formalises a four-stage pattern: retrieval, filtering, scoring and ordering [6]. Retrieval is "fast but coarse-grained," with efficiency taking priority over precision. Filtering removes items the model would struggle to exclude, such as already-consumed or out-of-stock items. Scoring estimates interest. Ordering aligns the output with business needs and constraints [6]. The explicit filtering stage reflects a practical lesson. Hard constraints belong in deterministic code, not in a model you hope learned them.

### 1.2 The recall-precision hand-off

Every stage hands off to the next under a different objective. Early stages are judged on recall, meaning whether the eventual winner survived. Late stages are judged on precision and calibrated prediction. Pinterest's Related Pins paper describes this shift explicitly: once machine-learned ranking was introduced, candidate generation moved from optimising precision to optimising recall (summarised in [14]). Uber Eats likewise states that its first-pass ranker must achieve low latency "while maintaining a high recall metric" [30].

The hand-off creates the main pathology of cascades, which is stage inconsistency. Each stage is trained on a different sample distribution. The ranker is usually trained only on exposed items, but it scores items selected by a pre-ranker, which in turn was trained on whatever the ranker let through. Kuaishou's RecFlow dataset exists precisely to expose this. It logs not only exposures but also unexposed items filtered at each stage: 1.9 billion stage samples from 9.3 million online requests over 37 days, spanning 6 stages [22].

## 2. Netflix: the online / nearline / offline blueprint (2013)

### 2.1 Problem

In March 2013, Xavier Amatriain and Justin Basilico published "System Architectures for Personalization and Recommendation" on the Netflix Tech Blog [1]. This was the first widely read reference architecture that separated where computation happens from what is computed. Netflix had to combine large batch-trained models (matrix factorisation and later ensembles, see also [34]) with responsiveness to what a member just played, under strict page-render latency.

### 2.2 Architecture

The post defines three computation modes [1]:

- Offline computation has few limits on data volume and algorithmic complexity because it runs in batch. Its results go stale between updates and it cannot react to the current context. Netflix ran model training and batch pre-computation of results here, using Hadoop and query tools such as Hive and Pig.
- Online computation responds to the current request and context but must meet a strict SLA. The post discusses fallbacks for when the online computation cannot complete in time, such as reverting to a pre-computed result.
- Nearline computation is "an intermediate compromise between these two modes in which we can perform online-like computations, but do not require them to be served in real-time" [1]. It is event-driven. A user action triggers computation, and the result is stored for later serving.

Event and data distribution was handled by an internal framework called Manhattan, which the post calls "a distributed computation system that is central to our algorithmic architecture for recommendation" and compares to Twitter's Storm. Logging flowed through Chukwa into Hadoop, and Hermes provided publish-subscribe delivery [1]. Results were persisted in different stores depending on access pattern. My recollection of the post is that it names MySQL, Cassandra and EVCache for this purpose; treat the exact mapping as approximate.

```
                         +----------------------- OFFLINE -------------------------+
   member events  --->   | Chukwa -> Hadoop/Hive/Pig -> model training               |
   (plays, ratings,      |   -> batch-computed recs / model params                  |
    browsing)            +---------------------------|-------------------------------+
        |                                            v
        |                +---------------------- NEARLINE ------------------------+
        +-- Manhattan -> | event-triggered compute: e.g. solve user factors,      |
           (event flow)  | dot products, refresh cached scores                    |
                         +---------------------------|-------------------------------+
                                                     v
                              result stores (relational / Cassandra / EVCache)
                                                     |
                         +---------------------- ONLINE --------------------------+
   page request  ----->  | context filtering, light model eval, merge, fallbacks  | -> UI
                         +--------------------------------------------------------+
                                    (after Amatriain & Basilico, 2013 [1])
```

Netflix slide decks from the same era give a concrete example of the nearline layer. On a play event, a matrix-factorisation model solves the user's factors, computes user-video dot products and stores the scores in a cache, so that the online layer only applies presentation-context filtering and serves the result (as summarised in search coverage of the decks around [1]).

### 2.3 Key ideas and lessons

- Learning, feature computation and model evaluation can each be placed in any layer. The architecture is a set of placement choices, not a fixed pipeline [1].
- Distinguish events (small, time-sensitive units such as a play) from data (larger, less latency-sensitive aggregates). They take different paths [1].
- Always keep a pre-computed fallback for when online computation misses its deadline.

In 2023 Amatriain revisited the post in "Blueprints for recommender system architectures: 10th anniversary edition" [2]. It summarises four existing blueprints, including the Netflix one, and proposes a generalised blueprint built around a scoring-and-ranking model trained on data from a feedback loop [2]. The online/nearline/offline split has survived a decade essentially unchanged. What changed is that "nearline" grew into streaming feature platforms and near-real-time training (Section 9).

## 3. YouTube: the two-stage deep network (2016)

### 3.1 Problem

Covington, Adams and Sargin's RecSys 2016 paper "Deep Neural Networks for YouTube Recommendations" describes what it calls "one of the largest scale and most sophisticated industrial recommendation systems in existence" [3]. The challenges it names are scale, freshness (new uploads every second) and noisy implicit feedback.

### 3.2 Architecture

The paper follows the "classic two-stage information retrieval dichotomy" [3]:

```
  video corpus           candidate generation          ranking             top-N
  (millions)   ------->  (deep net over user history ->  (deep net,       -> (dozens)
                          extreme multiclass softmax;     rich features,
  other candidate ---->   ANN search at serving)          watch-time
  sources                 => hundreds                     objective)
                                     (after Covington et al., 2016 [3][15])
```

- Candidate generation treats recommendation as extreme multiclass classification. Given the user's watch and search history (averaged embeddings), demographics and context, the network predicts which video will be watched next, trained with a sampled softmax. At serving time the softmax output layer is replaced by nearest-neighbour search over video embeddings with the user vector as the query, so selecting the top N classes reduces to ANN lookup [3][15]. It outputs a small subset, in the hundreds, from the corpus [3][15].
- Ranking uses a separate deep network with many more features, including impression-level and user-video interaction features. Its objective is a weighted logistic regression that predicts expected watch time rather than click probability, which counteracts clickbait [15].
- Other candidate sources, described as "earlier work," are blended in before ranking. Ranking's job is partly to make scores from heterogeneous sources comparable [3].

Reported scale: approximately one billion parameters, trained on hundreds of billions of examples (approximate, from a secondary summary of the paper [15]). The paper also introduced the "example age" feature to model freshness. To my recollection it frames the funnel as millions to hundreds to dozens under a serving budget of tens of milliseconds; these figures could not be re-verified for this chapter and should be treated as approximate.

### 3.3 Lessons

- Separating retrieval from ranking lets each be optimised for its own metric and cost profile.
- Training retrieval as classification plus ANN serving became the template for every later two-tower system.
- Ranking on watch time rather than clicks is an early example of moving the objective closer to long-term value.

## 4. Meta: Instagram Explore and Facebook Feed

### 4.1 Instagram Explore, 2019: the three-pass distillation funnel

Instagram's 2019 post "Powered by AI: Instagram's Explore recommender system" describes a three-part ranking funnel that "extracts 65 billion features and makes 90 million model predictions every second" [7]. Candidate sourcing relies on ig2vec, a word2vec-like framework that treats the sequence of account IDs a user interacts with as words in a sentence to learn account embeddings [7]. Given seed accounts, nearest-neighbour lookup yields candidate accounts and from those candidate media.

The ranking funnel then progressively distils:

```
  ig2vec account KNN + other sources
            |
            v
   ~500 candidates
            |   pass 1: distillation model (minimal features) mimics later stages
            v
   ~150 candidates
            |   pass 2: lightweight neural network
            v
   ~50 candidates
            |   pass 3: full deep neural network (all dense features)
            v
   ~25 shown first (after integrity/diversity rules)
                                 (after Instagram, 2019 [7]; counts approximate)
```

The 500, 150, 50 and 25 figures come from the post and its coverage [7]. The key trick is that the first-pass model is trained to imitate the combined output of the heavier stages using a fraction of the features. Distillation is how the funnel keeps its stages aligned.

### 4.2 Instagram Explore, 2023: four stages, caching everywhere

Meta's August 2023 engineering post "Scaling the Instagram Explore recommendations system" redraws the funnel as four stages: retrieval, first-stage ranking, second-stage ranking and final re-ranking. It explicitly says the "leitmotif" is "clever use of caching and pre-computation in different ranking stages," which allows heavier models at every stage and less reliance on heuristics [8].

- Retrieval combines heuristic sources (for example trending or followed-author content) with ML sources, chiefly two-tower networks. Because the user and item towers are independent after training, item embeddings are precomputed offline, typically daily. At request time a fresh user embedding is computed and an ANN search returns similar items [8].
- First-stage ranking reuses the two-tower architecture "because of its cacheability property." It is trained to predict the output of the second-stage ranker, a form of knowledge distillation [8].
- Second-stage ranking is a heavier multi-task multi-label (MTML) network with user-item interaction features. It predicts probabilities of several events (click, like, save, "see less" and so on). These are combined by a value model, a weighted sum such as w_click * P(click) + w_like * P(like) - w_seeless * P(see less) + ... [8].
- Final re-ranking applies integrity, diversity and business rules [8].

Meta's public transparency page for Explore gives concrete funnel sizes. Retrieval fetches up to 1,500 media. Early-stage ranking, a two-tower network, passes the top 100 onward. Late-stage ranking produces the final list [9]. The page cautions that "these models and their input signals ... change frequently" [9].

```
   +-------------+   up to   +-----------------+   top    +-----------------+   +--------+
   | Retrieval   |  ~1,500   | Early-stage     |  ~100    | Late-stage      |-->| Final  |
   | heuristics, |---------->| ranking (two-   |--------->| ranking (MTML,  |   | rerank |
   | two-tower   |           | tower, distilled|          | value model)    |   | rules, |
   | ANN, ...    |           | from LSR)       |          |                 |   | divers.|
   +-------------+           +-----------------+          +-----------------+   +--------+
     item tower: daily offline embeddings; user tower: online, fresh features
                                     (after Meta 2023 [8] and transparency page [9])
```

### 4.3 Facebook News Feed (2021)

Meta's January 2021 post "How machine learning powers Facebook's News Feed ranking algorithm" describes feed ranking as multiple passes over an inventory of more than 1,000 candidate posts per user, evaluated in parallel on machines called predictors [10]. Integrity checks run first. A lightweight model then narrows the pool to roughly 500 of the most relevant posts, and heavier neural networks in later passes produce the main per-story score that orders the feed [10]. The final ordering also enforces contextual rules such as content-type diversity. The structure is the same canonical shape: eligibility and integrity filtering, a light pass, a heavy pass and contextual re-ranking.

### 4.4 Meta ads: Andromeda and GEM (2024-2025)

Ads retrieval is the extreme case of the funnel. Secondary coverage of Meta's December 2024 Andromeda post reports that retrieval narrows tens of millions of eligible ads to a few thousand candidates per impression before ranking and auction. Andromeda replaced a stack of smaller models and heuristics with a deep network co-designed with NVIDIA Grace Hopper hardware and Meta's MTIA, with a reported 10,000x increase in model capacity, +6% retrieval recall and +8% ad-quality gains on selected segments [27]. All of these figures are secondhand and should be treated as approximate.

In November 2025 Meta described GEM, the "Generative Ads Recommendation Model," as "the largest foundation model for recommendation systems (RecSys) in the industry, trained at the scale of large language models" [28]. Architecturally, GEM is not served in the request path for every candidate. It acts as a central teacher whose learnings are propagated "across the entire ads model fleet" through post-training techniques (distillation and representation transfer) into the vertical models that actually run in each funnel stage. Meta credits it with a 5% increase in ad conversions on Instagram and 3% on Facebook Feed in Q2 2025 [28]. This is a new layer in the canonical diagram: an offline foundation-model tier that feeds every online stage.

## 5. X / Twitter: the open-sourced For You pipeline

### 5.1 2023 architecture

On 31 March 2023 Twitter published "Twitter's Recommendation Algorithm" together with the source repository twitter/the-algorithm [11][12]. The blog frames the problem as distilling the roughly 500 million tweets posted daily down to a handful for each user. It describes three stages: candidate sourcing, ranking with a machine learning model, and heuristics and filters [11]. To my recollection the post states that the pipeline runs about 5 billion times per day and completes in under 1.5 seconds on average; this could not be re-verified for this chapter and should be treated as approximate.

Components [11][12]:

- Home Mixer, the service that assembles For You, is built on Product Mixer, a Scala framework of pipelines. Product pipelines select mixer or recommendation pipelines, which in turn run multiple candidate pipelines.
- Candidate sources together produce roughly 1,500 candidates (approximate; reported in coverage of the release). They split roughly 50/50 between in-network and out-of-network, though this varies per user [11].
  - In-network: the Earlybird search index with a light ranker (logistic regression), with author selection heavily driven by RealGraph, a model of the probability that one user engages with another.
  - Out-of-network: social-graph traversal through UTEG (user-tweet-entity-graph, an in-memory interaction graph on GraphJet) and embedding spaces (SimClusters community embeddings and TwHIN knowledge-graph embeddings). Embedding approaches became the larger out-of-network source.
- Heavy Ranker: a multi-task neural network of approximately 48M parameters (reported by secondary sources) that predicts several engagement probabilities. These are combined with weights into a single score. Candidates are scored in batch without cross-candidate attention.
- Heuristics and filters: visibility filtering (blocked or muted accounts, NSFW settings), author diversity, content balance, feedback fatigue, de-duplication and social proof, then mixing with ads and Who-to-Follow modules.

```
  request -> Home Mixer (Product Mixer framework)
               |
               +-- in-network:  Earlybird index + light ranker (RealGraph) --+
               +-- out-of-net:  UTEG/GraphJet, SimClusters ANN, TwHIN ANN ---+--> ~1,500
               |                                                             |
               +--> feature hydration --> Heavy Ranker (~48M param MTL NN) <-+
               |
               +--> visibility filtering, author diversity, dedup, fatigue
               +--> mix ads / WTF modules --> For You timeline
                                     (after Twitter 2023 [11][12])
```

### 5.2 2026: the Grok-era x-algorithm

In January 2026 xAI released a new x-algorithm repository and updated it substantially on 15 May 2026 [13]. Secondary coverage describes the composition as Thunder for in-network candidates, Phoenix retrieval for out-of-network, Home Mixer for orchestration, and a generic candidate pipeline built from reusable Rust traits named Source, Hydrator, Filter, Scorer, Selector and SideEffect [13]. A single phoenix/run_pipeline.py now runs retrieval and ranking end to end from exported checkpoints, and a roughly 3 GB "mini Phoenix" model is downloadable. Production weights and training code remain closed [13]. The trait list is a useful canonical vocabulary in its own right: every stage of every funnel in this chapter is some composition of source, hydrate, filter, score and select.

## 6. Pinterest: candidate generators, a light-weight scorer and a blender

Pinterest's WWW 2022 slides draw the home feed as a funnel [14]:

- Candidate generators: token-based indexes, HNSW embedding indexes, graph-based random walks (Pixie) and explore/exploit sources.
- A Light-Weight Scorer (LWS) acting as pre-ranking.
- A Ranking Service containing a Utility Prediction Scorer (the multi-task ranker) and a Multi-Objective Blender.

Pixie, a real-time random-walk system over the pin-board bipartite graph, was reported to serve 3+ billion items to 200+ million users in real time. Systems backed by Pixie contributed more than 80% of user engagement at the time [15b]. An older Pinterest engineering post describes the same split in services. Polaris retrieves, filters and applies lightweight scoring, then passes a list to the Feed Generator, which sends it with real-time user signals to Scorpion for "a second pass of full scoring" [16]. Historically, each content type was ranked by its own specialised model and then "blended with a ratio-based round robin" to create the final home feed [33]. Later diversification work at retrieval and ranking (round-robin, DPP, bucketised ANN) is covered in a later chapter.

```
  Pixie walks | HNSW embeds | token idx | explore  --> merge+dedup --> LWS (light)
                                                                       |
                                     Utility Prediction Scorer (MTL) <-+
                                                 |
                                Multi-Objective Blender --> home feed
                                    (after Pinterest WWW 2022 [14], [16])
```

## 7. LinkedIn: first-pass rankers to a unified LLM-era stack

LinkedIn's 2019 feed architecture is a federated variant of the funnel. Multiple first-pass rankers (FPRs) preselect candidates from their own inventories, such as network updates, jobs and recommended content, and a second-pass ranker (SPR) merges their outputs into one list. FollowFeed, which serves updates from a member's network, is the dominant FPR: more than 80% of feed updates came from it [17]. The federated design matches the organisation (each inventory team owns its FPR) but makes global calibration the SPR's problem.

By 2025-2026 LinkedIn had published work on replacing multi-source retrieval with a single LLM-based dual encoder fine-tuned on engagement data [18]. Secondary coverage of a March 2026 LinkedIn post describes a unified LLM retrieval system plus a transformer-based sequential "Generative Recommender" ranker. These details are from secondary sources and are approximate. The direction is the same as at Meta and Kuaishou: fewer, larger, shared models with fewer hand-maintained sources.

## 8. Alibaba, Taobao and Kuaishou: the explicit pre-ranking stage

### 8.1 Why Chinese e-commerce and short-video stacks name "pre-ranking"

Chinese industrial papers almost always draw the cascade as matching, pre-ranking, ranking and re-ranking. Pre-ranking has become a research area in its own right because it handles orders of magnitude more items than the ranker under similar latency. A Taobao Search paper ("Rethinking the Role of Pre-ranking in Large-scale E-Commerce Searching System") describes billions of items in the corpus, with pre-ranking taking in hundreds of thousands of candidates and passing roughly the top thousands to ranking. Its input set is "tens or even hundreds of times larger" than the ranking stage's [20]. The paper argues that pre-ranking should be judged by whether it preserves the items the whole system would ultimately select, not by its standalone AUC [20].

### 8.2 COLD (Alibaba, 2020)

COLD ("Computing power cost-aware Online and Lightweight Deep pre-ranking system") rejected the then-standard two-tower pre-ranker, which cannot use user-item cross features. It replaced it with a small fully connected network with cross features, co-designed with the serving system [19]. The model and its computing cost are optimised together, so the trade-off between quality and compute is controllable. The design adds feature selection under a compute budget, inference acceleration and online learning to handle distribution shift. COLD was deployed from 2019 in almost all products with a pre-ranking module in Alibaba display advertising [19]. Its lesson generalises: the architecture of a stage should be chosen by its share of the latency and compute budget, not by tradition.

### 8.3 Kuaishou: a six-stage cascade, quantified

Kuaishou's public papers give the most complete stage-level numbers. RecFlow logs six stages of the production short-video funnel [22]. A 2026 paper on cross-stage fusion describes candidate sizes in that cascade shrinking from about 8,000 down to 6, with roughly 500 candidates selected at pre-ranking and about 10 items finally exposed [23]. KuaiFormer describes retrieval feeding a pre-rank stage, then a cascading rank stage, then a fine-grained full rank stage [24].

```
   corpus (10^8+)
      | retrieval: many channels (i2i, u2i, transformer retrieval e.g. KuaiFormer)
      v
   ~8,000 merged candidates
      | pre-ranking / coarse ranking
      v
   ~500
      | cascade ranking -> full (fine) ranking (multi-task)
      v
   ~tens
      | re-ranking (listwise, diversity, business)
      v
   ~6-10 exposed
                              (after RecFlow [22], UniRec [23]; approximate)
```

### 8.4 Collapsing the cascade: OneRec

Kuaishou's OneRec (February 2025) and its June 2025 technical report argue that the cascade itself is the problem. Each stage is optimised for a different objective, compute is fragmented, and earlier stages cap the quality of later ones [25][26]. OneRec replaces retrieval and ranking with a single encoder-decoder generative model (later decoder-only variants). It uses sparse Mixture-of-Experts for capacity and session-wise list generation, and it is aligned to the final objective with preference optimisation [25][26]. The reported motivation includes better engagement and lower operating cost than the legacy multi-stage pipeline [26]. In practice, generative systems so far keep a thin rule and filter layer and a re-ranking stage. What collapses is the learned middle of the funnel.

## 9. The time axis: offline, nearline, online in modern stacks

The Netflix three-layer split [1] maps onto modern components as follows:

| Layer | Netflix 2013 [1] | Modern equivalent | Examples in this chapter |
|---|---|---|---|
| Offline | Hadoop, Hive, Pig; model training; batch recs | Data lake + distributed training; daily item-embedding jobs; ANN index builds; foundation models | Item tower daily refresh [8]; GEM teacher [28]; Netflix FM [29] |
| Nearline | Manhattan event processing; cached scores | Stream processing, real-time feature store, streaming/online training, incremental index updates | Monolith online training [31]; COLD online learning [19] |
| Online | Request-time filtering, light models, fallbacks | Retrieval fan-out, feature hydration, pre-rank, rank, re-rank under a p99 SLA | Home Mixer [12]; Explore [8]; Feed [10] |

ByteDance's Monolith (2022) is the clearest published nearline-training design. It argues that standard frameworks keep batch training and serving "completely separated," which prevents models from reacting to feedback in real time. Monolith uses a collisionless hash embedding table with expirable embeddings and frequency filtering, plus an online training architecture that synchronises updated parameters from training to serving in near real time [31]. Netflix's 2025 foundation model shows the opposite trend at the offline end. A single transformer, trained autoregressively on hundreds of billions of interactions from over 300 million members, centralises member preference learning, and downstream models consume it through fine-tuning, embeddings or direct prediction [29]. Both Meta [28] and Netflix [29] thus add an explicit foundation-model tier beneath the classic offline layer.

## 10. Request flow, end to end

Putting the pieces together, a generic request in a modern feed looks like this (synthesised from [4][5][8][10][11][12][14]):

```
  client --> API/gateway --> mixer/orchestrator (per-surface pipeline)
     1. fetch user context: profile, recent actions (real-time feature store)
     2. fan out to N candidate sources in parallel, each with its own timeout
          two-tower ANN | i2i / graph walk | follow graph | trending | rules
     3. merge, dedup, eligibility & integrity filter          (~10^3-10^4)
     4. pre-rank: cached / distilled / two-tower or small DNN (-> ~10^2-10^3)
     5. hydrate rich features (batch RPC to feature store, embeddings)
     6. rank: multi-task NN -> P(event_i); value model sum w_i * P_i (-> ~10^1-10^2)
     7. re-rank: diversity, freshness, fatigue, policy, ads/module insertion
     8. log impressions + features served ("log-and-wait") for training
     9. respond; async side effects (cache next page, update seen-sets)
```

Three implementation details recur across sources:

- Per-source timeouts and graceful degradation. Candidate sources run in parallel, and slow ones are dropped rather than allowed to block the page. This echoes Netflix's insistence on fallbacks [1].
- Hydration is often the real latency hotspot. Uber Eats reported that retrieval was fetching tens of thousands of candidates that were then fully hydrated before ranking discarded most of them. Removing low-yield retrieval paths saved about 120 ms, and splitting hydration into ranking and presentation phases saved over 100 ms more [30].
- Logging the served features at request time (rather than recomputing them later) avoids training-serving skew. This is a theme in Yan's survey of discovery systems [5].

## 11. How each company draws it: comparison

| Company / system | Stages as named | Funnel sizes (public, approx.) | Retrieval | Light / pre-rank | Heavy rank | Final pass |
|---|---|---|---|---|---|---|
| Netflix 2013 [1] | offline / nearline / online | n/a | batch MF, precomputed | nearline cached scores | online light eval | context filter, fallbacks |
| YouTube 2016 [3] | candidate gen, ranking | millions -> hundreds -> dozens | softmax-trained DNN + ANN | none (2 stages) | DNN, weighted LR on watch time | not detailed |
| Google course [4] | candidate gen, scoring, re-ranking | billions -> 100s-1000s -> ~10 | multiple generators | none | scoring model | freshness, diversity, fairness |
| Instagram 2019 [7] | sourcing + 3-pass funnel | 500 -> 150 -> 50 -> 25 | ig2vec account KNN | distilled model, light NN | full DNN | integrity, diversity |
| Instagram 2023 [8][9] | retrieval, ESR, LSR, rerank | up to 1,500 -> 100 -> final | heuristics + two-tower ANN | cached two-tower, distilled | MTML + value model | rules, diversity |
| Facebook Feed 2021 [10] | multi-pass ranking | 1,000+ -> ~500 -> ordered | inventory of eligible posts | lightweight model | neural nets on predictors | contextual rules |
| Twitter 2023 [11][12] | sourcing, rank, heuristics | ~500M/day -> ~1,500 -> feed | Earlybird, UTEG, SimClusters, TwHIN | Earlybird light ranker | ~48M-param MTL NN | visibility, diversity, mixing |
| Pinterest [14][16] | CG, LWS, ranking, blender | not public | Pixie, HNSW, token idx | Light-Weight Scorer | utility prediction scorer | multi-objective blender |
| LinkedIn 2019 [17] | FPRs -> SPR | FollowFeed >80% of updates | per-inventory FPRs | FPR scoring | second-pass ranker | not detailed |
| Taobao search [20] | matching, pre-rank, rank | billions -> 10^5 -> 10^3 | multi-channel | pre-ranking | ranking | not detailed |
| Alibaba ads [19] | matching, pre-rank (COLD), rank | not public | multi-channel | COLD cross-feature DNN | ranking DNN | not detailed |
| Kuaishou [22][23][24] | 6 stages | ~8,000 -> ~500 -> ... -> 6-10 | KuaiFormer and others | pre-rank / cascade rank | full rank | re-rank |
| Kuaishou OneRec [25][26] | single generative model | session list generated | merged into generator | merged | merged | thin rule layer |
| Meta Ads [27][28] | retrieval (Andromeda), rank, auction | 10^7 ads -> few thousand | Andromeda DNN on GPUs | (not public) | ranking models distilled from GEM | auction |

## 12. Scale numbers at a glance

| Metric | Value | Source |
|---|---|---|
| Instagram Explore features extracted | 65 billion per second | [7] |
| Instagram Explore model predictions | 90 million per second | [7] |
| Instagram Explore retrieval output | up to 1,500 media; top 100 after ESR | [9] |
| Facebook Feed candidates per user | 1,000+, pruned to ~500 | [10] |
| Twitter daily tweets considered | ~500 million | [11] |
| Twitter candidates per request | ~1,500 (approx.) | [11][12] |
| Twitter heavy ranker size | ~48M parameters (approx.) | [12] |
| YouTube 2016 model size | ~1B parameters (approx.) | [15] |
| Kuaishou RecFlow | 9.3M requests, 1.9B stage samples, 6 stages | [22] |
| Pixie | 3+B items, 200+M users, real time | [15b] |
| Uber Eats hydration fix | ~120 ms and 100+ ms latency savings | [30] |
| Meta Andromeda | ~10,000x capacity; +6% recall (secondhand) | [27] |
| Meta GEM | +5% IG / +3% FB Feed ad conversions (Q2 2025) | [28] |

## 13. Recurring design tricks

1. Distil downward. Train each cheaper stage to imitate the next stage, not just the label: Instagram's first-pass distillation [7] and its two-tower first-stage ranker trained on second-stage outputs [8]. Meta GEM pushes the same idea fleet-wide [28].
2. Cache whatever is user-independent. Item towers are computed offline, and only the user tower and its dot product run online [8].
3. Make early stages expressive only when the budget allows. COLD shows that cross features in pre-ranking pay off when co-designed with serving [19].
4. Federate sources, centralise scoring. Many independent candidate sources feed one scorer, because raw retrieval scores are not comparable [4][11][17].
5. Separate hard rules from learned preferences. Filtering and ordering are explicit stages [6], and integrity checks run before expensive models [10].
6. Optimise a value model, not a single label. Weighted multi-task outputs produce one score [8][11], and objectives move towards long-term value (watch time [3]).
7. Measure the funnel, not the stage. Log what each stage dropped [22] and evaluate pre-ranking against end-to-end selection [20].
8. Push freshness into nearline. Event-triggered recomputation [1] and online training [31] are cheaper than making the online path heavier.
9. Collapse stages when one model can afford it. Generative recommenders [25][26] and unified LLM retrieval [18] remove learned stages, but keep rules and filtering.

## Key takeaways for system designers

- Start from the latency budget and the corpus size, then derive the funnel. Each stage should cut candidates by one to two orders of magnitude while its per-item cost rises by a similar factor. Public funnels cluster around 10^3-10^4 after retrieval, 10^2-10^3 after pre-ranking, and 10^1-10^2 after ranking [8][9][10][22][23].
- Treat online, nearline and offline as placement decisions for every component (features, models, results), as Netflix did in 2013 [1]. Most production wins come from moving work out of the online path through caching, pre-computation and nearline updates [8].
- Judge each stage by what it does for the final output. Recall at the next stage's cut-off matters more than standalone AUC [20]. Log stage-wise drops so that you can debug selection bias [22].
- Keep candidate sources heterogeneous but cheap. Graph walks, embeddings, social graph and trending each capture different intent, and the ranker reconciles them [4][11][14].
- Use distillation and shared towers to keep the stages consistent [7][8]. Inconsistent stages are the main source of wasted compute in cascades.
- Put integrity, eligibility and business logic in explicit filter and re-rank stages rather than hoping a model learns them [6][10][11].
- Feature hydration and fan-out are frequent latency hotspots. Prune low-yield sources and split hydration by need [30].
- Expect the 2025-2026 trend to continue. Foundation models trained offline [28][29] feed smaller serving models, and generative recommenders [25][26] and LLM-based retrievers [18] collapse the middle of the funnel. Filters, policy and listwise re-ranking survive in every published design.

## References

1. Xavier Amatriain and Justin Basilico (Netflix). "System Architectures for Personalization and Recommendation." Netflix Technology Blog, March 2013. https://netflixtechblog.com/system-architectures-for-personalization-and-recommendation-e081aa94b5d8
2. Xavier Amatriain. "Blueprints for recommender system architectures: 10th anniversary edition." Machine Learning Times (Predictive Analytics World), 2023. https://www.predictiveanalyticsworld.com/machinelearningtimes/blueprints-for-recommender-system-architectures-10th-anniversary-edition/12879/
3. Paul Covington, Jay Adams and Emre Sargin (Google). "Deep Neural Networks for YouTube Recommendations." RecSys 2016. https://research.google/pubs/pub45530 (PDF: https://cseweb.ucsd.edu/classes/fa17/cse291-b/reading/p191-covington.pdf)
4. Google for Developers. "Recommendation systems overview" (Machine Learning: Recommendation Systems course). https://developers.google.com/machine-learning/recommendation/overview/types
5. Eugene Yan. "System Design for Recommendations and Search." eugeneyan.com, 2021. https://eugeneyan.com/writing/system-design-for-discovery/
6. NVIDIA (Merlin team, citing Even Oldridge's "Recommender Systems, Not Just Recommender Models"). "Offline to Online: Feature Storage for Real-time Recommendation Systems with NVIDIA Merlin." NVIDIA Technical Blog, 2022-2023. https://developer.nvidia.com/blog/offline-to-online-feature-storage-for-real-time-recommendation-systems-with-nvidia-merlin
7. Ivan Medvedev, Haotian Wu and Taylor Gordon (Instagram/Facebook AI). "Powered by AI: Instagram's Explore recommender system." Instagram Engineering, 2019. https://instagram-engineering.com/powered-by-ai-instagrams-explore-recommender-system-7ca901d2a882
8. Vladislav Vorotilov and Ilnur Shugaepov (Meta). "Scaling the Instagram Explore recommendations system." Engineering at Meta, 9 August 2023. https://engineering.fb.com/2023/08/09/ml-applications/scaling-instagram-explore-recommendations-system/
9. Meta Transparency Center. "Instagram Explore AI system." Accessed 2026. https://transparency.meta.com/features/explaining-ranking/ig-explore/
10. Akos Lada, Meihong Wang and Tak Yan (Facebook). "How machine learning powers Facebook's News Feed ranking algorithm." Engineering at Meta, 26 January 2021. https://engineering.fb.com/2021/01/26/ml-applications/news-feed-ranking/
11. Twitter Engineering. "Twitter's Recommendation Algorithm." Twitter (X) Engineering Blog, 31 March 2023. https://blog.x.com/engineering/en_us/topics/open-source/2023/twitter-recommendation-algorithm
12. Twitter. "the-algorithm: Source code for the X Recommendation Algorithm" (including home-mixer README). GitHub, 2023. https://github.com/twitter/the-algorithm
13. Pasquale Pillitteri. "X Algorithm Open Source: May 15, 2026 Major Update Brings End-to-End Pipeline and Downloadable Phoenix Model." 2026 (secondary coverage of xAI's x-algorithm repository). https://pasqualepillitteri.it/en/news/2594/x-algorithm-open-source-may-2026
14. Pinterest. "Pinterest at WWW 2022" (sponsor slides on home feed recommendation funnel). 2022. https://archives.iw3c2.org/www2022/wp-content/uploads/Sponsors/WWW2022-Pinterest.pdf
15. Adrian Colyer. "Deep neural networks for YouTube recommendations" (paper summary). The Morning Paper, 19 September 2016. https://blog.acolyer.org/2016/09/19/deep-neural-networks-for-youtube-recommendations/
15b. Adrian Colyer. "Pixie: a system for recommending 3+ billion items to 200+ million users in real-time" (summary of Eksombatchai et al., WWW 2018). The Morning Paper, 2018. https://blog.acolyer.org/2018/05/23/pixie-a-system-for-recommending-3-billion-items-to-200-million-users-in-real-time/
16. Pinterest Engineering. "Building a dynamic and responsive Pinterest." Pinterest Engineering Blog (Medium). https://medium.com/@Pinterest_Engineering/building-a-dynamic-and-responsive-pinterest-7d410e99f0a9
17. LinkedIn Engineering. "Community-focused Feed optimization." LinkedIn Engineering Blog, June 2019. https://engineering.linkedin.com/blog/2019/06/community-focused-feed-optimization
18. LinkedIn. "Large Scale Retrieval for the LinkedIn Feed using Causal Language Models." arXiv 2510.14223, 2025. https://arxiv.org/html/2510.14223v1
19. Zhe Wang, Liqin Zhao, Biye Jiang et al. (Alibaba). "COLD: Towards the Next Generation of Pre-Ranking System." DLP-KDD 2020, arXiv 2007.16122. https://arxiv.org/abs/2007.16122
20. Alibaba (Taobao Search). "Rethinking the Role of Pre-ranking in Large-scale E-Commerce Searching System." arXiv 2305.13647, 2023. https://arxiv.org/abs/2305.13647
21. "A Comprehensive Survey on Retrieval Methods in Recommender Systems." arXiv 2407.21022, 2024. https://arxiv.org/abs/2407.21022
22. Kuaishou. "RecFlow: An Industrial Full Flow Recommendation Dataset." ICLR 2025, arXiv 2410.20868. https://arxiv.org/html/2410.20868v1
23. "UniRec: Cross-stage Multi-Task Fusion with Preference Alignment for Cascaded Recommender Systems." arXiv 2609.11052, 2026. https://arxiv.org/pdf/2609.11052
24. Kuaishou. "KuaiFormer: Transformer-Based Retrieval at Kuaishou." arXiv 2411.10057, 2024. https://arxiv.org/pdf/2411.10057
25. Jiaxin Deng et al. (Kuaishou). "OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment." arXiv 2502.18965, February 2025. https://arxiv.org/abs/2502.18965
26. Kuaishou OneRec Team. "OneRec Technical Report." arXiv 2506.13695, June 2025. https://arxiv.org/html/2506.13695v1
27. Jon Loomer. "Meta Andromeda: What It Means for Your Ad Strategy" (secondary summary of Meta's December 2024 Andromeda engineering post). 2025. https://www.jonloomer.com/meta-andromeda/
28. Meta. "Meta's Generative Ads Model (GEM): The central brain accelerating ads recommendation AI innovation." Engineering at Meta, 10 November 2025. https://engineering.fb.com/2025/11/10/ml-applications/metas-generative-ads-model-gem-the-central-brain-accelerating-ads-recommendation-ai-innovation/
29. ZenML LLMOps Database. "Foundation Model for Personalized Recommendation at Scale" (summary of Netflix Tech Blog, March 2025). https://www.zenml.io/llmops-database/foundation-model-for-personalized-recommendation-at-scale
30. Uber Engineering. "Uber Eats search pipeline" (retrieval, hydration and ranking latency work). Uber Blog, 2026. https://www.uber.com/us/en/blog/uber-eats-search-pipeline/
31. Zhuoran Liu et al. (ByteDance). "Monolith: Real Time Recommendation System With Collisionless Embedding Table." arXiv 2209.07663, 2022. https://arxiv.org/abs/2209.07663
32. Eugene Yan. "System Design for Discovery (RecSys and Search)." ApplyingML. https://applyingml.com/resources/discovery-system-design/
33. MLconf. Pinterest Homefeed talk abstract (ratio-based round-robin blending), 2016. https://mlconf.com/?p=8909
34. Carlos A. Gomez-Uribe and Neil Hunt. "The Netflix Recommender System: Algorithms, Business Value, and Innovation." ACM Transactions on Management Information Systems 6(4), 2015. https://dl.acm.org/doi/10.1145/2843948
