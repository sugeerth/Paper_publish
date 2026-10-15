# 23. LinkedIn Case Study: Feed, PYMK, Jobs, LiRank, LiGNN, 360Brew and the ML Platform Beneath Them

LinkedIn is one of the most thoroughly documented recommender organizations in industry. Over about fifteen years it has published production systems at every layer of the stack:

- statistical personalization models (GLMix in Photon ML, then GDMix);
- feature platforms (Feathr, with Venice as the online store);
- search and retrieval infrastructure (Galene, then GPU retrieval with LiNR);
- graph learning (LiGNN);
- large deep ranking models (LiRank);
- a 150B-parameter decoder-only foundation ranker (360Brew);
- LLM-based dual-encoder retrieval for the Feed (2025).

Its products span several recommendation problems with different shapes:

- **Feed:** a heterogeneous, multi-objective stream.
- **People You May Know (PYMK):** link prediction on a social graph.
- **Jobs:** a two-sided marketplace with sparse, high-value labels such as applications and recruiter replies.
- **Ads:** calibrated CTR prediction feeding an auction.
- **Recruiter / Talent Search:** search ranking with fairness constraints.

This chapter follows that history as a single case study. It starts with the statistical era (GLMix and "fixed plus random effects" personalization). It then covers the feature and serving substrate (Feathr, Venice, Galene), the 2019 federated Feed architecture and its multi-objective tuning, the 2024 deep-learning consolidation (LiRank, LiGNN, LiNR), and the 2025-2026 move to LLM-native ranking and retrieval (360Brew, LLM dual encoders). It closes with experimentation and fairness infrastructure (XLNT/T-REX, fairness-aware Recruiter ranking), a comparison table and lessons for designers.

> **Sourcing note.** For this chapter, the live web-search budget was exhausted and outbound access to arxiv.org, linkedin.com, engineering.linkedin.com, dl.acm.org and most blogs was blocked from the research environment. Primary sources read directly in this session: the 360Brew technical report PDF (full text, from the authors' GitHub repository) [1][2][3], and the GitHub repositories and READMEs of Photon ML [6], GDMix [8], Feathr (including its talk slides) [10][11], Venice [12], DeText [14], Liger-Kernel [24] and FastTreeSHAP [25]. Facts about LiRank, LiGNN, LiNR, the 2025 LLM retrieval paper, the Feed architecture, XLNT/T-REX and fairness-aware ranking come from abstracts and primary sources that other chapters of this Atlas fetched and verified earlier (Chapters 1-8, 12, 14-16). Those facts are cross-referenced, and they could not be re-fetched here. Numbers from those sources are marked approximate where the original wording could not be checked again. Readers should check exact figures against the cited papers.

---

## 23.1 The shape of LinkedIn's recommendation problems

LinkedIn's recommendation surfaces share one underlying asset, the "economic graph" of members, companies, jobs, skills, schools, posts and groups. Their objectives differ sharply, though:

| Surface | Item type | Key labels | Distinctive constraint |
|---|---|---|---|
| Feed | posts, articles, network updates, recommended content, ads | react, comment, share, dwell, click, "viral actions" | many inventory sources; creator-side value; multi-objective |
| PYMK | members | invitation sent, accepted | graph link prediction; reciprocity; cold-start members |
| Jobs (recs + search) | job postings | view, apply, dismiss, "qualified application", recruiter hearing back | two-sided marketplace; short-lived items; sparse labels |
| Ads | sponsored content, text ads | click, conversion | calibrated probabilities for auctions |
| Recruiter / Talent Search | member profiles | InMail sent, accepted | query-dependent ranking; fairness constraints |
| LinkedIn Learning | courses | enroll, watch | content cold start |

The 360Brew report says this directly. In the traditional design, each of these tasks "is conventionally addressed by dedicated models that have been developed and maintained over multiple years by teams of a similar or larger size" than the foundation-model team [1]. The report describes the resulting stack as "complex, multilayered architectures that ... often incorporate thousands of predictive models" [1]. Much of LinkedIn's published infrastructure attacks the cost of that model sprawl:

- Feathr: shared features.
- GDMix and Photon ML: shared personalization training.
- LiRank: shared ranking architecture.
- LiGNN: shared graph embeddings.
- 360Brew: one model for many tasks.

The canonical retrieval-then-ranking funnel, as drawn in the 360Brew report [1]:

```
  members (QPS = member count x interaction frequency)
        |
        v
  +-------------------+   large K (item pool >> member pool, e.g. posts),
  |  Retrieval layer  |   high recall, two-tower / GPU exhaustive / LLM dual-encoder
  +-------------------+
        |  hundreds to ~2,000 candidates
        v
  +-------------------+   small K, high precision; DCN-style rankers (LiRank),
  |   Ranking layer   |   or a text foundation ranker (360Brew)
  +-------------------+
        |
        v
  re-ranking / blending / fairness / ads slotting  -->  member-facing list
```

---

## 23.2 The statistical era: GLMix, Photon ML and GDMix

### 23.2.1 Problem

Before deep learning took over, LinkedIn needed per-member personalization for job recommendations, the feed and ads. It had to scale to hundreds of millions of members, each with relatively little data. A single global model underfits individual taste. A separate model per member overfits and cannot share signal.

### 23.2.2 GLMix and GAME (Photon ML)

**GLMix** (Generalized Linear Mixed models; Zhang et al., KDD 2016 [7]) solves this with the classical statistical idea of **fixed effects plus random effects**:

- a **global (fixed-effect) model** learns from all data;
- **per-entity random-effect models** learn per-member coefficients (and, for jobs, per-job coefficients) that capture individual deviations from the global trend;
- the score is the sum: global score + member-specific score + item-specific score, passed through a link function (logistic for clicks/applies).

**Photon ML** is LinkedIn's open-source Spark library for this family [6]:

- Supports GLMs (linear, logistic and Poisson regression) and **GAME (Generalized Additive Mixed Effects)** models, of which GLMix is the supported type.
- Training uses **coordinate descent**. The fixed effect and each random-effect "coordinate" are optimized in turn, while the others are held fixed and contribute as an **offset** (scores take the form x . B + offset).
- Can "scale model training up to hundreds of billions of coefficients" within Spark [6].
- Optimizers: L-BFGS and TRON. Regularization: L1, L2 and elastic net. Workflows: warm start, incremental training, partial retraining with locked coefficients, Avro I/O, and Sobol/Bayesian hyperparameter search (experimental) [6].
- Its example recommender is a fixed-effect model + per-user random effect + per-item random effect + a user-item matrix-factorization term [6].

The Photon ML README lists the LinkedIn products that adopted GLMix: **job search and recommendation, news feed ranking, ads CTR prediction and "People Also Viewed"** [6].

```
  GLMix / GAME (coordinate descent with offsets)
  ------------------------------------------------
  round r:
    1. fixed effect  f(x)       <- fit on ALL rows, offset = sum(random effects)
    2. per-member    g_m(x)     <- fit per member m on its rows, offset = f + h
    3. per-job       h_j(x)     <- fit per job j on its rows,  offset = f + g
  repeat until convergence
  serve: logit = f(x) + g_member(x) + h_job(x)
```

### 23.2.3 GDMix: deep fixed effects plus per-member random effects

**GDMix** (Generalized Deep Mixed model; LinkedIn, 2020 [8][9]) is the successor to Photon ML for deep learning. The design keeps the divide-and-conquer structure but allows the **fixed effect to be a deep model**:

- **Fixed effect:** logistic regression or a **DeText** deep NLP ranking model. Scores from any external model can be plugged in [8].
- **Random effects:** per-entity logistic regressions. There can be **hundreds of millions** of them, one per LinkedIn member [8].
- **Training pipeline** [8]:
  1. Distributed fixed-effect training. LR uses all-reduce gradients and SciPy L-BFGS. DeText uses parameter-server async or Horovod sync training.
  2. Score the training data with the fixed effect.
  3. A Spark job partitions records by entity ID.
  4. Workers train their per-entity LR models independently in parallel.
  5. Iterate (parallel blockwise coordinate descent).
- Runs on Kubernetes with Kubeflow and spark-on-k8s-operator, with shared storage [8].
- The README's worked example is job recommendation. Its MovieLens demo shows how much the random effects add. The LR global model reaches AUC 0.6237. Per-user random effects lift it to 0.7058 and per-movie random effects to 0.7599. A DeText global model gives 0.7090 global and 0.7680 overall [8].

**DeText** [14] is the deep text-ranking component used as a GDMix fixed effect:

- word embeddings, then a CNN/LSTM/BERT text encoder;
- query-document interaction features (cosine, Hadamard, concatenation);
- a wide-and-deep merge with hand-crafted features in an MLP;
- learning-to-rank heads including TF-Ranking losses.

Its papers (SIGIR 2019, KDD 2019, CIKM 2020, KDD 2020) cover deep NLP for LinkedIn search and recommendation [14].

### 23.2.4 Lessons from the mixed-model era

- **Per-member models are a structural answer to personalization.** Learned ID embeddings are the alternative. Random-effect coefficients are interpretable, cheap to retrain per entity, and parallelize naturally after a partition by member ID.
- **Offsets make modular stacking possible.** Any upstream score, including a deep model or an external model, can become the "fixed effect", and the personalization layer trains around it. The same idea of "train the residual" returns later in LiRank's calibration layers and 360Brew's in-context personalization.
- **Weakness:** random effects are linear and per-entity, so they need enough data per member. Cold-start members fall back to the global model. 360Brew later attacks this regime explicitly (Section 23.7).

---

## 23.3 Feature, storage and search substrate: Feathr, Venice, Galene

### 23.3.1 Feathr

**Problem.** Feature pipelines were rebuilt redundantly by every team. Teams had "no common type system, no common feature namespace", definitions were inconsistent, and features were inconsistent between training and serving (Chapter 8 quoting the Feathr community talk [11]). The Feathr talk slides list the problems as: features are not reused and are hard to track for compliance, definitions are inconsistent across teams, getting features into production is hard, offline/online inconsistency degrades models, and feature quality drifts and needs human review [11].

**Design** [10][11]:

- **Declarative feature definitions** [5] ("anchors" over batch or streaming sources, plus derived features) registered under shared names.
- **Point-in-time correct joins.** For each observation (impression) row, Feathr takes the feature value "closest from the past" for the same key [4]. This prevents label leakage. At billions of rows it relies on optimizations such as **bloom filters and salted joins** [10].
- Window aggregations (for example, 90-day averages), derived features and PySpark/Spark-SQL UDFs.
- A **unified API** where the same transformation runs offline, in streaming and online, plus a registry and UI for search, lineage and access control [10].

**History and scale.** Development began in 2017. The majority of LinkedIn ML applications were onboarded by 2020. Feathr was open-sourced in 2022 with Azure integration and joined LF AI & Data as a sandbox project in September 2022. The README states production use at LinkedIn for more than 6 years [10]. The Chapter 8 summary of the community talk reports "hundreds of models", "thousands of features", petabyte scale, feature experimentation going "from weeks to days", and running time improved "as much as 50%" [11]. Use cases named in the slides include **"Learning Hiring Preferences: The AI Behind LinkedIn Jobs"**, **"Personalized Recommendations in LinkedIn Learning"** and **"Near real-time features for near real-time personalization"** [11].

### 23.3.2 Venice: the derived-data online store

Ranking requests fan out to thousands of candidate keys, so the online feature store is read-dominated. LinkedIn's open-source **Venice** [12] is the serving substrate:

- **Ingestion:** batch push (full dataset swap from Hadoop/Spark), incremental push, streaming writes via Samza or an online producer, "write compute" partial updates, and hybrid batch-plus-stream stores with a configurable rewind time.
- **Read paths:** single and batch get, plus **server-side read compute** (dot product, cosine similarity, field projection). This lets embedding similarity run next to the data.
- **Three client tiers with documented latency** [12]:

| Client | Hops | Reported latency |
|---|---|---|
| Thin client | 2 network hops, stateless | < 10 ms |
| Fast client | 1 hop, partition-aware | < 2 ms |
| Da Vinci client | 0 hops, stateful local cache | < 1 ms |

- Active-active cross-region replication with **CRDT-based conflict resolution**, multi-tenancy, and change-data-capture streams. The README describes Venice as "particularly suitable as the stateful component backing a Feature Store, such as Feathr" [12].

```
  Hadoop/Spark offline features --batch push--\
                                               >--> Venice store (per region,
  Kafka -> Samza nearline features --stream---/     active-active, CRDT merge)
                                                        |   read compute:
  Feathr defs (anchors, derived, registry)              |   dot / cosine
     |-- offline: PIT join onto impressions --> training data
     `-- online: batch-get features <---------- ranking services
                         (thin <10ms | fast <2ms | Da Vinci in-process <1ms)
```

### 23.3.3 Galene: search infrastructure as a retrieval engine

**Galene** is LinkedIn's Lucene-based search architecture. It was announced on the LinkedIn engineering blog in 2014 [13] and powers member (people) search, job search and other verticals. In LinkedIn's design, search and recommendation overlap heavily. Job recommendations, Recruiter and PYMK-adjacent "people search" all retrieve from inverted indexes, and DeText-style models rank the results [13][14]. Based on the 2014 announcement, as summarized in secondary coverage that could not be re-fetched in this session [13], Galene's distinctive features were:

- offline index construction on Hadoop combined with live index updates;
- static rank and early termination;
- a federated broker/searcher tier.

Treat these architectural details as approximate.

**Lesson.** At LinkedIn the "retrieval layer" for many recommenders is a search engine with structured query rewriting (title, skills, location), not only an ANN index. This matters for Jobs, where hard constraints such as location, seniority and visa requirements are better expressed as index filters. The same insight returns in LiNR's attribute pre-filtering on GPUs (Section 23.6).

---

## 23.4 Feed ranking: from federated FPR/SPR to multi-objective autotuning

### 23.4.1 Architecture (2019)

LinkedIn's 2019 "Community-focused Feed optimization" post [15] describes a **federated funnel**:

- Multiple **first-pass rankers (FPRs)**, one per inventory (network updates, jobs, recommended content, and so on), preselect and score candidates.
- A **second-pass ranker (SPR)** merges them into one list.
- **FollowFeed**, which serves updates from a member's network, was the dominant FPR, supplying **more than 80% of feed updates** [15].

```
   FollowFeed FPR ----+   (network updates; >80% of feed items, 2019)
   Jobs FPR ----------+
   Recommended FPR ---+--> SPR: unified features, multi-objective score
   PYMK / modules ----+        sum_k w_k * P(action_k) (+ creator-side terms)
   Ads (separate auction) -------------+
                                       v
                      slotting (ads in allowed positions) -> Feed
```

Chapters 1 and 15 discuss the trade-off. The federated design matches the organization, since each inventory team owns its FPR, but it pushes **cross-source calibration** onto the SPR. If one FPR's score distribution drifts, its share of the feed drifts too.

### 23.4.2 Objectives and the value function

The Feed ranks by a **linear combination of per-objective probabilities**: viral actions (react, comment, share), contributions, clicks, and downstream effects on creators. The 2019 post adds **creator-side value**, meaning the value of timely feedback to the poster and their community, not only the viewer's own engagement [15].

### 23.4.3 Autotuning the weights (2018-2020)

ML engineers originally set the combination weights by hand, which LinkedIn described as very inefficient [16][17]. The automated approach, from LinkedIn's "Online parameter selection for web-based ranking problems" (KDD 2018, summarized in [17]) and covered in Chapter 4:

1. Treat each online metric as a black-box function of the control parameters (for example the click weight), and model it with a **Gaussian process** fitted on A/B data.
2. Optimize a **primary metric subject to constraints** on others. LinkedIn observed that contributions fall as the click weight rises while normalized clicks rise, so the optimum sits on the constraint boundary.
3. Choose the next parameters with a **Thompson-sampling-style** procedure. Draw functions from the GP posterior, take each draw's argmax, and estimate the probability that each parameter is optimal.
4. Because ramp volume is limited, run many low-traffic experiments in parallel.

**Lesson.** Separate *prediction* (calibrated per-action models) from *policy* (a small, versioned value function), and tune the policy online with constrained Bayesian optimization. LinkedIn, Meta, X and Pinterest all keep this split (Chapter 4).

### 23.4.4 Feed retrieval in the LLM era (2024-2026)

Two later systems rebuilt the retrieval side of the Feed:

- **LiNR (2024)** for out-of-network post retrieval on GPUs (Section 23.6).
- **LLM dual-encoder retrieval (2025, AAAI 2026)** [19]. LinkedIn fine-tuned **LLaMA 3** as a dual encoder. Member and post texts are encoded separately and pooled into embeddings. The system retrieves about **2,000 candidates from hundreds of millions of posts** within a few-millisecond budget at several thousand QPS (approximate). Two findings stand out:
  - numerical features written into prompts work better when **quantized into tokens**;
  - the largest online gains went to **newer members** with sparse networks, the cold-start regime where content-based LLM embeddings beat ID and graph signals [19].

Secondary coverage of a 2026 LinkedIn post describes a further consolidation. The multi-source retrieval described above is replaced by a unified LLM retrieval system, followed by a transformer-based sequential "generative recommender" ranker (Chapter 1). These details are from secondary sources and are approximate. A 2026 LinkedIn paper on a **Connected Content Retriever** describes "GPU Retrieval as Ranking": the whole candidate set sits in GPU memory and every candidate is scored with dense matrix operations. It also describes **SilverTorch**, which implements ANN search and filtering as tensor operations with a fused int8 IVF kernel and a GPU bloom-filter index [21].

```
  2019:  FPR_1 ... FPR_n (per-inventory)  -->  SPR (GLMix/DNN)  --> blend
  2024:  + LiNR GPU out-of-network retriever; LiRank deep SPR; LiGNN embeddings
  2025:  LLM (LLaMA-3) dual-encoder retrieval, ~2k cands, quantized numeric tokens
  2026:  unified LLM retrieval + sequential generative ranker (approx., secondary)
         GPU "retrieval as ranking" + SilverTorch tensor ANN for pre-ranking
```

---

## 23.5 LiRank: industrial large-scale ranking (KDD 2024)

### 23.5.1 Problem

By 2023 LinkedIn ran separate deep rankers for Feed, Jobs and Ads, each tuned independently. LiRank [18] is a framework that brings "state-of-the-art modeling architectures and optimization methods" into production across all three, along with the engineering needed to train and serve large rankers.

### 23.5.2 Architecture and key techniques

From the paper's abstract and the summaries verified in Chapters 3, 4 and 16 [18]:

- **Residual DCN.** DCNv2 extended with **attention and residual connections** in the cross network, using **low-rank** cross matrices. The paper reports a reduction of nearly **30% in parameters**.
- **A unified architecture** combining Residual DCN with **Dense Gating**, a large MLP and **Transformers** over member history sequences. The paper reports how to combine and tune these blocks.
- **Isotonic calibration layer trained jointly with the model.** It sits inside the network instead of a post-hoc Platt or isotonic fit. A 2026 follow-up paper generalizes it into one learnable component that also handles position debiasing and multi-task bias correction [20].
- **Deep-learning-based explore/exploit** in production.
- **Model compression for serving:** quantization and **vocabulary compression** of the large ID embedding tables, so large models fit within serving memory and latency budgets.
- **Incremental / warm-start training** to keep models fresh (as described in the paper; details not re-checked).

```
   sparse IDs --> embedding tables (compressed vocab, quantized)
   dense feats ----------------------------+
   member history seq --> Transformer -----+--> concat
                                           |
              +----------------------------+-----------------+
              |                            |                 |
        Residual DCN               Dense Gating          wide MLP
     (low-rank, attention,             |                    |
      residual cross layers)           |                    |
              +-------------+----------+--------------------+
                            v
                  multi-task heads (Feed actions / job apply / ad click)
                            v
                 jointly trained isotonic calibration layer
                            v
                calibrated P(action)  -->  value model / auction
```

### 23.5.3 Reported impact

From the abstract (not re-fetched; approximate): about **+0.5% member sessions** in Feed, about **+1.76% qualified job applications** for Jobs search and recommendations, and about **+4.3% Ads CTR** [18].

### 23.5.4 Lessons

- **Low-rank cross layers are a large parameter win at small quality cost.** LinkedIn and Pinterest both run low-rank DCN variants in production (Chapter 3).
- **Calibration belongs inside the model** when calibrated probabilities feed a value function or an auction. A jointly trained isotonic layer avoids the drift between a model and a separately fitted calibrator.
- **Compression is part of the modeling work.** Vocabulary compression and quantization decide whether a better model can ship at all.

---

## 23.6 Retrieval and graph learning: LiNR and LiGNN

### 23.6.1 LiNR: model-based neural retrieval on GPUs (2024)

**Problem.** Out-of-network Feed post retrieval needed attribute filters (language, freshness and so on). Standard KNN with **post-filtering** lost recall when filters removed many results [22].

**Design** [22]:

- **Item vectors and model weights live in the same model binary.** Index construction becomes part of training and export, so the index cannot fall out of sync with the query tower.
- **Exhaustive GPU scoring with attribute pre-filtering** over a **billion-sized index**. PyTorch and TensorFlow lacked efficient pre-filtering, so LinkedIn built custom indexing and filtering ops.
- Multi-embedding retrieval, cold-start strategies and **quantization** for larger indexes. Secondary reviews describe 1-bit random-projection compression and near-real-time index updates through change-data capture (approximate).
- Result: about **+3% relative professional daily active users** [22].

```
  model binary = { query tower weights, item matrix [N x d], item attrs }
  request:  q    = tower(member)                       (GPU)
            mask = prefilter(attrs)                    (GPU, exact)
            s    = q . items^T over masked rows        (dense / quantized)
            topk(s) --> ranking
```

LiNR is a step toward merging retrieval and ranking into one GPU model. The 2026 "retrieval as ranking" and SilverTorch work continues in the same direction [21].

### 23.6.2 LiGNN: production GNNs (KDD 2024)

**Problem.** LinkedIn's heterogeneous graph covers members, jobs, companies, posts, ads, groups and skills. Static shallow embeddings could not handle cold-start entities (new jobs, new posts), temporal drift, or the cost of training on a graph this large [23].

**Design** [23] (see Chapter 7 for the full treatment):

- **Temporal graph architectures with long-term losses.** Neighbor sampling respects edge timestamps, so the model learns to predict future engagement from past neighborhoods.
- **Graph densification for cold start.** Low-degree nodes get extra edges to similar entities, so the GNN has neighbors to aggregate.
- **ID embeddings plus multi-hop neighbor sampling**, combining transductive capacity with inductive generalization.
- **Training systems:** about a **7x training speedup** from adaptive neighbor sampling, grouping and slicing of training batches, a specialized shared-memory queue, and local gradient optimization.
- **Nearline inference** keeps embeddings fresh. Embeddings flow into downstream rankers and retrievers as features.

```
 heterogeneous graph (member/job/post/ad/company/skill, typed + timestamped edges)
     |  densify low-degree nodes (cold start)
     v
 temporal multi-hop sampling --> GNN (ID emb + content feats, long-term losses)
     |  7x faster: adaptive sampling, batch grouping/slicing, shm queue
     v
 embeddings (batch + nearline) --> Venice / ANN --> Feed, Jobs, PYMK, Ads models
```

**Reported A/B impact** (approximate) [23]:

- about +1% relative job-application **hearing-back rate**;
- about **+2% Ads CTR**;
- about +0.5% Feed engaged DAU;
- about +0.2% sessions;
- about +0.1% weekly active users from **people recommendations (PYMK)**.

---

## 23.7 People You May Know (PYMK)

PYMK is LinkedIn's oldest graph recommender and the main driver of network growth. Its classical shape is a **triangle-closing** candidate generator (friends of friends, plus co-attributes such as shared company, school or contact-book overlap), followed by a learned ranker that predicts invitation and acceptance (Chapter 7). The publicly verifiable recent facts in this session's sources are:

- PYMK is one of the four main consumers of **LiGNN** embeddings, with about a +0.1% WAU lift from people recommendations attributed to GNN features [23].
- GLMix was used for the related "**People Also Viewed**" recommender [6].

Detailed PYMK serving numbers, such as candidate counts and latency, were not found in sources reachable for this chapter. They are deliberately left out rather than estimated.

**Design notes for a PYMK-style system** (synthesis from the graph-recsys patterns in Chapter 7 and LiGNN):

```
  member m
    |-- 2-hop graph walk (friends-of-friends, weighted by edge strength)
    |-- co-attribute joins (same company/school/time overlap)
    |-- address-book / imported contacts (where consented)
    |-- GNN embedding ANN (LiGNN-style; covers cold-start members)
    v
  candidate union (dedupe, filter existing connections, blocked, pending)
    v
  ranker: P(invite) x P(accept | invite)  (reciprocity), plus growth objectives
    v
  diversification + frequency caps --> PYMK module (also slotted into Feed)
```

The key modeling point is **reciprocity**. A good PYMK suggestion must be accepted by the *other* member, so the objective is a product of both sides' propensities, unlike a one-sided content recommendation.

---

## 23.8 Job recommendations and the Jobs marketplace

Jobs is the surface where LinkedIn's history shows most clearly, because every generation of its stack was applied here:

| Generation | Model | Source |
|---|---|---|
| 2016 | GLMix: global + per-member + per-job random effects | [6][7] |
| 2020 | GDMix: DeText deep fixed effect + per-member LR | [8][9][14] |
| 2020-22 | Feathr features, for example "Learning Hiring Preferences" | [10][11] |
| 2024 | LiRank: about +1.76% qualified applications (approx.) | [18] |
| 2024 | LiGNN embeddings: about +1% hearing-back rate (approx.) | [23] |
| 2025 | 360Brew: job apply/view/dismiss as a text prompt | [1][2] |

**Marketplace considerations.** Job recommendations are two-sided. LinkedIn's reported metrics are not raw clicks. They are **"qualified job applications"** [18] and the **"hearing back" rate** [23], meaning applications that a recruiter actually responds to. Optimizing for the employer-side outcome keeps the system from flooding popular postings with poorly matched applications.

**Hiring preferences.** The Feathr talk cites "Learning Hiring Preferences: The AI Behind LinkedIn Jobs" as a flagship Feathr use case [11]. In that work, recruiter-side signals become features of the member-job ranker. The original post could not be fetched here.

---

## 23.9 360Brew: a decoder-only foundation model for ranking (2025)

### 23.9.1 Problem

The traditional stack relies on "hundreds of thousands of handcrafted features" and on ID-based embedding tables, with three structural weaknesses the report names [1]:

- **Cold start:** ID models cannot generalize to new members and items without frequent retraining.
- **Interactions:** cross features have to be hand-crafted or learned by special architectures such as DCN.
- **Domain generalization:** a model trained on Feed cannot move to Jobs, or from predicting interactions to predicting ratings.

360Brew's stated primary objective is "to enhance AI development productivity across LinkedIn by centralizing common modeling components into a foundational layer". It also enables governance and moderation at that layer [1].

### 23.9.2 Formulation

Each member's profile and history is treated as a **many-shot in-context learning** problem. The model approximates the joint distribution

P(m, (e1, i1), ..., (eN, iN))

over a member profile m and the member's historical entities e_j with interactions i_j, all verbalized as text. Any task then becomes a conditional marginal: P(i_t, i_t+1, ... | task instruction, m, history, new entities e_t, ...). This is next-token prediction, the same objective decoder LLMs train on [1].

**Prompt structure** (job recommendation example from the report) [1][2]:

```
 Instruction:  member profile + jobs + past interactions (applied / viewed /
               dismissed / no interaction); predict apply/view/dismiss for the
               "Question" job. (May include steering, e.g. "focus on skills,
               location, years of experience"; "30% weight to profile match,
               70% to historical activity")
 Member Profile:  current position, company, location
 Past job interaction data:
    applied: [Title, Location, Country, Company, Description] ...
    viewed:  [ ... ]
 Question:  Will the member apply to the following job: [ ... ]
 Answer:    The member will apply
```

For comparison with production binary rankers, prompts are written so the model answers yes or no. The **logits of the answer tokens** are used as scores to compute AUC and similar metrics [1].

### 23.9.3 Model and data

- **150B parameters**, decoder-only, built on the **Mixtral 8x22B** pre-trained MoE architecture [1].
- Further trained on LinkedIn raw entity data (member profiles, job descriptions, posts) and interaction data (including job applications) across **5+ surfaces**. The data is primarily first-party, from users outside the European Union [1].
- Evaluated on **30+ predictive tasks across 8+ surfaces**, split into T1 (in-domain, with at least a one-month gap between training and test data) and T2 (out-of-domain surfaces and tasks not seen in training) [1].
- Built by a small team over **9 months**. The model is described as "pre-production" (V1.0) [1].
- The report says the language models are used only for ranking and recommendation, not for content creation [1].

### 23.9.4 Results and scaling observations

- **Quality:** "comparable to or exceeding" current production systems on offline metrics, **without task-specific fine-tuning** [1].
- **Data scaling:** scaling processed data by **over 3x**, across **five releases over nine months**, improved T1 performance significantly over baselines, and "the slope ... does not appear to be flattening" [1].
- **Model scaling:** larger pre-trained architectures performed better [1].
- **History (context) scaling:** performance rises with maximum context length, normalized to an 8K-token baseline. The authors frame this as "replac[ing] the problem of feature engineering with the more tractable challenge of scaling up architecture depth and context length" [1].
- **Out-of-domain:** on 4 T2 tasks across 4 unseen surfaces, performance was similar to or better than production models [1].
- **Cold start:** the gap over production is largest when member history is cut from 100 interactions down to 5 [1].
- **Temporal robustness:** performance degrades less than baselines as test data moves further from the training period. The authors attribute this to in-context adaptation, which could reduce retraining frequency [1].

### 23.9.5 Lessons and open questions

- The report covers **ranking** only. Using the same model for **retrieval through embeddings** is stated as future work [1].
- Serving a 150B MoE at Feed QPS is not discussed in V1.0. The parallel 2025 Feed retrieval work fine-tunes a much smaller LLaMA 3 dual encoder [19]. That suggests the large model is a teacher or a ranker for high-value surfaces rather than a scorer for every candidate (an inference, not a published fact).
- Training LLM-scale models at LinkedIn is supported by open-source infrastructure such as **Liger-Kernel**. It provides Triton kernels (RMSNorm, RoPE, SwiGLU, fused linear cross-entropy) and reports about **+20% multi-GPU training throughput** and about **60% memory reduction**, with up to 80% memory savings for post-training losses such as DPO and ORPO [24].

---

## 23.10 Experimentation and responsible ranking

### 23.10.1 XLNT, Lix and T-REX

LinkedIn's A/B platform **XLNT** (Xu et al., KDD 2015 [26]) uses the **Lix** targeting DSL over member attributes and supports gradual ramps (for example 1% to 5% to 50%), with hundreds of metrics per test. The **T-REX** engine re-implemented Lix evaluation as compiled, in-process code, about **20x faster**, at a scale of trillions of variant evaluations per day (approximate) [27]. Speed matters because ranking services check experiment membership several times per request. The constrained-BO Feed tuning in Section 23.4.3 runs on top of this infrastructure.

### 23.10.2 Fairness-aware ranking in Recruiter

For LinkedIn Talent Search, Geyik, Ambler and Kenthapadi (KDD 2019) [28] describe a **post-processing re-ranker**. It makes the gender distribution of every **prefix** of the top-k results match the distribution among *qualified* candidates for that query, using the DetGreedy, DetCons and DetRelaxed algorithms. It shipped to all Recruiter users worldwide, with a reported near three-fold increase in queries with representative results and no significant drop in business metrics (Chapter 14).

### 23.10.3 Explainability tooling

**FastTreeSHAP** [25] is LinkedIn's open-source faster TreeSHAP. Variant v1 is about 1.5x faster at equal memory and v2 about 2.5x faster at higher memory, for explaining tree-based models with millions of rows.

---

## 23.11 Comparison of LinkedIn's published recommender systems

| System (year) | Layer | Core idea | Scale (public) | Reported impact | Ref |
|---|---|---|---|---|---|
| GLMix / Photon ML (2016) | ranking | fixed + per-member/per-item random effects, coordinate descent | up to hundreds of billions of coefficients | adopted by Jobs, Feed, Ads CTR, People Also Viewed | [6][7] |
| GDMix (2020) | ranking | deep fixed effect (DeText) + per-entity LR | hundreds of millions of random-effect models | MovieLens AUC 0.62 -> 0.77 with random effects | [8][9] |
| Feathr (2017; OSS 2022) | features | declarative features, PIT joins, registry | hundreds of models, thousands of features, PB | weeks -> days experimentation; up to 50% faster | [10][11] |
| Venice | serving store | batch + stream derived data, read compute | Da Vinci < 1 ms reads | substrate for Feathr | [12] |
| Feed FPR/SPR (2019) | funnel | federated per-inventory rankers + SPR | FollowFeed > 80% of updates | creator-side objectives | [15] |
| Feed autotuning (2018-20) | policy | GP + Thompson sampling, constrained | parallel low-traffic ramps | replaced hand tuning | [16][17] |
| LiRank (2024) | ranking | Residual DCN, gating, transformers, isotonic layer | ~30% fewer DCN params | Feed +0.5% sessions; Jobs +1.76% qualified apps; Ads +4.3% CTR (approx.) | [18] |
| LiGNN (2024) | embeddings | temporal GNN, densification, nearline | 7x training speedup | Ads +2% CTR; Jobs +1% hearing back (approx.) | [23] |
| LiNR (2024) | retrieval | GPU exhaustive + pre-filter, index in model | ~1B-item index | +3% professional DAU | [22] |
| LLM dual encoder (2025) | retrieval | LLaMA 3 dual encoder, quantized numeric tokens | ~2k of hundreds of millions | biggest gains for new members | [19] |
| 360Brew V1.0 (2025) | ranking FM | 150B Mixtral-based decoder, text prompts | 30+ tasks, 8+ surfaces | matches/exceeds prod offline, zero-shot OOD | [1][2] |
| CCR / SilverTorch (2026) | pre-ranking | GPU retrieval as ranking, tensor ANN | int8 IVF, GPU bloom filter | n/a here | [21] |

---

## 23.12 Key takeaways for system designers

1. **Model sprawl is the real enemy at scale.** LinkedIn's arc runs from GLMix/GDMix (a shared training paradigm) through Feathr (shared features), LiRank (a shared architecture) and LiGNN (shared embeddings) to 360Brew (one model for 30+ tasks). It is one long campaign against the cost of "thousands of predictive models" [1]. Count the models and features your organization maintains, and budget for that cost.
2. **Fixed plus random effects is still a strong personalization baseline.** Per-member coefficients trained after an ID-based partition scale to hundreds of millions of members on commodity hardware and stack cleanly on any global model through offsets [6][8].
3. **Point-in-time correctness and a single feature definition come first.** Feathr's PIT joins and unified offline/online definitions, plus Venice's tiered read paths (< 10 ms, < 2 ms, < 1 ms), are what make the models trustworthy and fast [10][12].
4. **Federated funnels need explicit calibration.** An FPR/SPR design fits the org chart but makes the SPR responsible for cross-source calibration. Calibrate inside the model (LiRank's isotonic layer) and tune the value function online with constrained Bayesian optimization [15][17][18].
5. **Ship compression together with the model.** Low-rank DCN, vocabulary compression and quantization are what let LiRank serve large models [18]. LiNR's quantized GPU index is what makes billion-item exhaustive retrieval affordable [22].
6. **Pre-filter, don't post-filter.** For filtered retrieval (language, freshness, location), exact pre-filtering on the GPU beats ANN plus post-filtering on recall [22].
7. **Graph learning pays off as a platform.** LiGNN's individual lifts are around 0.1-2%, but they apply to Feed, Jobs, PYMK and Ads at once. Most of the engineering effort goes into temporal sampling, densification and data-loader throughput, not the convolution [23].
8. **Optimize marketplace outcomes, not clicks.** For Jobs, report qualified applications and hearing-back rates. For PYMK, model reciprocity [18][23].
9. **Text foundation rankers trade compute for engineering effort.** 360Brew's evidence is that more data, more context and larger models keep helping. The gains are largest for cold-start members and out-of-domain surfaces, and the model ages more slowly over time [1]. Expect to pair such a model with smaller, cheaper LLM retrievers and rankers at high-QPS stages [19].
10. **Numbers in prompts need care.** Quantizing numeric features into tokens improved LLM retrieval [19]. 360Brew avoids time-window count features entirely by putting raw history into the context [1].
11. **Ranking ships with experimentation and fairness infrastructure.** A fast in-process assignment engine (T-REX) and auditable post-processing for fairness (Recruiter) are part of the recommender, not add-ons [26][27][28].

---

## References

1. 360Brew Team (Firooz, H., Sanjabi, M., et al.), LinkedIn Foundation AI Technologies. "360Brew: A Decoder-only Foundation Model for Personalized Ranking and Recommendation." arXiv 2501.16450, 2025 (technical report PDF read from the authors' repository). https://arxiv.org/abs/2501.16450
2. LLMPersonalization (LinkedIn authors). "360Brew-V1" repository README and report PDF. GitHub, 2025. https://github.com/LLMPersonalization/360Brew-V1
3. 360Brew Team, LinkedIn. "360Brew: A Decoder-only Foundation Model for Ranking and Recommendation" (V1.0 report PDF). GitHub, 2025. https://github.com/LLMPersonalization/360Brew-V1/blob/main/360brew_v1.pdf
4. Feathr project (LinkedIn). "Point-in-time Correctness and Point-in-time Join" (docs/concepts/point-in-time-join.md). GitHub, 2022. https://github.com/feathr-ai/feathr/blob/main/docs/concepts/point-in-time-join.md
5. Feathr project (LinkedIn). "Feathr Concepts for Beginners" (docs/concepts/feathr-concepts-for-beginners.md). GitHub, 2022. https://github.com/feathr-ai/feathr/blob/main/docs/concepts/feathr-concepts-for-beginners.md
6. LinkedIn. "Photon ML" repository README (GLM, GAME, GLMix on Spark). GitHub, 2016-2020. https://github.com/linkedin/photon-ml
7. Zhang, X., Zhou, Y., Ma, Y., Chen, B.-C., Zhang, L., Agarwal, D. (LinkedIn). "GLMix: Generalized Linear Mixed Models For Large-Scale Response Prediction." KDD 2016. http://www.kdd.org/kdd2016/papers/files/adf0562-zhangA.pdf
8. LinkedIn. "GDMix: Generalized Deep Mixed Model" repository README. GitHub, 2020. https://github.com/linkedin/gdmix
9. Shi, J., et al. (LinkedIn Engineering). "GDMix: A Deep Ranking Personalization Framework." LinkedIn Engineering Blog, 2020 (linked from [8]; not fetched). https://engineering.linkedin.com/blog/2020/gdmix--a-deep-ranking-personalization-framework
10. Feathr project (LinkedIn; LF AI & Data). "Feathr" repository README and docs (point-in-time joins, concepts). GitHub, 2022-2023. https://github.com/feathr-ai/feathr
11. Feathr project (LinkedIn and Microsoft). "Feathr Feature Store Talk" and "Feathr Community Talk - An Enterprise-Grade High Performance Feature Store" slide decks. GitHub, 2022. https://github.com/feathr-ai/feathr/tree/main/docs/talks
12. LinkedIn. "Venice: Derived Data Platform for Planet-Scale Workloads" repository README. GitHub, 2022-2026. https://github.com/linkedin/venice
13. LinkedIn Engineering. "Did you mean 'Galene'?" LinkedIn Engineering Blog, 2014 (not re-fetched; architectural details approximate). https://engineering.linkedin.com/blog/2014/03/did-you-mean-galene
14. LinkedIn. "DeText: A Deep Neural Text Understanding Framework for Ranking and Classification Tasks" repository README (cites Guo et al. SIGIR 2019, KDD 2019, CIKM 2020; Jia et al. KDD 2020). GitHub, 2020. https://github.com/linkedin/detext
15. LinkedIn Engineering. "Community-focused Feed optimization." LinkedIn Engineering Blog, June 2019. https://engineering.linkedin.com/blog/2019/06/community-focused-feed-optimization
16. LinkedIn (via KDnuggets). "Autotuning for Multi-Objective Optimization on LinkedIn's Feed Ranking." 2020. https://www.kdnuggets.com/2020/08/autotuning-multi-objective-optimization-linkedin-feed-ranking.html
17. Colyer, A. "Online parameter selection for web-based ranking problems" (summary of LinkedIn KDD 2018 paper). The Morning Paper, 2018. https://blog.acolyer.org/2018/10/08/online-parameter-selection-for-web-based-ranking-problems/
18. Borisyuk, F., et al. (LinkedIn). "LiRank: Industrial Large Scale Ranking Models at LinkedIn." KDD 2024, arXiv 2402.06859. https://arxiv.org/abs/2402.06859
19. LinkedIn. "Large Scale Retrieval for the LinkedIn Feed using Causal Language Models." arXiv 2510.14223, 2025 (AAAI 2026). https://arxiv.org/abs/2510.14223
20. LinkedIn. "Isotonic Layer: A Unified Framework for Recommendation Calibration and Debiasing." arXiv 2603.06589, 2026. https://arxiv.org/abs/2603.06589
21. LinkedIn. "Connected Content Retriever: Dense Graph Edge Features Powering Pre-Ranking at LinkedIn." arXiv 2609.22441, 2026. https://arxiv.org/pdf/2609.22441
22. Borisyuk, F., et al. (LinkedIn). "LiNR: Model Based Neural Retrieval on GPUs at LinkedIn." arXiv 2407.13218, 2024. https://arxiv.org/pdf/2407.13218
23. Borisyuk, F., et al. (LinkedIn). "LiGNN: Graph Neural Networks at LinkedIn." KDD 2024, arXiv 2402.11139. https://arxiv.org/abs/2402.11139
24. LinkedIn. "Liger Kernel: Efficient Triton Kernels for LLM Training" repository README and technical report (arXiv 2410.10989). GitHub, 2024-2025. https://github.com/linkedin/Liger-Kernel
25. LinkedIn. "FastTreeSHAP" repository README. GitHub, 2022. https://github.com/linkedin/FastTreeSHAP
26. Xu, Y., Chen, N., Fernandez, A., Sinno, O., Bhasin, A. (LinkedIn). "From Infrastructure to Culture: A/B Testing Challenges in Large Scale Social Networks." KDD 2015. https://dl.acm.org/doi/10.1145/2783258.2788602
27. LinkedIn Engineering. "Making the LinkedIn Experimentation Engine 20x Faster." LinkedIn Engineering Blog, 2019. https://engineering.linkedin.com/blog/2019/06/making-the-linkedin-experimentation-engine-20x-faster
28. Geyik, S. C., Ambler, S., Kenthapadi, K. (LinkedIn). "Fairness-Aware Ranking in Search and Recommendation Systems with Application to LinkedIn Talent Search." KDD 2019, arXiv 1905.01989. https://arxiv.org/abs/1905.01989
