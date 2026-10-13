# 19. Meta Case Study: From News Feed Ranking to Andromeda, Lattice, GEM and Generative Recommenders

> **Sourcing note.** This chapter was written when live web search and fetch were not
> available to the research agent: the shared search budget was used up and the egress proxy
> refused arXiv and Meta domains. All facts below come from primary and secondary sources that
> other chapters of this Atlas had already retrieved and checked (Chapters 1-4, 6, 9-11,
> 15, 16). Their URLs are listed in the References. Topics that could not be checked against
> a fetched source in this run (Threads feed ranking, Facebook Marketplace, Instagram Reels
> specifics, the "Scaling User Modeling" paper, MTIA chip internals) are marked as gaps in
> Section 9 and are not described in detail. Numbers marked "approx." are secondhand or
> rounded.

Meta runs some of the largest recommender systems in the world: Facebook Feed, Instagram Feed,
Explore, Reels and Stories, Threads, Marketplace, and an ads system that sits behind all of
these surfaces. Meta has also published more of its recommender architecture than almost any
other company. It released DLRM (the reference deep recommendation model for the industry),
TorchRec and FBGEMM (the open-source training and serving substrate), a series of
ranking-architecture papers (DHEN, Wukong, InterFormer), the generative-recommender line
(HSTU and its successors), and engineering posts on Feed, Explore, Lattice, Andromeda and GEM.

Across about a decade (2014-2026), the published record shows five changes:

1. **From features to embeddings.** The 2014 ads system used GBDT leaf features with online
   logistic regression [16]. DLRM (2019) made sparse-embedding plus MLP models the default,
   and the cost of embeddings drove the hardware [9][12].
2. **From single objectives to value models.** Feed and Explore predict many actions with
   multitask networks and combine them in an explicit, tunable value function [1][7].
3. **From many models to few.** Lattice merged the fragmented ads model fleet into
   multi-domain, multi-task models [22][23].
4. **From cheap retrieval to deep retrieval.** Andromeda put a deep network with a learned
   hierarchical index on GPUs at the stage that sees the most items [25].
5. **From discriminative to generative and foundation models.** HSTU recast recommendation
   as sequence transduction [30], and GEM became an LLM-scale teacher that is distilled into
   the whole ads fleet [27][28].

```
 2014   GBDT + LR ads CTR (online joiner, NE metric)                       [16]
 2018   MSI ("meaningful social interactions") Feed value model            [3]
 2019   DLRM open-sourced; Instagram Explore 3-pass distillation funnel     [9][5]
 2020   Instagram constrained exploration; FB Search embedding retrieval    [6][37]
 2021   Feed ranking explainer; survey-weighted objectives; Neo/ZionEX      [1][2][12]
 2022   DHEN; DSI data-ingestion paper; TorchRec (RecSys)                   [17][15][13]
 2023   Lattice (ads model consolidation); Explore scaled to 4 stages      [22][7]
 2024   HSTU generative recs; Wukong; InterFormer; Andromeda   [30][19][20][25]
 2025   GEM ads foundation model; Lattice paper; DHEN for CVR               [27][23][18]
 2026   GEM trained at LLM scale (MFU 20-25%); ULTRA-HSTU                   [28][34]
```

---

## 1. Facebook News Feed ranking

### 1.1 Problem

Feed has to order, for each of more than 2 billion people, the posts from friends, Pages and
Groups they follow (and later recommended content), in real time, so that the top of Feed
shows what that person will find most valuable [1][4]. An average user has more than 1,000
candidate posts available on a given visit [1].

### 1.2 Architecture: multi-pass ranking on "predictors"

Meta's January 2021 engineering post describes ranking as a series of passes run in parallel
on serving machines that Meta calls *predictors* [1]:

```
  inventory: all eligible posts for the viewer (> 1,000 on average)
        |
        v
  pass 0: integrity processes + eligibility (remove violating / ineligible content)
        |
        v
  pass 1: lightweight model narrows inventory to ~500 most relevant posts
        |
        v
  pass 2: heavy multitask neural networks predict P(action_i) per post
          (like, comment, share, hide, "worth your time" survey answer, ...)
          -> combined into one value score per post
        |
        v
  pass 3: contextual re-ranking (e.g. content-type diversity, avoid runs of
          similar posts) -> final ordered Feed
                                                         (after Meta 2021 [1][4])
```

The structure is the canonical funnel: cheap eligibility filters first, then a light pass,
a heavy pass and a contextual pass. Running integrity first saves heavy-model compute for
content that can actually be shown [1].

### 1.3 Models and the value function

Each post receives several predicted actions from multitask neural networks, and these "all
need to somehow aggregate up to a single value" [1]. The 2021 post uses a running example (a
user named Juan) to show that different predictions can disagree about which post Juan will
respond to most. The tie is broken by weighting each predicted interaction by how
*meaningful* it is, based on survey data about what people say they value [1].

Conceptually:

```
  value(post) = sum over events i of  w_i * P(event_i | viewer, post, context)
```

The weights are a product policy, set from data analysis and surveys, not learned end to end.

### 1.4 MSI and survey-calibrated objectives

- **Meaningful Social Interactions (2018).** Internal documents reported by CNN in 2021
  describe MSI as a point system. Likes, comments, re-shares, reactions and event RSVPs each
  had point values, adjusted for the relationship between the people involved. The weights
  came from surveys (one cited survey covered more than 69,000 people in five countries)
  together with analysis of user data [3]. A 2019 internal note said MSI was "beneficial for
  most metrics" in 2018. The same reporting describes how heavy weights on comments,
  reshares and reaction emojis also amplified outrage-provoking content [3].
- **Survey targets (2019-2021).** Meta added "Is this post worth your time?" surveys. In
  2021 it gave more weight to survey-derived predictions of how meaningful a friend, Page
  or Group is to a person. Classifiers trained to predict survey answers carry the signal to
  posts nobody was surveyed about [2].
- **Transparency.** Meta publishes a "system card" for the Facebook Feed AI system that
  lists the main prediction signals and the stages of the pipeline [4].

### 1.5 Lessons

- Keep prediction (calibrated multitask heads) separate from policy (a small, versioned
  value function). Product trade-offs then change through configuration, not retraining.
- Engagement-only weights optimize what is easy to measure, with second-order effects on the
  content ecosystem. Survey-predicted value is a counterweight [2][3].

---

## 2. Instagram Explore

Explore is Instagram's recommendation surface for content from accounts the user does not
follow. Meta has described it publicly three times (2019, 2020, 2023), which makes it one of
the best-documented recommender systems anywhere.

### 2.1 Explore 2019: ig2vec and the three-pass distillation funnel

The 2019 post "Powered by AI: Instagram's Explore recommender system" says the system
"extracts 65 billion features and makes 90 million model predictions every second" [5].

**Candidate generation.** Explore starts from *accounts*, not individual media. ig2vec is a
word2vec-style framework that treats the sequence of account IDs a user interacts with in a
session as a sentence and learns account embeddings from it [5]. Seed accounts (accounts the
user recently engaged with) are expanded by nearest-neighbor lookup in ig2vec space into
candidate accounts, and their media become candidates [5].

**Ranking funnel.**

```
  seed accounts --ig2vec KNN--> candidate accounts --> their media (+ other sources)
            |
            v
   ~500 candidates
            |   pass 1: distillation model (minimal features), trained to mimic the
            |           combined output of the later stages
            v
   ~150 candidates
            |   pass 2: lightweight neural network
            v
   ~50 candidates
            |   pass 3: full deep neural network (all dense features)
            v
   ~25 shown first (after integrity and diversity rules)
                                  (after Instagram 2019 [5]; counts approx.)
```

**Key trick.** The first pass is a *distilled* model that imitates the heavy stages with a
fraction of the features. Distillation keeps the stages aligned: the cheap stage keeps what
the expensive stage would have chosen, rather than what an independent proxy objective
prefers [5].

### 2.2 Explore 2020: constrained exploration and the log-linear value model

The 2020 post "How Instagram suggests new content" describes the multi-label ranking models
and how their outputs are combined [6]. The models predict positive engagement (like, save)
and negative feedback ("see less", "not interested"). Point-wise heads are trained with
cross-entropy, alongside GBDTs and list-wise LambdaRank models. The value model is
**log-linear** [6]:

```
  Value(post) = P(like) ^ w_like  *  (1 - P(not interested)) ^ w_see_less
```

The weights are tuned with offline replay over user sessions and with online Bayesian
optimization, and they are retuned often as the system changes [6]. In a multiplicative
form, a high predicted negative probability can veto an item however likely a like is.

### 2.3 Explore 2023: four stages and "caching everywhere"

The August 2023 post "Scaling the Instagram Explore recommendations system" (Vorotilov and
Shugaepov) redraws the system as four stages: **retrieval, first-stage ranking,
second-stage ranking and final re-ranking** [7]. Its stated "leitmotif" is the "clever use of
caching and pre-computation in different ranking stages". That is what made heavier models
affordable at every stage and reduced the use of heuristics [7].

**Retrieval.** Many sources run in parallel: heuristics (for example trending posts or
content from followed authors) and ML sources that capture real-time and pre-generated
interests. They are mixed with tunable weights, so different user groups get different
blends [7]. Meta's Explore transparency card lists source types: author-based (media from
accounts you engaged with), media-based (media similar to media you engaged with), item
collaborative filtering, personalized PageRank, and two-tower sparse-network sourcing. Each
source picks hundreds of items from a pool of billions [8]. The main ML source is a two-tower
network. Because the towers are independent after training, item embeddings are computed
offline (typically daily) and indexed. At request time, a fresh user embedding is computed
and an ANN search returns similar items [7].

**First-stage ranking.** This stage reuses the two-tower architecture "because of its
cacheability property". It is trained to predict the output of the second-stage ranker,
which is a form of knowledge distillation between stages [7]. The transparency card gives
the funnel sizes: retrieval returns up to **1,500** media, and the early-stage two-tower
ranker passes the top **100** onward [8].

**Second-stage ranking.** A heavier **multi-task multi-label (MTML)** network with
user-item interaction features predicts probabilities of several events. Its outputs are
combined by a now explicitly **linear** value model [7]:

```
  Expected Value = W_click * P(click) + W_like * P(like) - W_see_less * P(see less) + ...
```

Tuning the weights lets the team "explore different tradeoffs between online engagement
metrics" [7]. Because the MTML model is much more expensive than the two-tower models, Meta
precomputes some users' recommendations during off-peak hours [7].

**Final re-ranking.** Integrity filters plus business rules. Examples are diversity rules
that penalize repeated posts from the same author or seed account [7].

**Weight tuning.** The post contrasts offline tuning with Bayesian optimization, in which
engineers specify only the parameters, the goal metric and regression thresholds on
guardrail metrics [7].

```
 +--------------+  <=1,500  +------------------+  top 100  +-----------------+  +--------+
 | Retrieval    |---------->| 1st-stage rank   |---------->| 2nd-stage rank  |->| Final  |
 | heuristics,  |           | two-tower,       |           | MTML network +  |  | rerank |
 | item-CF, PPR,|           | cached towers,   |           | linear value    |  | integ.,|
 | two-tower ANN|           | distilled from   |           | model (BO-tuned)|  | divers.|
 | (weighted)   |           | 2nd stage        |           |                 |  |        |
 +--------------+           +------------------+           +-----------------+  +--------+
   item tower: offline daily embeddings | user tower: online, fresh features
   some users' heavy-stage results precomputed off-peak
                                   (after Meta 2023 [7] and Explore system card [8])
```

### 2.4 Explore over time

| Aspect | 2019 [5] | 2020 [6] | 2023 [7][8] |
|---|---|---|---|
| Candidate generation | ig2vec account KNN + other sources | (not focus) | weighted mix of heuristic, item-CF, PPR, two-tower ANN |
| Stage count | sourcing + 3 ranking passes | multi-label rankers | retrieval + 2 ranking stages + rerank |
| Funnel sizes | ~500 -> 150 -> 50 -> 25 | n/a | <=1,500 -> 100 -> final |
| Cheap stage | distilled light model | n/a | two-tower distilled from 2nd stage |
| Heavy stage | full DNN | multi-label NN + GBDT + LambdaRank | MTML network |
| Value model | (not detailed) | log-linear, BO-tuned | linear, BO with guardrails |
| Key infra idea | distillation | offline replay + online BO | caching and precomputation |

### 2.5 Lessons

- **Distil downward.** Both the 2019 and 2023 designs train the cheap stage to imitate the
  expensive one. Funnel consistency matters more than the cheap stage's own accuracy.
- **Pick the value-function form on purpose.** Linear is additive and easy to interpret.
  Log-linear lets one negative signal act as a veto.
- **Precompute what you can.** Daily item towers, cached user towers and off-peak heavy
  scoring buy model capacity without breaking the latency budget.

---

## 3. DLRM and the infrastructure it created

### 3.1 DLRM (2019)

"Deep Learning Recommendation Model for Personalization and Recommendation Systems"
(Naumov et al.) defined the reference architecture that most industrial training-systems
work assumes [9][10]:

```
            dense features            sparse features (multi-hot IDs)
                 |                      |        |        |
            +---------+             +------+ +------+ +------+
            | bottom  |             | EMB  | | EMB  | | EMB  |   <- almost all params,
            |  MLP    |             | T1   | | T2   | | Tk   |      model-parallel
            +---------+             +------+ +------+ +------+
                 |                      | pool   | pool   | pool
                 +----------+-----------+--------+--------+
                            |  pairwise interaction (dot products / concat)
                       +---------+
                       | top MLP |   <- data-parallel
                       +---------+
                            |
                        P(click)
```

DLRM deliberately stops at second-order (pairwise dot-product) interactions between
embedding vectors, for cost reasons. The open-source code supports synchronous distributed
training (gloo, NCCL, MPI) and includes MLPerf benchmark tooling [10].

**Why it shaped hardware.** Meta's study of production DLRM training found that embedding
operators are memory-bound and dominated by communication, and that the variety of table
sizes and sparse-feature profiles makes a single hardware configuration a poor fit for all
models [11]. Embedding tables, not FLOPs, set the cost.

### 3.2 Neo and ZionEX (2021-2022): synchronous training of trillion-parameter models

Before about 2020, Meta trained recommendation models asynchronously on CPU parameter-server
clusters. Asynchrony hurt quality and reproducibility, and parameter-server traffic limited
throughput [11]. Neo/ZionEX moved this to synchronous GPU training [12]:

- **ZionEX nodes:** 8 GPUs (A100-class in the paper) on NVLink, plus a dedicated RoCE
  scale-out network with several NICs per node, so every GPU has its own path for inter-node
  all-to-all traffic [12].
- **4D embedding parallelism** chosen per table: table-wise, row-wise, column-wise and
  data-parallel (replicated) [12].
- **Hybrid kernel fusion** (all tables on a device in one kernel, with the sparse optimizer
  update fused into backward), a cost-model **sharding planner**, software-managed embedding
  caching across HBM, DRAM and UVM, and **quantized collectives** for embedding all-to-all
  traffic [12].
- Reported scale: models of up to **12 trillion parameters**, and time-to-solution speedups of
  up to roughly 40x over the earlier CPU asynchronous setup (approx., headline figure) [12].

### 3.3 TorchRec, FBGEMM and the data pipeline

- **TorchRec** productionizes these ideas as a PyTorch library. It provides data-parallel,
  table-wise, row-wise, column-wise and hybrid sharders, an automatic planner, a pipelined
  training loop that overlaps the ID all-to-all with dense compute, and C++ inference
  export. The project says it "powers many production RecSys models at Meta" [13].
- **FBGEMM / FBGEMM_GPU** supplies table-batched embedding (TBE) kernels with fused
  optimizers, plus low-precision GEMMs with row-wise and outlier-aware quantization for
  inference [14].
- **Data Storage and Ingestion (DSI, ISCA 2022).** Features and events are logged through
  Scribe, joined with labels into columnar Hive tables (DWRF) on the Tectonic filesystem, and
  streamed to trainers by a disaggregated CPU preprocessing service (DPP). Meta found that
  storage and ingestion can cost as much as, or more than, GPU training [15].

```
 serving --> Scribe --> ETL / label join --> Hive (DWRF) on Tectonic
                                                  |
                         DPP CPU fleet (read cols, decode, transform)
                                                  |  tensors
                                                  v
                       GPU trainers (TorchRec: sharded EMB + DP MLP, FBGEMM TBE)
```

### 3.4 Retrieval building blocks

Two Meta systems underlie much of the industry's retrieval work:

- **Faiss** (2017), the similarity-search library used for ANN retrieval over learned
  embeddings, including billion-scale GPU search [39].
- **Embedding-based retrieval in Facebook Search** (KDD 2020). It reported that random
  negatives beat "impressed but not clicked" negatives for retrieval recall, and that
  offline hard negatives should come from a middle rank band, mixed with a large majority of
  easy negatives (approx. 100:1) [37].
- **PyTorch-BigGraph** learns shallow graph embeddings over billions of entities and
  trillions of edges for use as retrieval vectors and ranking features [38].
- **Horizon / ReAgent** (2018) is Facebook's applied batch-RL platform, with counterfactual
  policy evaluation before launch. It was used for decisions such as notifications [40].

---

## 4. Ads ranking: from GBDT+LR to DHEN, Wukong, InterFormer and Lattice

### 4.1 2014: "Practical Lessons from Predicting Clicks on Ads at Facebook"

At more than 750M daily active users and more than 1M advertisers, Facebook used boosted
trees as a learned feature transform. The leaf index of each tree became a one-hot feature
for an online logistic regression, which beat either model alone by more than 3% normalized
entropy (NE) [16]. The paper also introduced practices that are still standard: an online
joiner that matches impressions and clicks by request ID within a waiting window; the finding
that freshness matters (NE degrades as training data falls days behind); and negative
downsampling with recalibration q = p / (p + (1 - p) / w) [16]. NE (log loss normalized by the
entropy of the background CTR) is still Meta's headline offline metric.

### 4.2 Ranking-architecture research, 2022-2025

| Model | Year | Core idea | Reported result |
|---|---|---|---|
| DHEN [17] | 2022 | Each layer is an ensemble of heterogeneous interaction modules (dot, self-attention, conv, DCN, linear); layers stack hierarchically | +0.27% NE over AdvancedDLRM; Hybrid Sharded Data Parallel up to 1.2x throughput over FSDP |
| DHEN for CVR [18] | 2025 | DHEN in production ad conversion-rate prediction | production practice paper |
| AutoML [21] | 2023 | Neural architecture search over large-capacity ranking modules | (see paper) |
| Wukong [19] | 2024 | Stacked Factorization Machine Blocks + Linear Compress Blocks; layer i covers orders 1..2^i | scaling law held over two orders of magnitude, beyond 100 GFLOP per example |
| InterFormer [20] | 2024-25 | Interleaved Interaction Arch and Sequence Arch with bidirectional flow; Cross Arch summarizes | +0.15% NE, +24% QPS internal; +0.6% topline in pilot ads launches |

The direction is clear. Interaction order rises, sequence features and non-sequence features
interact in both directions, and architectures are chosen because quality keeps improving as
compute grows. Those properties make GEM possible (Section 6).

### 4.3 Meta Lattice (2023 blog, 2025 paper)

**Problem.** Ads ranking had split into many models across surfaces (Feed, Stories, Reels,
Audience Network), objectives (clicks and many conversion types) and funnel stages. That
slowed iteration and wasted compute [22].

**Design.** Lattice is a **multi-domain, multi-task** architecture with sparse activation. It
learns from many data sources and handles feedback delays from seconds to days. Meta's 2023
description cites **trillions of parameters** trained on **hundreds of billions of
examples** [22]. Thousands of domains and tens of objectives cannot be balanced by hand, so
Meta uses Pareto-optimality methods, including Pareto-front feature selection and
MetaBalance [22][24]. MetaBalance rescales each auxiliary task's gradient, per part of the
network, toward the magnitude of the target task's gradient, so auxiliary tasks neither
dominate nor vanish [24].

The 2025 paper, "Model Space Redesign for Cost-Effective Industry-Scale Ads
Recommendations", describes four moves: cross-domain knowledge sharing, data consolidation,
model unification, and distillation of the large consolidated model into lean user-facing
models. Meta reports +10% on revenue-driving top-line metrics, +11.5% user satisfaction, +6%
conversion rate and a 20% capacity saving [23].

```
 before Lattice                          after Lattice
 ---------------------------------       -------------------------------------------
 model(FB Feed, CTR)                     +---------------------------------------+
 model(FB Feed, purchase CVR)            | Lattice: one multi-domain multi-task  |
 model(IG Reels, CTR)          ==>       | model; domain-aware sparse experts;   |
 model(IG Stories, install)              | heads per objective; Pareto balancing |
 ... (N surfaces x M objectives)         +-------------------+-------------------+
                                                             | distillation
                                              lean serving models per stage/surface
```

**Lesson.** Consolidation pays when you can (a) balance losses across many tasks
automatically and (b) distil the big model into cheap serving models. Without (b), one
consolidated model would be too expensive to serve on every surface.

---

## 5. Andromeda: deep ads retrieval on GPUs (December 2024)

### 5.1 Problem

Generative-AI creative tools sharply increased the number of eligible ads. Meta said more
than a million advertisers had created more than 15 million ads with its GenAI tools in a
single month [25]. Retrieval had been limited to lightweight models (two-tower plus ANN, plus
heuristics). That limits how much user-ad interaction the model can capture, and the effect
is worst at the stage that has to look at the largest pool.

### 5.2 Architecture

- A heavily customized **deep neural network with sub-linear inference cost**, run with heavy
  parallelism on **NVIDIA Grace Hopper Superchips**. Meta's **MTIA** accelerator is also part
  of the hardware plan [25].
- Ads are organized in a **hierarchical index with multiple layers**, so inference only
  expands the most promising nodes.
- The **index and the retrieval model are trained jointly**. Meta says this gives better
  precision and recall than two-tower or ANN approaches [25].

```
 user + context features
        |
        v
 +---------------------------------------------------------------+
 | Andromeda retrieval DNN on Grace Hopper (unified CPU-GPU mem)  |
 |   level 0:  root clusters   --score--> keep top nodes          |
 |   level 1:  sub-clusters    --score--> keep top nodes          |
 |   level L:  individual ads  --score--> top ~10^3 ads           |
 |   (hierarchy learned jointly with the scoring network)         |
 +------------------------------+--------------------------------+
                                v
          ranking (Lattice / GEM-distilled models) --> auction --> feed insertion
```

### 5.3 Results and scale

- **+6% recall** for the retrieval system and **+8% ads quality** on selected segments, on
  Facebook and Instagram [25].
- Secondary write-ups cite about a **10,000x** increase in retrieval model complexity, and a
  narrowing from **tens of millions** of eligible ads to a **few thousand** candidates per
  request. Treat these as approximate [26].

### 5.4 Lessons

Co-designing model, index and hardware makes heavy computation affordable at the stage that
sees the most items, which moves the cascade's quality bottleneck upstream. When the dot
product becomes the ceiling, a learned hierarchical index is an alternative to ANN over a
factorized score.

---

## 6. GEM: the Generative Ads Recommendation Model (2025-2026)

### 6.1 What it is

In November 2025 Meta described GEM as "the largest foundation model for recommendation
systems (RecSys) in the industry, trained at the scale of large language models", trained on
thousands of GPUs with an LLM-inspired paradigm [27]. Meta frames the work in three pillars:
**scaling the model architecture**, **post-training knowledge transfer** to downstream
models, and **training scalability** [27]. Secondary coverage connects the architecture to
InterFormer-style sequence and non-sequence interaction and Wukong-style stacked
factorization [29]. The InterFormer and Wukong papers above are the published ingredients.

### 6.2 Role in the stack: central teacher, not online scorer

GEM does **not** score every candidate online. Its learnings are propagated "across the
entire ads model fleet" through distillation and representation transfer into the vertical
models that run in each funnel stage [27].

```
            +------------------------------------------------+
            | GEM (offline, LLM-scale, thousands of GPUs)    |
            | long user ad/organic sequences + non-sequence  |
            | features across surfaces and objectives         |
            +-----------------------+------------------------+
                                    | distillation / representation transfer
         +--------------------------+---------------------------+
         v                          v                           v
  retrieval models           light rankers               heavy rankers
  (Andromeda)                (per surface)               (Lattice family)
         \__________________________|___________________________/
                                    v
                       auction + allocation into feeds
```

### 6.3 Results and training efficiency

- **+5% ad conversions on Instagram and +3% on Facebook Feed** in Q2 2025 [27].
- An August 2026 follow-up reports **doubling end-to-end training efficiency to 20-25% model
  FLOPs utilization** while scaling training FLOPs **4x in twelve months**, through co-design
  of kernels, numeric precision, parallelism, networking and memory [28].

### 6.4 Lessons

When the best model is too expensive to serve, keep it offline as a teacher. You pay for it
once in training and collect the gains many times through student quality. This is the same
pattern as LLM distillation, applied fleet-wide. It also separates two release cycles: the
foundation model can be scaled without changing serving latency.

---

## 7. HSTU and generative recommenders (2024-2026)

### 7.1 Problem

Meta wanted to know whether recommendation quality scales with compute the way language
modeling does, and what architecture would make that possible on high-cardinality,
non-stationary streaming data with billions of users [30]. Standard transformers fit
poorly. Softmax normalizes away the intensity of user interest, and FFN-heavy blocks use too
much activation memory at industrial batch sizes [33].

### 7.2 Reformulation

"Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative
Recommendations" (Zhai et al., ICML 2024) interleaves content tokens and action tokens into
one chronological stream per user [30][33]:

- **Ranking:** p(a_i | Phi_0, a_0, ..., Phi_i). This is target-aware, because the candidate
  is in the context.
- **Retrieval:** p(Phi_i | Phi_0, a_0, ..., a_{i-1}), next-item generation.

With a causal mask, one forward pass yields a supervision signal at every position. For a
user with about 500 interactions, secondary analysis estimates roughly a 500x reduction in
training forward passes compared with impression-level DLRM training [33].

### 7.3 The HSTU block

```
   X --> Linear --> SiLU --> split --> U, V, Q, K
                                       |
          SiLU(Q K^T + rab(position, time)) * V     <- no softmax; relative pos+time bias
                                       |
                        LayerNorm(.) * U            <- gating replaces the wide FFN
                                       |
                             Linear --> + X (residual)
```

Secondary notes report per-layer activation memory falling from about 33x the hidden
dimension (standard transformer) to about 14x [33].

### 7.4 Efficiency tricks

- **Stochastic Length.** Long histories are sparsely subsampled during training (80%+
  sparsity with almost no quality loss, per secondary notes) [33].
- **Kernels.** HSTU runs 5.3x to 15.2x faster than FlashAttention2-based transformers at
  sequence length 8,192 [30]. The open-source repo adds Triton kernels, an FA3-based HSTU
  attention kernel for H100 and row-wise FP8 GEMMs [31].
- **M-FALCON (inference).** Many candidates are appended to the user sequence with a mask, so
  each candidate attends only to the history. They are processed in micro-batches that share
  the history's KV cache, which makes target-aware ranking roughly linear in the number of
  candidates [30][33].

### 7.5 Scale and results

- **1.5-trillion-parameter** models deployed on multiple surfaces of a billion-user platform,
  with **+12.4%** on online A/B metrics [30].
- Quality follows a power law of training compute across three orders of magnitude, up to
  roughly GPT-3/LLaMa-2 training compute [30][32].
- Up to 65.8% NDCG improvement over baselines on public and synthetic data. The repo's
  reproduction reports HR@10 0.3556 vs 0.2889 for SASRec on ML-20M [30][31].

### 7.6 Follow-ups

- **DLRM-v3** (an HSTU-based ranker with an MLCommons LoadGen inference harness) and
  **HSTU Ultra** (semi-local and target-aware attention, FP8 projections) in the
  `generative-recommenders` repo [31].
- **ULTRA-HSTU**: "Bending the Scaling Law Curve in Large-Scale Recommendation Systems"
  (2026) [34], and a study of how well generative recommenders generalize [35].
- **LIGER** (2024) combines semantic-ID generative retrieval with dense retrieval [36].

### 7.7 Lessons

Most of the gain comes from restructuring training (one causal sequence per user), not only
from adding parameters. Softmax is not required: unnormalized SiLU attention with a relative
time bias works better for recommendation. And a generative ranker needs its own inference
algorithm (M-FALCON) to be affordable.

---

## 8. Comparing Meta's systems

| System | Year | Stage | Scale (public) | Core technique | Reported impact |
|---|---|---|---|---|---|
| FB Feed [1][4] | 2021 | full funnel | >2B people; >1,000 -> ~500 posts | multipass, multitask NNs, survey-weighted value | n/a |
| IG Explore [5] | 2019 | full funnel | 65B features/s, 90M predictions/s; 500->150->50->25 | ig2vec KNN, distillation | n/a |
| IG Explore [7][8] | 2023 | full funnel | <=1,500 -> 100 -> final | two-tower retrieval + 1st stage, MTML, BO-tuned value | n/a |
| DLRM [9] | 2019 | ranking | reference model | sparse embeddings + MLPs + pairwise dot | industry standard |
| Neo/ZionEX [12] | 2021-22 | training | up to 12T params | 4D sharding, fused kernels, RoCE | up to ~40x (approx.) |
| Lattice [22][23] | 2023-25 | ads ranking | trillions of params | multi-domain multi-task + distillation | +10% topline, -20% capacity |
| Andromeda [25] | 2024 | ads retrieval | 10^7 ads -> ~10^3 (approx.) | joint DNN + hierarchical index on GH200/MTIA | +6% recall, +8% quality |
| HSTU [30] | 2024 | retrieval + ranking | 1.5T params | sequential transduction, M-FALCON | +12.4% online |
| GEM [27][28] | 2025-26 | fleet teacher | thousands of GPUs; MFU 20-25% | LLM-scale foundation model + distillation | +5% IG, +3% FB conversions |

---

## 9. Surfaces and systems not verified in this run

The chapter brief also covers Instagram Reels and Feed ranking, Threads, Facebook
Marketplace, "Scaling User Modeling" and MTIA. Live sources could not be retrieved for these,
so only verified, cross-referenced facts are given:

- **Per-surface system cards.** Meta's Transparency Center publishes AI system cards per
  surface. The Facebook Feed and Instagram Explore cards were verified [4][8]. The same
  Center is the primary public source for the other surfaces' ranking stages and signals,
  including Reels, Stories, Threads and Marketplace. Readers should consult it directly.
- **Reels and Instagram Feed.** The published patterns (multi-source retrieval, two-tower
  first stage, MTML second stage, value-model fusion, integrity-first filtering) are
  documented for Explore and Feed [1][7][8]. Reels-specific numbers were not verified here.
- **Threads and Marketplace.** No architecture details were verified in this run. On the
  retrieval side, Marketplace's product search sits next to the embedding-retrieval lineage
  of Facebook Search [37].
- **Scaling User Modeling (SUM).** Meta has a paper titled "Scaling User Modeling:
  Large-scale Online User Representations for Ads Personalization in Meta" on precomputed
  user representations shared across many downstream ads models. Its contents and URL were
  not re-verified here [41]. The idea of a shared upstream user model feeding many vertical
  models appears in the verified GEM post [27].
- **MTIA.** The Meta Training and Inference Accelerator is named as part of Andromeda's
  hardware plan [25]. Chip specifications were not verified in this run.

---

## 10. Key takeaways for system designers

1. **Run integrity and eligibility first, contextual rules last.** Feed's pass structure
   spends heavy compute only on showable content and keeps diversity as a separate pass [1].
2. **Separate prediction from policy.** Multitask models predict calibrated event
   probabilities. A small value function, tuned by surveys or Bayesian optimization with
   guardrails, encodes the trade-off [1][6][7].
3. **Choose the value-model form deliberately.** Linear sums are interpretable. Log-linear
   products let negative feedback veto an item [6][7].
4. **Optimize for real value, not only engagement.** MSI's history shows that heavy weights
   on reactive engagement can amplify harmful content. Survey-predicted objectives are a
   counterweight [2][3].
5. **Distil downward through the funnel.** Train cheap stages to imitate expensive ones
   (Explore 2019 and 2023), and distil big consolidated models into lean serving models
   (Lattice, GEM) [5][7][23][27].
6. **Cache and precompute aggressively.** Daily item towers, cacheable first-stage models
   and off-peak heavy scoring are what allowed heavier models at every Explore stage [7].
7. **Budget by embeddings and bytes, not FLOPs.** For DLRM-style models, embedding capacity,
   all-to-all bandwidth and data ingestion set the cost [11][12][15].
8. **Consolidate the model fleet when loss balancing is automated.** Lattice's Pareto
   methods and MetaBalance make thousands of domain-objective pairs workable [22][24].
9. **Push model quality upstream.** Andromeda shows that retrieval can carry deep models when
   index, model and hardware are co-designed [25].
10. **Keep the largest model offline.** GEM is an LLM-scale teacher, and the serving fleet
    inherits its gains through distillation and representation transfer [27][28].
11. **Restructure training before scaling it.** HSTU's one-sequence-per-user causal
    formulation gives large efficiency gains and a compute scaling law [30].
12. **Design inference with the model.** M-FALCON's candidate-isolated attention with a
    shared KV cache is what makes target-aware generative ranking affordable [30].

---

## References

1. Lada, A., Wang, M., Yan, T. "How machine learning powers Facebook's News Feed ranking algorithm." Meta, Engineering at Meta, 2021. https://engineering.fb.com/2021/01/26/ml-applications/news-feed-ranking/
2. Meta. "Incorporating More Feedback Into News Feed Ranking." Meta Newsroom, 2021. https://about.fb.com/news/2021/04/incorporating-more-feedback-into-news-feed-ranking/
3. CNN Business (syndicated). "Likes, anger emojis and RSVPs: the math behind Facebook's News Feed and how it backfired." 2021. https://abc17news.com/news/2021/10/27/likes-anger-emojis-and-rsvps-the-math-behind-facebooks-news-feed-and-how-it-backfired/
4. Meta Transparency Center. "Facebook Feed AI system" (system card). Meta, accessed 2026. https://transparency.meta.com/features/explaining-ranking/fb-feed
5. Medvedev, I., Wu, H., Gordon, T. "Powered by AI: Instagram's Explore recommender system." Instagram Engineering / Facebook AI, 2019. https://instagram-engineering.com/powered-by-ai-instagrams-explore-recommender-system-7ca901d2a882
6. Meta / Instagram. "How Instagram suggests new content" (designing a constrained exploration system). Engineering at Meta, 2020. https://engineering.fb.com/2020/12/10/web/how-instagram-suggests-new-content/
7. Vorotilov, V., Shugaepov, I. "Scaling the Instagram Explore recommendations system." Meta, Engineering at Meta, 2023. https://engineering.fb.com/2023/08/09/ml-applications/scaling-instagram-explore-recommendations-system/
8. Meta Transparency Center. "Instagram Explore AI system" (system card). Meta, accessed 2026. https://transparency.meta.com/features/explaining-ranking/ig-explore/
9. Naumov, M., et al. "Deep Learning Recommendation Model for Personalization and Recommendation Systems." Facebook, 2019. https://arxiv.org/abs/1906.00091
10. Meta Research. "DLRM: Deep Learning Recommendation Model" (GitHub repository). Meta, accessed 2026. https://github.com/facebookresearch/dlrm
11. Acun, B., et al. "Understanding Training Efficiency of Deep Learning Recommendation Models at Scale." Meta, HPCA 2021. https://arxiv.org/abs/2011.05497
12. Mudigere, D., et al. "Software-Hardware Co-design for Fast and Scalable Training of Deep Learning Recommendation Models" (Neo / ZionEX). Meta, ISCA 2022. https://arxiv.org/abs/2104.05158
13. PyTorch / Meta. "TorchRec" (GitHub repository). Accessed 2026. https://github.com/pytorch/torchrec
14. PyTorch / Meta. "FBGEMM / FBGEMM_GPU" (GitHub repository). Accessed 2026. https://github.com/pytorch/FBGEMM
15. Zhao, M., et al. "Understanding Data Storage and Ingestion for Large-Scale Deep Recommendation Model Training." Meta, ISCA 2022. https://arxiv.org/abs/2108.09373
16. He, X., Pan, J., Jin, O., et al. "Practical Lessons from Predicting Clicks on Ads at Facebook." Facebook, ADKDD 2014. https://quinonero.net/Publications/predicting-clicks-facebook.pdf
17. Zhang, B., Luo, L., Liu, X., et al. "DHEN: A Deep and Hierarchical Ensemble Network for Large-Scale Click-Through Rate Prediction." Meta, 2022. https://arxiv.org/abs/2203.11014
18. Meta. "On the Practice of Deep Hierarchical Ensemble Network for Ad Conversion Rate Prediction." Meta, 2025. https://arxiv.org/abs/2504.08169
19. Zhang, B., Luo, L., Chen, Y., et al. "Wukong: Towards a Scaling Law for Large-Scale Recommendation." Meta, ICML 2024. https://arxiv.org/abs/2403.02545
20. Zeng, Z., et al. "InterFormer: Effective Heterogeneous Interaction Learning for Click-Through Rate Prediction." Meta / UIUC, 2024-2025. https://arxiv.org/abs/2411.09852
21. Meta. "AutoML for Large Capacity Modeling of Meta's Ranking Systems." Meta, 2023. https://arxiv.org/abs/2311.07870
22. Meta AI. "Meta Lattice: AI ads performance and efficiency." Meta, 2023. https://ai.meta.com/blog/ai-ads-performance-efficiency-meta-lattice/
23. Meta. "Meta Lattice: Model Space Redesign for Cost-Effective Industry-Scale Ads Recommendations." Meta, 2025. https://arxiv.org/abs/2512.09200
24. He, Y., Feng, X., Cheng, C., Ji, G., Guo, Y., Caverlee, J. "MetaBalance: Improving Multi-Task Recommendations via Adapting Gradient Magnitudes of Auxiliary Tasks." Meta AI / TAMU, WWW 2022. https://arxiv.org/abs/2203.06801
25. Meta Engineering. "Meta Andromeda: Supercharging Advantage+ automation with the next-gen personalized ads retrieval engine." Meta, December 2024. https://engineering.fb.com/2024/12/02/production-engineering/meta-andromeda-advantage-automation-next-gen-personalized-ads-retrieval-engine/
26. Loomer, J. "Meta Andromeda: What It Means for Your Ad Strategy" (secondary summary of [25]). 2025. https://www.jonloomer.com/meta-andromeda/
27. Engineering at Meta. "Meta's Generative Ads Model (GEM): The central brain accelerating ads recommendation AI innovation." Meta, 10 November 2025. https://engineering.fb.com/2025/11/10/ml-applications/metas-generative-ads-model-gem-the-central-brain-accelerating-ads-recommendation-ai-innovation/
28. Engineering at Meta. "GEM Training: How Meta Doubled the Efficiency of Its LLM-Scale Ads Foundation Model." Meta, August 2026. https://engineering.fb.com/2026/08/03/ml-applications/training-gem-at-llm-scale-meta-ads-recommendation-foundation-model/
29. ByteByteGo. "How Meta Built a New AI-Powered Ads Model for 5% Better Conversions" (secondary). 2025. https://blog.bytebytego.com/p/how-meta-built-a-new-ai-powered-ads
30. Zhai, J., et al. "Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations." Meta, ICML 2024. https://arxiv.org/abs/2402.17152
31. meta-recsys. "generative-recommenders" (HSTU, M-FALCON, DLRM-v3, HSTU Ultra code). Meta, 2024-2026. https://github.com/meta-recsys/generative-recommenders
32. Zhai, J., et al. ICML 2024 slides, "Actions Speak Louder than Words." Meta, 2024. https://icml.cc/media/icml-2024/Slides/32684.pdf
33. Xing, H. "HSTU: The Cornerstone of Scaling Laws" (fun-rec-mdbook technical notes, secondary). 2026. https://github.com/Haozhe-Xing/fun-rec-mdbook/blob/main/src/en/part7-scaling/hstu.md
34. Meta. "Bending the Scaling Law Curve in Large-Scale Recommendation Systems" (ULTRA-HSTU). Meta, 2026. https://arxiv.org/abs/2602.16986
35. Meta. "How Well Does Generative Recommendation Generalize?" Meta, 2026. https://arxiv.org/abs/2603.19809
36. Meta. "Unifying Generative and Dense Retrieval for Sequential Recommendation" (LIGER). Meta, 2024. https://arxiv.org/abs/2411.18814
37. Huang, J.-T., et al. "Embedding-based Retrieval in Facebook Search." Facebook, KDD 2020. https://arxiv.org/abs/2006.11632
38. Lerer, A., Wu, L., Shen, J., Lacroix, T., Wehrstedt, L., Bose, A., Peysakhovich, A. "PyTorch-BigGraph: A Large-scale Graph Embedding System." Facebook AI, SysML 2019. https://mlsys.org/Conferences/2019/doc/2019/71.pdf
39. Meta Engineering. "Faiss: A library for efficient similarity search." Facebook, 2017. https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/
40. Gauci, J., Conti, E., Liang, Y., et al. "Horizon: Facebook's Open Source Applied Reinforcement Learning Platform." Facebook, 2018. https://arxiv.org/abs/1811.00260
41. Meta. "Scaling User Modeling: Large-scale Online User Representations for Ads Personalization in Meta." Meta, 2023-2024 (title cited from prior knowledge; URL not verified in this run, so none is given).
