# 03. The Evolution of Ranking and CTR Models: From Logistic Regression to Scaling-Law Architectures

## Overview

The ranking stage is where most of a recommender's modeling budget is spent. Retrieval
narrows millions of items to hundreds or a few thousand candidates; the ranker then scores
every candidate with a rich model that predicts click-through rate (CTR), conversion rate
(CVR), or a vector of engagement probabilities. Because ranking quality converts almost
directly into revenue in ads and into engagement in feeds, this is the most published-on
component of industrial recommendation.

The history of ranking models is mostly a history of one design question: how should a
model learn interactions between sparse, high-cardinality categorical features (user id,
item id, category, query, context) and a smaller set of dense features? The answers fall
into roughly five eras:

1. Linear models with hand-crafted crosses, trained online at huge scale
   (Google FTRL-Proximal, 2013 [1]).
2. Automated non-linear feature transforms and factorized interactions
   (Facebook GBDT+LR, 2014 [3]; Factorization Machines [4]; FFM [6]).
3. Hybrid "explicit interaction + deep MLP" networks (Wide & Deep [8], DeepFM [9],
   DCN [10], xDeepFM [13], AutoInt [14], DLRM [15]).
4. User-behavior-sequence modeling with target attention (Alibaba DIN [17], DIEN [18],
   SIM [19], Kuaishou TWIN [20]).
5. Scaling-oriented architectures that treat ranking like a foundation model (DCN-V2 [11],
   DHEN [21], Wukong [22], HSTU [23], InterFormer [24], RankMixer [30], Meta GEM [25]).

This chapter walks through each design with its problem statement, architecture, scale
numbers where public, key tricks, and lessons, then closes with a comparison table and
takeaways.

```
Evolution of CTR / ranking architectures (simplified)

2010  FM (Rendle) ---------------------------------------+
2013  LR + FTRL-Proximal (Google ads)                    |
2014  GBDT -> LR (Facebook ads)      FFM (Kaggle/Criteo) |
2016  Wide & Deep (Google Play)                          |
2017  DeepFM (Huawei)  DCN (Google)  DIN (Alibaba) <-----+ factorized / attention
2018  xDeepFM (CIN)  AutoInt (self-attn)  DIEN (AUGRU)
2019  DLRM (Facebook, open source)
2020  DCN-V2 (Google)  SIM (Alibaba, 54k-length sequences)
2021  MaskNet (Weibo) -> Twitter heavy ranker (2023)
2022  DHEN (Meta)
2023  TWIN (Kuaishou)
2024  Wukong (Meta)  HSTU (Meta)  LiRank (LinkedIn)  InterFormer (Meta)
2025  RankMixer (ByteDance)  GEM (Meta ads foundation model)
2026  TokenMixer-Large (ByteDance)  ML-DCN (Pinterest)  GEM training at LLM scale
```

## 1. The Linear Era: Logistic Regression and FTRL-Proximal (Google, 2013)

### Problem

Google search ads needed a CTR model that could be trained online on an enormous stream of
impressions, with billions of sparse features, while staying small enough in memory to
serve at very high QPS. Calibration matters as much as ranking, because ad auctions
multiply predicted CTR by bids.

### Architecture and algorithm

McMahan et al.'s KDD 2013 paper "Ad Click Prediction: a View from the Trenches" [1]
describes a logistic-regression model trained with the FTRL-Proximal online learning
algorithm, chosen for its combination of sparsity (via L1 regularization that produces
exact zeros) and convergence properties, plus per-coordinate learning rates. Per-coordinate
learning rates require tracking the sum of gradients and the sum of squared gradients for
each feature, which improves accuracy at the cost of memory [2].

```
 impression stream --> feature extraction --> hashed / sparse feature ids
                                                   |
                                                   v
                        +-------------------------------------------+
                        |  FTRL-Proximal logistic regression        |
                        |  per-coordinate state: z_i, n_i           |
                        |  L1 => exact zeros => sparse model        |
                        +-------------------------------------------+
                                  |                     |
                       weights (q2.13, 16-bit)   calibration layer
                                  |                     |
                                  v                     v
                             serving replicas  -->  pCTR to auction
```

### Memory tricks (the most copied part of the paper)

- Probabilistic feature inclusion. Rare features dominate the feature space but carry
  little signal. Instead of admitting every feature, the paper tests Poisson inclusion
  (admit a new feature with probability p) and Bloom-filter inclusion (admit after the
  feature has been seen n times, using a rolling set of counting Bloom filters). Reported
  results: Poisson p = 0.03 saves 60% RAM at 0.020% AucLoss detriment; p = 0.1 saves 40%
  at 0.006%; Bloom with n = 2 saves 66% at 0.008%, so Bloom filtering gives the better
  trade-off [1][2].
- Reduced-precision weights. Coefficients are stored in a 16-bit fixed-point "q2.13"
  encoding with randomized rounding; the authors report no measurable loss versus 64-bit
  floats, and summaries attribute roughly 75% RAM savings for coefficient storage to the
  combination of tricks [1][2]. The same group's companion paper "Large-Scale Learning
  with Less RAM via Randomization" formalizes the randomized-rounding idea [39].
- Training many model variants simultaneously, sharing feature-index structures, and
  subsampling training data (keeping all clicks but only a fraction of non-click queries,
  with importance weights) are also described [1].

### Negative results

A distinctive contribution is the list of ideas that did not pay off in production despite
good literature results. Feature hashing, for instance, needed too large a table to avoid
damaging collisions [2]. This "what did not work" reporting style became a model for later
industrial papers.

### Lessons

- A linear model can be extremely strong when the feature engineering (crosses) is rich and
  the data fresh; its weakness is that humans must design the crosses.
- Memory, not FLOPs, was the binding constraint. Feature admission and low-precision
  storage remain standard in every modern embedding system (they reappear as embedding
  admission policies, frequency thresholds, and FP16/INT8 embedding tables).
- Calibration is a first-class output, not an afterthought.

## 2. Automating Feature Crosses: GBDT + LR at Facebook (2014)

### Problem

Facebook ads needed better CTR accuracy than a linear model with hand-built crosses, while
still training online on fresh data.

### Architecture

He et al. ("Practical Lessons from Predicting Clicks on Ads at Facebook", ADKDD 2014) [3]
combine gradient-boosted decision trees with logistic regression. Each boosted tree acts as
a learned, non-linear feature transform: the index of the leaf an example lands in becomes a
one-hot categorical feature, and the concatenation of leaf indices across all trees is fed
to a linear classifier.

```
 raw features (dense + categorical)
          |
          v
 +-----------------------------+     trained in batch (e.g. daily)
 |  Boosted trees T1 ... TK    |
 +-----------------------------+
   | leaf id   | leaf id    | leaf id
   v           v            v
 [one-hot(T1) | one-hot(T2) | ... | one-hot(TK)]  + other sparse features
          |
          v
 +-----------------------------+     trained online, near real time
 | Logistic regression (SGD)   |
 +-----------------------------+
          |
          v
     pCTR  --> recalibrate for negative downsampling --> auction
```

### Key findings

- The hybrid beats either trees or LR alone by over 3% in normalized entropy (NE) [3];
  summaries report LR + trees at about 96.6% of the trees-only NE and a 3.4% NE improvement
  from the tree transform [3][40].
- Normalized entropy (log loss divided by the entropy of the background CTR) is introduced
  as the primary metric because raw log loss looks deceptively good when base CTR is near
  0 or 1 [40].
- Feature choice matters most: historical features describing the user's or ad's past
  behavior dominate contextual features. Data freshness, learning-rate schedule, and data
  sampling help only modestly by comparison [3].
- Data freshness: shorter delay between training and serving lowers NE, which motivates
  training the LR part online while retraining trees less frequently [40].
- Negative downsampling speeds training but requires re-calibration: with negative
  sampling rate w and in-sample prediction p, the live-space estimate is
  q = p / (p + (1 - p) / w) [3][40].
- Subsampling to 10% of the data reportedly costs about 1% NE (secondary summary) [40].

### Lessons

GBDT+LR was the first widely-copied "learned feature crossing" design and the template for
splitting a model into a slow-trained feature transformer and a fast-trained head. That
split reappears later as frozen embeddings/towers plus online-updated top layers.

## 3. Factorized Interactions: FM and FFM

### Factorization Machines (Rendle, ICDM 2010)

FM [4] models every pairwise interaction between features i and j as the dot product of
two learned latent vectors, <v_i, v_j>, instead of a free weight w_ij. This lets the model
estimate interactions between feature pairs that never co-occurred in training, which is
essential under extreme sparsity, and it generalizes matrix factorization and several
specialized factorization models through choice of input encoding [4]. The pairwise term
looks quadratic in the number of features n, but can be rewritten as one half of the sum
over latent dimensions of (square of sum minus sum of squares), giving O(k*n) cost where k
is the latent dimension [5][41].

### Field-aware FM (Juan et al., RecSys 2016)

FFM [6] gives each feature a separate latent vector per field it interacts with: the
interaction between feature i (field F_i) and feature j (field F_j) uses
<v_{i,F_j}, v_{j,F_i}>. This increases parameters by roughly the number of fields but lets
the model represent that, for example, "advertiser" interacts differently with "publisher"
than with "gender". FFM-based solutions won two Kaggle CTR competitions in 2014, including
the Criteo display-ad challenge [6]. Criteo later reported running FFM in a production
display-ads system for CTR and conversion prediction, adding a seeding algorithm and a
distributed learning mechanism to cut training time [7].

### Lessons

- FM introduced the idea that would dominate the next decade: every feature gets an
  embedding, and interactions are computed from embeddings rather than memorized as cross
  weights.
- FFM showed that interaction structure should respect field semantics, at a large
  parameter cost. Later designs (FwFM, DCN-V2's full-rank matrix, field-specific
  token mixing in RankMixer) are all attempts to get field awareness more cheaply.

## 4. Hybrid Deep Models: Memorization plus Generalization

### 4.1 Wide & Deep (Google Play, 2016)

Problem: Google Play app recommendations, with over one billion active users and over one
million apps [8]. Linear models with cross-product features memorize well but generalize
poorly to unseen combinations; pure embedding models generalize but over-generalize for
niche users.

Architecture: a jointly trained wide linear part (raw sparse features plus cross-product
transformations such as AND(user_installed_app, impression_app)) and a deep part
(categorical features mapped to dense embeddings, concatenated with dense features, passed
through a ReLU MLP). Their logits are summed and passed through a sigmoid. The wide part is
trained with FTRL with L1 regularization; the deep part with AdaGrad [8].

```
      wide (memorization)                     deep (generalization)
 sparse ids + cross products         sparse ids -> embeddings (~32-d each)
            |                                  + dense features
            |                                     |
            |                               MLP (ReLU layers)
            |                                     |
            +----------- sum of logits -----------+
                               |
                            sigmoid -> P(app acquisition)
```

Scale and serving: models were trained on over 500 billion examples, with warm-starting
from the previous model so that each retrain does not start from scratch. Scoring all
candidates in a single-threaded batch took 31 ms; splitting into smaller batches across
threads brought client-side latency to 14 ms (batch 50, 4 threads), against a serving
target on the order of 10 ms [8][42]. A three-week live A/B test showed +3.9% app
acquisitions over a wide-only control and about +1% over deep-only [8][42].

Lesson: joint training of a memorization path and a generalization path became the default
template. Its weakness is that the wide part still requires hand-designed crosses.

### 4.2 DeepFM (Huawei, 2017)

DeepFM [9] replaces the hand-crafted wide part with an FM component, and crucially shares
the same embedding layer between the FM and the DNN. Because both parts take the same raw
input, no feature engineering beyond raw features is needed. It reported AUC and logloss
improvements over Wide & Deep variants on Criteo and on a commercial (Huawei) app-store
dataset, with training efficiency comparable to the fastest deep baselines [9].

### 4.3 Deep & Cross Network (Google, 2017)

DCN [10] introduces a cross network in which each layer explicitly multiplies the
original input x0 with the current layer's output, so the maximum polynomial degree of
interaction grows by one with each layer. In the original formulation, each cross layer
used a weight vector, making it very cheap (parameters linear in input width). The paper
reports lower logloss than a plain DNN with far fewer parameters and no manual feature
engineering [10].

### 4.4 xDeepFM and the Compressed Interaction Network (Microsoft, 2018)

xDeepFM [13] argues that DCN's crosses operate "bit-wise" (on individual embedding
elements) and proposes the Compressed Interaction Network (CIN), which generates
interactions "vector-wise" (treating each field embedding as a unit). Each CIN layer takes
outer products between the previous hidden feature maps and the base field embeddings,
then compresses them with convolution-like filters, so interaction degree grows with
depth. xDeepFM combines a linear part, CIN, and a DNN. A commonly noted downside is CIN's
time complexity, about an order of magnitude higher than a plain DNN (secondary analysis)
[13][43], which limited its adoption in latency-constrained production systems.

### 4.5 AutoInt (2018)

AutoInt [14] maps all numerical and categorical features into a shared low-dimensional
space and applies stacked multi-head self-attention with residual connections over the
field embeddings. Each attention layer models interactions among fields; stacking layers
increases the order. A side benefit is interpretability: attention weights show which
field pairs interact [14]. AutoInt was an early indicator of the later move toward
Transformer-style token mixing over feature "tokens".

### 4.6 DLRM (Facebook/Meta, 2019)

Problem: provide a reference architecture and benchmark that reflects production
personalization models, which are dominated by huge embedding tables rather than dense
compute.

Architecture [15][16]:

```
 dense features --> bottom MLP --> d-dim vector ----+
                                                    |
 sparse feat 1 --> embedding table 1 --> d-dim -----+
 sparse feat 2 --> embedding table 2 --> d-dim -----+--> pairwise dot products
 ...                                                |    (all pairs of vectors)
 sparse feat N --> embedding table N --> d-dim -----+          |
                                                               v
                               concat(dot products, bottom-MLP output)
                                                               |
                                                           top MLP
                                                               |
                                                           sigmoid
```

- The bottom MLP projects dense features to the same dimension as embeddings so that all
  vectors can be dotted together [16][44].
- Interactions are deliberately limited to second order (dot products), in the spirit of
  FM, which keeps the interaction cheap and dimensionality small; the authors contrast this
  with DCN-style higher-order crossing [16].
- Parallelism: embedding tables are model-parallel (sharded across devices because they do
  not fit on one), while the MLPs are data-parallel. Moving from the model-parallel
  embedding lookup to the data-parallel interaction and top MLP requires a personalized
  all-to-all communication [16][44].
- Released open source in July 2019 with PyTorch and Caffe2 implementations [15]; the
  repository later added MLPerf benchmark support and TorchRec integration [15].

Lesson: DLRM made explicit that industrial recommenders are memory- and
communication-bound, and established the "embeddings + bottom MLP + interaction + top MLP"
skeleton that DHEN, Wukong and many internal models extend.

### 4.7 DCN-V2 (Google, 2020)

Problem: the original DCN's vector-parameterized cross layer was too weak to learn
predictive crosses at web scale [11].

Architecture: the cross layer becomes x_{l+1} = x0 * (W_l x_l + b_l) + x_l, where * is
element-wise product and W_l is a full matrix. This makes the cross network much more
expressive while retaining bounded-degree, explicit crosses. Cross and deep networks can be
combined either stacked (cross output feeds the DNN) or in parallel (outputs concatenated)
[11][12].

Efficiency trick: the authors observed that learned W matrices are effectively low-rank,
so they factor W = U V^T with rank r much smaller than d, reducing per-layer cost from
O(d^2) to roughly O(2dr), and further use a mixture of low-rank experts with an
input-dependent gate (often called DCN-Mix) [11][45]. The paper reports state-of-the-art
results on public benchmarks and significant offline accuracy and online business-metric
gains across many web-scale learning-to-rank systems at Google; it also stresses that
production must respect extremely high QPS and strict latency [11].

Industrial echoes:
- TensorFlow Recommenders ships DCN as a first-class layer with a tutorial [12].
- LinkedIn's LiRank (KDD 2024) extends DCNv2 with attention and residual connections
  ("Residual DCN"), uses low-rank matrices, and reports nearly 30% parameter reduction; it
  also introduces an isotonic calibration layer co-trained inside the deep model, used
  across Feed, Jobs and Ads CTR [29].
- Pinterest's ads models used DCNv2 as the production baseline when adding Transformer
  user-sequence features, and reported DCNv2 cross-layer overflow issues under float16
  training [32]. In 2026 Pinterest reported that naively scaling DCNv2 and MaskNet gave
  rapidly diminishing offline gains, and proposed ML-DCN (an instance-conditioned mask on a
  low-rank crossing layer), deployed with neutral serving cost [34].

### 4.8 MaskNet and the Twitter heavy ranker

MaskNet (Sina Weibo, 2021) [27] argues plain feed-forward layers are inefficient at
learning multiplicative crosses and introduces the instance-guided mask: an element-wise
product applied to embeddings and hidden layers, with mask values computed from the input
instance. A MaskBlock combines layer normalization, the mask, and a feed-forward layer;
analysis on Criteo shows over half of mask values near zero, acting like bit-wise
attention [27]. When Twitter open-sourced its recommendation algorithm in 2023, its heavy
ranker was a roughly 48M-parameter neural network, continuously trained, that predicts
about ten engagement probabilities (likes, retweets, replies, and so on) on about 1,500
candidates per request [28]; secondary analyses of the released code identify it as a
parallel MaskNet [28].

## 5. User-Behavior Modeling: The Alibaba Line (DIN, DIEN, SIM) and Kuaishou TWIN

Feature-interaction networks treat a user's history as a bag of features pooled into a
fixed vector. In e-commerce display ads, that loses information: a user's interest in
running shoes is irrelevant when scoring a kitchen appliance. Alibaba's line of work makes
the user representation a function of the candidate item.

### 5.1 DIN: Deep Interest Network (KDD 2018)

Architecture: a local activation unit computes, for each historical behavior, a relevance
weight with respect to the candidate ad, then takes a weighted sum pooling of behavior
embeddings. Behaviors related to the candidate dominate the representation [17]. The
weights are not normalized by softmax, so the sum also reflects the intensity of interest
(a point emphasized in the paper's design discussion [17]).

```
 candidate ad emb ------------------------------+
                                                |
 behavior_1 emb --[act unit(b1, ad)]-- w1 --+   |
 behavior_2 emb --[act unit(b2, ad)]-- w2 --+-> sum_i w_i * b_i --> user interest
 ...                                        |                         |
 behavior_T emb --[act unit(bT, ad)]-- wT --+                         |
                                                                       v
           concat(user interest, user profile, ad, context) --> MLP --> pCTR
```

Training tricks [17]:
- Mini-batch aware regularization: L2 over billions of parameters every mini-batch is
  infeasible, so regularization is applied only to the parameters of features present in
  the mini-batch. It added 0.0031 absolute AUC over dropout on the Alibaba dataset.
- Dice activation: a data-adaptive generalization of PReLU whose rectification point
  depends on input distribution; +0.0015 AUC over PReLU.

Scale and results: the production dataset has over 2 billion samples; an online A/B test
(May to June 2017) showed up to +10.0% CTR and +3.8% RPM over the base model; DIN was
deployed to serve the main traffic of Alibaba display advertising [17].

### 5.2 DIEN: Deep Interest Evolution Network (AAAI 2019)

DIEN [18] adds time. An interest-extractor layer (GRU) turns the behavior sequence into
latent interest states, with an auxiliary loss that supervises each hidden state with the
next actual behavior so states carry real interest signal. An interest-evolving layer uses
AUGRU, a GRU whose update gate is scaled by attention to the target item, so relevant
interests evolve while irrelevant ones (interest drift) are suppressed [18].

Results and serving: deployed in Taobao display advertising with a reported 20.7% CTR
improvement. Serving a GRU per candidate is expensive; the paper reports reducing latency
from 38.2 ms to 6.6 ms using element-parallel GRU with kernel fusion, batching adjacent
requests from different users on GPU, and model compression, with the system serving more
than one million users per second at peak [18][46].

### 5.3 SIM: Search-based Interest Model (CIKM 2020)

Problem: Alibaba's earlier MIMN could handle about 1,000 behaviors, but accuracy degraded
as sequences grew 10x or more [19]. Lifelong histories carry valuable long-term signal but
cannot be attended over in full within latency.

Architecture: a two-stage cascade [19][47]:

```
 lifelong behaviors (up to ~54,000)
          |
          v
 +------------------------------------------+
 | General Search Unit (GSU)                |
 |   hard search: behaviors with same       |
 |     category as candidate (index lookup) |
 |   soft search: embedding MIPS            |
 +------------------------------------------+
          | top-K relevant behaviors (K ~ hundreds)
          v
 +------------------------------------------+
 | Exact Search Unit (ESU)                  |
 |   multi-head target attention            |
 +------------------------------------------+
          |
          v
   concat with short-term DIN/DIEN features --> MLP --> pCTR
```

- Sequence length up to 54,000, about 54x the previous state of the art [19][47].
- Hard search is non-parametric (category match) and was the variant deployed; soft search
  was only slightly better offline. Behaviors are stored in a two-level user behavior tree
  keyed by user and category; a secondary summary reports this index at about 22 TB and
  under 5 ms of added serving latency (approximate, unverified against the paper) [47].
- Deployed in Alibaba display advertising since 2019 with 7.1% CTR and 4.4% RPM lift [19].

### 5.4 TWIN (Kuaishou, KDD 2023)

TWIN [20] identifies a flaw in GSU/ESU cascades: the cheap GSU relevance metric differs
from the ESU's target attention, so the GSU retrieves behaviors the ESU would rank
low and misses ones it would rank high. TWIN's Consistency-Preserved GSU uses the same
target-attention relevance metric as the ESU (hence "twins"). To make this affordable over
tens of thousands of behaviors, item features are split into item-only (cacheable,
precomputed) and user-item cross parts, which the authors say cut the computational
bottleneck by 99.3%. The GSU scans tens of thousands of behaviors and passes the top 100 to
the ESU; offline evaluation used a 46-billion-sample production dataset, and the model is
deployed on Kuaishou's main traffic [20]. TWIN-V2 later compresses lifelong sequences with
hierarchical clustering [20].

### Lessons from the sequence line

- Target attention ("represent the user relative to the item") is the single most durable
  idea from this line; nearly every modern ranker has a DIN-style attention block.
- Long sequences are a retrieval problem inside the ranker; the GSU/ESU split mirrors the
  retrieval/ranking cascade of the whole system.
- Serving engineering (kernel fusion, cross-user batching, precomputed item-side
  projections) is as important as architecture.

## 6. Scaling Era: Ensembles, Stacked FMs, and Foundation Rankers

### 6.1 DHEN: Deep and Hierarchical Ensemble Network (Meta, 2022)

Observation: interaction modules claiming the same interaction order (e.g., DCN cross,
self-attention, FM-style dot products) perform differently on different datasets, so no
single module is best [21]. DHEN stacks layers, each of which is an ensemble of
heterogeneous interaction modules (such as AdvancedDLRM-style dot products, self-attention,
convolution, linear, DCN), with outputs combined and fed to the next layer. Stacking yields
a hierarchy of interactions of increasing order [21].

```
 feature embeddings
       |
 +-----------------------------------------------+
 | Layer 1: [DotProd] [SelfAttn] [DCN] [Linear]  |  -> ensemble combine + residual
 +-----------------------------------------------+
       |
 +-----------------------------------------------+
 | Layer 2: [DotProd] [SelfAttn] [DCN] [Linear]  |
 +-----------------------------------------------+
       |
      ...  -> top MLP -> task heads
```

Systems co-design: the deeper model is harder to train, so the paper introduces Hybrid
Sharded Data Parallel (HSDP), reported at up to 1.2x throughput over fully sharded data
parallel. On a large-scale CTR dataset DHEN improved normalized entropy by 0.27% over an
AdvancedDLRM baseline [21]. Meta later applied DHEN to ad conversion-rate prediction [38]
and built AutoML over DHEN-style module search for its ranking models [37].

### 6.2 Wukong (Meta, ICML 2024)

Problem: unlike LLMs, ranking models had not shown predictable quality gains from scaling
dense compute; upscaling DLRM-like or DCN-like models saturates [22].

Architecture: a stack of identical Wukong layers. Each layer has two parallel blocks whose
outputs are concatenated: a Factorization Machine Block (FMB: an FM-style pairwise
interaction over embeddings, followed by an MLP that emits new embeddings) and a Linear
Compress Block (LCB: linear recombination of embeddings without raising interaction
order). Because each layer interacts the outputs of the previous one, layer i covers
interaction orders from 1 to 2^i, analogous to binary exponentiation [22]. Low-rank FM
reduces cost from O(n^2 d) to O(n k d) (secondary analysis) [22].

Scaling knobs are number of layers, number of embeddings emitted by FMB and LCB, FM rank,
and MLP width. Wukong beat state-of-the-art models on six public datasets, and on an
internal dataset held a scaling law across two orders of magnitude of model complexity,
beyond 100 GFLOP per example, where prior architectures stopped improving [22].

### 6.3 HSTU and Generative Recommenders (Meta, ICML 2024)

HSTU [23] reframes ranking and retrieval as sequential transduction over user action
sequences, using a modified attention block (Hierarchical Sequential Transduction Unit).
It reports 5.3x to 15.2x speedups over FlashAttention2-based Transformers on 8,192-length
sequences, power-law quality scaling over three orders of magnitude of training compute,
and 1.5-trillion-parameter models that improved online A/B metrics by 12.4% on surfaces of
a platform with billions of users [23]. HSTU is covered in depth in the generative
recommendation chapter; here the point is that it merges the feature-interaction line and
the sequence line into one scalable architecture.

### 6.4 InterFormer (Meta, 2024-2025)

InterFormer [24] targets heterogeneous inputs (non-sequential features plus behavior
sequences). It identifies two weaknesses of prior models: one-directional information flow
between modes, and early, aggressive summarization of sequences. It interleaves an
Interaction Arch (behavior-aware feature interaction) and a Sequence Arch (context-aware
sequence modeling) with bidirectional flow, plus a Cross Arch for selective summarization.
It reports up to 0.14% AUC gain on public benchmarks, 0.15% NE and 24% QPS gain on Meta
internal data, and a 0.6% topline gain in pilot ads launches [24].

### 6.5 Meta GEM (2025-2026)

Meta's Generative Ads Recommendation Model (GEM), announced November 2025, is described as
the largest foundation model for recommendation in the industry, trained across thousands
of GPUs on an LLM-inspired paradigm [25]. Its role is not a single online ranker: Meta
describes three pillars (scaling the model architecture, post-training knowledge transfer
to downstream ads models, and training scalability), with GEM's learnings transferred to
the many production ranking models across the ads stack. Reported launch results were +5%
ad conversions on Instagram and +3% on Facebook Feed in Q2 2025 [25]. A follow-up post
(August 2026) reports doubling end-to-end training efficiency to 20-25% model FLOPs
utilization while scaling training FLOPs 4x in twelve months via co-design of kernels,
precision, parallelism, networking and memory [26]. Secondary coverage links GEM's
architecture to InterFormer and Wukong [48].

```
              +-----------------------------------------+
              |  GEM foundation model (offline, huge)   |
              |  sequence + non-sequence interaction    |
              +-----------------------------------------+
                  |  knowledge transfer / distillation
        +---------+-----------+-----------------+
        v                     v                 v
  vertical model A      vertical model B   vertical model C   (served online,
  (e.g. surface/CVR)    (e.g. CTR)         (...)              latency-bounded)
```

Design lesson: when the best model is too expensive to serve, the industry pattern is
"teacher foundation model + many distilled students", the same pattern LLM serving uses.

### 6.6 RankMixer and TokenMixer-Large (ByteDance, 2025-2026)

Problem: industrial rankers had accumulated many handcrafted interaction modules with poor
hardware efficiency. RankMixer [30] reports that such models ran at about 4.5% model FLOPs
utilization (MFU).

Architecture: features are grouped into tokens; each RankMixer block applies multi-head
token mixing (split each token into heads and recombine across tokens, a parameter-free
substitute for quadratic self-attention over heterogeneous feature spaces) and a per-token
FFN with separate parameters per token (feature subspace), optionally extended into a
sparse mixture-of-experts [30].

Results: MFU rose from 4.5% to 45%, ranking-model parameters scaled about 100x at roughly
constant inference latency, and full-traffic deployment in Douyin feed ranking gave +0.2%
active days and +0.5% app duration [30]. The successor TokenMixer-Large (2026) reports
offline scaling to 7B and 15B parameters for Douyin ads and e-commerce [31].

### 6.7 A cautionary thread: embedding collapse and benchmark reality

- Guo et al. (Tsinghua and Tencent, ICML 2024) show that scaling embedding size often fails
  because embedding matrices collapse into low-dimensional subspaces; feature interaction
  both worsens collapse and limits overfitting. Their multi-embedding design (several
  independent embedding sets with set-specific interaction modules) keeps improving up to
  10x embedding size [35].
- The BARS/FuxiCTR open benchmark re-ran 24 CTR models with over 7,000 experiments and
  12,000+ GPU hours, finding that with adequate tuning many deep models differ far less
  than their papers claim [36].
- Pinterest notes that benchmark-leading interaction modules face industry obstacles:
  latency, GPU memory limits, reproducibility [33].

## 7. Comparison Table

| Model (org, year) | Interaction mechanism | Order | Hand-crafted crosses? | Production evidence (public) |
|---|---|---|---|---|
| LR + FTRL (Google, 2013) [1] | Linear over hashed/sparse crosses | Manual | Yes | Google search ads |
| GBDT + LR (Facebook, 2014) [3] | Tree leaves as features, linear head | Tree depth | Partly | FB ads; >3% NE gain vs either alone |
| FM (2010) [4] | Dot product of latent vectors | 2 | No | Basis of later models |
| FFM (2016) [6][7] | Field-aware latent vectors | 2 | No | Kaggle wins; Criteo prod |
| Wide & Deep (Google, 2016) [8] | Linear crosses + MLP | Manual + implicit | Wide part | Play: +3.9% acquisitions |
| DeepFM (Huawei, 2017) [9] | FM + MLP, shared embeddings | 2 + implicit | No | Huawei app store data |
| DCN (Google, 2017) [10] | Vector cross layers + MLP | Bounded, grows w/ depth | No | Criteo |
| xDeepFM (2018) [13] | CIN (vector-wise outer products) | Bounded | No | Costly (approx 10x DNN) |
| AutoInt (2018) [14] | Multi-head self-attention over fields | Grows w/ depth | No | Public datasets |
| DLRM (Meta, 2019) [16] | Pairwise dot products | 2 | No | Reference / MLPerf |
| DIN (Alibaba, 2017-18) [17] | Target attention over behaviors | n/a (sequence) | No | +10.0% CTR, +3.8% RPM |
| DIEN (Alibaba, 2019) [18] | GRU + AUGRU | n/a | No | +20.7% CTR; 6.6 ms |
| DCN-V2 (Google, 2020) [11] | Matrix cross, low-rank MoE | Bounded | No | Many Google LTR systems |
| SIM (Alibaba, 2020) [19] | GSU search + ESU attention | n/a | No | 54k behaviors; +7.1% CTR |
| MaskNet (Weibo, 2021) [27] | Instance-guided multiplicative mask | Implicit | No | Twitter heavy ranker [28] |
| DHEN (Meta, 2022) [21] | Hierarchical ensemble of modules | Grows | No | +0.27% NE |
| TWIN (Kuaishou, 2023) [20] | Consistent GSU/ESU attention | n/a | No | Main traffic |
| Wukong (Meta, 2024) [22] | Stacked FM + linear compress | Up to 2^L | No | Scaling >100 GFLOP/ex |
| HSTU (Meta, 2024) [23] | Sequential transduction | n/a | No | 1.5T params; +12.4% |
| RankMixer (ByteDance, 2025) [30] | Token mixing + per-token FFN/MoE | Implicit | No | MFU 4.5% -> 45% |

## 8. Feature-Interaction Design Choices

Pulling the threads together, the main design axes are:

1. Explicit vs implicit interactions. MLPs can in principle learn crosses but are
   inefficient at multiplicative structure [27]; explicit modules (FM, cross layers, CIN,
   dot products) add inductive bias. Nearly all production models combine both
   (Wide & Deep, DeepFM, DCN, DLRM).
2. Bit-wise vs vector-wise. DCN crosses individual elements; CIN, FM and DLRM treat a field
   embedding as a unit [13]. RankMixer's per-token parameters are a modern answer that keeps
   field identity while mixing across tokens [30].
3. Bounded order vs depth-growing order. DLRM stops at order 2 for cost reasons [16];
   DCN/DCN-V2 add one order per layer [10][11]; Wukong doubles order per layer [22].
4. Parameter budget for interactions. Full matrices (DCN-V2) are expressive but O(d^2);
   low-rank and mixture-of-low-rank factorization recovers most quality [11]; LinkedIn and
   Pinterest both rely on low-rank variants in production [29][34].
5. Static features vs behavior sequences. Interaction networks over static features and
   target-attention over sequences evolved separately and are now being merged
   (InterFormer, HSTU, GEM) [23][24][25].
6. Hardware efficiency. Modules that look cheap in FLOPs may run at very low MFU; ByteDance
   reported 4.5% MFU before RankMixer [30], and Meta reports 20-25% MFU for GEM training
   after co-design [26]. Architecture choices increasingly follow GPU-friendliness
   (dense matmuls, uniform tokens) rather than theoretical elegance.
7. Calibration and online freshness. From FTRL [1] through GBDT+LR [3] to LiRank's
   co-trained isotonic layer [29], calibrated probabilities and fresh models remain
   non-negotiable for auctions and multi-objective blending.

## Key takeaways for system designers

- Start from the DLRM skeleton (embeddings, bottom MLP, interaction, top MLP); it is the
  shared ancestor of most modern rankers and maps cleanly onto model-parallel embeddings
  plus data-parallel dense layers.
- Use an explicit interaction module, and prefer low-rank DCN-V2-style crosses as a strong,
  cheap default; consider MaskNet-style multiplicative gating as an alternative.
- Make the user representation target-aware: a DIN-style attention block over recent
  behaviors is usually the highest-ROI addition after the base model.
- Treat long histories as an in-model retrieval problem (SIM, TWIN), keep the GSU and ESU
  relevance metrics consistent, and precompute item-side projections.
- Budget memory before FLOPs: feature admission (Bloom filters, frequency thresholds) and
  low-precision storage saved Google 60-75% RAM a decade ago and still matter for
  embedding tables.
- Expect benchmark gains to shrink in production: tuned baselines close most gaps [36], and
  latency, GPU memory and reproducibility constrain choices [33].
- If you want to scale, pick architectures with demonstrated scaling behavior (Wukong,
  HSTU, RankMixer) and measure MFU; watch for embedding collapse when scaling embedding
  width [35].
- When the best model cannot be served, train a large foundation ranker and transfer its
  knowledge into smaller serving models (Meta GEM).
- Never ship uncalibrated probabilities; correct for negative downsampling and consider a
  calibration layer trained jointly with the model.

## References

1. McMahan, H. B., Holt, G., Sculley, D., et al. "Ad Click Prediction: a View from the
   Trenches." Google, KDD 2013.
   https://research.google/pubs/ad-click-prediction-a-view-from-the-trenches/
2. Qun Le. "Paper Notes: Ad Click Prediction: a View from the Trenches." Blog.
   https://le.qun.ch/en/blog/paper-notes-ad-click-prediction-a-view-from-the-trenches/
3. He, X., Pan, J., Jin, O., et al. "Practical Lessons from Predicting Clicks on Ads at
   Facebook." Facebook, ADKDD 2014.
   https://quinonero.net/Publications/predicting-clicks-facebook.pdf
   (also https://ai.meta.com/research/publications/practical-lessons-from-predicting-clicks-on-ads-at-facebook/)
4. Rendle, S. "Factorization Machines." IEEE ICDM 2010.
   https://doi.org/10.1109/ICDM.2010.127
5. Dive into Deep Learning. "Factorization Machines" chapter. d2l.ai.
   https://d2l.ai/chapter_recommender-systems/fm.html
6. Juan, Y., Zhuang, Y., Chin, W.-S., Lin, C.-J. "Field-aware Factorization Machines for
   CTR Prediction." National Taiwan University, RecSys 2016.
   https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf
7. Juan, Y., Lefortier, D., Chapelle, O. "Field-aware Factorization Machines in a
   Real-world Online Advertising System." Criteo, WWW 2017 Companion.
   https://arxiv.org/abs/1701.04099
8. Cheng, H.-T., et al. "Wide & Deep Learning for Recommender Systems." Google, DLRS 2016.
   https://arxiv.org/abs/1606.07792
9. Guo, H., Tang, R., Ye, Y., Li, Z., He, X. "DeepFM: A Factorization-Machine based Neural
   Network for CTR Prediction." Huawei Noah's Ark, IJCAI 2017.
   https://arxiv.org/abs/1703.04247
10. Wang, R., Fu, B., Fu, G., Wang, M. "Deep & Cross Network for Ad Click Predictions."
    Google/Stanford, ADKDD 2017. https://arxiv.org/abs/1708.05123
11. Wang, R., et al. "DCN V2: Improved Deep & Cross Network and Practical Lessons for
    Web-scale Learning to Rank Systems." Google, WWW 2021. https://arxiv.org/abs/2008.13535
12. TensorFlow Recommenders. "Deep & Cross Network (DCN)" tutorial. Google.
    https://www.tensorflow.org/recommenders/examples/dcn
13. Lian, J., et al. "xDeepFM: Combining Explicit and Implicit Feature Interactions for
    Recommender Systems." Microsoft/USTC, KDD 2018. https://arxiv.org/abs/1803.05170
14. Song, W., et al. "AutoInt: Automatic Feature Interaction Learning via Self-Attentive
    Neural Networks." Peking University/Mila, CIKM 2019. https://arxiv.org/abs/1810.11921
15. Meta AI. "DLRM: An advanced, open source deep learning recommendation model." Blog,
    July 2019.
    https://ai.meta.com/blog/dlrm-an-advanced-open-source-deep-learning-recommendation-model/
16. Naumov, M., et al. "Deep Learning Recommendation Model for Personalization and
    Recommendation Systems." Facebook, 2019. https://arxiv.org/abs/1906.00091
17. Zhou, G., et al. "Deep Interest Network for Click-Through Rate Prediction." Alibaba,
    KDD 2018. https://arxiv.org/abs/1706.06978
18. Zhou, G., Mou, N., et al. "Deep Interest Evolution Network for Click-Through Rate
    Prediction." Alibaba, AAAI 2019. https://arxiv.org/abs/1809.03672
19. Pi, Q., et al. "Search-based User Interest Modeling with Lifelong Sequential Behavior
    Data for Click-Through Rate Prediction." Alibaba, CIKM 2020.
    https://arxiv.org/abs/2006.05639
20. Chang, J., et al. "TWIN: TWo-stage Interest Network for Lifelong User Behavior Modeling
    in CTR Prediction at Kuaishou." Kuaishou, KDD 2023. https://arxiv.org/abs/2302.02352
21. Zhang, B., Luo, L., Liu, X., et al. "DHEN: A Deep and Hierarchical Ensemble Network for
    Large-Scale Click-Through Rate Prediction." Meta, 2022.
    https://arxiv.org/abs/2203.11014
22. Zhang, B., Luo, L., Chen, Y., et al. "Wukong: Towards a Scaling Law for Large-Scale
    Recommendation." Meta, ICML 2024. https://arxiv.org/abs/2403.02545
23. Zhai, J., et al. "Actions Speak Louder than Words: Trillion-Parameter Sequential
    Transducers for Generative Recommendations." Meta, ICML 2024.
    https://arxiv.org/abs/2402.17152
24. Zeng, Z., et al. "InterFormer: Effective Heterogeneous Interaction Learning for
    Click-Through Rate Prediction." Meta/UIUC, 2024-2025. https://arxiv.org/abs/2411.09852
25. Engineering at Meta. "Meta's Generative Ads Model (GEM): The Central Brain Accelerating
    Ads Recommendation AI Innovation." November 2025.
    https://engineering.fb.com/2025/11/10/ml-applications/metas-generative-ads-model-gem-the-central-brain-accelerating-ads-recommendation-ai-innovation/
26. Engineering at Meta. "GEM Training: How Meta Doubled the Efficiency of Its LLM-Scale Ads
    Foundation Model." August 2026.
    https://engineering.fb.com/2026/08/03/ml-applications/training-gem-at-llm-scale-meta-ads-recommendation-foundation-model/
27. Wang, Z., She, Q., Zhang, J. "MaskNet: Introducing Feature-Wise Multiplication to CTR
    Ranking Models by Instance-Guided Mask." Sina Weibo, 2021.
    https://arxiv.org/abs/2102.07619
28. Twitter Engineering. "Twitter's Recommendation Algorithm." March 2023.
    https://blog.x.com/engineering/en_us/topics/open-source/2023/twitter-recommendation-algorithm
    (secondary analysis: https://blog.reachsumit.com/posts/2023/04/the-twitter-ml-algo/)
29. Borisyuk, F., et al. "LiRank: Industrial Large Scale Ranking Models at LinkedIn."
    LinkedIn, KDD 2024. https://arxiv.org/abs/2402.06859
30. ByteDance. "RankMixer: Scaling Up Ranking Models in Industrial Recommenders." 2025.
    https://arxiv.org/abs/2507.15551
31. ByteDance. "TokenMixer-Large: Scaling Up Large Ranking Models in Industrial
    Recommenders." 2026. https://arxiv.org/abs/2602.06563
32. Pinterest Engineering. "User Action Sequence Modeling for Pinterest Ads Engagement
    Modeling." Medium.
    https://medium.com/pinterest-engineering/user-action-sequence-modeling-for-pinterest-ads-engagement-modeling-21139cab8f4e
33. Pinterest. "Improving feature interactions at Pinterest under industry constraints."
    CIKM 2024 workshop. https://arxiv.org/abs/2412.01985
34. Li, J., et al. "ML-DCN: Masked Low-Rank Deep Crossing Network Towards Scalable Ads
    Click-through Rate Prediction at Pinterest." Pinterest, 2026.
    https://arxiv.org/abs/2602.09194
35. Guo, X., et al. "On the Embedding Collapse When Scaling up Recommendation Models."
    Tsinghua/Tencent, ICML 2024. https://arxiv.org/abs/2310.04400
36. Zhu, J., Liu, J., Yang, S., Zhang, Q., He, X. "BARS-CTR: Open Benchmarking for
    Click-Through Rate Prediction." Huawei, CIKM 2021. https://arxiv.org/abs/2009.05794
37. Meta. "AutoML for Large Capacity Modeling of Meta's Ranking Systems." 2023.
    https://arxiv.org/abs/2311.07870
38. Meta. "On the Practice of Deep Hierarchical Ensemble Network for Ad Conversion Rate
    Prediction." 2025. https://arxiv.org/abs/2504.08169
39. Golovin, D., Sculley, D., McMahan, H. B., Young, M. "Large-Scale Learning with Less RAM
    via Randomization." Google, ICML 2013. https://arxiv.org/abs/1303.4664
40. Sodhani, S. "Practical Lessons from Predicting Clicks on Ads at Facebook" (paper notes);
    and E. Melt notes.
    https://shagunsodhani.com/papers-I-read/Practical-Lessons-from-Predicting-Clicks-on-Ads-at-Facebook
    and https://notes.elimelt.com/ml/recommender-systems/predicting-clicks-on-ads-at-facebook
41. NextRoll Tech Blog. "Factorization Machines." 2015.
    https://tech.nextroll.com/blog/data-science/2015/08/25/factorization-machines.html
42. Lacuna summary of "Wide & Deep Learning for Recommender Systems."
    https://lacuna.tiptreesystems.com/work/wide-deep-learning-for-recommender-systems/wrk_808f4b394e1c85f393b84b3964d18fd7
43. "xDeepFM: Combining Explicit and Implicit Feature Interactions" (review). Substack.
    https://happystrongcoder.substack.com/p/xdeepfm-combining-explicit-and-implicit
44. Emergent Mind. "DLRM: Deep Learning Recommendation Model" (topic summary).
    https://www.emergentmind.com/topics/deep-learning-recommendation-model-dlrm
45. Salamone, L. "Summary: Deep & Cross Net v2." Blog.
    https://blog.lukesalamone.com/posts/deep-cross-net-v2/
46. Zhou, G., et al. DIEN paper PDF (serving section). https://arxiv.org/pdf/1809.03672
47. Emergent Mind. "Search-based Interest Model (SIM)" (topic summary).
    https://www.emergentmind.com/topics/search-based-interest-model-sim
48. ByteByteGo. "How Meta Built a New AI-Powered Ads Model for 5% Better Conversions."
    https://blog.bytebytego.com/p/how-meta-built-a-new-ai-powered-ads
