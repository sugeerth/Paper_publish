# 16. Ads Recommendation Systems: pCTR/pCVR Prediction, Auctions, Pacing, Calibration and the Meta, Google, Pinterest, Snap and Twitter Ads Stacks

> Research note: live web search and page fetching were not available when this chapter
> was written. The shared search budget for the run was used up, and outbound fetches to
> company blogs and arXiv failed DNS resolution or were refused by the egress proxy. The
> chapter is therefore built from two kinds of source. The first is primary sources that
> other chapters of this Atlas had already fetched and checked: Meta's Andromeda, Lattice
> and GEM posts and papers, Google's FTRL and Photon papers, Facebook's GBDT+LR paper,
> Twitter's delayed-feedback paper, Pinterest, Snap and LinkedIn ads write-ups. The second
> is a small set of canonical auction-theory papers, cited by DOI. Numbers that could not be
> rechecked are marked "approximately" or "reported". Verify figures against the originals
> before quoting them.

## Overview

An ads recommender is a recommender with a price tag on every slot. It has the same funnel
as an organic feed (retrieval, light ranking, heavy ranking, re-ranking), but three things
change the design.

1. **The output is money, so predictions must be calibrated probabilities, not just
   scores.** An auction ranks ads by something like bid x pCTR (or bid x pCTR x pCVR for
   conversion objectives) and charges a price derived from the next competitor's score. If
   pCTR is 2x too high, the advertiser is ranked and charged as if the ad were twice as
   good. Every ads paper since 2013 treats calibration as a first-class output [1][3].
2. **There is a second optimizer: the advertiser.** Advertisers set budgets and targets
   (cost per click, cost per action, return on ad spend). The platform must spend each
   budget smoothly over the day (pacing) and, more and more often, bid on the advertiser's
   behalf (auto-bidding) [39][40].
3. **Ads compete with organic content for the user's attention.** Ad load, spacing and the
   long-term "ad blindness" users learn are page-level costs. Google showed that cutting
   mobile ads could raise long-term value [9].

This chapter goes through the stack in the order a request flows, then company by
company: Google (FTRL, Photon, DCN), Facebook/Meta (GBDT+LR, DHEN, Wukong, InterFormer,
Lattice, Andromeda, GEM), Pinterest, Snap, Twitter/X, LinkedIn, Spotify, and the 2025-2026
generative-advertising wave (Tencent GPR, Meituan EGA, ByteDance TokenMixer-Large).

```
 ad request (user, context, placement)
      |
      v
 +-----------------+   eligibility: targeting, budget left, policy, frequency caps
 | ad index / ACL  |   10^6 .. 10^7+ eligible ads (Meta: "tens of millions") [10][11]
 +--------+--------+
          v
 +-----------------+   two-tower / learned index (Andromeda) / GPU brute force
 | retrieval       |   -> 10^3 .. 10^4 candidates
 +--------+--------+
          v
 +-----------------+   cheap model, often distilled from the heavy ranker
 | light ranking   |   -> 10^2 candidates
 +--------+--------+
          v
 +-----------------+   multi-task DNN: pCTR, pCVR (per conversion type), quality,
 | heavy ranking   |   negative feedback; calibrated per task
 +--------+--------+
          v
 +-----------------+   score = bid(or auto-bid) x p(action) + quality terms
 | auction         |   allocation + pricing (GSP-like, VCG-like), reserve prices
 +--------+--------+
          v
 +-----------------+   ad load, spacing, slot choice vs organic items
 | allocation      |
 +--------+--------+
          v
 impression --> click/conversion logs --> joiner --> online trainer (minutes-hours)
          ^                                                  |
          +-------------- pacing / bid controller <----------+ (spend telemetry)
```

## 1. Foundations: pCTR, pCVR and the expected-value score

### 1.1 What the ranker predicts

The core predictions are:

- **pCTR** = P(click | impression, user, ad, context).
- **pCVR** = P(conversion | click, ...). It is trained on clicked impressions but used on
  all impressions, which creates sample-selection bias (Section 4.3) [33].
- **pCTCVR** = pCTR x pCVR = P(click and convert | impression), the quantity a
  cost-per-action (CPA) advertiser actually pays for [33].
- Quality and negative-feedback predictions (hide, report, low-quality click), which enter
  the auction as a penalty or a quality term.

The ranking score is an expected value. Under the usual convention, an ad billed per click
is scored by bid x pCTR (its expected cost per thousand impressions, eCPM, up to a
constant). An ad optimized for conversions is scored by target CPA x pCTR x pCVR. Both
Google's and Facebook's ads papers state that the auction multiplies predicted CTR by bids,
which is why calibration matters as much as ranking quality [1][3].

### 1.2 Metrics: normalized entropy, log loss, calibration ratio

Facebook introduced **normalized entropy (NE)**: log loss divided by the entropy of the
background CTR. Raw log loss looks deceptively good when the base rate is near 0, and NE
makes datasets with different base rates comparable [3]. Google's paper relies on log loss
and AUC and tracks calibration slices as a core monitoring tool [1]. Both papers stress that
improvements of a fraction of a percent in log loss or NE matter online [1][3].

Calibration is tracked as a global ratio (sum of predicted / sum of observed) and as
reliability curves per slice and per predicted-score bucket [1][3].

## 2. Google: the linear era and the plumbing that made it work

### 2.1 Ad Click Prediction: a View from the Trenches (KDD 2013)

**Problem.** Search ads needed a CTR model trained online on an enormous impression
stream, with billions of sparse features, small enough in memory to serve at very high QPS
[1].

**Model.** Logistic regression trained with **FTRL-Proximal**. The algorithm combines the
sparsity of L1-regularized updates (exact zeros) with good convergence, and it uses
**per-coordinate learning rates**. These require storing the sum of gradients and the sum of
squared gradients per feature [1][2].

**Memory tricks.** Memory, not FLOPs, was the binding constraint [1]:

| Trick | Setting | RAM saved | AucLoss cost |
|---|---|---|---|
| Poisson feature inclusion | p = 0.03 | ~60% | 0.020% |
| Poisson feature inclusion | p = 0.1 | ~40% | 0.006% |
| Bloom-filter inclusion | admit after n = 2 sightings | ~66% | 0.008% |
| 16-bit q2.13 weights, randomized rounding | vs 64-bit floats | large (summaries: ~75% for coefficients) | no measurable loss |

(Figures as reported in the paper and in summaries of it [1][2].)

The paper also trains many model variants at once with shared feature-index structures, and
subsamples non-click queries while keeping all clicks, correcting with importance weights
[1]. The companion paper formalizes randomized rounding for low-RAM learning [2].

**Negative results.** The paper lists ideas that did not pay off in production, for example
feature hashing, which needed too large a table to avoid damaging collisions [1]. This
"what did not work" section became a model for later industrial papers.

**Calibration.** A calibration layer sits between the model and the auction, and the paper
treats calibration plots across many slices as a primary monitoring tool [1].

### 2.2 Photon: exactly-once click joins (SIGMOD 2013)

Training labels and billing come from the same join. Photon joins each click to its query
event by a unique query ID across several datacenters with exactly-once semantics. The core
is the **IdRegistry**, a Paxos-replicated, geo-distributed key-value store of IDs that have
already been joined. A click is never emitted twice even when several datacenters process
the same click stream. Clicks that arrive before their query are retried. Throughput is
reported in the millions of joined events per minute, with seconds of latency
(approximate) [4].

Lesson: in ads, the label pipeline is also the billing pipeline. Duplicate positives inflate
CTR, corrupt calibration and overcharge advertisers.

### 2.3 From crosses to deep crosses: DCN and DCN-V2

Google's Deep & Cross Network (ADKDD 2017) was introduced explicitly for ad click
prediction. Each cross layer multiplies the original input with the current layer's output,
so the polynomial degree grows by one per layer [6]. DCN-V2 (WWW 2021) replaced the cross
vector with a full (optionally low-rank) matrix and reported gains across many web-scale
learning-to-rank systems at Google, under very high QPS and strict latency [7]. DCN-V2
became the production baseline at Pinterest ads and LinkedIn (Section 6).

### 2.4 Long-term ad load: "Focusing on the Long-term" (KDD 2015)

Hohnhold, O'Brien and Tang showed that users exposed to more or worse ads gradually learn
to ignore them, which drives ad CTR down over months. Short A/B tests cannot see this. They
fit a model that predicts long-term user learning from short-term ad quality and ad load,
so that later launches can be judged with short tests plus that model. Google then
**reduced** the number of ads on mobile search (reported as roughly 50% on some surfaces,
approximate), judging that long-term gains outweighed short-term revenue loss [9].

This is the canonical public evidence that ad load is a long-horizon decision, not a
per-request revenue maximization.

### 2.5 Microsoft Bing adPredictor (ICML 2010)

Bing's sponsored-search CTR model was a Bayesian probit regression over sparse binary
features, trained online with assumed-density filtering. Each weight carries a Gaussian
mean and variance, so the model knows how uncertain it is about a new ad. Sampling from the
posterior gives new ads impressions in proportion to their chance of being good, which is
Thompson sampling [5]. It is the earliest well-documented production answer to ad
cold-start exploration.

## 3. Facebook/Meta: from GBDT+LR to foundation models

### 3.1 Practical Lessons from Predicting Clicks on Ads at Facebook (ADKDD 2014)

**Scale at the time.** More than 750M daily active users and over 1M active advertisers
[3].

**Architecture.** Boosted decision trees act as a learned feature transform: the leaf index
of each tree becomes a one-hot feature, and the concatenation goes into a logistic
regression trained online. The hybrid beat either trees or LR alone by more than 3% NE [3].

```
 raw features --> boosted trees T1..TK (retrained ~daily)
                    | leaf ids (one-hot)
                    v
              logistic regression (online SGD, near real time)
                    |
                    v
          p (downsampled space) --> q = p / (p + (1 - p) / w) --> auction
```

**Online joiner.** Impressions and clicks are joined by request ID inside a waiting window.
Too short a window mislabels late clicks as negatives. Too long a window delays training and
costs memory. The joiner needs alarms that cut the trainer off if the click stream stops
[3].

**Key lessons [3].**

- Historical features (past behavior of user and ad) dominate contextual features.
- Freshness matters: NE worsens as training data falls days behind the serving day.
- **Negative downsampling with recalibration**: keep negatives at rate w, then correct with
  q = p / (p + (1 - p) / w). This formula became standard practice.
- Uniform subsampling to about 10% of data reportedly costs about 1% NE (approximate).

### 3.2 Scaling the ranker: DHEN, Wukong, InterFormer

Meta's ranking architecture research since 2022 has mostly been published in ads terms.

- **DHEN (2022).** Each layer is an ensemble of heterogeneous interaction modules
  (dot-product, self-attention, convolution, DCN, linear). Layers stack into a hierarchy of
  increasing interaction order. It improved NE by 0.27% over an AdvancedDLRM baseline on a
  large CTR dataset and introduced Hybrid Sharded Data Parallel training, reported at up to
  1.2x throughput over FSDP [18]. A 2025 paper describes running DHEN for **ad
  conversion-rate prediction** in production [19], and Meta built AutoML over DHEN-style
  module search for its ranking models [23].
- **Wukong (ICML 2024).** A stack of identical layers, each with a Factorization Machine
  Block and a Linear Compress Block. Layer i covers interaction orders 1 to 2^i. On Meta's
  internal data it held a scaling law across two orders of magnitude of model complexity,
  beyond 100 GFLOP per example, where earlier architectures saturated [20].
- **InterFormer (2024-2025).** Interleaves an Interaction Arch and a Sequence Arch with
  bidirectional information flow and a Cross Arch for selective summarization. It reports
  0.15% NE and 24% QPS gains on Meta internal data and a **0.6% topline gain in pilot ads
  launches** [21].
- **HSTU (ICML 2024).** Generative-recommender architecture with reported 1.5-trillion-
  parameter models and +12.4% online metrics on surfaces with billions of users [22]. It is
  covered in Chapter 6.

### 3.3 Lattice: consolidating the ads model fleet (2023, paper 2025)

**Problem.** Ads ranking had fragmented into many models across surfaces (Feed, Stories,
Reels, Audience Network), objectives (click, many conversion types) and funnel stages. That
slowed iteration and wasted compute [12].

**Design.** Lattice is a **multi-domain, multi-task** architecture with sparse activation.
It learns from many data sources and handles delayed feedback ranging from seconds to days.
Meta's 2023 description cites **trillions of parameters** trained on **hundreds of billions
of examples** [12]. Hand-balancing thousands of domains and tens of objectives is
impossible, so Meta uses Pareto-optimality-based methods, including Pareto-front feature
selection and MetaBalance [12][14]. MetaBalance rescales each auxiliary task's gradient,
per network part, toward the magnitude of the target task's gradient so auxiliary tasks
neither dominate nor vanish [14].

The 2025 paper ("Model Space Redesign for Cost-Effective Industry-Scale Ads
Recommendations") describes four moves: cross-domain knowledge sharing, data
consolidation, model unification, and **distillation** of a large consolidated model into
lean user-facing models. Reported results: +10% on revenue-driving top-line metrics, +11.5%
user satisfaction, +6% conversion rate, and 20% capacity saving (Meta's own figures) [13].

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

### 3.4 Andromeda: deep-learning ads retrieval (December 2024)

**Problem.** Generative-AI creative tools sharply increased the number of eligible ads.
Meta said more than a million advertisers created more than 15 million ads in one month with
its GenAI tools. Two-tower retrieval with ANN caps how much user-ad interaction the model
can express [10].

**Architecture [10].**

- A highly customized deep neural network with **sub-linear inference cost**, running with
  heavy parallelism on **NVIDIA Grace Hopper Superchips**, with Meta's **MTIA** accelerator
  also part of the hardware plan.
- Ads are organized into a **hierarchical index with multiple layers**, so inference only
  expands the most relevant nodes.
- The **index and the retrieval model are trained jointly**, which Meta says gives better
  precision and recall than two-tower or ANN approaches.

**Results.** On Facebook and Instagram: **+6% recall** for the retrieval system and **+8% ads
quality** on selected segments [10]. Secondary write-ups cite a roughly **10,000x** increase
in retrieval model complexity and a narrowing from **tens of millions** of eligible ads to a
**few thousand** candidates per request; treat these as approximate [11].

```
 user + context features
        |
        v
 +--------------------------------------------------------------+
 | Andromeda retrieval DNN on Grace Hopper (unified CPU-GPU mem) |
 |                                                              |
 |   level 0:  root clusters   --score--> keep top nodes         |
 |   level 1:  sub-clusters    --score--> keep top nodes         |
 |   level L:  individual ads  --score--> top ~10^3 ads          |
 |   (hierarchy learned jointly with the scoring network)        |
 +-----------------------------+--------------------------------+
                               v
                 ranking (Lattice / GEM-distilled models) -> auction
```

**Lesson.** Co-designing model, index and hardware makes heavy computation affordable at the
stage that sees the most items, which moves the cascade's quality bottleneck upstream [10].

### 3.5 GEM: the Generative Ads Recommendation Model (November 2025)

Meta describes GEM as "the largest foundation model for recommendation systems (RecSys) in
the industry, trained at the scale of large language models" [15]. It is trained across
thousands of GPUs with an LLM-inspired paradigm.

**Role in the stack.** GEM does **not** score every candidate online. It is a central teacher
whose learnings are propagated "across the entire ads model fleet" through post-training
knowledge transfer (distillation and representation transfer) into the vertical models that
run in each funnel stage [15]. Meta frames the work in three pillars: scaling the model
architecture, post-training knowledge transfer to downstream models, and training
scalability [15]. Secondary coverage links the architecture to InterFormer-style
sequence/non-sequence interaction and Wukong-style stacked factorization [17].

**Results.** +5% ad conversions on Instagram and +3% on Facebook Feed in Q2 2025 [15].

**Training (August 2026 follow-up).** Meta reports doubling end-to-end training efficiency to
**20-25% model FLOPs utilization** while scaling training FLOPs **4x in twelve months**,
through co-design of kernels, numeric precision, parallelism, networking and memory [16].

```
            +------------------------------------------------+
            | GEM (offline, LLM-scale, thousands of GPUs)    |
            | inputs: long user ad/organic sequences +       |
            |         non-sequence features across surfaces  |
            +-----------------------+------------------------+
                                    | distillation / representation transfer
         +--------------------------+---------------------------+
         v                          v                           v
  retrieval models           light rankers               heavy rankers (Lattice)
  (Andromeda)                (per surface)               per surface / objective
         \__________________________|___________________________/
                                    v
                          auction + allocation
```

**Lesson.** When the best model is too expensive to serve, keep it offline as a teacher and
pay for it once in training, many times in student quality. This is the same pattern LLM
serving uses, and Meta now applies it fleet-wide [15].

### 3.6 Meta ads evolution at a glance

| Year | System | Stage | Key idea | Reported impact |
|---|---|---|---|---|
| 2014 | GBDT + LR [3] | ranking | trees as feature transform, online LR | >3% NE vs either alone |
| 2022 | DHEN [18] | ranking | hierarchical ensemble of interaction modules | +0.27% NE |
| 2023 | Lattice [12] | ranking | multi-domain multi-task consolidation | trillions of params |
| 2024 | Wukong [20] | ranking | stacked FMs with scaling law | scaling beyond 100 GFLOP/ex |
| 2024 | InterFormer [21] | ranking | bidirectional seq / non-seq interaction | +0.6% topline (pilot ads) |
| 2024 | Andromeda [10] | retrieval | learned hierarchical index on GPUs | +6% recall, +8% quality |
| 2025 | Lattice paper [13] | fleet | redesign + distillation | +10% topline, -20% capacity |
| 2025 | GEM [15] | teacher | LLM-scale foundation model | +5% IG / +3% FB conversions |
| 2026 | GEM training [16] | infra | MFU to 20-25%, 4x FLOPs/yr | n/a |

## 4. Conversion modeling: delayed and biased labels

Conversions (purchases, installs, sign-ups) arrive minutes to weeks after the click. This
gives ads two label problems that organic feeds mostly avoid.

### 4.1 Delayed feedback

- **Criteo's joint delay model (Chapelle, KDD 2014).** Each click has a latent "will ever
  convert" variable and an exponential delay. A click unconverted after elapsed time e is
  either a real negative or a slow positive. Likelihood terms: converted at delay d gives
  P(C=1) x lambda x exp(-lambda d); not converted after e gives 1 - P(C=1) + P(C=1) x
  exp(-lambda e) [30].
- **Twitter's fake negatives (RecSys 2019).** Every impression is ingested immediately as a
  negative and again as a positive when the engagement arrives. No waiting window, so
  freshness is maximal. The bias is corrected either by importance weighting (FN-weighted)
  or at prediction time with p = b / (1 - b), where b is the biased prediction
  (FN-calibration). Both beat waiting-window baselines and the delayed-feedback model
  offline and in an online A/B test on Twitter ads [28].
- **Alibaba ES-DFM (AAAI 2021) and DEFER (KDD 2021).** Wait a short elapsed window. Emit
  early conversions as positives and others as negatives, and re-emit late positives
  (ES-DFM) [31]. DEFER also re-ingests real negatives once the attribution window closes, so
  every sample has its final label, with an importance-sampling loss for the duplication.
  DEFER reports offline and online gains in Alibaba display advertising [32].
- **Lattice** must handle feedback delays from seconds to days in one model [12].

### 4.2 Streaming training and freshness

ByteDance's Monolith (2022) shows the modern ads/feeds streaming design. A Flink joiner
merges serving-time features with user actions from Kafka. Negatives are sampled inside the
joiner with a **log-odds correction** at serving time. Sparse parameters sync from training
to serving parameter servers at minute granularity, which measurably improves AUC.
Embedding tables are collisionless (cuckoo hashing) with frequency filtering and expiry [38].

### 4.3 Sample-selection bias: ESMM

CVR models trained only on clicked impressions are applied to all impressions. Alibaba's
ESMM (SIGIR 2018) trains a CTR tower and a CVR tower with shared embeddings, supervising
pCTR (on clicks) and pCTCVR = pCTR x pCVR (on click-and-convert) over the entire impression
space. The CVR tower is never directly supervised. It uses the product, not a division,
because dividing by small CTR values is numerically unstable [33]. On 8.9 billion Taobao
samples, reported gains were roughly +2.18% AUC on CVR and +2.32% on CTCVR
(approximate) [33].

```
 impression features --> shared embeddings
                              |                 |
                          CTR tower         CVR tower
                              |                 |
                            pCTR ------(x)---- pCVR
                              |                 |
                     loss on click       pCTCVR: loss on click & convert
```

## 5. Calibration: the contract with the auction

Every step that changes the training distribution moves calibration: negative downsampling,
fake-negative duplication, importance weights, new features, and a new architecture [3][28].

| Method | Where used | Notes |
|---|---|---|
| Downsampling correction q = p / (p + (1-p)/w) | Facebook ads [3] | exact for uniform negative sampling |
| Log-odds correction at serving | ByteDance Monolith [38] | same idea in logit space |
| FN-calibration p = b / (1 - b) | Twitter ads [28] | undoes fake-negative duplication |
| Post-hoc calibration layer + slice plots | Google ads [1] | calibration monitored per slice |
| Isotonic regression / field-level calibration | general practice [37] | per advertiser or placement |
| Per-task temperature scaling | Spotify audio ads [36] | one parameter per task; chosen for simplicity |
| Isotonic layer trained jointly in the DNN | LinkedIn LiRank, Feed/Jobs/Ads CTR [34] | 2026 version also debiases position [35] |

Spotify states the reason plainly: miscalibration distorts auction ordering and pricing,
so each task head of its multi-task audio-ads model gets its own temperature [36]. LinkedIn
argues that a single learnable isotonic layer can capture interactions across task, position
and platform that a global Platt transform cannot [35].

Operational rule from these sources: recalibrate whenever a head's training distribution
changes, and monitor predicted versus observed rates per head, surface, advertiser segment
and position bucket. A calibration drift is a pricing change.

## 6. Pinterest, Snap, Twitter/X, LinkedIn and others

### 6.1 Pinterest ads ranking

- **Sequence features in ads engagement models.** Pinterest added Transformer-encoded user
  action sequences to its ads engagement models, with DCN-V2 as the production baseline. It
  reported DCN-V2 cross-layer overflow problems under float16 training, a practical warning
  for anyone moving crossing layers to mixed precision [24].
- **Feature interactions under industry constraints (2024).** Pinterest describes how it
  evaluated interaction modules under latency and cost limits rather than on public
  benchmarks alone [25].
- **ML-DCN (2026).** Naively scaling DCN-V2 and MaskNet gave rapidly diminishing offline
  gains. ML-DCN adds an instance-conditioned mask to a low-rank crossing layer and was
  deployed for ads CTR prediction with neutral serving cost [26].

### 6.2 Snap ads ranking

Snap's ad ranking predicts several conversion events at once (app installs, purchases,
sign-ups) with **MMoE and PLE** multi-task architectures, so sparse conversion types borrow
signal from dense ones [27]. Snap has also published extensively on embedding-based
retrieval and generative recommendation with semantic IDs (Chapter 6), which is relevant to
ad candidate generation.

### 6.3 Twitter/X ads

- **Continuous training with delayed feedback** (Section 4.1) is Twitter ads' best-known
  public contribution [28].
- **TwHIN** (KDD 2022) is a heterogeneous-network embedding over users, tweets, advertisers
  and engagement relations. The same embeddings are reused for candidate generation, ranking
  features and ads [51].
- In the timeline, ads are ranked by a separate system and then **mixed** into the For You
  feed by Home Mixer alongside Who-to-Follow modules, after organic ranking and filtering
  [29].

### 6.4 LinkedIn

LiRank (KDD 2024) serves Feed, Jobs and **Ads CTR**. It extends DCN-V2 into a Residual DCN
with attention and low-rank matrices (nearly 30% fewer parameters) and adds the jointly
trained isotonic calibration layer [34]. LinkedIn's LiGNN graph embeddings gave about +2%
Ads CTR in A/B tests (approximate) [50].

### 6.5 Criteo

Criteo ran Field-aware Factorization Machines in production for display-ads CTR and
conversion prediction, adding a seeding algorithm and distributed learning to cut training
time [8]. Its "Offline A/B testing" work proposed capped and normalized importance-sampling
estimators for top-K display ads that predicted the sign and ordering of online A/B results
better than standard offline metrics [48].

## 7. Auctions: allocation and pricing

### 7.1 GSP, VCG and why ads use "GSP-like" auctions

- **Vickrey (1961)** showed that a sealed-bid second-price auction for a single item makes
  truthful bidding a dominant strategy [43].
- **Position auctions.** Search and feed ads sell several slots with different click rates.
  The **Generalized Second-Price (GSP)** auction ranks advertisers by bid (or by bid x
  quality) and charges each the minimum it needs to keep its slot, roughly the next
  advertiser's score divided by its own quality. Edelman, Ostrovsky and Schwarz showed that
  **GSP is not truthful**, but it has a "locally envy-free" equilibrium whose payments equal
  those of the **VCG** mechanism [41]. Varian independently analyzed the same symmetric Nash
  equilibria of position auctions and fit them to data [42].
- **VCG** charges each winner the externality it imposes on others. It is truthful but harder
  to explain to advertisers, and it composes awkwardly with reserve prices and quality
  scores. Industry practice is "GSP-like ranking by expected value, with quality
  adjustments", or VCG-style pricing on some surfaces; exact mechanisms are rarely
  published.

```
 rank by  score_i = bid_i x pCTR_i (x pCVR_i) + quality_i         (expected value)
 slot k goes to the k-th highest score
 GSP price per click for winner k:
     cpc_k = (score_{k+1} - quality_k) / pCTR_k     (+ minimum increment, >= reserve)
 => winner k pays just enough to beat k+1, so pCTR errors move both rank AND price
```

### 7.2 Generative auctions

The 2025-2026 generative-advertising papers move parts of the auction into the model.
Meituan's **EGA-V2** generates ads with RQ-VAE semantic IDs (3 levels x 1024 codes),
decomposes generation into P(POI) x P(creative | POI), and adds token-level bids and a
**payment network** that guarantees individual rationality (payment never exceeds bid).
Incentive compatibility is enforced only approximately, through a Lagrangian on ex-post
regret [44]. Tencent's **GPR** for WeChat Channels ads uses four token types (user, organic,
environment, ad), RQ-KMeans+ semantic IDs, a heterogeneous hierarchical decoder and
value-guided trie beam search. It reports latency falling from 200 ms+ to about 80 ms and five
cascade models collapsing into one [45]. Even these designs keep an explicit payment and
policy layer: pricing is too consequential to leave implicit.

## 8. Budgets, pacing and auto-bidding

Advertisers declare budgets and goals, not per-auction bids. Two control loops sit around
the auction.

- **Pacing** spreads spend over the campaign. The two common families are probabilistic
  **throttling** (participate in a fraction of eligible auctions) and **bid modification**
  (scale the bid by a multiplier driven by a feedback controller on spend versus plan). This
  is standard practice across platforms; detailed production descriptions are scarce.
- **Auto-bidding** chooses bids to maximize conversions under budget and CPA/ROAS
  constraints. Alibaba's **budget-constrained bidding** (CIKM 2018) uses model-free RL (DQN)
  to adjust a bid-scaling factor over the day, with a reward design that avoids myopic
  overspending [39]. Alibaba's **multi-agent RTB** clusters advertisers into agents trained
  with a distributed multi-agent actor-critic, balancing competition and cooperation on
  Taobao [40].

```
 advertiser goal (budget B, target CPA)          spend telemetry (per minute)
           |                                               |
           v                                               v
 +-------------------+   multiplier lambda_t   +--------------------------+
 | bid controller /  |------------------------>| auction: bid_t = lambda_t |
 | RL policy         |<------------------------|  x pCVR x target CPA      |
 +-------------------+  spend, conversions,    +--------------------------+
     state: remaining budget, time of day, pace error
```

The shared design pattern: the RL policy outputs a **scalar control** (a bid multiplier),
not an item choice. That keeps the action space small, the policy easy to roll back, and the
rest of the stack unchanged [39][40]. Since bids are multiplied by pCVR, the bidder inherits
every calibration error from the ranker. Calibration and pacing are one system.

## 9. Ad load and allocation against organic content

Public posts from Meta do not describe ad slotting in detail. The pattern visible across the
industry, and documented for X's Home Mixer [29], is that ads go through their own
retrieval, ranking and auction and are then **inserted** into the organic stream at allowed
positions under ad-load and spacing constraints.

```
 organic ranked list O = [o1, o2, ...]      auctioned ad list A = [a1, a2, ...]
                \                                 /
                 v                               v
        +----------------------------------------------------+
        | allocator                                          |
        |  slots: fixed (every k-th) or dynamic per user     |
        |  constraints: min gap, max ad load per session,    |
        |               no ad in slot 1, category rules      |
        |  objective: revenue(a, pos) + lambda x             |
        |             engagement_loss(pos)                   |
        +----------------------------------------------------+
```

Design decisions:

1. **Shared currency.** Organic value (predicted engagement) and ad value (expected revenue)
   are in different units. The exchange rate lambda is a policy, tuned against long-term
   retention [9].
2. **Long-horizon measurement.** Ad load changes need long-running holdouts or a model of
   user learning, because ad blindness develops over months [9].
3. **Generative allocation.** Tencent's GPR feeds organic and ad tokens through one decoder,
   which is a step toward joint allocation inside the model [45].

## 10. The 2025-2026 scaling wave in ads ranking

| System | Org / year | Stage | Architecture | Reported scale / result |
|---|---|---|---|---|
| Andromeda [10] | Meta 2024 | retrieval | learned hierarchical index, GH200 | +6% recall, +8% quality |
| Lattice [13] | Meta 2025 | ranking fleet | multi-domain MTL + distillation | +10% topline, 20% capacity saved |
| GEM [15][16] | Meta 2025-26 | teacher | LLM-scale foundation model | +5% IG / +3% FB conv.; 20-25% MFU |
| RankMixer [47] | ByteDance 2025 | ranking | token mixing + per-token FFN | MFU 4.5% -> 45%; ~100x params |
| TokenMixer-Large [46] | ByteDance 2026 | ranking | RankMixer successor | 7B and 15B offline (Douyin ads) |
| GPR [45] | Tencent 2025 | one model | generative, semantic IDs | 5 models -> 1; ~80 ms |
| EGA-V2 [44] | Meituan 2025 | one model | generative + payment network | IR guaranteed, approx. IC |
| ML-DCN [26] | Pinterest 2026 | ranking | masked low-rank crossing | neutral serving cost |
| LiRank [34] | LinkedIn 2024 | ranking | Residual DCN + isotonic layer | Feed / Jobs / Ads CTR |

Two directions are visible. Meta keeps a classic cascade (Andromeda, Lattice students, the
auction) and adds a giant offline teacher. Tencent and Meituan collapse the cascade into one
generative model while keeping a small explicit payment and policy module. ByteDance's
contribution is hardware efficiency: the jump from about 4.5% to 45% MFU is what makes a
100x parameter increase affordable at constant latency [47].

## Key takeaways for system designers

1. **Calibration is the API between ML and money.** Rank and price both depend on pCTR x
   bid. Correct every sampling choice (downsampling formula, log-odds, FN-calibration),
   calibrate per task head, and monitor per slice [1][3][28][34][36].
2. **Model the funnel explicitly.** Train CVR over the entire impression space (ESMM), and
   treat delayed conversions with fake-negative, elapsed-time or real-negative schemes
   rather than long waits [28][31][32][33].
3. **Freshness pays, so the label join is critical infrastructure.** Mint a unique
   impression ID, join idempotently (Photon), and train online or at minute granularity.
   Log features at serving time rather than recomputing them [3][4][38][49].
4. **Push model quality upstream.** Andromeda shows that a deep, hardware-co-designed
   retrieval stage gains recall that no ranker can recover [10].
5. **Consolidate, then distill.** Many per-surface, per-objective models waste compute and
   iteration speed. One large multi-domain model (Lattice) or an offline foundation teacher
   (GEM), distilled into lean serving students, is the pattern at the largest scale [13][15].
6. **Scale only architectures that are known to scale, and measure MFU.** Wukong and
   RankMixer/TokenMixer-Large show scaling laws, and naively widening DCN or MaskNet does not
   [20][26][47].
7. **Keep the mechanism explicit and auditable.** Even generative ads systems keep a payment
   network and policy layer. GSP-like pricing means every model error is also a pricing
   error [41][44][45].
8. **Let control policies turn knobs.** Pacing and auto-bidding work best as scalar
   multipliers on top of calibrated predictions [39][40].
9. **Decide ad load on long-term metrics.** Users learn to ignore ads. Use long holdouts or
   user-learning models, and set the organic-versus-ads exchange rate as policy [9].
10. **Publish your negative results internally.** Google's "what did not work" list saved
    the industry years. Keep one for your stack [1].

## References

1. McMahan, H. B., Holt, G., Sculley, D., et al. "Ad Click Prediction: a View from the Trenches." Google, KDD 2013. https://research.google/pubs/ad-click-prediction-a-view-from-the-trenches/
2. Golovin, D., Sculley, D., McMahan, H. B., Young, M. "Large-Scale Learning with Less RAM via Randomization." Google, ICML 2013. https://arxiv.org/abs/1303.4664 (paper notes on [1]: Qun Le, https://le.qun.ch/en/blog/paper-notes-ad-click-prediction-a-view-from-the-trenches/)
3. He, X., Pan, J., Jin, O., et al. "Practical Lessons from Predicting Clicks on Ads at Facebook." Facebook, ADKDD 2014. https://quinonero.net/Publications/predicting-clicks-facebook.pdf
4. Ananthanarayanan, R., et al. "Photon: Fault-tolerant and Scalable Joining of Continuous Data Streams." Google, SIGMOD 2013. https://research.google/pubs/photon-fault-tolerant-and-scalable-joining-of-continuous-data-streams/
5. Graepel, T., Quiñonero Candela, J., Borchert, T., Herbrich, R. "Web-Scale Bayesian Click-Through Rate Prediction for Sponsored Search Advertising in Microsoft's Bing Search Engine." Microsoft Research, ICML 2010. https://www.microsoft.com/en-us/research/publication/web-scale-bayesian-click-through-rate-prediction-for-sponsored-search-advertising-in-microsofts-bing-search-engine/
6. Wang, R., Fu, B., Fu, G., Wang, M. "Deep & Cross Network for Ad Click Predictions." Google/Stanford, ADKDD 2017. https://arxiv.org/abs/1708.05123
7. Wang, R., et al. "DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems." Google, WWW 2021. https://arxiv.org/abs/2008.13535
8. Juan, Y., Lefortier, D., Chapelle, O. "Field-aware Factorization Machines in a Real-world Online Advertising System." Criteo, WWW 2017 Companion. https://arxiv.org/abs/1701.04099
9. Hohnhold, H., O'Brien, D., Tang, D. "Focusing on the Long-term: It's Good for Users and Business." Google, KDD 2015. https://dl.acm.org/doi/10.1145/2783258.2788583
10. Meta Engineering. "Meta Andromeda: Supercharging Advantage+ automation with the next-gen personalized ads retrieval engine." Meta, December 2024. https://engineering.fb.com/2024/12/02/production-engineering/meta-andromeda-advantage-automation-next-gen-personalized-ads-retrieval-engine/
11. Loomer, J. "Meta Andromeda: What It Means for Your Ad Strategy" (secondary summary of [10]). 2025. https://www.jonloomer.com/meta-andromeda/
12. Meta AI. "Meta Lattice: AI ads performance and efficiency." Meta, 2023. https://ai.meta.com/blog/ai-ads-performance-efficiency-meta-lattice/
13. Meta. "Meta Lattice: Model Space Redesign for Cost-Effective Industry-Scale Ads Recommendations." Meta, 2025. https://arxiv.org/abs/2512.09200
14. He, Y., Feng, X., Cheng, C., Ji, G., Guo, Y., Caverlee, J. "MetaBalance: Improving Multi-Task Recommendations via Adapting Gradient Magnitudes of Auxiliary Tasks." Meta AI / TAMU, WWW 2022. https://arxiv.org/abs/2203.06801
15. Engineering at Meta. "Meta's Generative Ads Model (GEM): The central brain accelerating ads recommendation AI innovation." Meta, November 2025. https://engineering.fb.com/2025/11/10/ml-applications/metas-generative-ads-model-gem-the-central-brain-accelerating-ads-recommendation-ai-innovation/
16. Engineering at Meta. "GEM Training: How Meta Doubled the Efficiency of Its LLM-Scale Ads Foundation Model." Meta, August 2026. https://engineering.fb.com/2026/08/03/ml-applications/training-gem-at-llm-scale-meta-ads-recommendation-foundation-model/
17. ByteByteGo. "How Meta Built a New AI-Powered Ads Model for 5% Better Conversions" (secondary). 2025. https://blog.bytebytego.com/p/how-meta-built-a-new-ai-powered-ads
18. Zhang, B., Luo, L., Liu, X., et al. "DHEN: A Deep and Hierarchical Ensemble Network for Large-Scale Click-Through Rate Prediction." Meta, 2022. https://arxiv.org/abs/2203.11014
19. Meta. "On the Practice of Deep Hierarchical Ensemble Network for Ad Conversion Rate Prediction." Meta, 2025. https://arxiv.org/abs/2504.08169
20. Zhang, B., Luo, L., Chen, Y., et al. "Wukong: Towards a Scaling Law for Large-Scale Recommendation." Meta, ICML 2024. https://arxiv.org/abs/2403.02545
21. Zeng, Z., et al. "InterFormer: Effective Heterogeneous Interaction Learning for Click-Through Rate Prediction." Meta/UIUC, 2024-2025. https://arxiv.org/abs/2411.09852
22. Zhai, J., et al. "Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations." Meta, ICML 2024. https://arxiv.org/abs/2402.17152
23. Meta. "AutoML for Large Capacity Modeling of Meta's Ranking Systems." Meta, 2023. https://arxiv.org/abs/2311.07870
24. Pinterest Engineering. "User Action Sequence Modeling for Pinterest Ads Engagement Modeling." Pinterest, Medium. https://medium.com/pinterest-engineering/user-action-sequence-modeling-for-pinterest-ads-engagement-modeling-21139cab8f4e
25. Pinterest. "Improving feature interactions at Pinterest under industry constraints." Pinterest, CIKM 2024 workshop. https://arxiv.org/abs/2412.01985
26. Li, J., et al. "ML-DCN: Masked Low-Rank Deep Crossing Network Towards Scalable Ads Click-through Rate Prediction at Pinterest." Pinterest, 2026. https://arxiv.org/abs/2602.09194
27. InfoQ. "Snap ad ranking (MMoE and PLE for multiple conversion events)." 2022. https://www.infoq.com/news/2022/10/snap-ad-ranking/
28. Ktena, S. I., Tejani, A., Theis, L., et al. "Addressing Delayed Feedback for Continuous Training with Neural Networks in CTR Prediction." Twitter, RecSys 2019. https://arxiv.org/abs/1907.06558
29. Twitter Engineering. "Twitter's Recommendation Algorithm." Twitter, 2023. https://blog.x.com/engineering/en_us/topics/open-source/2023/twitter-recommendation-algorithm (Home Mixer: https://github.com/twitter/the-algorithm/blob/main/home-mixer/README.md)
30. Chapelle, O. "Modeling Delayed Feedback in Display Advertising." Criteo, KDD 2014. https://dl.acm.org/doi/10.1145/2623330.2623634
31. Yang, J.-Q., et al. "Capturing Delayed Feedback in Conversion Rate Prediction via Elapsed-Time Sampling." Alibaba, AAAI 2021. https://arxiv.org/abs/2012.03245
32. Gu, S., et al. "Real Negatives Matter: Continuous Training with Real Negatives for Delayed Feedback Modeling." Alibaba, KDD 2021. https://arxiv.org/abs/2104.14121
33. Ma, X., et al. "Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate." Alibaba, SIGIR 2018. https://arxiv.org/abs/1804.07931
34. Borisyuk, F., et al. "LiRank: Industrial Large Scale Ranking Models at LinkedIn." LinkedIn, KDD 2024. https://arxiv.org/abs/2402.06859
35. "Isotonic Layer: A Unified Framework for Recommendation Calibration and Debiasing." LinkedIn, 2026. https://arxiv.org/abs/2603.06589
36. "An Audio-centric Multi-task Learning Framework for Streaming Ads Targeting on Spotify." Spotify, 2025. https://arxiv.org/abs/2506.18735
37. "Ranking model calibration in recommender systems." RecSys ML Substack. https://recsysml.substack.com/p/ranking-model-calibration-in-recommender (see also parametric ranking calibration, https://arxiv.org/abs/2112.07428)
38. Liu, Z., et al. "Monolith: Real Time Recommendation System With Collisionless Embedding Table." ByteDance, 2022. https://arxiv.org/abs/2209.07663
39. Wu, D., Chen, X., Yang, X., et al. "Budget Constrained Bidding by Model-free Reinforcement Learning in Display Advertising." Alibaba, CIKM 2018. https://arxiv.org/abs/1802.08365
40. Jin, J., Song, C., Li, H., et al. "Real-Time Bidding with Multi-Agent Reinforcement Learning in Display Advertising." Alibaba, CIKM 2018. https://arxiv.org/abs/1802.09756
41. Edelman, B., Ostrovsky, M., Schwarz, M. "Internet Advertising and the Generalized Second-Price Auction: Selling Billions of Dollars Worth of Keywords." American Economic Review 97(1), 2007. https://doi.org/10.1257/aer.97.1.242
42. Varian, H. R. "Position Auctions." International Journal of Industrial Organization 25(6), 2007. https://doi.org/10.1016/j.ijindorg.2006.10.002
43. Vickrey, W. "Counterspeculation, Auctions, and Competitive Sealed Tenders." Journal of Finance 16(1), 1961. https://doi.org/10.1111/j.1540-6261.1961.tb02789.x
44. "EGA-V2: An End-to-end Generative Framework for Industrial Advertising." Meituan, 2025. https://arxiv.org/abs/2505.17549
45. "GPR: Towards a Generative Pre-trained One-Model Paradigm for Large-Scale Advertising Recommendation." Tencent, 2025. https://arxiv.org/abs/2511.10138
46. ByteDance. "TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders." ByteDance, 2026. https://arxiv.org/abs/2602.06563
47. ByteDance. "RankMixer: Scaling Up Ranking Models in Industrial Recommenders." ByteDance, 2025. https://arxiv.org/abs/2507.15551
48. Gilotte, A., Calauzènes, C., Nedelec, T., Abraham, A., Dollé, S. "Offline A/B Testing for Recommender Systems." Criteo, WSDM 2018. https://arxiv.org/abs/1801.07030
49. Zinkevich, M. "Rules of Machine Learning: Best Practices for ML Engineering." Google. https://developers.google.com/machine-learning/guides/rules-of-ml
50. Borisyuk, F., et al. "LiGNN: Graph Neural Networks at LinkedIn." LinkedIn, KDD 2024. https://arxiv.org/abs/2402.11139
51. El-Kishky, A., et al. "TwHIN: Embedding the Twitter Heterogeneous Information Network for Personalized Recommendation." Twitter, KDD 2022. https://arxiv.org/abs/2202.05387
