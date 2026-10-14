# 06. Design an Ad Click-Through-Rate Prediction and Ad Ranking System (with Auction and Calibration)

> Prompt variants: "Ad click prediction on social platforms" (ByteByteGo), "Ad prediction
> system" (Educative), "Ads click prediction / ads ranking" (alirezadir), "Ad Click
> Prediction" (Khang Pham), "Design the ads ranking system for a feed" (Meta-style phrasing).

## Company tags

| Company | Evidence | Source |
|---|---|---|
| Meta (Facebook / Instagram) | The textbook production system for this prompt (GBDT+LR, NE metric, calibration). Ads/feed ranking is a commonly reported Meta MLE design theme | [7] [8] [21] [22] |
| Google | Search ads CTR at scale (FTRL-Proximal LR, isotonic calibration layer, Ad Rank); DCN-V2 paper reports production use in Google rankers | [9] [23] [27] |
| Snap | Engineering write-up on Snapchat ad ranking (listed in [24]) | [12] [24] |
| LinkedIn | Deep-learning ads CTR model; ads allocation in feed via constrained optimisation | [13] [24] [25] |
| Twitter / X | Ads engagement with delayed feedback, and prediction of ad-request value | [4] [11] [14] [24] |
| Pinterest, Etsy, Wayfair | Lookalike ad targeting, real-time ads personalisation, ads bidding (all listed in [24]) | [15] [16] [17] [24] |

**Portals that cover it**

- ByteByteGo / Aminian & Xu, *ML System Design Interview*, chapter "Ad Click Prediction on Social Platforms" [1]
- Educative, *Grokking the Machine Learning Interview*, "Ad Prediction System" module [2]
- Educative / Khang Pham, *Machine Learning System Design*, plus the khangich GitHub "Ad Click Prediction" case and Adtech reading list [3] [4]
- alirezadir/Machine-Learning-Interviews (AIMLInterviews), `mlsd-ads-ranking.md` "Ads click prediction" [5] [6]
- Hello Interview (recsys pattern), Exponent / Aced, IGotAnOffer, Prepfully: generic ranking frameworks used for this prompt [18] [19] [20] [21]

## Requirements & scale

| Dimension | Value | Source / status |
|---|---|---|
| Business goal | Maximise long-term revenue = advertiser value + user experience (not raw CTR) | [1] [5] [22] |
| Functional | For each ad slot in a feed or search page, pick and price ads; predict P(click), optionally P(conv) | [5] |
| ML task | Pointwise LTR: binary classification on (user, ad, context) | [5] |
| Label | Click within t seconds of impression = 1, else 0; "hide/report ad" = negative signal | [5] |
| Users / ads | 1B+ DAU; 10M+ active ads, millions of advertisers | typical assumption |
| QPS | 100k-1M ad requests/s at peak (Khang Pham's worked case assumes 4k/s); ranker scores 100s-1000s ads per request | typical assumption; [4] |
| Latency | Ads pipeline p99 ~100-150 ms total, ranker ~10-30 ms | typical assumption; 150 ms in [4] |
| Base rate | CTR is low (often ~1-2% or less), so classes are very imbalanced | [4] [7] |
| Freshness | New ads and campaigns arrive constantly; stale models lose accuracy daily | [7] [9] |
| Hard constraints | Calibrated probabilities (they feed the auction and billing); budget pacing; privacy; ad policy | [7] [9] [22] |

## DIAGRAM 1: online serving path

**Diagram 1. Ad request to response: targeting retrieval, filtering, pCTR/pCVR ranking, auction and pricing, with feature store, model store and logging (synthesised from [1] [5] [7] [12] [22])**

```
 +-----------+  ad request (user, slot, ctx)  +-----------------------------------+
 | Client /  |------------------------------->|  Ad Server / Ads Mixer            |
 | Feed svc  |<-------------------------------|  (fan-out, timeouts, ~100ms SLA)  |
 +-----------+   winning ads + price + log id +----+------------------------------+
                                                  | 1
                                                  v
 +----------------------+  targeting index  +------------------------------------+
 | Ad index (inverted:  |------------------>| CANDIDATE GENERATION  (~10M -> 10k)|
 | geo, age, interests, |                   | targeting match + ANN user->ad emb |
 | lookalike, keywords) |                   +----------------+-------------------+
 +----------------------+                                    | 2
                                                             v
 +----------------------+  budget/pacing    +------------------------------------+
 | Budget & pacing svc  |------------------>| FILTERING          (10k -> ~1-2k)  |
 | (spend, freq caps)   |                   | policy, freq cap, budget left,     |
 +----------------------+                   | blocked/hidden advertisers         |
                                            +----------------+-------------------+
 +----------------------+  online feats                      | 3
 | FEATURE STORE        |------------+                       v
 | batch: ad/adv stats  |            |      +------------------------------------+
 | stream: last-N clicks|            +----->| LIGHT RANKER (two-tower / small    |
 | counts 1h/1d/7d      |            |      | DNN)                (~2k -> ~200)  |
 +----------------------+            |      +----------------+-------------------+
                                     |                       | 4
 +----------------------+  weights   |                       v
 | MODEL STORE /        |------------+----->+------------------------------------+
 | registry (pCTR, pCVR,|                   | HEAVY RANKER  pCTR, pCVR (DCN /    |
 | calibrators)         |------------------>| DeepFM / MTL)  -> CALIBRATION      |
 +----------------------+  calib maps       +----------------+-------------------+
                                                             | 5 calibrated p
                                                             v
 +----------------------+                   +------------------------------------+
 | Advertiser bids /    |------------------>| AUCTION + RE-RANK                  |
 | objectives (CPC,CPM, |   bid, goal       | eCPM = bid x pCTR [x pCVR]         |
 | oCPM, target CPA)    |                   | total = eCPM + quality - penalty   |
 +----------------------+                   | GSP / VCG price, reserve, ad load, |
                                            | diversity, slot allocation         |
                                            +----------------+-------------------+
                                                             | 6
                                                             v
 +------------------------------------------------------------------------------+
 | LOGGING: impression id, features-at-serve-time, pCTR, bid, price, position  |
 |  --> Kafka --> online joiner (with clicks/conversions) --> training data     |
 +------------------------------------------------------------------------------+
```

**Funnel numbers (typical assumption)**

| Stage | In -> out | Model | Budget |
|---|---|---|---|
| Targeting + retrieval | 10M -> 10k | Inverted index + ANN embeddings | 10-20 ms |
| Filtering | 10k -> 2k | Rules (budget, frequency cap, policy) | 2-5 ms |
| Light ranking | 2k -> 200 | Two-tower / small GBDT/DNN | 5-10 ms |
| Heavy ranking + calibration | 200 -> 200 scored | DCN-V2 / DeepFM / MTL DNN | 10-30 ms |
| Auction + re-rank | 200 -> 1-5 slots | eCPM, GSP/VCG, business rules | 1-5 ms |

## DIAGRAM 2: offline training and data pipeline

**Diagram 2. Logging, online joining with delayed feedback, negative downsampling, training, calibration fit, evaluation and deployment (synthesised from [5] [7] [9] [11])**

```
 +-------------+   +--------------+   +-----------------+
 | Impression  |   | Click events |   | Conversions     |
 | logs        |   | (seconds)    |   | (hours-days,    |
 | +features   |   |              |   |  pixel/SDK/CAPI)|
 +------+------+   +------+-------+   +--------+--------+
        |                 |                    |
        v                 v                    v
 +--------------------------------------------------------------+
 | ONLINE JOINER (stream): join on impression id, wait window t |
 | click in t -> label 1; window expires -> label 0             |
 | late conversions -> correction / delayed-feedback model      |
 +-----------------------------+--------------------------------+
                               |  labelled examples
                               v
 +-----------------------+   +--------------------------------------+
 | Feature backfill /    |-->| SAMPLING: keep all positives,        |
 | point-in-time joins   |   | downsample negatives at rate w       |
 | (no future leakage)   |   | (record w for later recalibration)   |
 +-----------------------+   +-------------------+------------------+
                                                 |
                                                 v
 +----------------------------------------------------------------+
 | TRAINING                                                       |
 |  a) daily/weekly full retrain (big DNN on GPU)                 |
 |  b) online / incremental updates every minutes-hours           |
 |     (FTRL for LR; warm-start fine-tune for DNN)                |
 +-----------------------------+----------------------------------+
                               |
                               v
 +---------------------------+   +-----------------------------------+
 | CALIBRATION FIT           |-->| OFFLINE EVAL on next-day holdout  |
 | undo downsampling, then   |   | NE / log loss, AUC, calibration   |
 | Platt / isotonic per slice|   | ratio by segment, sliced by       |
 +---------------------------+   | new ads, advertiser, country      |
                                 +----------------+------------------+
                                                  | pass gates?
                                       no <-------+-------> yes
                                       |                   |
                                       v                   v
                                 +-----------+   +---------------------+
                                 | debug /   |   | MODEL REGISTRY      |
                                 | rollback  |   | -> shadow -> canary |
                                 +-----------+   | -> A/B -> ramp 100% |
                                                 +----------+----------+
                                                            |
           feedback loop: served ads become next day's data |
 <----------------------------------------------------------+
```

## DIAGRAM 3: model architecture

**Diagram 3. Multi-task CTR/CVR ranker: embeddings, explicit cross network plus deep tower, shared bottom, task heads, calibration and eCPM (DCN-V2 / DeepFM / ESMM style; [1] [5] [26] [27] [30])**

```
  INPUTS
 +------------+ +------------+ +--------------+ +-------------+ +-------------+
 | User IDs,  | | Ad / adv / | | Context:     | | User seq:   | | Dense:      |
 | demo, geo  | | campaign   | | slot, device,| | last N ads  | | hist CTR,   |
 | (sparse)   | | IDs, categ.| | time, query  | | clicked     | | counts 1/7d |
 +-----+------+ +-----+------+ +------+-------+ +------+------+ +------+------+
       |              |               |                |               |
       v              v               v                v               v
 +-----------------------------------------------+ +------------+ +-----------+
 | Embedding tables (hashed IDs, dim 16-64)      | | Target     | | Normalise |
 | creative: image/text encoder -> vector        | | attention  | | / log /   |
 +----------------------+------------------------+ | (DIN-style)| | bucketize |
                        |                          +-----+------+ +-----+-----+
                        +------------------+-------------+--------------+
                                           | concat x0
                     +---------------------+---------------------+
                     v                                           v
        +--------------------------+               +--------------------------+
        | CROSS NETWORK (explicit) |               | DEEP NETWORK (implicit)  |
        | x1 = x0*(W.x + b) + x    |               | MLP 1024-512-256, ReLU   |
        | or FM pairwise <vi,vj>   |               |                          |
        +------------+-------------+               +------------+-------------+
                     +---------------------+---------------------+
                                           v
                          +-----------------------------------+
                          | shared bottom / MMoE experts      |
                          +----+-------------------------+----+
                               |                         |
                               v                         v
                        +-------------+          +---------------+
                        | pCTR head   |          | pCVR head     |
                        | sigmoid     |          | (ESMM: no     |
                        | BCE(click)  |          | direct loss)  |
                        +------+------+          +-------+-------+
                               |                         |
                               +--------------+----------+
                                              v
                        +-------------------------------------------+
                        | pCTCVR = pCTR x pCVR -> BCE(click & conv) |
                        | over ALL impressions (entire space)       |
                        +---------------------+---------------------+
                                              v
                        +-------------------------------------------+
                        | CALIBRATE pCTR, pCVR: q = p/(p+(1-p)/w)   |
                        | then isotonic / Platt per segment         |
                        +---------------------+---------------------+
                                              v
                        +-------------------------------------------+
                        | eCPM = bid x pCTR (CPC)                   |
                        |      = bid x pCTR x pCVR (oCPM / CPA)     |
                        +-------------------------------------------+
```

## DIAGRAM 4: the real system (Facebook, He et al. 2014)

**Diagram 4. Facebook's hybrid GBDT + logistic regression CTR model with online joiner and online learning [7] [8]**

```
 +--------------------+     raw features x (continuous + categorical)
 | Ad impression +    |--------------------------------------------+
 | user/ad/context    |                                            |
 +--------------------+                                            v
                         +-----------------------------------------------+
                         | Boosted trees (batch, daily / every few days) |
                         |  tree 1    tree 2    ...    tree T            |
                         |   leaf k1   leaf k2          leaf kT          |
                         +----------------------+------------------------+
                                                | one-hot leaf indices
                                                v   (learned feature transform)
                         +-----------------------------------------------+
                         | LR, near-real-time SGD, per-coordinate LR     |
                         | p = sigmoid(w . leaf_onehot + b)              |
                         +-----------+------------------------+----------+
                                     ^                        |
                                     | updated LR weights     | p -> recalibrate
                                     |                        v   (if downsampled)
 +--------------------+  labels  +---+----------------+  +-----------------+
 | ONLINE JOINER      |--------->| Online trainer     |  | Auction (eCPM)  |
 | impressions+clicks |          | (Scribe stream)    |  +-----------------+
 | waiting window t   |          +--------------------+
 +--------------------+
 Findings: hybrid beat either alone (trees cut NE >3.4%); weekly -> daily
 training cuts NE ~1%; best negative-downsampling rate 0.025 (then recalibrate);
 NE + calibration are the offline metrics; historical features dominate top 10.
```

## Data & labels

| Item | Choice | Notes / source |
|---|---|---|
| Positive | Click within t s of impression | [5] [7] |
| Negative | Impression with no click by end of window; hide/report as strong negative | [5] |
| Conversion label | Purchase/install after click, arrives hours-days later | delayed feedback [11] |
| Window t | Trade-off: short = fresher but more false negatives; long = accurate but stale | [7] [11] |
| Imbalance | Downsample negatives at rate w; recalibrate q = p / (p + (1-p)/w) | [7] |
| Position bias | Log position; use as training feature and fix it at serve time, or use a shallow position tower | typical practice |
| Leakage | Point-in-time feature joins; split train/test by time, not randomly | [5] |

## Features

| Group | Examples | Encoding |
|---|---|---|
| Ad | ad id, advertiser id, campaign, category, creative image/text/video | Embeddings; pretrained encoders (e.g. SimCLR-style) [5] |
| Ad engagement | impressions, clicks, CTR at ad/advertiser/campaign level, 1h/1d/7d | Dense, log-scaled, smoothed for new ads |
| User | demographics, location, device, language | Embeddings / one-hot |
| User history | historical CTR, last N clicked ads/categories | Sequence + attention |
| User-ad cross | user x ad-category CTR, previous impressions of this ad (fatigue) | Crosses or learned by DCN/FM |
| Context | slot/position, time of day, surface (feed/stories/search), query | One-hot / embeddings |
| Bid / objective | bid type and objective (not usually a CTR feature; used in auction) | Kept out of pCTR to avoid bias |

## Model choices (baseline -> advanced)

| Model | Strengths | Weaknesses | Source |
|---|---|---|---|
| Logistic regression (+ hashing) | Fast, calibrated-ish, easy online learning (FTRL) | No interactions without manual crosses | [5] [9] |
| LR + manual feature crosses | Captures key pairwise interactions | Manual, combinatorial, sparse | [1] [5] |
| GBDT | Strong on dense features, interpretable | Poor with huge sparse IDs; no continual training | [5] |
| GBDT + LR | Trees as feature transform; LR learns online | Two systems to maintain | [7] |
| Factorization Machine (FM/FFM) | Learned pairwise interactions via embeddings | Only 2nd-order | [1] [5] |
| Wide & Deep / DeepFM | Memorisation + generalisation, no manual crosses | Bigger, harder to calibrate | [26] [28] |
| DCN / DCN-V2 | Explicit bounded-degree crosses + deep tower; production use at Google | Tuning cross depth | [27] |
| DLRM / DIN / MMoE multi-task | Huge embedding tables, user-sequence attention, joint pCTR/pCVR | Infra cost, GPU serving | [29] [30] |

## Loss

| Loss | Use |
|---|---|
| Binary cross-entropy (log loss) | Default for pCTR; directly optimises calibrated probability [1] [5] |
| Weighted BCE / importance weights | Correct for negative downsampling [7] |
| Multi-task sum: L = BCE(click, pCTR) + BCE(click & conv, pCTR x pCVR) over all impressions | ESMM: pCVR learned implicitly, avoiding sample-selection bias and click sparsity [30] |
| Pairwise / listwise (optional) | Improves ordering but breaks calibration; usually avoided as the only loss because the auction needs absolute probabilities |
| L1/L2 + FTRL regularisation | Sparsity for huge hashed feature spaces [9] |

## Calibration

| Topic | Detail | Source |
|---|---|---|
| Why | eCPM = bid x pCTR: a 10% overestimate means overcharging/mis-ranking advertisers | [7] [9] [22] |
| Calibration ratio | sum(predicted pCTR) / sum(actual clicks); target 1.0 overall and per slice | [7] |
| Downsampling fix | q = p / (p + (1-p)/w), w = negative sampling rate | [7] |
| Post-hoc methods | Google: fit tau(p) = gamma x p^kappa by Poisson regression, or piecewise-linear isotonic map per data slice; Platt / temperature scaling for DNNs | [9] [31] |
| Monitoring | Calibration plots by bucket, per advertiser; alert when ratio drifts beyond about +-5% | typical assumption |
| Training-time subsampling | Google keeps all clicked queries and a fraction r of others, weighting kept negatives by 1/r so the loss stays unbiased | [9] |

## Auction and ranking score

| Concept | Formula / rule | Source |
|---|---|---|
| Expected value (CPC bid) | eCPM = 1000 x bid_CPC x pCTR | standard practice |
| Optimised CPM / CPA | value = bid_CPA x pCTR x pCVR | [12] |
| Meta "total value" | advertiser bid x estimated action rate + ad quality | [22] |
| Google "Ad Rank" | bid, ad quality (expected CTR, relevance, landing page), thresholds, context | [23] |
| Pricing | GSP: pay the minimum needed to keep the slot; VCG: pay the externality imposed; plus reserve prices | standard practice |
| Re-rank / policy | Ad load limits, advertiser diversity, frequency caps, pacing multiplier on bid | [1] [5] |
| Ads allocation in feed | Choose ad slots in an organic feed by maximising revenue subject to user-engagement constraints (LinkedIn) | [25] |
| Autobidding / pacing | Platform adjusts bids per auction to hit advertiser budget and CPA/ROAS goals | [17] |

## Offline metrics

| Metric | Why | Source |
|---|---|---|
| Normalised entropy (NE) / normalised cross-entropy | Log loss divided by the entropy of the background CTR; lower is better | [1] [7] |
| Log loss | Measures probability quality (needed by the auction) | [5] |
| ROC-AUC / PR-AUC | Ranking quality; insensitive to calibration, so do not use alone | [7] |
| Calibration ratio, reliability diagram | Absolute probability correctness | [7] |
| Sliced metrics | New ads, new users, country, surface, advertiser size | typical practice |

## Online metrics / A-B

| Metric | Type |
|---|---|
| Revenue per mille (RPM) / revenue lift | Primary business metric [5] |
| CTR, conversion rate | Engagement [5] |
| Advertiser ROI / cost per action, advertiser retention | Advertiser health |
| Hide/report rate, organic engagement, session time | User-experience guardrails [5] |
| Calibration in production, auction-price changes | Safety |
| A/B design | Randomise users; watch budget-split interference between arms (advertiser budgets are shared), so use budget-split or advertiser-side experiments |

## Serving & latency budget

| Component | Budget (p99) | Technique |
|---|---|---|
| Feature fetch | 5-10 ms | In-memory KV store, batched multi-get, cached ad features |
| Retrieval | 10-20 ms | Inverted targeting index + ANN |
| Light ranker | 5-10 ms | Two-tower dot product, ad embeddings precomputed |
| Heavy ranker | 10-30 ms | GPU/CPU batch inference, quantised embeddings, model distillation |
| Auction + logging | <5 ms | In-process; async log write |
| Total | ~100 ms | Timeouts with fallback to a cached or simpler model |

## Monitoring

| What | Signal |
|---|---|
| Prediction drift | Mean pCTR vs observed CTR per hour/slice (calibration ratio) |
| Feature drift / missing | Null rate, distribution shift (PSI) per feature; online/offline skew check |
| Data pipeline | Joiner lag, label delay, event loss |
| Business | Revenue, spend pacing, advertiser complaints, auction clearing price |
| Model freshness | Hours since last successful update; NE on latest hour |

## What the production papers add (often missed in portal answers)

| Item | Detail | Source |
|---|---|---|
| Online joiner click coverage | Too short a wait window loses clicks and biases empirical CTR low; too long delays data and costs memory | [7] |
| Probabilistic feature inclusion | Add a new sparse feature only after it is seen n times (counting Bloom filter) or with probability p (Poisson) to save RAM | [9] |
| Progressive validation | Score each example before training on it in online learning; no separate holdout needed | [9] |
| Confidence estimates | Per-prediction uncertainty to drive exploration and guard bids on rare ads | [9] |
| Automated feature management | Feature metadata, owners and automatic alerts on signal changes across many models | [9] |
| Model size vs trees | Fewer boosted trees keep latency and memory down at small NE cost | [7] |

## Failure modes

| Failure | Mitigation |
|---|---|
| Mis-calibration after downsampling or a model swap | Recalibration step; calibration gate in deployment [7] |
| Cold-start ads (no history) | Content embeddings, advertiser priors, exploration budget (bandits / boosted bids) |
| Delayed conversions labelled as negative | Delayed-feedback model, longer windows, label correction [11] |
| Training-serving skew | Log features at serve time; reuse those logs for training [7] |
| Feedback loop / selection bias (only shown ads get labels) | Exploration traffic, inverse propensity weighting |
| Ad fatigue | Frequency caps; impression-count features [5] |
| Click fraud / bots | Invalid-traffic filters before labelling |
| Catastrophic forgetting with continual training | Replay buffer, periodic full retrain [5] |
| Privacy changes (lost cross-app signals) | On-platform signals, aggregated conversion APIs, modelled conversions |

## Interviewer follow-ups

- Why must pCTR be calibrated, and how would you measure and fix miscalibration? [7]
- You downsampled negatives to 1%. How do you correct the predicted probabilities? [7]
- Why is AUC not enough as an offline metric for ads? What is normalised entropy? [7]
- How do you pick the click attribution window t, and how do you handle delayed conversions? [11]
- How does the auction combine bid, pCTR, pCVR and quality? GSP vs VCG? [22] [23]
- How do you rank ads for a CPA (conversion) advertiser vs a CPC advertiser?
- Walk through LR -> FM -> DeepFM -> DCN. Why do plain MLPs struggle with sparse crosses? [1] [5]
- How fresh must the model be? Online learning vs daily retrain trade-offs? [7] [9]
- How do you handle a new ad or new advertiser with no history (cold start)?
- How do you deal with position bias in click logs?
- How do you A/B test when the treatment and control arms share advertiser budgets?
- How would you keep the heavy ranker within 30 ms for 1,000 candidates?
- How do you trade off ad revenue against user experience (ad load, hides)?
- What happens to the system when third-party tracking signals disappear?
- How do you prevent the model from learning from fraudulent clicks?
- How long should the online joiner wait before labelling a no-click, and what does that do to click coverage? [7]
- How would you add millions of new sparse features without blowing up model RAM? [9]
- Where should ads go in an organic feed, and how many? [25]

## Portal comparison

| Portal | Framing | Model path | Distinctive emphasis |
|---|---|---|---|
| ByteByteGo (Aminian & Xu) [1] | "Ad click prediction on social platforms", pointwise LTR | LR -> feature crossing -> GBDT -> GBDT+LR -> NN -> DCN / FM / DeepFM | NCE metric, continual learning, targeting-based candidate generation |
| alirezadir AIMLInterviews [5] | Follows [1] closely, in outline form | LR baseline, then DCN and DeepFM | Fatigue period, hide-rate metric, catastrophic forgetting, leakage; no auction section |
| Educative Grokking ML Interview [2] | "Ad prediction system": ad selection funnel + engagement prediction | LR / GBDT -> NN, online learning | Funnel stages and incremental training (from course outline) |
| Khang Pham (Educative + GitHub) [3] [4] | "Ad Click Prediction" case study | Paper-driven: Facebook GBDT+LR, Google FTRL | Adtech reading list: delayed feedback, entity embeddings |
| Hello Interview / Exponent / IGotAnOffer [18] [19] [20] | Generic ranking or recsys template applied to ads | Two-stage retrieve + rank | Delivery framework and time-boxing rather than ads detail |
| Company papers [7] [9] [12] | Production reality | GBDT+LR, FTRL-LR, DCN, MMoE | Calibration, NE, freshness, auction value |

## References

1. A. Aminian & A. Xu, "Ad Click Prediction on Social Platforms" (Machine Learning System Design Interview, ByteByteGo), bytebytego.com, 2023. https://bytebytego.com/courses/machine-learning-system-design-interview/ad-click-prediction-on-social-platforms
2. Educative, "Grokking the Machine Learning Interview" (Ad Prediction System module), educative.io, n.d. https://www.educative.io/courses/grokking-the-machine-learning-interview
3. Educative / K. Pham, "Machine Learning System Design" (course), educative.io, 2021. https://educative.io/courses/machine-learning-system-design/introduction
4. K. Pham, "machine-learning-interview" (Ad Click Prediction case; Adtech list), GitHub, 2021. https://github.com/khangich/machine-learning-interview
5. A. Dirafzoon, "Ads click prediction" (mlsd-ads-ranking.md), alirezadir/AIMLInterviews, GitHub, 2023. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/mlsd-ads-ranking.md
6. alirezadir, "ML System Design" template (lists ads click prediction and ads serving cases), GitHub, 2025. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/ml-system-design.md
7. X. He et al., "Practical Lessons from Predicting Clicks on Ads at Facebook", ADKDD, 2014. https://quinonero.net/Publications/predicting-clicks-facebook.pdf
8. Facebook Research, same paper (PDF mirror), research.fb.com, 2014. https://research.fb.com/wp-content/uploads/2016/11/practical-lessons-from-predicting-clicks-on-ads-at-facebook.pdf
9. H. B. McMahan et al., "Ad Click Prediction: a View from the Trenches", KDD / Google, 2013. https://storage.googleapis.com/pub-tools-public-publication-data/pdf/41159.pdf
10. alirezadir, "ML at Companies" (ads and auction resources), GitHub, 2025. https://github.com/alirezadir/AIMLInterviews/blob/main/src/MLSD/ml-companies.md
11. Twitter Engineering, "Improving engagement on digital ads with delayed feedback", blog.twitter.com, 2019. https://blog.twitter.com/engineering/en_us/topics/insights/2019/improving-engagement-on-digital-ads-with-delayed-feedback.html
12. Snap Engineering, "Machine Learning for Snapchat Ad Ranking", eng.snap.com, 2022. https://eng.snap.com/machine-learning-snap-ad-ranking
13. LinkedIn Engineering, "Challenges and practical lessons from building a deep-learning-based ads CTR prediction model", engineering.linkedin.com, 2022. https://engineering.linkedin.com/blog/2022/challenges-and-practical-lessons-from-building-a-deep-learning-b
14. Twitter Engineering, "Using machine learning to predict the value of ad requests", blog.twitter.com, 2020. https://blog.twitter.com/engineering/en_us/topics/insights/2020/using-machine-learning-to-predict-the-value-of-ad-requests
15. Pinterest Engineering, "The machine learning behind delivering relevant ads", Medium, 2021. https://medium.com/pinterest-engineering/the-machine-learning-behind-delivering-relevant-ads-8987fc5ba1c0
16. Etsy, "Leveraging Real-Time User Actions to Personalize Etsy Ads", arXiv, 2023. https://arxiv.org/pdf/2302.01255.pdf
17. Wayfair Tech, "Evolution of Ads Bidding at Wayfair", aboutwayfair.com, 2021. https://www.aboutwayfair.com/careers/tech-blog/evolution-of-ads-bidding-at-wayfair
18. Hello Interview, "Recommendation Systems (ML System Design pattern)", hellointerview.com, 2025. https://www.hellointerview.com/learn/ml-system-design/patterns/recommendation-systems
19. Aced (formerly Exponent), "Machine Learning System Design Interview (2026 Guide)", aced.io, 2026. https://www.aced.io/blog/machine-learning-system-design-interview-guide
20. IGotAnOffer, "Machine Learning (ML) System Design Interview (examples, answers, prep)", igotanoffer.com, 2025. https://igotanoffer.com/en/advice/machine-learning-system-design-interview
21. Prepfully, "Meta Machine Learning Engineer interview questions", prepfully.com, 2025. https://prepfully.com/interview-questions/meta/machine-learning-engineer?page=8
22. Meta Business Help Center, "About ad auctions", facebook.com, n.d. https://www.facebook.com/business/help/430291176997542
23. Google Ads Help, "About Ad Rank", support.google.com, n.d. https://support.google.com/google-ads/answer/1722122
24. Engineer1999, "A Curated List of ML System Design Case Studies" (ads section), GitHub, 2024. https://github.com/Engineer1999/A-Curated-List-of-ML-System-Design-Case-Studies
25. J. Yan et al., "Ads Allocation in Feed via Constrained Optimization" (LinkedIn), KDD, 2020. https://dl.acm.org/doi/abs/10.1145/3394486.3403391
26. H. Guo et al., "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction", arXiv, 2017. https://arxiv.org/abs/1703.04247
27. R. Wang et al., "DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems", arXiv, 2020. https://arxiv.org/abs/2008.13535
28. H.-T. Cheng et al., "Wide & Deep Learning for Recommender Systems", arXiv, 2016. https://arxiv.org/abs/1606.07792
29. M. Naumov et al., "Deep Learning Recommendation Model for Personalization and Recommendation Systems" (DLRM), arXiv, 2019. https://arxiv.org/abs/1906.00091
30. X. Ma et al., "Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate" (ESMM), arXiv, 2018. https://arxiv.org/abs/1804.07931
31. C. Guo et al., "On Calibration of Modern Neural Networks", arXiv, 2017. https://arxiv.org/abs/1706.04599

> Research note: in this review, non-GitHub hosts were blocked by the network proxy and the
> web-search budget was exhausted. [4] [5] [6] [10] [24] were re-read from GitHub; the texts
> of [7] and [9] were checked from PDF mirrors, confirming NE, calibration ratio, online
> joiner, downsampling recalibration, Poisson/isotonic calibration and subsampling weights.
> URLs for [7]-[9], [11]-[17] and [31] are confirmed as links in [4] [10] [24]; their pages
> were not loaded. [1]-[3], [18]-[23] and [26]-[30] are cited from prior knowledge. An
> unsupported TikTok/Amazon/Reddit/Uber tag row was removed. Scale numbers marked
> "typical assumption" are not sourced.
