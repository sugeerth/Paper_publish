# 04. Multi-Task and Multi-Objective Ranking: From Shared-Bottom to Value Models

## Overview

Almost every large recommender ranks by more than one thing. A feed item can be clicked, liked, shared, replied to, watched to the end, hidden, or reported. A product can be clicked, carted, bought, or returned. An ad can be clicked, can convert, and can earn revenue. Production systems split this into two separate problems, and it helps to keep them apart:

1. **Multi-task learning (MTL), the modeling problem.** How to train one network, or a few, that predicts many behaviors (pClick, pLike, pShare, pHide, expected watch time, ...) well. Data volumes differ across tasks, tasks conflict, and some labels are only seen after another event happens. Shared-bottom, MMoE, PLE, ESMM and their successors live here.
2. **Multi-objective optimization (MOO), the decision problem.** How to turn those predictions into one ordering that serves the business and the user over the long run. This is the "value model" or "ensemble sort" or "utility" layer: a weighted sum or product of calibrated probabilities whose weights get tuned by hand, by Bayesian optimization, or more and more by reinforcement learning.

```
                 +--------------------------------------------+
  features  ---> |  multi-task ranker (shared-bottom / MMoE / |
  (user, item,   |  PLE / MaskNet / transformer)              |
   context)      +--------------------------------------------+
                   | pClick | pLike | pShare | pHide | E[dwell] ...
                   v        v       v        v       v
                 +--------------------------------------------+
                 | calibration (per task: Platt / temperature |
                 | / isotonic / learned isotonic layer)       |
                 +--------------------------------------------+
                                  |
                                  v
                 +--------------------------------------------+
                 | value model / fusion:                      |
                 |   score = sum_i w_i * p_i   (linear)       |
                 |   score = prod_i p_i ^ w_i  (log-linear)   |
                 |   w = f(user, request)      (personalized) |
                 +--------------------------------------------+
                                  |
                                  v
                 diversity / out-of-network / freshness
                 multipliers, DPP rerank, integrity demotions
```

This chapter goes through the canonical MTL architectures, then the value-model layer as it is publicly documented at X/Twitter, Meta (Facebook Feed, Instagram Explore, ads via Lattice), YouTube, LinkedIn, Pinterest and Kuaishou. It ends with calibration, loss balancing, Pareto tuning and lessons for designers.

---

## 1. Multi-task learning architectures

### 1.1 Shared-bottom (hard parameter sharing)

**Design.** All tasks share one embedding layer and a bottom MLP. Each task then gets its own small "tower" with a sigmoid or regression head. The loss is a weighted sum of per-task losses.

```
  input embeddings -> [shared MLP] -> h
                                    |-> tower_click -> pClick
                                    |-> tower_like  -> pLike
                                    |-> tower_share -> pShare
```

**Strengths.** It is the cheapest option, and sparse tasks get regularized by dense ones. It is still the default in many production rankers. Pinterest describes its Homefeed ranker (Pinnability) as a point-wise multi-task model: a wide-and-deep stack feeds feature-interaction layers and MLPs that produce per-action scores such as click, repin and hide [33].

**Weakness.** When tasks are weakly correlated or in conflict, the shared representation causes *negative transfer*: improving one head hurts another. The MMoE paper used synthetic tasks with controlled Pearson correlation and showed that shared-bottom quality drops as task correlation falls [1].

### 1.2 MMoE: Multi-gate Mixture-of-Experts (Google, KDD 2018)

**Problem.** Learn task relationships from data instead of fixing how much tasks share.

**Architecture.** The single shared bottom is replaced by *n* expert networks shared by all tasks. Each task k has its own gating network g_k(x), a linear transformation followed by a softmax over experts. Task k's tower input is the sum over experts of g_k(x)_i * expert_i(x) [1].

```
                    +----------+  +----------+       +----------+
  x ------------->  | expert 1 |  | expert 2 |  ...  | expert n |
  |                 +----------+  +----------+       +----------+
  |                        \            |            /
  |-> gate_A (softmax) ---> weighted sum_A --> tower_A --> y_A
  |-> gate_B (softmax) ---> weighted sum_B --> tower_B --> y_B
```

**Findings.**
- On the synthetic data, a one-gate MoE (OMoE) and MMoE perform about the same when tasks are highly correlated. As correlation drops, MMoE is consistently better [1].
- The authors report that the model was trained on hundreds of billions of examples in a large-scale Google content recommendation system and improved online metrics in live experiments [1].

**Known pathology: gate polarization / expert collapse.** Later industrial work reports that MMoE gates can collapse onto a few experts. A Kuaishou paper notes that under ReLU some experts output mostly zeros, so gates cannot weight experts fairly. It also notes that "shared" experts can degrade into serving only one task [9].

### 1.3 YouTube "What Video to Watch Next" (RecSys 2019)

**Problem.** YouTube ranks several hundred candidates against *multiple competing objectives* and has to deal with *implicit selection bias*, especially position bias, in the feedback it logs [2].

**Objectives.** The paper groups them in two families [2][3]:
- *Engagement objectives*: clicks and the degree of engagement (for example watch time).
- *Satisfaction objectives*: liking a video, leaving a rating on the recommendation, and so on.

**Architecture.** The paper extends Wide & Deep:
- **Main tower (deep, MMoE).** Soft parameter sharing through MMoE, so each objective can choose which experts to share [2]. The paper compares shared-bottom with MMoE at 4 and 8 experts and reports that, at the same model complexity, MMoE significantly improves both engagement and satisfaction metrics (per secondary summaries of the paper's tables).
- **Shallow tower (wide).** A small model that takes bias-related inputs such as the position where the item was shown and the device. It outputs a bias logit that is *added* to the main model's logit during training. At serving time the position input is dropped, so the main tower learns position-debiased utility [2][3]. The KDnuggets write-up reports that the position feature gets 10% dropout during training, so the model does not lean on it too heavily [3]. Treat that number as secondary.

```
  user/video/context features --> [MMoE: experts + per-objective gates]
                                        |               |
                                  engagement towers  satisfaction towers
                                        |               |
  position, device -> [shallow tower] --+ (bias logit added at train only)
                                        v
              per-objective predictions --> weighted combination --> rank
```

**Fusion.** Each objective's prediction is combined into one ranking score by a weighted combination whose weights are *manually tuned* for the best online engagement/satisfaction trade-off. That description comes from secondary summaries of the paper [3]. This "MTL model plus hand-tuned fusion" pattern shows up again and again below.

**Lessons.** (a) Debiasing belongs in the model structure: a separate additive tower absorbs position effects. (b) MMoE gains held at equal compute, so the gain comes from better sharing, not more parameters.

### 1.4 PLE: Progressive Layered Extraction (Tencent, RecSys 2020 Best Long Paper)

**Problem.** Tencent's video recommender saw a *seesaw phenomenon*: MMoE and similar models improved one task at the expense of another, especially when tasks had complex or competing correlations [4].

**Architecture.**
- **CGC (Customized Gate Control).** Each task gets its own *task-specific experts* plus a pool of *shared experts*. Task k's gate mixes only its own experts and the shared ones. The paper removes connections from a task's tower to other tasks' specific experts [4].
- **PLE.** Several CGC layers are stacked. Intermediate layers also have a *shared gate* that mixes all experts to feed the next level's shared experts. Shared and task-specific knowledge are separated step by step [4].

```
  Level 1:  [A-experts] [shared-experts] [B-experts]
              gate_A      gate_shared      gate_B
                |              |              |
  Level 2:  [A-experts] [shared-experts] [B-experts]
              gate_A                       gate_B
                |                             |
             tower_A -> VTR (AUC)          tower_B -> VCR (MSE)
```

**Tasks and data.** The main experiments use a Tencent video dataset of about 1 billion samples with two tasks: VTR, a binary "effective view" classification measured by AUC, and VCR, a view-completion-ratio regression measured by MSE [4].

**Results.** MMoE beat single-task models, but its VCR gain was tiny (about +0.0001 MSE improvement). CGC improved VTR more than VCR, and PLE improved both significantly over MMoE [4]. Online, PLE raised view count by 2.23% and watch time by 1.84% over the earlier SOTA MTL model, and it was deployed in Tencent's video recommender [4].

**Lesson.** Explicit task-specific capacity guards against negative transfer. Shared capacity still allows positive transfer, and stacking the levels lets the separation happen gradually.

### 1.5 HoME: Hierarchy of Multi-Gate Experts (Kuaishou, KDD 2025)

**Problem.** In production at Kuaishou, MMoE-style models showed three anomalies [9]:
- *Expert collapse*: experts whose ReLU outputs are mostly zero. Gates cannot balance them.
- *Expert degradation*: shared experts that end up serving a single task.
- *Expert underfitting*: sparse tasks lean on shared experts, which dense tasks' gradients dominate, and neglect their own experts.

**Architecture.** Experts are arranged in a hierarchy: a global shared level, then task-group levels (for example passive watch-time tasks vs. interaction tasks), then task-specific experts, with cascaded gates. The paper also adds normalization to expert outputs and feature-gating to fight collapse [9][44].

**Scale.** The paper reports deployment on Kuaishou short-video services serving about 400 million daily users [9].

**Lesson.** At very large scale, getting MoE-style MTL to work well needs normalization and grouping by task semantics. Gates alone are not enough.

### 1.6 STEM: Shared and Task-specific EMbeddings (Tencent, AAAI 2024)

**Problem.** Negative transfer is worst on samples where users give *comparable* positive feedback across tasks. The STEM authors trace this to the single shared embedding table, which gives every user and item one universal embedding [10].

**Design.** STEM-Net keeps one shared embedding table *plus* a separate embedding table per task. Experts are built on shared or task-specific embeddings. An "All Forward Task-specific Backward" gate lets every expert contribute forward, while stop-gradient keeps one task's gradients out of other tasks' embeddings [10].

**Lesson.** Negative transfer is not only a tower/expert problem. In ID-heavy recommenders the embedding table is where most parameters sit, and it is also where most interference happens.

### 1.7 Sequential-dependency models: ESMM and successors

#### ESMM: Entire Space Multi-Task Model (Alibaba, SIGIR 2018)

**Problem.** Conventional CVR (post-click conversion) models train only on clicked impressions but score *all* impressions, which is *sample selection bias*. Conversions are also very sparse. In Taobao data the CVR task had only about 4% as many samples as the CTR task [5].

**Key identity.** Over the impression space:

```
  pCTCVR = P(click=1, conv=1 | x) = pCTR(x) * pCVR(x)
```

**Architecture.** Two towers, CTR and CVR, share the embedding layer. Supervision goes on pCTR (click labels on all impressions) and on pCTCVR = pCTR * pCVR (click-and-convert labels on all impressions). The CVR tower is never directly supervised, so it is trained implicitly over the entire space. The model uses the product form instead of dividing pCTCVR by pCTR, because division by small CTR values is numerically unstable [5].

```
  impression features
        |
   [shared embeddings]
     /            \
  [CTR tower]   [CVR tower]
     |               |
   pCTR ----(x)---- pCVR
     |         |
  loss(click)  pCTCVR -> loss(click & conversion)
```

**Scale and results.** The full Taobao dataset has 8.9 billion samples with sequential click and conversion labels. A 1% sample was released publicly as the first such dataset. On the production data, ESMM beat the baseline by roughly 2.18% AUC on CVR and 2.32% on CTCVR (approximate: secondary sources quote slightly different numbers) [5].

#### Successors
- **ESM2 (Alibaba, 2019).** Breaks the click-to-purchase path into intermediate post-click actions such as add-to-cart and wishlist, so the sparse purchase label gets denser supervision [6].
- **ESCM2 (SIGIR 2022).** Shows that ESMM's CVR estimates are themselves biased: they over-estimate and ignore the causal click-to-conversion link. It adds a counterfactual risk-minimization regularizer to the entire-space framework [7].
- **AITM (Meituan, KDD 2021).** For multi-step conversion funnels in credit-card customer acquisition (impression -> click -> application -> approval -> activation), an Adaptive Information Transfer module learns what and how much information to pass from earlier steps to later ones. A Behavioral Expectation Calibrator in the loss keeps later-step probabilities consistent with earlier ones. It was deployed in the Meituan app [8].

**Lesson.** When tasks form a funnel, encode the funnel in the model: either as probability chain rules (ESMM) or as learned information transfer between task towers (AITM). Do not train later stages only on survivors of earlier stages.

### 1.8 Comparison of MTL architectures

| Architecture | Sharing mechanism | Handles conflict | Handles funnel / SSB | Public production use | Typical failure mode |
|---|---|---|---|---|---|
| Shared-bottom | Hard: one bottom MLP | Poorly | No | Pinterest Homefeed (MTL heads) [33], many others | Negative transfer, seesaw |
| MMoE [1] | Shared experts, per-task softmax gates | Moderately | No | Google, YouTube [2]; Snap ads [41] | Gate polarization, expert collapse [9] |
| PLE / CGC [4] | Task-specific + shared experts, stacked | Well | No | Tencent video [4]; Snap ads [41] | More params, more tuning |
| HoME [9] | Hierarchical expert groups, normalized | Well | No | Kuaishou, ~400M DAU [9] | Complexity, group design |
| STEM [10] | Shared + per-task embedding tables | Well (embedding level) | No | Tencent (paper) | Embedding memory multiplies |
| ESMM [5] | Shared embeddings, product of towers | N/A | Yes (CTR x CVR) | Taobao / Alibaba ads | CVR bias (fixed by ESCM2) [7] |
| AITM [8] | Learned transfer between step towers | N/A | Yes (multi-step) | Meituan [8] | Transfer is one-directional |
| Parallel MaskNet [11] | Shared trunk, per-engagement heads | N/A | No | Twitter heavy ranker [11] | Calibration across heads |

---

## 2. The value-model layer: how industry blends objectives

The MTL model outputs a vector of probabilities. The *value model* (Meta's term), *scoring weights* (X), *utility* (Pinterest), or *ensemble sort* (Kuaishou) collapses that vector into one score. Most of the real product policy lives here.

### 2.1 Twitter (2023): heavy ranker plus a public weight table

**Pipeline.** Twitter's 2023 open-source release describes the For You timeline as: candidate sourcing (search-index in-network, which supplies roughly 50% of posts; tweet-mixer out-of-network; UTEG graph traversal on GraphJet; Follow Recommendation Service), then a light ranker, then the *heavy ranker*, then filters/heuristics and mixing in Home Mixer [12][13]. Home Mixer hydrates about 6,000 features per candidate for ranking [13]. Its heuristics include author diversity, in-/out-of-network content balance, feedback fatigue, deduplication, and visibility filtering [13].

**Model.** The heavy ranker is a *parallel MaskNet* that outputs a probability in [0, 1] for each engagement type [11].

**Fusion.** The final score is a plain weighted sum:

```
  score = sum_i  weight_i * P(engagement_i)
```

The weights published in the README (as of April 5, 2023) [11]:

| Predicted engagement | Weight |
|---|---|
| Favorite (like) | 0.5 |
| Retweet | 1.0 |
| Reply | 13.5 |
| Good profile click (open profile, then like or reply) | 12.0 |
| Video playback 50% | 0.005 |
| Reply engaged by author | 75.0 |
| Good click (open conversation, then reply or like) | 11.0 |
| Good click v2 (open conversation, stay 2+ minutes) | 10.0 |
| Negative feedback v2 (show less, block, mute) | -74.0 |
| Report | -369.0 |

**How the weights were set.** The README says the weights were first chosen so that *each weighted engagement contributes roughly equally on average* to the score. They were then adjusted periodically to optimize platform metrics, and they live in a config file that can be changed at any time [11]. That initialization is a useful heuristic: a weight scaled by the inverse of the base rate. A rare action like a reply or report gets a large weight because its probability is tiny.

### 2.2 X / xAI (2026): Phoenix transformer and the `xai-value-model`

The 2026 `xai-org/x-algorithm` repository documents the successor system in more detail [14][15][16].

**Model.** *Phoenix*, a transformer (JAX training, Rust serving), handles both embedding-based retrieval and ranking. The main input is the viewer's recent engagement history. It predicts probabilities for many actions instead of a single relevance score. Hash-based embeddings mean there is no vocabulary to maintain. *Candidate isolation* stops candidates from attending to each other: they attend only to viewer context, so a post's score does not depend on what else is in the batch, and scores can be cached [14].

**Predicted heads** in five groups [14]:
- Engagement: favorite, reply, repost, quote, share, share via DM, share via copy link.
- Clicks: post, profile, link, photo expand, video open, quoted post.
- Attention: video quality view, dwell, dwell time, click dwell time, video continuation seconds, profile visit seconds.
- Author: follow author.
- Negative: not interested, mute author, block author, report, not dwelled.

**Weights.** Default weights mirrored from feature switches in `home-mixer/params/param.rs` (live values may differ) [15]:

| Head | Weight | Head | Weight |
|---|---|---|---|
| Favorite | 0.5 | Share via copy link | 20.0 |
| Reply | 5.0 | Share via DM | 5.0 |
| Bidirectional-follow reply boost | 15.0 | Share | 2.0 |
| Retweet | 1.0 | Quote | 5.0 |
| Follow author | 4.0 | Click | 0.3 |
| Open link | 0.2 | Video open | 0.07 |
| Photo expand | 0.05 | Dwell | 0.05 |
| Continuous click dwell time | 0.4 | Continuous dwell time | 0.004 |
| Not interested | -47.52 | Block author | -31.2 |
| Mute author | -58.8 | Report | -234.0 |
| Not dwelled | -0.02 | Post unexplored (OON) | 0.015 |

The source comments make two points worth repeating to anyone who reads these tables [15]:
- Weights multiply *predicted probabilities or continuous values*, not raw counts. "One report cancels 468 likes" is a misreading.
- The report weight is large because the report base rate is "more than 1000x lower than a Like", which is the same base-rate normalization logic as Twitter 2023.

**Scoring arithmetic** (`xai-value-model/scoring.rs`) [16]:
1. Weighted sum of heads. Two-factor heads (video continuation, profile visit time) use p * seconds. Eligibility gates zero out some weights, for example video-quality-view for ineligible posts. Reply and dwell weights can vary per candidate.
2. **Negative-score offset.** Let T be the total weight sum and N the magnitude of the negative weights. Negative raw scores are squashed into [0, OFFSET) as (s + N) / T * OFFSET. Non-negative scores become s + OFFSET. This keeps scores positive and order-preserving, so the later multiplicative adjustments behave.
3. **Exploration bonus** added in unoffset space.
4. **Author diversity multiplier** (1 - floor) * decay^k + floor, where k is the number of higher-ranked posts already shown from the same author.
5. **Out-of-network multiplier** below 1 for posts from accounts not followed (optionally also for in-network replies and reposts).
6. **Cold-start lift.** A new-author candidate is raised to at least a target rank, never lowered.
7. A separate `vm-ranker` then reorders with a **determinantal point process (DPP)** over embeddings, giving up a little score for less similarity between neighbors [14].

```
  Phoenix (transformer, candidate-isolated) -> ~25 heads
        -> weighted sum (param.rs weights) -> offset negatives
        -> + exploration bonus -> x author-diversity decay
        -> x out-of-network factor -> cold-start lift
        -> vm-ranker DPP rerank -> filters -> timeline
```

**Lesson.** The value function is a small, versioned, auditable piece of code that sits apart from the model. Policy changes (more replies, fewer reports) ship as config changes with no retraining.

### 2.3 Meta / Facebook News Feed: MSI and survey-calibrated value

**Scale.** Feed ranking must score candidates for more than 2 billion people in real time. On average there are more than 1,000 candidate posts per user per day. A lightweight first pass narrows them to about 500 for the heavier multitask neural models [17][20].

**Model and fusion.** Each post gets several predicted actions (like, comment, share, ...) from *multitask neural networks*. These "all need to somehow aggregate up to a single value" [17]. Meta's 2021 post uses a running example (a user named Juan). Several prediction models can disagree about which post he will respond to most. The disagreement is resolved by weighting predictions by how *meaningful* the corresponding interactions are, using survey data on what people say they value [17].

**MSI (Meaningful Social Interactions, 2018).** Internal documents reported by CNN in 2021 describe MSI as a point system. Likes, comments, re-shares, reactions and even event RSVPs had different point values, adjusted for the relationship between the people involved. Weights were set using surveys (one cited survey covered over 69,000 people in five countries) together with analysis of user data. A 2019 internal note said MSI was "beneficial for most metrics" in 2018, including likes and comments between users, daily users, revenue, and time on Feed [18]. The same reporting describes how the weighting backfired: heavy weights on comments, reshares and reaction emojis also amplified outrage-provoking content [18].

**Survey-based objectives.** From 2019 Meta added "Is this post worth your time?" surveys. In 2021 it gave more weight to survey-derived predictions about how meaningful friends, Pages and Groups are to a person. Classifiers trained to predict survey answers let survey signals generalize to posts without surveys [19].

**Lesson.** Engagement-derived weights optimize what can be measured. Survey-predicted objectives give a value model a counterweight that is closer to what users say they want. Weight choices also have large second-order effects on the content ecosystem.

### 2.4 Instagram Explore: log-linear vs. linear value models

**2020 design (constrained exploration).** Instagram's multi-label model predicts engagement (like, save) and negative feedback (see less / not interested). Point-wise heads are trained with cross-entropy, alongside GBDTs and list-wise LambdaRank models. The value model is *log-linear* [21]:

```
  Value(post) = P(like) ^ w_like  *  (1 - P(not interested)) ^ w_see_less
```

The weights are tuned with (a) *offline replay over user sessions* and (b) *online Bayesian optimization*, and they are retuned often as the system changes [21]. A multiplicative form has a useful property: a high predicted negative probability can veto an item however likely a like is.

**2023 design (scaling Explore).** Four stages: retrieval, first-stage ranking (lightweight), second-stage ranking with a *multi-task multi-label (MTML)* neural network, and final reranking. The value model became an explicitly *linear* weighted sum [22]:

```
  Expected Value = W_click * P(click) + W_like * P(like)
                   - W_see_less * P(see less) + ...
```

Meta describes the value model as approximating how much value each media item brings to the user. Tuning its weights lets the team "explore different tradeoffs between online engagement metrics" [22]. The second-stage MTML model is much heavier than two-tower retrieval, so Meta precomputes some users' recommendations off-peak. Business rules in the value model handle diversity, such as penalizing repeated posts from the same author or seed account [22]. For tuning, the post contrasts Bayesian optimization, where engineers specify only the parameters, the goal metric and regression thresholds on guardrail metrics, with offline tuning [22].

**Lesson.** Linear and log-linear value models make different trade-offs. Linear is easy to interpret and lets each term add its own contribution. Log-linear lets a single term dominate (a veto). In both cases the weights are search parameters for online BO, not something learned by gradient descent.

### 2.5 Meta ads: Lattice, multi-domain multi-task consolidation

**Problem.** Ads ranking had grown into many models across surfaces and objectives. That slowed iteration and wasted compute [23].

**Design.** Lattice consolidates models into a *multi-domain, multi-task* architecture with sparse activation. It learns from many data sources and handles delayed feedback ranging from seconds to days. Meta's 2023 description cites *trillions of parameters* trained on *hundreds of billions of examples* [23]. Balancing thousands of domains and tens of objectives by hand is impossible, so Meta uses Pareto-optimality-based methods, including Pareto-front feature selection and MetaBalance [23][25]. The 2025 paper describes cross-domain knowledge sharing, data consolidation, model unification, and *distillation* of a large consolidated model into lean user-facing models. It reports +10% on revenue-driving top-line metrics, +11.5% user satisfaction, +6% conversion rate, and 20% capacity saving (Meta's own figures) [24].

**MetaBalance.** At each iteration, and per part of the network, the method rescales each *auxiliary* task's gradient so its magnitude is closer to the *target* task's gradient. Auxiliary tasks then neither dominate nor vanish. The paper reports +8.34% NDCG@10 over the strongest baseline on two datasets [25].

### 2.6 LinkedIn Feed: constrained objectives and autotuning

**Formulation.** LinkedIn treats feed ranking as multi-objective optimization. Separate models predict actions such as viral actions (react, comment, share), contributions, and downstream effects on creators. A linear combination of per-objective probabilities ranks items [26][27][28]. A 2019 post describes objectives that include a member's likelihood to view updates, to join conversations, and the value of timely feedback to content creators [27].

**Autotuning (2018-2020).** The weights had been set by hand by ML engineers, which LinkedIn called very inefficient [26]. The automated approach [26][28]:
- Treat online metrics as black-box functions of the control parameters (for example the click weight). Model each metric with a *Gaussian process* from A/B data.
- Optimize a primary metric subject to constraints on others. LinkedIn observed that contributions *fall* as click weight rises while the constraint metric (normalized clicks) *rises*, so the optimum sits on the constraint boundary.
- Choose the next parameters by sampling functions from the GP posterior and computing each sample's argmax. That gives a probability that each parameter value is optimal, a Thompson-sampling-style procedure.
- Ramp volume and duration are limited in practice, so many experiments run in parallel at low traffic.

**LiRank calibration (KDD 2024).** LinkedIn's LiRank paper adds an *isotonic calibration layer trained jointly* inside the deep model instead of post-hoc Platt/isotonic regression, and reports significant gains in predictive accuracy [29]. The 2026 "Isotonic Layer" paper extends this to a single learnable component that handles calibration, position debiasing and multi-task bias correction, first deployed in LiRank [30].

### 2.7 Pinterest Homefeed: utility weights, from global to per-request RL

**Model.** The Homefeed ranker (Pinnability) predicts actions such as click, repin and hide, with hide as a negative signal. TransAct adds a real-time user-action sequence transformer to it [33].

**Utility.** A utility-weight vector defines each item's utility as a weighted sum of predicted objective scores. Historically the weights were set by offline analysis and online experiments, applied *globally* (one vector for all requests), and left in place for long periods [32].

**PRL-PUTS (2026).** Pinterest reframes weight selection as a *one-step, value-based RL* problem. For each request an agent picks the weight vector that maximizes expected request-level engagement reward. An inference-time scalarization parameter yields a family of policies and an *empirical Pareto frontier*, used as a governance artifact for choosing the operating point. It runs alongside ranking inference, adds no serving latency, and needs no ranker change. Training uses unbiased exploration logs. Online it gave +0.13% successful sessions, with controllable Repin vs. P2P trade-offs [32]. Pinterest has also published on optimizing longer-horizon downstream rewards beyond immediate engagement on Homefeed [34] and on lifecycle-dependent utility weights (secondary reporting).

### 2.8 Kuaishou: from formula fusion to learned ensemble sort

Kuaishou's papers describe the industry-standard "ensemble sort" stage: the ranking model emits many pXTRs (pLTR, pWTR, pFTR, ...), and a manually tuned formula merges them into one score before the top dozen items are picked. The Pantheon authors call this formula-based adjustment the "wide-accepted industry" practice [35].

A line of Kuaishou and related work replaces the hand formula with learned, personalized fusion:
- **RL-based multi-task fusion (MTF).** Users are the environment, the recommender is the agent, and the action is the fusion-weight vector. Fusion is often a log-transformed weighted sum with a bias term. The reward is long-term user satisfaction [38][37].
- **xMTF.** A "formula-free" MTF model that drops the fixed functional form [37].
- **Pantheon (2025).** Personalized ensemble sort trained jointly with the real-time ranking model using Iterative Pareto Policy Optimization (IPPO). It searches objective weights automatically and is deployed at Kuaishou for about 400 million daily users [35].
- **GRADE (2025).** Critic-free, GRPO-style group-relative RL over candidate weight vectors with Dirichlet exploration. In Kuaishou marketplace search it reports +0.595% CTR, +1.193% CVR, +1.788% OPM and +1.568% order volume [36].
- **Rank-consistency ensembles (2026).** Moves the ensemble from score fusion toward rank consistency across objectives, using Kuaishou data [43]. UMRE proposes a unified monotonic transformation of pXTRs before fusion [46].

```
  ranking model --> pLTR, pWTR, pFTR, pCMTR, E[watch] ...
                          |
     +--------------------+--------------------+
     | 2018-era: hand formula, global weights  |
     | 2022+:    RL agent picks weights/user   |
     | 2025+:    Pareto policy trained jointly |
     +--------------------+--------------------+
                          v
                   top-k by fused score
```

### 2.9 Other public data points

- **Snap ads** use MMoE and PLE to predict multiple conversion events (app installs, purchases, sign-ups) jointly [41].
- **Spotify audio ads** use multi-task learning and calibrate each task with *per-task temperature scaling* (one parameter per task, a simplified Platt scaling). They chose it over Platt or isotonic for simplicity, because miscalibration distorts auction ordering and pricing [31].
- **Netflix** reports consolidating several related search and recommendation models into one multi-task model to cut maintenance cost and improve quality [42].
- **Alibaba PE-LTR (RecSys 2019)** optimizes GMV and CTR together with a weighted aggregation. It derives a condition for Pareto efficiency and a two-step Pareto-efficient optimization algorithm that can also generate a Pareto frontier [39].

### 2.10 Value-model comparison

| System | MTL model | Fusion form | Weight setting | Post-fusion adjustments |
|---|---|---|---|---|
| Twitter 2023 [11] | Parallel MaskNet | Linear sum, 10 heads | Base-rate equalization, then periodic hand tuning | Author diversity, content balance, fatigue [13] |
| X 2026 [14][15][16] | Phoenix transformer, candidate-isolated | Linear sum, ~25 heads, negative offset | Feature-switch config, A/B | Author decay, OON factor, cold-start lift, DPP |
| Facebook Feed [17][18][19] | Multitask NNs | Weighted value; MSI points | Surveys + data analysis | Diversity, integrity demotions [20] |
| Instagram Explore 2020 [21] | Multi-label NN + GBDT + LambdaRank | Log-linear product | Offline replay + online BO | Exploration constraints |
| Instagram Explore 2023 [22] | MTML NN | Linear sum | BO with guardrail thresholds | Same-author penalty |
| YouTube 2019 [2][3] | MMoE + shallow bias tower | Weighted combination | Manual tuning | n/a (not public) |
| LinkedIn Feed [26][28] | Per-objective models (LiRank) | Linear combination | GP-based constrained BO | n/a |
| Pinterest Homefeed [32] | Pinnability MTL + TransAct | Linear utility | Global -> per-request RL, Pareto sweep | n/a |
| Kuaishou [35][36] | MMoE/PLE/HoME | Formula -> learned ensemble | Hand -> RL (IPPO, GRPO) | n/a |

---

## 3. Calibration: the contract between the model and the value model

A value model that adds w_i * p_i is only meaningful if each p_i means what it says. If pReply is systematically 2x too high, the effective reply weight is 2x too. Changes to training data, sampling or architecture then silently shift product policy. Practitioners describe calibration as giving the fusion layer "a stable contract about score scale." For pure ordering inside one objective calibration does not matter, but for ads auctions and feed blending it does [45][31].

**Techniques seen in industry:**
- **Platt / temperature scaling per task.** A cheap, global monotonic rescaling of logits. Spotify uses one temperature per task [31].
- **Isotonic regression.** Non-parametric and monotone, applied post hoc per task. Applied to ranking calibration since Menon et al. (2012) [45].
- **Field-level calibration.** Separate calibration per advertiser, placement or other field value, using posterior statistics (for example AdaCalib, per secondary sources) [45].
- **Learned isotonic layers inside the network.** LinkedIn trains the calibration jointly with the model. The 2026 follow-up argues that Platt (two parameters, one global transform) and per-task isotonic regression cannot capture interactions across task x position x platform. It replaces them with one learnable isotonic layer that also handles position debiasing [29][30].
- **Negative sampling correction.** If negatives are downsampled at rate r, predicted odds must be rescaled to bring probabilities back to the true base rate before fusion (standard practice; ESMM-style funnels need the same care at every step).

**Rule of thumb.** Recalibrate whenever a head's training distribution changes. Monitor predicted vs. observed rate per head, per surface, per position bucket. Treat a calibration drift as a value-model change.

---

## 4. Training-time balancing: loss weights and gradient surgery

Fusion weights act at *serving* time. MTL training also needs per-task *loss* weights, which are a separate set of hyperparameters:

- **Static loss weights.** Hand-tuned. Simple, but they interact with label sparsity.
- **Uncertainty weighting (Kendall et al., 2018).** Scale each task loss by 1/sigma_k^2 and add a log sigma_k regularizer, with sigma learned (homoscedastic uncertainty) [48]. The GradNorm authors argue these weights can change without constraint and push the effective learning rate up [40].
- **GradNorm (2018).** Learn per-task loss weights so that per-task gradient norms on shared parameters sit on a common scale, adjusted by relative training rates. It has one asymmetry hyperparameter alpha [40].
- **PCGrad (2020).** When two tasks' gradients conflict (negative cosine), project each onto the normal plane of the other [47].
- **MetaBalance (Meta, 2022).** Scale *auxiliary* gradients relative to the *target* task's gradient magnitude, per network part. Built for recommenders where one task is primary [25].
- **Structural decoupling.** Task-specific experts (PLE), task-specific embeddings with stop-gradient (STEM), and hierarchical expert groups (HoME) avoid much of the gradient conflict by construction [4][10][9].

**Practical note.** Loss weights change *what the model learns*. Fusion weights change *what gets shown*. Teams that confuse the two often try to fix a serving trade-off by retraining, or the reverse.

---

## 5. Pareto tuning and objective blending in practice

### 5.1 The workflow most companies converge on

```
  1. Train calibrated MTL heads (one model, many heads).
  2. Define the value function: score = sum w_i * p_i (+ multipliers).
  3. Initialize w_i ~ 1 / mean(p_i) * importance_i  (equal-contribution
     heuristic, as in Twitter 2023 [11]).
  4. Offline: replay logged sessions; sweep w; trace a Pareto front of
     proxy metrics (e.g. predicted likes vs predicted hides) [21][32].
  5. Online: constrained Bayesian optimization or GP Thompson sampling
     over w, primary metric s.t. guardrails [22][26].
  6. Personalize: learn w = f(user, request) with RL / policy
     optimization, keep a Pareto frontier for governance [32][35][36].
  7. Re-check calibration on every model change; re-tune w on every
     product-goal change.
```

### 5.2 Why weighted sums persist

Even teams with RL-based fusion keep a weighted sum as the interface. It is interpretable ("what does a share buy us?"), it can be audited (X publishes its weights [15]), it can be changed without retraining, and it composes with business multipliers. Learned fusion usually outputs the weights, not the score, so the human-readable layer survives [32][35].

### 5.3 Pareto frontiers as governance artifacts

Pinterest's PRL-PUTS makes the empirical Pareto frontier an explicit artifact for choosing among Repin vs. P2P trade-offs [32]. Meta's Lattice uses Pareto-front methods to avoid hand-tuning thousands of domains [23]. Alibaba's PE-LTR gives a theoretical Pareto-efficiency condition for weighted aggregation [39]. The general pattern: compute the frontier, then make the operating point a product decision.

### 5.4 Multiplicative and offset tricks

- **Log-linear products** (Instagram 2020) let a negative-feedback term veto an item [21].
- **Negative-score offsets** (X 2026) map negative weighted sums into a small positive band, so later multiplicative adjustments (diversity decay, out-of-network discount) cannot accidentally *promote* a negatively scored item by scaling it toward zero [16].
- **Monotone transforms before fusion** (UMRE) align heads with very different scales and distributions [46].

### 5.5 Failure modes documented in public sources

- **Engagement weights amplifying harmful content.** Facebook MSI's heavy weights on comments, reshares and reactions are reported to have boosted divisive content [18].
- **Seesaw.** Improving one head degrades another (PLE motivation) [4].
- **Expert collapse and sparse-task underfitting** in MoE MTL [9].
- **Biased CVR from entire-space models** [7].
- **Global weights fitting no one.** Pinterest and Kuaishou both motivate personalized fusion by noting that one global weight vector is slow to adapt and ignores individual intent [32][36].
- **Manual tuning cost.** LinkedIn [26] and Kuaishou [35] both describe hand-tuning as a bottleneck.

---

## Key takeaways for system designers

1. **Separate prediction from policy.** Train an MTL model that predicts calibrated probabilities of concrete actions. Keep the business trade-off in a small, versioned value function. X, Meta, Pinterest and LinkedIn all keep this split.
2. **Start with shared-bottom and move up only when you see the seesaw.** Use MMoE when tasks are moderately related. Use PLE/CGC (task-specific experts) when they conflict. Watch for expert collapse at scale, and use normalization or hierarchy as HoME does. Consider task-specific embeddings (STEM) when the conflict is in the ID embeddings.
3. **Encode funnels explicitly.** For impression -> click -> conversion, use entire-space modeling (ESMM: pCTCVR = pCTR * pCVR) or learned transfer (AITM). Add counterfactual corrections (ESCM2) when you need unbiased CVR.
4. **Model bias structurally.** An additive shallow tower for position and device (YouTube) or a learned isotonic layer (LinkedIn) keeps presentation bias out of the utility heads.
5. **Calibrate every head the value model consumes.** Without calibration, fusion weights silently drift with every retrain. Monitor predicted vs. observed rates per head.
6. **Initialize weights by base-rate equalization**, then tune. Twitter's "equal average contribution" rule and X's ">1000x rarer than a like" comment explain why rare actions (replies, reports) get large weights.
7. **Give negatives real teeth.** Large negative weights on report, block, mute and "not interested" (X), or a multiplicative veto term (Instagram), are the cheapest safety mechanisms in the stack.
8. **Tune online with constrained BO.** Optimize one primary metric subject to guardrail thresholds (Meta, LinkedIn). Expect optima on the constraint boundary.
9. **Personalize the weights, not just the predictions.** Per-request RL weight selection (Pinterest PRL-PUTS, Kuaishou Pantheon/GRADE) gives small but real online gains with no extra serving latency, and a Pareto frontier for governance.
10. **Include survey and long-term objectives.** Pure engagement weights can amplify harmful content (MSI). Survey-predicted value (Facebook) and downstream or long-horizon rewards (Pinterest) give a counterweight.
11. **Apply diversity and network-mix multipliers after fusion**, and protect against score-sign bugs (X's negative-score offset).
12. **Consolidate models when you can afford one big MTL model plus distillation** (Meta Lattice, Netflix). Fewer models means faster iteration and less compute, provided loss balancing (MetaBalance, Pareto methods) is handled.

---

## References

1. Ma, J., Zhao, Z., Yi, X., Chen, J., Hong, L., Chi, E. H. "Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts." Google, KDD 2018. https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/
2. Zhao, Z., et al. "Recommending What Video to Watch Next: A Multitask Ranking System." Google/YouTube, RecSys 2019. https://research.google/pubs/recommending-what-video-to-watch-next-a-multitask-ranking-system/ (ACM: https://dl.acm.org/doi/10.1145/3298689.3346997)
3. "How YouTube is Recommending Your Next Video." KDnuggets, 2019. https://www.kdnuggets.com/2019/10/youtube-recommending-next-video.html
4. Tang, H., Liu, J., Zhao, M., Gong, X. "Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations." Tencent, RecSys 2020. https://dl.acm.org/doi/fullHtml/10.1145/3383313.3412236
5. Ma, X., Zhao, L., et al. "Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate." Alibaba, SIGIR 2018. https://arxiv.org/abs/1804.07931
6. Wen, H., et al. "Entire Space Multi-Task Modeling via Post-Click Behavior Decomposition for Conversion Rate Prediction (ESM2)." Alibaba, 2019. https://arxiv.org/abs/1910.07099
7. Wang, H., et al. "ESCM2: Entire Space Counterfactual Multi-Task Model for Post-Click Conversion Rate Estimation." SIGIR 2022. https://arxiv.org/abs/2204.05125
8. Xi, D., Chen, Z., et al. "Modeling the Sequential Dependence among Audience Multi-step Conversions with Multi-task Learning in Targeted Display Advertising (AITM)." Meituan, KDD 2021. https://arxiv.org/abs/2105.08489
9. Wang, X., et al. "HoME: Hierarchy of Multi-Gate Experts for Multi-Task Learning at Kuaishou." Kuaishou, 2024 (KDD 2025). https://arxiv.org/abs/2408.05430
10. Su, L., Pan, J., et al. "STEM: Unleashing the Power of Embeddings for Multi-task Recommendation." Tsinghua/Tencent, AAAI 2024. https://arxiv.org/abs/2308.13537
11. Twitter. "Heavy Ranker README (projects/home/recap)." the-algorithm-ml, 2023. https://github.com/twitter/the-algorithm-ml/blob/main/projects/home/recap/README.md
12. Twitter. "the-algorithm README." 2023. https://github.com/twitter/the-algorithm
13. Twitter. "Home Mixer README." 2023. https://github.com/twitter/the-algorithm/blob/main/home-mixer/README.md
14. xAI / X. "x-algorithm README." 2026. https://github.com/xai-org/x-algorithm
15. xAI / X. "home-mixer/params/param.rs (scoring weights)." 2026. https://github.com/xai-org/x-algorithm/blob/main/home-mixer/params/param.rs
16. xAI / X. "xai-value-model/scoring.rs." 2026. https://github.com/xai-org/x-algorithm/blob/main/xai-value-model/scoring.rs
17. Lada, A., Wang, M., Yan, T. "How machine learning powers Facebook's News Feed ranking algorithm." Engineering at Meta, 2021. https://engineering.fb.com/2021/01/26/ml-applications/news-feed-ranking/
18. CNN Business (syndicated). "Likes, anger emojis and RSVPs: the math behind Facebook's News Feed and how it backfired." 2021. https://abc17news.com/news/2021/10/27/likes-anger-emojis-and-rsvps-the-math-behind-facebooks-news-feed-and-how-it-backfired/
19. Meta. "Incorporating More Feedback Into News Feed Ranking." 2021. https://about.fb.com/news/2021/04/incorporating-more-feedback-into-news-feed-ranking/
20. Meta Transparency Center. "Facebook Feed AI system." https://transparency.meta.com/features/explaining-ranking/fb-feed
21. Meta / Instagram. "How Instagram suggests new content (Designing a constrained exploration system)." Engineering at Meta, 2020. https://engineering.fb.com/2020/12/10/web/how-instagram-suggests-new-content/
22. Meta / Instagram. "Scaling the Instagram Explore recommendations system." Engineering at Meta, 2023. https://engineering.fb.com/2023/08/09/ml-applications/scaling-instagram-explore-recommendations-system/
23. Meta AI. "Meta Lattice: AI ads performance and efficiency." 2023. https://ai.meta.com/blog/ai-ads-performance-efficiency-meta-lattice/
24. Meta. "Meta Lattice: Model Space Redesign for Cost-Effective Industry-Scale Ads Recommendations." 2025. https://arxiv.org/abs/2512.09200
25. He, Y., Feng, X., Cheng, C., Ji, G., Guo, Y., Caverlee, J. "MetaBalance: Improving Multi-Task Recommendations via Adapting Gradient Magnitudes of Auxiliary Tasks." Meta AI / TAMU, WWW 2022. https://arxiv.org/abs/2203.06801
26. LinkedIn (via KDnuggets). "Autotuning for Multi-Objective Optimization on LinkedIn's Feed Ranking." 2020. https://www.kdnuggets.com/2020/08/autotuning-multi-objective-optimization-linkedin-feed-ranking.html
27. LinkedIn Engineering. "Community-focused Feed optimization." 2019. https://engineering.linkedin.com/blog/2019/06/community-focused-feed-optimization
28. Colyer, A. "Online parameter selection for web-based ranking problems." The Morning Paper, 2018. https://blog.acolyer.org/2018/10/08/online-parameter-selection-for-web-based-ranking-problems/
29. Borisyuk, F., et al. "LiRank: Industrial Large Scale Ranking Models at LinkedIn." LinkedIn, KDD 2024. https://arxiv.org/abs/2402.06859
30. "Isotonic Layer: A Unified Framework for Recommendation Calibration and Debiasing." LinkedIn, 2026. https://arxiv.org/abs/2603.06589
31. "An Audio-centric Multi-task Learning Framework for Streaming Ads Targeting on Spotify." Spotify, 2025. https://arxiv.org/abs/2506.18735
32. "A Production-Ready RL Framework for Personalized Utility Tuning with Pareto Sweeping in Pinterest Recommender Systems (PRL-PUTS)." Pinterest, 2026. https://arxiv.org/abs/2605.16344
33. Xia, X., et al. "TransAct: Transformer-based Realtime User Action Model for Recommendation at Pinterest." Pinterest, KDD 2023. https://arxiv.org/abs/2306.00248
34. "Long-term User Engagement Optimization through Model-agnostic Downstream Rewards Learning." Pinterest, 2026. https://arxiv.org/abs/2607.14192
35. Cao, J., et al. "Pantheon: Personalized Multi-objective Ensemble Sort via Iterative Pareto Policy Optimization." Kuaishou, 2025. https://arxiv.org/abs/2505.13894
36. "GRADE: Personalized Multi-Task Fusion via Group-relative Reinforcement Learning with Adaptive Dirichlet Exploration." Kuaishou, 2025. https://arxiv.org/abs/2510.07919
37. "xMTF: A Formula-Free Model for Reinforcement-Learning-Based Multi-Task Fusion in Recommender Systems." 2025. https://arxiv.org/abs/2504.05669
38. "Multi-Task Fusion via Reinforcement Learning for Long-Term User Satisfaction in Recommender Systems." 2022. https://arxiv.org/abs/2208.04560
39. Lin, X., et al. "A Pareto-Efficient Algorithm for Multiple Objective Optimization in E-Commerce Recommendation (PE-LTR)." Alibaba, RecSys 2019. https://www.doi.org/10.1145/3298689.3346998
40. Chen, Z., Badrinarayanan, V., Lee, C.-Y., Rabinovich, A. "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks." ICML 2018. https://arxiv.org/abs/1711.02257
41. InfoQ. "Snap ad ranking (MMoE and PLE for multiple conversion events)." 2022. https://www.infoq.com/news/2022/10/snap-ad-ranking/
42. Netflix Technology Blog. "Lessons learnt from consolidating ML models in a large scale recommendation system." 2023. https://netflixtechblog.medium.com/lessons-learnt-from-consolidating-ml-models-in-a-large-scale-recommendation-system-870c5ea5eb4a
43. "Rethinking Multi-objective Ranking Ensemble in Recommender System: From Score Fusion to Rank Consistency." 2026. https://arxiv.org/abs/2601.02955
44. "Advances and Challenges of Multi-task Learning Method in Recommender System: A Survey." 2023. https://arxiv.org/abs/2305.13843
45. "Ranking model calibration in recommender systems." RecSys ML Substack. https://recsysml.substack.com/p/ranking-model-calibration-in-recommender (and parametric ranking calibration, https://arxiv.org/abs/2112.07428)
46. "UMRE: A Unified Monotonic Transformation for Ranking Ensemble in Recommender Systems." 2025. https://arxiv.org/abs/2508.07613
47. Yu, T., et al. "Gradient Surgery for Multi-Task Learning (PCGrad)." NeurIPS 2020. https://arxiv.org/abs/2001.06782
48. Kendall, A., Gal, Y., Cipolla, R. "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics." CVPR 2018. https://arxiv.org/abs/1705.07115
