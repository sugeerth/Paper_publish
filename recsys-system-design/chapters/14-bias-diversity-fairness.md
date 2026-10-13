# 14. Debiasing, Diversity, Fairness and Cold Start: The Corrective Layers of Production Recommenders

> Research note: when this chapter was written, live web search and page fetching were not
> available in the research environment (the shared search budget was used up and outbound
> fetches failed DNS resolution). The chapter is therefore built from well-known, widely
> cited primary papers and engineering write-ups whose content and identifiers the author
> knows. The claims follow those sources. Any number not checked against the source is
> marked "approximately" or "reported as". Check figures against the cited originals before
> quoting them.

## Overview

Every production recommender learns from logs that it produced itself. A user can only
click what was shown, items shown at the top get more clicks, popular items get more
impressions and so look more relevant, and new users, items and creators have almost no
data at all. Left alone, the system's feedback loop makes all of these effects stronger. The
result is a narrow, popularity-heavy, position-driven product that can be unfair to
suppliers.

Large platforms answer this with a set of "corrective layers" that sit around the core
retrieval and ranking stack (Chapters 2-4):

```
                 +-------------------- feedback loop --------------------+
                 |                                                        |
 logs (biased) --+--> [A] debiasing in training -----> retrieval / ranking |
                      - position towers (YouTube, Airbnb, Huawei PAL)      |
                      - IPS / propensity weighting (Joachims, Schnabel)    |
                      - logQ sampling correction (YouTube two-tower)       |
                      - popularity deconfounding (PDA, MACR)               |
                      - duration deconfounding (Kuaishou D2Q)              |
                                                     |                     |
                                                     v                     |
                      [B] slate construction / re-ranking                  |
                      - MMR, DPP (YouTube, Hulu), calibration (Netflix)    |
                      - listwise diversity (Airbnb)                        |
                      - fairness constraints (LinkedIn, Spotify)           |
                                                     |                     |
                                                     v                     |
                      [C] cold start and exploration                       |
                      - content towers, meta-embeddings, DropoutNet        |
                      - cohort gating (Kuaishou POSO), bandits (Yahoo)     |
                      - new-listing priors (Airbnb)                        |
                                                     |                     |
                                                     v                     |
                                            impressions shown to user -----+
```

This chapter goes through each layer. For each one it covers the problem, the deployed
designs, and the lessons for system designers.

---

## 14.1 A Taxonomy of Bias in Recommender Logs

The survey by Chen et al. [19] is the standard map of the field. It separates:

| Bias | Where it enters | Symptom | Typical fix |
|---|---|---|---|
| Position bias | User clicks on what is ranked high | CTR depends on slot, not relevance | Position feature + dropout, IPS, PAL |
| Selection bias | Users rate only what they choose | Explicit ratings are MNAR | Propensity-weighted losses [13] |
| Exposure bias | Users only see a subset of items | Unobserved does not mean negative | Exposure modelling [14], off-policy correction [31] |
| Popularity bias | Training data dominated by head | Long tail under-recommended | Regularisers [16], deconfounding [17][18], logQ [11] |
| Conformity | Users imitate the crowd | Labels not individual preference | Causal disentangling |
| Duration / length bias | Longer videos get more watch time | Watch-time model prefers long content | Duration-bucketed targets [27] |
| Inductive / feedback-loop bias | Model's own outputs become training data | Homogenisation, filter bubbles | Exploration, randomisation, diversity |
| Unfairness | Unequal exposure across groups | Creators or sellers starved | Exposure constraints [20][21][22] |

Two kinds of fix come up again and again:

1. **Model the bias explicitly as a separate factor** (a "bias tower" or propensity), train
   it jointly with the relevance model, and remove it at serving time.
2. **Reweight the data** by inverse propensities so that the expected loss matches the loss
   on unbiased data.

Production systems prefer (1). It is easier to run, has lower variance, and fits into
multitask DNN rankers.

---

## 14.2 Position Bias

### 14.2.1 Unbiased learning to rank: IPS and propensity estimation

**Problem.** A click on position 1 says less about relevance than a click on position 10,
because users look at position 1 far more often. Joachims, Swaminathan and Schnabel [8] cast
this as counterfactual learning. Under an examination hypothesis,

    P(click | q, d, k) = P(examined | k) * P(relevant | q, d)

and each click is weighted by 1 / P(examined | k). The IPS-weighted ranking loss is then an
unbiased estimate of the loss under full-information relevance labels.

**Estimating propensities.** The original work used randomisation interventions, such as
swapping results at rank 1 and rank k, to estimate a propensity curve that roughly follows a
power law in rank. Swaps hurt the user experience. Agarwal et al. [29] showed that
propensities can instead be estimated without intrusive interventions, using
"intervention harvesting": when several historical ranking functions put the same document
at different positions, the logs already contain natural swaps. Google's personal-search
team (Wang et al. [28][34]) estimated position bias with regression EM, again avoiding
heavy randomisation, and used selection-bias weights in email search ranking.

**Lessons.** IPS is unbiased only if the propensity model is correct. Weights at deep ranks
blow up variance, so practitioners clip them. IPS also struggles with features that
interact with position, such as device type or presentation layout. That limitation
motivated the "bias tower" designs below.

### 14.2.2 YouTube: the shallow tower in the Watch Next ranker

**System.** YouTube's "Recommending What Video to Watch Next" ranking system (Zhao et al.,
RecSys 2019) [1] ranks a few hundred candidates from multiple candidate generators. It uses a
Multi-gate Mixture-of-Experts (MMoE) network with engagement objectives (clicks, watch time)
and satisfaction objectives (likes, dismissals, survey responses).

**Bias problem.** Watch Next logs carry strong selection bias. Users click videos near the
top of the list partly because of position, not utility. The paper calls this an implicit
feedback loop: training on these logs keeps reinforcing the current ranking.

**Architecture.**

```
 user / context / candidate features          position, device (+ other bias feats)
                 |                                        |
         +---------------+                         +--------------+
         | shared hidden |                         | shallow tower|
         |  layers       |                         | (small MLP / |
         +---------------+                         |  linear)     |
                 |                                 +--------------+
         +---------------+                                |
         | MMoE experts  |                                |
         | + task gates  |                                |
         +---------------+                                |
           |         |                                    |
   engagement     satisfaction                            |
   task towers    task towers                             |
           |                                              |
       logit_main  ------------------ (+) --------------- logit_bias
                                       |
                                    sigmoid ----> training loss (click etc.)

  Serving: the position feature is treated as missing, so logit_bias collapses
           to a constant and ranking uses the main towers only.
```

**Key tricks.**

- The shallow tower gets features that drive bias but not relevance: the position of the
  item when it was shown, plus device information. Device matters because position effects
  differ between mobile and desktop.
- The bias logit is **added** to the main model's logit before the sigmoid. This factorises
  the click probability into a relevance part and a bias part.
- During training the position feature is dropped out at a reported rate of about 10%. This
  stops the main model from relying on position, and the model learns sensible behaviour
  when position is missing, which is exactly the serving setting.
- At serving, position is set to missing.

**Results.** The paper reports that the shallow tower improved the engagement metric in live
experiments over baselines, including over using position as a plain input feature. The
learned position biases fall monotonically with position, as expected.

**Lesson.** A "bias-only" tower that is added in logit space is cheap (a few extra
parameters), works with any multitask ranker, and has become the de facto industrial pattern.
Later work in industry and academia extends it with context-aware bias towers (position
crossed with device, UI surface, or session depth).

### 14.2.3 Huawei PAL: position as a separate probability

Huawei's PAL (Guo et al., RecSys 2019) [9], deployed in the Huawei App Store, factorises
CTR multiplicatively:

    pCTR(click | x, pos) = pSeen(pos) * pCTR(click | x, seen)

Two modules, a position module and a CTR module, are trained jointly on the click label.
At serving only the CTR module ranks candidates, so the system never needs to know the
position before choosing it. That breaks the chicken-and-egg problem of using position as a
feature in a ranker that decides positions. In online A/B tests PAL reported gains in CTR
and conversion rate over a "position as feature" baseline. The multiplicative form is the
probabilistic twin of YouTube's additive-logit tower.

### 14.2.4 Airbnb: position features with dropout in search ranking

**System.** Airbnb's search ranking moved from GBDT to deep networks in 2018-2019 [6]. In
"Improving Deep Learning for Airbnb Search" (Haldar et al., KDD 2020) [5] the team describes
its next round of changes, including handling of position bias.

**Problem.** Guest bookings depend heavily on where a listing appears. Airbnb is a
two-sided marketplace with limited inventory, so a model that learns "top position means
good listing" also entrenches incumbents and hurts new hosts.

**Design.** Airbnb added the listing's search position as a training feature and applied
dropout to it during training. At inference the position feature is set to a constant
(zero), so all candidates are scored as if they sat in the same position. The paper treats
the dropout rate as a regularisation knob. Too little dropout and the model over-relies on
position, which hurts offline relevance with position removed. Too much and the model cannot
separate position effects. The team tuned the rate to balance the two and reported a
statistically significant booking gain online.

**Lessons.**

- Training with "position as feature" and serving with "position = constant" is the
  simplest debiasing you can deploy. It needs no propensity estimation.
- In marketplaces, position debiasing is also a fairness and cold-start lever. It lowers
  the advantage that listings get from having ranked well in the past.

### 14.2.5 Comparison of position-debiasing designs

| Design | Org | Bias model | Combination | Serving | Needs randomisation? |
|---|---|---|---|---|---|
| IPS-LTR [8] | Cornell / academic | Propensity per rank | Loss weight 1/p_k | Unchanged ranker | Yes (or harvested [29]) |
| Regression-EM [28] | Google personal search | Propensity per rank | Loss weight | Unchanged ranker | No |
| Shallow tower [1] | YouTube | Small net on position + device | Additive logit | Position missing | No |
| PAL [9] | Huawei App Store | pSeen(pos) module | Multiplicative probability | CTR module only | No |
| Position feature + dropout [5] | Airbnb search | Position as input feature | Inside main DNN | Position = 0 | No |

---

## 14.3 Popularity, Exposure and Sampling Bias

### 14.3.1 Popularity bias and the long tail

Abdollahpouri et al. [15] show that common collaborative filtering algorithms not only
over-recommend popular items but also serve users with niche tastes worse than mainstream
users, because their recommendations drift toward the head. Earlier work by the same group
[16] added a regulariser to a learning-to-rank objective that penalises lists made only of
head items. This made an explicit accuracy versus long-tail-coverage trade-off.

Causal methods treat popularity as a confounder:

- **PDA (Popularity-bias Deconfounding and Adjusting)**, Zhang et al., SIGIR 2021 [17].
  Item popularity affects both exposure and the click label. PDA trains a deconfounded
  model by removing the confounding popularity effect through backdoor adjustment. It then
  optionally injects *predicted future* popularity at inference. The point is that some
  popularity is real signal (trending items) and should be kept on purpose, not by accident.
- **MACR**, Wei et al., KDD 2021 [18]. Builds a causal graph with user, item and
  user-item-matching branches. Training uses multitask learning with auxiliary item-only and
  user-only modules. At inference, counterfactual reasoning subtracts the direct item
  (popularity) effect from the prediction.

### 14.3.2 Sampling bias in two-tower retrieval: logQ correction (YouTube)

In-batch softmax for two-tower retrieval (Chapter 2) treats the other items in a batch as
negatives. Popular items show up in batches far more often, so they are over-penalised as
negatives. That would bias the model against the head. Yi et al., RecSys 2019 [11], built
YouTube's retrieval around **logQ correction**:

    s_c(x, y_j) = s(x, y_j) - log(p_j)

Here p_j is the probability that item j appears in a batch. Because the training data
streams, p_j cannot be counted offline. The paper therefore introduces a
**streaming frequency estimator**: a hash array that records the last global step at which
each item was seen and keeps a moving average of the gap between sightings (delta). The
estimate is p_j ≈ 1 / delta. The design was deployed for YouTube candidate generation over a
corpus of (reported) tens of millions of videos. LogQ correction is now standard in almost
every in-batch two-tower system.

**Design lesson.** Popularity bias can enter through the *training procedure* (negative
sampling) as well as through the data. Check the sampler before blaming the logs.

### 14.3.3 Exposure bias and off-policy correction

Implicit feedback mixes "did not like" with "never saw". Liang et al. [14] model exposure as
a latent variable (ExpoMF): a missing interaction counts as negative evidence only to the
degree that the user was probably exposed to the item. Schnabel et al. [13] frame
recommendation as treatments and use propensity-scored estimators for both training and
evaluation of rating models under missing-not-at-random data.

At industry scale, YouTube's REINFORCE recommender (Chen et al., WSDM 2019, "Top-K
Off-Policy Correction") [31] tackles the exposure bias of training a policy on logs produced
by earlier policies:

- A separate head estimates the **behaviour policy** beta(a|s), meaning the probability
  that the logged system showed item a. Gradients are importance-weighted by pi/beta, with
  weight capping to control variance.
- A **top-K correction** multiplier, lambda_K = K * (1 - pi)^(K-1), accounts for the fact
  that the policy recommends a slate of K items, not a single one.
- The paper reports that the off-policy-corrected agent increased viewing time in live
  experiments, and that the gains came disproportionately from items that the earlier
  system rarely surfaced. The correction expands exposure toward the tail.

Chapter 13 covers RL and exploration in more depth.

### 14.3.4 YouTube's "example age" feature: debiasing toward the past

Covington et al. [12] made a simpler but influential observation. A model trained on
historical windows learns the *average* popularity of an item over the training window.
That under-predicts fresh, viral uploads. YouTube added the **age of the training example**
as a feature during training and set it to zero (or slightly negative) at serving. This
lets the model represent time-dependent popularity and shows "now" at serving. The same
paper also takes a **fixed number of training examples per user**, so that a small set of
heavy users does not dominate the loss, which is a form of reweighting against activity
bias.

### 14.3.5 Duration bias in watch time: Kuaishou D2Q

Short-video platforms optimise watch time. Longer videos naturally collect more watch time,
so a naive regressor favours length. Kuaishou's D2Q (Zhan et al., KDD 2022) [27] treats
video duration as a confounder that affects both exposure and watch time:

```
      duration --------> exposure (system shows long videos more)
         |                  |
         v                  v
               watch time (label)

  D2Q: split training data into duration buckets
       -> within each bucket, convert watch time into a quantile label
       -> fit one model (shared parameters) predicting the within-bucket quantile
       -> serve on quantiles: compares "how well this video holds attention
          relative to similar-length videos"
```

The paper reports online gains on Kuaishou over naive watch-time regression and over
value-weighted baselines. Later short-video work builds on the same "normalise by length
group" idea.

---

## 14.4 Diversity: From MMR to Learned Listwise Rankers

### 14.4.1 Classic re-ranking: MMR and topic diversification

**Maximal Marginal Relevance** (Carbonell and Goldstein, SIGIR 1998) [10] builds the list
greedily:

    next = argmax_{i not in S} [ lambda * rel(i) - (1 - lambda) * max_{j in S} sim(i, j) ]

MMR is O(N * K) similarity look-ups for K picks from N candidates. It is easy to run on the
re-ranking tier and is still the default diversity baseline in many systems, including
RAG retrieval today. Ziegler et al. [30] introduced **intra-list similarity** and topic
diversification for book recommendations. User studies showed that users liked lists that
were diversified even when accuracy metrics dropped slightly. That is an early piece of
evidence that offline accuracy is the wrong only-metric.

Limitations of MMR: it only penalises the *maximum* pairwise similarity, it has no
probabilistic meaning, and lambda must be tuned per surface.

### 14.4.2 Determinantal point processes

A DPP over a ground set of items defines P(S) proportional to det(L_S), where L is a
positive semi-definite kernel. A common decomposition is

    L_ij = q_i * q_j * S_ij

where q_i is item quality (relevance) and S_ij is similarity. The determinant grows with the
qualities and shrinks as items become collinear, so one score covers both relevance and
set-level diversity. It rewards whole-set "volume", not only pairwise distance.

#### YouTube: practical DPP for the home feed

Wilhelm et al., "Practical Diversified Recommendations on YouTube with Determinantal Point
Processes" (CIKM 2018) [2], describe how YouTube put a DPP into its mobile home feed.

**Problem.** The pointwise ranker scores each video on its own, so a feed can fill up with
near-duplicate content (for example, several videos on the same topic or from the same
creator). Users then scroll past the whole block, and a pointwise ranker never sees that
cost.

**Architecture.**

```
 candidate generators --> pointwise ranker (q_i = predicted utility)
                                   |
                                   v  top-N ranked candidates (N ~ a few hundred, approx.)
                    +-------------------------------+
                    | DPP re-ranker                 |
                    |  kernel:                      |
                    |  L_ii = q_i^2                 |
                    |  L_ij = alpha*q_i*q_j*        |
                    |        exp(-D_ij / (2 s^2))   |
                    |  D_ij from video embeddings / |
                    |  topical distance             |
                    |  greedy MAP, window by window |
                    +-------------------------------+
                                   |
                                   v
                   feed served in windows of k items; next window
                   chosen conditioned on items already selected
```

**Key tricks.**

- **Parameterised kernel with two knobs.** alpha trades relevance against diversity, and
  sigma sets how fast similarity decays with distance. Early versions found the values by
  grid search against online metrics. Later the paper describes learning a deep kernel from
  feed interactions, treating the set of items the user engaged with in a window as a DPP
  sample.
- **Windowed greedy inference.** The feed is long. Instead of one global MAP, the system
  greedily picks a window of items (roughly the number visible on a phone screen), fixes
  it, and moves to the next window. This keeps latency bounded.
- **Engagement as the objective.** The paper reports that DPP-diversified feeds raised
  long-term user satisfaction and engagement metrics compared with heuristic diversity
  rules such as per-channel caps, which YouTube had used before.

#### Hulu: fast greedy MAP inference

Exact DPP MAP is NP-hard, and naive greedy costs O(M * N^3) for selecting N items from M if
determinants are recomputed. Chen, Zhang and Zhou (Hulu, NeurIPS 2018) [3] made greedy DPP
practical:

- **Incremental Cholesky updates.** Each time an item is added, the marginal gain of every
  remaining candidate is updated in O(N) work per candidate, giving O(N^2 * M) total. That
  is fast enough for online re-ranking at hundreds of candidates and tens of selected items.
- **Sliding-window variant.** For long sequences, diversity is enforced only within a window
  of the last w items. This matches how users perceive a scrolling feed and lowers cost
  further.
- Experiments showed that the method was orders of magnitude faster than earlier greedy
  implementations and gave a better relevance-diversity frontier than MMR and other
  baselines. The paper also reports an online A/B test in a production movie-recommendation
  setting.

This algorithm (often called "fast greedy DPP") is the implementation most teams use in
practice, including several large short-video and e-commerce platforms that describe
DPP-style re-ranking.

### 14.4.3 Airbnb: learning to rank diversely

Airbnb's search team (Abdool et al., "Learning To Rank Diversely At Airbnb") [7] points out a
basic gap in pairwise learning to rank. The loss assumes each listing's utility does not
depend on what sits above it. In a marketplace, if the top 5 results are all similar
high-priced lofts, a sixth loft adds little.

**Design.** The team keeps the trained pairwise relevance model and adds a model that scores
each listing *in the context of the listings already ranked above it*. It learns a
representation of the "query context of results so far" and ranks listing by listing
(sequentially), so that a listing similar to those already placed is de-prioritised when the
data shows that such redundancy lowers bookings. The paper presents this as moving from
pointwise or pairwise to a listwise, context-aware formulation without hand-tuned diversity
weights, and reports a statistically significant booking gain in online A/B tests.

**Lesson.** Diversity does not have to be a heuristic re-ranker bolted on after the model. It
can be learned from outcome data once the model sees the slate context. This is the same
direction taken by generative, slate-aware rankers (Chapter 6).

### 14.4.4 Choosing a diversity mechanism

| Mechanism | Objective | Cost per request | Tunability | Learned? | Production users (public) |
|---|---|---|---|---|---|
| Hard rules (per-creator cap, category quota) | Constraint | O(N) | Very easy | No | Nearly everyone |
| MMR [10] | Relevance minus max-sim | O(NK) | 1 knob | Rarely | Search, RAG, feeds |
| Fast greedy DPP [3] | log det of quality-similarity kernel | O(K^2 N) | alpha, sigma | Kernel can be learned [2] | YouTube [2], Hulu [3] |
| Calibration [4] | Match user's genre mix (KL) | O(NK * genres) | lambda | No | Netflix research |
| Listwise context model [7] | Outcome (bookings) given items above | Model inference per position | Implicit | Yes | Airbnb |

---

## 14.5 Calibrated Recommendations (Netflix)

**Problem.** Harald Steck (Netflix, RecSys 2018) [4] gives a simple example. A user watched
70% romance and 30% action. An accuracy-optimised top-10 list may be 100% romance, because
romance items score highest one by one. The list is accurate but *mis-calibrated*: the
user's minority interest disappears. Over time the feedback loop can wipe that interest out
of the profile altogether.

**Formulation.**

- p(g | u): the target distribution over genres g, computed from the user's watch history.
  Each item's genre mass is spread across its genres, and items can be weighted, for
  example by recency.
- q(g | u): the genre distribution of the recommended list.
- Mis-calibration is measured with KL divergence, C_KL(p, q) = sum_g p(g|u) log(p(g|u) /
  q~(g|u)). Here q~ = (1 - a) * q + a * p with a small a (the paper uses a value around
  0.01) so that the divergence stays finite when the list misses a genre.

**Re-ranking objective.**

    I* = argmax_{I, |I| = N} (1 - lambda) * sum_{i in I} s(u, i) - lambda * C_KL(p, q(I))

Steck notes that the objective is close to submodular. A greedy algorithm that adds the
item with the largest marginal gain therefore has the usual (1 - 1/e) approximation
guarantee and runs fast enough for post-processing. Experiments on MovieLens 20M show that
a small lambda removes most of the mis-calibration at a modest accuracy cost. Lists from a
standard model trained for accuracy came out *more* mis-calibrated for users with diverse
tastes.

**Why it matters for system design.**

- Calibration is a *personalised* form of diversity. The target mix comes from the user, not
  from a global "be diverse" prior.
- The same machinery can calibrate along any categorical axis that matters to the product:
  content type (movies vs series), creator popularity bucket (popular vs emerging), language,
  or freshness. Many later "calibrated" papers do exactly that.
- It is a cheap re-ranker over the top few hundred candidates and needs no retraining.

---

## 14.6 Cold Start

Cold start has three cases with different solutions: **new users**, **new items**, and
**new creators/sellers**. The last is item cold start plus a marketplace-health goal.

```
             +-------------------+      +------------------------+
 new item -->| content encoders  |----> | ID embedding warm-up   |--> ranker
 (meta,     | (text, image,     |      | (meta-embedding [25],   |
  media)    |  audio, creator)  |      |  DropoutNet [24])       |
             +-------------------+      +------------------------+
                      |                             ^
                      v                             |
            exploration traffic (bandits [32], boosted slots, priors [5])
                      |
                      v
            real-time training (minutes) -> ID embedding converges

 new user --> onboarding signals + context (geo, device, referrer)
          --> cohort-specific gating (POSO [26]) --> popular-within-cohort fallback
```

### 14.6.1 Item cold start with content and dropout

**DropoutNet** (Volkovs et al., Layer 6 AI, NeurIPS 2017) [24] trains a network that takes
both preference embeddings (from collaborative filtering) and content features as input.
During training it **randomly drops out the preference input** for some users and items.
The network is forced to rebuild the warm-start relevance score from content alone, so the
same model works for warm, cold and partly cold entities with no separate cold model. This
idea, input dropout on ID/CF features, carries over directly to modern two-tower and DLRM
systems: drop ID embeddings for some examples so that content and side-feature towers learn
to stand alone.

### 14.6.2 Meta-learned ID embeddings for new ads

Pan et al. (SIGIR 2019, Tencent-related ad data) [25] attack the new-ad problem in CTR
models. A new ad's randomly initialised ID embedding is noise until it has gathered enough
clicks. **Meta-Embedding** trains a generator that maps an ad's attributes (category,
advertiser, and so on) to an initial ID embedding. It uses a two-phase, MAML-style objective:
the generated embedding should (a) predict well in the cold phase and (b) adapt fast after a
few gradient steps on early data. The paper reports better CTR prediction for new ads in both
the cold and warm-up phases on large ad datasets. Production descendants appear across ad
systems: initialise from content or neighbours, never from zero.

### 14.6.3 Airbnb: priors for new listings

The 2020 Airbnb paper [5] describes the cold-start issue for new listings. Engagement
features such as historical booking rate are missing or noisy, so the DNN ranks new listings
poorly, and fewer impressions mean even less data. Airbnb's fix was to **estimate the
missing engagement features for new listings from similar nearby listings** (same
geography, capacity and so on) instead of using defaults. The paper reports a large relative
booking gain *for new listings* in online tests (reported as approximately double-digit
percent) and a smaller but significant overall booking gain. Combined with the position
debiasing above (14.2.4), this is a clear example of debiasing and cold start being the same
marketplace problem.

### 14.6.4 New users: Kuaishou POSO

Kuaishou's POSO (Personalized Cold Start Modules, Dai et al., 2021) [26] identifies a
structural problem. In a large shared ranking model, signals from new users are swamped by
the majority of active users. Even when "is_new_user" is an input feature, the network
mostly ignores it because it is one feature among hundreds. POSO adds **user-group-specific
gating** (personalised modules) to existing layers such as MLP, MHA and MMoE. A small gating
network driven by user-cohort features (new vs regular, activity level) rescales hidden
units, so cohort differences shape the whole computation. Kuaishou reports gains in
new-user engagement and retention metrics in production, at negligible serving cost.

**Pattern.** Give minority cohorts a *structural* path through the network (gates, separate
experts, separate towers), not just an input feature.

### 14.6.5 Exploration for cold start: contextual bandits

When features cannot fill the gap, the system must *explore*. Li et al.'s **LinUCB**
(Yahoo!, WWW 2010) [32] scored news articles with a linear model plus an upper-confidence
bonus alpha * sqrt(x^T A^-1 x). New articles have high uncertainty, so they automatically
get exploratory traffic. On the Yahoo! front page the paper reports a click lift (roughly
12.5%) over a context-free bandit, using offline replay evaluation on randomly logged
traffic. Chapter 13 covers modern neural bandits and exploration budgets.

### 14.6.6 Real-time training as a cold-start tool

Infrastructure matters too. ByteDance's **Monolith** [33] (collisionless embedding tables
with expiry, plus online training that syncs parameters to serving at minute-level
granularity) shortens the time a new item or user spends with an untrained embedding from
hours or days to minutes. For fast-churning catalogues such as short video or news, the
freshness of the training loop is often the biggest cold-start lever (see Chapter 9).

### 14.6.7 Cold-start strategy matrix

| Entity | Signal available | Primary technique | Secondary | Example |
|---|---|---|---|---|
| New item | Content, creator, metadata | Content tower / meta-embedding [25] | ID dropout [24], fast online training [33] | Ads CTR, short video |
| New user | Context, onboarding choices | Cohort gating [26], popular-in-cohort | Bandits [32], onboarding questionnaire | Kuaishou, news |
| New creator / seller | Content + creator profile | Boosted exploration slots | Priors from similar suppliers [5] | Airbnb hosts, marketplaces |
| New surface / market | Transfer from other surfaces | Shared embeddings, multi-domain models | Calibration to local mix [4] | Any multi-product platform |

---

## 14.7 Fairness for Creators, Sellers and Users

In multi-sided platforms, "fairness" usually means **fair exposure for suppliers**: creators,
artists, hosts, sellers and job candidates. User-side group fairness (for example, equal
recommendation quality across demographic groups) is the other main form.

### 14.7.1 Fairness of exposure: the theory

Singh and Joachims (KDD 2018) [20] define fairness constraints on the *exposure* that
ranking policies give to groups. Exposure is the position-weighted attention, using the same
position-bias curves as 14.2. Examples are demographic parity of exposure and
exposure proportional to merit (relevance). The optimal fair ranking policy is computed as a
linear program over doubly stochastic matrices, then decomposed (Birkhoff-von Neumann) into
a distribution over rankings. Position bias therefore drives both relevance debiasing *and*
fairness: the same propensity curve sets how much exposure each slot is worth.

FA*IR (Zehlike et al.) [35] is a simpler, top-k ranking algorithm. It guarantees that every
prefix of the ranking holds at least a minimum share of a protected group, checked with a
binomial significance test, while staying as close as possible to the score order.

### 14.7.2 LinkedIn Recruiter: fairness-aware re-ranking at scale

**Problem.** In LinkedIn Talent Search, recruiters search for candidates. LinkedIn wanted the
gender distribution of top results to match the gender distribution of the qualified
candidate pool for that query, a "representative" ranking.

**Design** (Geyik, Ambler, Kenthapadi, KDD 2019) [21]:

```
 query --> candidate retrieval --> ML ranker (relevance scores)
                                         |
                                         v
            desired distribution p_q(attribute) = distribution among candidates
            matching the query (e.g., qualified pool), computed per query
                                         |
                                         v
            fairness re-ranker: per-attribute queues sorted by score;
            fill ranks greedily so that every prefix k has at least
            floor(p_a * k) and at most ceil(p_a * k) of each attribute a
            (DetGreedy / DetCons / DetRelaxed variants)
                                         |
                                         v
                              top-k shown to recruiter
```

**Key properties.**

- **Post-processing, model-agnostic.** The ranker is untouched. That made the change easy to
  ship and audit.
- **Per-query targets.** The target is the attribute mix of the *qualified* candidates for
  that query, not a global quota.
- **Prefix constraints.** Fairness holds at every k the recruiter might look at, not only in
  aggregate.
- **Results.** The paper reports deployment to all LinkedIn Recruiter users worldwide and a
  large increase (reported as nearly three-fold) in the share of queries with representative
  results, with no significant drop in business metrics in A/B tests.

### 14.7.3 Spotify: relevance, fairness and satisfaction trade-offs

Mehrotra et al. (Spotify, CIKM 2018) [22] studied a music recommender as a two-sided
marketplace. Artists differ widely in popularity, and a pure relevance policy pushes exposure
to the head. Using **counterfactual (off-policy) evaluation** on logged data, they estimated
user satisfaction under policies that mix relevance with a supplier-fairness score
(exposure across artist popularity groups). Main findings:

- Pure fairness policies hurt satisfaction. Pure relevance policies concentrate exposure.
- Users differ in how open they are to less-popular artists. **Adaptive policies** that give
  fairness more weight for users with broader tastes reached better trade-offs than global
  interpolation.

This links fairness to calibration (14.5). Pushing exposure toward emerging suppliers works
best when it follows each user's demonstrated appetite.

### 14.7.4 Google: pairwise fairness in production recommenders

Beutel et al. (Google, KDD 2019) [23] define **pairwise fairness** for recommenders. Among
pairs where one item was engaged with and one was not, the model should rank the engaged item
higher with equal probability whether the item belongs to group A or group B (with
intra-group and inter-group variants). Measuring this needs data where both items were
actually exposed, which they get from randomised experiments. They add a **pairwise
regulariser** that penalises the correlation between group membership and residual pairwise
error. They report that, in a large-scale production recommender at Google, the regulariser
greatly improved inter-group pairwise fairness with little engagement loss.

### 14.7.5 User-side fairness

Yao and Huang [36] define unfairness metrics for collaborative filtering across user groups,
for example under- or over-estimating ratings for a minority group, and add them as
regularisers. Abdollahpouri et al. [15] show that popularity bias is itself a user-fairness
issue, because niche-taste users get worse service. In practice, user-side fairness is
monitored by slicing evaluation metrics by cohort (Chapter 12) and fixed with structural
cohort modelling such as POSO [26].

### 14.7.6 Fairness design options compared

| Approach | Where | Guarantee | Cost | Examples |
|---|---|---|---|---|
| Prefix-constrained re-ranking | Post-ranking | Hard, per prefix | Low | LinkedIn [21], FA*IR [35] |
| Exposure LP / stochastic ranking | Post-ranking | Expected exposure | Medium (LP per query) | Singh and Joachims [20] |
| Regulariser in training | Model | Soft, metric-level | Training only | Google pairwise [23], Yao and Huang [36] |
| Adaptive mixing per user | Policy | Soft, user-aware | Low | Spotify [22] |
| Exploration boosts for new suppliers | Serving | Soft | Exploration budget | Marketplaces, creator platforms |

---

## 14.8 Putting It Together: A Reference Corrective Stack

A typical large-scale feed or marketplace stack that combines the designs above:

```
 [Training data]
   impressions + engagements + position + device + example age + duration
     |
     |-- reweight: fixed examples per user [12]; IPS for explicit ratings [13]
     v
 [Retrieval: two-tower]
   in-batch softmax with logQ correction via streaming frequency estimator [11]
   ID dropout so content towers handle cold items [24]
   meta-embedding initialisation for new IDs [25]
     |
     v  ~ thousands of candidates
 [Ranker: multitask DNN / MMoE]
   main towers (engagement, satisfaction)
   + shallow position/device tower, additive logit, position dropout [1][5]
   duration-bucketed quantile targets for watch time [27]
   cohort gates for new users [26]
   example_age = 0 and position = missing at serving [12][1]
     |
     v  ~ hundreds of candidates
 [Re-ranking / slate construction]
   fast greedy DPP or MMR within sliding windows [2][3][10]
   calibration to user's genre / creator-popularity mix [4]
   fairness prefix constraints where policy requires [21]
   exploration slots for cold items/creators (bandit bonus) [32]
     |
     v
 [Serving + logging]  log true positions, exploration flags, propensities
                      (needed by every debiasing method above)
```

---

## Key Takeaways for System Designers

1. **Log what the debiasers need.** Position, device, surface, exploration flag, sampling
   probability and (for off-policy work) the serving policy's propensity must be in the
   impression logs. Without them almost none of this chapter is possible.
2. **Prefer factorised bias towers to raw IPS in industrial rankers.** YouTube's additive
   shallow tower [1], Huawei's multiplicative PAL [9] and Airbnb's position-feature-with-
   dropout [5] share one recipe: let a small component absorb the bias in training, then
   neutralise it at serving. Dropout on the bias feature is what makes neutralisation safe.
3. **Popularity bias hides in the training procedure.** In-batch negatives need logQ
   correction [11]. Training windows need an example-age feature [12]. Watch-time targets
   need duration normalisation [27].
4. **Treat diversity as a slate-level property.** Pointwise rankers cannot see redundancy.
   Use DPP or MMR with windowing for cheap gains [2][3]. Move to learned, context-aware
   listwise models once you can [7].
5. **Calibrate to the user, not to a global prior.** Netflix's calibration [4] and Spotify's
   adaptive fairness [22] both show that the best trade-off follows each user's demonstrated
   breadth of interest.
6. **Cold start is a structural problem, not just a missing feature.** Use content towers
   with ID dropout [24], meta-learned embedding initialisation [25], cohort-specific gating
   [26], neighbour-based priors [5], and fast online training [33]. Keep an explicit
   exploration budget [32].
7. **Fairness of exposure reuses the position-bias curve.** The same propensity model that
   debiases clicks tells you how much exposure each slot gives. Measure supplier exposure
   with it [20].
8. **Ship fairness as auditable post-processing first.** LinkedIn's per-query, prefix-
   constrained re-ranker [21] was model-agnostic, explainable and deployed globally with no
   measurable business loss.
9. **Evaluate counterfactually and by cohort.** Off-policy estimators [13][22][31] and
   cohort-sliced metrics (new users, niche users, emerging creators) catch problems that
   aggregate NDCG hides.
10. **Expect long-term rather than immediate wins.** Diversity, calibration and fair exposure
    often look neutral or slightly negative on short-term engagement. The reported gains in
    the YouTube DPP work [2] and the off-policy REINFORCE work [31] show up in satisfaction,
    retention and tail coverage. Design experiments long enough to measure them.

---

## References

1. Zhe Zhao, Lichan Hong, Li Wei, Jilin Chen, Aniruddh Nath, Shawn Andrews, Aditee Kumthekar, Maheswaran Sathiamoorthy, Xinyang Yi, Ed Chi (Google/YouTube). "Recommending What Video to Watch Next: A Multitask Ranking System." RecSys 2019. https://doi.org/10.1145/3298689.3346997
2. Mark Wilhelm, Ajith Ramanathan, Alexander Bonomo, Sagar Jain, Ed H. Chi, Jennifer Gillenwater (Google/YouTube). "Practical Diversified Recommendations on YouTube with Determinantal Point Processes." CIKM 2018. https://doi.org/10.1145/3269206.3272018
3. Laming Chen, Guoxin Zhang, Hanning Zhou (Hulu). "Fast Greedy MAP Inference for Determinantal Point Process to Improve Recommendation Diversity." NeurIPS 2018. https://arxiv.org/abs/1709.05135
4. Harald Steck (Netflix). "Calibrated Recommendations." RecSys 2018. https://doi.org/10.1145/3240323.3240372
5. Malay Haldar et al. (Airbnb). "Improving Deep Learning For Airbnb Search." KDD 2020. https://arxiv.org/abs/2002.05515
6. Malay Haldar et al. (Airbnb). "Applying Deep Learning To Airbnb Search." KDD 2019. https://arxiv.org/abs/1810.09591
7. Mustafa Abdool et al. (Airbnb). "Learning To Rank Diversely At Airbnb." 2022 (CIKM 2023). https://arxiv.org/abs/2210.07774
8. Thorsten Joachims, Adith Swaminathan, Tobias Schnabel (Cornell). "Unbiased Learning-to-Rank with Biased Feedback." WSDM 2017. https://arxiv.org/abs/1608.04468
9. Huifeng Guo, Jinkai Yu, Qing Liu, Ruiming Tang, Yuzhou Zhang (Huawei). "PAL: A Position-bias Aware Learning Framework for CTR Prediction in Live Recommender Systems." RecSys 2019. https://doi.org/10.1145/3298689.3347033
10. Jaime Carbonell, Jade Goldstein (CMU). "The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries." SIGIR 1998. https://doi.org/10.1145/290941.291025
11. Xinyang Yi, Ji Yang, Lichan Hong, Derek Zhiyuan Cheng, Lukasz Heldt, Aditee Kumthekar, Zhe Zhao, Li Wei, Ed Chi (Google/YouTube). "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations." RecSys 2019. https://doi.org/10.1145/3298689.3346996
12. Paul Covington, Jay Adams, Emre Sargin (Google/YouTube). "Deep Neural Networks for YouTube Recommendations." RecSys 2016. https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf
13. Tobias Schnabel, Adith Swaminathan, Ashudeep Singh, Navin Chandak, Thorsten Joachims. "Recommendations as Treatments: Debiasing Learning and Evaluation." ICML 2016. https://arxiv.org/abs/1602.05352
14. Dawen Liang, Laurent Charlin, James McInerney, David M. Blei. "Modeling User Exposure in Recommendation." WWW 2016. https://arxiv.org/abs/1510.07025
15. Himan Abdollahpouri, Masoud Mansoury, Robin Burke, Bamshad Mobasher. "The Unfairness of Popularity Bias in Recommendation." RMSE workshop at RecSys 2019. https://arxiv.org/abs/1907.13286
16. Himan Abdollahpouri, Robin Burke, Bamshad Mobasher. "Controlling Popularity Bias in Learning-to-Rank Recommendation." RecSys 2017. https://doi.org/10.1145/3109859.3109912
17. Yang Zhang, Fuli Feng, Xiangnan He, Tianxin Wei, Chonggang Song, Guohui Ling, Yongdong Zhang. "Causal Intervention for Leveraging Popularity Bias in Recommendation." SIGIR 2021. https://arxiv.org/abs/2105.06067
18. Tianxin Wei, Fuli Feng, Jiawei Chen, Ziwei Wu, Jinfeng Yi, Xiangnan He. "Model-Agnostic Counterfactual Reasoning for Eliminating Popularity Bias in Recommender System." KDD 2021. https://arxiv.org/abs/2010.15363
19. Jiawei Chen, Hande Dong, Xiang Wang, Fuli Feng, Meng Wang, Xiangnan He. "Bias and Debias in Recommender System: A Survey and Future Directions." ACM TOIS 2023 (arXiv 2020). https://arxiv.org/abs/2010.03240
20. Ashudeep Singh, Thorsten Joachims. "Fairness of Exposure in Rankings." KDD 2018. https://arxiv.org/abs/1802.07281
21. Sahin Cem Geyik, Stuart Ambler, Krishnaram Kenthapadi (LinkedIn). "Fairness-Aware Ranking in Search and Recommendation Systems with Application to LinkedIn Talent Search." KDD 2019. https://arxiv.org/abs/1905.01989
22. Rishabh Mehrotra, James McInerney, Hugues Bouchard, Mounia Lalmas, Fernando Diaz (Spotify). "Towards a Fair Marketplace: Counterfactual Evaluation of the trade-off between Relevance, Fairness and Satisfaction in Recommendation Systems." CIKM 2018. https://doi.org/10.1145/3269206.3272027
23. Alex Beutel, Jilin Chen, Tulsee Doshi, Hai Qian, Li Wei, Yi Wu, Lukasz Heldt, Zhe Zhao, Lichan Hong, Ed H. Chi, Cristos Goodrow (Google). "Fairness in Recommendation Ranking through Pairwise Comparisons." KDD 2019. https://arxiv.org/abs/1903.00780
24. Maksims Volkovs, Guangwei Yu, Tomi Poutanen (Layer 6 AI). "DropoutNet: Addressing Cold Start in Recommender Systems." NeurIPS 2017. https://papers.nips.cc/paper/7081-dropoutnet-addressing-cold-start-in-recommender-systems
25. Feiyang Pan, Shuokai Li, Xiang Ao, Pingzhong Tang, Qing He. "Warm Up Cold-start Advertisements: Improving CTR Predictions via Learning to Learn ID Embeddings." SIGIR 2019. https://arxiv.org/abs/1904.11547
26. Shangfeng Dai, Haobin Lin, Zhichen Zhao, Jianying Lin, Honghuan Wu, Zhe Wang, Sen Yang, Ji Liu (Kuaishou). "POSO: Personalized Cold Start Modules for Large-scale Recommender Systems." 2021. https://arxiv.org/abs/2108.04690
27. Ruohan Zhan et al. (Kuaishou). "Deconfounding Duration Bias in Watch-time Prediction for Video Recommendation." KDD 2022. https://arxiv.org/abs/2206.06003
28. Xuanhui Wang, Nadav Golbandi, Michael Bendersky, Donald Metzler, Marc Najork (Google). "Position Bias Estimation for Unbiased Learning to Rank in Personal Search." WSDM 2018. https://doi.org/10.1145/3159652.3159732
29. Aman Agarwal, Ivan Zaitsev, Xuanhui Wang, Cheng Li, Marc Najork, Thorsten Joachims. "Estimating Position Bias without Intrusive Interventions." WSDM 2019. https://arxiv.org/abs/1812.05161
30. Cai-Nicolas Ziegler, Sean M. McNee, Joseph A. Konstan, Georg Lausen. "Improving Recommendation Lists Through Topic Diversification." WWW 2005. https://doi.org/10.1145/1060745.1060754
31. Minmin Chen, Alex Beutel, Paul Covington, Sagar Jain, Francois Belletti, Ed H. Chi (Google/YouTube). "Top-K Off-Policy Correction for a REINFORCE Recommender System." WSDM 2019. https://arxiv.org/abs/1812.02353
32. Lihong Li, Wei Chu, John Langford, Robert E. Schapire (Yahoo! Research). "A Contextual-Bandit Approach to Personalized News Article Recommendation." WWW 2010. https://arxiv.org/abs/1003.0146
33. Zhuoran Liu et al. (ByteDance). "Monolith: Real Time Recommendation System With Collisionless Embedding Table." 2022. https://arxiv.org/abs/2209.07663
34. Xuanhui Wang, Michael Bendersky, Donald Metzler, Marc Najork (Google). "Learning to Rank with Selection Bias in Personal Search." SIGIR 2016. https://doi.org/10.1145/2911451.2911537
35. Meike Zehlike, Francesco Bonchi, Carlos Castillo, Sara Hajian, Mohamed Megahed, Ricardo Baeza-Yates. "FA*IR: A Fair Top-k Ranking Algorithm." CIKM 2017. https://arxiv.org/abs/1706.06368
36. Sirui Yao, Bert Huang. "Beyond Parity: Fairness Objectives for Collaborative Filtering." NeurIPS 2017. https://arxiv.org/abs/1705.08804
