# 12. Evaluation and Experimentation: Offline Metrics, Off-Policy Evaluation, A/B Platforms, Interleaving and Long-Term Measurement

> Research note: live web search and page fetching were unavailable when this chapter
> was compiled. The search quota was used up and outbound DNS to the source sites failed.
> Everything below comes from primary sources (papers, official engineering blogs, books)
> as the author already knew them, not from a fresh crawl. Numbers marked "approx." should
> be checked against the cited source before anyone quotes them. URLs are listed only where
> the author was confident they are canonical.

## Overview

The quality of a recommender is only as believable as the measurement behind it. Every mature recommendation organization covered in this atlas, including Netflix, Airbnb, LinkedIn, Spotify, Google/YouTube, Microsoft, Booking.com, Criteo and Uber, has ended up with the same layered evaluation stack:

1. **Offline metrics.** NDCG, recall@K, MAP, AUC, log loss / normalized entropy and calibration are computed on held-out logs. They are cheap and fast, but biased by the logging policy. Teams use them to screen ideas, not to make launch decisions.
2. **Counterfactual / off-policy evaluation (OPE).** IPS, self-normalized IPS, the direct method and doubly robust estimators reweight logged interactions to estimate how a *new* policy would have performed. They need logged propensities and enough exploration.
3. **Interleaving.** Two rankers are merged into one list for the same user, and attributed engagement shows which one the user prefers. It is roughly one to two orders of magnitude more sensitive than A/B testing for ranking changes, and teams use it to triage candidates quickly.
4. **A/B testing.** Randomized controlled experiments measure business metrics such as retention, streaming hours, bookings and sessions. This is the final arbiter, run on platforms like Netflix's ABlaze, Airbnb's ERF, LinkedIn's XLNT/T-REX, Spotify's Experimentation Platform/Confidence, Google's overlapping layers and Microsoft ExP.
5. **Long-term measurement.** Holdouts, surrogate indices and long-term proxy metrics address the fact that most launches are decided on 1 to 4 weeks of data, while the metrics that matter (retention, LTV) move over months.

```
   COST / LATENCY OF A DECISION                         FIDELITY TO BUSINESS GOAL
   low                                                                       high
   +-----------+   +-------------+   +--------------+   +---------+   +-----------+
   | Offline   |-->| Off-policy  |-->| Interleaving |-->|  A/B    |-->| Long-term |
   | metrics   |   | evaluation  |   | (ranking)    |   |  test   |   | holdout / |
   | NDCG,AUC  |   | IPS, DR     |   | team-draft   |   | 1-4 wks |   | surrogate |
   +-----------+   +-------------+   +--------------+   +---------+   +-----------+
   minutes-hours     hours             days               weeks          months
   1000s of ideas    100s              10s                handful        portfolio
```

The funnel is the main architectural idea of this chapter. Each stage is cheaper but less faithful than the next, so it is used to cut down the candidate set for the stage after it. The rest of the chapter goes through the stages in order, then the long-term layer, and ends with design lessons.

## 12.1 Offline metrics

### 12.1.1 Ranking metrics: NDCG, recall@K, MRR, MAP

**NDCG** (normalized discounted cumulative gain) comes from Järvelin and Kekäläinen [1]. DCG@K = sum over positions i <= K of (2^rel_i - 1) / log2(i + 1). That sum is then divided by the ideal DCG, which is the DCG of the best possible ordering. NDCG rewards putting highly relevant items near the top, and it is the default listwise metric for search and feed ranking. Airbnb, for example, reports NDCG of the booked listing over its search logs as the main offline signal for search ranking models [2].

**Recall@K / HitRate@K** measures what fraction of the held-out positives appear in the top K. It is the standard metric for *retrieval* stages, where the question is "did the candidate generator surface the item at all", not "is it in the right position". **MRR** (mean reciprocal rank) fits single-answer settings. **MAP** averages precision at each relevant position.

**Pointwise metrics for rankers** include AUC, log loss, and Facebook's *normalized entropy* (NE). NE is log loss divided by the entropy of the background CTR, which makes it comparable across datasets with different base rates [3]. Google's ads paper [4] relies on log loss and AUC and also tracks calibration slices. Both papers stress that tiny relative improvements in log loss/NE (fractions of a percent) can matter online.

### 12.1.2 Calibration

A ranking model can have excellent AUC and still be badly miscalibrated. That breaks every downstream consumer that treats scores as probabilities: auctions (bid x pCTR), multi-task value models that add pCTR, pLike and pWatch, and thresholds. Production teams track:

- **Global calibration ratio**: sum of predicted / sum of observed (Facebook reports "calibration" this way [3]).
- **Calibration by slice and by predicted-score bucket**: a reliability diagram, as in Google's ads paper, which lists calibration plots across many slices as a core monitoring tool [4].

Steck's **Calibrated Recommendations** [5] (Netflix, RecSys 2018) gives the word a second, distributional meaning. A recommendation list is *calibrated* if its genre distribution q(g) matches the user's historical distribution p(g). Steck measures miscalibration with KL divergence and re-ranks greedily to maximize (1 - lambda) * utility - lambda * KL(p || q~), where q~ is a slightly smoothed q. The paper shows that accuracy-trained models tend to crowd out a user's minority interests. That is useful as an offline diagnostic, not only as a re-ranker.

### 12.1.3 Pitfalls of offline evaluation

**Sampled metrics are inconsistent.** Krichene and Rendle [6] (Google, KDD 2020) showed that the common shortcut of ranking the true item against ~100 random negatives and reporting HR@10/NDCG@10 is *inconsistent*: the ranking of algorithms under sampled metrics can disagree with the ranking under full-catalogue metrics. They propose corrections, but the main advice is to compute metrics over the full catalogue whenever you can.

**Weak baselines and leakage.** Ferrari Dacrema et al. [7] tried to reproduce 18 neural recommendation papers from top venues. They could reproduce only about 7, and 6 of those were beaten by well-tuned simple baselines (kNN, graph-based, linear). Rendle et al. [8] showed that a properly tuned dot-product MF beats the MLP-based NCF. The lesson for practitioners is to tune baselines as carefully as the new model, split by time rather than at random, and remove future information from features.

**Logging-policy bias.** Every logged interaction happened because the *production* policy chose to show that item. Offline metrics computed on these logs reward models that imitate the current policy and penalize models that would surface items users never saw. Schnabel et al. [9] cast this as missing-not-at-random data and propose propensity-weighted evaluation (see 12.2). Joachims et al. [10] do the same for position bias in learning-to-rank.

```
  OFFLINE EVAL PIPELINE (typical)
  impression+engagement logs ---> time-based split ----> train window | eval window
                                                                        |
     candidate model --(score full catalogue or logged slate)-----------+
                                                                        v
           +----------------+--------------+---------------+------------------+
           | retrieval:     | ranking:     | calibration:  | beyond-accuracy: |
           | recall@K       | NDCG, AUC,   | sum(p)/sum(y) | coverage, KL     |
           | (full catalog) | NE/logloss   | by slice      | genre calib.,    |
           |                | per slice    | reliability   | diversity, nov.  |
           +----------------+--------------+---------------+------------------+
                     \__________________ gate: "worth an online test?" ______/
```

### 12.1.4 The offline-online gap

The best-documented industrial evidence comes from three sources:

- **Booking.com** [11] reported from about 150 production models that offline model-performance gains are *not* a reliable predictor of online business gains. In their words, increased performance does not necessarily translate to increased value. They describe cases where better click models saturated, or optimized proxies that diverged from conversion. Their remedy is to A/B test everything and to treat offline metrics only as a health check.
- **Airbnb** [2] documented, in "Applying Deep Learning to Airbnb Search", a long series of model iterations where offline NDCG gains did not carry over to online bookings. One example was a model that learned to favour listings that looked attractive but did not lead to bookings. The team learned to rely on online tests and to be suspicious of offline wins that rest on features which are effectively proxies for the label.
- **Netflix** [12]: Gomez-Uribe and Hunt describe offline metrics (ranking accuracy on held-out plays) as useful for screening only. Final decisions rest on A/B tests of member retention and streaming hours, because offline metrics cannot capture the counterfactual "what would the member have played otherwise".

The usual causes of the gap are (a) logging-policy bias, (b) mismatch between proxy and objective (clicks vs. satisfaction), (c) feedback loops and position effects that offline replay cannot simulate, (d) train/serve skew in features, and (e) metric saturation, where an already-good model has little room to improve on the measured axis. The rest of the chapter covers techniques that attack (a) through (c).

## 12.2 Counterfactual and off-policy evaluation

### 12.2.1 Problem setup

The logs contain tuples (x, a, r, p): context x, action (item or slate) a chosen by the logging policy pi_0 with probability p = pi_0(a|x), and reward r. We want V(pi) = E_x E_{a~pi(.|x)}[r] for a new target policy pi, *without deploying it*.

### 12.2.2 Estimators

| Estimator | Formula (per-sample, averaged over logs) | Bias | Variance | Needs |
|---|---|---|---|---|
| Replay / rejection (Li et al. [13]) | keep only events where pi picks the logged a; average r | unbiased if pi_0 uniform | high (discards data) | uniformly random logging bucket |
| Direct Method (DM) | r_hat(x, pi(x)) from a reward model | biased by model error | low | reward model |
| IPS | (pi(a\|x) / p) * r | unbiased if p > 0 wherever pi > 0 | high when p small | logged propensities |
| Clipped / capped IPS | min(w, M) * r | biased (downward) | bounded | propensities, cap M |
| Self-normalized IPS (SNIPS) | sum(w r) / sum(w) | small, consistent | lower than IPS | propensities |
| Doubly Robust (Dudik et al. [14]) | r_hat(x, pi) + w * (r - r_hat(x, a)) | unbiased if either p or r_hat is correct | lower than IPS | both |
| Pseudo-inverse for slates (Swaminathan et al. [15]) | linear decomposition over slot-item pairs | unbiased under additivity | far lower than slate-level IPS | per-slot propensities |

**Li, Chu, Langford and Wang (Yahoo!, WSDM 2011)** [13] introduced the replay method for news recommendation on the Yahoo! Today module. A slice of traffic was served *uniformly at random*. Any candidate bandit policy can be evaluated by streaming through those logs and keeping only events where it agrees with the random action. The paper showed that offline replay estimates closely matched the policies' online CTR. This set the template: **pay for a small exploration bucket, gain unbiased offline evaluation**.

**Dudik, Langford and Li (ICML 2011)** [14] formalized the doubly robust estimator, which combines a direct reward model with an IPS correction on the residual. It is unbiased if *either* the propensities *or* the reward model is correct, and usually has much lower variance than IPS. DR and its variants (switch-DR, DR with optimistic shrinkage, and others) are now the default in OPE libraries.

**Swaminathan and Joachims** [16] (JMLR 2015) developed counterfactual risk minimization (CRM) for *learning* from logged bandit feedback. Its key idea is to penalize the empirical variance of the IPS estimate (POEM). Related work introduced the self-normalized estimator to deal with "propensity overfitting".

### 12.2.3 Slates and rankings

Recommenders show slates. The naive propensity of a whole slate is the product of per-slot probabilities, which is vanishingly small, so IPS on slates explodes. Two industrial-grade answers:

- **Pseudo-inverse estimator (Microsoft Research / Cornell, NeurIPS 2017)** [15] assumes that slate reward decomposes additively over (slot, item) pairs. Under that assumption it gives an unbiased estimator whose variance scales with the number of slot-item pairs, not with the number of slates. It was validated on Microsoft search (MSLR/Bing-derived) data.
- **Criteo's "Offline A/B testing for Recommender Systems" (WSDM 2018)** [17] targets the top-K display ads setting. It proposes capped and normalized IPS variants that use only the top-K positions, trading a controlled bias for a large drop in variance. The estimators were evaluated against real online A/B outcomes at Criteo, and normalized capped importance sampling predicted the A/B *sign and ranking* of policies more reliably than standard offline metrics.

### 12.2.4 Off-policy correction for learning: YouTube REINFORCE

Chen et al. (Google/YouTube, WSDM 2019) [18] trained a REINFORCE candidate generator over an action space of *millions* of videos. Because training data came from earlier, different policies, they added (1) an importance weight pi_theta/beta, with the behaviour policy beta *estimated* by a separate softmax head because YouTube did not log propensities directly, and (2) a **top-K off-policy correction** multiplier lambda_K = K * (1 - pi)^(K-1). The second term accounts for the fact that K items are recommended at once. They also capped weights to control variance. They reported a statistically significant online gain in viewing time, which they described as one of the largest single launches in a couple of years at the time (exact effect sizes in [18]; treat as approx.). The point for designers is that **OPE is not only for evaluation**. The same weights correct training bias, and the missing ingredient in most stacks is *logging the propensity at serve time*.

### 12.2.5 Tooling and benchmarks

- **Open Bandit Pipeline / Open Bandit Dataset (ZOZO, Saito et al.)** [19] is a public dataset from a fashion e-commerce site. It was logged under *two* policies (uniform random and Bernoulli Thompson sampling), which makes it possible to validate OPE estimators against an on-policy ground truth. The paper benchmarks DM, IPS, SNIPS, DR and variants and finds that estimator rankings depend heavily on the setting. No estimator wins everywhere.
- **Practical requirements** that recur in all the sources above:
  1. Log p(a|x) at serve time, including post-ranking randomization and re-ranker effects.
  2. Keep a small exploration bucket (epsilon-greedy, softmax sampling or random reshuffle of the top-N) so that the support condition holds.
  3. Monitor the effective sample size, (sum w)^2 / sum w^2. When it collapses, the target policy is too far from logging and only an A/B test will do.

```
  OPE DATA FLOW
   request --> logging policy pi_0 (ranker + sampler) --> slate shown
                         |  logs: x, slate, per-slot p(a|x), model_version
                         v
                 [ propensity-aware log store ]
                         |
     candidate pi ------>+-----> IPS / SNIPS / DR / PI estimators
     reward model r_hat->+            |
                                      +--> V_hat(pi), CI, ESS
                                      v
                  gate: V_hat significantly > V_hat(pi_0) and ESS ok
                         --> interleaving / A/B
```

## 12.3 Interleaving

### 12.3.1 Method

Interleaving goes back to Joachims and was refined by Radlinski, Kurup and Joachims [20]. It shows each user a *single merged list* built from rankers A and B, then credits each engagement to the ranker that contributed the item. The variant used in industry is **team-draft interleaving (TDI)**: like picking teams in a schoolyard, a coin flip decides who picks first in each round, and each ranker adds its highest-ranked item not already in the list. Chapelle, Joachims, Radlinski and Yue [21] validated interleaving at scale on Yahoo! and Bing search. They found that interleaving agreed with the direction of offline relevance judgments and of A/B results, and needed one to two orders of magnitude less data than absolute-metric A/B tests to reach the same statistical power.

The sensitivity comes from a within-user comparison. Every user sees both rankers, so between-user variance (heavy vs. light users) cancels out.

```
  TEAM-DRAFT INTERLEAVING
  Ranker A: a1 a2 a3 a4 ...           Ranker B: b1 a1 b2 b3 ...
  round 1: coin=A -> A picks a1 ; B picks b1
  round 2: coin=B -> B picks b2 (a1 taken) ; A picks a2
  ...
  merged list: [a1(A), b1(B), b2(B), a2(A), ...]
  user engages with b1, b2  --> preference(B) for this user
  aggregate over users: Delta = (#users prefer A - #users prefer B) / N  -> sign test
```

### 12.3.2 Netflix

Netflix's 2017 post "Innovating Faster on Personalization Algorithms at Netflix Using Interleaving" [22] describes a **two-stage online experimentation process**:

1. **Stage 1, interleaving.** Use interleaving to quickly prune a large set of candidate ranking algorithms down to the most promising few.
2. **Stage 2, A/B.** Run a traditional A/B test on the survivors to measure effects on core metrics (streaming hours, retention).

Key points from the post:

- Engagement is attributed by **hours streamed** from titles contributed by each ranker, not by clicks.
- Netflix applies interleaving to *rows* of personalized videos and uses team-draft with care for position. Each ranker must have an equal chance of filling each position, so they "pick" in randomized order to avoid position bias.
- **Sensitivity**: in Netflix's simulations, interleaving reached the same statistical power with a much smaller sample. The post reports that A/B testing needed on the order of 100x more subscribers than interleaving to detect the same preference reliably (approx.; see the sensitivity plot in [22]).
- **Agreement**: the interleaving preference metric was strongly correlated with the A/B test's core engagement metric across the experiments they compared. That justified using it as a pruning stage.
- **Limitations**: interleaving measures *relative preference between two rankings* on one surface. It cannot measure absolute effects on retention, page-level effects (for example, how the whole homepage composes), or non-ranking changes.

### 12.3.3 Airbnb

Airbnb's 2022 post "Beyond A/B Test: Speeding up Airbnb Search Ranking Experimentation through Interleaving" [23] adapts TDI to a two-sided marketplace with sparse, delayed conversions (bookings):

- **Competitive pairs.** Many interleaved positions hold the same listing from both rankers. Airbnb only uses "competitive pairs" for attribution, meaning the positions where A and B actually differ. Searches where the rankers agree contribute no signal and are filtered out, which raises sensitivity.
- **Attribution across a search journey.** A guest issues many searches before booking, so engagement (clicks, then bookings) is attributed back over the guest's search session. Listings are *deduplicated* so that a listing first contributed by A stays credited to A.
- **Metrics**: interleaving variants of click-through and booking metrics, with booking-based metrics as the primary decision signal.
- **Results**: Airbnb reports roughly **50x speed-up** in experiment sensitivity relative to A/B testing for search ranking. It also reports that interleaving results agreed directionally with the subsequent A/B tests in the large majority of paired comparisons (approx. 80+%; see [23]).
- **Usage**: interleaving runs on a small slice of traffic and is used to triage ranking candidates before A/B. That frees A/B capacity, which is scarce because marketplace experiments need large samples.

### 12.3.4 Design notes

- Interleaving needs **both rankers to be served live in the same request**, which roughly doubles the ranking compute on that traffic slice. Cache the candidate set and only re-score.
- It suits **ranking-only changes**. It does not fit retrieval changes that alter the candidate pool size, UI changes, or anything that affects the whole page.
- Debiasing matters. If one ranker systematically wins "first pick" positions or produces items with longer descriptions, the attribution is skewed. Netflix and Airbnb both describe position-balanced picking and filtering for this reason.

## 12.4 A/B testing platforms

All the large platforms converge on a common architecture:

```
  CANONICAL EXPERIMENTATION PLATFORM
  +-------------------+     +--------------------------+     +---------------------+
  | Experiment config |---->| Assignment / allocation  |---->| Product services    |
  | UI + API (owners, |     | hash(unit_id, salt)%N -> |     | read variant, emit  |
  | metrics, layers,  |     | bucket -> variant; layers|     | exposure events     |
  | targeting, ramps) |     | / domains for overlap    |     +----------+----------+
  +-------------------+     +--------------------------+                |
                                                                        v
  +-------------------+     +--------------------------+     +---------------------+
  | Scorecards / UI   |<----| Stats engine: deltas,    |<----| Metrics pipeline    |
  | alerts, guardrail |     | CIs, CUPED/variance red, |     | (batch DAGs: exposure|
  | SRM checks, ship  |     | sequential tests, SRM,   |     | x metric joins,      |
  | decision records  |     | multiple-test control    |     | per-unit aggregates) |
  +-------------------+     +--------------------------+     +---------------------+
```

### 12.4.1 Google: overlapping experiment infrastructure

Tang, Agarwal, O'Brien and Meyer (KDD 2010) [24] described Google search's **layers and domains**. Parameters are partitioned into *layers*. Each layer splits traffic independently, so one request can be in one experiment per layer. That made it possible to run many more concurrent experiments than with single-membership bucketing. *Domains* give exclusive traffic slices for experiments that conflict with everything else, and *launch layers* handle gradual rollouts. Tooling enforces data-file checks and pre-defined metrics. Nearly every later platform (LinkedIn, Spotify, Airbnb, Netflix) uses a variant of this layering idea.

### 12.4.2 Microsoft ExP and the "trustworthiness" canon

Kohavi et al. (KDD 2013) [25] reported that Bing ran hundreds of concurrent experiments, with each user in roughly 15 concurrent experiments (approx.). They catalogued pitfalls: carry-over effects, novelty/primacy, and the need for A/A tests. The book *Trustworthy Online Controlled Experiments* [26] by Kohavi, Tang and Xu (Microsoft, Google, LinkedIn) is the standard reference. It introduces the **Overall Evaluation Criterion (OEC)**, guardrail metrics, and **sample ratio mismatch (SRM)** checks. Fabijan et al. (KDD 2019) [27] taxonomize SRM causes (bot filtering, redirect timing, residual effects, telemetry loss) and report that SRMs occurred in a notable fraction of Microsoft experiments (on the order of 6%, approx.). That makes an automated SRM chi-square test a must-have platform feature.

**CUPED** (Deng, Xu, Kohavi, Walker, WSDM 2013) [28] reduces variance by using pre-experiment covariates: Y_cuped = Y - theta * (X - mean(X)), with theta = cov(X, Y)/var(X). Using the same metric in the pre-period as X typically cuts variance by around half for engagement metrics at Bing (approx.). That is equivalent to doubling traffic. Nearly every platform below implements CUPED or a regression-adjustment generalization.

### 12.4.3 Netflix: ABlaze and the experimentation analysis rewrite

Netflix runs essentially every product change, including every recommendation algorithm, through A/B testing. Gomez-Uribe and Hunt [12] describe the core OEC as **member retention**, with **streaming hours** as a more sensitive proxy that correlates with it. Their paper also estimates that personalization and recommendations save Netflix more than USD 1B per year in reduced churn.

The 2016 post "It's All A/Bout Testing: The Netflix Experimentation Platform" [29] describes:

- **ABlaze**, the front-end for configuring tests, allocation and viewing results.
- **Allocation**: test cells, a choice between **batch allocation** (a pre-computed population of existing members meeting criteria) and **real-time allocation** (members allocated at the moment of a qualifying action, such as signup or first device use), plus a test-allocation service that clients call to get cell membership.
- **Metrics/analysis**: an analysis pipeline producing per-test results with statistical significance on retention, streaming and other metrics, and support for the many concurrent tests run across the product.

The 2019 post "Reimagining Experimentation Analysis at Netflix" [30] covers the move to a modular, open stack. Analyses are written in Python/R, and a shared library of **statistical "causal models"** (including quantile bootstrapping and regression-based variance reduction) is applied across metrics, so data scientists can add new methods without platform rewrites. Netflix's 2022 survey of causal inference applications [31] adds context on quasi-experiments, synthetic control and heterogeneous treatment effects, which are used where randomization is impossible.

### 12.4.4 Airbnb: from ERF to a scaled platform

Airbnb's early post "Experiments at Airbnb" (2014) [32] documents the pitfalls specific to a marketplace with long booking funnels:

- **Context dependence**: the same change can show opposite effects at different times or in different markets.
- **Stopping too early**: their search experiments showed p-values dipping below 0.05 transiently before settling. They argue for deciding the run time in advance and for dynamic p-value thresholds.
- **Assignment hygiene**: users must be assigned at the moment they experience the change, not earlier, or the signal is diluted.

The **Experiment Reporting Framework (ERF)** became Airbnb's central system. The 2017 post "Scaling Airbnb's Experimentation Platform" [33] describes how it was rebuilt to keep up with growth:

- Metric computation moved to **Airflow DAGs over Hive/Spark**, with assignments joined to a library of pre-defined metric "events" and per-subject aggregation.
- Scale: the post describes ERF supporting on the order of hundreds of concurrent experiments and thousands of metrics, computing results daily (approx.; see [33]).
- **Metric hierarchy**: core company metrics computed for every experiment, plus target metrics chosen by the team, plus certified metrics shared across teams.
- Pipeline optimizations reduced daily compute time by pre-aggregating assignment-metric joins and splitting the work into smaller, parallel DAG stages.

Airbnb search adds two further layers. **Interleaving** (12.3.3) handles fast triage, and offline NDCG [2] is the first screen.

### 12.4.5 LinkedIn: XLNT and T-REX

Xu, Chen, Fernandez, Sinno and Bhasin (LinkedIn, KDD 2015) [34] describe **XLNT**, LinkedIn's end-to-end A/B platform:

- **Lix** (LinkedIn Experiment) for targeting and assignment. Experiments are defined with a targeting DSL over member attributes (country, industry, activity level and so on) and ramped gradually (for example 1%, 5%, 50%).
- An offline analysis pipeline built on Hadoop that computes **many hundreds of metrics** for every experiment each day and produces reports with statistical significance.
- Scale at the time: about 200 experiments running concurrently, and well over a thousand metrics in the library (approx.; see [34]).
- Cultural mechanisms: an experimentation review process, metric "ownership", and A/B results as a gate for launches.
- Social-network challenges: **network effects/interference** (a treated member's posts reach control members), which motivated cluster-based and ego-cluster randomization later on.

LinkedIn's 2019 post "Making the LinkedIn experimentation engine 20x faster" [35] covers the next-generation **T-REX** online assignment engine. The Lix DSL was re-implemented so that evaluating which variant a member gets is a very low-latency, in-process computation. Moving from interpreted to compiled evaluation gave the 20x speed-up, at the scale of **trillions of variant evaluations per day** (approx.; see [35]). That speed matters because ranking services evaluate experiment membership per request, often several times per request.

### 12.4.6 Spotify: Experimentation Platform and Confidence

Spotify's 2020 post "Spotify's New Experimentation Platform (Part 1)" [36] describes a replacement for its earlier homegrown "ABBA" system. It has three parts: a **Remote Configuration** service for feature flags and variant properties, a **Metrics Catalog** for certified metrics computed in a shared pipeline, and an **Experiment Planner** UI for designing, sizing and launching experiments. Experiments are allocated via **"domains"** and exclusive/overlapping **layers**, with salted-hash bucketing. The post cites hundreds of teams running experiments on the platform and many experiments per year (exact figures in [36]).

Spotify's statistical work is unusually public:

- **Risk-aware decision rules with multiple metrics** (2024) [37] separates *success metrics* (shown to be better), *guardrail metrics* (shown to be non-inferior) and *deterioration* checks. It derives how to set alpha and beta across them so that the decision rule, not each metric separately, has the intended false-positive and false-negative rates. A notable result is that guardrail non-inferiority tests do not need multiple-testing correction on alpha, but power must be corrected for.
- Sequential testing (group sequential tests with spending functions) is used so that teams can check results without inflating false positives.
- **Confidence** (announced 2023) [38] is the platform offered as a commercial product to other companies. It keeps the same concepts: flags, metrics on the customer's warehouse, and Spotify's sequential and risk-aware statistics.

### 12.4.7 Uber XP

Uber's 2018 post "Under the Hood of Uber's Experimentation Platform" [39] describes a platform supporting A/B tests, multi-armed bandits, switchback experiments (for marketplace interference) and causal-inference methods. It runs on the order of 1,000+ concurrent experiments (approx.). Its statistics engine picks the test per metric type (proportions, continuous, ratio via the delta method), includes sequential testing for monitoring, and uses CUPED-style variance reduction.

### 12.4.8 Platform comparison

| Platform | Org | Assignment model | Notable statistics | Scale (public, approx.) | Ref |
|---|---|---|---|---|---|
| Overlapping layers | Google | layers + domains, per-layer hash | pre-defined metrics, launch layers | many concurrent expts per request | [24] |
| ExP | Microsoft/Bing | concurrent overlapping | CUPED, SRM checks, OEC | ~15 expts per user (approx.) | [25][27][28] |
| ABlaze + analysis lib | Netflix | batch + real-time allocation cells | retention OEC, bootstrap, regression adj. | essentially all product changes | [29][30] |
| ERF | Airbnb | user/visitor hashing; search-level for interleave | metric hierarchy, dynamic p-value stopping | hundreds expts, thousands metrics | [32][33] |
| XLNT / T-REX | LinkedIn | Lix DSL targeting + ramps | hundreds of metrics per test, network-effect designs | trillions evals/day (approx.) | [34][35] |
| EP / Confidence | Spotify | domains + layers, remote config | sequential, risk-aware multi-metric rules | hundreds of teams | [36][37][38] |
| XP | Uber | A/B, switchback, bandits | delta method, sequential, CUPED | 1000+ concurrent (approx.) | [39] |

### 12.4.9 Recommender-specific A/B pitfalls

- **Shared-model interference / feedback loops.** If control and treatment both train on the pooled logs, the treatment's data leaks into the control model and vice versa. For ranking tests that change data collection (exploration, new surfaces), teams use *separate training data per arm* or "data-diverted" arms. Kohavi et al. [26] discuss leakage through shared resources in general.
- **Two-sided marketplace interference.** At Airbnb and Uber, treatment and control compete for the same inventory, so effects are biased. Remedies are switchbacks [39] or market-level/cluster randomization.
- **Novelty and primacy effects.** Short tests overestimate (novelty) or underestimate (learning) long-run effects. Google's long-term work [40] quantifies this (see 12.5).
- **Metric sensitivity.** Retention barely moves in a 2-week test. Netflix uses streaming hours as a proxy validated against retention [12]. Generalized: surrogate indices (12.5.2).

## 12.5 Long-term metrics, surrogates and holdouts

### 12.5.1 Measuring long-term effects: Google's "Focusing on the Long-term"

Hohnhold, O'Brien and Tang (Google, KDD 2015) [40] studied how ads-quality changes affect *user learning*. Users exposed to more or worse ads gradually learn to ignore them. That drives ad CTR down over months, an effect invisible in short A/B tests. Their method:

- **Post-period / cookie-cohort design**: expose a cohort to treatment for a long period, then measure that cohort under the *same* (control) system, to isolate the learned behaviour change from the direct treatment effect.
- **Model the long-term effect from short-term metrics**: they fit a model predicting long-term user learning from short-term measurable ad quality and ad load, so future launches can be judged using short tests plus this model.
- **Outcome**: Google *reduced* the number of ads shown on mobile search (by roughly 50% on some surfaces, approx.). The long-term user-learning gains were judged to outweigh the short-term revenue loss.

This is the classic demonstration that **short-term OEC optimization can harm long-term value**, and that an explicit long-term model can fix the decision rule.

### 12.5.2 Surrogate index

Athey, Chetty, Imbens and Kang [41] formalized the **surrogate index**. Train a model on historical data that predicts the long-term outcome Y (for example 1-year retention) from many short-term surrogates S (engagement in weeks 1 to 4). Under a "surrogacy" assumption (treatment affects Y only through S), the treatment effect on Y equals the treatment effect on E[Y | S]. Using *many* surrogates combined into an index makes surrogacy more plausible than relying on any single proxy.

Recommender teams use this in practice:

- **Netflix** has publicly discussed using surrogate indices to make launch decisions from short tests in place of waiting for retention, including a large-scale evaluation of surrogate-based decisions across a few hundred historical A/B tests (approx.). The finding was that surrogate-index decisions agreed with long-term retention decisions far more often than single-proxy decisions. This is a 2023 Netflix research paper. The author could not verify its URL during compilation, so it is not cited by URL here.
- **Netflix recommendations** [42] (2024 post "Recommending for Long-Term Member Satisfaction at Netflix") treats the recommender as a bandit and argues that the *reward* used for training should itself be a proxy for long-term satisfaction, not the immediate click or play. The proxy is built by "reward engineering" from signals such as completion, thumbs-up and return visits. Its validity is checked against the online A/B results of the north-star (retention-related) metric. Practical complications are **delayed feedback**, where a show may be finished days later, and predicting the missing part of the reward for recent interactions. That ties this chapter to Chapter 11.

```
  SURROGATE-INDEX WORKFLOW
  historical cohorts: short-term S (wk1-4 engagement) --> long-term Y (retention 6-12 mo)
                     \____ fit f(S) ~ E[Y|S]  (validated on past A/Bs) ____/
  new A/B (2-4 wks):   S_treat, S_ctrl  --> f(S_treat) - f(S_ctrl) = estimated effect on Y
  safety net:          long-term holdout periodically checks that f still predicts Y
```

### 12.5.3 Holdouts

A **holdout** is a slice of users kept off a whole class of changes (for example "all recommendation launches this half") for months. It has three uses:

1. **Cumulative impact**: sums of individually measured lifts tend to overstate the combined effect, because of winner's curse, interactions and novelty. A holdout measures the real cumulative effect. Kohavi et al. [26] discuss this, along with "long-term holdouts" and "reverse experiments".
2. **Long-term validation of surrogates/proxies**: the holdout is the ground truth that the surrogate index in 12.5.2 is checked against.
3. **Ecosystem effects** that A/B tests on individuals cannot see, such as creator-side effects or content supply.

Costs are an ethical/product cost (users miss improvements), loss of statistical power for regular tests, and engineering cost (keeping old code paths alive). Common practice described in [26] is a small holdout (for example 1 to 5%) refreshed each period, or *reverse experiments*, where a launched feature is briefly turned off for a small group to re-measure it.

### 12.5.4 Winner's curse and metric inflation

When many experiments are run and only "significant winners" ship, the reported effects of winners are biased upward. Kohavi et al. [26] recommend shrinkage or holdout-based correction and replication runs for important launches. Airbnb's observation about transient significance [32] is the same phenomenon in time. Sequential designs [37] and pre-registered durations are the platform-level fixes.

## 12.6 Putting it together: a reference evaluation architecture

```
                         RECOMMENDER EVALUATION REFERENCE ARCHITECTURE
  +--------------------------- LOGGING (serve time) -------------------------------+
  | request_id, unit_id, experiment assignments, model_version, candidate set,      |
  | final slate, per-slot propensity p(a|x), features snapshot, exposure event      |
  +----------+-------------------------------------+--------------------------------+
             |                                     |
             v                                     v
  +---------------------+                 +--------------------------+
  | OFFLINE EVAL        |                 | OPE SERVICE              |
  | full-catalogue      |                 | IPS/SNIPS/DR/PI on       |
  | recall@K, NDCG, NE, |                 | exploration logs; ESS;   |
  | calibration slices, |                 | CI; compares vs pi_0     |
  | beyond-accuracy     |                 +------------+-------------+
  +----------+----------+                              |
             +-------------------+---------------------+
                                 v
                    +--------------------------+   triage 10s of candidates
                    | INTERLEAVING (ranking)   |   ~1-2 orders of mag. more sensitive
                    +------------+-------------+
                                 v
                    +--------------------------+   OEC + guardrails + SRM + CUPED
                    | A/B PLATFORM             |   sequential monitoring
                    +------------+-------------+
                                 v
                    +--------------------------+   surrogate index predicts LT
                    | LONG-TERM LAYER          |   holdouts validate cumulative
                    +--------------------------+   impact & surrogate models
```

**Design rules distilled from the sources:**

- **Log for the next evaluation, not just the current model.** Propensities [13][18], exposure events [26], model version and candidate sets cost little to log and are impossible to reconstruct later.
- **Budget exploration explicitly.** A uniformly random bucket on a small percentage of traffic made Yahoo!'s offline replay possible [13]. Lighter alternatives are softmax sampling or randomized re-ranking of the top-N.
- **Match the stage to the change type.** Ranking-only change: offline, then interleaving, then A/B. Retrieval or UI change: offline, then A/B. Exploration/RL policy: OPE, then A/B. Ecosystem change: A/B plus holdout, with cluster/switchback designs.
- **Validate every proxy against the stage above.** Netflix validated interleaving against A/B [22], Airbnb did the same [23], Criteo validated OPE against A/B [17], and Netflix validated surrogates against long-term retention.

## 12.7 Case summaries at a glance

| Company / work | Stage | Core technique | Headline result (approx.) | Ref |
|---|---|---|---|---|
| Yahoo! Today module | OPE | replay on uniform-random logs | offline estimates matched online CTR | [13] |
| Microsoft/Cornell | OPE (slates) | pseudo-inverse estimator | big variance cut vs slate-IPS | [15] |
| Criteo | OPE (top-K) | capped/normalized IS | better prediction of A/B outcomes than offline metrics | [17] |
| YouTube | OPE in training | top-K off-policy corrected REINFORCE | significant online watch-time gain | [18] |
| ZOZO | OPE benchmark | Open Bandit Dataset, two logging policies | no estimator dominates | [19] |
| Netflix | interleaving | team-draft, hours-attributed | ~100x fewer users than A/B | [22] |
| Airbnb | interleaving | competitive pairs, booking attribution | ~50x speed-up, high A/B agreement | [23] |
| Google | A/B infra | overlapping layers | many more concurrent experiments | [24] |
| Microsoft | A/B stats | CUPED, SRM | ~50% variance reduction; SRM in ~6% of expts | [27][28] |
| LinkedIn | A/B infra | XLNT, Lix, T-REX | trillions evals/day, 20x faster engine | [34][35] |
| Spotify | A/B stats | risk-aware multi-metric decision rules | principled alpha/power across metric types | [37] |
| Google ads | long-term | user-learning model, post-period | fewer mobile ads, long-term gain | [40] |
| Netflix | long-term | proxy reward for long-term satisfaction | reward engineering validated by A/B | [42] |
| Booking.com | offline-online gap | A/B everything | offline gains did not predict business gains | [11] |

## Key takeaways for system designers

1. **Treat evaluation as a funnel, not a single metric.** Offline metrics generate ideas, OPE and interleaving triage them, A/B tests decide, and holdouts or surrogates protect the long term. Each stage should be validated against the next one.
2. **Compute offline metrics honestly.** Use time-based splits, full-catalogue ranking (not sampled negatives [6]), well-tuned baselines [7][8], and calibration by slice [3][4]. Report beyond-accuracy diagnostics such as genre calibration [5].
3. **Expect the offline-online gap.** Booking.com [11], Airbnb [2] and Netflix [12] all report that offline wins often do not transfer. Budget for online tests and do not over-invest in squeezing offline metrics.
4. **Log propensities and keep an exploration slice.** Without them, IPS, SNIPS, DR and slate estimators [13][14][15][17] cannot be used. With them, the same weights also de-bias training [18].
5. **Prefer DR or SNIPS to raw IPS, and watch the effective sample size.** When the target policy moves far from the logging policy, OPE becomes unreliable and only an online test will settle the question.
6. **Use interleaving for ranking changes.** Netflix (~100x) [22] and Airbnb (~50x) [23] show very large sensitivity gains. Ignore positions where both rankers agree (competitive pairs), attribute the metric you actually care about (hours, bookings), and keep A/B as the final gate.
7. **Build the A/B platform around trust.** Overlapping layers [24], fast in-process assignment [35], SRM checks [27], CUPED [28], sequential and risk-aware decision rules [37], and a curated metric catalogue [33][36] matter more than any single statistical trick.
8. **Recommenders add their own pitfalls.** Watch for shared-training-data leakage between arms, marketplace interference (use switchbacks or clusters [39]), and network effects (use ego/cluster designs [34]).
9. **Optimize for the long term explicitly.** Short-term wins can hurt users over time [40]. Use surrogate indices [41], proxy rewards built for long-term satisfaction [42], and standing holdouts [26] to keep the decision rule aligned with retention and value.

## References

1. Järvelin, K., Kekäläinen, J. "Cumulated Gain-Based Evaluation of IR Techniques." ACM TOIS, 2002. https://dl.acm.org/doi/10.1145/582415.582418
2. Haldar, M., et al. "Applying Deep Learning to Airbnb Search." KDD 2019, Airbnb. https://arxiv.org/abs/1810.09591
3. He, X., Pan, J., Jin, O., et al. "Practical Lessons from Predicting Clicks on Ads at Facebook." ADKDD 2014, Facebook. https://dl.acm.org/doi/10.1145/2648584.2648589
4. McMahan, H. B., et al. "Ad Click Prediction: a View from the Trenches." KDD 2013, Google. https://dl.acm.org/doi/10.1145/2487575.2488200
5. Steck, H. "Calibrated Recommendations." RecSys 2018, Netflix. https://dl.acm.org/doi/10.1145/3240323.3240372
6. Krichene, W., Rendle, S. "On Sampled Metrics for Item Recommendation." KDD 2020, Google. https://arxiv.org/abs/1912.02263
7. Ferrari Dacrema, M., Cremonesi, P., Jannach, D. "Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches." RecSys 2019. https://arxiv.org/abs/1907.06902
8. Rendle, S., Krichene, W., Zhang, L., Anderson, J. "Neural Collaborative Filtering vs. Matrix Factorization Revisited." RecSys 2020, Google. https://arxiv.org/abs/2005.09683
9. Schnabel, T., Swaminathan, A., Singh, A., Chandak, N., Joachims, T. "Recommendations as Treatments: Debiasing Learning and Evaluation." ICML 2016. https://arxiv.org/abs/1602.05352
10. Joachims, T., Swaminathan, A., Schnabel, T. "Unbiased Learning-to-Rank with Biased Feedback." WSDM 2017. https://arxiv.org/abs/1608.04468
11. Bernardi, L., Mavridis, T., Estevez, P. "150 Successful Machine Learning Models: 6 Lessons Learned at Booking.com." KDD 2019, Booking.com. https://dl.acm.org/doi/10.1145/3292500.3330744
12. Gomez-Uribe, C. A., Hunt, N. "The Netflix Recommender System: Algorithms, Business Value, and Innovation." ACM TMIS, 2015, Netflix. https://dl.acm.org/doi/10.1145/2843948
13. Li, L., Chu, W., Langford, J., Wang, X. "Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms." WSDM 2011, Yahoo! Research. https://arxiv.org/abs/1003.5956
14. Dudik, M., Langford, J., Li, L. "Doubly Robust Policy Evaluation and Learning." ICML 2011. https://arxiv.org/abs/1103.4601
15. Swaminathan, A., Krishnamurthy, A., Agarwal, A., Dudik, M., Langford, J., Jose, D., Zitouni, I. "Off-policy Evaluation for Slate Recommendation." NeurIPS 2017, Microsoft Research. https://arxiv.org/abs/1605.04812
16. Swaminathan, A., Joachims, T. "Batch Learning from Logged Bandit Feedback through Counterfactual Risk Minimization." JMLR 16, 2015. https://jmlr.org/papers/v16/swaminathan15a.html
17. Gilotte, A., Calauzènes, C., Nedelec, T., Abraham, A., Dollé, S. "Offline A/B Testing for Recommender Systems." WSDM 2018, Criteo. https://arxiv.org/abs/1801.07030
18. Chen, M., Beutel, A., Covington, P., Jain, S., Belletti, F., Chi, E. "Top-K Off-Policy Correction for a REINFORCE Recommender System." WSDM 2019, Google/YouTube. https://arxiv.org/abs/1812.02353
19. Saito, Y., Aihara, S., Matsutani, M., Narita, Y. "Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation." NeurIPS Datasets and Benchmarks 2021, ZOZO. https://arxiv.org/abs/2008.07146
20. Radlinski, F., Kurup, M., Joachims, T. "How Does Clickthrough Data Reflect Retrieval Quality?" CIKM 2008. https://dl.acm.org/doi/10.1145/1458082.1458092
21. Chapelle, O., Joachims, T., Radlinski, F., Yue, Y. "Large-Scale Validation and Analysis of Interleaved Search Evaluation." ACM TOIS, 2012. https://dl.acm.org/doi/10.1145/2094072.2094078
22. Parks, J., Aurisset, J., Ramm, M. (Netflix Technology Blog). "Innovating Faster on Personalization Algorithms at Netflix Using Interleaving." 2017, Netflix. https://netflixtechblog.com/interleaving-in-online-experiments-at-netflix-a04ee392ec55
23. Airbnb Engineering. "Beyond A/B Test: Speeding up Airbnb Search Ranking Experimentation through Interleaving." 2022, Airbnb. https://medium.com/airbnb-engineering/beyond-a-b-test-speeding-up-airbnb-search-ranking-experimentation-through-interleaving-7087afa09c8e
24. Tang, D., Agarwal, A., O'Brien, D., Meyer, M. "Overlapping Experiment Infrastructure: More, Better, Faster Experimentation." KDD 2010, Google. https://dl.acm.org/doi/10.1145/1835804.1835810
25. Kohavi, R., Deng, A., Frasca, B., Walker, T., Xu, Y., Pohlmann, N. "Online Controlled Experiments at Large Scale." KDD 2013, Microsoft. https://dl.acm.org/doi/10.1145/2487575.2488217
26. Kohavi, R., Tang, D., Xu, Y. "Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing." Cambridge University Press, 2020. https://experimentguide.com/
27. Fabijan, A., Gupchup, J., Gupta, S., Omhover, J., Qin, W., Vermeer, L., Dmitriev, P. "Diagnosing Sample Ratio Mismatch in Online Controlled Experiments." KDD 2019, Microsoft. https://dl.acm.org/doi/10.1145/3292500.3330722
28. Deng, A., Xu, Y., Kohavi, R., Walker, T. "Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data (CUPED)." WSDM 2013, Microsoft. https://dl.acm.org/doi/10.1145/2433396.2433413
29. Netflix Technology Blog. "It's All A/Bout Testing: The Netflix Experimentation Platform." 2016, Netflix. https://netflixtechblog.com/its-all-a-bout-testing-the-netflix-experimentation-platform-4e1ca458c15
30. Netflix Technology Blog. "Reimagining Experimentation Analysis at Netflix." 2019, Netflix. https://netflixtechblog.com/reimagining-experimentation-analysis-at-netflix-71356393af21
31. Netflix Technology Blog. "A Survey of Causal Inference Applications at Netflix." 2022, Netflix. https://netflixtechblog.com/a-survey-of-causal-inference-applications-at-netflix-b62d25175e6f
32. Overgoor, J. "Experiments at Airbnb." 2014, Airbnb Engineering. https://medium.com/airbnb-engineering/experiments-at-airbnb-e2db3abf39e7
33. Parks, J. "Scaling Airbnb's Experimentation Platform." 2017, Airbnb Engineering. https://medium.com/airbnb-engineering/https-medium-com-jonathan-parks-scaling-erf-23fd17c91166
34. Xu, Y., Chen, N., Fernandez, A., Sinno, O., Bhasin, A. "From Infrastructure to Culture: A/B Testing Challenges in Large Scale Social Networks." KDD 2015, LinkedIn. https://dl.acm.org/doi/10.1145/2783258.2788602
35. LinkedIn Engineering. "Making the LinkedIn Experimentation Engine 20x Faster." 2019, LinkedIn. https://engineering.linkedin.com/blog/2019/06/making-the-linkedin-experimentation-engine-20x-faster
36. Spotify Engineering. "Spotify's New Experimentation Platform (Part 1)." 2020, Spotify. https://engineering.atspotify.com/2020/10/spotifys-new-experimentation-platform-part-1/
37. Spotify Engineering. "Risk-Aware Product Decisions in A/B Tests with Multiple Metrics." 2024, Spotify. https://engineering.atspotify.com/2024/03/risk-aware-product-decisions-in-a-b-tests-with-multiple-metrics
38. Spotify Engineering. "Coming Soon: Confidence, an Experimentation Platform from Spotify." 2023, Spotify. https://engineering.atspotify.com/2023/08/coming-soon-confidence-an-experimentation-platform-from-spotify
39. Uber Engineering. "Under the Hood of Uber's Experimentation Platform." 2018, Uber. https://www.uber.com/blog/xp/
40. Hohnhold, H., O'Brien, D., Tang, D. "Focusing on the Long-term: It's Good for Users and Business." KDD 2015, Google. https://dl.acm.org/doi/10.1145/2783258.2788583
41. Athey, S., Chetty, R., Imbens, G., Kang, H. "The Surrogate Index: Combining Short-Term Proxies to Estimate Long-Term Treatment Effects More Rapidly and Precisely." NBER / arXiv, 2016-2019. https://arxiv.org/abs/1603.09326
42. Netflix Technology Blog. "Recommending for Long-Term Member Satisfaction at Netflix." 2024, Netflix. https://netflixtechblog.com/recommending-for-long-term-member-satisfaction-at-netflix-ac15cada49ef
