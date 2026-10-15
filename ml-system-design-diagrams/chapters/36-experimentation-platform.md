# 36. Design an A/B Testing / Experimentation Platform for ML Models (Airbnb ERF, Netflix XP, Amazon Weblab)

> Sourcing note (review run 2026-10-14): web search was unavailable and most non-GitHub hosts did not resolve from this sandbox. Verified this run: [10], [11], [12], [13], [15] (HTTP 200). Other references are well-known public sources cited from prior knowledge; URLs not re-checked. Company rows below are tied to each company's published platform, not to verified interview reports.

## Company tags

| Company (published platform) | Framing seen | Ref |
|---|---|---|
| Airbnb | "Design an experiment reporting framework (ERF)", plus metric pipelines and dimension cuts | [1][2] |
| Netflix | "Design the A/B platform behind personalization", covering allocation, cells, interleaving | [3][4][5] |
| Amazon | "Design Weblab": treatment dialing, launch gating, metric scorecards | [9] |
| Uber | "Design XP": many experiment types, a stats engine, staged rollout | [6] |
| Google | "Overlapping experiments at scale": layers and domains | [7] |
| Microsoft / Spotify | Scorecards, variance reduction, SRM alerts, guardrails | [8][9][15][18] |
| Any MLSD loop | "How do you A/B test a new ranker?" is the online-testing step of most MLSD answers | [10][16] |

Portals that cover this problem, either as a full problem or as the "online evaluation" step of every design:

| Portal | Coverage | Link |
|---|---|---|
| alirezadir/Machine-Learning-Interviews | Step 8 "A/B, bandits, shadow, canary" checklist | [10] |
| khangich/machine-learning-interview | Lists the Kohavi et al. A/B testing book as a resource | [11] |
| Chip Huyen, "Designing ML Systems" (Ch. 9) | Shadow, A/B, canary, interleaving, bandits | [16] |
| ByteByteGo (Xu & Aminian) | Online metrics and A/B test step in every chapter | [17] |
| Kohavi, Tang, Xu book | The reference text on platform architecture | [9] |
| GrowthBook (open source) | Reference implementation: flags + SDK, warehouse-native stats | [12][13] |

## Requirements & scale

| Category | Requirement | Value (typical assumption unless cited) |
|---|---|---|
| Functional | Define experiment: variants, targeting, allocation %, metrics, duration | - |
| Functional | Deterministic, sticky assignment (user, device, session, or cluster unit) | hash(unit_id, salt) [7] |
| Functional | Route traffic to model variants (A = prod ranker, B = new ranker) | - |
| Functional | Concurrent overlapping experiments without collisions | layers/domains [7] |
| Functional | Automated scorecards: lift, CI, p-value, guardrails, SRM | [1][4][8] |
| Functional | Ramp, stop, ship; kill-switch | - |
| Functional | Feature flags as the delivery vehicle; SDK fires a tracking callback on exposure | [12][13] |
| Functional | Warehouse-native analysis: stats from aggregates (sums, sums of squares), no row-level PII leaves the warehouse | [13] |
| Functional | Frequentist or Bayesian engine, custom priors, multi-armed bandits | [13] |
| Non-functional | Assignment latency | < 1-5 ms p99 (local SDK eval) |
| Non-functional | Availability | 99.99%; fail-open to control |
| Non-functional | Trustworthiness | SRM checks, A/A tests, pre-registered metrics [9] |
| Scale | Concurrent experiments | 1k-10k+ (Microsoft, Booking, Google scale [9]) |
| Scale | Assignment / exposure events | 100k-1M QPS; 10-100B events/day |
| Scale | Users | 100M-1B units |
| Scale | Metrics per scorecard | 100s-1000s; daily batch plus near-real-time guardrails |
| Freshness | Scorecard refresh | Daily batch (ERF [1]); minutes for guardrails |

## Diagrams

**DIAGRAM 1 - Online path: request -> assignment -> model-variant routing -> logged response**

```
+--------+ request(unit_id, ctx) +---------------------------+
| Client |---------------------->| API Gateway / Product Svc |
+--------+                       +---------------------------+
    ^                                  | getVariants(unit_id, ctx)
    |                                  v
    |                    +------------------------------+  push/poll  +-----------+
    |                    | Assignment SDK (in-process)  |<------------| Config    |
    |                    |  bucket = hash(id+salt)%1000 |   (10-60 s) | Service   |
    |                    |  layer / domain lookup       |             +-----------+
    |                    |  targeting + holdouts        |                   ^
    |                    +------------------------------+                   |
    |                       | variant map {ranker: B}           Exp UI / API (owners)
    |                       v
    |      +---------------------------------------------+     +----------------+
    |      | Model Router (by experiment flag)           |<----| Model Registry |
    |      |   A --> ranker_v12 (prod)                   |load | (artifact per  |
    |      |   B --> ranker_v13 (candidate)              |     |  variant)      |
    |      |   S --> shadow v14: scored, not served      |     +----------------+
    |      +---------------------------------------------+
    |                       | request + chosen model
    |                       v
    |      +-------------------------------+  +---------------+
    |      | Retrieval -> Filter -> Rank   |<-| Feature Store |
    |      |  -> Re-rank (chosen variant)  |  +---------------+
    |      +-------------------------------+
    |   response            |
    +-----------------------+
                            | async: exposure {unit, exp, variant, ts} + actions
                            v
             +----------------------------------+
             | Event Log (Kafka)                |---> Diagram 2 (offline)
             |  exposures + user actions        |---> RT guardrail stream
             +----------------------------------+
```

**DIAGRAM 2 - Offline metrics pipeline: logs -> joins -> metrics -> stats -> scorecard -> decision (feedback loop)**

```
+------------+  +-------------+
| Exposure   |  | Action logs |  (clicks, bookings, plays, revenue)
| log        |  | (Kafka)     |
+------------+  +-------------+
      |               |
      v               v
+----------------------------------+    hourly / daily (Airflow, Spark, Hive)
| Raw lake / warehouse (S3, Hive)  |
+----------------------------------+
      |  dedupe, bot filter, first-exposure time
      v
+----------------------------------+
| Assignment table                 |   unit -> (exp, variant, first_exposed_ts)
+----------------------------------+
      |  join on unit_id, ts >= exposure
      v
+----------------------------------+    +---------------------------+
| Metric layer (semantic defs)     |<---| Metric repo (YAML / DSL)  |
|  per-unit aggregates             |    | owners, guardrails, OEC   |
+----------------------------------+    +---------------------------+
      |  + pre-period covariates (for CUPED)
      v
+----------------------------------+
| Stats engine                     |   t-test, delta method, CUPED,
|  (see Diagram 3)                 |   sequential, multiple-testing fix
+----------------------------------+
      |
      v
+----------------------------------+    +---------------------------+
| Scorecard store + UI             |--->| Alerts: SRM, guardrail    |
|  lift, CI, p, power, slices      |    | breach -> auto-stop       |
+----------------------------------+    +---------------------------+
      |  ship / iterate / kill
      v
+----------------------------------+
| Decision log + model registry    |---> promote ranker_v13 to prod
| (learnings, meta-analysis)       |---> priors for next experiment
+----------------------------------+        (feedback loop)
```

**DIAGRAM 3 - The "model": stats engine inputs/outputs**

```
 inputs per experiment
 +------------------------+   +------------------------+   +----------------------+
 | per-unit metric Y      |   | pre-period covariate X |   | variant labels, n_A, |
 | (sum, ratio num/den)   |   | (same metric, -14d)    |   | n_B, expected split  |
 +------------------------+   +------------------------+   +----------------------+
            |                           |                            |
            v                           v                            v
 +-------------------------------------------------------------------------------+
 | 1. Data quality: SRM chi-square test (observed vs expected split)  --> block  |
 | 2. Variance reduction: CUPED  Y' = Y - theta*(X - mean(X))                    |
 |    theta = cov(Y,X)/var(X)                                                    |
 | 3. Ratio metrics (CTR per session): delta method for var(num/den)             |
 | 4. Test: Welch t-test / z-test | Bayesian posterior | sequential (mSPRT)      |
 | 5. Multiple metrics: Benjamini-Hochberg FDR; primary OEC pre-registered       |
 | 6. Heterogeneity: slice by platform/country (flag, not ship criterion)        |
 +-------------------------------------------------------------------------------+
            |                  |                    |                    |
            v                  v                    v                    v
      +-----------+     +-------------+     +--------------+     +--------------+
      | lift, 95% |     | p-value /   |     | power, MDE,  |     | guardrail    |
      | CI        |     | P(B>A)      |     | days-to-sig  |     | pass/fail    |
      +-----------+     +-------------+     +--------------+     +--------------+
```

**DIAGRAM 4 - Variant: Google-style layered assignment and the ML rollout ladder**

```
 traffic (100%)
 +------------------------------------------------------------------------------+
 | Domain 1 (non-overlapping, 10%): big UI changes, exclusive experiments       |
 +------------------------------------------------------------------------------+
 | Domain 2 (overlapping, 85%)                                                  |
 |   Layer "ranking model":  [ctl 50%][ranker_v13 25%][ranker_v14 25%]          |
 |   Layer "UI":             [ctl][blue btn][big card]   <- independent hash    |
 |   Layer "ads":            [ctl][bid model B]          <- salt per layer      |
 +------------------------------------------------------------------------------+
 | Global holdout (5%): no launches, long-term cumulative impact                |
 +------------------------------------------------------------------------------+

 ML model rollout ladder:
 +--------+    +---------+    +-----------+    +-----------+    +------------+
 | offline|--->| shadow  |--->| canary 1% |--->| A/B 50/50 |--->| ramp 100%  |
 | eval   |    | (no UX) |    | latency,  |    | OEC +     |    | + holdout  |
 | AUC,   |    | score   |    | errors    |    | guardrails|    | long-term  |
 | NDCG   |    | diff    |    |           |    | 1-4 weeks |    |            |
 +--------+    +---------+    +-----------+    +-----------+    +------------+
                    \                                 ^
                     \--> interleaving (rankers) ----/  ~100x fewer users [5]  
```

## Data & labels

| Data | Source | Notes |
|---|---|---|
| Assignment/exposure | SDK log at decision time | Log only when the user is actually exposed (trigger analysis) |
| Outcome events | Product logs (click, book, play, revenue) | Joined after first exposure |
| Pre-period data | Same metrics, 1-4 weeks before | Covariates for CUPED [8] |
| Experiment metadata | Config service | Hypothesis, owner, OEC, guardrails, MDE |

## Features (unit-level covariates)

| Feature | Use |
|---|---|
| Pre-experiment metric value | CUPED variance reduction (about 50% variance cut on Bing metrics [8]) |
| Platform, country, tenure | Stratification and slices |
| New vs returning | Novelty/primacy checks |
| Bot / internal flags | Exclusion |

## Model choices (statistical method)

| Level | Method | Tradeoff |
|---|---|---|
| Baseline | Fixed-horizon t-test, 50/50 split | Simple, but invalid if people peek at results |
| + | CUPED / regression adjustment | Smaller CI, needs pre-period data |
| + | Sequential testing (mSPRT, always-valid p) [14] | Safe to peek; slightly wider CI |
| + | Bayesian (posterior P(B>A), expected loss) | Intuitive; prior choice matters |
| + | Interleaving for rankers (Netflix reports ~100x fewer users needed) [5] | Very sensitive, but only gives a preference, not business lift |
| Advanced | Multi-armed / contextual bandits | Less regret; harder inference, biased estimates |
| Advanced | Switchback / cluster randomization (marketplaces) [6] | Handles interference; less power |

## Loss / objective

| Item | Definition |
|---|---|
| OEC | One pre-registered primary metric (e.g. bookings per user) [9] |
| Decision rule | Ship if OEC CI > 0, no guardrail breach, SRM pass |
| Bandit objective | Minimize cumulative regret |

## Offline metrics (platform health)

| Metric | Target |
|---|---|
| A/A false-positive rate | about alpha (5%) |
| SRM rate across experiments | < 1-2% of experiments |
| Bucketing uniformity | chi-square pass |
| Offline-to-online correlation | Track for each model family |

## Online metrics / A-B

| Type | Examples |
|---|---|
| OEC / primary | Bookings/user, watch hours, GMV/user |
| Guardrails | Latency p99, errors, crash rate, unsubscribes, revenue |
| Model-specific | CTR, NDCG@k from logs, coverage, diversity |
| Platform | Time to decision, % experiments that are well-powered |

## Serving & latency budget

| Step | Budget |
|---|---|
| SDK hash + config lookup (in memory) | < 0.5 ms |
| Config refresh (push/poll) | 10-60 s staleness OK |
| Model router to variant endpoint | + 0 ms (pre-loaded) |
| Async exposure logging | Off the critical path |
| Shadow model | Async; never blocks response |

## Monitoring

| Signal | Action |
|---|---|
| SRM (p < 0.001) | Invalidate scorecard, alert owner |
| Guardrail breach (near-real-time) | Auto-ramp-down or kill-switch |
| Logging drop or delay | Pipeline SLA alert |
| Interaction between concurrent experiments | Pairwise interaction detection |
| Novelty decay | Plot daily lift trend |

## Failure modes

| Failure | Mitigation |
|---|---|
| Peeking leads to false positives | Sequential tests or fixed horizon [14] |
| SRM from bot filtering or redirects | Exposure-based logging, SRM gate |
| Network effects / interference (marketplace) | Cluster or switchback randomization, two-sided tests |
| Feedback loop: B model trains on A's data | Train per-variant or use a holdout; flag data leakage |
| Carryover from reused buckets | Re-salt per experiment, cooldown |
| Too many metrics, false discoveries | FDR control, pre-registered OEC |
| Long-term vs short-term mismatch | Long-running holdouts, surrogate metrics |

## Interviewer follow-ups

- How do you guarantee sticky assignment across devices and logged-out users?
- How do you run 1,000 concurrent experiments without collisions (layers/domains)?
- What is SRM, how do you detect it, and what causes it?
- How do you compute sample size / MDE, and what if traffic is too low?
- Explain CUPED. Why does it reduce variance and not bias?
- How do you handle peeking? Sequential vs Bayesian?
- How do you compute variance for ratio metrics like CTR (delta method)?
- How do you A/B test a ranking model: interleaving vs A/B?
- Marketplace interference (Airbnb, Uber): how do you randomize?
- How do you avoid the feedback loop where the new model trains on logs from the old model?
- How do you pick the OEC and guardrails? What if metrics disagree?
- Offline metrics improved but online did not. Why?
- How do you measure long-term effects and novelty or primacy effects?
- When would you use bandits instead of A/B?
- How do you make the platform fail open, and what is the latency impact?
- Warehouse-native (query the customer warehouse) vs a streaming metrics pipeline: tradeoffs in freshness, cost, privacy? [13]
- Why compute every test from sufficient statistics (n, sum, sum of squares) per variant, and how does that scale to 1000s of metrics? [13]

## Portal comparison

| Portal | Framing | Emphasis | Gaps |
|---|---|---|---|
| alirezadir GitHub [10] | Step 8 of every MLSD answer | Checklist: A/B, bandits, shadow, canary | No architecture or stats depth |
| Chip Huyen DMLS [16] | Ch. 9 "Continual learning & test in prod" | Shadow, canary, interleaving, bandits | No platform infra |
| ByteByteGo ML book [17] | Online eval section per chapter | Online metrics, A/B per problem | Platform not a standalone chapter |
| Kohavi et al. book [9] | Full platform design | SRM, OEC, CUPED, trust | Not interview-formatted |
| Company blogs [1]-[6] | Real systems | Pipelines, scale, sequential, interleaving | No interview structure |
| GrowthBook docs [12][13] | Open-source platform | SDK tracking callback, fact tables, CUPED, sequential, SRM, bandits | Not interview-formatted |

## References

1. "Experiment Reporting Framework", Airbnb Engineering (Medium), 2014, https://medium.com/airbnb-engineering/experiment-reporting-framework-4e3fcd29e6c0
2. "Scaling Airbnb's Experimentation Platform", Airbnb Engineering (Medium), 2017, https://medium.com/airbnb-engineering/https-medium-com-jonathan-parks-scaling-erf-23fd17c91166
3. "It's All A/Bout Testing: The Netflix Experimentation Platform", Netflix Tech Blog, 2016, https://netflixtechblog.com/its-all-a-bout-testing-the-netflix-experimentation-platform-4e1ca458c15
4. "Reimagining Experimentation Analysis at Netflix", Netflix Tech Blog, 2019, https://netflixtechblog.com/reimagining-experimentation-analysis-at-netflix-71356393af21
5. "Innovating Faster on Personalization Algorithms at Netflix Using Interleaving", Netflix Tech Blog, 2017, https://netflixtechblog.com/interleaving-in-online-experiments-at-netflix-a04ee392ec55
6. "Under the Hood of Uber's Experimentation Platform", Uber Engineering, 2018, https://www.uber.com/blog/xp/
7. Tang et al., "Overlapping Experiment Infrastructure: More, Better, Faster Experimentation", Google Research (KDD), 2010, https://research.google/pubs/overlapping-experiment-infrastructure-more-better-faster-experimentation/
8. Deng, Xu, Kohavi, Walker, "Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data" (CUPED), WSDM, 2013, https://exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf
9. Kohavi, Tang, Xu, "Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing", Cambridge University Press, 2020, https://experimentguide.com/
10. alirezadir, "Machine-Learning-Interviews: ML System Design", GitHub, 2023-2025, https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/ml-system-design.md
11. khangich, "machine-learning-interview", GitHub, 2021-2024, https://github.com/khangich/machine-learning-interview
12. GrowthBook, "growthbook: open-source feature flagging and experimentation", GitHub, 2024, https://github.com/growthbook/growthbook
13. GrowthBook, "docs/overview.mdx" (feature flags, warehouse analysis, CUPED, sequential, SRM, bandits), GitHub, 2025, https://github.com/growthbook/growthbook/blob/main/docs/overview.mdx
14. Johari, Pekelis, Walsh, "Always Valid Inference: Bringing Sequential Analysis to A/B Testing", arXiv, 2015, https://arxiv.org/abs/1512.04922
15. "Experimentation Platform (ExP)", Microsoft Research, ongoing, https://www.microsoft.com/en-us/research/group/experimentation-platform-exp/
16. Chip Huyen, "Designing Machine Learning Systems" (Ch. 9), O'Reilly, 2022, https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/
17. Ali Aminian and Alex Xu, "Machine Learning System Design Interview", ByteByteGo, 2023, https://bytebytego.com/intro/machine-learning-system-design-interview
18. "Spotify's New Experimentation Platform (Part 1)", Spotify Engineering, 2020, https://engineering.atspotify.com/2020/10/spotifys-new-experimentation-platform-part-1
