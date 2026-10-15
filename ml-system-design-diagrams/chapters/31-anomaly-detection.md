# 31. Design Anomaly Detection for Metrics/Monitoring and Transactions

> Sourcing note: for this chapter the shared web-search budget had run out and most portal and
> company-blog domains could not be reached from the build environment. The facts below come from
> pages that were fetched (GitHub repos: Microsoft anomalydetector, Twitter AnomalyDetection, Yahoo
> EGADS, LinkedIn luminol, Numenta NAB, Meta Kats, curated case-study indexes, interview repos).
> References marked "(not fetched)" are well-known engineering posts/papers whose URLs were not
> re-checked. A later review pass also found every external host blocked (proxy 403), so no URL
> was re-fetched in that pass either; unverifiable company tags and portal claims were removed or
> marked. Rows marked "typical assumption" are interview-standard estimates.

## Company tags

| Company (reported / mirrored) | Evidence | Ref |
|---|---|---|
| Microsoft | Built it: SR-CNN service (KDD 2019); multivariate MTAD-GAT; Azure Anomaly Detector (retiring 1 Oct 2026) | [1][2][27][28] |
| LinkedIn | Built it: ThirdEye business-metric monitoring + RCA; luminol; isolation forest | [3][4][5] |
| Uber | Built it: Argos real-time outage alerts; Risk Entity Watch (fraud anomalies) | [6][7] |
| Twitter / X | Built it: Seasonal Hybrid ESD (S-H-ESD) open-source detector | [8][9] |
| Yahoo | Built it: EGADS generic time-series anomaly framework (KDD 2015) | [10][11] |
| Netflix | Built it: RAD, robust-PCA outlier detection on big data | [12] |
| Meta | Built it: Kats toolkit (CUSUM, outlier, trend, change-point detectors) | [13] |
| Grab, Capital One, Walmart | Graph anomaly fraud; app-failure detection; sensor-anomaly forecasting | [14][15][16] |

Portals that cover the problem:

| Portal | Coverage | Ref |
|---|---|---|
| alirezadir/Machine-Learning-Interviews | No named chapter; MLSD template covers drift detection, alarms, metric dashboards | [17] |
| chiphuyen/machine-learning-systems-design | Exercise 2: labelled fraud vs safe card purchases (transaction variant) | [18] |
| khangich/machine-learning-interview | "Fraud detection (TBD)" placeholder | [19] |
| Engineer1999 curated case studies | Indexes Uber, Grab, Walmart anomaly posts; Capital One, Microsoft incident mgmt | [20] |
| Educative Grokking ML System Design | Generic 6-step framework; no anomaly chapter verified (fraud variant unconfirmed) | [21] |
| ByteByteGo (Aminian & Xu) | No dedicated chapter; monitoring covered inside every chapter | [22] |
| Exponent, Hello Interview | Generic MLSD frameworks applied to detection problems | [23][24] |
| Atlas cross-ref | Ch. 14 (payment fraud) for the supervised transaction path | ch. 14 |

## Requirements & scale

| Dimension | Value | Source |
|---|---|---|
| Functional | Ingest millions of time series (KPIs, infra metrics); flag anomalous points/windows | [1][3] |
| Functional | Severity score + anomaly window + peak timestamp (not just a binary flag) | [5] |
| Functional | Group related alerts; root-cause hints (dimension drill-down, correlated metrics) | [3][5] |
| Functional | Transaction mode: score each event, route to block / review / allow | [7][18] |
| Functional | User feedback (true/false alert) and per-metric sensitivity tuning | [1][3] |
| Non-functional | Detection delay: seconds-1 min (infra), <100 ms per event (transactions) | [2] / typical |
| Non-functional | Low alert fatigue: precision prioritized for paging; recall for fraud | typical |
| Scale | 1M-100M series, 1-60 s granularity => 100k-10M points/s | typical assumption |
| Scale (Microsoft) | Service handles millions of time series across Microsoft products | [2] (not fetched) |
| Labels | Very sparse (<0.1-1% anomalous); mostly unlabeled; delayed fraud labels | [2][7] |
| Seasonality | Daily/weekly cycles, holidays, deploy-driven level shifts | [8][10] |

## Diagram 1: Online detection path (streaming)

**Diagram 1 - End-to-end online architecture for metric and transaction anomaly detection**

```
 +----------------+    +----------------+    +----------------+
 | Infra agents   |    | Business KPI   |    | Transactions   |
 | (StatsD/OTel)  |    | events (orders)|    | (pay, logins)  |
 +-------+--------+    +-------+--------+    +-------+--------+
         | points              | events              | events
         v                     v                     v
 +--------------------------------------------------------------+
 | Stream bus (Kafka), partitioned by series id / entity id     |
 +-----------+---------------------------------------+----------+
             | metric points                         | txn events
             v                                       v
 +------------------------+             +------------------------+
 | Stream aggregator      |             | Online feature store   |
 | (Flink): 1-min roll-up,|             | velocity counts, entity|
 | dimension cubes,       |             | history, graph feats   |
 | no-data / late check   |             |                        |
 +-----------+------------+             +-----------+------------+
             | series window                        | feature vector
             v                                      v
 +------------------------+  +----------+  +------------------------+
 | Expected-value model   |<-| Model    |->| Event scorer           |
 | STL / Holt-Winters /   |  | store:   |  | GBDT (p_fraud) +       |
 | SR-CNN / MTAD-GAT      |  | weights, |  | iForest / AE novelty   |
 +-----------+------------+  | per-ser. |  +-----------+------------+
             | residual      | params   |              | event score
             v               +----------+              v
 +------------------------+             +------------------------+
 | Scoring + threshold:   |             | Decision policy: rules |
 | robust z / ESD / SR    |             | + score, cost-based    |
 | prob, per-series dyn.  |             | cut-offs               |
 | threshold, min duration|             +-----+------------+-----+
 +-----------+------------+                   | allow      | review / block
             | candidate anomalies            v            v
             v                          +----------+ +-------------+
 +------------------------+             | Approve  | | Step-up or  |
 | Post-process: dedupe,  |             +----------+ | analyst     |
 | group by service/dim,  |                          | queue       |
 | deploy/holiday suppress|                          +------+------+
 +-----------+------------+                                 |
             | alerts                                       |
             v                                              |
 +------------------------+                                 |
 | Alert router: page /   |                                 |
 | ticket / Slack + RCA   |                                 |
 +-----------+------------+                                 |
             | ack / false alarm                            | labels, chargebacks
             v                                              v
 +-----------------------------------------------------------------------+
 | Feedback + prediction log (TSDB / lake) --> offline training (Diag 2) |
 +-----------------------------------------------------------------------+
```

## Diagram 2: Offline training & data pipeline

**Diagram 2 - Offline pipeline: history, labels, model fitting, evaluation, deployment**

```
 +--------------+   +-----------------+   +-----------------------+
 | Metric TSDB  |   | Event lake      |   | Feedback store        |
 | (history,    |   | (txns, logins,  |   | alert acks, false     |
 |  90d-2y)     |   |  sessions)      |   | alarms, chargebacks   |
 +------+-------+   +--------+--------+   +-----------+-----------+
        |                    |                        |
        v                    v                        v
 +-----------------------------------------------------------------+
 | Label builder: incident windows from postmortems, user feedback,|
 | delayed fraud labels (30-90 d), synthetic injected anomalies    |
 +--------------------------------+--------------------------------+
                                  | labelled windows / events
                                  v
 +-----------------------------------------------------------------+
 | Feature / transform jobs: STL decomposition, SR transform,      |
 | lags, calendar, entity aggregates, embeddings                   |
 +---------------+----------------------------+--------------------+
                 | per-series fit             | global model train
                 v                            v
 +----------------------------+   +----------------------------------+
 | Per-series baselines       |   | Global models: SR-CNN, LSTM-AE,  |
 | (seasonality period,       |   | GBDT on txns, iForest; meta-     |
 |  thresholds, sensitivity)  |   | learner picks algo per series    |
 +-------------+--------------+   +----------------+-----------------+
               |                                   |
               v                                   v
 +-----------------------------------------------------------------+
 | Backtest / evaluation: point-adjusted F1, delay-aware F1,       |
 | NAB-style score, alerts per series per week, PR-AUC (txns)      |
 +--------------------------------+--------------------------------+
                                  | pass gates
                                  v
 +--------------------+   +------------------+   +------------------+
 | Model registry     |-->| Shadow mode on   |-->| Canary / ramp,   |
 | + param versions   |   | live stream      |   | alert-rate guard |
 +--------------------+   +------------------+   +--------+---------+
                                                          | deployed
                                                          v
 +-----------------------------------------------------------------+
 | Online path (Diagram 1) -> new acks / false alarm / chargebacks |
 | -> Feedback store (top) -> next retrain / threshold re-tune     |
 +-----------------------------------------------------------------+
```

## Diagram 3: Model architecture

**Diagram 3 - Forecast-residual detector with SR-CNN scorer (metrics) and hybrid event scorer (transactions)**

```
       METRIC PATH                                TRANSACTION PATH
 +---------------------------+            +-------------------------------+
 | Window x[t-w .. t]        |            | Event e: amount, merchant,    |
 | (e.g. w = 1440 points)    |            | device, geo, user history     |
 +-------------+-------------+            +---------------+---------------+
               |                                          |
     +---------+----------+                   +-----------+-----------+
     |                    |                   |                       |
     v                    v                   v                       v
 +----------------+ +--------------+   +---------------+     +-----------------+
 | Forecaster     | | Spectral     |   | Supervised    |     | Unsupervised    |
 | STL / Holt-    | | Residual:    |   | GBDT / DNN on |     | iForest / auto- |
 | Winters /      | | FFT -> log   |   | labelled fraud|     | encoder recon   |
 | Prophet / LSTM | | amp - avg -> |   | (p_fraud)     |     | error (novelty) |
 | -> y_hat, band | | iFFT saliency|   +-------+-------+     +--------+--------+
 +-------+--------+ +------+-------+           |                      |
         | residual        | saliency map      +----------+-----------+
         v                 v                              v
 +----------------+ +--------------+             +-----------------+
 | Robust z /     | | 1-D CNN      |             | Score fusion    |
 | S-H-ESD test   | | (trained on  |             | (calibrated     |
 | (median, MAD)  | |  injected    |             |  stack / rules) |
 +-------+--------+ |  anomalies)  |             +--------+--------+
         |          +------+-------+                      |
         +--------+--------+                              v
                  v                              +-----------------+
         +-------------------+                   | Action policy:  |
         | Ensemble score in |                   | allow / review /|
         | [0,1] + severity  |                   | block by cost   |
         +---------+---------+                   +-----------------+
                   v
         +-------------------+
         | Dynamic threshold |
         | + min duration    |
         | -> anomaly window |
         +-------------------+
```

## Diagram 4: Real-world system (ThirdEye-style monitoring + RCA)

**Diagram 4 - Business-metric monitoring with dimension drill-down root-cause analysis (LinkedIn ThirdEye / luminol pattern)**

```
 +------------------+    +---------------------+    +--------------------+
 | OLAP store       |--->| Detection pipeline  |--->| Anomaly + merger   |
 | (Pinot cubes:    |    | (per metric x dim,  |    | (merge adjacent    |
 |  metric x dims)  |    |  baseline: wo1w,    |    |  windows, dedupe)  |
 +------------------+    |  wo2w, model-based) |    +---------+----------+
                         +---------------------+              |
                                                              v
 +------------------+    +---------------------+    +--------------------+
 | Correlated       |<---| RCA engine:         |<---| Alert + UI         |
 | metrics, events  |    | dimension drill-down|    | (user marks true / |
 | (deploys,        |    | contribution score, |    |  false, tunes      |
 |  holidays)       |    | cross-correlation   |    |  sensitivity)      |
 +------------------+    | (luminol)           |    +---------+----------+
                         +---------------------+              |
                                                              v
                                                    +--------------------+
                                                    | Feedback -> retune |
                                                    | detection config   |
                                                    +--------------------+
```

## Data & labels

| Source | Label type | Notes |
|---|---|---|
| Incident tickets / postmortems | Weak positive windows | Coarse timing; map to affected series |
| On-call ack / "not an anomaly" clicks | Explicit feedback | Biased to alerted points only (no FN signal) [3] |
| Synthetic injection (spikes, level shifts, dips) | Strong, free | Used to train SR-CNN without manual labels [1][2] |
| Chargebacks, analyst reviews | Delayed positive (fraud) | 30-90 day delay; censoring for recent data [7] |
| Public benchmarks (NAB, Yahoo S5) | Labelled windows | NAB: 58 series, real + artificial [25][10] |

## Features

| Group | Examples | Path |
|---|---|---|
| Raw window | Last w points, normalized | Metrics |
| Decomposition | Trend, seasonal, residual (STL); spectral saliency | Metrics |
| Calendar | Hour-of-day, day-of-week, holiday, deploy flag | Both |
| Cross-series | Sibling dimension values, correlated metric deltas | Metrics / RCA |
| Multivariate window | N correlated metrics x w steps (input to MTAD-GAT) | Metrics [27] |
| Velocity | Txns per card/device/IP in 1m/1h/24h | Transactions |
| Entity history | Typical amount, merchant category, geo distance | Transactions |
| Graph | Shared device/payment instrument, community anomaly | Transactions [14] |

## Model choices (baseline -> advanced)

| Stage | Model | Pros | Cons |
|---|---|---|---|
| 0 | Static thresholds / rules | Simple, explainable | Breaks with seasonality, high fatigue |
| 1 | Rolling mean + k-sigma, week-over-week | Cheap, per-series | Sensitive to outliers in baseline |
| 2 | Robust stats: median/MAD, S-H-ESD | Handles seasonality, robust [8][9] | Batch-ish, tuning max_anoms |
| 3 | Forecast + residual (Holt-Winters, Prophet, EGADS TMM/ADM) | Interpretable band [10] | One model per series; cold start |
| 4 | Robust PCA (RAD), isolation forest | Multivariate, unlabeled [12][4] | Hard to explain per point |
| 5 | SR-CNN (global, trained on synthetic) | No labels, one model for millions [1][2] | Needs good injection design |
| 6 | LSTM/Transformer AE or forecaster | Complex patterns, multivariate | Cost, drift, needs GPUs |
| 7 | Supervised GBDT/DNN on events (fraud) | Best precision with labels [18] | Label delay, misses novel attacks |
| 8 | Meta-learner choosing algo per series | Scales tuning [13] | Extra system complexity |
| 9 | Multivariate GAT (MTAD-GAT: feature + time attention, forecast + reconstruction) | Catches cross-metric anomalies; per-variable attribution [27] | Group of metrics must be pre-defined; GPU training |

## Loss

| Model | Loss / objective |
|---|---|
| Forecaster | MSE or quantile (pinball) loss for prediction band |
| SR-CNN | Cross-entropy on injected-anomaly labels per point [2] |
| Autoencoder | Reconstruction MSE on (mostly) normal data |
| Isolation forest | None (path length => score) |
| Supervised fraud | Weighted BCE / focal loss for class imbalance; cost-sensitive threshold |

## Offline metrics

| Metric | Why |
|---|---|
| Precision / recall / F1 on points | Baseline [1] |
| Point-adjusted or delay-aware F1 | Segment counts if detected within delay k of start [1][30] |
| Caveat: point-adjust inflation | PA can make random scores look strong; report PA%K or raw F1 too [29] |
| NAB score (Standard, Low-FP, Low-FN profiles) | Rewards early detection, penalizes FP [25] |
| Detection delay (median, p95) | Time-to-detect matters for outages |
| Alerts per series per week | Proxy for alert fatigue |
| PR-AUC, recall @ fixed FPR, $ loss caught | Transaction / fraud mode |

## Online metrics / A-B

| Metric | Notes |
|---|---|
| MTTD (mean time to detect) | vs. incidents discovered by humans/customers |
| % incidents caught by detector first | Coverage of real outages |
| Alert precision (ack rate as true) | From on-call feedback [3] |
| Pages per on-call shift | Fatigue guardrail |
| Fraud $ loss, false decline rate, review queue size | Transaction mode |
| A-B design | Shadow mode + interleaved alert rules; can't randomize outages, so backtest + shadow |

## Serving & latency budget

| Step | Metrics path | Transaction path |
|---|---|---|
| Ingest + aggregate | 10-60 s roll-up window | 5 ms |
| Feature fetch | Window from in-memory state | 10-20 ms (online store) |
| Model scoring | <1 ms per series per tick (vectorized) | 5-20 ms |
| Threshold + post-process | ~1 s batching | 2 ms |
| Total | under 1-2 min point-to-alert | under 50-100 ms |
| Scaling trick | Partition by series id; batch thousands of series per CPU call | Cache entity aggregates |

## Monitoring (of the detector itself)

| Signal | Action |
|---|---|
| Alert volume spike across many series | Likely upstream data issue; global suppression |
| Missing data / late points | Distinguish "no data" from "drop to zero" (no-data check in aggregator, Diagram 1) |
| Score distribution drift | Recalibrate thresholds |
| Feedback FP rate per series | Auto-lower sensitivity or switch algo |
| Fraud: score PSI, approval rate | Rollback model |

## Failure modes

| Failure | Mitigation |
|---|---|
| Alert storms from one root cause | Group by topology/dimension; RCA [3][5] |
| Seasonality / holiday false positives | Seasonal decomposition, holiday calendar, wo1w/wo2w baselines |
| Level shift after deploy keeps alerting | Change-point detection, adapt baseline quickly [13] |
| Anomaly contaminates baseline | Robust stats (median/MAD), exclude flagged points from fit [8] |
| Cold-start series | Global model (SR-CNN), borrow from sibling series |
| Sparse / low-volume metrics | Poisson models, larger windows, minimum-volume gates |
| Adversaries adapt (fraud) | Unsupervised + supervised hybrid; graph features [14] |
| Feedback only on alerted points | Periodic random audits for false negatives |

## Interviewer follow-ups

- How do you handle daily/weekly seasonality and holidays without per-series manual tuning?
- Supervised vs unsupervised: what if you have almost no labels?
- How do you generate training labels (synthetic injection, incident logs) and avoid leakage?
- How do you evaluate when anomalies are segments, not points (point-adjusted F1, delay)?
- How do you reduce alert fatigue and group alerts from one root cause?
- How would you do root-cause analysis across dimensions (country x platform x version)?
- How do you scale to 10M series with one model vs one model per series?
- How do you distinguish a real drop from missing/late data?
- What happens after a deploy causes a permanent level shift?
- How do you set thresholds per series; how do you let users tune sensitivity?
- For transactions: how do you combine rules, supervised model and anomaly score?
- How do you handle delayed fraud labels and feedback loops from blocked transactions?
- How do you monitor the detector itself (meta-monitoring)?
- Multivariate: detect anomalies across correlated metrics jointly, and attribute to a variable? [27]
- Why can point-adjusted F1 overstate a detector, and what would you report instead? [29]
- Build vs buy: what if the managed service you depend on is retired? [28]
- Can an LLM help summarize incidents / suggest root causes? [26]

## Portal comparison

| Portal / source | Framing | Model emphasis | Distinctive point |
|---|---|---|---|
| Educative Grokking MLSD [21] | Generic 6-step framework (no verified anomaly chapter) | Applied by candidate | Framework walk-through |
| Chip Huyen exercises [18] | Labelled fraud vs safe records | Supervised; class imbalance | Short prompt, open-ended |
| alirezadir MLSD template [17] | Monitoring section of every design | Drift detection via stats/hypothesis tests | Alarms on pipeline, metrics |
| Microsoft SR-CNN / MTAD-GAT [1][2][27] | Metric anomaly service at scale | SR + CNN (univariate); graph attention (multivariate) | No labels; delay-aware F1 |
| LinkedIn ThirdEye / luminol [3][5] | Business KPI monitoring | Baselines + RCA correlation | Dimension drill-down, feedback UI |
| Twitter S-H-ESD / Yahoo EGADS [8][10] | Time-series outliers | Robust stats / forecast + threshold | Pluggable TMM + ADM modules |

## References

1. Microsoft, "anomalydetector: SR-CNN implementation", GitHub. https://github.com/microsoft/anomalydetector
2. H. Ren et al., "Time-Series Anomaly Detection Service at Microsoft", KDD 2019, arXiv, 2019. https://arxiv.org/abs/1906.03821 (not fetched)
3. LinkedIn Engineering, "Introducing ThirdEye: LinkedIn's Business-Wide Monitoring Platform", engineering.linkedin.com, 2019. https://engineering.linkedin.com/blog/2019/01/introducing-thirdeye--linkedins-business-wide-monitoring-platfor (not fetched)
4. LinkedIn Engineering, "Detecting and preventing abuse on LinkedIn using isolation forests", engineering.linkedin.com, 2019. https://engineering.linkedin.com/blog/2019/isolation-forest
5. LinkedIn, "luminol: anomaly detection and correlation library", GitHub. https://github.com/linkedin/luminol
6. Uber Engineering, "Identifying Outages with Argos, Uber Engineering's Real-Time Monitoring and Root-Cause Exploration Tool", uber.com, 2015. https://www.uber.com/blog/argos-real-time-alerts/ (not fetched)
7. Uber Engineering, "Risk Entity Watch: Using Anomaly Detection to Fight Fraud", uber.com. https://www.uber.com/en-IN/blog/risk-entity-watch/
8. Twitter, "AnomalyDetection (Seasonal Hybrid ESD) R package", GitHub, 2015. https://github.com/twitter/AnomalyDetection
9. Twitter Engineering, "Introducing practical and robust anomaly detection in a time series", blog.twitter.com, 2015. https://blog.twitter.com/engineering/en_us/a/2015/introducing-practical-and-robust-anomaly-detection-in-a-time-series (not fetched)
10. Yahoo, "EGADS: Extensible Generic Anomaly Detection System", GitHub. https://github.com/yahoo/egads
11. N. Laptev, S. Amizadeh, I. Flint, "Generic and Scalable Framework for Automated Time-series Anomaly Detection", KDD 2015. https://s.yimg.com/ge/labs/v2/uploads/kdd2015.pdf
12. Netflix Tech Blog, "RAD - Outlier Detection on Big Data", netflixtechblog.com, 2015. https://netflixtechblog.com/rad-outlier-detection-on-big-data-d6b0494371cc (not fetched)
13. Meta (Facebook Research), "Kats: toolkit to analyze time series data", GitHub. https://github.com/facebookresearch/Kats
14. Grab Engineering, "Unsupervised graph anomaly detection: Catching new fraudulent behaviours", engineering.grab.com. https://engineering.grab.com/graph-anomaly-model
15. Capital One Tech, "Automated detection, diagnosis & remediation of app failure", capitalone.com. https://www.capitalone.com/tech/machine-learning/automated-detection-diagnosis-remediation-of-application-failure/
16. Walmart Global Tech, "Forecast Anomalies in Refrigeration with PySpark & Sensor-data", Medium. https://medium.com/walmartglobaltech/forecast-anomalies-in-refrigeration-with-pyspark-sensor-data-195f23ae24e2
17. alirezadir, "ML System Design" (ml-system-design.md), GitHub, 2025. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/ml-system-design.md
18. C. Huyen, "Machine Learning Systems Design - Exercises", GitHub. https://github.com/chiphuyen/machine-learning-systems-design/blob/master/content/exercises.md
19. K. Pham, "machine-learning-interview", GitHub. https://github.com/khangich/machine-learning-interview
20. Engineer1999, "A Curated List of ML System Design Case Studies", GitHub, 2024. https://github.com/Engineer1999/A-Curated-List-of-ML-System-Design-Case-Studies
21. Educative, "The 6-step ML SD framework" (Grokking the ML System Design Interview), educative.io, 2025. https://www.educative.io/courses/grokking-the-machine-learning-system-design-interview/the-6-step-ml-sd-framework
22. ByteByteGo, "ml-bytebytego: references for Machine Learning System Design Interview", GitHub, 2023. https://github.com/ByteByteGoHq/ml-bytebytego
23. Exponent, "Introduction to the ML System Design Interview", tryexponent.com, 2025. https://www.tryexponent.com/courses/ml-system-design/mlsd-intro
24. Hello Interview, "ML System Design in a Hurry: Delivery Framework", hellointerview.com, 2025. https://www.hellointerview.com/learn/ml-system-design/in-a-hurry/delivery
25. Numenta, "NAB: The Numenta Anomaly Benchmark", GitHub. https://github.com/numenta/NAB
26. Microsoft Research, "Large-language models for automatic cloud incident management", microsoft.com, 2023. https://www.microsoft.com/en-us/research/blog/large-language-models-for-automatic-cloud-incident-management/
27. H. Zhao et al., "Multivariate Time-series Anomaly Detection via Graph Attention Network" (MTAD-GAT, Microsoft), ICDM 2020. https://arxiv.org/abs/2009.02040 (not fetched)
28. Microsoft Learn, "What is Anomaly Detector?" (retirement notice: no new resources from 20 Sep 2023; service retired 1 Oct 2026), learn.microsoft.com. https://learn.microsoft.com/en-us/azure/ai-services/anomaly-detector/overview (not fetched)
29. S. Kim et al., "Towards a Rigorous Evaluation of Time-series Anomaly Detection", AAAI 2022. https://arxiv.org/abs/2109.05257 (not fetched)
30. H. Xu et al., "Unsupervised Anomaly Detection via Variational Auto-Encoder for Seasonal KPIs in Web Applications" (Donut; introduces point-adjust), WWW 2018. https://arxiv.org/abs/1802.03903 (not fetched)
