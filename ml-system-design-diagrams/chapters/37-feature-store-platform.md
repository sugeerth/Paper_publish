# 37. Design an ML Platform: Feature Store + Training + Model Serving (Uber Michelangelo, Meta FBLearner)

> Sourcing note (review, 2026-10): non-GitHub hosts (uber.com, engineering.fb.com, doordash.engineering, infoq.com, arxiv.org) did not resolve from this environment, and the web-search budget was exhausted. Fetched and read: GitHub sources [5][20][21][22][23][30]. The curated lists [20][21] confirm titles, years, and URLs for [1][2][4][6]-[10][14][15][18][31][33][34]. Platform numbers for Michelangelo and FBLearner are quoted from those posts and were not re-read in this pass. Tags below name the company whose published system the framing comes from. How often each company asks this question was not verified.

## Company tags

| Company (system the framing comes from) | Typical framing | Ref |
|---|---|---|
| Uber | "Design Michelangelo": self-serve feature store, training, deploy, and online predictions for ETA, pricing, and fraud | [1][4][35] |
| Meta | "Design FBLearner": feature store, Flow (training DAGs), Predictor (serving); also "design a training data pipeline for recsys" | [2][3][11][19] |
| Airbnb | "Design a feature platform with point-in-time-correct backfills" (Zipline, open-sourced as Chronon) | [10][23] |
| DoorDash | "Design a low-latency online feature store" (Redis, billions of features); declarative real-time features (Riviera, Fabricator) | [8][18][34] |
| LinkedIn | "Design a feature store / feature registry" (Feathr) | [9][30] |
| Netflix, Lyft, Shopify | "Design an ML platform for data scientists"; fact store / time travel (Netflix), feature serving (Lyft) | [15][16][17][31][33] |
| Google | "Design a TFX-style pipeline": data validation, training, model analysis, push to serving | [12] |

Portals and sources that cover this problem, either as a full problem or as the "serving / MLOps" step of every design:

| Portal / source | Coverage | Link |
|---|---|---|
| alirezadir/Machine-Learning-Interviews | Step 5: static (from feature store) vs dynamic (computed online) features; step 7: batch / online / hybrid prediction; step 8: A/B, bandits, shadow, canary; step 9: logging, monitoring, drift | [22] |
| Chip Huyen, "Designing ML Systems" (Ch. 7, 10) and the earlier GitHub notes | Batch vs online prediction, feature stores, ML platform components | [25][26] |
| ByteByteGo (Xu & Aminian) | Serving section of each chapter: batch vs online, feature store, model compression | [27] |
| Exponent / Hello Interview | Infra-style "design an ML platform / model serving" prompts; delivery framework | [28][29] |
| Educative (Khang Pham MLSD) | Feature store, training pipeline, and serving chapters | [24][32] |
| eugeneyan/applied-ml, visenger/awesome-mlops | Curated feature-store and ML-platform case studies (Feature Stores, Existing ML Systems sections) | [20][21] |
| Open-source feature platforms (Feast, Chronon, Feathr) | Reference designs: offline + online store, PIT-correct backfills, registry, online/offline consistency checks | [5][23][30] |

## Requirements & scale

| Category | Requirement | Value (typical assumption unless cited) |
|---|---|---|
| Functional | Register features once and reuse them in training and serving | Michelangelo: about 10k shared features (2017) [1] |
| Functional | Point-in-time-correct training sets (no label leakage) | [5][10] |
| Functional | Batch, streaming, and on-demand (request-time) features | Hive + Kafka/Flink + request [1][4] |
| Functional | Reproducible training workflows (DAGs), hyperparameter search, experiment tracking | FBLearner Flow [2] |
| Functional | Model registry, evaluation reports, one-click deploy, rollback | [1] |
| Functional | Online, batch, and embedded (library) prediction; shadow/A-B | [1][11] |
| Functional | Monitoring: prediction logging, joining with outcomes, drift | [1] |
| Non-functional | Online feature read latency | p99 under 5-10 ms for a batch of about 100 features |
| Non-functional | Prediction latency | Michelangelo: P95 under 5 ms (no Cassandra features), under 10 ms with Cassandra [1] |
| Scale | Prediction QPS | Michelangelo: 250k+ pred/s for the top model (2017) [1]; about 10M pred/s at peak (2024) [35]; FBLearner: 6M+ pred/s (2016) [2] |
| Scale | Models / users | Thousands of models in production; 25%+ of FB engineering used FBLearner Flow [2] |
| Scale | Online store size | TB-scale key-value; DoorDash stores billions of feature values in Redis [8] |
| Non-functional | Freshness | Batch: daily; streaming: seconds to minutes |
| Non-functional | Availability | 99.99% serving; fall back to default or last-known feature values |
| Governance | Lineage, ownership, ACLs, PII tagging, feature deprecation | [9][14][30] |
| Functional | Online/offline consistency measured, not assumed: log online fetches, replay as a backfill, compare | [23] |
| Functional | Input data validation before training (schema, ranges, missing values) | [12] |

## Diagrams

**Diagram 1. End-to-end online path: request to feature assembly to inference to policy to response, with the feature store, model store, and logging**

```text
 +-------------+   predict(entity_ids,ctx)   +------------------------------------+
 | Client svc  |---------------------------->| Prediction Gateway (RPC / REST)    |
 | (ETA, feed, |<----------------------------| authn, quotas, validation          |
 | fraud ...)  |   score + model_version     +-----------------+------------------+
 +-------------+                                               | model_key
                                                               v
 +-------------------------+  route / split  +------------------------------------+
 | Experiment / config svc |---------------->| Model Router                       |
 | A/B, shadow, canary %,  |                 | champion / challenger /            |
 | kill switch             |                 | shadow copy (async)                |
 +-------------------------+                 +-----------------+------------------+
                                                               | feature spec (DSL)
                                                               v
 +--------------------------------------------------------------------------------+
 | Feature Assembly (inside the prediction container)                             |
 | (a) request features: from ctx (time, geo, device)                             |
 | (b) on-demand transforms: DSL (log, bucketize, nearest-zone)                   |
 | (c) precomputed: batched multi-get to the online feature store                 |
 +-----------------+-------------------------------------------+------------------+
                   ^ batched multi-get                         | feature vector
                   v                                           |
 +------------------------------------+                        |
 | ONLINE FEATURE STORE (KV)          |                        |
 | Cassandra / Redis / RocksDB        |                        |
 | key: entity_id -> packed feats     |                        |
 +------------------------------------+                        |
                                                               v
 +------------------------------------+      +------------------------------------+
 | MODEL STORE / REGISTRY             |----->| Inference Engine                   |
 | artifact + metadata + feature spec | load | GBDT / DNN / linear; CPU or GPU    |
 +------------------------------------+      | batching; partitioned (per city)   |
                                             +-----------------+------------------+
                                                               | raw score
                                                               |
                                                               v
                                             +------------------------------------+
                                             | Post-processing / Policy           |
                                             | calibration, thresholds, rules,    |
                                             | fallback model, default on timeout |
                                             +-----------------+------------------+
                                                               | response; async log
                                                               |
                                                               v
                                             +------------------------------------+
                                             | Prediction logger (Kafka)          |
                                             | request_id, features-as-served,    |
                                             | score, model_version, ts           |
                                             +-----------------+------------------+
                                                               |
                                                               v
 +------------------------------------+      +------------------------------------+
 | Outcomes / labels (delayed)        |----->| Log store (Hive / data lake)       |
 | trip done, chargeback, click       | join | monitoring + training data         |
 +------------------------------------+      | (feeds Diagram 2)                  |
                                             +------------------------------------+
```

**Diagram 2. Offline pipeline: logging to labels/joins to features to training to evaluation to registry to deployment, with the feedback loop**

```text
 +----------------+   +----------------+     +------------------------------------+
 | Service DBs    |   | Event logs     |     | Prediction + outcome logs          |
 | (CDC dumps)    |   | (Kafka)        |     | (from Diagram 1)                   |
 +--------+-------+   +--------+-------+     +-----------------+------------------+
          | batch              | stream                        | delayed labels
          v                    v                               v
 +----------------+   +----------------+     +------------------------------------+
 | Batch jobs     |   | Stream jobs    |     | Label service                      |
 | Spark on Hive  |   | Flink / Samza  |     | label windows, maturity cutoff,    |
 | daily / hourly |   | windowed aggs  |     | dedupe, negative sampling          |
 +--------+-------+   +--------+-------+     +-----------------+------------------+
          |                    |                               |
          |                    |                               |
          v                    v                               |
 +-------------------------------------+                       |
 | FEATURE STORE                       |                       |
 | offline: Hive/Parquet (history)     |                       |
 | online : KV (latest per entity)     |                       |
 | registry: owner, schema, version    |                       |
 +------------------+------------------+                       |
                    | point-in-time join                       |
                    |                                          |
                    v                                          |
 +-------------------------------------+      label spine      |
 | Training-set builder (Spark)        |<----------------------+
 | spine(entity,ts,label) AS-OF join   |
 | split train/val/test by time        |
 +------------------+------------------+
                    | versioned snapshot
                    |
                    v
 +-------------------------------------+     +------------------------------------+
 | Training workflows (Flow-style DAG) |     | Experiment tracking / metadata     |
 | operators on Spark / GPU pool       |---->| params, data snapshot, code sha,   |
 | HPO; distributed (Horovod/DDP)      |     | metrics, artifacts                 |
 +------------------+------------------+     +------------------------------------+
                    | candidate model + feature spec
                    |
                    v
 +-------------------------------------+     +------------------------------------+
 | Evaluation gate                     |---->| Fail: report to owner, stop        |
 | AUC, MAE, calibration vs champion;  |fail +------------------------------------+
 | slices, fairness, latency + size    |
 +------------------+------------------+
                    | pass
                    |
                    v
 +-------------------------------------+
 | MODEL REGISTRY                      |
 | version, lineage, stage:            |
 | staging > shadow > canary > prod    |
 +------------------+------------------+
                    | deploy
                    |
                    v
 +-------------------------------------+     +------------------------------------+
 | Serving fleet (Diagram 1)           |---->| Monitoring: drift, skew, KPIs      |
 | online / batch scoring / library    |     | logs -> top-right source box;      |
 +-------------------------------------+     | trigger -> retrain (FEEDBACK)      |
                                             +------------------------------------+
```

**Diagram 3. The core component, feature store internals: a feature definition becomes offline and online materialization**

```text
 +--------------------------------------------------------------------------------+
 | Feature definition (declarative DSL / Python, reviewed in git)                 |
 | entity: driver_id   source: trips_stream   ttl: 2h                             |
 | agg: count(trips), avg(rating) over windows [1h, 24h, 7d]                      |
 | owner, tags (PII), version                                                     |
 +---------------------------------------+----------------------------------------+
                                         | compile / register
                                         |
                                         v
 +--------------------------------------------------------------------------------+
 | FEATURE REGISTRY (metadata DB): schemas, lineage, ACLs, stats, consumers       |
 +------------------+------------------------------------------+------------------+
                    | batch plan                               | stream plan
                    |                                          |
                    |                                          |
                    v                                          v
 +------------------------------------+      +------------------------------------+
 | Batch engine (Spark)               |      | Stream engine (Flink)              |
 | backfill full history              |      | sliding / tumbling windows         |
 +------------------+-----------------+      +-----------------+------------------+
                    |                                          |
                    |                                          |
                    v                                          v
 +--------------------------------------------------------------------------------+
 | Materialization / writers                                                      |
 | batch: append history to offline; upsert latest value to online                |
 | stream: upsert online within seconds; append the same rows to offline log      |
 +------------------+------------------------------------------+------------------+
                    | append                                   | upsert
                    |                                          |
                    |                                          |
                    v                                          v
 +------------------------------------+      +------------------------------------+
 | OFFLINE STORE                      |      | ONLINE STORE (KV)                  |
 | Hive / Parquet / Iceberg           |      | Cassandra / Redis                  |
 | rows: (entity, feature_ts, value)  |      | hash(entity) -> packed values, TTL |
 +------------------+-----------------+      +-----------------+------------------+
                    |                                          |
                    |                                          |
                    |                                          |
                    v                                          v
 +------------------------------------+      +------------------------------------+
 | Training sets                      |      | Serving (Diagram 1)                |
 | get_historical_features            |      | get_online_features(entity_ids)    |
 | (entity, ts) AS-OF join            |      | p99 < 5-10 ms                      |
 +------------------+-----------------+      +-----------------+------------------+
                    | offline values                           | online values (logged)
                    |                                          |
                    |                                          |
                    v                                          v
 +--------------------------------------------------------------------------------+
 | Consistency monitor: log online fetches, replay offline, compare per key/ts    |
 | -> training-serving skew alerts; freshness lag per feature; null rate; drift   |
 +--------------------------------------------------------------------------------+
```

**Diagram 4. Point-in-time (AS-OF) join, which prevents leakage when the spine is built**

```text
 time ---------------------------------------------------------------------->
 driver_42 feature "trips_24h":  v=10 @t1      v=14 @t3           v=20 @t6
 label event (spine row):                 ts=t4, label=late_eta=1
 AS-OF join picks latest value with feature_ts <= t4    --> 14  (correct)
 naive join on "latest" picks                           --> 20  (LEAKAGE)
 ttl check: if t4 - t3 > ttl                            --> null / default
```

**Diagram 5. Real systems side by side: Uber Michelangelo (2017, with Palette) and Meta FBLearner (2016)**

```text
 UBER MICHELANGELO [1][4][35]               META FBLEARNER [2][3]
 +-----------------------------------+      +-----------------------------------+
 | Data mgmt / Palette feature store |      | FBLearner Feature Store           |
 | Hive (offline) + Cassandra        |      | catalog of features for training  |
 | (online); Kafka -> Samza/Flink    |      | and real-time prediction          |
 | for near-real-time features       |      +-----------------+-----------------+
 | DSL for select / transform        |                        |
 +----------------+------------------+                        v
                  v                         +-----------------------------------+
 +-----------------------------------+      | FBLearner Flow                    |
 | Training: Spark / distributed     |      | workflows = Python DAGs           |
 | GBDT, linear, DNN (Horovod);      |      | operators + typed channels        |
 | HPO; partitioned models           |      | futures -> parallel exec on fleet |
 +----------------+------------------+      | experiment UI, reuse of pipelines |
                  v                         +-----------------+-----------------+
 +-----------------------------------+                        v
 | Model eval reports + repository   |      +-----------------------------------+
 | (Cassandra): config, metrics,     |      | FBLearner Predictor               |
 | feature report, lineage           |      | online inference service          |
 +----------------+------------------+      | 6M+ predictions/s (2016)          |
                  v                         +-----------------------------------+
 +-----------------------------------+      Later Meta: Looper (product-level
 | Deploy: offline batch | online    |      AutoML / decision platform) [11];
 | containers | library; UUID/tag    |      DSI data ingestion for DLRM
 | routing, A/B by tag               |      training [19]
 +----------------+------------------+
                  v
 +-----------------------------------+
 | Serving P95 < 5-10 ms; monitoring |
 | by joining logged preds with      |
 | outcomes                          |
 +-----------------------------------+
```

## Data & labels

| Data | Source | Notes |
|---|---|---|
| Entity snapshots | DB CDC dumps into the lake | Daily partitions; keyed by entity_id |
| Events | Kafka topics | Feed both stream features and the offline log |
| Request context | Prediction request | Only available at serve time; it must be logged so training can use it |
| Labels | Outcome tables (trip ETA actual, chargeback, click) | Delayed; label window and maturity cutoff per use case |
| Prediction logs | Serving fleet | Features-as-served give skew-free training data ("log and wait") |

## Features (platform-level feature types)

| Type | Example | Store / compute | Freshness |
|---|---|---|---|
| Batch aggregate | restaurant avg prep time over 30d | Spark to offline, then copied online | Daily |
| Streaming aggregate | restaurant prep time, last 1h | Flink to online and offline log | Seconds to minutes |
| On-demand / request | distance(pickup, dropoff), hour-of-day | Computed in the serving container (DSL) | Real time |
| Embeddings | user / item vectors | Batch job to the online KV or a vector index | Daily |
| Cross / derived | ratio of two features | DSL transform, shared code for training and serving | Inherits from inputs |

## Model choices (platform design choices, baseline to advanced)

| Decision | Baseline | Advanced | Trade-off |
|---|---|---|---|
| Feature store | Shared tables + ad-hoc Redis | Registry + dual store + PIT joins (Feast, Palette, Feathr) | Consistency and reuse vs build cost |
| Transform logic | Separate SQL (train) and Java (serve) | One DSL compiled for both [1] | Removes skew; DSL expressiveness is limited |
| Training orchestration | Cron + notebooks | DAG workflows (FBLearner Flow, Airflow, Kubeflow, Flyte) [2][16] | Reproducibility vs flexibility |
| Model formats | Per-team servers | Standard artifact + feature spec; GBDT/DNN runtimes | Simpler ops; harder for exotic models |
| Serving | One service per model | Multi-tenant prediction service with routing by tag | Efficiency vs noisy neighbors |
| Online store | Redis | Cassandra / ScyllaDB / custom KV with compression [8] | Latency vs cost per TB |
| Scope | Platform for ML engineers | Self-serve for analysts (Looper-style AutoML) [11] | Adoption vs guardrails |

## Loss / objective

| Level | Objective |
|---|---|
| Per model | Defined by the team (log-loss, MAE, ranking loss); the platform supplies standard evaluators |
| Platform | Minimize time-to-production and skew incidents; maximize feature reuse and serving efficiency (cost per 1k predictions) |

## Offline metrics

| Metric | Target (typical) |
|---|---|
| Model quality vs champion (AUC, MAE, NDCG) by slice | Gate: no regression above threshold |
| Training-serving skew (feature value mismatch rate) | Under 0.1% of sampled keys |
| PIT-join correctness tests (no feature_ts > label_ts) | 0 violations |
| Backfill time for a new feature (1 year of history) | Hours, not days |
| Pipeline success rate / reproducibility (same snapshot gives same metrics) | Above 99% |

## Online metrics / A-B

| Metric | Use |
|---|---|
| Business KPI of each model (ETA error, fraud loss, CTR) | A/B of challenger vs champion, routed by the platform |
| Shadow-mode score diff | Safety check before any user exposure |
| Platform KPIs: models shipped/week, time idea to prod, % features reused | Platform-team success |
| Serving p50/p99, error rate, cost per prediction | SLO dashboards |

## Serving & latency budget

| Stage (online prediction, about 10 ms p99) | Budget |
|---|---|
| Gateway + routing | 1 ms |
| Online feature multi-get (batched, local-AZ KV) | 3-4 ms |
| On-demand transforms | 0.5 ms |
| Inference (GBDT on CPU, or a small DNN) | 2-4 ms |
| Post-processing + async logging | 0.5 ms |
| Options | Batch prediction precomputed into the KV for slow-changing outputs; library mode (in-process) for ultra-low latency; GPU servers with dynamic batching for large DNNs |

## Monitoring

| Signal | How |
|---|---|
| Feature freshness lag | Per-feature watermark vs SLA; page the owner |
| Feature distribution drift (PSI, null %) | Compare online samples with training stats from the registry |
| Prediction drift / calibration | Logged predictions joined with delayed outcomes [1] |
| Skew | Log online fetches, replay them as an offline backfill, compare per key and ts (Chronon-style) [23]; Diagram 3 |
| Data validation | Schema and distribution checks on every training snapshot; block the run on anomalies [12] |
| Serving health | p99, QPS, timeouts, fallback rate, per-tenant quotas |
| Lineage / audit | Model to features to data sources, for incidents and deprecation |

## Failure modes

| Failure | Mitigation |
|---|---|
| Training-serving skew (different code paths) | Single DSL, logging of features-as-served, skew monitors [1][23] |
| Hidden pipeline debt (glue code, undeclared consumers, feedback loops) | Registry of consumers, typed feature contracts, ownership [13] |
| Label leakage from non-PIT joins | AS-OF join API by default; tests that flag future timestamps |
| Stale online features (stream job down) | TTLs, freshness alerts, fallback to batch values |
| Hot keys / KV overload | Caching, request coalescing, replicas, compression [8] |
| Bad model pushed | Eval gate, shadow, canary, auto-rollback on SLO or KPI breach |
| Feature deleted or changed under a live model | Versioned features, consumer registry, deprecation workflow |
| Cost explosion (unused features) | Usage tracking, auto-expire unused features, tiered storage |
| Noisy neighbor in a multi-tenant serving fleet | Per-model resource isolation, quotas, separate pools for tier-0 models |

## Interviewer follow-ups

- How do you guarantee a training row never sees a feature value from after its label timestamp?
- Why keep both an offline and an online store? Could one store serve both?
- How do you make the transform code identical in training and serving (DSL, shared library, or logging features-as-served)?
- A new streaming feature needs 1 year of history to train on. How do you backfill it?
- How would you bring p99 online feature reads under 5 ms with 300 features per request at 1M QPS?
- How do you detect and debug training-serving skew in production?
- How do you deploy and roll back a model safely (shadow, canary, A/B by tag)?
- Batch vs online vs embedded prediction: when would you choose each?
- How do you support thousands of per-city or partitioned models without thousands of services?
- What metadata goes into the model registry, and how do you reproduce a model from 6 months ago?
- How do you handle delayed labels (fraud chargebacks after 60 days) in retraining and monitoring?
- How would you make the platform self-serve for non-ML engineers (Looper / AutoML)?
- How do you manage GPU training capacity across teams (quotas, preemption, priority)?
- How do you govern PII features (ACLs, retention, deletion requests)?
- How does the platform change for LLMs (prompt/model registry, gateway, evaluation)? [35]
- How do you prove online and offline features match (log-and-replay consistency score) and alert on it? [23]
- How do you backfill a windowed aggregate for a new feature over a year of events without recomputing the world (tiled / incremental aggregation)? [23][33]
- Where would you put data validation, and what happens to a scheduled retrain when it fails? [12]

## Portal comparison

| Source | Framing | Emphasis | Gaps |
|---|---|---|---|
| Uber Michelangelo blog [1][35] | End-to-end platform as six workflow steps | Shared feature store, DSL, eval reports, deploy modes, P95 latency | Light on governance and cost |
| Meta FBLearner [2][3] | Feature Store + Flow + Predictor | Reusable DAG workflows, experiment management at scale | Few details on the feature store internals |
| Feast / Tecton docs [5][7] | Feature store as a data layer | Offline/online stores, PIT correctness, materialization | Training and serving are out of scope |
| Chronon / Feathr READMEs [23][30] | Feature platform as code | PIT backfills, online fetch API, registry and lineage, consistency checks | No model training or serving |
| Chip Huyen DMLS [25] | Infra and tooling chapter | Batch vs online prediction, feature store, build vs buy | No interview-style walkthrough |
| alirezadir template [22] | MLSD step checklist | Static vs dynamic features, deployment, monitoring | Treats the platform as a sub-step |
| ByteByteGo / Educative [27][32] | Per-problem serving sections | Batch vs online serving, model compression | No standalone platform chapter |

## References

1. Uber Engineering, "Meet Michelangelo: Uber's Machine Learning Platform", uber.com, 2017. https://www.uber.com/blog/michelangelo-machine-learning-platform/
2. Meta Engineering, "Introducing FBLearner Flow: Facebook's AI backbone", engineering.fb.com, 2016. https://engineering.fb.com/ml-applications/introducing-fblearner-flow-facebook-s-ai-backbone/
3. ZenML MLOps Database, "Meta FBLearner: Introducing FBLearner Flow, Facebook's AI backbone" (summary), zenml.io, 2016. https://www.zenml.io/mlops-database/meta-fblearner-introducing-fblearner-flow-facebooks-ai-backbone
4. Uber, "Michelangelo Palette: A Feature Engineering Platform at Uber" (talk), InfoQ, 2019. https://www.infoq.com/presentations/michelangelo-palette-uber/
5. Feast, "feast: open source feature store for machine learning" README, GitHub, 2025. https://github.com/feast-dev/feast
6. Gojek / Google Cloud, "Introducing Feast: an open source feature store for machine learning", cloud.google.com, 2019. https://cloud.google.com/blog/products/ai-machine-learning/introducing-feast-an-open-source-feature-store-for-machine-learning
7. Tecton, "What is a Feature Store?", tecton.ai, 2020. https://www.tecton.ai/blog/what-is-a-feature-store/
8. DoorDash Engineering, "Building a Gigascale ML Feature Store with Redis, Binary Serialization, String Hashing, and Compression", doordash.engineering, 2020. https://doordash.engineering/2020/11/19/building-a-gigascale-ml-feature-store-with-redis/
9. LinkedIn Engineering, "Open sourcing Feathr: LinkedIn's feature store for productive machine learning", engineering.linkedin.com, 2022. https://engineering.linkedin.com/blog/2022/open-sourcing-feathr---linkedin-s-feature-store-for-productive-m
10. Airbnb, "Zipline: Airbnb's Machine Learning Data Management Platform" (talk), YouTube, 2018. https://www.youtube.com/watch?v=Tg5VEMEsC-0
11. Meta AI, "Inside Meta's AI optimization platform for engineers across the company" (Looper), ai.facebook.com, 2022. https://ai.facebook.com/blog/looper-meta-ai-optimization-platform-for-engineers/ (paper: https://arxiv.org/abs/2110.07554)
12. D. Baylor et al., "TFX: A TensorFlow-Based Production-Scale Machine Learning Platform", KDD / Google Research, 2017. https://research.google/pubs/tfx-a-tensorflow-based-production-scale-machine-learning-platform/
13. D. Sculley et al., "Hidden Technical Debt in Machine Learning Systems", NeurIPS, 2015. https://proceedings.neurips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf
14. Hopsworks, "Feature Store: The missing data layer for Machine Learning pipelines?", hopsworks.ai, 2018. https://www.hopsworks.ai/post/feature-store-the-missing-data-layer-in-ml-pipelines
15. Netflix Tech Blog, "Evolution of ML Fact Store", netflixtechblog.com, 2022. https://netflixtechblog.com/evolution-of-ml-fact-store-5941d3231762
16. Lyft Engineering, "LyftLearn: ML Model Training Infrastructure built on Kubernetes", eng.lyft.com, 2021. https://eng.lyft.com/lyftlearn-ml-model-training-infrastructure-built-on-kubernetes-aef8218842bb
17. Shopify Engineering, "The Magic of Merlin: Shopify's New Machine Learning Platform", shopify.engineering, 2022. https://shopify.engineering/merlin-shopify-machine-learning-platform
18. DoorDash Engineering, "Introducing Fabricator: A Declarative Feature Engineering Framework", doordash.engineering, 2022. https://doordash.engineering/2022/01/11/introducing-fabricator-a-declarative-feature-engineering-framework/
19. M. Zhao et al. (Meta), "Understanding Data Storage and Ingestion for Large-Scale Deep Recommendation Model Training", arXiv, 2021. https://arxiv.org/abs/2108.09373
20. E. Yan, "applied-ml: papers and tech blogs on ML in production" (Feature Stores, MLOps Platforms sections), GitHub, 2024. https://github.com/eugeneyan/applied-ml
21. visenger, "awesome-mlops" (Feature Stores, Existing ML Systems sections), GitHub, 2024. https://github.com/visenger/awesome-mlops
22. alirezadir, "ML System Design" template, GitHub, 2025. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/ml-system-design.md
23. Airbnb, "Chronon: a data platform for serving AI/ML applications" README, GitHub, 2025. https://github.com/airbnb/chronon
24. khangich, "machine-learning-interview" (Educative MLSD course companion notes), GitHub, 2021. https://github.com/khangich/machine-learning-interview
25. C. Huyen, "Designing Machine Learning Systems" (Ch. 7 Deployment, Ch. 10 Infrastructure and Tooling), O'Reilly, 2022. https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/
26. C. Huyen, "machine-learning-systems-design", GitHub, 2019. https://github.com/chiphuyen/machine-learning-systems-design
27. A. Aminian & A. Xu, "Machine Learning System Design Interview", ByteByteGo, 2023. https://bytebytego.com/intro/machine-learning-system-design-interview
28. Exponent, "Introduction to the ML System Design Interview" (course), tryexponent.com, 2025. https://www.tryexponent.com/courses/ml-system-design/mlsd-intro
29. Hello Interview, "ML System Design in a Hurry: Delivery Framework", hellointerview.com, 2025. https://www.hellointerview.com/learn/ml-system-design/in-a-hurry/delivery
30. Feathr (LinkedIn, now LF AI & Data), "feathr" README, GitHub, 2024. https://github.com/feathr-ai/feathr
31. Lyft Engineering, "ML Feature Serving Infrastructure at Lyft", eng.lyft.com, 2021. https://eng.lyft.com/ml-feature-serving-infrastructure-at-lyft-d30bf2d3c32a
32. Educative / K. Pham, "Machine Learning System Design" (course), educative.io, 2021. https://educative.io/courses/machine-learning-system-design/introduction
33. Netflix Tech Blog, "Distributed Time Travel for Feature Generation", netflixtechblog.com, 2016. https://netflixtechblog.com/distributed-time-travel-for-feature-generation-389cccdd3907
34. DoorDash Engineering, "Building Riviera: A Declarative Real-Time Feature Engineering Framework", doordash.engineering, 2021. https://doordash.engineering/2021/03/04/building-a-declarative-real-time-feature-engineering-framework/
35. Uber Engineering, "From Predictive to Generative: How Michelangelo Accelerates Uber's AI Journey", uber.com, 2024 (host unreachable in this review). https://www.uber.com/blog/from-predictive-to-generative-ai/
