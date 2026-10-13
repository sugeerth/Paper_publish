# 13. Infrastructure for Closed Loops: Continuous Training, Logging, Monitoring, Registries and Safe Rollout

## Overview

Every closed-loop system in this atlas, from ads click models to fine-tuned LLM agents, eventually hits the same plumbing problem. A loop of the form "serve, observe, learn, redeploy" only works if five pieces of infrastructure exist and agree with one another:

1. **A continuous training (CT) pipeline.** An orchestrated, repeatable DAG that turns new data into a candidate model without a human running notebooks.
2. **Feature and label logging.** A record of what the model actually saw at serving time and what happened afterwards, joined into training examples.
3. **Data quality and drift monitoring.** Checks on inputs, predictions and (when labels arrive) quality, which act as both safety interlocks and retraining triggers.
4. **A model registry with lineage.** The system of record for which artifact is approved, which one is serving, and what data and code produced each.
5. **Progressive delivery.** Canary, shadow and A/B rollout with automated analysis and a rollback path that is faster than the loop can do damage.

Google's MLOps guidance frames this as a maturity ladder. Level 0 is manual and notebook-driven. Level 1 adds an orchestrated pipeline that runs continuous training in production, triggered by a schedule, by new data, or by degradation and drift alerts, and gated by data and model validation, a feature store and a metadata store. Level 2 adds CI/CD for the pipeline itself [1]. Chip Huyen describes a parallel ladder on the training side: manual stateless retraining, scheduled retraining, stateful (warm-started) retraining, and finally continual learning triggered by data or performance signals [54].

This chapter is about the machinery at Levels 1 and 2. It does not cover streaming parameter-server training, which Chapter 5 treats in depth. The chapter is organized around the loop's data flow: pipelines (Section 2), logging (Section 3), validation and drift (Section 4), registries and lineage (Section 5), triggers (Section 6) and rollout (Section 7). It then compares the designs and lists failure modes and takeaways.

A note on sources. Most of the systems here are open source, so the primary sources are the projects' own design documents and READMEs. Where a figure comes from a project's self-description (for example, Netflix's Metaflow scale numbers or whylogs benchmark costs), it is reported as the project states it and has not been independently reproduced.

### Reference architecture

The figure below is a synthesis of the designs in this chapter. Each box lists the systems that publish a version of it.

```
                 +-----------------------------------------------------+
                 |  SERVING  (KServe, Seldon, SageMaker endpoint)      |
 requests ------>|  model vN (stable)  <-- traffic split -->  vN+1     |
                 |   |  payload logger [35] / data capture [36]        |
                 +---|------------------------------------------------+
                     | logged features + predictions (+ keys, ts)
                     v
 +-----------+   +-------------------+   +---------------------------+
 | outcomes  |-->| label join / log- |-->| training data spans       |
 | (clicks,  |   | and-wait [29][30] |   | (ExampleGen {SPAN} [7],   |
 |  returns) |   | backfill (Chronon)|   |  Airflow assets [27])     |
 +-----------+   +-------------------+   +-------------+-------------+
                                                       |
             data validation: schema, skew, drift      v
             (TFDV [13], Deequ [37], GX [39]) --> [ CT PIPELINE ]
                                                  TFX / KFP / Metaflow
                                                  Trainer (warm start)
                                                       |
             model validation: baseline compare,       v
             slices, thresholds (Evaluator [8]) --> blessing
             infra validation (InfraValidator [9]) --> infra blessing
                                                       |
                                                       v
                         +--------------------------------------------+
                         | REGISTRY + METADATA (MLflow aliases [44],  |
                         | MLMD lineage [12], Metaflow namespaces)    |
                         +---------------------+----------------------+
                                               | promote alias / push
                                               v
                         progressive delivery: canary %, analysis runs,
                         shadow/mirror, auto-abort (KServe [46], Argo
                         Rollouts [48][49], Flagger [50], Seldon [51])
                                               |
   monitors (Model Monitor [36], Evidently [43], NannyML [42], alibi
   [41], Feast DQM [33]) ---- alerts ----> retraining trigger / rollback
```

## 1. Why closed loops need dedicated infrastructure

### 1.1 The debt argument

Sculley et al.'s "Hidden Technical Debt in Machine Learning Systems" is the canonical statement of the problem. Its best-known figure shows ML code as a small box surrounded by much larger systems for data collection, feature extraction, data verification, configuration, serving, monitoring and resource management [2]. The paper's loop-specific warnings map directly onto the infrastructure in this chapter:

- **Direct and hidden feedback loops**, where a model influences its own future training data, call for logging of what was served and why [2].
- **Changes in the external world** call for monitoring of **prediction bias** (the distribution of predicted labels should match observed labels) and for **action limits** as sanity checks [2].
- **Undeclared consumers and data dependencies** call for lineage, which is what a metadata store and registry provide [2].

The follow-up "ML Test Score" turned these concerns into 28 tests across data, model, infrastructure and monitoring, scored for production readiness [3]. Zinkevich's Rules of ML adds the logging rule that underpins every loop in this chapter. Rule #29 says to log the features actually used at serving time and to train on them, at least for a sample of traffic, to remove training/serving skew [4].

### 1.2 The control-system view

It helps to treat this infrastructure as a control system with two nested loops.

- The **inner loop** is CT. A trigger fires, the pipeline trains a candidate, validators bless or reject it, and the registry records it.
- The **outer loop** is release. The candidate is exposed to a growing slice of traffic while monitors compare it with the incumbent, and the rollout either completes or reverts.

Each loop needs a sensor (logging and monitoring), a controller (trigger policy and gate thresholds) and an actuator (the pipeline run and the traffic split). Most production incidents in closed loops come from a mismatch in timescale between these parts. A typical example is a rollout analysis that measures infrastructure metrics within minutes while the quality signal that matters, the label, arrives days later.

## 2. Continuous training pipelines

### 2.1 TFX (Google): the validated-pipeline reference design

**Problem.** Google built TFX as "an end-to-end platform for deploying production ML pipelines" [5]. The design goal is that every run of a pipeline produces a model that is validated on data, on quality and on servability before anything is pushed.

**Architecture.** The standard components form a fixed DAG [6]:

```
ExampleGen -> StatisticsGen -> SchemaGen -> ExampleValidator
     |                                          |
     +--------------> Transform -----------> Trainer (+ Tuner)
                                                 |
           Resolver(LatestBlessedModel) -> Evaluator --(blessing)--+
                                                 |                  |
                                         InfraValidator --(infra)--+--> Pusher
                         every step records artifacts/executions in MLMD
```

- **ExampleGen** ingests data and organizes it into **spans, versions and splits** [7]. A span is a grouping of examples (for example, a day or an hour). A version is a revision of a span, such as a cleaned copy. A split is a subdivision of a version, most often train and eval. Input glob patterns use `{SPAN}`, `{VERSION}` or date specs (`{YYYY}{MM}{DD}`, where the span number is days since 1970-01-01). By default the pipeline processes the latest span, records the span number in metadata, and picks up a newly arrived span when it is re-triggered [7]. This is the mechanism that turns a scheduled or event-triggered pipeline run into continuous training.
- **StatisticsGen, SchemaGen and ExampleValidator** compute statistics, infer a schema and flag anomalies such as missing values and wrong types [6]. These are TFDV under the hood (Section 4.1).
- **Transform** emits a SavedModel that contains its preprocessing logic [6]. The same transformation graph is therefore used in training and serving, which removes one class of skew by construction.
- **Trainer** runs a user `run_fn(fn_args)` and writes the model to `fn_args.serving_model_dir`. Its optional inputs include "pre-trained models used for scenarios such as warmstart" [11]. This is the hook for Huyen's stateful retraining stage [54].
- **Evaluator** is the quality gate. When validation is enabled, it evaluates the candidate and a **baseline** on the same eval data. The baseline is normally the latest blessed model, supplied by a Resolver that uses `LatestBlessedModelStrategy` (marked experimental in the guide) [8]. Two kinds of threshold apply:
  - a `GenericChangeThreshold` relative to the baseline, with a direction (for example, `HIGHER_IS_BETTER`) and an absolute tolerance. The guide's example uses a tolerance of -1e-10, meaning "not meaningfully worse";
  - a `GenericValueThreshold`, an absolute floor, for example `BinaryAccuracy` with a `lower_bound` of 0.5.

  Thresholds apply to every metric and every slice defined in the `SlicingSpec`, so a model that improves overall but regresses on a slice is not blessed. Change thresholds are skipped when there is no baseline, as on the first run [8].
- **InfraValidator** is a pre-production canary for servability. It launches a sandboxed model server with the candidate and either checks that the model loads (`LOAD_ONLY`) or also sends sample requests (`LOAD_AND_QUERY`). It does not check whether predictions are correct. The guide insists that the sandbox must match production: same server binary, same resources and same configuration, or the check proves little. It supports local Docker and Kubernetes, and it is designed mainly for TensorFlow Serving and SavedModel. With `RequestSpec.make_warmup=True` it also emits a model bundled with warmup requests [9].
- **Pusher** deploys only when it has the Evaluator's blessing and, optionally, the InfraValidator's `infra_blessing`. It writes the SavedModel plus versioning metadata to a push destination [10].
- **ML Metadata (MLMD)** records every artifact and execution (Section 5.1) [12].

**Orchestration.** The same pipeline definition runs on Apache Airflow, on Kubeflow Pipelines, or with Apache Beam, which also serves as the data-parallel backend on Flink, Spark or Dataflow [6].

**Lessons.**
- *Blessing is a first-class artifact.* Because blessings are artifacts in metadata, "the latest blessed model" is a query, not a convention. The baseline for the next run's comparison is therefore always well defined.
- *Validation is layered.* Data validation, quality validation and infrastructure validation are separate gates with separate failure semantics.
- *Failure modes.* An absolute floor with no change threshold lets slow regressions through, one acceptable step at a time. A baseline that is the *latest blessed* model rather than the *serving* model can diverge from production if pushes fail. An InfraValidator sandbox that differs from production gives false confidence [8][9].

### 2.2 Kubeflow Pipelines: portable container DAGs

**Problem.** Kubeflow Pipelines (KFP) targets "end-to-end orchestration of machine learning pipelines", easy experimentation and reuse of components on Kubernetes [16].

**Architecture.** A pipeline is a DAG of components, each run as one container execution that may produce ML artifacts, with control flow allowed [17]. The Python SDK compiles a pipeline into a platform-neutral **IR YAML** that can be submitted to any KFP-conformant backend. Two backends are named: the open-source KFP backend, either inside Kubeflow or standalone, and Google Cloud Vertex AI Pipelines [17]. The open-source backend runs on Argo Workflows (the README lists v3.7 or v4.1) with MySQL v8, and it uses the Argo Emissary executor. Argo 3.x will not be supported in KFP 3.0 [16]. KFP manages and visualizes pipeline definitions, runs, experiments and artifacts [17].

**Caching.** KFP reuses a component's outputs when it runs again "with the same inputs and parameters" and the earlier output still exists. Caching can be disabled per task (`set_caching_options(enable_caching=False)`), per run, at compile time, or with an environment variable [18]. The caching guide does not describe expiry [18]. For closed loops this is the main KFP pitfall. A data-loading step whose *parameters* do not change, such as "read table X", will hit the cache and silently retrain on stale data unless the data version is passed in as a parameter or caching is disabled for ingestion steps.

**Lessons.** The IR is the important design choice: the same compiled pipeline runs on a self-managed cluster or a managed service [17]. Content-addressed caching saves compute, but only if every input that can change, *including data snapshots*, is part of the cache key.

### 2.3 Metaflow and Maestro (Netflix): human-centric flows with reactive orchestration

**Problem and scale.** Metaflow is a Python framework for taking ML systems from prototype to production. It was developed at Netflix and is now supported by Outerbounds. The README states that at Netflix it supports over 3,000 AI and ML projects and runs hundreds of millions of compute jobs that process petabytes of data. It also manages tens of petabytes of models and artifacts [19]. Its listed features include versioning and experiment tracking through a Client API, CPU and GPU scaling, dependency management, one-command deployment to production orchestrators, and "reactive orchestration" through event triggering [19].

**Production orchestration.** Metaflow defines a production deployment as one that runs without human intervention and in a highly available manner. The docs list four schedulers. **Argo Workflows** is the only one that supports event triggering, which makes it the recommended default. **AWS Step Functions** suits minimal operations, **Airflow** suits existing Airflow estates but has more limitations, and **Kubeflow** suits existing Kubeflow deployments [20]. At Netflix the internal orchestrator is **Maestro**, a workflow-as-a-service that its README says serves thousands of users, schedules hundreds of thousands of workflows and runs millions of jobs a day under a strict SLO [26].

**Event triggering, the closed-loop primitive.** The Metaflow docs frame events as "a clear interface between workflows", which lets teams build and run upstream and downstream flows independently [21]:

- `@trigger(event='data_updated')` starts a deployed flow when a named event occurs. Event names behave like topics. A list of events can be required, all within a configured time window. An external system publishes with `ArgoEvent(name=...).publish(payload=...)`, and payload fields map onto flow `Parameter`s, explicitly when names differ [22].
- `@trigger_on_finish(flow='FirstFlow')` chains flows. The triggered run can read the upstream run's artifacts, for example `current.trigger.run.data.model` [23].
- The docs explicitly warn that flows which trigger each other can create an **infinite loop** that consumes cluster resources, to be broken by deleting the workflow in the Argo UI [23]. This is a direct, documented instance of closed-loop runaway at the orchestration layer.

**Champion/challenger through namespaces.** `@project` gives every deployment an isolated namespace. User deployments are prefixed with the user name, `--branch` creates shareable test branches, and `--production` creates the official deployment, which requires an authorized production token. `--production --branch` allows several production variants [25]. Event-triggered chains can be deployed per branch, so a `new_model` branch runs fully isolated next to production for an A/B comparison. A variant can also consume upstream production results by naming the full project-branch flow in `trigger_on_finish` [24]. Code can check `current.is_production` to make sure only production writes to production tables [25].

**Lessons.** Netflix's design treats isolation as the precondition for safe iteration. A challenger is a full parallel copy of the pipeline, not a flag inside one. Event triggering replaces calendar retraining with data-driven retraining, but it brings the risk of trigger storms and cycles, so the event graph must be acyclic or rate-limited.

### 2.4 Data-aware scheduling in general-purpose orchestrators

Two other widely used orchestrators provide the same primitives.

- **Apache Airflow assets** (called "datasets" before Airflow 3.0) let a DAG run when an upstream asset, identified by a URI, is updated. A producer task declares the asset in `outlets`, and its completion emits an asset event that can carry metadata such as row counts. Consumer DAGs can read past events through `inlet_events`, and partitioned assets map upstream partition keys to downstream runs [27]. Airflow treats the URI as opaque and does not check that the data actually changed. Its trust model is "implicit": anyone allowed to create asset events can trigger the downstream DAGs [27]. For a retraining DAG, that means the trigger is only as trustworthy as the permissions on asset-event creation.
- **Flyte** is a Kubernetes-native orchestrator for data and ML pipelines with output caching (`cache=True`), immutable executions for reproducibility, and scheduled runs. Named adopters include LinkedIn and Spotify [28].

## 3. Feature and label logging

### 3.1 Log-and-wait versus backfill

A closed loop needs training examples that pair the features the model saw with the outcome that followed. There are two ways to get the features, and Airbnb's Chronon README states the trade-off clearly [29]:

- **Log-and-wait.** "Users start with the data that is available in the online serving environment." Features are logged at serving time to the warehouse, and the model trains once enough data has accumulated. Every training feature is guaranteed to exist at serving time. The costs are slow data accumulation, difficulty with windowed aggregations, and transformation logic leaking into the application layer.
- **Replicated offline-online.** Models train on warehouse data and teams re-implement the features online. This gives broad data access and warehouse compute, but it is error-prone, produces training/serving inconsistency, and does not scale across many models [29].

Chronon's answer is to define features once (`GroupBy` aggregations and `Join`s keyed by entity and timestamp), compile them to Spark for **point-in-time-correct backfills** and serve the same definitions from a KV store [29]. Feast takes the same approach. Its point-in-time joins ensure that "future feature values do not leak to models during training", and materialization copies offline values into the online store [31]. Feast **push sources** let a stream write fresh values to the online store, the offline store or both. The docs describe periodic offline-to-online materialization as a way to reduce training/serving skew [32].

### 3.2 Chronon: measuring online/offline consistency with logs

Even with a single feature definition, online and offline values diverge. Chronon names streaming latency and delayed batch uploads around midnight as causes, and says perfect consistency is impossible [30]. Its design for *measuring* the gap is the most detailed public logging design for features:

```
online fetch --> log {key bytes, value bytes, ts, caller}  (sampled by KEY HASH)
                       | partitioned by date + join
                       v
Spark job: filter logs -> replace Join's left source with logged keys/ts
                       -> recompute offline join for same period
                       v
compare logged (a) vs backfilled (b) per feature:
   equality: mismatch %, missing %, extra %
   numeric : symmetric % diff, delta distribution
   sequence: length dist, Levenshtein (ins/del separately)
                       v
Hive table per join (date, time, group-by) with ReqSketch quantiles
```

Key techniques [30]:
- **Sample by key hash, not by request.** All requests for a sampled key are kept. This keeps per-key histories complete for aggregation features and keeps the bloom filter used in the join small.
- **Replay the log as the training spine.** Replacing the join's left side with the logged keys and timestamps makes the backfill an exact counterfactual of what serving should have returned.
- **Type-specific diff metrics, stored as sketches.** ReqSketch was chosen for accuracy and speed, and a few binned values per feature keep the output small.

The same logs also serve as **log-and-wait training data** for features that cannot be backfilled, which is why logging and backfill are complementary rather than alternatives.

### 3.3 Logging at the serving layer

Model servers and managed platforms provide payload logging as a built-in sensor:

- **KServe** has an inference logger that captures requests and responses for "monitoring performance, debugging, auditing predictions, and collecting data for model improvement". It can send them to a message-dumper service, through Knative Eventing, or to blob storage, with optional TLS and request-header capture [35].
- **SageMaker Model Monitor** enables data capture on an endpoint with a `sampling_percentage` (100 in the reference notebook). Captured records are written as JSON Lines to S3, organized by endpoint, variant and hour [36].
- **Feast** can log the features actually served by feature services. Its data quality monitoring can then compute metrics on them with `feast monitor run --source-type log`, separately from metrics on batch data [33]. The Python feature server also has opt-in audit logs (request metadata such as entity count, feature views, status and latency) and a Prometheus "seconds since last materialization" freshness gauge [34].

### 3.4 Label joining and delayed feedback

Labels usually arrive after the prediction: a click within minutes, a return within weeks, a chargeback within months. The label join must therefore choose an attribution window, and the choice biases the data. Chapter 5 covers the streaming case. Twitter's continuous-training work ingests every impression immediately as a negative and adds a positive copy if an engagement arrives later, then corrects the resulting "fake negatives" in the loss [53]. For batch CT, the practical rules are:

- version the label-join window as part of the training data span;
- train only on spans whose window has closed;
- monitor the *label* delay distribution itself, because a change in label latency looks like a change in base rate.

## 4. Data quality and drift monitoring

### 4.1 TFDV (Google): schema-as-contract

**Problem.** The SysML 2019 paper "Data Validation for Machine Learning" by Breck, Polyzotis, Roy, Whang and Zinkevich describes Google's production data validation [15]. TensorFlow Data Validation (TFDV) is the open-source implementation. It computes statistics, infers schemas and detects anomalies, and is "designed to be highly scalable" [14].

**Techniques** [13]:
- **Schema as a contract.** The schema "codifies properties which the input data is expected to satisfy, such as data types or categorical values". `infer_schema()` is "best-effort", so the schema is meant to be reviewed and then version-controlled. Anomalies are violations of the schema by new statistics.
- **Schema environments.** A label can be required in `TRAINING` and absent in `SERVING`, so the same schema validates both training data and logged serving data.
- **Training/serving skew.** Distribution skew is when the "distribution of feature values for training data is significantly different from serving data". Typical causes are a different corpus or a faulty sampling method, such as downsampling without upweighting.
- **Drift between consecutive spans.** Span N is compared with span N+1. Categorical features use **L-infinity distance** and numeric features use an approximate **Jensen-Shannon divergence**. Thresholds are user-set, and the guide says choosing them is "an iterative process requiring domain knowledge and experimentation".

TFDV runs on every ExampleGen split and on both pre-transform and post-transform data [13]. A drift anomaly therefore blocks the CT pipeline *before* training, which is the cheapest place to stop a bad loop iteration.

### 4.2 Deequ (Amazon): unit tests for data on Spark

Deequ is a Spark library for "unit tests for data" [37]. A `VerificationSuite` holds `Check`s such as completeness, uniqueness, allowed values, non-negativity and approximate quantiles. Deequ can also suggest constraints automatically. It stores computed metrics in a **MetricsRepository** and runs anomaly detection on the time series of those metrics. The project describes its target as datasets of "billions of rows" [37], and the design is published in the VLDB 2018 paper by Schelter et al. [38]. The metric-history design matters for closed loops. A completeness of 0.97 may be fine in isolation and still be a sharp drop from yesterday's 0.999, and only a stored history can tell the two apart.

### 4.3 Profiles and expectations: whylogs and Great Expectations

- **whylogs** builds compact statistical **profiles** (distributions, missing values, custom metrics) that are **mergeable**, so profiles from distributed workers or time slices can be combined at any granularity. It supports constraints for data unit tests and drift reports against a reference profile [40]. The README puts overhead at usually under 1%, and about 5% for very large workloads (thousands of features, 10M+ QPS). Its benchmarks report about 2.6 minutes on 2 c5a.2xlarge instances for a 10 GB, ~34M-row table (estimated at ~$2.45 per TB), and 2.7 minutes on 16 instances for 100 GB, ~290M rows (~$0.221) [40]. These are vendor estimates at on-demand prices. The structural point is that cost scales with the number of features, not rows, which makes always-on logging affordable.
- **Great Expectations (GX Core)** expresses checks as "Expectations", described as "expressive and extensible unit tests for your data", and auto-generates documentation of validation results [39].

### 4.4 Statistical drift detectors and label-free performance estimation

Three open-source libraries cover most of the statistical toolkit:

- **Alibi Detect** (Seldon) offers feature-level tests (Kolmogorov-Smirnov, Chi-squared, Cramér-von Mises, Fisher's exact test), multivariate kernel tests (MMD, learned-kernel MMD, context-aware MMD, LSDD), classifier-based drift and spot-the-diff, and model-uncertainty drift. Several have **online** variants: MMD, LSDD, Cramér-von Mises and Fisher's exact test. Backends are TensorFlow, PyTorch and KeOps, for example `MMDDrift(x_ref, backend='pytorch', p_val=.05)` [41].
- **NannyML** targets the gap between drift and impact. It estimates performance *without ground truth*, using confidence-based performance estimation (CBPE) for classification and direct loss estimation (DLE) for regression. Its multivariate drift method tracks **PCA reconstruction error** against a threshold set on a reference period. Univariate tests include Jensen-Shannon and L-infinity distance, with multiple-testing correction. The README claims that estimating performance filters out drift alerts that do not affect expected performance, which reduces alert fatigue [42].
- **Evidently** packages 100+ metrics into Reports, and pass/fail **Test Suites** whose conditions can be auto-generated from a reference dataset for CI/CD. It offers "20+ statistical tests and distance metrics" for drift (for example, `DataDriftPreset(method="psi")`), a self-hostable monitoring dashboard, and LLM evaluation through row-level descriptors and LLM-as-judge [43].

### 4.5 Managed monitoring: SageMaker Model Monitor

SageMaker's reference workflow shows the managed pattern end to end [36]:

1. enable data capture on the endpoint;
2. run `suggest_baseline` on the training data, a processing job (for example, ml.m5.xlarge with a 3,600 s limit) that produces baseline statistics and suggested constraints;
3. create a monitoring schedule (for example, `CronExpressionGenerator.hourly()`), noting that runs may start up to ~20 minutes after the hour because of a buffer period;
4. read execution states, including `CompletedWithViolations`, and the violations report;
5. optionally emit CloudWatch metrics for alarms.

The design is "baseline once, compare every hour". It catches sudden breakage well. It is weak at slow drift unless the baseline is refreshed together with the model, so the baseline must be versioned *with* the model in the registry.

### 4.6 Comparison of data and model monitors

| Tool | Where it runs | What it checks | Drift method(s) | Label-free quality? | Closed-loop role |
|---|---|---|---|---|---|
| TFDV [13][14] | In-pipeline (Beam) | Schema, anomalies, skew, span-to-span drift | L-infinity (categorical), approx. JS (numeric) | No | Pre-training gate in TFX |
| Deequ [37][38] | Spark batch | Constraint checks, suggested constraints | Anomaly detection on metric history | No | Upstream table gate |
| Great Expectations [39] | Python, many sources | Expectations, data docs | Via expectations | No | Pipeline gate, docs |
| whylogs [40] | Pipeline or sidecar | Mergeable profiles, constraints | Profile-vs-reference reports | No | Cheap always-on logging |
| Alibi Detect [41] | Python (TF, PyTorch, KeOps) | Drift, outliers | KS, Chi2, MMD, LSDD, classifier, context-aware | No | Detector service behind serving |
| NannyML [42] | Python batch | Estimated performance, drift | PCA reconstruction, JS, L-infinity | Yes (CBPE, DLE) | Prioritizes alerts by impact |
| Evidently [43] | Python and UI | Reports, tests, dashboards, LLM evals | 20+ tests (e.g. PSI) | No | CI tests plus monitoring |
| SageMaker Model Monitor [36] | Managed, scheduled | Captured data vs baseline constraints | Baseline statistics vs constraints | No | Hourly violations to alarms |
| Feast DQM [33] | Offline store | Distributions, nulls, percentiles on batch or logged features | Baseline comparison | No | Monitors served features |

## 5. Model registries, metadata and lineage

### 5.1 ML Metadata (MLMD): lineage as a graph

MLMD records ML workflow metadata as a typed graph [12]:

- **ArtifactType/Artifact** (datasets, models, statistics, blessings);
- **ExecutionType/Execution** (a component run, with its runtime parameters);
- **Event** (an execution consumed or produced an artifact);
- **ContextType/Context** (a project, pipeline run or experiment), with **Attribution** linking an artifact to a context and **Association** linking an execution to one.

The guide's use cases are the queries a closed loop needs [12]:
- list all models;
- trace how an artifact was produced, for auditing;
- **find every model derived from a given dataset**, which is exactly what you need when a data bug is discovered after the fact;
- check whether an execution already ran on the same inputs, so its output can be reused.

Backends are in-memory SQLite, SQLite, MySQL and PostgreSQL, optionally behind a gRPC server [12].

### 5.2 MLflow Model Registry: from stages to aliases

MLflow's registry is the most widely used open registry, and its recent design change is instructive. A **registered model** has a unique name, numbered **versions**, **tags** and **aliases**. Each version links to the run, logged model or notebook that produced it [44]. An **alias** is "a mutable, named reference to a particular version", resolved through URIs such as `models:/MyModel@champion` [44].

Model **stages** (Staging/Production) "are deprecated and will be removed in a future major release". They were deprecated in MLflow 2.9.0 [45]. The migration guidance [45]:
- replace stage labels with version **tags** such as `validation_status`;
- replace stage references with aliases (Production becomes `champion`);
- keep separate registered models per environment (for example, `dev`, `staging` and `prod` prefixes), with access control;
- promote by `copy_model_version` into the higher environment's model.

One version can carry several aliases, which the docs suggest for A/B tests and staged rollouts [45].

Why this matters for closed loops:
- **Promotion and rollback become pointer moves.** Serving code loads `@champion`, so promotion is moving the alias forward and rollback is moving it back, with no code change [45]. The previous champion must stay registered and loadable.
- **Environment separation is an access-control boundary.** An automated CT pipeline can be allowed to write to `dev`/`staging` but not to `prod`, which puts a human or a policy check at exactly one gate.
- **Tags carry gate results.** Validation status, evaluation metrics and the drift baseline version can be stored on the version, which makes the registry the single place a rollout controller reads.

### 5.3 Other registry patterns

- **TFX** uses no separate registry. The blessing artifacts in MLMD play that role, and the Resolver's "latest blessed model" query is the champion pointer [8][12].
- **Metaflow** uses namespaces and production tokens. The registry is the set of artifacts produced by the `prod` namespace, accessible through the Client API, and producing them requires the production token [19][25].

## 6. Automated retraining triggers

### 6.1 Trigger taxonomy

| Trigger | Mechanism examples | Strength | Main risk |
|---|---|---|---|
| Schedule (cron) | Model Monitor hourly schedule [36]; Metaflow `@schedule` [21]; Flyte schedules [28] | Predictable cost; simple | Retrains when nothing changed; misses sudden shifts |
| New data arrival | TFX latest-span pickup [7]; Airflow asset events [27]; Metaflow `@trigger` [22] | Freshness follows data | Partial or bad data triggers runs; event storms |
| Upstream completion | Metaflow `@trigger_on_finish` [23]; Airflow consumer DAGs [27] | Composable pipelines | Cycles ("infinite loop" warning [23]) |
| Drift alert | TFDV drift anomaly [13]; Alibi Detect [41]; Evidently tests [43] | Reacts to distribution change | Drift without impact causes needless retrains |
| Performance degradation | Realized metrics on joined labels; NannyML estimated metrics [42] | Retrains when it matters | Label delay; estimators can be wrong under concept shift |
| Human or policy | Production token for prod deploy [25]; MLflow environment copy [45] | Accountability at the gate | Bottleneck if every loop iteration needs approval |

### 6.2 Designing the trigger controller

The published designs suggest a controller with the following properties. Each is grounded in a documented failure or feature.

```
 signals: [new span]  [drift score]  [est. perf]  [realized perf]  [cron]
              \            |              |              |            /
               v           v              v              v           v
         +-------------------------------------------------------------+
         | TRIGGER POLICY                                              |
         |  - debounce / min interval between runs                     |
         |  - require data-validation PASS before training             |
         |  - weight drift by estimated impact (NannyML-style)         |
         |  - dedupe events (multi-event window, Metaflow [22])        |
         |  - acyclic trigger graph; cap concurrent runs               |
         +----------------------------+--------------------------------+
                                      v
                         CT pipeline run (data version pinned)
                                      v
                 gates: data -> model vs baseline -> infra -> registry
```

- **Debounce and windowing.** Metaflow's multi-event triggers require all events within a time window [22]. The same idea, applied to one event type, gives a minimum interval between retrains.
- **Validate before training, not after.** TFDV blocks a span before compute is spent and before a bad model can be blessed [13].
- **Prefer impact-weighted signals.** Raw drift p-values fire constantly on high-dimensional data. NannyML's argument is that estimated performance is the better trigger, with drift used for diagnosis [42].
- **Make triggers authenticated.** Under Airflow's "implicit trust" model, anyone who can create asset events can trigger retraining [27].
- **Pin the data version in the run's parameters.** Otherwise KFP-style caching can reuse stale outputs [18], and TFX-style "latest span" semantics can make re-runs non-reproducible [7].

## 7. Canarying, shadowing and rollback

### 7.1 Pre-production: servability and shadow traffic

Before any user traffic, two checks catch most mechanical failures:

- **Servability.** TFX's InfraValidator loads the candidate into a production-matched sandbox and optionally queries it [9].
- **Shadow or mirror deployment.** In a Seldon Core 2 **Experiment**, the `mirror` model receives a copy of traffic whose responses are not returned to callers [51]. This tests latency, errors and prediction distribution on live inputs without user impact. Logged mirror predictions can also be compared offline with the incumbent's predictions on the same requests.

### 7.2 Traffic-split canaries

- **KServe** sets `canaryTrafficPercent` on an InferenceService. KServe tracks the last revision rolled out to 100% as the known-good baseline. Promotion is removing the field. Rollback is setting it to 0, which returns traffic to the previous revision. A failed step pins traffic to `PreviousRolledoutRevision` [46][47]. Tag routing (`serving.kserve.io/enable-tag-routing`) exposes `latest-` and `prev-` URLs so tests can target each revision directly. The canary strategy is supported only in serverless deployment mode [46][47].
- **Seldon Core 2** splits traffic between candidates by weight (for example, 50/50) behind an existing model endpoint, with optional sticky routing through a returned `x-seldon-route` header [51].

### 7.3 Metric-gated progressive delivery

General-purpose Kubernetes delivery controllers add automated analysis and automatic abort:

- **Argo Rollouts.** A canary is a list of steps (`setWeight`, `pause`, analysis). An **AnalysisTemplate** defines metrics, intervals and `successCondition`/`failureCondition`. An **AnalysisRun** ends Successful, Failed or Inconclusive, which continues, aborts or pauses the rollout. The reference example queries Prometheus every 5 minutes for the non-5xx ratio with `successCondition: result[0] >= 0.95` and `failureLimit: 3` [49]. On failure the controller aborts and sets canary weight to zero. By default the stable ReplicaSet stays fully scaled so traffic can move back immediately. `dynamicStableScale` saves cost by shrinking stable as canary grows, at the price of slower rollback [48]. **Experiments** can run a baseline and a canary side by side for a fixed time and compare them [49]. This baseline-versus-canary comparison is statistically better than comparing a canary with the whole stable fleet.
- **Flagger** supports canary, A/B, blue/green and traffic-mirroring strategies across Istio, Linkerd, Kuma, Knative and several ingress controllers. Built-in Prometheus checks cover request success rate and P99 duration, and custom metric templates and pre-rollout or rollout webhooks (for example, load and conformance tests) can be added. Rollback is automatic when failed checks exceed a threshold or progress stalls past the deadline (default 600 s) [50].

```
 weight:  0% --10%--> [analysis] --25%--> [analysis] --50%--> ... --100%
                        |   success: continue   failure x failureLimit
                        |                         |
                        +--------> ABORT: weight=0, stable still scaled
                                   registry alias stays on previous version
```

### 7.4 The gap: infrastructure metrics versus model quality

All of these controllers gate on what they can measure within minutes: error rates, latency and custom Prometheus metrics [48][49][50]. A model can be fast, error-free and wrong. Closing that gap takes three things:

1. **Export model-level online metrics to the metrics system**, so they can be AnalysisTemplate metrics. Examples are prediction mean and calibration against a holdout, drift scores from an Alibi Detect service, and NannyML-style estimated performance [41][42].
2. **Hold the canary long enough for the fastest reliable label**, or gate on a proxy that has been validated against the slow label.
3. **Keep the rollback target warm.** Argo Rollouts' default keeps stable at full scale [48], KServe keeps the previous revision [46], and MLflow keeps the old version behind an alias that can be moved back [45].

A cautionary case from Chapters 1, 3 and 11 shows the stakes. OpenAI's April 2025 GPT-4o update added a reward signal based on thumbs-up/down feedback. Offline evaluations and A/B tests did not catch the resulting sycophancy, and a full rollback took about 24 hours [55]. For loops that retrain on user feedback, behavioral evaluations must be launch-blocking gates in the same pipeline, not advisory dashboards.

For online-learning systems that push parameter deltas every few minutes, version-level rollback is too coarse. Chapter 5 describes Ekko's update-level rollback as the corresponding primitive [52].

## 8. Cross-system comparisons

### 8.1 Pipeline and orchestration layers

| System | Origin | Unit of work | Trigger primitives | Validation built in | Lineage and registry | Notable numbers |
|---|---|---|---|---|---|---|
| TFX [5]-[12] | Google | Typed components over artifacts | Re-run picks up latest span/version [7] | Data (TFDV), model (Evaluator), infra (InfraValidator) | MLMD; blessing artifacts | n/a |
| Kubeflow Pipelines [16]-[18] | Kubeflow / Google | Container per component; IR YAML | Runs via UI, SDK or CLI; recurring runs in UI/API | None built in (components) | Run, experiment and artifact tracking | n/a |
| Metaflow [19]-[25] | Netflix | Python steps in flows | `@schedule`, `@trigger`, `@trigger_on_finish` | User code | Client API; `@project` namespaces | 3,000+ projects; hundreds of millions of jobs [19] |
| Maestro [26] | Netflix | Workflows as a service | Scheduler (signals not documented in README) | n/a | n/a | Hundreds of thousands of workflows; millions of jobs a day [26] |
| Airflow assets [27] | Apache | Tasks in DAGs | Asset events, partitions | None built in | Asset event history | n/a |
| Flyte [28] | Open source (Kubernetes) | Typed tasks | Schedules | None built in | Immutable executions | n/a |

### 8.2 Rollout controllers

| Controller | Traffic mechanism | Automated analysis | Shadow/mirror | Rollback semantics |
|---|---|---|---|---|
| KServe canary [46][47] | `canaryTrafficPercent` on InferenceService (serverless mode) | Not built in | n/a | Set to 0, or failure pins to previous rolled-out revision |
| Seldon Core 2 Experiments [51] | Weighted candidates, default endpoint | Not built in | `mirror` model | Remove experiment |
| Argo Rollouts [48][49] | `setWeight` steps, mesh or ingress routing | AnalysisTemplate/Run, Experiments | Via traffic routing integrations | Abort sets weight 0; stable kept scaled by default |
| Flagger [50] | Mesh or ingress weights, A/B headers | Prometheus success rate, P99, custom metrics, webhooks | Blue/green mirroring | Automatic on threshold or deadline (600 s default) |
| MLflow aliases [44][45] | n/a (serving loads `@alias`) | n/a | n/a | Move alias back to previous version |

## 9. Failure modes and mitigations

| Failure mode | Where it bites | Symptom | Mitigation from the sources |
|---|---|---|---|
| Training/serving skew | Feature computation | Offline wins, online losses | Single feature definition [29][31]; Transform graph in the SavedModel [6]; log-and-train (Rule #29) [4]; measure with logged-vs-backfill diffs [30] |
| Stale data through caching | KFP, Flyte caching | Retrain produces an identical model | Put data version in parameters; disable caching on ingestion [18] |
| Trigger storms and cycles | Event-driven CT | Runaway cluster use | Acyclic trigger graph; multi-event windows; deletion procedure [22][23] |
| Untrusted triggers | Airflow assets | Unauthorized retrains | Restrict asset-event creation [27] |
| Slow regression through absolute gates | Evaluator thresholds | Quality ratchets down | Change thresholds vs blessed baseline; per-slice thresholds [8] |
| Sandbox mismatch | Infra validation | Model loads in CI, fails in prod | Match binary, resources and config [9] |
| Alert fatigue from drift | Monitoring | Ignored alerts | Impact-weighted alerts; multiple-testing correction [42] |
| Stale baselines | Scheduled monitors | Permanent violations or none | Version baselines with the model [36] |
| Infra-only canary gates | Progressive delivery | Fast, error-free, wrong model ships | Model-level metrics in the analysis; baseline-vs-canary Experiments [49] |
| Rollback target gone | Registry or serving | Rollback slower than incident | Keep stable scaled [48]; previous revision [46]; alias history [45] |
| Feedback-signal reward hacking | User-feedback loops | Behavioral regression | Launch-blocking behavior evals; fast rollback [55] |

## 10. Further company case studies

The following primary sources describe in-house platforms that implement parts of this architecture. They are listed for completeness, with titles and URLs as catalogued in the community "applied-ml" index [56]. They are not analyzed in detail in this chapter.

| Organization | Source | Topic |
|---|---|---|
| Uber | "Meet Michelangelo: Uber's Machine Learning Platform" (2017) [57] | End-to-end platform |
| Uber | "Monitoring Data Quality at Scale with Statistical Modeling" (2017) [58] | Data quality |
| Netflix | "Runway: Model Lifecycle Management at Netflix" (OpML 2020) [59] | Model lifecycle |
| Intuit | "Managing ML Models @ Scale" (OpML 2020) [60] | Model management |
| Apple | "Overton: A Data System for Monitoring and Improving Machine-Learned Products" (2019) [61] | Monitoring and improvement |
| Meta | "Inside Meta's AI optimization platform for engineers" (Looper, 2022) [62] | Product-decision ML platform |
| Netflix | "Evolution of ML Fact Store" (2022) [63] | Feature and fact logging |
| Nubank | "ML Model Monitoring: 9 Tips From the Trenches" (2021) [64] | Monitoring practice |
| Nubank | "Dealing with Train-serve Skew in Real-time ML Models" (2023) [65] | Skew |
| Google | "Data Management Challenges in Production Machine Learning" (2017) [66] | Data management |
| Shopify | "The Magic of Merlin: Shopify's New Machine Learning Platform" (2022) [67] | Platform |

## Key takeaways for system designers

1. **Treat validation outputs as artifacts, not logs.** TFX's blessing and infra-blessing artifacts make "is this model allowed to ship?" a queryable fact. They also make the next run's baseline well defined [8][9][12].
2. **Gate on change relative to a baseline, per slice.** An absolute floor alone lets a loop ratchet quality down. Pair value thresholds with change thresholds against the latest blessed model on every slice [8].
3. **Validate data before training, and validate serving data with the same schema.** Schema environments let one contract cover training and serving, while L-infinity and JS drift checks between spans stop bad spans before compute is spent [13].
4. **Log what serving actually used, sampled by entity key.** Logged features are both the ground truth for skew measurement and a source of log-and-wait training data. Key-hash sampling keeps per-entity histories intact [4][30].
5. **Pin data versions in every pipeline run.** Content-addressed caching and "latest span" semantics are both convenient and dangerous. A run should be reproducible from its parameters alone [7][18].
6. **Use event triggers, but engineer the event graph.** Data-driven retraining beats calendars, but chained triggers can loop, and unauthenticated events can retrain production models. Debounce, deduplicate, keep the graph acyclic and restrict who can emit events [22][23][27].
7. **Prefer impact-weighted retraining signals.** Drift is a diagnosis, not a decision. Estimated or realized performance should decide when to retrain, and drift should explain why [42].
8. **Make the registry a pointer system.** Aliases (`@champion`), per-environment registered models and tags for gate results turn promotion and rollback into auditable pointer moves, with access control at the prod boundary [44][45].
9. **Keep rollback cheaper than the incident.** Keep the stable version scaled, keep the previous revision addressable, and rehearse alias rollback. A loop that cannot be reverted within its own cadence is open-loop in practice [45][46][48].
10. **Put model-quality signals into the canary analysis.** Rollout controllers gate on HTTP success and latency by default [49][50]. Export prediction-distribution, drift and estimated-performance metrics so the same analysis can abort a wrong-but-healthy model. For feedback-trained models, make behavioral evals launch-blocking [55].
11. **Isolate challengers completely.** Netflix's `@project` branches run a challenger as a full parallel pipeline in its own namespace. This prevents experiments from writing to production state [24][25].
12. **Version monitoring baselines with the model.** A baseline from an old training set makes every hourly check fail or pass for the wrong reason [36].

## References

1. Google Cloud. "MLOps: Continuous delivery and automation pipelines in machine learning." Google Cloud Architecture Center. https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning
2. D. Sculley, G. Holt, D. Golovin, E. Davydov, T. Phillips, D. Ebner, V. Chaudhary, M. Young, J.-F. Crespo, D. Dennison. "Hidden Technical Debt in Machine Learning Systems." Google, NeurIPS 2015. https://proceedings.neurips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf
3. E. Breck, S. Cai, E. Nielsen, M. Salib, D. Sculley. "The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction." Google, IEEE Big Data 2017. https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/
4. M. Zinkevich. "Rules of Machine Learning: Best Practices for ML Engineering." Google. https://developers.google.com/machine-learning/guides/rules-of-ml
5. TensorFlow. "TFX: an end-to-end platform for deploying production ML pipelines" (repository README). Google, GitHub. https://github.com/tensorflow/tfx
6. TensorFlow. "The TFX User Guide." Google, GitHub. https://github.com/tensorflow/tfx/blob/master/docs/guide/index.md
7. TensorFlow. "The ExampleGen TFX Pipeline Component" (span, version, split). Google, GitHub. https://github.com/tensorflow/tfx/blob/master/docs/guide/examplegen.md
8. TensorFlow. "The Evaluator TFX Pipeline Component." Google, GitHub. https://github.com/tensorflow/tfx/blob/master/docs/guide/evaluator.md
9. TensorFlow. "The InfraValidator TFX Pipeline Component." Google, GitHub. https://github.com/tensorflow/tfx/blob/master/docs/guide/infra_validator.md
10. TensorFlow. "The Pusher TFX Pipeline Component." Google, GitHub. https://github.com/tensorflow/tfx/blob/master/docs/guide/pusher.md
11. TensorFlow. "The Trainer TFX Pipeline Component." Google, GitHub. https://github.com/tensorflow/tfx/blob/master/docs/guide/trainer.md
12. TensorFlow. "ML Metadata" guide. Google, GitHub. https://github.com/tensorflow/tfx/blob/master/docs/guide/mlmd.md
13. TensorFlow. "TensorFlow Data Validation" guide (schema, environments, skew, drift). Google, GitHub. https://github.com/tensorflow/tfx/blob/master/docs/guide/tfdv.md
14. TensorFlow. "TensorFlow Data Validation" (repository README). Google, GitHub. https://github.com/tensorflow/data-validation
15. E. Breck, N. Polyzotis, S. Roy, S. E. Whang, M. Zinkevich. "Data Validation for Machine Learning." Google, SysML 2019. https://mlsys.org/Conferences/2019/doc/2019/167.pdf
16. Kubeflow. "Kubeflow Pipelines" (repository README). GitHub. https://github.com/kubeflow/pipelines
17. Kubeflow. "Kubeflow Pipelines overview" (website source). GitHub. https://github.com/kubeflow/website/blob/master/content/en/docs/components/pipelines/overview.md
18. Kubeflow. "Use Caching" (KFP user guide, website source). GitHub. https://github.com/kubeflow/website/blob/master/content/en/docs/components/pipelines/user-guides/core-functions/caching.md
19. Netflix and Outerbounds. "Metaflow" (repository README). GitHub. https://github.com/Netflix/metaflow
20. Metaflow. "Production deployments: introduction." Netflix, GitHub (metaflow-docs). https://github.com/Netflix/metaflow-docs/blob/master/docs/production/introduction.md
21. Metaflow. "Event triggering." Netflix, GitHub (metaflow-docs). https://github.com/Netflix/metaflow-docs/blob/master/docs/production/event-triggering/README.md
22. Metaflow. "Triggering flows based on external events." Netflix, GitHub (metaflow-docs). https://github.com/Netflix/metaflow-docs/blob/master/docs/production/event-triggering/external-events.md
23. Metaflow. "Triggering flows based on other flows." Netflix, GitHub (metaflow-docs). https://github.com/Netflix/metaflow-docs/blob/master/docs/production/event-triggering/flow-events.md
24. Metaflow. "Deploying variants of event-triggered flows." Netflix, GitHub (metaflow-docs). https://github.com/Netflix/metaflow-docs/blob/master/docs/production/event-triggering/project-events.md
25. Metaflow. "Coordinating larger Metaflow projects" (@project). Netflix, GitHub (metaflow-docs). https://github.com/Netflix/metaflow-docs/blob/master/docs/production/coordinating-larger-metaflow-projects.md
26. Netflix. "Maestro: workflow orchestrator" (repository README). GitHub. https://github.com/Netflix/maestro
27. Apache Airflow. "Asset definitions / data-aware scheduling" (docs source). GitHub. https://github.com/apache/airflow/blob/main/airflow-core/docs/authoring-and-scheduling/assets.rst
28. Flyte. "Flyte" (repository README). GitHub. https://github.com/flyteorg/flyte
29. Airbnb. "Chronon" (repository README: log-and-wait vs replicated approaches, backfills, consistency). GitHub. https://github.com/airbnb/chronon
30. Airbnb. "Online/Offline Consistency" (Chronon docs). GitHub. https://github.com/airbnb/chronon/blob/main/docs/source/test_deploy_serve/Online_Offline_Consistency.md
31. Feast. "Feast: open source feature store for machine learning" (repository README). GitHub. https://github.com/feast-dev/feast
32. Feast. "Push source" (reference docs). GitHub. https://github.com/feast-dev/feast/blob/master/docs/reference/data-sources/push.md
33. Feast. "Data quality monitoring" (reference docs). GitHub. https://github.com/feast-dev/feast/blob/master/docs/reference/dqm.md
34. Feast. "Python feature server" (reference docs). GitHub. https://github.com/feast-dev/feast/blob/master/docs/reference/feature-servers/python-feature-server.md
35. KServe. "Inference logger" (website docs). GitHub. https://github.com/kserve/website/blob/main/docs/model-serving/predictive-inference/logger/index.md
36. AWS. "Amazon SageMaker Model Monitor" introduction notebook. amazon-sagemaker-examples, GitHub. https://github.com/aws/amazon-sagemaker-examples/blob/main/sagemaker_model_monitor/introduction/SageMaker-ModelMonitoring.ipynb
37. AWS Labs. "Deequ: unit tests for data" (repository README). GitHub. https://github.com/awslabs/deequ
38. S. Schelter, D. Lange, P. Schmidt, M. Celikel, F. Biessmann, A. Grafberger. "Automating Large-Scale Data Quality Verification." Amazon, PVLDB 11(12), 2018. http://www.vldb.org/pvldb/vol11/p1781-schelter.pdf
39. Great Expectations. "GX Core" (repository README). GitHub. https://github.com/great-expectations/great_expectations
40. WhyLabs. "whylogs" (Python README: profiles, overhead and benchmarks). GitHub. https://github.com/whylabs/whylogs/blob/mainline/python/README.md
41. Seldon. "Alibi Detect" (repository README). GitHub. https://github.com/SeldonIO/alibi-detect
42. NannyML. "NannyML" (repository README). GitHub. https://github.com/NannyML/nannyml
43. Evidently AI. "Evidently" (repository README). GitHub. https://github.com/evidentlyai/evidently
44. MLflow. "MLflow Model Registry" (docs source). GitHub. https://github.com/mlflow/mlflow/blob/master/docs/docs/classic-ml/model-registry/index.mdx
45. MLflow. "Model Registry workflows" (aliases, environment copies, migration from stages). GitHub. https://github.com/mlflow/mlflow/blob/master/docs/docs/classic-ml/model-registry/workflow.mdx
46. KServe. "Canary rollout strategy" (website docs). GitHub. https://github.com/kserve/website/blob/main/docs/model-serving/predictive-inference/rollout-strategies/canary.md
47. KServe. "Canary rollout example" (website docs). GitHub. https://github.com/kserve/website/blob/main/docs/model-serving/predictive-inference/rollout-strategies/canary-example.md
48. Argo Project. "Argo Rollouts: Canary deployment strategy." GitHub. https://github.com/argoproj/argo-rollouts/blob/master/docs/features/canary/index.md
49. Argo Project. "Argo Rollouts: Analysis and progressive delivery." GitHub. https://github.com/argoproj/argo-rollouts/blob/master/docs/features/analysis.md
50. Flux project. "Flagger: progressive delivery Kubernetes operator" (repository README). GitHub. https://github.com/fluxcd/flagger
51. Seldon. "Seldon Core 2: Experiments" (docs). GitHub. https://github.com/SeldonIO/seldon-core/blob/v2/docs-gb/kubernetes/resources/experiment.md
52. C. Sima, Y. Fu, M.-K. Sit, et al. "Ekko: A Large-Scale Deep Learning Recommender System with Low-Latency Model Update." University of Edinburgh and Tencent, OSDI 2022. https://www.usenix.org/conference/osdi22/presentation/sima
53. S. I. Ktena, A. Tejani, L. Theis, et al. "Addressing Delayed Feedback for Continuous Training with Neural Networks in CTR prediction." Twitter, RecSys 2019. https://arxiv.org/abs/1907.06558
54. C. Huyen. "Real-time machine learning: challenges and solutions." 2022. https://huyenchip.com/2022/01/02/real-time-machine-learning-challenges-and-solutions.html
55. OpenAI. "Expanding on what we missed with sycophancy." May 2025. https://openai.com/index/expanding-on-sycophancy/
56. E. Yan et al. "applied-ml: Papers and tech blogs by companies sharing their work on data science and machine learning in production." GitHub. https://github.com/eugeneyan/applied-ml
57. J. Hermann, M. Del Balso. "Meet Michelangelo: Uber's Machine Learning Platform." Uber Engineering, 2017. https://eng.uber.com/michelangelo-machine-learning-platform/
58. Uber Engineering. "Monitoring Data Quality at Scale with Statistical Modeling." 2017. https://eng.uber.com/monitoring-data-quality-at-scale/
59. Netflix. "Runway: Model Lifecycle Management at Netflix." USENIX OpML 2020. https://www.usenix.org/conference/opml20/presentation/cepoi
60. Intuit. "Managing ML Models @ Scale: Intuit's ML Platform." USENIX OpML 2020. https://www.usenix.org/conference/opml20/presentation/wenzel
61. C. Ré et al. "Overton: A Data System for Monitoring and Improving Machine-Learned Products." Apple, 2019. https://arxiv.org/abs/1909.05372
62. Meta AI. "Inside Meta's AI optimization platform for engineers" (Looper). 2022. https://ai.facebook.com/blog/looper-meta-ai-optimization-platform-for-engineers/
63. Netflix Technology Blog. "Evolution of ML Fact Store." 2022. https://netflixtechblog.com/evolution-of-ml-fact-store-5941d3231762
64. Nubank. "ML Model Monitoring: 9 Tips From the Trenches." 2021. https://building.nubank.com.br/ml-model-monitoring-9-tips-from-the-trenches/
65. Nubank. "Dealing with Train-serve Skew in Real-time ML Models: A Short Guide." 2023. https://building.nubank.com.br/dealing-with-train-serve-skew-in-real-time-ml-models-a-short-guide/
66. N. Polyzotis, S. Roy, S. E. Whang, M. Zinkevich. "Data Management Challenges in Production Machine Learning." Google, SIGMOD 2017 tutorial. https://research.google/pubs/pub46178/
67. Shopify Engineering. "The Magic of Merlin: Shopify's New Machine Learning Platform." 2022. https://shopify.engineering/merlin-shopify-machine-learning-platform
