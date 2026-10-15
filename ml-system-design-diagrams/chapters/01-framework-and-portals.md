# 01. The ML System Design Interview Framework, Master Architecture, and Portal Index

> Problem 01 does not ask you to design one product. It covers the reusable answer
> template, a generic production ML system you can adapt to any prompt, and an index of
> the prep portals with what each one teaches. Later chapters fill this template in for
> specific problems.

## Company tags

| Company | What is reported about the MLSD round | Source |
|---|---|---|
| Meta | Onsite with 1-2 ML system design rounds (E5/E6 often 2), usually ranking or recsys, e.g. news feed, Marketplace | [9] [35] [36] |
| Google | Design rounds at L5+ include content ranking and recommender scaling; reported prompts include YouTube recs, Gmail spam, YouTube copyright detection | [10] [33] |
| Amazon | Covered by IGotAnOffer as a source of real MLSD questions | [8] [11] |
| Apple | Has a dedicated MLE interview guide that includes ML design | [12] |
| OpenAI / Anthropic | Newer question banks lean toward LLM systems (e.g. "Design ChatGPT") | [13] [14] [37] |
| Google, LinkedIn, Snap, Coupang, Stitch Fix | Khang Pham reports MLE offers from these five; his question material draws on Facebook, Amazon, Apple, Google, Microsoft, Snap, LinkedIn interviews | [17] |
| LinkedIn, Airbnb, YouTube | Used as the basis for Khang Pham's case studies (LinkedIn feed ranking, Airbnb search ranking, YouTube recommendation) | [17] |

**Portals and books that cover the framework itself:** Exponent / Aced [3][4], ByteByteGo
(Aminian & Xu) [15], Educative Grokking MLSD [5] and Khang Pham's course [6][17], Hello
Interview [7], IGotAnOffer [8], alirezadir/AIMLInterviews [1][2], Chip Huyen [18][19][20][21],
Patrick Halina [23], Babushkin & Kravchenko (Manning) [22], interviewing.io / Prepfully mocks
[34][35], DesignGurus [38], Interview Kickstart [39].

## Requirements & scale (the template you fill in for every prompt)

| Dimension | What to pin down in minutes 0-8 | Typical default if the interviewer stays vague |
|---|---|---|
| Business goal | What gets better: engagement, revenue, safety, cost | Pick one primary metric and 1-2 guardrails [1] |
| Functional | Inputs, outputs, surfaces, personalization yes/no | "Return top-K items per request" (assumption) |
| ML objective | Classification / regression / ranking / retrieval / generation | Pointwise P(engage) ranker (assumption) |
| Users | DAU, requests per user per day | 100M-1B DAU for FAANG-scale prompts (assumption) |
| Catalog | Number of candidates | 1M-10B items, so retrieval must be ANN (assumption) |
| QPS | Peak online requests | 10k-100k QPS; Uber Michelangelo reported >250k pred/s [28] |
| Latency | End-to-end p99 | 100-200 ms total, model scoring 10-50 ms (assumption) |
| Freshness | How fast new items, users and features take effect | Features within minutes, retrain daily (assumption) |
| Data | Labels available? Logging exists? Privacy limits? | Implicit feedback from logs [1] |
| Constraints | Cost, on-device, interpretability, fairness, regions | Server-side GPU or CPU serving (assumption) |
| Is ML needed? | Heuristic baseline first | "Launch without ML is fine" (Rule #1) [31] |

## The canonical answer template (what every portal converges on)

**DIAGRAM 0: The 45-60 minute answer flow with time budget (synthesised from [1][3][5][7][15])**

```
  0-8 min             8-15 min            15-25 min           25-35 min
+----------------+  +----------------+  +----------------+  +----------------+
| 1 CLARIFY      |  | 2 FRAME AS ML  |  | 3 DATA &       |  | 4 MODEL        |
| goal, scope,   |->| inputs/outputs,|->| FEATURES       |->| baseline ->    |
| scale, latency,|  | objective,     |  | sources, labels|  | advanced, loss,|
| constraints    |  | metrics        |  | features       |  | training       |
+-------+--------+  +----------------+  +----------------+  +-------+--------+
        ^                                                           |
        |                                                           v
        |           45-55 min           40-45 min           35-40 min
        |           +----------------+  +----------------+  +----------------+
        |           | 7 MONITOR      |  | 6 SERVE        |  | 5 EVALUATE     |
        |           | drift, alerts, |<-| batch/online,  |<-| offline + A/B  |
        |           | retrain cadence|  | ANN, latency   |  | metrics,       |
        |           |                |  | budget, rollout|  | guardrails     |
        |           +-------+--------+  +----------------+  +----------------+
        |                   |
        +-------------------+  feedback loop: new data, failures, retraining
  55-60 min: wrap up -- summarize, name trade-offs, say what you'd do next
```

**Step-name mapping across portals**

| Generic step | ByteByteGo 7-step [15] | alirezadir 9-step [1] | Educative 6-step [5] | Chip Huyen [18] | Hello Interview [7] |
|---|---|---|---|---|---|
| Clarify | Clarifying requirements | Problem formulation | Problem scoping / clarify | Project setup | Problem framing (5-7 min) |
| Frame as ML | Frame problem as ML task | Problem formulation + Metrics | Problem formulation | Project setup | Problem framing |
| Architecture | (inside serving) | Architectural components (MVP) | System architecture | Serving | High-level design |
| Data | Data preparation | Data collection & preparation | Data strategy | Data pipeline | Data & features |
| Features | Data preparation | Feature engineering | Data strategy | Data pipeline | Data & features |
| Model | Model development | Model dev & offline eval | Model design + trade-offs | Modeling | Modeling |
| Evaluation | Evaluation | Metrics + online testing | Offline & online eval | Modeling (debug) | Evaluation |
| Serving | Deployment & serving | Prediction service + deployment | Deploy under constraints | Serving | Inference |
| Monitoring | Monitoring & infrastructure | Scaling, monitoring, updates | Monitoring & iteration | Serving (maintain) | Deep dives |

- The ByteByteGo labels come from the book's Chapter 1 framework. The other columns paraphrase the cited pages.
- alirezadir orders the steps differently: metrics come 2nd and the MVP architecture 3rd, before any data work, so the interviewer sees the whole box diagram early [1].
- Hello Interview says its steps are "guideposts, not hard rules": if the interviewer pulls you off course, follow them [7].

## DIAGRAM 1: Master end-to-end architecture (online serving path)

**DIAGRAM 1: Generic production ML system, online path (retrieval -> filter -> rank -> re-rank) [1][27][28]**

```
                       +--------------------+
  user / client ------>| API gateway /      |  request: user_id, context, query
                       | app server         |
                       +---------+----------+
                                 | (1) fetch user + context features
                                 v
+------------------+   +--------------------------------------------------------+
| ONLINE FEATURE   |-->| ML ORCHESTRATOR / PREDICTION SERVICE                   |
| STORE (KV, <5ms) |   |                                                        |
+------------------+   | (2) CANDIDATE GENERATION (parallel sources)            |
                       |     +------------+  +------------+  +------------+     |
+------------------+   |     | ANN on     |  | co-visit / |  | trending / |     |
| ANN INDEX        |-->|     | embeddings |  | graph      |  | rules      |     |
| (HNSW / IVF-PQ)  |   |     +-----+------+  +-----+------+  +-----+------+     |
+------------------+   |           +---------------+---------------+            |
                       |                           | merge + dedupe, ~1k-10k    |
+------------------+   | (3) FILTER                v                            |
| MODEL REGISTRY   |-->|     seen, blocked, policy, eligibility, geo            |
| (versioned)      |   |                           | ~500                       |
+------------------+   | (4) RANK                  v                            |
                       |     heavy model (GBDT / DNN / multi-task)              |
                       |     score = f(user, item, context, cross features)     |
                       |                           | top ~100                   |
                       | (5) RE-RANK / POLICY      v                            |
                       |     diversity, freshness, fairness, business rules,    |
                       |     ads blending, exploration (bandit)                 |
                       +---------------------------+----------------------------+
                                                   | top-K + model_version
                                                   v
+------------------+                     +---------+---------+
| FALLBACK         |---- on timeout ---->| response to UI    |
| (popular/cache)  |                     +---------+---------+
+------------------+                               | impressions, clicks,
                                                   | dwell, conversions
                                                   v
                                         +---------+---------+  stream  +-------------+
  served features logged at request ---->| EVENT LOGGING     |--------->| streaming   |
  time, so training sees the same values | (Kafka-style)     |          | features    |
                                         +---------+---------+          +------+------+
                                                   | to offline lake           |
                                                   v                           v
                                              [DIAGRAM 2]           online feature store
```

## DIAGRAM 2: Offline training & data pipeline

**DIAGRAM 2: Offline data -> features -> training -> evaluation -> registry -> deploy loop
[1][28][29][30]**

```
+--------------+   +--------------+   +--------------+   +------------------------+
| event logs   |-->| data lake /  |-->| FEATURE      |-->| OFFLINE + ONLINE       |
| impr, click, |   | warehouse    |   | PIPELINES    |   | FEATURE STORE          |
| conversions  |   | (by date)    |   | (Spark/SQL)  |   | (same definitions)     |
+--------------+   +------+-------+   +------+-------+   +-----------+------------+
                          ^                  ^                       |
+--------------+          |                  |                       | features as of
| user / item  |----------+------------------+                       | event time
| DBs (CDC)    |          |                                          v
+--------------+          v                              +-----------+------------+
                   +------+-------+       labels         | TRAINING DATA GENERATOR|
                   | LABEL        |--------------------->| point-in-time join,    |
                   | GENERATION   |                      | neg sampling, down-    |
                   | natural /    |                      | sampling, time split   |
                   | human / weak |                      +-----------+------------+
                   +--------------+                                  |
                                                                     v
                                                         +-----------+------------+
                                                         | TRAINING (distributed) |
                                                         | baseline -> DNN, HPO,  |
                                                         | warm-start daily       |
                                                         +-----------+------------+
                                                                     |
                                                                     v
+--------------+   +--------------+   +--------------+   +-----------+------------+
| PRODUCTION   |<--| CANARY /     |<--| MODEL        |<--| OFFLINE EVALUATION     |
| online A/B   |   | SHADOW       |   | REGISTRY     |   | AUC / NDCG, calib.,    |
| ramp 1->50%  |   | deployment   |   | version,     |   | slice analysis, data + |
|              |   |              |   | lineage      |   | model validation gate  |
+------+-------+   +--------------+   +--------------+   +-----------+------------+
       |                                (pass gate)                  | fail: back to
       | online metrics, drift, alerts                               | features/model
       v                                                             v
+------+-------------------------------------------------------+   (rework)
| MONITORING: feature drift, prediction drift, latency, label  |
| delay, business KPIs -> trigger retrain ---------------------+--> TRAINING
+--------------------------------------------------------------+
```

## DIAGRAM 3: Model architecture (the generic two-stage model pair)

**DIAGRAM 3: Two-tower retriever + multi-task ranker, the default answer for recsys,
feed, ads and search prompts [27]**

```
       RETRIEVAL: TWO-TOWER                         RANKING: MULTI-TASK DNN
+-------------------+ +-------------------+  +-------------------------------------+
| user feats: id,   | | item feats: id,   |  | user | item | context | cross feats |
| history, demo,    | | text/img emb,     |  +------------------+------------------+
| context           | | category, age     |                     | embeddings + dense
+---------+---------+ +---------+---------+                     v   normalization
          |                     |            +------------------+------------------+
          v                     v            | feature interaction: DCN cross /    |
+---------+---------+ +---------+---------+  | attention over user history / FM    |
| user tower (MLP)  | | item tower (MLP)  |  +------------------+------------------+
+---------+---------+ +---------+---------+                     | shared bottom/MMoE
          | u (d=64-256)        | v                 +-----------+------------+
          +------> dot(u,v) <---+                   |           |            |
                      |                             v           v            v
  loss: sampled softmax with in-batch          +--------+  +--------+  +----------+
  negatives; v precomputed into ANN index      | P(clk) |  | P(like)|  | E[dwell] |
                      |                        +---+----+  +---+----+  +----+-----+
                      v                            | BCE       | BCE        | Huber
       top ~1k candidates ---- fed to ranker       +-----------+------------+
                                                               v
                                    final score = w1*P(clk) + w2*P(like) + w3*E[dwell]
                                    (weights tuned online: the "value model")
```

| Problem family | Swap DIAGRAM 3 for | Output |
|---|---|---|
| Recsys / feed / ads / search | two-tower + multi-task ranker (above) | ranked list |
| Moderation / fraud / spam | multimodal encoder + classifier heads | P(violation) per policy |
| Visual / semantic search | image or text encoder + ANN (single tower) | top-K neighbors |
| ETA / pricing / forecasting | GBDT or DNN regressor | continuous value |
| RAG / LLM assistant | retriever + reranker + LLM generator | grounded text |

## DIAGRAM 4: The real system the template mirrors (Uber Michelangelo, 2017)

**DIAGRAM 4: Michelangelo's 6-step workflow and online/offline split, the canonical
public reference for the "master diagram" [28]**

```
  OFFLINE (HDFS, Spark)                                    ONLINE (Samza, Cassandra)
+---------------------------------------+       +--------------------------------------+
| 1 MANAGE DATA                         |       |  streaming features (Kafka->Samza)   |
|   batch feature pipelines --> Hive    |       |            v                         |
|   feature store (shared, curated) ----+-------+--> Cassandra KV (online features)    |
+-------------------+-------------------+       +------------+-------------------------+
                    v                                        |
+-------------------+-------------------+                    |
| 2 TRAIN MODELS  (XGBoost, DL, MLlib)  |                    |
| 3 EVALUATE (reports, comparisons)     |                    v
+-------------------+-------------------+       +------------+-------------------------+
                    | model artifact            | 5 MAKE PREDICTIONS                   |
                    v                           |   serving containers, network call   |
+-------------------+-------------------+       |   from microservices (e.g. ETA for   |
| 4 DEPLOY MODELS                       |------>|   UberEATS), low-latency online path |
|   offline batch / online / library    |       +------------+-------------------------+
+---------------------------------------+                    |
                                                             v
                               +-----------------------------+---------------------+
                               | 6 MONITOR PREDICTIONS: log preds, join to labels  |
                               |   later, track accuracy + distributions over time |
                               +---------------------------------------------------+
```

- Other public platforms with the same shape: Google TFX (data validation -> trainer ->
  model validation -> serving, used for Google Play) [29], Meta FBLearner Flow (workflows,
  feature store marketplace, per-model inference containers) [32].
- Sculley et al. explain why the diagram is mostly not the model: glue code, data
  dependencies, hidden feedback loops and configuration debt [30].

## Data & labels (generic checklist)

| Item | Options | Interview tip |
|---|---|---|
| Label source | Natural (click, purchase), explicit feedback, human annotation, programmatic/weak, transfer learning, active learning [1] | Say which label you use and why it matches the business goal |
| Positives / negatives | Engaged vs impressed-not-engaged; random or hard negatives for retrieval | Do not use non-impressed items as negatives in a ranker |
| Label delay | Conversions arrive late; spam reports late | Use attribution windows and delayed-feedback correction |
| Bias | Position bias, exposure bias, selection bias | Log position, use exploration traffic, IPS weighting |
| Split | Time-based train/val/test; no user leakage | Random splits leak the future |
| Imbalance | Downsample negatives, class weights, focal loss | Recalibrate after downsampling |
| Privacy | PII, consent, regional retention; human annotation exposes data [1] | Mention it as a constraint; weak labels or on-device reduce exposure [1] |

## Features

| Group | Examples | Freshness / store |
|---|---|---|
| User | demographics, locale, long-term interests, embeddings | Batch daily, online KV |
| Item | category, text/image embeddings, age, quality score | Batch on create/update |
| Context | time, device, surface, session state | Request time |
| User-item cross | past interactions with author/category, similarity | Streaming (minutes) |
| Aggregates / counters | CTR over 1h/1d/7d windows, popularity | Streaming + batch backfill |
| Representations | one-hot, hashing, learned embeddings, normalization, bucketization [1] | Shared feature store [28] |
| Training-serving skew | log features at serve time; same transform code | Rule: reuse code offline and online [31] |

## Model choices (baseline -> advanced)

| Stage | Option | Pros | Cons |
|---|---|---|---|
| 0 | Heuristic / popularity / rules | Ships in days, strong baseline | No personalization [31] |
| 1 | Logistic regression + crosses | Fast, calibrated, interpretable | Manual feature crosses |
| 2 | GBDT (XGBoost/LightGBM) | Strong on tabular data, little tuning | Weak on sparse IDs and raw text/images |
| 3 | Two-tower DNN (retrieval) | Scales to billions of items with ANN | No late user-item interaction |
| 4 | Deep ranker (DCN, DLRM, attention over history) | Learns interactions, uses embeddings | GPU cost, latency |
| 5 | Multi-task / MMoE | One model, several objectives | Tuning the task weights |
| 6 | Pretrained / LLM components | Cold-start, semantics | Cost, latency, hallucination |

## Loss functions

| Task | Loss |
|---|---|
| Binary engagement / classification | Binary cross-entropy (weighted or focal for imbalance) |
| Multi-class / policy labels | Softmax CE, or multi-label sigmoid BCE |
| Retrieval embeddings | Sampled softmax with in-batch negatives, contrastive, triplet |
| Learning to rank | Pairwise (RankNet, BPR), listwise (LambdaRank, softmax) |
| Regression (ETA, dwell) | MSE, MAE, Huber, quantile loss for intervals |
| Multi-task | Weighted sum of per-head losses |

## Offline metrics

| Task | Metrics |
|---|---|
| Classification | ROC-AUC, PR-AUC, precision/recall@threshold, log loss, calibration (ECE) |
| Retrieval | Recall@K, hit rate@K |
| Ranking | NDCG@K, MAP, MRR |
| Regression | MAE, RMSE, MAPE |
| System | p50/p99 latency, throughput, cost per 1k predictions [1] |
| Always | Slice metrics: new users, regions, cold items, protected groups |

## Online metrics & A/B testing

| Item | Generic answer |
|---|---|
| Primary metric | CTR, session time, conversion, revenue, task success [1] |
| Counter metrics | direct negative feedback: hides, reports [1] |
| Guardrail metrics | unsubscribes, latency, crash rate (assumption) |
| Rollout | shadow -> canary -> A/B with control/test split and null hypothesis [1]; ramp 1% -> 5% -> 50% -> full (assumption) |
| Alternatives | multi-armed bandits [1]; interleaving for ranking (assumption) |
| Pitfalls | novelty effect, network effects, sample-ratio mismatch, peeking |
| Decision | ship if primary is significantly up and guardrails are flat |

## Serving & latency budget (typical 200 ms p99, assumption)

| Component | Budget | Technique |
|---|---|---|
| Gateway + auth | 10 ms | |
| Feature fetch | 10-20 ms | KV store, batched multi-get, caching |
| Retrieval (ANN, several sources) | 20-30 ms | HNSW / IVF-PQ, parallel fan-out |
| Filtering | 5 ms | Bloom filters for already-seen items |
| Ranking ~500 candidates | 30-60 ms | GPU batching; compression: quantization, pruning, distillation, factorization [1] |
| Re-rank / policy | 5-10 ms | Rule engine, MMR diversity |
| Slack / network | 50 ms | Timeout falls back to cached or popular results |
| Mode choice | batch / online / hybrid / on-device [1] | Precompute for stable users, online for context; on-device when memory, compute, energy or privacy dominate [1] |

## Monitoring

| Layer | What to watch | Action |
|---|---|---|
| System | QPS, p99 latency, errors, GPU utilization | Autoscale, alert |
| Data | Missing or null features, schema changes, feature drift (PSI/KL) | Data validation gate [29] |
| Model | Prediction distribution, calibration, online AUC once labels arrive | Roll back to the previous registry version |
| Business | Primary KPI, guardrails, per-slice metrics | Experiment review |
| Retraining | Scheduled (daily/weekly) or triggered by drift | Continual training [1] |

## Failure modes

| Failure | Symptom | Mitigation |
|---|---|---|
| Training-serving skew | Offline AUC up, online flat | Log features at serve time, share transform code [31] |
| Feedback loop | Popular items get more popular | Exploration slot, debiasing [30] |
| Label leakage | Offline metrics too good | Point-in-time joins, time-based splits |
| Cold start | New users or items get no exposure | Content features, popularity priors, explore |
| Data drift / seasonality | Gradual metric decay | Drift alerts, frequent retraining |
| Stale features / pipeline outage | Sudden drop in metrics | Freshness SLAs, default values, fallback model |
| Optimizing a proxy | CTR up, satisfaction down (clickbait) | Multi-objective value model, guardrails |
| Latency blow-up | Timeouts at peak | Candidate caps, distillation, fallbacks |

## Interviewer follow-ups (reported or likely across portals)

- Why ML at all? What heuristic baseline would you launch first? [31]
- How exactly do you turn the business goal into a training label and loss? [1][15]
- Which offline metric correlates with your online metric, and what if they disagree?
- How do you build negatives, and how do you handle class imbalance?
- How do you avoid data leakage when you join labels and features?
- How do you handle position bias and the feedback loop created by your own recommendations?
- Cold start for new users and new items?
- Walk through the latency budget. What do you cut if p99 doubles?
- Batch vs online prediction: which features are precomputed and which are real-time?
- How do you design the A/B test? Which guardrails? How long do you run it? [8]
- How do you detect performance drift after deployment, and when do you retrain? [8]
- How would the design change at 10x scale, or on-device?
- The model is too big for the latency or device budget: quantize, prune, distill or factorize? [1]
- Which predictions are batch-precomputed, which online, and why not hybrid? [1]
- Real labels are scarce: weak supervision, transfer learning or active learning? [1]
- Fairness, privacy and policy constraints: where do they enter the pipeline?
- What would you build next with another quarter?
- (GenAI rounds) How would the design change if an LLM generates the output? [14]

## Portal comparison

| Portal / resource | Steps | Signature emphasis | Format | Notes |
|---|---|---|---|---|
| ByteByteGo, Aminian & Xu (2023) [15] | 7 | Problem -> ML task framing; 10 case studies with 211 diagrams | Book + online course | Blind reviewers call the framework decent but some find it shallow [25] |
| alirezadir AIMLInterviews [1][2] | 9 | Most granular checklist; MVP architecture as its own step; bandits, shadow and canary rollouts | Free GitHub | Updated for GenAI and agentic design in 2026 [2] |
| Educative Grokking MLSD [5] | 6 | Clarify -> data strategy -> model -> offline/online eval -> deploy -> monitor; fraud detection as the running example | Paid course, mock interviews | Separate Khang Pham course: feed, ads, Airbnb search [6][17] |
| Exponent / Aced [3][4] | 6 | Treats the interview as "convincing the team to launch"; rubric module | Course + video mocks | Time budgets per step (~8 min to define the problem) [3] |
| Hello Interview [7] | 5-6 | Timed delivery; problem framing 5-7 min; deep dives driven by the interviewer | Free guide + paid mocks | Steps are guideposts, not hard rules [7] |
| Chip Huyen [18][19][21] | 4 | Project setup -> data -> modeling -> serving; 27 open questions; production MLOps depth | Free booklet, O'Reilly book, Stanford CS 329S | The ml-interviews-book points to a separate resource for design [19] |
| Patrick Halina [23] | ~5 | Compact; heavy stress on clarifying first | Free blog | Blind calls it one of the best compact resources [24] |
| Babushkin & Kravchenko (Manning, 2025) [22] | 16 chapters | Design-doc-driven: problem -> research -> loss/metrics -> baseline -> error analysis | Book | Closer to real industry design docs than interview scripts |

## Portal and resource index (what each offers)

| # | Resource | Type | Offers |
|---|---|---|---|
| 1 | alirezadir/AIMLInterviews MLSD template [1][2] | GitHub (free) | 9-step template, classic + GenAI + agentic design, MCP-based AI tutor |
| 2 | ByteByteGo "ML System Design Interview" [15] | Book / course | 7-step framework; visual search, YouTube video search, harmful content, video recs, news feed, People You May Know, and more |
| 3 | Educative Grokking the ML System Design Interview [5] | Course | 6-step framework, fraud detection mock, RAG knowledge assistant case |
| 4 | Educative / Khang Pham ML System Design [6][16][17] | Course + book (2022) | YouTube recs, LinkedIn feed, ad click, delivery ETA, Airbnb search |
| 5 | Exponent / Aced ML System Design [3][4] | Course + mocks | Rubric, data pipeline, model selection, evaluation, deployment modules |
| 6 | Hello Interview ML System Design [7] | Guide + mocks | Timed delivery framework, worked problems |
| 7 | IGotAnOffer [8]-[14] | Articles + coaching | Company-specific MLE guides (Meta, Google, Amazon, Apple, OpenAI), GenAI design |
| 8 | Chip Huyen [18]-[21] | Free booklet + book + course | Booklet (2019), 27 questions, DMLS book (O'Reilly 2022), CS 329S |
| 9 | Patrick Halina [23] | Blog | Compact MLSD interview guide |
| 10 | Babushkin & Kravchenko [22][41] | Book (Manning 2025) | 16-chapter design-doc method |
| 11 | interviewing.io [34], Prepfully [35] | Mock marketplaces | Anonymous mocks with FAANG ML engineers |
| 12 | InterviewQuery [33] | Question bank | Company MLE guides (e.g. Google) |
| 13 | DesignGurus Grokking series [38] | Courses | Grokking the AI System Design Interview |
| 14 | Interview Kickstart [39], Codemia [36] | Guides | Framework write-ups, company guides |
| 15 | LeetCode Discuss framework post [26], Blind threads [24][25] | Community | Candidate frameworks, resource reviews |
| 16 | Eugene Yan, "System Design for RecSys & Search" [27] | Blog | Offline/online x retrieval/ranking 2x2 |
| 17 | Engineer1999 curated case studies [40] | GitHub | 300+ case studies from 80+ company blogs |
| 18 | Google Rules of ML [31], Sculley 2015 [30], TFX [29] | Papers / guides | Production best practices and the reference platform |

## References

1. alirezadir, "ML System Design" template (src/MLSD/ml-system-design.md), GitHub, 2025. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/ml-system-design.md
2. alirezadir, "AIMLInterviews (formerly Machine-Learning-Interviews)" README, GitHub, 2026. https://github.com/alirezadir/Machine-Learning-Interviews
3. Aced (formerly Exponent), "Machine Learning System Design Interview (2026 Guide)", aced.io, 2026. https://www.aced.io/blog/machine-learning-system-design-interview-guide
4. Exponent, "Introduction to the ML System Design Interview" (course), tryexponent.com, 2025. https://www.tryexponent.com/courses/ml-system-design/mlsd-intro
5. Educative, "The 6-step ML SD framework" (Grokking the ML System Design Interview), educative.io, 2025. https://www.educative.io/courses/grokking-the-machine-learning-system-design-interview/the-6-step-ml-sd-framework
6. Educative / Khang Pham, "Machine Learning System Design" (course), educative.io, 2021. https://educative.io/courses/machine-learning-system-design/introduction
7. Hello Interview, "ML System Design in a Hurry: Delivery Framework", hellointerview.com, 2025. https://www.hellointerview.com/learn/ml-system-design/in-a-hurry/delivery
8. IGotAnOffer, "Machine Learning (ML) System Design Interview (examples, answers, prep)", igotanoffer.com, 2025. https://igotanoffer.com/en/advice/machine-learning-system-design-interview
9. IGotAnOffer, "Meta Machine Learning Engineer Interview", igotanoffer.com, 2025. https://igotanoffer.com/blogs/tech/facebook-machine-learning-engineer-interview
10. IGotAnOffer, "Google Machine Learning Engineer Interview", igotanoffer.com, 2025. https://igotanoffer.com/blogs/tech/google-machine-learning-engineer-interview
11. IGotAnOffer, "Amazon Machine Learning Engineer Interview", igotanoffer.com, 2025. https://igotanoffer.com/blogs/tech/amazon-machine-learning-engineer-interview
12. IGotAnOffer, "Apple Machine Learning Engineer Interview", igotanoffer.com, 2025. https://igotanoffer.com/en/advice/apple-machine-learning-engineer-interview
13. IGotAnOffer, "OpenAI System Design Interviews", igotanoffer.com, 2025. https://igotanoffer.com/en/advice/openai-system-design-interview
14. IGotAnOffer, "Generative AI System Design Interview", igotanoffer.com, 2025. https://igotanoffer.com/en/advice/generative-ai-system-design-interview
15. A. Aminian & A. Xu, "Machine Learning System Design Interview: An Insider's Guide" (ByteByteGo), library record, 2023. https://library.bracu.ac.bd/vufind/Record/47626 (book overview: https://opencourser.com/book/xop33f/machine-learning-system-design-interview)
16. K. Pham, "Machine Learning Design Interview", Goodreads, 2022. https://www.goodreads.com/book/show/61098541-machine-learning-design-interview
17. khangich, "machine-learning-interview", GitHub, 2021. https://github.com/khangich/machine-learning-interview
18. C. Huyen, "machine-learning-systems-design" booklet, GitHub, 2019. https://github.com/chiphuyen/machine-learning-systems-design
19. C. Huyen, "Introduction to Machine Learning Interviews Book", huyenchip.com, 2021. https://huyenchip.com/ml-interviews-book/
20. C. Huyen, "Machine Learning Systems Design" table of contents, huyenchip.com, 2019. https://huyenchip.com/machine-learning-systems-design/toc.html
21. DataTalks.Club, "Designing Machine Learning Systems" (C. Huyen, O'Reilly) book of the week, 2022. https://datatalks.club/books/20220627-designing-machine-learning-systems.html
22. V. Babushkin & A. Kravchenko, "Machine Learning System Design", Manning, 2025. https://manning.com/books/machine-learning-system-design
23. P. Halina, "ML Systems Design Interview Guide", patrickhalina.com, 2020. http://patrickhalina.com/posts/ml-systems-design-interview-guide/
24. Blind, "MLE interview preparation that led to successful offers", teamblind.com, 2023. https://www.teamblind.com/post/mle-interview-preparation-that-led-to-successful-offers-3k2az8d8
25. Blind, "Resources for ML system design for interviews", teamblind.com, 2024. https://www.teamblind.com/post/resources-for-ml-system-design-for-interviews-51wa76zv
26. LeetCode Discuss, "Machine Learning System Design: A framework for the interview day", leetcode.com, 2020. https://leetcode.com/discuss/interview-question/system-design/566057/machine-learning-system-design-a-framework-for-the-interview-day
27. E. Yan, "System Design for Recommendations and Search", eugeneyan.com, 2021. https://eugeneyan.com/writing/system-design-for-discovery/
28. Uber Engineering, "Meet Michelangelo: Uber's Machine Learning Platform", uber.com, 2017. https://www.uber.com/en-DE/blog/michelangelo-machine-learning-platform/
29. D. Baylor et al., "TFX: A TensorFlow-Based Production-Scale Machine Learning Platform", KDD / Google Research, 2017. https://research.google/pubs/tfx-a-tensorflow-based-production-scale-machine-learning-platform/
30. D. Sculley et al., "Hidden Technical Debt in Machine Learning Systems", NeurIPS, 2015. https://proceedings.neurips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf
31. M. Zinkevich, "Rules of Machine Learning: Best Practices for ML Engineering", Google Developers, 2017. https://developers.google.com/machine-learning/guides/rules-of-ml/
32. ZenML MLOps Database, "Meta FBLearner: Introducing FBLearner Flow, Facebook's AI backbone", zenml.io, 2016 (summary). https://www.zenml.io/mlops-database/meta-fblearner-introducing-fblearner-flow-facebooks-ai-backbone
33. InterviewQuery, "Google Machine Learning Engineer Interview Questions + Guide", interviewquery.com, 2024. https://interviewquery.com/interview-guides/google-machine-learning-interview-questions
34. interviewing.io, home page (anonymous mock interviews including ML system design), 2025. https://www.interviewing.io
35. Prepfully, "Meta Machine Learning Engineer interview questions", prepfully.com, 2025. https://prepfully.com/interview-questions/meta/machine-learning-engineer?page=8
36. Codemia, "Meta Machine Learning Engineer guide", codemia.io, 2025. https://codemia.io/guides/meta-machine-learning-engineer
37. PracHub, "ML System Design Interview Questions", prachub.com, 2026. https://prachub.com/categories/ml-system-design?page=20
38. DesignGurus, "The Grokking series", designgurus.io, 2025. https://www.designgurus.io/grokking
39. Interview Kickstart, "The Ultimate Machine Learning System Design Interview Guide (2026)", interviewkickstart.com, 2026. https://interviewkickstart.com/blogs/articles/machine-learning-system-design-interview-guide
40. Engineer1999, "A Curated List of ML System Design Case Studies", GitHub, 2024. https://github.com/Engineer1999/A-Curated-List-of-ML-System-Design-Case-Studies
41. ML-SystemDesign, "MLSystemDesign BookOutline.md", GitHub, 2024. https://github.com/ML-SystemDesign/MLSystemDesign/blob/main/BookOutline.md
