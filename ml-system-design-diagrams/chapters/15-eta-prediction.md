# 15. Design ETA / Delivery Time Prediction (Uber, DoorDash, Amazon Delivery)

> Sourcing note: in this review run, company blogs, arXiv and ACM could not be fetched
> (proxy denial) and the web-search budget was spent. The GitHub portal repositories
> [7][8][9][12] were cloned and checked line by line. Company and paper references
> ([1]-[6], [10], [11], [13]) use their canonical URLs; four of them ([1], [10], [11],
> [13]) are also linked from those portal repositories. Rows marked "typical
> assumption" are standard interview estimates, not quoted numbers.

## Company tags

| Company | Evidence | Ref |
|---|---|---|
| Uber / Uber Eats | DeepETA: a transformer that predicts a residual on top of the routing-engine ETA | [1][2] |
| Uber Eats | Trip inferences and ML (food prep time) used to optimize delivery times | [11] |
| DoorDash | Multi-task deep ETA models with probabilistic outputs; long-tail ETA work | [3][13] |
| Google Maps | GNN over road "supersegments" for travel-time ETA | [4][5] |
| DiDi | Wide-Deep-Recurrent (WDR) travel-time model | [6] |
| Generic question | "Food delivery time approximation" is a sample MLSD question | [7][12] |
| Generic question | "Estimate Delivery time" is a module of Khang Pham's ML design course | [8] |

Amazon, Instacart, Lyft and Grab use the same pattern in interviews, but this run found
no source that ties them to the question, so they are not tagged.

Portals that cover the problem:

| Portal | Coverage | Ref |
|---|---|---|
| khangich/machine-learning-interview | "Estimate Delivery time" course module (paid, rebrand.ly/mldesign); README lists Uber Eats and DoorDash dispatch reading | [8] |
| alirezadir (AIMLInterviews / Machine-Learning-Interviews) | "Food delivery time approximation" listed under "Other" sample questions; links Uber DeepETA | [7][12] |
| chiphuyen/machine-learning-systems-design | Generic framework only (setup, data, modeling, serving); no ETA case | [9] |

## Requirements & scale

| Dimension | Value | Source |
|---|---|---|
| Functional | Predict ETA before the order or request (quote) and refresh it during the trip | [1][3] |
| Functional | Serve several ETA stages (browse page, checkout, post-checkout tracking) | [3] |
| Functional | Decompose: prep time + courier to store + wait + store to customer | [10][11] |
| Functional | Output a point estimate plus an interval ("25-35 min") | [3] |
| Consumers | Users (quote), dispatch/matching, pricing, supply positioning | [1][10] |
| Latency | Low single-digit ms per model call; ETA is called many times per request | [1] |
| Throughput | Uber: DeepETA is described as one of the highest-QPS models at the company | [1] |
| Throughput | 50k-500k ETA calls/s at peak (dispatch scores many driver-order pairs) | typical assumption |
| Scale | Millions of orders a day, millions of road segments, 100k+ stores | typical assumption |
| Accuracy | Minimize MAE; bound the late rate (under-prediction costs more) | [1][3] |
| Freshness | Traffic and store-load features updated within seconds to minutes | [3][4] |

## DIAGRAM 1: Online ETA serving path

**Figure 15.1 - Request -> routing base ETA -> feature fetch -> model -> post-process -> quote**

```
 +-----------+   order / trip   +----------------------+
 | Client    |----------------->| ETA API gateway      |
 | app       |<-----------------| (quote, refresh)     |
 +-----------+  ETA + range     +----------+-----------+
                                           | origin, store, dest, time
                                           v
 +------------------------+  base ETA  +------------------------+
 | Routing engine         |----------->| Leg decomposer         |
 | (road graph, shortest  |  per leg   | prep, to-store, wait,  |
 |  path / contraction)   |            | to-customer            |
 +------------------------+            +-----------+------------+
                                                   | legs + base ETAs
                                                   v
 +------------------------+  features  +------------------------+
 | Online feature store   |----------->| Feature assembler      |
 | live traffic, store    |            | geo, time, categorical |
 | load, courier supply   |            +-----------+------------+
 +-----------^------------+                        | feature vector
             | Kafka / Flink streams               v
 +-----------+------------+            +------------------------+
 | GPS pings, order and   |            | ETA model server       |
 | store events           |            | residual + quantiles   |
 +------------------------+            +-----------+------------+
                                                   ^      | raw ETA, p50/p90
 +------------------------+  load model            |      v
 | Model registry         |------------------------+  +------------------------+
 +------------------------+                           | Post-process           |
 +------------------------+  buffers, SLA             | calibrate, clip, round |
 | Business rules         |-------------------------->| interval, fallback to  |
 +------------------------+                           | routing ETA on timeout |
                                                      +-----------+------------+
                                                                  | final ETA
                                                                  v
                                                      +------------------------+
                                                      | Gateway -> client;     |
                                                      | log req id, features,  |
                                                      | ETA, model version     |
                                                      +------------------------+
```

## DIAGRAM 2: Offline training & data pipeline

**Figure 15.2 - Logs -> actual-time labels -> features -> training -> eval -> registry -> rollout**

```
 +----------------+   +------------------+   +-------------------+
 | Prediction log |   | Order / trip     |   | Courier GPS pings |
 | (ETA, features,|   | events (created, |   | + map matching    |
 |  model ver.)   |   | picked, dropped) |   | (segment speeds)  |
 +-------+--------+   +--------+---------+   +---------+---------+
         |                     |                       |
         +---------------------+-----------------------+
                               | join on order id after completion (label delay)
                               v
 +--------------------------------------+   +-------------------------------+
 | Label builder                        |-->| Data quality: drop cancels,   |
 | total = dropoff - created            |   | GPS outliers, fraud; flag     |
 | per-leg: prep, travel, wait          |   | batched orders                |
 | residual = actual - routing ETA      |   +---------------+---------------+
 +--------------------------------------+                   |
                                                            v
 +--------------------------------------+   +-------------------------------+
 | Offline feature store                |<--| Feature pipelines (Spark):    |
 | point-in-time snapshots (no leakage) |   | store prep history, route     |
 +------------------+-------------------+   | stats, weather, events        |
                    |                       +-------------------------------+
                    | split by time (train < val < test), not random
                    v
 +----------------------+   +-------------------------+   +--------------------+
 | Training (GPU)       |-->| Offline eval            |-->| Model registry     |
 | GBDT baseline, deep  |   | MAE, p50/p90 error,     |   | version, metrics,  |
 | residual, quantile / |   | late rate, coverage by  |   | feature schema     |
 | multi-task heads     |   | city, hour, distance    |   +---------+----------+
 +----------^-----------+   +-------------------------+             |
            |                                                       v
            |                                     +--------------------------------+
            |   retrain daily / weekly,           | Shadow -> canary -> switchback |
            +---- or on drift alert --------------| / A-B, rolled out per city     |
                                                  +--------------------------------+
```

## DIAGRAM 3: Model architecture

**Figure 15.3 - Residual transformer over a routing-engine base ETA (DeepETA-style [1][2])**

```
  INPUTS
  +-----------------+ +-----------------+ +----------------+ +-----------------+
  | Continuous      | | Geo features    | | Categorical    | | Base ETA from   |
  | dist, base ETA, | | origin/dest     | | city, product, | | routing engine  |
  | hour, traffic   | | lat/lng -> geo- | | request type,  | | (y_route)       |
  |                 | | hash, multi-res | | weekday        | |                 |
  +--------+--------+ +--------+--------+ +-------+--------+ +--------+--------+
           |                   |                  |                   |
           v                   v                  v                   |
  +----------------+ +-------------------+ +----------------+         |
  | quantile bucket| | feature hashing   | | embedding      |         |
  | -> embedding   | | -> embeddings     | | tables         |         |
  +-------+--------+ +---------+---------+ +-------+--------+         |
          +--------------------+-------------------+                  |
                               | one token per feature (K x d)        |
                               v                                      |
                 +----------------------------+                       |
                 | Linear-attention           |                       |
                 | transformer encoder        |                       |
                 | (feature interactions)     |                       |
                 +-------------+--------------+                       |
                               | pooled vector                        |
                               v                                      |
                 +----------------------------+                       |
                 | Bias-adjustment decoder    |                       |
                 | FC layers + per-segment    |                       |
                 | bias (request type, city)  |                       |
                 +-------------+--------------+                       |
                               | residual r_hat                       |
                               v                                      v
                 +---------------------------------------------------------+
                 | ETA_hat = y_route + r_hat   (optional p50 / p90 heads)  |
                 | loss: asymmetric Huber, or quantile (pinball)           |
                 +---------------------------------------------------------+
```

## DIAGRAM 4: Variant - delivery ETA with leg heads and a distribution output

**Figure 15.4 - Shared encoder, per-leg heads (interview pattern) and a probabilistic
total-ETA head [3]**

```
 +----------------+  +----------------+  +----------------+  +----------------+
 | Order          |  | Store          |  | Courier supply |  | Route/traffic  |
 | items, size,   |  | hist prep time,|  | idle couriers, |  | base travel    |
 | time, promo    |  | open orders    |  | dist to store  |  | time, weather  |
 +-------+--------+  +-------+--------+  +-------+--------+  +-------+--------+
         |                   |                   |                   |
         +-------------------+---------+---------+-------------------+
                                       v
                        +-------------------------------+
                        | Shared encoder (MLP or        |
                        | transformer)                  |
                        +---------------+---------------+
                                        | shared representation
         +-------------------+----------+--------+---------------------+
         v                   v                   v                     v
 +----------------+  +----------------+  +----------------+  +---------------------+
 | Head: prep     |  | Head: courier  |  | Head: store -> |  | Head: total ETA     |
 | time (store)   |  | arrival + wait |  | customer drive |  | distribution params |
 +-------+--------+  +-------+--------+  +-------+--------+  +----------+----------+
         |                   |                   |                      |
         +-------------------+-------------------+                      |
                             | sum of legs ~ total (consistency loss)   |
                             v                                          v
              +-----------------------------+          +-----------------------------+
              | Dispatch / batching uses    |          | Customer quote: pick the    |
              | per-leg predictions [10]    |          | quantile for on-time target |
              +-----------------------------+          +-----------------------------+
```

## Data & labels

| Item | Detail | Ref |
|---|---|---|
| Label (rides) | Actual pickup or trip duration from trip events | [1] |
| Label (delivery) | created -> dropoff; per-leg: prep, courier to store, wait, drive | [10][11] |
| Prep-time label | Not observed directly; inferred from courier arrival and pickup events | [11] |
| Residual label | actual - routing-engine ETA (narrower, easier target) | [1][2] |
| Label delay | Known only after completion (30-60 min); join by order id | typical assumption |
| Noise | Cancelled orders, GPS drift, courier batching, manual "picked up" taps | typical assumption |
| Splits | Time-based; hold out whole days; check per city and holidays | typical assumption |
| Volume | Hundreds of millions of historical trips for training | [1] |

## Features

| Group | Examples | Freshness |
|---|---|---|
| Route | Distance, base routing ETA, turn count, highway share | per request |
| Traffic | Live segment speeds, historical speeds by hour of week | seconds-minutes [4] |
| Geo | Origin/destination geohash at several resolutions, hashed embeddings | static [1] |
| Time | Hour, weekday, holiday, minute of day (cyclical) | per request |
| Store | Historical prep time, open orders, cuisine, order size | minutes [3][11] |
| Supply | Idle couriers nearby, courier distance, acceptance rates | seconds [10] |
| Real-time load | Orders in the last N minutes at the store and in the area (long-tail signal) | minutes [13] |
| Context | Weather, events, product type (Eats vs rides, scheduled vs ASAP) | minutes |

## Model choices

| Stage | Model | Pros | Cons |
|---|---|---|---|
| Baseline 0 | Routing engine ETA (graph shortest path) | No training, explainable | Misses prep time and traffic quirks |
| Baseline 1 | Historical average by (store, hour) + routing | Trivial | No interactions, cold start |
| Classic | GBDT (XGBoost/LightGBM) on tabular features | Strong, fast, interpretable | Hits a ceiling at Uber's data scale; Uber moved from XGBoost to DeepETA [1] |
| Deep residual | Transformer/MLP predicting the residual over base ETA | Scales with data, embeddings for geo | Needs careful latency work [1] |
| Graph | GNN over road supersegments | Captures how traffic spreads in space | Heavy infra; mainly for the travel leg [4][5] |
| Sequence | Wide-Deep-Recurrent over route links | Uses the link sequence | Long sequences, slower [6] |
| Multi-task / probabilistic | Shared encoder, stage or leg heads, distribution output | One model for many ETA stages; intervals for quotes | Harder to train and calibrate [3] |

## Loss

| Loss | Why |
|---|---|
| MAE / L1 | Robust to outliers, matches the MAE KPI |
| Huber | Smooth near 0, robust tails |
| Asymmetric Huber | Penalize lateness more than earliness; tunable [1] |
| Quantile (pinball) for p50/p80/p90 | Gives intervals and SLA-aligned quotes |
| Distribution NLL (predict distribution parameters) | Full probabilistic forecast for quotes [3] |
| Multi-task sum + leg consistency | Total and sum of legs agree |

## Offline metrics

| Metric | Notes |
|---|---|
| MAE, RMSE, MAPE | Overall and sliced by city, hour, distance, product |
| p50 / p90 absolute error | The tail matters for user trust |
| Late rate | Share of orders where actual > quoted (or > upper bound) |
| Interval coverage & width | Calibration of p10-p90 bands |
| Long-tail error | Error on the slowest few percent of orders (rush, storms) [13] |
| Bias by segment | Systematic early or late in new cities, rain, peak |

## Online metrics & A-B

| Metric | Direction |
|---|---|
| On-time rate vs quote | Up |
| Conversion (checkout after quote) | Up; pessimistic ETAs lower it |
| Cancellations, "where is my order" support contacts | Down |
| Courier wait time at store (dispatch quality) | Down [10] |
| Guardrails: p99 latency, error rate, revenue | No regression |
| Design | Switchback or geo-cluster experiments (dispatch network effects) |

## Serving & latency budget

| Step | Budget (typical assumption) |
|---|---|
| Routing engine base ETA | 5-20 ms |
| Online feature fetch (Redis/Cassandra) | 2-5 ms |
| Model inference (CPU, linear attention) | 1-5 ms [1] |
| Post-processing, calibration, rules | <1 ms |
| Total per quote | under ~50 ms; dispatch scores thousands of pairs in batches |
| Tactics | Residual on base ETA, embedding caches, quantized or distilled models, fall back to routing ETA on timeout |

## Monitoring

| Signal | Action |
|---|---|
| Rolling MAE and late rate by city-hour (once labels arrive) | Alert, roll back |
| Feature drift (traffic feed outage, stale store load) | Fall back to historical features |
| Prediction shift (mean ETA jump) | Check map or routing releases |
| Interval coverage | Recalibrate quantiles |
| Latency, timeouts, fallback rate | Scale out or simplify the model |

## Failure modes

| Failure | Mitigation |
|---|---|
| Feedback loop: the quote shapes courier behavior and dispatch | Log exploration, causal eval, per-leg labels |
| Label leakage (future features in training) | Point-in-time feature snapshots |
| Cold-start stores or new cities | Hierarchical fallbacks (cuisine, region), shared embeddings |
| Extreme events (storms, concerts, rush peaks) | Real-time load features, surge flags, wider intervals [13] |
| Routing-engine changes shift the base ETA | Retrain the residual model, monitor bias |
| Quotes that are consistently optimistic | Asymmetric loss, quantile quoting, buffers |
| Batched multi-order trips | Explicit batch features or a separate model |

## Interviewer follow-ups

- Why predict a residual over the routing engine instead of the raw ETA? [1]
- How do you split a delivery ETA into legs, and do you train one model or many?
- How do you handle the higher cost of being late than early?
- How would you output an interval instead of one number, and calibrate it? [3]
- How do you encode lat/lng and very high-cardinality geo IDs? (geohash, feature hashing) [1]
- How do you get real-time traffic and store-load features into the model fast?
- How do you avoid leakage when building training data from logs?
- How do you update the ETA mid-trip as new GPS pings arrive?
- XGBoost vs a deep model: when does the deep model win, and at what latency? [1]
- How do you A/B test ETA changes given dispatch network effects? (switchback)
- How do you handle a new city or a new restaurant with no history?
- How do you keep accuracy on long-tail events such as storms or rush peaks? [13]
- How do you label food prep time when nobody records it? [11]
- What happens if the traffic feed or feature store goes down? (fallbacks)
- How does ETA feed dispatch, pricing and batching decisions? [10][11]
- Which metric would you report to the business, and why not only MAE?

## Portal comparison

| Portal | Framing | Model emphasis | Distinctive point |
|---|---|---|---|
| Khang Pham (khangich) [8] | "Estimate Delivery time" course module | Not public (paid course) | README pairs it with Uber Eats and DoorDash dispatch posts |
| alirezadir [7][12] | "Food delivery time approximation" sample question | Generic MLSD template | Listed next to proximity service; links DeepETA |
| Chip Huyen MLSD booklet [9] | Framework only | Data, modeling, serving steps | Stresses label definition and serving constraints |
| Uber DeepETA (company) [1][2] | Residual over routing engine | Linear-attention transformer, asymmetric Huber | Engineered for very high QPS |
| DoorDash (company) [3][13] | Delivery ETA at several stages | Multi-task, probabilistic forecasts | Quantile quotes; long-tail handling |
| Google Maps (company) [4][5] | Road travel time | GNN on supersegments | Models how traffic spreads in space |

## References

1. "DeepETA: How Uber Predicts Arrival Times Using Deep Learning", Uber Engineering Blog, 2022. https://www.uber.com/blog/deepeta-how-uber-predicts-arrival-times/
2. Hu, X., Binaykiya, T., Frank, E., Cirit, O., "DeeprETA: An ETA Post-processing System at Scale", arXiv, 2022. https://arxiv.org/abs/2206.02127
3. "Improving ETAs with multi-task models, deep learning, and probabilistic forecasts", DoorDash Engineering Blog, 2024. https://careersatdoordash.com/blog/improving-etas-with-multi-task-models-deep-learning-and-probabilistic-forecasts/
4. Derrow-Pinion, A. et al., "ETA Prediction with Graph Neural Networks in Google Maps", CIKM, 2021. https://arxiv.org/abs/2108.11482
5. "Traffic prediction with advanced Graph Neural Networks", DeepMind Blog, 2020. https://deepmind.google/discover/blog/traffic-prediction-with-advanced-graph-neural-networks/
6. Wang, Z., Fu, K., Ye, J., "Learning to Estimate the Travel Time", KDD, 2018. https://dl.acm.org/doi/10.1145/3219819.3219900
7. alirezadir, "ML System Design" notes (AIMLInterviews), GitHub. https://github.com/alirezadir/AIMLInterviews/blob/main/src/MLSD/ml-system-design.md
8. Khang Pham, "machine-learning-interview" repository, GitHub. https://github.com/khangich/machine-learning-interview
9. Chip Huyen, "machine-learning-systems-design", GitHub, 2019. https://github.com/chiphuyen/machine-learning-systems-design
10. "Next-Generation Optimization for Dasher Dispatch at DoorDash", DoorDash Engineering Blog, 2020. https://doordash.engineering/2020/02/28/next-generation-optimization-for-dasher-dispatch-at-doordash/
11. "How Trip Inferences and Machine Learning Optimize Delivery Times on Uber Eats", Uber Engineering Blog, 2018. https://eng.uber.com/uber-eats-trip-optimization/
12. alirezadir, "Machine-Learning-Interviews" repository, GitHub. https://github.com/alirezadir/Machine-Learning-Interviews
13. "Improving ETA Prediction Accuracy for Long-tail Events", DoorDash Engineering Blog, 2021. https://doordash.engineering/2021/04/28/improving-eta-prediction-accuracy-for-long-tail-events/
