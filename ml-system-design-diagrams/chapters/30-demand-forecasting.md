# 30. Design Demand Forecasting for Inventory (Amazon Supply Chain, Instacart)

> Sourcing note: web search was exhausted and company-blog / portal domains were
> blocked in this review run. Every company tag and portal row below was re-checked
> against cloned copies of the GitHub indexes [23]-[27], which list the cited posts
> and papers ([1]-[22], [30], [31]) under forecasting or availability. Rows marked
> "typical" are standard interview assumptions, not quoted numbers.

## Company tags

| Company (reported / mirrored) | Evidence | Ref |
|---|---|---|
| Amazon | Multi-horizon quantile forecaster (MQ-RNN -> MQ-CNN -> MQ-Transformer) feeds buying; RL ordering policy research | [1][2][4][5] |
| Amazon (AWS) | DeepAR, GluonTS, Chronos: open probabilistic forecasting models | [3][6][8] |
| Instacart | Real-time item availability (P(in stock)) for hundreds of millions of items; lost-demand modeling | [9][10][11][30][31] |
| DoorDash | Ensemble time-series forecasting; holiday cascade model; supply/demand balance | [12][13][14][15] |
| Uber | Forecasting platform; LSTM extreme-event forecasting | [16][17] |
| Zalando, Ocado, Mercado Libre, Grubhub, Lyft | Item / order-volume demand forecasting in production | [18][19][20][21][22] |

Portals that cover the problem:

| Portal | Coverage | Ref |
|---|---|---|
| eugeneyan/applied-ml | "Forecasting" section: Amazon, Uber, DoorDash, Grubhub, Lyft, LinkedIn Greykite | [23] |
| Engineer1999 ML case-study list | Zalando, DoorDash (ensemble, holiday), Grubhub, Ocado, Mercado Libre, Lyft forecasting; 4 Instacart availability posts; Instacart lost demand | [24] |
| khangich/machine-learning-interview (Khang Pham) | Per-company applied-ML index; forecasting posts under Uber, Airbnb, Gojek | [25] |
| alirezadir/Machine-Learning-Interviews | No forecasting chapter; framework covers split-by-time, seasonality/trend features, batch vs online prediction | [26] |
| chiphuyen/machine-learning-systems-design; "Designing ML Systems" | Generic framework; no forecasting chapter; batch vs online prediction, drift | [27][29] |

## Requirements & scale

| Dimension | Value | Source |
|---|---|---|
| Functional | Forecast demand per (SKU, location, day) for horizon 1-52 weeks; output quantiles (P10/P50/P90) | [1][2] |
| Functional (downstream) | Turn forecast into order quantity / safety stock; flag stock-outs | [5] |
| Functional (Instacart) | P(item in stock at store now) for every item-store, used at search/checkout | [9][10] |
| Scale (Amazon) | Hundreds of millions of products, many fulfillment centers | [1] typical |
| Scale (Instacart) | Hundreds of millions of item-store pairs | [9] |
| Freshness | Batch: daily/weekly reforecast; availability: minutes | typical |
| Latency | Batch job hours (SLA before buying run); availability lookup < 10 ms from cache | typical |
| Non-functional | Calibrated uncertainty, cold-start for new SKUs, robustness to holidays/shocks (COVID) | [1][13][15] |
| Business objective | Minimize holding + lost-sales cost; maximize in-stock rate | [5] |

## Diagram 1 - End-to-end architecture

**Diagram 1. Serving path: batch forecasts -> inventory policy -> orders, plus real-time
availability lookups.**

```
                       +------------------------+
 scheduler (daily) --->| Forecast Orchestrator  |<--- model registry (champion)
                       | partitions SKU x node  |
                       +-----------+------------+
                                   | batch of series ids
                                   v
+---------------+ features +------------------------+ P10..P90 +-----------------+
| Feature Store |--------->| Forecast Inference     |--------->| Forecast Store  |
| sales, price, |          | (MQ-CNN / TFT / GBDT)  |          | (SKU,node,h,q)  |
| promo, cal.   |          | many batch workers     |          +--------+--------+
+---------------+          +------------------------+                   |
                                                                        | quantiles
+---------------+ on-hand, +------------------------+                   |
| Inventory DB  |--------->| Inventory Policy       |<------------------+
| in-transit,   | lead time| newsvendor / (s,S) /   |
| vendor LT     |          | RL buying policy [5]   |
+---------------+          +-----------+------------+
                                       | proposed order qty
                                       v
                           +------------------------+ approved +-----------------+
                           | Business Rules Layer   |--------->| Purchase Orders |
                           | caps, budget, overrides|          | to vendors / FC |
                           +-----------+------------+          +-----------------+
                                       | exceptions, what-if
                                       v
                           +------------------------+
                           | Planner UI / alerts    |
                           +------------------------+

 Real-time availability (Instacart-style) [9][10][30]:
 app / search request (item, store)
       |
       v
 +----------------+  lookup  +-------------------+  upsert  +-------------------+
 | Search/Ranking |--------->| KV cache          |<---------| Availability      |
 | filter, demote |<---------| P(in stock)       |  scores  | scorer (batch +   |
 +-------+--------+  score   +-------------------+          | stream signals)   |
         |                                                  +---------+---------+
         v in-stock badge                                             ^
 response to app            shopper found / not-found events, retailer feeds

 Logging: forecasts, orders, sales, stock-outs --> data lake --> Diagram 2
```

## Diagram 2 - Offline training & data pipeline

**Diagram 2. Training loop with censored-demand correction, backtesting, and feedback.**

```
+-------------+   +-------------+   +--------------+   +----------------------+
| POS / orders|   | Inventory   |   | Price, promo,|   | Calendar, holidays,  |
| clickstream |   | snapshots   |   | catalog attrs|   | weather, events      |
+------+------+   +------+------+   +------+-------+   +----------+-----------+
       |                 |                 |                      |
       +--------+--------+--------+--------+----------------------+
                v                 v
       +----------------+  +--------------------------+
       | Label builder  |  | Censoring fix: sales on  |
       | demand = units |->| stock-out days != demand |
       | per SKU-node-d |  | -> mask or impute (Tobit)|
       +-------+--------+  +------------+-------------+
               |                        |
               v                        v
       +--------------------------------------------+
       | Feature pipeline (lags, rolling stats,     |
       | price elasticity, hierarchy embeddings)    |---> Feature Store (offline)
       +----------------------+---------------------+
                              v
       +--------------------------------------------+
       | Training: global model over all series,    |
       | quantile loss, forking-sequence decoders   |
       +----------------------+---------------------+
                              v
       +--------------------------------------------+
       | Backtest: rolling-origin windows, wQL,     |
       | WAPE, bias by segment, holiday slices      |
       +----------------------+---------------------+
                              | pass gate? yes (no -> reject, keep champion)
                              v
                     +------------------+
                     | Model Registry   |
                     | (versioned)      |
                     +--------+---------+
                              v
                     +---------------------+   +----------------------+
                     | Shadow forecast vs  |-->| Champion swap; feed  |
                     | champion (n weeks)  |   | Diagram 1 inference  |
                     +---------------------+   +----------+-----------+
                                                          |
   feedback: realized sales, stock-outs, overrides <------+
   (retrain weekly; event-triggered retrain on drift, e.g. COVID [15])
```

## Diagram 3 - Model architecture

**Diagram 3. MQ-CNN style multi-horizon quantile forecaster (Amazon) [1][2].**

```
 history x_1..x_t (sales, price, promo, stock flags)     static: SKU/category/node
            |                                                     |
            v                                                     v
 +------------------------+                            +--------------------+
 | Encoder: dilated causal|                            | Embeddings         |
 | CNN (or LSTM/Transf.)  |                            | (item, brand, node)|
 +-----------+------------+                            +---------+----------+
             | context h_t  (forking: one per forecast creation date)
             v                                                    |
 +-----------------------------------------+                      |
 | Global MLP decoder: h_t + known future  |<---------------------+
 | covariates (planned price/promo, hols)  |
 | -> horizon-specific contexts c_1..c_K   |
 +--------------------+--------------------+
                      v
 +-----------------------------------------+
 | Local MLP decoder (shared across k):    |
 | c_k + covariates at t+k -> quantiles    |
 +--------------------+--------------------+
                      v
   y_hat(t+k, q) for k=1..K, q in {0.1,0.5,0.9}
   loss = sum_k sum_q  pinball_q(y_{t+k}, y_hat)   (quantile / wQL)
```

## Diagram 4 - Variant: forecast-then-optimize vs end-to-end policy

**Diagram 4. Two-stage (classic) vs Amazon's deep RL buying policy [5].**

```
 Classic two-stage
 +-----------+  quantiles  +--------------------+  order q* = F^-1(cu/(cu+co))
 | Forecaster|-----------> | Newsvendor / (s,S) |---------------------------->
 +-----------+             | + lead-time demand |   safety stock = z*sigma*sqrt(L)
                           +--------------------+

 End-to-end policy (Deep Inventory Management)
 +--------------------------+   order qty   +--------------------------+
 | Policy net (MQ encoder + |-------------->| Differentiable simulator |
 | state: on-hand, pipeline)|<--------------| replays historical demand|
 +--------------------------+  reward:      +--------------------------+
                               revenue - holding - lost sales
```

## Data & labels

| Item | Choice | Note |
|---|---|---|
| Label | Units demanded per SKU x location x day/week | Sales are censored when out of stock |
| Censoring | Mask stock-out periods, or impute via availability model | Instacart availability doubles as a censoring signal [9] |
| Lost demand | Estimate demand never observed (searches / carts for out-of-stock items) and add back to labels | Instacart "Modeling the unseen" [31] |
| Hierarchy | SKU -> category -> dept; store -> region -> national | Reconcile (top-down / MinT) |
| Splits | Rolling-origin backtests by time, never random | [26] |
| Cold start | New SKUs: attribute/text embeddings, similar-item priors | [18] |
| Anomalies | Remove one-off bulk orders, data outages; tag holidays | [13] |

## Features

| Group | Examples |
|---|---|
| Lags / windows | Demand lag 1/7/14/28/364, rolling mean/std, trend, days since last sale |
| Calendar | Day of week, holidays, paydays, seasonality Fourier terms |
| Price & promo | Current/planned price, discount depth, promo flag, competitor price |
| Inventory | On-hand, stock-out flags, in-transit, vendor lead time |
| Item static | Category, brand, size, embeddings, life-cycle stage |
| Location | Store/FC id embedding, region, demographics, weather |
| Real-time (availability) | Shopper "found / not found", last restock time, store feed recency [9][10] |

## Model choices

| Stage | Model | Pros | Cons |
|---|---|---|---|
| Baseline | Seasonal naive, moving avg, ETS/ARIMA per series | Simple, interpretable | No cross-series learning, poor on sparse SKUs |
| Classic ML | GBDT (LightGBM) on lag features, quantile objective | Strong on tabular (M5 winners [28]) | Separate model per horizon/quantile |
| Deep global | DeepAR (autoregressive, likelihood) | Probabilistic, cold-start via sharing [3] | Error accumulation over long horizons |
| Deep multi-horizon | MQ-RNN / MQ-CNN / MQ-Transformer | Direct multi-horizon quantiles, Amazon prod [1][2][4] | Larger infra |
| Attention | Temporal Fusion Transformer | Interpretable variable selection [7] | Heavier training |
| Foundation | Chronos (pretrained on many series) | Zero-shot for new series [6] | Less domain covariates |
| Ensemble | Stack of above (DoorDash ELITE-style) | Robust | Ops complexity [12] |
| Decision | Newsvendor / RL policy | Optimizes cost directly [5] | Needs simulator |

## Loss

| Loss | When |
|---|---|
| Pinball (quantile) loss, summed over horizons | Default for inventory (asymmetric costs) [2] |
| Negative log-likelihood (neg. binomial / Tweedie) | Count, intermittent demand (DeepAR) [3] |
| Tweedie / Poisson in GBDT | Zero-inflated SKUs |
| Cost-weighted loss (holding vs lost-sale) | Align with business critical ratio |
| Policy reward (RL) | End-to-end buying [5] |

## Offline metrics

| Metric | Why |
|---|---|
| wQL / CRPS | Quality of full distribution |
| WAPE / MAPE / sMAPE | Point accuracy, volume-weighted |
| Bias (mean error) | Systematic over/under-forecasting |
| Quantile coverage (P90 hit rate ~ 90%) | Calibration |
| Simulated in-stock rate & holding cost | Decision-level backtest |
| Sliced: new SKUs, holidays, promos, long tail | Catch hidden failures |

## Online metrics / A-B

| Metric | Note |
|---|---|
| In-stock rate / fill rate | Primary customer outcome |
| Lost sales, substitution rate | Instacart: items not found [9] |
| Inventory holding cost, days of supply | Over-stock signal |
| Waste / spoilage (perishables) | Grocery |
| Delivery promise accuracy | Downstream |
| Test design | Store/FC-level or SKU-cluster randomization, switchback; long tests (weeks) |

## Serving & latency budget

| Path | Budget | Mechanism |
|---|---|---|
| Nightly forecast run | ~2-4 h for all series | Distributed batch (Spark/Ray), partition by SKU (typical) |
| Policy / order generation | ~1 h | Vectorized newsvendor per SKU-node |
| Availability lookup | < 10 ms p99 | Precomputed scores in KV cache [10] |
| Availability refresh | Minutes | Stream updates from shopper events [10] |
| What-if planner query | ~1 s | Cached forecasts + on-demand re-solve |

## Monitoring

| Signal | Action |
|---|---|
| Rolling WAPE / bias vs backtest | Alert, roll back champion |
| Input drift (price feed, missing stores) | Block run, fallback to last forecast |
| Regime shift (COVID, viral item) | Event-triggered retrain, shorter windows [15] |
| Override rate by planners | Proxy for trust |
| Stock-out and overstock outliers | Root-cause by SKU |

## Failure modes

| Failure | Mitigation |
|---|---|
| Censored demand -> forecast spiral down | Stock-out masking, availability-aware labels |
| Holiday / promo spikes | Holiday features, cascade model [13] |
| New SKU cold start | Hierarchical priors, similar-item embeddings, foundation model [6] |
| Intermittent demand | Count likelihoods, Croston-style baselines |
| Feedback loop (forecast drives stock drives sales) | Treat as causal; simulate policy [5][22] |
| Data outages | Freshness checks, last-good fallback |
| Overconfident intervals | Calibrate quantiles, conformal adjustment |

## Interviewer follow-ups

- How do you get true demand when sales are censored by stock-outs?
- Point vs probabilistic forecast: why quantiles for inventory? Which quantile to order at?
- One global model vs per-series models: tradeoffs at 100M+ series?
- How do you forecast for a brand-new SKU or a new store?
- How do you handle promotions and price changes you plan in the future (known covariates)?
- How would you reconcile forecasts across SKU / category / region hierarchy?
- How do you backtest correctly without leakage?
- How did COVID-like shocks break models, and how would you retrain? [15]
- How do lead time and vendor uncertainty enter the order quantity?
- How do you A/B test a forecasting model when inventory takes weeks to react?
- How does forecasting interact with pricing (forecast depends on price)? [22]
- Batch vs real-time: when would Instacart-style availability need streaming? [10]
- How would you use an RL policy instead of forecast-then-optimize? [5]
- How do you handle intermittent, long-tail SKUs?
- How do you estimate demand you never observed (lost demand from out-of-stock searches)? [31]
- How do you keep the availability score fresh for 200M+ item-store pairs without rescoring everything? [30]
- What latency/compute budget for the nightly run, and how to parallelize?

## Portal comparison

| Portal | Framing | Model emphasis | Distinctive |
|---|---|---|---|
| Amazon Science / papers [1][2][5] | Multi-horizon quantile forecast feeding buying | MQ-CNN, MQ-Transformer, RL policy | Decision-aware, forking sequences |
| Instacart engineering [9][10][11] | P(in stock) per item-store in real time | ML scorer + streaming signals | Serving architecture, consistency |
| DoorDash engineering [12][13][14] | Order/supply volume forecasting | Ensemble stacking, holiday cascade | Ensemble ops, holiday handling |
| eugeneyan applied-ml / Khang Pham [23][25] | Case-study index | Points to Uber, Amazon, Lyft posts | Reading list, not a walkthrough |
| alirezadir / Chip Huyen [26][27][29] | Generic MLSD framework | Time-based splits, batch prediction, drift | Applies to any problem |

## References

1. "The history of Amazon's forecasting algorithm", Amazon Science, 2021. https://www.amazon.science/latest-news/the-history-of-amazons-forecasting-algorithm
2. Wen, R. et al., "A Multi-Horizon Quantile Recurrent Forecaster", arXiv, 2017. https://arxiv.org/abs/1711.11053
3. Salinas, D. et al., "DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks", arXiv, 2017. https://arxiv.org/abs/1704.04110
4. Eisenach, C. et al., "MQTransformer: Multi-Horizon Forecasts with Context Dependent and Feedback-Aware Attention", arXiv, 2020. https://arxiv.org/abs/2009.14799
5. Madeka, D. et al., "Deep Inventory Management", arXiv, 2022. https://arxiv.org/abs/2210.03137
6. Ansari, A. F. et al., "Chronos: Learning the Language of Time Series", arXiv, 2024. https://arxiv.org/abs/2403.07815
7. Lim, B. et al., "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting", arXiv, 2019. https://arxiv.org/abs/1912.09363
8. "GluonTS: Probabilistic time series modeling in Python", GitHub (awslabs), 2024. https://github.com/awslabs/gluonts
9. "How Instacart Modernized the Prediction of Real Time Availability for Hundreds of Millions of Items While Saving Costs", Instacart Tech Blog, 2023. https://tech.instacart.com/how-instacart-modernized-the-prediction-of-real-time-availability-for-hundreds-of-millions-of-items-59b2a82c89fe
10. "Instacart's Item Availability Architecture: Solving for scale and consistency", Instacart Tech Blog, 2023. https://tech.instacart.com/instacarts-item-availability-architecture-solving-for-scale-and-consistency-f5661acb20a6
11. "How Instacart's Item Availability Evolved Over the Pandemic", Instacart, 2023. https://www.instacart.com/company/how-its-made/how-instacarts-item-availability-evolved-over-the-pandemic/
12. "How DoorDash Built an Ensemble Learning Model for Time Series Forecasting", DoorDash Engineering Blog, 2023. https://doordash.engineering/2023/06/20/how-doordash-built-an-ensemble-learning-model-for-time-series-forecasting/
13. "How DoorDash Improves Holiday Predictions via Cascade ML Approach", DoorDash Engineering Blog, 2023. https://doordash.engineering/2023/08/31/how-doordash-improves-holiday-predictions-via-cascade-ml-approach/
14. "Managing Supply and Demand Balance Through Machine Learning", DoorDash Engineering Blog, 2021. https://doordash.engineering/2021/06/29/managing-supply-and-demand-balance-through-machine-learning/
15. "Retraining Machine Learning Models in the Wake of COVID-19", DoorDash Engineering Blog, 2020. https://doordash.engineering/2020/09/15/retraining-ml-models-covid-19/
16. "Forecasting at Uber: An Introduction", Uber Engineering Blog, 2018. https://eng.uber.com/forecasting-introduction/
17. "Engineering Extreme Event Forecasting at Uber with Recurrent Neural Networks", Uber Engineering Blog, 2017. https://eng.uber.com/neural-networks/
18. "Deep Learning based Forecasting: a case study from the online fashion industry" (Zalando), arXiv, 2023. https://arxiv.org/abs/2305.14406
19. "Finding the sweet spot", Ocado Group, 2021. https://careers.ocadogroup.com/blogs/careers-blogs/our-technologies/finding-the-sweet-spot
20. "Marketplace Forecasting: Sales or Demand? Why not both?", Mercado Libre Tech (Medium), 2021. https://medium.com/mercadolibre-tech/global-time-series-forecasting-models-for-item-level-demand-and-sales-forecasts-in-our-marketplace-aee2956957ae
21. "Forecasting Grubhub Order Volume At Scale", Grubhub Bytes, 2022. https://bytes.grubhub.com/forecasting-grubhub-order-volume-at-scale-a966c2f901d2
22. "Causal Forecasting at Lyft (Part 1)", Lyft Engineering Blog, 2022. https://eng.lyft.com/causal-forecasting-at-lyft-part-1-14cca6ff3d6d
23. Eugene Yan, "applied-ml", GitHub, 2023. https://github.com/eugeneyan/applied-ml
24. Engineer1999, "A Curated List of ML System Design Case Studies", GitHub, 2024. https://github.com/Engineer1999/A-Curated-List-of-ML-System-Design-Case-Studies
25. Khang Pham, "machine-learning-interview" applied ML index, GitHub, 2021. https://github.com/khangich/machine-learning-interview/blob/master/appliedml.md
26. alirezadir, "Machine-Learning-Interviews" (MLSD notes), GitHub, 2024. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/ml-system-design.md
27. Chip Huyen, "machine-learning-systems-design" exercises, GitHub, 2019. https://github.com/chiphuyen/machine-learning-systems-design/blob/master/content/exercises.md
28. "M5 Forecasting - Accuracy", Kaggle, 2020. https://www.kaggle.com/competitions/m5-forecasting-accuracy
29. Chip Huyen, "Designing Machine Learning Systems", O'Reilly, 2022. https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/
30. "Predicting the real-time availability of 200 million grocery items in US & Canada stores", Instacart Tech Blog. https://tech.instacart.com/predicting-real-time-availability-of-200-million-grocery-items-in-us-canada-stores-61f43a16eafe
31. "Modeling the unseen", Instacart Tech Blog. https://tech.instacart.com/modeling-the-unseen-6a51c9a02430
