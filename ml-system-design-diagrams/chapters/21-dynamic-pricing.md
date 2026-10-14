# 21. Design Dynamic Pricing (Uber Surge, Airbnb Smart Pricing, Amazon Pricing)

> Sourcing note: live web fetch and search were unavailable when this chapter was
> reviewed. Every company post and paper below was checked against the curated GitHub
> indexes [10][11] (cloned locally), which link to it under the URL given here; [2], [6],
> [14] and the portal pages [17]-[21] are canonical URLs not re-fetched. Rows marked
> "typical" are standard interview assumptions, not quoted numbers.

## Company tags

| Company (reported / mirrored) | Evidence | Ref |
|---|---|---|
| Airbnb | Smart Pricing: booking-probability model, then price strategy model, then personalization; KDD paper with custom loss | [1][2] |
| Uber | Surge multiplier per hexagonal zone; demand forecasting with LSTMs and a forecasting platform | [3][4][5][6] |
| Lyft | "Pricing at Lyft" (indexed under Anomaly Detection in [11]); causal forecasting for marketplace levers | [7][8] |
| Alibaba (Tmall) | Dynamic pricing framed as deep reinforcement learning (MDP over price changes), with field experiments | [13] |
| Alibaba (second-hand items) | Price suggestion for C2C listings from images + text (cold-start pricing for sellers) | [22] |
| Amazon | Probabilistic demand forecasting (DeepAR, MQ-RNN/CNN lineage); published work is about supply-chain forecasting, so its use as a pricing input is an interview assumption | [14][15] |
| DoorDash | Supply/demand balance via ML forecasts plus levers (pay, promotions) | [16] |

Portals that cover the problem:

| Portal | Coverage | Ref |
|---|---|---|
| khangich/machine-learning-interview (Khang Pham) | Applied-ML index lists Airbnb "Optimal Pricing", Airbnb revenue-forecasting platform and Alibaba DRL pricing as study cases | [10][23] |
| eugeneyan/applied-ml | Lists "Pricing at Lyft", Alibaba DRL pricing (Reinforcement Learning section), Alibaba price suggestion, Uber/Lyft/DoorDash/Amazon forecasting posts, Airbnb and DoorDash causal-inference posts | [11] |
| alirezadir/Machine-Learning-Interviews (MLSD notes) | No pricing chapter; its question list has "Ride matching system" and "Food delivery time approximation" (same marketplace stack) | [9] |
| chiphuyen/machine-learning-systems-design | Generic 4-step framework (project setup, data pipeline, modeling, serving); no pricing case | [12] |
| ByteByteGo (Aminian & Xu) | No pricing chapter in the book; its Airbnb "similar listings" chapter is the closest marketplace case | [18] |
| Exponent, Educative, IGotAnOffer, Patrick Halina | Generic MLSD frameworks only; no pricing walkthrough confirmed | [17][19][20][21] |

## Requirements & scale

| Dimension | Value | Source |
|---|---|---|
| Functional (rides) | Return a price / surge multiplier for a (pickup zone, time) at request time | [3][7] |
| Functional (stays) | Suggest a nightly price per listing-night for the next ~1 year; host keeps min/max bounds | [1][2] |
| Functional (e-com) | Recommend price per SKU subject to margin, competitor and MAP constraints | [13] typical |
| Objective | Maximize revenue / GMV / bookings while balancing supply and demand | [1][3][13] |
| Guardrails | Caps, smoothing, fairness, legal (e.g. emergency price caps), price-change frequency | typical |
| Latency | Rides: price quote under ~100 ms p99 (precomputed per zone); stays: batch daily | typical |
| Freshness | Rides: zone multiplier refreshed every ~1-2 min; stays: daily batch; e-com: hourly-daily | typical |
| Scale | Rides: ~1e5 hex zones x city, ~1e4-1e5 quote QPS peak; stays: ~7e6 listings x 365 nights | typical |
| Explainability | Hosts/drivers need reasons ("high demand", "local events") | [1] |
| Upfront price | Rides: fare fixed at request time from predicted distance/duration x zone multiplier, so ETA/route error becomes pricing error | typical |
| Host control | Stays: suggestions are opt-in; host can accept per night or turn on automatic pricing within min/max | [1] |

## Diagrams

**DIAGRAM 1: Online serving - ride quote path plus per-zone surge loop (Uber/Lyft style)**

```
 REQUEST PATH (per quote, p99 < ~100 ms)
 +-----------+ 1. quote(origin, dest, t, product) +----------------------+
 | Rider app |----------------------------------->| Pricing API gateway  |
 |           |<-----------------------------------| (auth, rate limit)   |
 +-----------+ 6. price, surge, reason, quote_id  +----------+-----------+
                                                             |
                                                             v
 +-----------------------+                        +----------+-----------+
 | ETA / route models    |  pred. distance, time  | 2. Base fare         |
 | (see ETA chapter)     |----------------------->|  f(dist, time, prod) |
 +-----------------------+                        +----------+-----------+
                                                             |
 +-----------------------+  m(H3 zone of origin)  +----------v-----------+
 | Zone multiplier KV    |----------------------->| 3. Price = base x m  |
 | (Redis, TTL ~2 min)   |                        |  + fees, promos      |
 +-----------^-----------+                        +----------+-----------+
             |                                               |
             | 5. write m(z) every 1-2 min                   v
             |                                    +----------+-----------+
             |                                    | 4. Guardrails        |
             |                                    |  caps, legal limits, |
             |                                    |  emergency override  |--> to 6
             |                                    +----------+-----------+
             |                                               |
             |                log(quote_id, price, ctx, arm) v
             |                                    +----------+-----------+
             |                                    | Quote log -> Kafka   |
             |                                    | -> warehouse (labels)|
             |                                    +----------------------+
             |   SURGE LOOP (per H3 hex, every 1-2 min)
 +-----------+-----------+    +-----------------------+    +----------------------+
 | Surge optimizer job   |<---| Zone state + forecast |<---| Stream processor     |
 | m = g(D/S, elasticity)|    | open reqs, idle       |    | (Kafka + Flink)      |
 | clip [1.0, cap],      |    | drivers, ETA, D(t+h), |    | 1-5 min windows      |
 | spatial/temporal      |    | S(t+h) quantiles      |    +----------^-----------+
 | smoothing, hysteresis |    +-----------------------+               |
 +-----------------------+                                 app events: open,
                                                           request, accept,
                                                           cancel, driver GPS
```

**DIAGRAM 2: Offline training and data pipeline with feedback loop**

```
 +---------------+   +----------------+   +-------------------+
 | Quote log     |-->| Outcome join   |-->| Label builder     |
 | (price shown, |   | booked? when?  |   | - booked/not      |
 |  context, t)  |   | cancel? lead t |   | - unbooked nights |
 +---------------+   +----------------+   | - conversion@price|
                                          +---------+---------+
 +---------------+   +----------------+             |
 | Supply / GPS  |-->| Zone x 5-min   |             v
 | Demand events |   | aggregates     |---->+-------+---------+
 +---------------+   +----------------+     | Feature pipeline|
 +---------------+                          | (Spark, offline |
 | External:     |------------------------->|  = online defs) |
 | events, wx,   |                          +-------+---------+
 | holidays, comp|                                  |
 +---------------+                    point-in-time |  train / val (time split)
                                                    v
 +--------------------+   +--------------------+   +-------------------+
 | Demand forecaster  |   | Price-response     |   | Policy / strategy |
 | (zone x horizon)   |   | (booking prob.)    |   | model (optimal    |
 | LSTM / DeepAR      |   | GBDT / DNN         |   | price, custom loss|
 +---------+----------+   +---------+----------+   +---------+---------+
           |                        |                        |
           +------------+-----------+------------+-----------+
                        v                        v
            +-----------+-----------+  +---------+----------+
            | Offline eval: MAPE,   |  | Simulator / replay |
            | AUC, calibration,     |  | counterfactual rev,|
            | booking regret, PDR   |  | off-policy (IPS)   |
            +-----------+-----------+  +---------+----------+
                        |   pass gates           |
                        v                        v
            +-----------+------------------------+--------+
            | Model registry -> shadow -> canary -> A/B   |
            | (switchback by city x time for marketplaces)|
            +-----------+---------------------------------+
                        | deploy
                        v
            +-----------+-----------+   new prices change demand
            | Online serving (D1)   |---------------------------+
            +-----------------------+                           |
                        ^       feedback: quotes + outcomes     |
                        +---------------------------------------+
```

**DIAGRAM 3: Model architecture (Airbnb-style 3-stage + rides demand/supply forecaster)**

```
  Stage A: Price-response (booking probability) model
  +--------------------------------------------------------------+
  | inputs: listing feats (beds, loc, reviews, amenities),       |
  |  night feats (lead time, DOW, season, events),               |
  |  market feats (search volume, comps' prices, occupancy),     |
  |  candidate price p                                           |
  +------------------------------+-------------------------------+
                                 v
                    +------------+------------+
                    | GBDT / DNN per market   |--> P(book | p, x)   (demand curve)
                    +------------+------------+
                                 |
  Stage B: Strategy model        v
  +------------------------------+-------------------------------+
  | regress optimal price p* with asymmetric custom loss:        |
  |  booked night   -> penalize p* < price paid (left money)     |
  |  unbooked night -> penalize p* > price shown (too high)      |
  | inputs: Stage A demand signal + listing/market features      |
  +------------------------------+-------------------------------+
                                 v
  Stage C: Personalization / policy layer
  +--------------------------------------------------------------+
  | host goal (occupancy vs ADR), min/max bounds, smoothing,     |
  | guardrails -> final suggested price + reason codes           |
  +--------------------------------------------------------------+

  Rides variant: forecast-then-optimize
  +-------------+    +---------------------+    +--------------------+
  | zone history|--->| Seq. encoder (LSTM) |--->| demand D(z,t+h)    |
  | x 5-min     |    | + exogenous feats   |    | supply S(z,t+h)    |
  +-------------+    +---------------------+    | quantiles P10-P90  |
                                                +---------+----------+
                                                          v
                          +-------------------------------+---------+
                          | multiplier m = f(D/S, elasticity),      |
                          | clip [1.0, cap], spatial + temporal     |
                          | smoothing across neighbouring hexes     |
                          +-----------------------------------------+
```

**DIAGRAM 4: Variant - e-commerce pricing as reinforcement learning / contextual bandit**

```
 +----------------------+  state s_t (price, sales, stock,   +----------------------+
 | Environment:         |  comp. price, traffic), reward r_t | Agent: DRL / bandit  |
 | SKU market           |----------------------------------->| (DQN / DDPG, or      |
 | (shoppers, rivals)   |                                    |  Thompson sampling   |
 +----------+-----------+                                    |  over price arms)    |
            ^                                                +----------+-----------+
            | apply price                                               |
            |                                                           | proposed
 +----------+-----------+   clipped action a_t                          | price a_t
 | Guardrails:          |<----------------------------------------------+
 | MAP, margin floor,   |
 | max delta per day    |
 +----------+-----------+
            |
            | log (s, a, r, propensity)
            v
 +----------+-----------+   train on logs       +----------------------+
 | Replay buffer / logs |---------------------->| Offline RL pretrain  |
 | (historical prices)  |                       | + off-policy eval    |
 +----------------------+                       | (IPS / DR) gate      |
                                                | -> warm-start Agent  |
                                                +----------------------+
 reward r_t = revenue or profit, or its change vs. the last period (Alibaba [13])
```

## Components often missed

| Component | Why it matters | Ref |
|---|---|---|
| Revenue / booking forecasting platform | Market-level forecasts feed pricing and finance planning; separate from per-listing models | [23] |
| Causal impact measurement (synthetic control / ML counterfactuals) | Pricing changes cannot always be randomized; estimate effect from observational data | [24] |
| Variance reduction / heterogeneous effects on flat A/B results | Pricing tests are noisy; heterogeneous-effect models find segment-level lift hidden in a flat average | [25] |
| Supply levers beside price (driver pay, incentives) | Balance can be restored on the supply side, not only by raising price | [16] |
| Price anomaly detector | Catches bad multipliers or stale zone state before customers see them | [7][11] |
| Image/text price suggestion for new items | Cold start when no booking or sales history exists | [22] |

## Data & labels

| Source | Label / use | Pitfall |
|---|---|---|
| Quote log (price shown, context) | Conversion at shown price (booked=1) | Only one price observed per context: no counterfactual |
| Bookings / trips | Revenue, accepted price | Survivorship: unbooked nights look "too expensive" |
| Unbooked nights | Upper bound signal for strategy loss [2] | Missing-not-at-random (host blocked, low demand) |
| Randomized price tests / explore traffic | Unbiased elasticity estimates | Cost of exploration, fairness concerns |
| Supply (driver GPS, availability calendars) | Supply forecast targets | Supply reacts to price (endogenous) |
| External (events, weather, holidays, competitor prices) | Exogenous features | Sparse, late-arriving |

## Features

| Group | Examples |
|---|---|
| Spatial | H3 hex id, neighbour demand, distance to airport/venue [6] |
| Temporal | hour, DOW, holiday, lead time (days to check-in), seasonality |
| Real-time marketplace | open requests, idle drivers, ETA, acceptance rate, search volume (last 5-60 min) |
| Item / listing | beds, location, review score, amenities, historical ADR, occupancy |
| Market | competitor/comparable prices, market occupancy, price index |
| User (careful) | price sensitivity segment; often excluded for fairness / legal reasons |
| Forecasts | predicted demand, supply, quantiles as features to the policy |

## Model choices

| Stage | Baseline | Advanced | Tradeoff |
|---|---|---|---|
| Demand forecast | Seasonal naive, Prophet/ARIMA per zone | LSTM / DeepAR global models, quantile outputs [3][14] | Global DL handles cold zones and events; harder to debug |
| Price response | Logistic regression on log price | GBDT / DNN with monotonic constraint in price | Monotonicity keeps demand curve sane |
| Elasticity | Log-log regression per segment | Causal ML (DML, uplift), randomized tests [8] | Confounding: high price coincides with high demand |
| Price decision | Rule: m = clip(D/S) | Constrained optimization over demand curve; strategy regression [2] | Rules are explainable; optimizers need guardrails |
| Exploration | Fixed small random price jitter | Contextual bandit / DRL [13] | Bandit regret vs. learning speed; trust |

## Loss functions

| Model | Loss |
|---|---|
| Booking probability | Binary cross-entropy (+ calibration) |
| Demand forecast | MSE / MAPE / quantile (pinball) loss for P10-P90 |
| Strategy (Airbnb) | Asymmetric hinge-like loss: booked nights penalize under-pricing, unbooked nights penalize over-pricing [2] |
| RL policy | Expected discounted revenue/profit; TD loss (DQN) or policy gradient [13] |

## Offline metrics

| Metric | Use |
|---|---|
| MAPE / WAPE / pinball loss | Forecast quality per zone x horizon |
| AUC, log loss, calibration | Booking-probability model |
| Price Decrease Recall (PDR) | Of unbooked nights, share where model suggested a lower price [2] |
| Booking Regret | How far below booked price the suggestion went (money left on table) [2] |
| Counterfactual revenue (IPS / DR estimators) | Off-policy value of a new pricing policy |
| Backtest in simulator | Supply/demand balance, wait times under replayed demand |

## Online metrics & A/B

| Metric | Notes |
|---|---|
| Revenue / GMV / profit per session | Primary business objective |
| Conversion (quote -> book), occupancy, ADR | Airbnb host outcomes; host adoption of suggestions [1] |
| Rider wait time / ETA, fulfilment rate, unfulfilled requests | Rides marketplace health |
| Driver utilization and earnings | Supply-side health |
| Long-term retention, complaints, price-shock rate | Guardrail metrics |
| Host adoption of price suggestions | Share of nights where hosts accept / enable suggestions [1] |
| Experiment design | Marketplace interference: use switchback (city x time slices) or geo-randomization, not user-level A/B |

## Serving & latency budget (rides, typical)

| Step | Budget |
|---|---|
| Gateway + auth | 10 ms |
| Feature fetch (zone state, forecasts) from Redis | 10 ms |
| Candidate grid + constraints | 5 ms |
| Model scoring (or read precomputed zone multiplier) | 20 ms |
| Policy / smoothing / logging | 10 ms |
| Total p99 | ~60-100 ms |
| Batch path (stays / e-com) | Nightly Spark job writes price per listing-night to KV store |

## Monitoring

| Signal | Alert |
|---|---|
| Price distribution per market | Sudden jumps, multiplier stuck at cap |
| Forecast error drift | MAPE above threshold per city |
| Feature freshness | Stream lag > 2 min |
| Conversion vs. expected | Calibration drift |
| Price anomalies | Erroneous or stale prices per region; [11] files "Pricing at Lyft" [7] under anomaly detection |
| Fairness / legal | Prices in protected areas, emergency-state caps |

## Failure modes

| Failure | Mitigation |
|---|---|
| Feedback loop: price changes demand, model learns its own policy | Exploration traffic, causal estimation, logged propensities |
| Oscillation (surge on/off, drivers chase zones) | Temporal + spatial smoothing, hysteresis |
| Extreme events (concerts, storms, emergencies) | Event features, caps, manual overrides [3] |
| Cold-start listing / new city | Hierarchical / global models, comparable listings |
| Data leakage (future bookings in features) | Point-in-time joins, time-based split |
| Adversarial supply (drivers going offline to trigger surge) | Anomaly detection on supply, use forecasts not raw counts |
| PR / trust damage | Reason codes, caps, transparency to hosts/riders |

## Interviewer follow-ups

- How do you estimate price elasticity when price is confounded with demand?
- You only observe the outcome at the price you showed; how do you build labels for "optimal price"?
- Why might a custom asymmetric loss beat plain regression on booked prices?
- How do you A/B test pricing in a two-sided marketplace with interference (switchback vs. geo)?
- How do you prevent surge oscillation and drivers gaming zones?
- What granularity for zones and time windows, and why hexagons?
- How do you handle a sudden extreme event or a public emergency (price gouging laws)?
- Forecast-then-optimize vs. end-to-end RL: tradeoffs and safety?
- How do you evaluate a new policy offline before exposing users (IPS, simulators)?
- How do you cold-start pricing for a new listing, new SKU or new city?
- How do you incorporate host/seller preferences and bounds into the model output?
- Would you personalize price per user? Discuss fairness, legal and trust issues.
- How fresh must features be, and what happens if the stream is delayed?
- How would you explain a price to a host or rider?
- When is raising driver pay (supply lever) better than raising rider price?
- How do you measure pricing impact when you cannot randomize (synthetic control, ACE-style counterfactuals)?
- Upfront fares: how do ETA/distance prediction errors propagate into price and who absorbs them?
- E-commerce: how do you avoid price wars or tacit collusion when competitors also run repricing bots?

## Portal comparison

| Portal | Framing | Core model | Distinctive emphasis |
|---|---|---|---|
| Airbnb engineering [1][2] (cited via Khang Pham index [10]) | Suggest listing-night price | Booking prob. -> strategy regression -> personalization | Custom asymmetric loss; PDR and booking regret metrics |
| Uber engineering [3][4][5] | Zone surge multiplier | Demand forecasting (LSTM) + supply/demand ratio | Extreme events; H3 hex spatial index; forecasting platform |
| Lyft engineering [7][8] | Ride pricing system | Forecasts + causal models | Causal forecasting of lever effects; post indexed under anomaly detection [11] |
| Alibaba paper [13] (via eugeneyan index [11]) | Tmall SKU pricing as MDP | DQN / DDPG | Reward design (difference of revenue conversion), offline pretraining |
| Generic MLSD portals (Exponent, Educative, IGotAnOffer, Halina) [17][19]-[21]; ByteByteGo has no pricing chapter [18] | Apply the generic framework: forecast + business objective | GBDT baseline -> DL | Requirements, metrics, serving, monitoring (no pricing walkthrough confirmed) |

## References

1. "Learning Market Dynamics for Optimal Pricing", Airbnb Engineering & Data Science (Medium), 2018. https://medium.com/airbnb-engineering/learning-market-dynamics-for-optimal-pricing-97cffbcc53e3
2. Ye, P. et al., "Customized Regression Model for Airbnb Dynamic Pricing", KDD (ACM), 2018. https://dl.acm.org/doi/10.1145/3219819.3219830
3. "Engineering Extreme Event Forecasting at Uber with Recurrent Neural Networks", Uber Engineering Blog, 2017. https://eng.uber.com/neural-networks/
4. "Forecasting at Uber: An Introduction", Uber Engineering Blog, 2018. https://eng.uber.com/forecasting-introduction/
5. "Meet Michelangelo: Uber's Machine Learning Platform", Uber Engineering Blog, 2017. https://eng.uber.com/michelangelo-machine-learning-platform/
6. "H3: Uber's Hexagonal Hierarchical Spatial Index", Uber Engineering Blog, 2018. https://www.uber.com/blog/h3/
7. "Pricing at Lyft", Lyft Engineering Blog, 2022. https://eng.lyft.com/pricing-at-lyft-8a4022065f8b
8. "Causal Forecasting at Lyft (Part 1)", Lyft Engineering Blog, 2022. https://eng.lyft.com/causal-forecasting-at-lyft-part-1-14cca6ff3d6d
9. alirezadir, "Machine-Learning-Interviews" (MLSD notes), GitHub, 2024. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/ml-system-design.md
10. Khang Pham, "machine-learning-interview" applied ML index, GitHub, 2021. https://github.com/khangich/machine-learning-interview/blob/master/appliedml.md
11. Eugene Yan, "applied-ml" (papers and tech blogs on ML in production), GitHub, 2023. https://github.com/eugeneyan/applied-ml
12. Chip Huyen, "machine-learning-systems-design", GitHub, 2019. https://github.com/chiphuyen/machine-learning-systems-design
13. Liu, J. et al., "Dynamic Pricing on E-commerce Platform with Deep Reinforcement Learning", arXiv, 2019. https://arxiv.org/abs/1912.02572
14. Salinas, D. et al., "DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks", arXiv, 2017. https://arxiv.org/abs/1704.04110
15. "The history of Amazon's forecasting algorithm", Amazon Science, 2021. https://www.amazon.science/latest-news/the-history-of-amazons-forecasting-algorithm
16. "Managing Supply and Demand Balance Through Machine Learning", DoorDash Engineering Blog, 2021. https://doordash.engineering/2021/06/29/managing-supply-and-demand-balance-through-machine-learning/
17. Exponent, "ML System Design" course introduction, tryexponent.com, 2024. https://www.tryexponent.com/courses/ml-system-design/mlsd-intro
18. Ali Aminian & Alex Xu, "Machine Learning System Design Interview", ByteByteGo, 2023. https://bytebytego.com
19. Educative, "Grokking the Machine Learning Interview", educative.io, 2023. https://www.educative.io/courses/grokking-the-machine-learning-interview
20. IGotAnOffer, "Machine learning system design interview guide", igotanoffer.com, 2024. https://igotanoffer.com/en/advice/machine-learning-system-design-interview
21. Patrick Halina, "ML Systems Design Interview Guide", patrickhalina.com, 2021. http://patrickhalina.com/posts/ml-systems-design-interview-guide/
22. Alibaba, "Vision-based Price Suggestion for Online Second-hand Items", arXiv, 2020. https://arxiv.org/abs/2012.06009
23. Jerry Chu, "Airbnb's data platform of revenue forecasting", Medium (listed under Airbnb in [10]). https://medium.com/@jerry.chu/airbnbs-data-platform-of-revenue-forecasting-2e95a01122e6
24. "Artificial Counterfactual Estimation (ACE): Machine Learning-Based Causal Inference at Airbnb", Airbnb Engineering (Medium), 2022. https://medium.com/airbnb-engineering/artificial-counterfactual-estimation-ace-machine-learning-based-causal-inference-at-airbnb-ee32ee4d0512
25. "Leveraging Causal Modeling to Get More Value from Flat Experiment Results", DoorDash Engineering Blog, 2020. https://doordash.engineering/2020/09/18/causal-modeling-to-get-more-value-from-flat-experiment-results/
