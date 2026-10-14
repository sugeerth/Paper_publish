# 14. Design Payment Fraud Detection (Stripe Radar / PayPal) with Real-Time Scoring

> Sourcing note (review, 2026-10-14): GitHub-hosted references [16][17][19][20][21] were
> re-fetched and their claims confirmed. Company blog domains (stripe.com, medium.com,
> eng.lyft.com, uber.com, portals) did not resolve from the review sandbox; refs [1]-[15],
> [25]-[27] were confirmed to exist with these exact titles/URLs via the curated index [20],
> and their content is summarized from the posts themselves, not re-read in this pass.
> Educative/Exponent/Hello Interview/Halina [18][22]-[24] were not re-fetched. Rows marked
> "typical assumption" are interview-standard estimates, not quoted numbers.

## Company tags

| Company | Evidence (built it = public engineering post; not proof of interview question) | Ref |
|---|---|---|
| Stripe | Built it: Radar, ML scorer on every card payment; similarity clustering for rings | [1][2][3] |
| PayPal | Built it: fraud-model CI/CD + shadow platform, real-time graph DB, multi-domain model | [4][5][6] |
| Uber, Lyft, Grab, Netflix, Wayfair, BlaBlaCar | Built it: production fraud/risk ML posts indexed in [20] | [8]-[15][25][26] |
| Capital One | Adjacent only: ML for anti-money-laundering, not card-fraud scoring | [7] |
| Generic "Fraud detection system" | Listed under "Harmful content / Spam detection" in sample MLSD questions | [16] |
| Chip Huyen exercise | "each record is labelled as fraudulent or safe, how would you build a fraud detection algorithm?" | [17] |

Removed in review: "Amazon" and "Airbnb" (no fetched source supports either tag).

Portals that cover the problem:

| Portal | Coverage | Ref |
|---|---|---|
| alirezadir/Machine-Learning-Interviews | "Fraud detection system" under Other > Harmful content / Spam (verified) | [16] |
| chiphuyen/machine-learning-systems-design | Exercise 2: labelled fraudulent vs safe records (verified) | [17] |
| khangich/machine-learning-interview | "Fraud detection (TBD)" heading, no content (verified) | [19] |
| Engineer1999 curated case studies | ~20 fraud/risk/payments posts indexed (verified) | [20] |
| ByteByteGo ML SDI (Aminian & Xu) | No fraud chapter; closest is ch. 5 Harmful Content Detection (verified) | [21] |
| Educative "Grokking the ML SD Interview" | Reported to use fraud detection in its 6-step framework (not re-fetched) | [18] |
| Exponent, Hello Interview, Patrick Halina | Generic MLSD frameworks only; no fraud-specific page claimed | [22][23][24] |

## Requirements & scale

| Dimension | Value | Source |
|---|---|---|
| Functional | Score each payment/authorization; return allow / review / 3DS step-up / block | [1][2] |
| Functional | Merchant-defined rules on top of score; explanations for decisions | [1][2] |
| Functional | Analyst review queue; feedback (disputes, manual labels) into training | [4][9] |
| Latency | Score in the authorization path, well under ~100 ms end to end | [1] |
| Throughput | 5k-50k payments/s peak (Black Friday 3-5x average) | typical assumption |
| Feature count | 1,000+ signals per payment at Stripe Radar | [1] |
| Entities | 100M-1B cards, devices, emails, IPs; millions of merchants | typical assumption |
| Fraud base rate | 0.05%-0.5% of transactions (extreme imbalance) | typical assumption |
| Label delay | Chargebacks arrive days to ~90+ days after the charge | [1][3] |
| Availability | 99.99%; fail-open or fail-to-rules if model is down | typical assumption |
| Cost asymmetry | Missed fraud = amount + fee; false decline = lost revenue + churn | [3][6] |

## DIAGRAM 1: Online real-time scoring path

**Figure 14.1 - Authorization-time fraud scoring, steps 1-10 (request -> features -> model -> policy -> action)**

```
+-----------+  1 charge    +-------------------+  2 ids   +---------------------------+
| Merchant  |------------->| Payment API /     |--------->| Enrichment: BIN, geo-IP,  |
| checkout  |<-------------| auth gateway      |          | device fingerprint, proxy |
+-----------+ 10 allow /   +-------------------+          +---------------------------+
               3DS / block     |           ^                            |
                   3 request   |           |  9 decision + reasons      | 4 attributes
                               v           |                            v
        +------------------------------------------------------------------------+
        | Fraud Scoring Service: parallel fan-out, p99 < ~100 ms; timeout->rules |
        +------------------------------------------------------------------------+
            |              |  ^                |  ^                       |
     5 read |       6 feats|  | 7 score  score |  | 8 action        5 read|
            v              v  |                v  |                       v
+-------------------+ +---------------+ +-----------------+ +-----------------------+
| Online feature    | | Model server  | | Policy engine   | | Real-time graph store |
| store (KV):       | | DNN or GBDT   | | merchant rules, | | card-device-IP-email  |
| velocity 1m/1h/24h| | -> p(fraud);  | | cost-aware      | | links, shared-entity  |
| per card/IP/device| | shadow models | | thresholds      | | and ring features     |
+-------------------+ +---------------+ +-----------------+ +-----------------------+
          ^                   ^                  |
          | windowed aggs     | load weights     | review
          |                   |                  v
+-------------------+ +---------------+ +-----------------+
| Stream processor  | | Model registry| | Manual review   |---> analyst labels
| (Flink)           | | (Fig 14.2)    | | queue           |     (to Fig 14.2)
+-------------------+ +---------------+ +-----------------+
          ^
          | auth requests, scores, decisions (published by gateway + scoring svc)
+--------------------------------------------------+     +--------------------------+
| Event bus (Kafka)                                |---->| Data lake: features and  |
+--------------------------------------------------+     | decisions as of auth time|
                                                         +--------------------------+
```

## DIAGRAM 2: Offline training and data pipeline

**Figure 14.2 - Delayed-label pipeline with feedback loop**

```
+---------------+  +----------------+  +----------------+  +------------------+
| Scoring logs  |  | Disputes /     |  | Analyst review |  | Issuer / network |
| (features at  |  | chargebacks    |  | labels, refunds|  | fraud reports    |
| decision time)|  | (T+7..T+120d)  |  | (T+hours)      |  | (TC40/SAFE-like) |
+---------------+  +----------------+  +----------------+  +------------------+
        |                  |                  |                     |
        +------------------+--------+---------+---------------------+
                                    v
                     +-------------------------------+
                     | Label join on charge_id       |
                     | - maturity window (e.g. 60d)  |
                     | - fraud vs service dispute    |
                     | - blocked txns = unlabeled    |
                     +-------------------------------+
                                    |
                                    v
+-----------------+       +-------------------------------+
| Offline feature |------>| Point-in-time correct join    |
| store (history) |       | (no leakage from the future)  |
+-----------------+       +-------------------------------+
                                    |
                                    v
                     +-------------------------------+
                     | Sampling: downsample negatives|
                     | + weights; add exploration    |
                     | holdout (small % not blocked) |
                     +-------------------------------+
                                    |
                                    v
                     +-------------------------------+
                     | Train (daily/weekly): GBDT,   |
                     | DNN; time-based split         |
                     +-------------------------------+
                                    |
                                    v
                     +-------------------------------+  fail
                     | Evaluate: PR-AUC, recall @    |-------> back to features
                     | fixed FPR, $ saved, per-slice |
                     +-------------------------------+
                                    | pass
                                    v
                           +-----------------+    +-----------------+
                           | Model registry  |--->| Shadow mode     |
                           | (versioned)     |    | (score, no      |
                           +-----------------+    | action) [4]     |
                                                  +-----------------+
                                                           |
                                                           v
                           +-----------------+    +-----------------+
                           | Online model    |<---| Canary / A-B +  |
                           | server (14.1)   |    | threshold recal.|
                           +-----------------+    +-----------------+
                                    |
                                    v
          feedback loop: new decisions -> scoring logs -> labels (top of figure)
```

## DIAGRAM 3: Model architecture

**Figure 14.3 - Hybrid tabular + sequence + graph fraud classifier**

```
 Inputs per payment
+--------------------+ +--------------------+ +-------------------+ +-------------+
| Dense numeric      | | Categorical ids    | | Recent event seq. | | Graph nbhd  |
| amount, velocity,  | | BIN, MCC, country, | | last N txns/logins| | card-device-|
| ratios, z-scores,  | | merchant, device,  | | of card & device  | | IP-email    |
| time since last    | | email domain       | | (time-ordered)    | | 2-hop       |
+--------------------+ +--------------------+ +-------------------+ +-------------+
          |                      |                      |                  |
          v                      v                      v                  v
   +-------------+      +----------------+     +----------------+  +-------------+
   | normalize / |      | embedding      |     | GRU/Transformer|  | GNN / graph |
   | log-bucket  |      | tables         |     | encoder        |  | embeddings  |
   +-------------+      +----------------+     +----------------+  +-------------+
          |                      |                      |                  |
          +----------+-----------+----------+-----------+------------------+
                     |                      |
                     v                      v
          +---------------------+   +------------------------+
          | GBDT leaf / score   |   | Concatenate -> deep    |
          | ("wide" part)       |-->| residual MLP blocks    |
          +---------------------+   +------------------------+
                                               |
                         +---------------------+--------------------+
                         v                     v                    v
                 +---------------+    +----------------+    +----------------+
                 | Head 1:       |    | Head 2:        |    | Head 3 (opt):  |
                 | p(fraud       |    | p(dispute any) |    | p(ATO / stolen |
                 | chargeback)   |    | aux task       |    | creds)         |
                 +---------------+    +----------------+    +----------------+
                         |
                         v
          calibrated risk score 0-100 + SHAP-style top reasons -> policy engine
```

## DIAGRAM 4: Real system - Stripe Radar model evolution (as described in [1])

**Figure 14.4 - From hand-built ensemble to a single deep network**

```
+--------------------+      +----------------------------+      +-------------------+
| Early: logistic    | ---> | Wide & Deep ensemble:      | ---> | Pure DNN (ResNeXt |
| regression + rules |      | XGBoost (wide) + DNN (deep)|      | inspired), more   |
|                    |      | hard to retrain/extend     |      | features, faster  |
+--------------------+      +----------------------------+      | training          |
                                                                +-------------------+
                                                                         |
   Network effect: signals from millions of businesses on the network    |
   (card seen before? at which merchants? disputed elsewhere?)  ---------+
                                                                         v
                                   Risk score + "risk insights" explanations
                                   + merchant Radar rules (allow/review/block)
```

## Data & labels

| Source | Label / use | Caveat |
|---|---|---|
| Fraudulent chargebacks | Positive label (strongest) | Delay of weeks; friendly fraud noise |
| Early fraud warnings / issuer reports | Faster positive signal | Coverage varies by network |
| Manual review outcomes | Positive/negative | Biased to borderline scores |
| Refunds flagged as fraud by merchant | Weak positive | Merchant-dependent |
| Matured non-disputed charges | Negative after window | Blocked charges never get labels |
| Exploration holdout (tiny % allowed) | Unbiased estimate of true fraud rate | Costs real money |
| Weak supervision (heuristic labeling functions) | Extra noisy labels for unlabeled/new patterns [27] | Label noise; needs denoising |

## Features

| Group | Examples |
|---|---|
| Transaction | amount, currency, MCC, card-present flag, time of day, amount vs merchant avg |
| Velocity (stream) | txns per card/IP/device/email in 1m, 1h, 24h; distinct cards per device |
| Identity / device | device fingerprint, browser, IP geo vs billing country, proxy/VPN flag |
| Card / issuer | BIN country, card age on network, prior declines, prior disputes |
| Network / graph | shared device/IP with known fraud, connected-component size, fraud-ring cluster [3][12] |
| Behavioral | checkout dwell time, typing speed, copy-paste of card number |
| Merchant | merchant dispute rate, vertical, account age |

## Model choices

| Stage | Model | Pros | Cons |
|---|---|---|---|
| Baseline | Rules + logistic regression | Explainable, fast | Low recall, brittle |
| Standard | GBDT (XGBoost/LightGBM) | Strong on tabular, fast CPU inference | Weak on raw sequences/ids |
| Hybrid | Wide & Deep (GBDT + DNN) | Combines memorization + generalization | Two systems to maintain [1] |
| Advanced | Single deep net (residual MLP + embeddings) | Scales with features, one pipeline [1][11] | Needs more data, explainability tooling |
| Sequence | RNN/Transformer over event history; customer-journey embeddings [25] | Captures behavior change, ATO | Latency, state storage |
| Graph | GNN / graph features, similarity clustering | Catches rings, mule networks [3][6][12] | Real-time graph cost |
| Unsupervised | Isolation forest, autoencoder, graph anomaly | Novel attacks, no labels [10][13] | High FP, hard to tune |

## Loss

| Option | When |
|---|---|
| Weighted binary cross-entropy | Default; weight positives or by dollar amount |
| Focal loss | Extreme imbalance, many easy negatives |
| Cost-sensitive / amount-weighted loss | Optimize dollars, not counts |
| Multi-task BCE (fraud, dispute, ATO) | Shared tower, sparse fraud labels |
| Calibration (Platt/isotonic) after training | Thresholds must map to stable probabilities |

## Offline metrics

| Metric | Why |
|---|---|
| PR-AUC (not ROC-AUC) | Base rate under 1% |
| Recall at fixed FPR / block rate | Matches business constraint (e.g. block 0.5%) |
| Dollar-weighted recall / $ fraud caught | Large tickets matter more |
| Calibration (ECE, reliability) | Thresholds and merchant rules depend on it |
| Per-slice metrics | Country, MCC, new vs existing card, merchant size |
| Time-based backtest | Train on t, test on t+k; detects decay |

## Online metrics & A-B

| Metric | Direction |
|---|---|
| Fraud chargeback rate (bps of volume) | Down |
| False-positive / good-user decline rate | Down [6] |
| Authorization / conversion rate | Up |
| Manual review volume & precision | Down / up |
| Net $ = fraud saved - lost good revenue - review cost | Up |
| A-B design | Randomize by merchant or card; long horizon because labels lag; use early-warning proxies |

## Serving & latency budget

| Step | Budget (typical assumption) |
|---|---|
| Enrichment (BIN, geo-IP, device) | 5-10 ms |
| Online feature store reads (KV, parallel) | 5-15 ms |
| Graph lookups (1-2 hop, cached) | 5-15 ms |
| Model inference (GBDT CPU or small DNN) | 5-20 ms |
| Policy / rules engine | 1-5 ms |
| Total p99 | under ~100 ms [1] |
| Fallback | Timeout -> rules-only or last cached score; fail-open for low amounts |

## Monitoring

| Signal | Action |
|---|---|
| Score distribution / block-rate drift | Alert; recalibrate thresholds |
| Feature null-rate, freshness lag of stream aggregates | Page on-call; fall back to rules |
| Early-warning fraud rate by segment | Detect new attack campaigns |
| Analyst-confirmed precision on review queue | Track quality daily |
| Shadow vs prod disagreement | Gate model promotion [4] |
| Per-merchant spikes | Auto-tighten rules for that merchant |
| Attack-campaign detector (time-series anomaly on fraud signals) | Propose rules; analyst approves before deploy (human in loop) [8] |
| Entity-level anomaly watch (riders/drivers/cards) | Flag risky entities missed by txn model [9] |

## Failure modes

| Failure | Mitigation |
|---|---|
| Label delay -> stale model | Proxy labels (early warnings, reviews); frequent retrains |
| Selection bias (blocked txns unlabeled) | Exploration holdout, counterfactual/IPS weighting |
| Adversarial adaptation | Fast retrain, rules for hotfixes, graph/ring features |
| Train/serve skew in velocity features | Same feature definitions offline and online, point-in-time joins |
| Feedback loop from own decisions | Log propensity; keep randomized holdout |
| Friendly fraud mislabeled as fraud | Separate dispute reason codes; multi-task heads |
| Cold start (new merchant/card) | Network-level priors, population features |
| Explainability / regulation | Reason codes, SHAP, merchant-facing insights [1][15] |

## Interviewer follow-ups

- Labels arrive 30-90 days late: how do you train and evaluate on recent data?
- You never see the outcome of blocked transactions: how do you estimate true recall?
- How do you choose the threshold, and does it differ per merchant or per amount?
- ROC-AUC is 0.99 but the business is unhappy: why, and what metric do you use instead?
- How do you compute "transactions per card in the last hour" consistently online and offline?
- How do you detect a fraud ring of many new cards using one device or IP?
- Fraudsters change tactics within days: how fast can you retrain and ship safely?
- How do you combine merchant rules with the ML score without double counting?
- What happens if the feature store or model server times out in the auth path?
- How do you run an A-B test when the outcome takes two months?
- How do you explain a decline to a merchant or regulator?
- How do you handle a new merchant with no history (cold start)?
- When do you add a 3DS step-up instead of a hard block?
- Would you use a GNN online, or precompute graph features? Why?
- How do you separate friendly fraud from true fraud in labels?
- A card-testing attack sends thousands of tiny charges from one merchant: which features and rules catch it?
- How do analysts and auto-generated rules coexist with the model, and who can ship a rule in minutes? [8]
- How would you reuse one model across products/domains without hurting good-user approval? [6]

## Portal comparison

| Portal | Framing | Model | Distinctive emphasis |
|---|---|---|---|
| Educative Grokking MLSD [18] | Reported fraud example in 6-step framework (not re-fetched) | Classical ML -> deep models | Data strategy, imbalance, offline vs online eval |
| Chip Huyen exercises [17] | Short open prompt on labelled fraud/safe records | Open | Imbalance, label quality, metric choice |
| alirezadir MLI [16] | Grouped with spam / harmful content detection | Classifier template | Reuse of detection template |
| Khang Pham repo [19] | Placeholder; framework from ads/feed chapters applies | GBDT / DNN | Feature engineering at scale |
| Stripe Radar blog [1] | Production system description | LR -> Wide&Deep -> pure DNN | Network data, 1000+ signals, sub-100 ms, explanations |
| PayPal tech blog [4][5][6] | Production deployment | Ensemble + graph | Shadow platform, real-time graph, reduce good-user declines |
| Uber Project RADAR [8] | Production attack detection | Time-series anomaly + rules | Early detection, human-in-the-loop rule generation |

## References

1. Stripe, "How we built it: Stripe Radar", stripe.com blog, 2023. https://stripe.com/blog/how-we-built-it-stripe-radar
2. Stripe, "A primer on machine learning for fraud detection", stripe.com guides. https://stripe.com/en-mx/guides/primer-on-machine-learning-for-fraud-protection
3. Stripe, "Similarity clustering to catch fraud rings", stripe.com blog. https://stripe.com/blog/similarity-clustering
4. PayPal Tech, "Deploying Large-scale Fraud Detection Machine Learning Models at PayPal", Medium. https://medium.com/paypal-tech/machine-learning-model-ci-cd-and-shadow-platform-8c4f44998c78
5. PayPal Tech, "How PayPal Uses Real-time Graph Database and Graph Analysis to Fight Fraud", Medium. https://medium.com/paypal-tech/how-paypal-uses-real-time-graph-database-and-graph-analysis-to-fight-fraud-96a2b918619a
6. PayPal Tech, "Multi-Domain Fraud Detection While Reducing Good User Declines, Part I", Medium. https://medium.com/paypal-tech/multi-domain-fraud-detection-while-reducing-good-user-declines-part-i-e8bba5b07da8
7. Capital One Tech, "How Machine Learning Can Help Fight Money Laundering", capitalone.com. https://www.capitalone.com/tech/machine-learning/how-machine-learning-can-help-fight-money-laundering/
8. Uber Engineering, "Project RADAR: Intelligent Early Fraud Detection System with Humans in the Loop", uber.com. https://www.uber.com/en-GB/blog/project-radar-intelligent-early-fraud-detection/
9. Uber Engineering, "Risk Entity Watch: Using Anomaly Detection to Fight Fraud", uber.com. https://www.uber.com/en-IN/blog/risk-entity-watch/
10. Grab Engineering, "Unsupervised graph anomaly detection: Catching new fraudulent behaviours", engineering.grab.com. https://engineering.grab.com/graph-anomaly-model
11. Lyft Engineering, "From shallow to deep learning in fraud", eng.lyft.com. https://eng.lyft.com/from-shallow-to-deep-learning-in-fraud-9dafcbcef743
12. Grab Engineering, "Graph for fraud detection", engineering.grab.com. https://engineering.grab.com/graph-for-fraud-detection
13. Lyft Engineering, "Fingerprinting fraudulent behavior", eng.lyft.com. https://eng.lyft.com/fingerprinting-fraudulent-behavior-6663d0264fad
14. Netflix Tech Blog, "Machine Learning for Fraud Detection in Streaming Services", Medium. https://netflixtechblog.medium.com/machine-learning-for-fraud-detection-in-streaming-services-b0b4ef3be3f6
15. Wayfair Tech, "Explainable Fraud Detection", aboutwayfair.com. https://www.aboutwayfair.com/tech-innovation/explainable-fraud-detection
16. alirezadir, "ML System Design" (ml-system-design.md), GitHub, 2025. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/ml-system-design.md
17. C. Huyen, "Machine Learning Systems Design - Exercises", GitHub. https://github.com/chiphuyen/machine-learning-systems-design/blob/master/content/exercises.md
18. Educative, "The 6-step ML SD framework" (Grokking the ML System Design Interview), educative.io, 2025. https://www.educative.io/courses/grokking-the-machine-learning-system-design-interview/the-6-step-ml-sd-framework
19. K. Pham, "machine-learning-interview", GitHub. https://github.com/khangich/machine-learning-interview
20. Engineer1999, "A Curated List of ML System Design Case Studies", GitHub, 2024. https://github.com/Engineer1999/A-Curated-List-of-ML-System-Design-Case-Studies
21. ByteByteGo, "ml-bytebytego: references for Machine Learning System Design Interview", GitHub, 2023. https://github.com/ByteByteGoHq/ml-bytebytego
22. Exponent, "Introduction to the ML System Design Interview", tryexponent.com, 2025. https://www.tryexponent.com/courses/ml-system-design/mlsd-intro
23. Hello Interview, "ML System Design in a Hurry: Delivery Framework", hellointerview.com, 2025. https://www.hellointerview.com/learn/ml-system-design/in-a-hurry/delivery
24. P. Halina, "ML Systems Design Interview Guide", patrickhalina.com, 2021. http://patrickhalina.com/posts/ml-systems-design-interview-guide/
25. Wayfair Tech, "Introducing Melange: A Customer Journey Embedding System for Improving Fraud and Policy Abuse Detection", aboutwayfair.com. https://www.aboutwayfair.com/careers/tech-blog/introducing-melange-a-customer-journey-embedding-system-for-improving-fraud-and-scam-detection
26. BlaBlaCar Engineering, "How we built our machine learning pipeline to fight fraud at BlaBlaCar - Part 2", Medium. https://medium.com/blablacar/how-we-built-our-machine-learning-pipeline-to-fight-fraud-at-blablacar-part-2-476335f459b4
27. Swiggy Bytes, "DeFraudNet: An End-to-End Weak Supervision Framework to Detect Fraud in Online Food Delivery", bytes.swiggy.com. https://bytes.swiggy.com/defraudnet-an-end-to-end-weak-supervision-framework-to-detect-fraud-in-online-food-delivery-22366ddce461
