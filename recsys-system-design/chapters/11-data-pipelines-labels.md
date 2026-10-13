# 11. Training Data and Labeling: Impression Logs, Label Joins, Delayed Feedback, Negatives and Pipelines

> Research note: when this chapter was compiled, live web search and page fetching were
> unavailable (search quota exhausted, egress blocked). Everything below comes from
> primary sources (papers, official engineering blogs, open-source READMEs) as the author
> already knew them, not from a fresh crawl. Numbers marked "approx." should be checked
> against the cited source before they are quoted.

## Overview

People usually picture a recommender as a model, but in production most of the work and most of the bugs sit in the data that trains it. Every large public system covered in this atlas runs roughly the same loop. It serves a ranked list and logs what was shown (the impression) along with the features used at that moment. It then waits for user feedback such as clicks, watch time, purchases or hides, joins that feedback to the impression to make a labeled example, chooses or creates negatives, reweights, and feeds the result to a batch or streaming trainer. Each step forces a design decision with real consequences:

- **What counts as a positive?** A click, a "long click", watch time, a conversion within 30 days, or a survey answer.
- **When is a label final?** Conversions can arrive days after the click, and the model has to train before then.
- **What counts as a negative?** An impression that wasn't clicked, a random item, an in-batch item, or a hard negative.
- **Where do the features come from?** Logged at serving time, or reconstructed later ("time travel").
- **How fresh does the data have to be?** Daily batch, hourly, or minute-level streaming.

This chapter collects the public designs that answer these questions. It starts with impression logging and the label join (Facebook, Google Photon, ByteDance Monolith, Netflix). It then covers delayed feedback and conversion attribution (Criteo, Twitter, Alibaba, CyberAgent), negative sampling (YouTube, Google Play, Facebook Search, Pinterest, Airbnb), sample weighting and label definitions (YouTube, Twitter/X, Yahoo, Kuaishou, Pinterest), and the large data-ingestion pipelines behind them (Meta DSI, TFX/Beam, Spotify Scio). Feature stores are covered in Chapter 8 and training infrastructure in Chapter 9. Here the focus is the *example*: how it is defined, labeled, sampled, weighted and delivered.

```
                 THE CANONICAL TRAINING-DATA LOOP
  +-----------+   request    +-------------------+  ranked list  +--------+
  |  Client   |------------->|  Serving stack    |-------------->| Client |
  +-----------+              | (retrieval+rank)  |               +--------+
                             +---------+---------+                   |
                                       | impression log              | engagement
                                       | (req_id, item, pos,         | events
                                       |  features, model_ver)       | (click, dwell,
                                       v                             |  convert...)
                             +-------------------+                   v
                             |  Log bus (Kafka / |<--------- event log bus
                             |  Scribe / PubSub) |
                             +---------+---------+
                                       v
                      +----------------------------------+
                      |  LABEL JOINER  (stream or batch) |
                      |  key = req_id/impression_id      |
                      |  wait window W, dedup, attribution|
                      +----------------+-----------------+
                                       v
                +---------------------------------------------+
                | Sampling: negative downsampling, hard negs, |
                | per-user caps, importance / label weights    |
                +----------------------+----------------------+
                                       v
               +-------------------+       +-----------------------+
               | Training tables   |------>| Batch / online trainer|
               | (Hive/Parquet/    |       | (+ calibration fix-up)|
               |  TFRecord, Kafka) |       +-----------+-----------+
               +-------------------+                   | new model
                                                       v
                                                back to serving
```

---

## 1. Impression logging and the label join

### 1.1 Facebook Ads (2014): the Online Joiner

**Problem.** Click-through-rate (CTR) prediction for ads at Facebook's scale, at the time more than 750M daily active users and over 1M active advertisers. The model had to be retrained often because, as the paper reports, prediction quality decays measurably as the gap between the training data and the serving day grows [1].

**Architecture.** He et al. describe an *online data joiner*. Ad impressions go into one stream, keyed by a request ID, and clicks go into another. The joiner holds impressions in a window. If a click with the same key arrives within the **waiting window** the example is labeled positive, and otherwise it is emitted as a negative once the window closes. The labeled stream feeds an online trainer for the logistic-regression layer, which sits on top of boosted-decision-tree feature transforms, and the trained models are pushed back to the ranker [1].

```
  impressions --(req_id)--+
                          |   +----------------------+   labeled   +---------+
                          +-->| Online joiner        |------------>| Online  |
  clicks -----(req_id)----+   | hold until click or  |   examples  | trainer |
                              | window W expires     |             +----+----+
                              +----------------------+                  |
                                     ^                                  v
                                     |      models <-------------- ranker
```

**Key tricks and lessons.**

- **The waiting-window trade-off.** If the window is too short, real clicks arrive too late and are mislabeled as negatives, which biases CTR downward. If it is too long, examples reach the trainer late and the trainer needs more memory to buffer them. The paper treats this explicitly as a tuning knob and says the joiner needs to be protected against anomalies, for example by raising alarms and cutting the trainer off if the click stream stops [1].
- **Freshness matters.** The paper's experiments show normalized entropy getting worse as the training data falls days behind the evaluation day, which is the reason for daily or online retraining [1].
- **Negative downsampling plus re-calibration.** Negatives are kept at rate w, and predictions are then corrected with q = p / (p + (1 - p) / w), where p is the model's prediction in the downsampled space. Since then this formula has been standard in CTR systems [1].
- **Uniform subsampling** of all data: according to the paper, keeping about 10% of the data cost only about 1% in normalized entropy (approx.). That makes subsampling a cheap cost lever [1].

### 1.2 Google Photon (2013): exactly-once joins across datacenters

**Problem.** Google's ads system has to join a *query/impression* log with a *click* log continuously, across several datacenters, with exactly-once semantics, because each joined click is billed and also used for training [2].

**Architecture.** Photon joins each click to its originating query event by a unique query ID. Its core is the **IdRegistry**, a Paxos-replicated, geo-distributed key-value store holding the IDs of events that have already been joined. Every joiner checks it before emitting, so a click is never emitted twice even when several datacenters process the same click stream at once for fault tolerance. When a click arrives before its query, which is possible across datacenters, the click is retried until the query becomes visible. The paper reports throughput in the millions of joined events per minute, with end-to-end latency in the seconds range (approx.) [2].

**Lessons.**
- **Use stable join keys.** The serving system must mint a unique ID per request or impression and pass it through to every downstream event. This is the single most important logging decision.
- **Make the join idempotent.** Duplicate positives inflate CTR and corrupt calibration. Deduplication has to be part of the joiner's design, not a cleanup job added later.

### 1.3 ByteDance Monolith (2022): streaming joiner for online training

**Problem.** TikTok/Douyin-style recommenders need minute-level freshness. User interests shift within a session, and new videos have to be learnable right away [3].

**Architecture.** Monolith uses Kafka queues for (a) user actions and (b) the features logged at serving time. A **Flink online joiner** joins the two by request key and writes training examples to another Kafka queue, which the online trainer consumes. User actions can arrive long after the features, so the joiner keeps an **in-memory cache backed by an on-disk key-value store** for features still waiting for their actions. The same examples are also written to HDFS for batch training, so a single code path produces both batch and streaming data [3].

```
  serving --> [Kafka: features] --+
                                  +--> Flink joiner --> [Kafka: examples] --> online
  client  --> [Kafka: actions] ---+    (mem cache +                          trainer
                                        disk KV for                            |
                                       pending feats)                         | param sync
                                         |                                    v  (minutes)
                                          +--> HDFS (batch training)       serving PS
```

**Key tricks.**
- **Negative sampling with log-odds correction inside the joiner.** Negatives far outnumber positives, so the joiner samples them, and the bias this introduces is corrected at serving time so the prediction approximates the original distribution [3].
- **Parameter sync measured in minutes**, from the training parameter servers to the serving parameter servers. Only touched sparse IDs are pushed, and dense weights are synced less often. The paper reports that more frequent syncing measurably improves AUC [3].
- **Collisionless embedding tables** (cuckoo hashing), plus **ID frequency filtering** and **embedding expiry**. These keep the streaming tables bounded, which matters because the example stream keeps introducing new IDs [3].

### 1.4 Netflix: from "time travel" snapshots to a logged fact store

**Problem.** If features are computed offline when a new model is trained, they reflect *today's* state of the world and not the state at the moment the recommendation was made. That is label leakage plus training/serving skew.

**DeLorean (2016).** Netflix's first answer was to snapshot the online facts that features are computed from (viewing history, My List, and similar) at regular intervals. A service calls the online microservices, through the Prana sidecar, for a sampled set of contexts (profile, time, device) and stores the snapshots on S3 as Parquet. Feature generation then runs in Spark: for a given **context set** and **label set**, it "time travels" by loading the snapshot for that timestamp and running the *same* feature encoders that the online system uses [4].

**Evolution (2022, Axion fact store).** Snapshotting turned out to be expensive and covered only a sample of contexts. Netflix moved to logging the facts that were actually used *at compute time*. The online system logs the facts, Axion stores and indexes them, and offline feature generation reads the logged facts and reuses the shared feature-encoder library. The blog describes how the storage design evolved: Iceberg tables, and an EVCache layer for low-latency lookups during offline jobs [5].

**Lesson.** This is the clearest public account of the industry-wide move from **"backfill features offline" to "log and wait"**. Google's Rules of ML gives the same advice. Rule #29 says the best way to train like you serve is to save the features used at serving time and pipe them to a log for training, and notes that YouTube Home switched to serving-time feature logging and saw significant quality gains with less code complexity [6].

### 1.5 Comparison: label-join designs

| System | Join type | Join key | Waiting / attribution | Freshness | Notable trick |
|---|---|---|---|---|---|
| Facebook Ads 2014 [1] | Streaming joiner | request ID | fixed waiting window W | online LR + daily trees | negative downsampling + recalibration |
| Google Photon [2] | Streaming, multi-DC | query ID | until matching query visible | seconds (approx.) | Paxos IdRegistry for exactly-once |
| ByteDance Monolith [3] | Flink streaming | request key | cache + disk KV for late actions | minutes | in-joiner neg sampling, PS sync |
| Netflix DeLorean [4] | Batch, Spark | context set + timestamp | n/a (offline labels) | daily | snapshot-based time travel |
| Netflix Axion [5] | Batch over logged facts | logged fact keys | n/a | daily | log facts at compute time |
| Twitter delayed fb [7] | Streaming, no wait | impression | none: negatives now, positives later | real time | fake negatives + loss correction |
| Alibaba ES-DFM / DEFER [8][9] | Streaming | click / impression | short window e + duplicate | near real time | elapsed-time sampling, real negatives |

---

## 2. Delayed feedback and conversion attribution

Conversions (purchases, installs, subscriptions) can arrive minutes or weeks after the click. A trainer that waits for the full attribution window (often 7 to 30 days) gets the right labels but uses stale data. A trainer that does not wait sees **fake negatives**: clicks that will convert later but are labeled 0 for now. Four families of fixes appear in the public literature.

### 2.1 Criteo: joint conversion and delay model (Chapelle, KDD 2014)

Chapelle's paper started this line of work [10]. Each click x has a latent outcome C (whether it will ever convert). If C = 1, the conversion arrives after a delay D, modeled with an exponential distribution whose rate depends on x. At training time, a click that has not converted after elapsed time e is either a real negative or a positive whose delay exceeds e. The likelihood combines the two models:

- converted at delay d: P(C=1|x) * lambda(x) * exp(-lambda(x) d)
- not converted after elapsed e: 1 - P(C=1|x) + P(C=1|x) * exp(-lambda(x) e)

Both models are logistic/linear and are trained jointly, either by EM or by direct gradient optimization. This lets the system train on recent clicks without treating every unconverted click as a negative. The paper uses Criteo display-advertising data, and Criteo released a related conversion-log dataset for research [10].

### 2.2 Twitter: fake negatives with loss correction (Ktena et al., RecSys 2019)

**Setting.** Ads at Twitter, with continuous training of neural networks [7].

**Design.** Instead of waiting, each engagement-eligible impression is ingested **immediately as a negative**. When the positive engagement arrives, the example is ingested **again as a positive**. This is the "fake negative" pipeline. It has no waiting window and the best possible freshness, but the training distribution is biased: positives appear both as negatives and as positives. The paper derives and compares several losses that undo the bias [7]:

- **Fake-negative weighted (FN-weighted):** importance-sample each example with weights derived from the model's own p(y=1|x), so the loss matches the true distribution in expectation.
- **Fake-negative calibration (FN-calibration):** train on the biased data and invert the bias at prediction time, p = b / (1 - b), where b is the biased prediction.
- **Positive-unlabeled (PU) loss** and a **delayed-feedback loss** in the style of Chapelle's model.

According to the paper, FN-weighted and FN-calibration gave the best results in offline experiments on Twitter data and in an online A/B test. They outperformed both waiting-window baselines and the delayed-feedback model [7].

```
 t0 impression ------> emit (x, y=0)           <- fake negative, immediately
 t1 click/convert ---> emit (x, y=1)           <- duplicate as positive
 trainer: loss reweights each example by f(p_model(x)) to undo the duplication
```

### 2.3 Feedback-shift importance weighting (CyberAgent, WWW 2020)

Yasui et al. frame delayed conversions as a **feedback shift**: the label distribution at training time differs from the label distribution at test time. Their importance-weighting method (FSIW) trains an auxiliary model that estimates the probability that an observed negative is a real negative given its elapsed time. Each example is then reweighted with those estimates [11].

### 2.4 Alibaba: elapsed-time sampling (ES-DFM) and real negatives (DEFER)

**ES-DFM (AAAI 2021).** This approach sits between "wait for everything" and "wait for nothing". Yang et al. wait a short **elapsed window e**. A click that converts within e is emitted once as a positive. Otherwise it is emitted as a negative, and if it converts later it is re-emitted as a positive. Importance weights derived from the modeled elapsed-time distribution correct the resulting bias [8].

**DEFER (KDD 2021).** Gu et al. point out that in fake-negative schemes, real negatives never get their own corrected entry. DEFER therefore **re-ingests real negatives too** once the attribution window closes, so every sample appears with its final label in addition to the early copy, and an importance-sampling loss accounts for the duplication. The paper reports offline and online gains in Alibaba's display-advertising system, and gives variants of the design for different attribution-window lengths [9].

Later work such as DEFUSE (WWW 2022) refines unbiased estimation for the duplicated-sample streams. The common pattern is to emit early, emit again when the label changes, and correct with importance weights.

### 2.5 Sample selection bias in conversion: ESMM (Alibaba, SIGIR 2018)

A separate problem from delay is that conversion-rate (CVR) models are traditionally trained only on *clicked* impressions but are used to score *all* impressions. ESMM (Entire Space Multi-task Model) trains two towers over the **entire impression space**. One predicts pCTR and the other predicts pCVR, and the supervised outputs are pCTR and pCTCVR = pCTR * pCVR. In this way, CVR is learned implicitly from impression-level data and the selection bias is avoided. Alibaba released the Ali-CCP dataset with this paper [12].

### 2.6 Comparison of delayed-feedback strategies

| Strategy | Freshness | Label noise | Extra storage | Correction | Source |
|---|---|---|---|---|---|
| Wait full window (7-30 d) | worst | none | buffer all clicks | none | baseline |
| Fixed short window W | good | fake negs after W | buffer W | none / calibration | [1] |
| Joint delay model | good | modeled | none | survival likelihood | [10] |
| Fake negatives (no wait) | best | high, corrected | duplicates positives | FN-weight / FN-calib | [7] |
| FSIW | good | modeled | none | aux elapsed-time model | [11] |
| Elapsed-time sampling | good | moderate | duplicates late positives | importance weights | [8] |
| Real-negative re-ingest | good | lowest | duplicates all | importance weights | [9] |

---

## 3. Negative sampling strategies

The right negative-sampling strategy depends on the stage. **Ranking** models see real impressions, so the negatives are shown-but-not-engaged items, usually downsampled. **Retrieval** models (two-tower) score the whole corpus, so they need negatives drawn from the corpus, and impressions alone are a biased choice.

### 3.1 YouTube candidate generation (Covington et al., RecSys 2016)

- **Extreme multiclass formulation.** The model predicts the next watch out of millions of videos with a sampled softmax. According to the paper, it samples several thousand negatives and corrects with importance weighting [13].
- **Training examples come from all watches**, including watches of YouTube videos embedded on other sites, and not only from recommendation impressions. This keeps the model from just reinforcing the current recommender's exploitation loop [13].
- **Fixed number of examples per user.** This keeps a small group of very active users from dominating the loss [13].
- **Predict the *next* watch rather than a random held-out watch**, and withhold information that would leak the label (for example, the search query that immediately led to the watch). Episodic viewing makes "future" leakage very harmful [13].
- **"Example age" feature.** The model gets the age of each training example as a feature, set to zero (or slightly negative) at serving time. This removes the bias toward older content that comes from training on past windows, and it shows directly how dataset freshness enters the model [13].

### 3.2 Sampling-bias-corrected in-batch softmax (YouTube, RecSys 2019)

Two-tower retrieval models commonly use **in-batch negatives**: the other items in the mini-batch serve as negatives. That is cheap, but popular items appear as negatives far more often, so the model over-penalizes them. Yi et al. apply the **logQ correction**: subtract log(p_j) from each logit, where p_j is the item's sampling probability. Because the item vocabulary changes in a streaming setting, they also propose a **streaming frequency estimator**. Two hashed arrays, A and B, record the last step at which an item was seen and a moving average of the gap between sightings, and p_j is approximately 1/B[h(j)]. The paper reports gains in offline recall and in live YouTube experiments [14].

### 3.3 Mixed negative sampling (Google Play, WWW 2020)

In-batch negatives contain only items that someone engaged with, so long-tail items rarely appear as negatives. **Mixed Negative Sampling (MNS)** adds a second set of negatives sampled **uniformly from the whole corpus** and combines them with the in-batch negatives in one softmax. This reduces the selection bias and improves retrieval of fresh and long-tail apps in Google Play experiments [15].

### 3.4 Facebook Search embedding-based retrieval (KDD 2020)

Huang et al. give some of the most quoted negative-sampling results [16]:

- **Random negatives beat "impressed but not clicked" negatives for retrieval.** The authors found that using non-click impressions as negatives clearly lowered recall compared with random negatives. Their explanation is that retrieval has to separate relevant items from the whole corpus, while impressed items are already near-relevant (the reported drop was large; see paper for exact figure).
- **Hard negative mining, online and offline.** Online: within a batch, use the highest-scoring non-positive items as hard negatives. Offline: retrieve with the current model and take results from a middle rank band (for example, around rank 101-500), because the very top ranks contain unlabeled positives.
- **Mix easy and hard negatives.** Hard negatives alone hurt the model. The paper found blends heavily weighted toward easy/random negatives worked best (on the order of 100:1 easy:hard, approx.).
- **Hard positives.** Positives mined from failed searches that the production system had not retrieved.

### 3.5 PinSage hard negatives and curriculum (Pinterest, KDD 2018)

PinSage trains with a max-margin loss on (query, positive) pairs taken from co-engagement. Negatives are mostly random, but **hard negatives** are added. These are items that are related to the query but not as related as the positive, drawn from roughly ranks 2000-5000 of personalized PageRank scores for the query. **Curriculum training** adds one more hard negative per epoch. The paper reports that the hard negatives roughly doubled the number of epochs needed to converge but improved quality [17].

### 3.6 Airbnb listing embeddings (KDD 2018)

Grbovic and Cheng train skip-gram embeddings on **click sessions** [18]:

- **Session definition.** Sessions are split when two clicks are more than 30 minutes apart. Only clicks where the user stayed on the listing page long enough (about 30 seconds, approx.) count, which removes accidental clicks. This is a dwell-based implicit-feedback filter.
- **Booked listing as global context.** The eventually booked listing is added to the context of every listing in the session, so the embeddings are pulled toward conversion.
- **Market-aware negatives.** Random negatives come from the whole world, so they are trivially easy: a Paris listing versus a Tokyo listing. Airbnb also samples negatives **from the same market** as the central listing, which forces the model to learn distinctions within a market.
- **Rejections as explicit negatives** for user-type and listing-type embeddings. Host rejections are used as negatives.

### 3.7 Ranking-stage downsampling and calibration

For ranking, negatives are the impressions the user did not engage with, and the main question is **how many to keep**. Facebook's uniform negative downsampling with the q = p/(p + (1-p)/w) correction [1] and Monolith's log-odds correction [3] are the standard public designs. Rule of thumb: downsampling negatives changes the *base rate*, so any consumer that uses probabilities directly (an auction, a weighted-sum value model, multi-task blending) must recalibrate. Ranking-only consumers do not need to.

### 3.8 Negative sampling summary

| Negative type | Stage | Pros | Cons | Public examples |
|---|---|---|---|---|
| Non-engaged impressions | ranking | true counterfactual of exposure | position/selection bias; useless for retrieval | [1], [3] |
| Uniform random corpus | retrieval | covers long tail; unbiased | too easy alone | [15], [16] |
| In-batch | retrieval | free; scales | popularity bias, needs logQ | [14] |
| Mixed (in-batch + uniform) | retrieval | balances both | extra compute | [15] |
| Hard (model-mined) | retrieval | sharper boundaries | false negatives; slower convergence | [16], [17] |
| Same-market / contextual | retrieval / embeddings | in-context discrimination | domain-specific | [18] |
| Explicit (hide, reject, report) | both | strong signal | sparse | [18], [19] |

---

## 4. Label definitions and implicit feedback

### 4.1 Clicks are not enough: dwell time (Yahoo, RecSys 2014)

Yi et al. ("Beyond Clicks") argue that clicks on news and content streams are noisy proxies for satisfaction, and propose **dwell time** as the training signal. Their practical findings [20]:
- Dwell time must be **computed from client-side events**: page focus and blur, scrolling and app backgrounding. Naive server-side timestamp differences are inflated by idle tabs.
- Raw dwell time differs systematically by **content type** (article, slideshow, video), **content length** and **device**. They normalize dwell time per context, for example by z-scoring the log of dwell time within each group, before using it as a label or weight.
- Models trained on normalized dwell did better than click-trained models on engagement metrics.

### 4.2 YouTube: from clicks to watch time to "valued" watch time

**Weighted logistic regression (2016).** YouTube's ranking model predicts *expected watch time* using weighted logistic regression. Positive (clicked) impressions are weighted by observed watch time and negatives have unit weight. The learned odds then approximate E[T] * (1 + P), which is roughly E[T] when CTR P is small, so serving uses exp(logit) as the expected watch time [13]. This is the standard example of **turning a continuous engagement signal into a sample weight**.

**Satisfaction (2021).** YouTube's public explanation of its recommender describes how the optimization target moved: first clicks (which encouraged clickbait), then watch time, then "valued watch time" measured by **user surveys** ("How satisfied were you with this video?"). Only a small fraction of views get a survey, so a model is trained to predict survey responses across all watches, and that prediction becomes a label or objective for the recommender [21].

### 4.3 Short-video watch time: duration bias (Kuaishou, KDD 2022)

In short-video feeds, longer videos mechanically collect more watch time, so raw watch time as a label favors long content. D2Q (Duration-Deconfounded Quantile) splits videos into **duration buckets** and trains the model to predict the *quantile* of watch time within each bucket. This removes the confounding effect of duration while keeping the useful signal [22]. Industry teams also often define binary labels such as "effective view" or "complete play" from watch ratio thresholds that depend on duration (thresholds are company-specific and seldom published).

### 4.4 Twitter/X: a weighted portfolio of engagement labels

The 2023 open-sourcing of Twitter's recommendation code made public a heavy ranker that predicts about ten engagement probabilities. The final score is a **weighted sum** of those probabilities, with weights published in the README [19][23]. Approximate values from the April 2023 README: like 0.5, retweet 1.0, reply 13.5, profile click then engagement 12.0, "good click" 11.0, reply that the author engages with 75.0, video watched at least 50% 0.005, negative feedback (show less, block, mute) -74.0, report -369.0 [19].

Two label definitions are worth noting:
- **"Good click"** is a click into the conversation followed by a reply or like, or by staying for at least a few minutes (about 2 minutes in one variant). This is a dwell-qualified click, built in the same spirit as Airbnb's and Yahoo's dwell filters.
- **Negative labels with large negative weights** let rare explicit negative feedback outweigh many positive engagements.

### 4.5 Facebook News Feed: multi-signal labels and survey-derived targets

Meta's description of News Feed ranking lists many predicted events: like, comment, share, hide, and "worth your time" survey responses. Each is a separate binary label from the joined impression stream, combined into a value score with tunable weights [24]. As at YouTube, survey responses are used to calibrate and validate engagement-based objectives against what users say they value.

### 4.6 Pinterest PinnerFormer: long-horizon labels

PinnerFormer learns a user embedding meant to predict the user's positive engagements over a **future window** (up to 28 days), not just the next action. The "dense all action" loss uses every positive action in the window as a target. Pinterest counts repins, close-ups and long clicks as positive actions (outbound clicks with long dwell, so short bounces are excluded). Training data is produced in daily batches, and the paper reports that a daily-refreshed embedding largely matched a real-time sequence model while being much cheaper to serve [25].

### 4.7 Position and selection bias in labels

Logged labels depend on what was *shown* and *where*. Two public lines of work are standard references:
- **Google personal search (Wang et al., SIGIR 2016):** estimate position propensities from result randomization and reweight clicks by inverse propensity [26].
- **Unbiased learning-to-rank (Joachims et al., WSDM 2017):** a counterfactual inverse-propensity-scoring framework for learning from biased click logs [27].

In practice, recommender pipelines handle this in one of three ways. They log **position** and other presentation features into the impression record, then set them to a constant at serving time (the same idea as YouTube's "example age"). They weight examples by inverse propensity. Or they keep a small **randomized exploration slice** of traffic to provide propensity estimates and unbiased evaluation data.

---

## 5. Sample weighting

Sample weights in recommender training data fall into five groups:

1. **Correction weights**, which undo a sampling choice. Examples are 1/w for downsampled negatives (or recalibration of the outputs instead) [1], logQ for in-batch sampling [14], importance weights for fake and duplicated delayed-feedback samples [7][8][9][11], and inverse propensity for position bias [26][27].
2. **Value weights**, which encode how much an outcome is worth. Examples are watch time as the weight on positives [13] and a value-weighted sum of predicted engagements [19][24].
3. **User-balance weights.** YouTube's fixed number of examples per user [13] can be implemented as truncation or as per-user weights. This keeps heavy users from dominating the loss and, in doing so, shapes whose preferences the model learns.
4. **Recency weights**, which up-weight recent examples in batch training to approximate freshness. These are often used as a cheap substitute for a streaming pipeline.
5. **Label-quality weights**, which down-weight noisy labels such as short-dwell clicks or bot traffic, or drop them entirely as Airbnb does with its dwell filter [18].

A practical warning that runs through all of these sources: **every weight changes the calibration of the output.** Any downstream system that uses scores as probabilities (ads auctions, multi-objective blending, thresholds) needs a calibration stage after weighting, and calibration needs to be monitored per segment.

---

## 6. Dataset freshness, retraining cadence and leakage

### 6.1 Evidence that freshness matters

- **Facebook Ads:** model quality degrades as the training data falls days behind, which supports daily or online training [1].
- **Monolith:** shorter parameter-sync intervals between training and serving give higher AUC, measured in production-like experiments [3].
- **YouTube:** the "example age" feature lets the model learn time-dependent popularity, especially for fresh uploads, so it does not average over the training window [13].
- **Pinterest PinnerFormer:** shows the opposite case. With a properly defined long-horizon objective, a *daily* batch embedding can come close to real-time quality [25]. How much freshness you need depends on the objective.

### 6.2 Training/serving skew and leakage

Rules of ML [6] gives the most compact list of failure modes:
- **Rule #29:** log features at serving time ("log and wait") to get exact training/serving parity.
- **Rule #31:** if training joins against a table that changes over time, the features seen in training will differ from those seen at serving.
- **Rule #32:** reuse code between the training and serving pipelines.
- **Rule #33:** test on data collected *after* the training window (temporal split), not on random splits.

Netflix's DeLorean and Axion [4][5], Feathr's point-in-time joins at LinkedIn [28], and Monolith's serving-time feature logging [3] are all ways of putting these rules into practice.

```
 LEAKAGE TIMELINE (what a point-in-time join must enforce)

   feature_ts <= impression_ts  <  label_ts <= impression_ts + attribution_window
   |------- allowed features ---|-- label collection --|
                                ^ impression            ^ label final
   Using any feature value with ts > impression_ts = leakage.
```

---

## 7. Data pipelines at scale

### 7.1 Meta: Data Storage and Ingestion (DSI), ISCA 2022

Zhao et al. describe the end-to-end pipeline that feeds Meta's DLRM-style recommendation trainers, and the finding that the resources spent on **data storage and ingestion can be comparable to, or exceed, those spent on GPU training** [29].

**Architecture** [29]:
- **Logging.** Serving hosts log features and events through **Scribe**, Meta's distributed log transport.
- **ETL and label join.** Streaming and batch jobs join features with labels and write partitioned **Hive tables** (columnar DWRF, a variant of ORC) to **Tectonic**, Meta's exabyte-scale distributed filesystem.
- **DPP (Data PreProcessing service).** A disaggregated fleet of CPU workers reads columnar data, runs feature transforms (filtering, hashing, normalization) and streams tensors to trainers. This frees trainer hosts from the preprocessing bottleneck.
- **Feature-level reads.** Each model reads only a subset of the thousands of logged features, so column projection and file layout (feature flattening, coalesced reads) dominate I/O efficiency.

```
 serving --> Scribe --> streaming/batch ETL (label join) --> Hive tables (DWRF)
                                                               on Tectonic
                                                                   |
                                     +-----------------------------+
                                     v
                       DPP workers (CPU fleet: read cols, decode,
                       transform, batch)  ==tensors==>  GPU trainers
```

**Lessons** [29]:
- Data volume grows faster than compute, and datasets are reported in the petabyte range per model family (approx.).
- Many trainings run on the same tables (exploration, hyperparameter sweeps, combo jobs), so storage and decoding efficiency become fleet-level costs.
- Preprocessing has to scale independently of training. Hence the disaggregated DPP.

Related Meta work deduplicates training samples within a session, since many impressions from one request share identical user features, to cut ingestion and training cost. This general idea of **request-level deduplication** works in any pipeline that logs per impression.

### 7.2 Google TFX and Apache Beam

TensorFlow Extended (TFX) builds the training-data pipeline from components (ExampleGen, StatisticsGen, SchemaGen, ExampleValidator, Transform). These run on **Apache Beam** for distributed processing, typically on Google Cloud Dataflow. The pattern provides: (a) schema inference and **data validation** that catch skew and drift before training; (b) **tf.Transform**, which computes full-pass statistics such as vocabularies and normalization constants in Beam and exports them as a TensorFlow graph that runs identically at serving time, closing one major source of skew [30].

### 7.3 Spotify Scio

Spotify built and open-sourced **Scio**, a Scala API for Apache Beam that runs on Dataflow. It is the backbone of much of Spotify's batch and streaming data processing, including generating training data for personalization models [31]. Scio is one public example of a typed, functional Beam API used at scale for recommendation data.

### 7.4 Spark-based pipelines

Spark is the most common public substrate for batch label joins and feature generation. Netflix DeLorean [4] and Axion [5] run offline feature generation in Spark. LinkedIn's Feathr provides point-in-time-correct feature joins on Spark, used to build training datasets from observation data (impressions with timestamps) and timestamped feature sources [28]. In the typical Spark design, impressions are the "observation" table, labels are joined with a time-bounded window, and features are joined "as of" the impression time.

### 7.5 Reference pipeline (synthesis)

```
 ONLINE                                 NEARLINE                       OFFLINE
 ------                                 --------                       -------
 ranker logs per request:              Flink/Beam streaming joiner:    Spark/Beam daily:
  req_id, user_id, ts, model_ver,       - key by req_id/imp_id         - re-join with full
  [item_id, position, features,       - short window (min-hours)        attribution window
   scores] per impression              - emit early (fake neg) +       - dedup, bot filter
        |                                 re-emit on label change       - neg downsampling
        v                               - importance weight column      - weights, splits
 Kafka/PubSub/Scribe topics  -------->  -> online trainer topic         - write Parquet /
 (impressions, engagements,             -> warehouse sink                 TFRecord / Hive
  conversions, surveys)                                                 - data validation
                                                                         -> batch trainer
```

---

## 8. Key takeaways for system designers

1. **Mint an impression ID at serving time and pass it through every downstream event.** Photon, Facebook's online joiner and Monolith all depend on a stable join key [1][2][3]. If you get this wrong, the label join becomes guesswork based on timestamps.
2. **Log features at serving time ("log and wait")** instead of reconstructing them offline. Netflix moved from snapshot time travel to logged facts, and Google's Rules of ML explicitly recommends logging [4][5][6].
3. **Treat the label window as a first-class hyperparameter.** For clicks, a short waiting window works. For conversions, use a delayed-feedback method: fake negatives with FN weighting or calibration [7], elapsed-time sampling [8], real-negative re-ingest [9], or a delay model [10].
4. **Choose negatives for the stage.** Ranking uses non-engaged impressions, downsampled and then recalibrated [1]. Retrieval uses corpus negatives: in-batch with logQ [14], plus uniform [15], plus a small dose of hard negatives [16][17]. Using impressions as retrieval negatives is a known way to lose recall [16].
5. **Define implicit labels with dwell or quality thresholds and normalize by context.** Raw clicks reward clickbait, and raw watch time rewards long videos [13][20][21][22]. Dwell-qualified clicks (Twitter's "good click", Airbnb's session filter, Pinterest's long clicks) are now standard [18][19][25].
6. **Keep explicit negative feedback as heavily weighted labels.** It is rare but highly informative (Twitter's -74 and -369 weights) [19].
7. **Every sampling and weighting choice moves calibration.** Recalibrate after downsampling and weighting [1][3], and monitor calibration per segment.
8. **Balance users and correct exposure bias.** Use per-user caps [13], log position and set it to a constant at serving time, and keep a small randomized slice for propensity estimates [26][27].
9. **Match freshness to the objective.** Minute-level streaming pays off for fast-moving feeds and ads [3][7], while long-horizon objectives can run on daily batches [25]. Measure the decay curve before building streaming infrastructure [1].
10. **Budget for data ingestion as much as for training.** At Meta's scale, storage, decoding and preprocessing are comparable to GPU training in resources. Use columnar storage, column projection, disaggregated preprocessing and request-level deduplication [29].
11. **Validate the data before training.** Schema checks, distribution-drift checks and transform code shared between training and serving (TFX/tf.Transform) catch silent pipeline breaks that offline AUC will not [30].

---

## References

1. He, X., Pan, J., Jin, O., et al. "Practical Lessons from Predicting Clicks on Ads at Facebook." ADKDD 2014, Facebook. https://dl.acm.org/doi/10.1145/2648584.2648589
2. Ananthanarayanan, R., et al. "Photon: Fault-tolerant and Scalable Joining of Continuous Data Streams." SIGMOD 2013, Google. https://research.google/pubs/photon-fault-tolerant-and-scalable-joining-of-continuous-data-streams/
3. Liu, Z., et al. "Monolith: Real Time Recommendation System With Collisionless Embedding Table." 2022, ByteDance. https://arxiv.org/abs/2209.07663
4. Netflix Technology Blog. "Distributed Time Travel for Feature Generation." 2016, Netflix. https://netflixtechblog.com/distributed-time-travel-for-feature-generation-389cccdd3907
5. Netflix Technology Blog. "Evolution of ML Fact Store." 2022, Netflix. https://netflixtechblog.com/evolution-of-ml-fact-store-5941d3231762
6. Zinkevich, M. "Rules of Machine Learning: Best Practices for ML Engineering." Google. https://developers.google.com/machine-learning/guides/rules-of-ml
7. Ktena, S. I., Tejani, A., Theis, L., et al. "Addressing Delayed Feedback for Continuous Training with Neural Networks in CTR Prediction." RecSys 2019, Twitter. https://arxiv.org/abs/1907.06558
8. Yang, J.-Q., et al. "Capturing Delayed Feedback in Conversion Rate Prediction via Elapsed-Time Sampling." AAAI 2021, Alibaba. https://arxiv.org/abs/2012.03245
9. Gu, S., et al. "Real Negatives Matter: Continuous Training with Real Negatives for Delayed Feedback Modeling." KDD 2021, Alibaba. https://arxiv.org/abs/2104.14121
10. Chapelle, O. "Modeling Delayed Feedback in Display Advertising." KDD 2014, Criteo. https://dl.acm.org/doi/10.1145/2623330.2623634
11. Yasui, S., Morishita, G., Fujita, K., Shibata, M. "A Feedback Shift Correction in Predicting Conversion Rates under Delayed Feedback." WWW 2020, CyberAgent. https://dl.acm.org/doi/10.1145/3366423.3380032
12. Ma, X., et al. "Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate." SIGIR 2018, Alibaba. https://arxiv.org/abs/1804.07931
13. Covington, P., Adams, J., Sargin, E. "Deep Neural Networks for YouTube Recommendations." RecSys 2016, Google. https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/
14. Yi, X., Yang, J., Hong, L., et al. "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations." RecSys 2019, Google. https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/
15. Yang, J., Yi, X., et al. "Mixed Negative Sampling for Learning Two-tower Neural Networks in Recommendations." WWW 2020 Companion, Google. https://research.google/pubs/mixed-negative-sampling-for-learning-two-tower-neural-networks-in-recommendations/
16. Huang, J.-T., et al. "Embedding-based Retrieval in Facebook Search." KDD 2020, Facebook. https://arxiv.org/abs/2006.11632
17. Ying, R., He, R., Chen, K., et al. "Graph Convolutional Neural Networks for Web-Scale Recommender Systems" (PinSage). KDD 2018, Pinterest. https://arxiv.org/abs/1806.01973
18. Grbovic, M., Cheng, H. "Real-time Personalization using Embeddings for Search Ranking at Airbnb." KDD 2018, Airbnb. https://dl.acm.org/doi/10.1145/3219819.3219885
19. Twitter. "the-algorithm-ml: Heavy Ranker README (projects/home/recap)." 2023, Twitter/X. https://github.com/twitter/the-algorithm-ml/blob/main/projects/home/recap/README.md
20. Yi, X., Hong, L., Zhong, E., Liu, N. N., Rajan, S. "Beyond Clicks: Dwell Time for Personalization." RecSys 2014, Yahoo. https://dl.acm.org/doi/10.1145/2645710.2645724
21. Goodrow, C. "On YouTube's recommendation system." 2021, YouTube Official Blog. https://blog.youtube/inside-youtube/on-youtubes-recommendation-system/
22. Zhan, R., et al. "Deconfounding Duration Bias in Watch-time Prediction for Video Recommendation." KDD 2022, Kuaishou. https://arxiv.org/abs/2206.06003
23. Twitter Engineering. "Twitter's Recommendation Algorithm." 2023, Twitter/X. https://blog.x.com/engineering/en_us/topics/open-source/2023/twitter-recommendation-algorithm
24. Lada, A., Wang, M., Yan, T. "How machine learning powers Facebook's News Feed ranking algorithm." 2021, Meta Engineering. https://engineering.fb.com/2021/01/26/ml-applications/news-feed-ranking/
25. Pancha, N., Zhai, A., Leskovec, J., Rosenberg, C. "PinnerFormer: Sequence Modeling for User Representation at Pinterest." KDD 2022, Pinterest. https://arxiv.org/abs/2205.04507
26. Wang, X., Bendersky, M., Metzler, D., Najork, M. "Learning to Rank with Selection Bias in Personal Search." SIGIR 2016, Google. https://research.google/pubs/learning-to-rank-with-selection-bias-in-personal-search/
27. Joachims, T., Swaminathan, A., Schnabel, T. "Unbiased Learning-to-Rank with Biased Feedback." WSDM 2017, Cornell. https://arxiv.org/abs/1608.04468
28. LinkedIn / Feathr AI. "Feathr: A scalable, unified data and AI engineering platform" (point-in-time feature joins). 2022, LinkedIn. https://github.com/feathr-ai/feathr
29. Zhao, M., et al. "Understanding Data Storage and Ingestion for Large-Scale Deep Recommendation Model Training." ISCA 2022, Meta. https://arxiv.org/abs/2108.09373
30. TensorFlow. "TensorFlow Extended (TFX)" documentation (ExampleGen, Transform, data validation on Apache Beam). Google. https://www.tensorflow.org/tfx
31. Spotify. "Scio: A Scala API for Apache Beam and Google Cloud Dataflow." Spotify (open source). https://github.com/spotify/scio
