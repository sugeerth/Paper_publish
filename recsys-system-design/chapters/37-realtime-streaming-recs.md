# 37. Real-Time and Session-Based Recommendation: Streaming Updates, Online Learning, Nearline Pipelines and In-Session Intent

A recommender reacts to user behaviour on several clocks. The **item corpus** changes by the minute: new videos, posts, listings and ads appear constantly. **User intent** changes within a session: a user who has just saved three kitchen Pins wants kitchens now, whatever their 90-day profile says. **Model parameters** go stale as the world drifts: Facebook showed in 2014 that prediction quality degrades measurably for every day the training data falls behind the serving day [3]. Real-time recommendation is the set of designs that shorten these clocks without breaking cost, latency or stability budgets.

This chapter surveys the public designs along four axes:

1. **Fresh features.** Streaming counters, real-time action sequences and session embeddings computed from event streams (Kafka plus Flink, Heron or Samza) and served from a low-latency store. Examples: Twitter's aggregation framework, Airbnb's session embeddings, Chronon's tiled Flink aggregations, DoorDash Riviera.
2. **Fresh candidates.** Indexes and graphs that absorb new edges and items within seconds. Examples: GraphJet, Pixie, Manas realtime HNSW and X's Thunder.
3. **Fresh user representations inside the model.** Real-time sequence encoders that read the last N actions at request time. Examples: TransAct, TransAct V2, SIM/TWIN-style search over long histories, and the academic session-based line GRU4Rec, SASRec and BERT4Rec.
4. **Fresh parameters.** Online or continual training with streaming label joins and incremental parameter sync. Examples: Facebook's online joiner, Google Photon, ByteDance Monolith, Kuaishou PERSIA, and delayed-feedback corrections from Twitter and Alibaba.

The recurring tension is **freshness versus stability**. Fresher means more infrastructure, more ways to diverge between training and serving, more exposure to feedback loops and noise, and less diversity. The best designs pair a cheap, stable long-term component (a batch embedding, a daily model, a lifelong history) with a small, fast real-time component, and they measure the decay curve before buying freshness.

*Sourcing note.* The live web-search budget for this run was used up and the sandbox egress proxy blocked direct fetches of arXiv, Medium and company blogs. This chapter therefore draws on primary sources (papers, engineering posts and open-source READMEs) that sibling chapters of this Atlas had already fetched and summarized (Chapters 5, 8, 9, 11, 18, 19, 20, 21, 22, 26 and 28). Figures marked "approx." come from secondary summaries or could not be re-verified here.

---

## 37.1 The landscape: where freshness lives

### 37.1.1 Offline, nearline, online: the Netflix blueprint (2013)

Netflix's "System Architectures for Personalization and Recommendation" by Amatriain and Basilico is still the clearest vocabulary for placing computation [27]. It separates **where** computation runs from **what** is computed:

```
 member events (plays, ratings, browsing, search)
        |
        +--> OFFLINE  (Hadoop/Hive/Pig): training, batch scoring; no data or
        |             complexity limit, but stale and context-blind
        |
        +--> NEARLINE (event-driven, no request waiting): e.g. on a play event
        |             recompute member factors and scores; results written to
        |             Cassandra / EVCache / MySQL
        |
        v
     ONLINE (per request, strict SLA): fetch precomputed + live signals,
            light model eval, filtering, context; FALLBACK to precomputed
            or unpersonalized results on timeout
```

Key rules from the post [27]:

- **Nearline** is the middle tier. It reacts to events within seconds, can be more complex than online code because no request waits on it, and writes its results to fast stores.
- **Events vs data.** Events are small, time-sensitive messages that must be processed fast and flow over a pub/sub system (Hermes in the post). Data is the denser record collected into Hadoop for later processing.
- **A model can be split across tiers.** Item factors are trained offline, while user-dependent parts are re-solved nearline when the member acts. Netflix slides from the period describe re-solving a member's MF factors on a play event (approx.).

Amatriain's 2023 "10th anniversary" revisit says the split still holds, but "nearline" has grown into streaming feature platforms and near-real-time training [28]. Almost every system in this chapter can be placed in one of the three tiers.

### 37.1.2 Four kinds of freshness

| Freshness kind | Typical latency | Mechanism | Representative systems |
|---|---|---|---|
| Feature freshness | ms to seconds | Stream processor -> KV store; request-time sequence fetch | Twitter RTA [24], Chronon tiles [32], Airbnb [30], TransAct [9] |
| Candidate freshness | seconds | In-memory graphs, incremental ANN inserts | GraphJet [18], Pixie [15], Manas realtime [16], Thunder [26] |
| Representation freshness | per request | Real-time sequence encoder in the ranker | TransAct [9], TransAct V2 [11], TWIN [46] |
| Parameter freshness | minutes to hours | Online joiner + streaming trainer + incremental sync | FB 2014 [3], Monolith [1], PERSIA [47] |

These are complementary, not alternatives. A short-video feed typically uses all four. A marketplace like Airbnb can get most of the benefit from feature freshness alone [30].

---

## 37.2 Event backbone: a canonical action stream

Every real-time design starts with a reliable, low-latency stream of user actions with stable keys. Two public examples:

**Twitter/X Unified User Actions (UUA).** UUA is described as "a centralized, real-time stream of user actions". It merges client-side and server-side events into one Kafka stream, which is replicated to HDFS, Pub/Sub, GCS and BigQuery [22]. The **User Signal Service** then standardizes explicit and implicit signals (likes, retweets, follows, video views and so on) so that retrieval and ranking consume one schema [23].

**Recos-Injector.** For Twitter's GraphJet-based services, Recos-Injector is the streaming preprocessor. It consumes favorites, retweets, follows and client events, filters and combines them, and publishes them to a **dedicated Kafka topic per GraphJet service** [21].

Design rules that recur across sources:

- **Mint an impression or request ID at serving time and carry it into every downstream event.** Facebook's online joiner, Google Photon and Monolith all depend on such a key for the label join [1][3][4].
- **Log features at serving time** rather than recomputing them later. This is the only robust defence against training/serving skew for real-time features, because the value served at time t is hard to reconstruct offline [1][34].
- **Per-consumer topics.** Recos-Injector's per-service topics let each consumer get a filtered, compact stream and scale independently [21].

---

## 37.3 Streaming features: counters, windows and tiles

### 37.3.1 Twitter's aggregation framework: one definition, two engines

**Problem.** The Home ranker, push notifications and email recommendations all need user, author, user x author and tweet engagement aggregates, conditioned on content type and engagement label, both over long horizons and over the last few minutes [24].

**Design.** An `AggregateGroup` declares keys (for example `USER_ID` or `(USER_ID, AUTHOR_ID)`), features, labels, metrics and **half-lives** (exponential decay rather than hard windows). The same definition drives a daily summingbird-scalding batch job, whose output is uploaded to Manhattan, and a summingbird-storm/Heron **real-time** job backed by memcached [24].

```
            client events (Kafka client-event topics)
                 |                               |
   daily DataRecord dumps (HDFS)        Storm/Heron (summingbird)
                 |                               |
   batch aggregates -> Manhattan KV     real-time aggregates -> memcached
                 \___________________   _________________/
                                     \ /
              Home Mixer / pushservice feature hydration
```

**Scale.** The 2023 open-source release describes ranking features kept over long windows (50 days) and real-time windows (under 3 days, typically 30 minutes), with roughly 6,000 features hydrated per ranked candidate for the For You heavy ranker (approx.) [24][25].

**Tricks.** Decayed counters are monoid-friendly: an old value plus elapsed time is enough to roll forward, so each key stores O(1) state instead of a window of events. The real-time path also updates "realtime features" such as like, reply and retweet counts on tweets for Earlybird-based retrieval [25].

### 37.3.2 Airbnb/Stripe Chronon: tiled Flink aggregation

Chronon (open-sourced by Airbnb and co-maintained with Stripe) makes the freshness model an explicit part of the feature definition.

- A `GroupBy` with `accuracy = SNAPSHOT` is refreshed daily and backfilled as of the previous midnight. With `TEMPORAL` (the default when a Kafka `topic` is set) it is updated in real time online and backfilled with millisecond point-in-time accuracy offline [33].
- **Untiled vs tiled serving.** In the untiled design the streaming job writes individual events and the fetcher folds them at request time. With 10 events/s for one key and a 12-hour window, one request has to fold 432,000 events. In the tiled design a stateful **Flink** job pre-aggregates events into tiles of intermediate representations (for example sum and count for an average), and the fetcher reads 12 or 13 one-hour tiles. Stripe reported a **33% reduction in feature-serving latency** after the initial tiled rollout [32].
- The Flink job evaluates the same Spark SQL expressions as the batch engine by embedding Catalyst inside Flink, so batch and streaming share one expression language [35].
- Chronon logs served values and runs a daily job comparing them with backfilled values. It names the inherent sources of skew: Kafka plus KV write latency makes streaming features lag by milliseconds to seconds, and batch features are inconsistent "centered around midnight" until the bulk upload finishes [34].

```
 Kafka -> Flink [source] -> [Spark SQL eval] -> [tiling: IRs per tile] -> KV
 fetcher: get(key, tiles in [now - W, now)) -> merge IRs -> finalize -> feature
 batch upload holds hop-aligned partials; stream covers everything after batch end
```

### 37.3.3 Marketplace variants: DoorDash Riviera, Uber, Lyft

DoorDash's **Riviera** (2021) is a declarative framework for real-time features on Flink: features are defined in configuration and SQL, and the framework builds the streaming job. Its output lands in a Redis-based online feature store next to batch features from Fabricator [36]. Uber's Michelangelo used the same dual-path pattern: Kafka plus streaming jobs (Samza, later Flink) for near-real-time features such as a restaurant's recent average prep time, written to the same Cassandra store as batch features (approx.; see Chapter 8).

**Lesson.** The streaming feature problem is solved. The remaining design work is (1) one declarative definition for batch, stream and backfill, (2) pre-aggregation on the write path for hot keys, and (3) logging served values to measure and remove skew.

---

## 37.4 Real-time candidate generation

### 37.4.1 GraphJet (Twitter, VLDB 2016)

**Problem.** Tweets have a useful life of hours. Collaborative filtering on a nightly batch graph is too stale, so Twitter needed "users who engaged with X also engaged with Y" over roughly the last day, with new edges visible within seconds [18].

**Architecture.** GraphJet "maintains a full graph index over a sliding time window in memory on a single server" [19]. The bipartite graph has users on one side and tweets (or URLs, hashtags) on the other. The index is a sequence of **time-sliced segments**: new edges go into the newest segment and whole segments are dropped when they leave the window, which avoids per-edge deletion and garbage-collection pressure. Recommendations come from SALSA-style personalized random walks [18].

| Metric (per server, from README [19]) | Value |
|---|---|
| Edge ingestion | up to 1,000,000 edges/s |
| Recommendations, steady state | up to 500/s |
| Edge reads | "several million" per second |

**Production descendant: UTEG.** The User-Tweet-Entity-Graph service is "a Finagle thrift service built on the GraphJet framework". It produces the "XXX liked" out-of-network tweets on the home timeline. It is fed by Kafka through Recos-Injector, keeps **in-memory engagements for the past 24-48 hours**, takes the viewer's RealGraph-weighted followees as input, and returns tweets ranked by weighted followee engagement [20][21]. Twitter's 2023 blog attributes roughly 15% of For You tweets to this kind of social-graph traversal (approx.) [25].

```
 Kafka (fav, RT, follow, client events)
        |
  recos-injector (filter, combine, per-service topic)
        |
  UTEG / GraphJet (24-48 h window, all in RAM, full replica per server)
        |  input: viewer's weighted followees
        v
  tweets ranked by followee engagement, with social proof
```

### 37.4.2 Thunder (X, 2026)

In the 2026 open-source X algorithm, in-network retrieval moved from the Lucene-based Earlybird search cluster to **Thunder**, "an in-memory post store and realtime ingestion pipeline". It consumes post create and delete events from Kafka, serves the posts of the viewer's followees, trims posts older than a retention period, and promises "sub-millisecond lookups for in-network content without hitting an external database". It also receives the already-seen list so it can exclude those posts at the source [26]. Architecturally this is GraphJet again: a time-windowed in-memory index replacing a general-purpose search engine for very fresh content.

### 37.4.3 Pixie (Pinterest, WWW 2018)

**Problem.** Recommend from more than 3 billion Pins to more than 200 million users and react to what the user just did. The previous Hadoop system precomputed results and could not react within a session [15].

**Design.** A pruned Pin-Board bipartite graph of about 3 billion nodes and 17 billion edges, about 120 GB, is loaded into RAM on each server. Every server holds a full replica, so walks never cross machines. The graph is rebuilt offline. The **query** is real time: the user's most recent Pins, each weighted, seed many short random walks with restarts. Per-Pin visit counts are combined as V[p] = (sum over q of sqrt(V_q[p]))^2, which rewards candidates reached from several query Pins. Walks stop early once enough Pins pass a visit threshold [15].

**Scale.** p99 latency under 60 ms, about 1,200 requests/s per server and about 100,000 requests/s across the cluster, with up to 50% higher engagement than the Hadoop system [15].

**Lesson.** Freshness can come from the **query**, not the index. A daily graph with real-time seed actions already gives in-session responsiveness.

### 37.4.4 Manas realtime HNSW (Pinterest)

Pinterest's ANN platform Manas added a **realtime HNSW** mode in which new Pins are inserted incrementally into the graph index, so embedding-based retrieval can surface fresh content without waiting for a full rebuild. Filtering was first done by over-fetching K neighbours and post-filtering [16]. **Streaming filters** later reused the inverted index as the filter store, with bitmap and array-backed posting lists replacing skiplists for faster lookups [17].

### 37.4.5 Google News co-visitation (2007)

The earliest large-scale example of the split is Google News personalization [49]. MinHash user clusters and PLSI classes are computed in batch with MapReduce. **Co-visitation counts** (two stories clicked by the same user within a time window) and per-cluster click counts are updated online in Bigtable, so a story gets a score as soon as clicks arrive. Serving blends the three signals linearly. The pattern of batch structure plus streaming counts is the same one Twitter and Pinterest later used.

---

## 37.5 Real-time user intent inside the ranker

### 37.5.1 Pinterest TransAct (KDD 2023)

**Problem.** Pinterest's user embedding PinnerFormer is computed in a **daily batch**. It was explicitly trained to tolerate staleness: moving from real-time to daily inference cut Recall@10 by 13.9% with a next-action objective but only by 8.3% with the dense all-action objective [13]. Even so, a daily embedding cannot react to what the user did seconds ago [9].

**Hybrid design.** TransAct encodes the user's most recent actions (on the order of 100) **in real time inside the Homefeed ranker (Pinnability)**, while PinnerFormer and other batch features provide long-term context. The authors present this as combining real-time responsiveness with the cost efficiency of batch representations [9].

```
 real-time action stream                                  candidate Pin
         |                                                      |
 latest ~100 actions: PinSage emb + action type + timestamp     |
         |                                                      |
 early fusion: concat each action with candidate embedding <---+
         v
 small Transformer encoder
         v
 compress (first K tokens + max pooling) -> flatten
         v
 concat with PinnerFormer + other features -> DCN v2 -> MLP
         v
 multi-task heads (repin, click, hide, ...) -> utility
```

**Findings and tricks** [9][10]:

- **Early fusion** of each action with the candidate makes the sequence encoder target-aware.
- **Action-type embeddings** mattered more than positional encodings.
- **Random time-window masking** during training hides the most recent actions for some examples, so the model does not over-rely on the last few clicks. This protects diversity (as reported in summaries).
- The paper lists production problems: serving complexity, reduced diversity, and **engagement decay**, which means the model needs frequent retraining.
- **Serving.** The new model raised CPU latency by more than 20x, so Pinterest moved Homefeed ranking to **GPU serving** to stay latency-neutral at the same cost [10].
- **Results.** Secondary summaries report about +11% Homefeed repin volume and -10% hides (approx.). TransAct was deployed in Homefeed, Related Pins, Notifications and Search [9].

### 37.5.2 TransAct V2 (2025): real-time plus lifelong

TransAct V2 extends Homefeed ranking from about 100 real-time actions to lifelong histories of up to about **16,000 actions** [11][12]. It keeps three sequences, lifelong, real-time and impressions, and selects a short subsequence by **candidate-anchored nearest-neighbour search** in PinSage space. The **most recent real-time actions are always kept**, so in-session intent is never lost to the long-history search [11].

- Each token carries a timestamp, a multi-hot action type, the surface and a 32-d PinSage embedding. Lifelong actions are stored with int8 quantization [11].
- An auxiliary **Next Action Loss** predicts what the user does next [11].
- Serving kernels (fused dequantization plus NN search, a single-kernel Triton Transformer) improved end-to-end inference latency by 103x (p50), 338x (p90) and 250x (p99) over the unoptimised baseline [11].
- Reported online results: +6.35% repins, +1.41% time spent, -12.8% hides [12].

### 37.5.3 Long histories with real-time recency: SIM and TWIN

Alibaba's **SIM** (CIKM 2020) introduced the two-stage pattern for lifelong histories. A General Search Unit (GSU) retrieves behaviours relevant to the candidate. In production this is "hard search" by category over a **User Behavior Tree** keyed by user ID then category ID, about 22 TB in size. An Exact Search Unit (ESU) then runs target attention over the top-K. The output is concatenated with **short-term, real-time DIN features**. SIM modelled up to 54,000 behaviours and reported +7.1% CTR and +4.4% RPM in display advertising [45].

Kuaishou's **TWIN** (KDD 2023) makes the GSU use the same target-attention relevance as the ESU. It keeps it affordable by precomputing and caching linear projections of item-inherent features per item, and compressing user-item cross features into a scalar attention bias. The CP-GSU scores about 10^4 behaviours, keeps the top 100, and the authors report a 99.3% reduction of the compute bottleneck, enough to deploy on main traffic [46].

The real-time lesson: **lifelong and real-time are separate inputs with separate freshness**. The long history is indexed or cached and can lag. The last few minutes of actions are streamed and always included.

### 37.5.4 Airbnb: session embeddings as real-time features (KDD 2018)

**Problem.** Search ranking should react to what the guest is doing in this session, for example clicking several cheap, modern apartments near the centre, without retraining a model [30].

**Design.** Grbovic and Cheng trained skip-gram listing embeddings on click sessions, splitting sessions at gaps of more than 30 minutes. They added two changes: the **booked listing as a global context** for every listing in a booking session, and **same-market negatives**. Secondary summaries report about 800 million click sessions, about 4.5 million active listings and 32-dimensional embeddings (approx.) [30][31].

**Real-time use.** Kafka streams record the user's clicked, skipped, long-clicked, wishlisted and contacted listings over roughly the last two weeks. For each candidate the ranker computes features such as "similarity to clicked listings" and "similarity to skipped listings" online and feeds them to the GBDT ranker in production at the time. The paper reports an online booking gain, and the embedding similarity features ranked high in importance [30].

**Lesson.** Freshness can come from **cheap online similarity features over a streamed short history**, with no online training at all. Skipped items are a valuable real-time negative signal.

### 37.5.5 Netflix: in-session adaptation for search

Netflix research described augmenting search with recommendations adapted to the member's **current session**. Browsing and partial-query signals in the session predict intent before or while the member types, which matters most on TV devices where typing is slow (approx.) [29].

### 37.5.6 Instagram Explore: online user tower, offline item tower

Instagram Explore's main ML retrieval source is a two-tower network. **Item embeddings are computed offline (typically daily) and indexed**, and **a fresh user embedding is computed at request time** from online features before ANN search. The first-stage ranker reuses two towers for cacheability, and some users' heavy-stage results are precomputed off-peak [37]. This asymmetry, where the user side is fresh and the item side is batched, is the standard compromise for two-tower retrieval.

---

## 37.6 Session-based models: the academic line

Session-based recommendation predicts the next item from a short, often anonymous, sequence. It is the modelling counterpart of real-time intent.

| Model | Year / org | Core idea | Notable result or caveat |
|---|---|---|---|
| GRU4Rec [38] | 2015, Gravity R&D / Telefonica | GRU over session clicks; session-parallel mini-batches; in-batch negatives; BPR/TOP1 ranking losses | Beat item-kNN in the paper |
| Session-kNN [39] | 2017, Jannach & Ludewig | Nearest-neighbour over sessions | Beat GRU4Rec in most tested setups; hybrids best |
| HRNN [44] | 2017 | Hierarchical RNN: session RNN initialized from a user-level RNN | Personalizes sessions for logged-in users |
| SASRec [40] | 2018, UCSD | Causal self-attention for next item | About 10x faster than CNN/RNN baselines |
| BERT4Rec [41] | 2019, Alibaba | Bidirectional Cloze training | Not consistently better than SASRec on replication [42] |
| gSASRec [42] | 2023, Glasgow | SASRec + gBCE to fix negative-sampling overconfidence | +9.47% NDCG vs BERT4Rec on ML-1M; ~73% faster training |
| Transformers4Rec [43] | 2021, NVIDIA | Library bridging NLP Transformers and session recs | Used in competitions and industry pipelines |

Key details:

- **GRU4Rec** made RNNs practical for sessions with **session-parallel mini-batches**: many sessions run in parallel, and when one ends the next takes its slot and its hidden state is reset. Scoring is restricted to items in the mini-batch, an early form of in-batch negatives [38].
- **Strong simple baselines.** Jannach and Ludewig showed that session-based nearest neighbours often beat GRU4Rec, and the best results came from combining the two [39]. Co-visitation-style item-to-item methods (as in Google News [49]) remain hard to beat for very short sessions.
- **Loss and sampling matter more than architecture.** gSASRec traces most of the BERT4Rec vs SASRec gap to overconfidence caused by negative sampling [42].

**From academia to production.** Industrial systems rarely deploy a pure session model. Instead the session encoder becomes one component: a real-time sequence feature in a ranker (TransAct [9]), a user tower in retrieval (Instagram [37]), or the short-term branch next to a lifelong search unit (SIM [45], TWIN [46]). Meta's HSTU goes further, treating recommendation as sequential transduction over a single chronological stream of content and action tokens per user. The paper motivates the design explicitly by high-cardinality, **non-stationary streaming data** [48].

---

## 37.7 Online learning: streaming labels and continual training

### 37.7.1 Facebook ads online joiner (2014)

He et al. describe an online data joiner. Impressions and clicks are keyed by request ID. Impressions wait in a window: a click within the **waiting window** makes a positive, and otherwise a negative is emitted when the window closes. The labeled stream feeds an online trainer for the logistic-regression layer on top of boosted-tree feature transforms, and models are pushed back to the ranker [3].

- **Waiting-window trade-off.** Too short and late clicks become false negatives, biasing CTR down. Too long and examples arrive late and need more buffer memory.
- **Safety.** The joiner needs anomaly protection, for example alarms and cutting off the trainer if the click stream stops [3].
- **Freshness evidence.** Normalized entropy worsens as training data falls days behind evaluation, which motivates daily or online training [3].
- **Negative downsampling with recalibration** q = p / (p + (1 - p)/w) became the standard correction [3].

### 37.7.2 Google Photon (2013): exactly-once joins

Photon joins clicks to queries across datacenters with exactly-once semantics, because each joined click is billed and used for training. A Paxos-replicated **IdRegistry** records joined event IDs, so duplicate processing in several datacenters never emits a click twice. Clicks that arrive before their query are retried. The paper reports millions of joined events per minute with end-to-end latency of seconds (approx.) [4]. The lesson for recommenders is that **the join must be idempotent**: duplicate positives inflate CTR and break calibration.

### 37.7.3 ByteDance Monolith (2022): minute-level training at TikTok scale

**Problem.** ByteDance's feed and ads models face an unbounded, non-stationary sparse ID space, long-tailed feature frequency and concept drift within hours. Standard TensorFlow keeps batch training and serving "completely separated", and fixed-size hashed tables cause collisions [1].

**Architecture.**

```
 client actions --> Kafka: actions ---+
 serving-time feature log --> Kafka --+
                                      v
           Flink online joiner (join on request key; in-memory cache +
           on-disk KV for late actions; negative sampling)
                  |                                     |
          Kafka: training examples              HDFS (batch dumps)
                  |                                     |
          online training workers <--- same code ---> batch training
                  |
          training PS: Cuckoo-hash tables (frequency admission + expiry)
                  |  sparse: incremental sync of touched IDs (minute-level)
                  |  dense : synced far less often
                  v
          serving PS -> online inference
```

**Key tricks** [1][2]:

- **Collisionless embeddings.** Cuckoo-hash key-value tables on the parameter servers give every ID its own row. **Frequency filtering** admits an ID only after it has been seen a threshold number of times, and **expirable embeddings** evict inactive IDs. Together they bound memory.
- **One codebase, two modes.** Batch training on HDFS bootstraps the model, and online training then consumes the Kafka example stream indefinitely.
- **Asymmetric parameter sync.** Pushing terabytes every minute is impractical. Only sparse rows touched since the last sync are pushed, at minute-level intervals, while dense weights, which change slowly, are synced much less often.
- **Cheap fault tolerance.** Daily PS snapshots lose little quality on failure, because each shard holds a small fraction of IDs and online training heals quickly.

**Results.** Collisionless tables beat hashed ones. Shorter sync intervals (from hours down toward about 30 minutes in the experiments, approx.) gave consistently higher online AUC, and real-time training beat batch training for ads [1]. The repository is TensorFlow plus Bazel and was archived by ByteDance in October 2025 [2].

### 37.7.4 Kuaishou PERSIA (2022)

PERSIA splits training the same way. Embedding layers are updated **asynchronously** on CPU parameter servers, because sparse updates tolerate staleness, and the dense network is trained **synchronously** on GPUs, because its accuracy suffers under asynchrony. It reports models up to 100 trillion parameters in production at Kuaishou [47]. Monolith and PERSIA agree: **the sparse part is fast-moving and staleness-tolerant, and the dense part is slow-moving and sensitive**.

### 37.7.5 Delayed feedback in streaming training

Online training on conversions faces labels that arrive minutes to weeks late.

| Strategy | Freshness | Bias handling | Source |
|---|---|---|---|
| Waiting window | Delayed by window | Late positives become false negatives | Facebook [3] |
| Delay model (exponential delay) | Medium | Joint conversion + delay likelihood | Criteo, Chapelle [8] |
| Fake negatives, no wait | Best | FN-weighted or FN-calibrated loss | Twitter [5] |
| Elapsed-time sampling | Good | Short wait + re-emit late positives + importance weights | Alibaba ES-DFM [6] |
| Real negatives re-ingest | Good | Also re-ingest real negatives after window | Alibaba DEFER [7] |

Twitter's design (Ktena et al., RecSys 2019) ingests every eligible impression **immediately as a negative** and again as a positive when the engagement arrives. FN-weighted and FN-calibrated losses undo the bias, and they beat waiting-window and delay-model baselines offline and in an online A/B test [5]. For continuous training this is the freshest option and has become a common default.

### 37.7.6 Online learning for the policy layer

Freshness also applies to exploration and fusion. Yahoo's LinUCB updates a per-article ridge regression online for a small, fast-changing pool of news stories [50]. Kuaishou's **Pantheon** learns personalized fusion weights jointly with its real-time ranker, deployed for about 400 million daily users [52]. When the candidate pool churns faster than a model can be retrained, a bandit or online-updated policy layer on top of a slower ranker is a cheap way to stay fresh.

---

## 37.8 TikTok and the short-video case

TikTok's 2020 newsroom post describes the For You feed as ranking on user interactions (likes, shares, follows, comments, content created), video information and, with lower weight, device and account settings. Signals are weighted by strength: finishing a longer video is a stronger signal than shared country. It also describes diversity rules (no two consecutive videos from the same creator or with the same sound) and cold start through interest selection, which "early interactions quickly replace" [51].

Short video stresses every clock at once. A user may swipe through dozens of items in minutes, each swipe is a label, and the corpus churns hourly. ByteDance's published answer is the Monolith loop: streaming joins, online training, and minute-level sparse sync [1]. Kuaishou's is PERSIA plus real-time sequence features and TWIN-style lifelong search [46][47].

```
     +---------------- short-video real-time stack (composite) ----------------+
     | client swipe/watch events -> Kafka                                      |
     |   |-> Flink features (counters, last-N actions) -> KV  -> ranker input  |
     |   |-> fresh-item / graph indexes (incremental insert)  -> retrieval     |
     |   '-> Flink joiner(serving feature log) -> online trainer               |
     |          sparse rows: minute-level incremental push -> serving PS       |
     |          dense weights: hourly or slower                                |
     | request: lifelong history (cached, may lag) + last-N actions (fresh)    |
     +-------------------------------------------------------------------------+
```

---

## 37.9 Freshness versus stability

Freshness has costs, and the sources name them explicitly.

1. **Feedback loops and over-reaction.** A model that reacts to the last click can collapse diversity. TransAct reports reduced diversity and uses random time-window masking to stop the model from over-weighting the most recent actions [9]. TikTok describes explicit diversity rules on top of the ranker [51].
2. **Train/serve skew in real-time features.** Streaming features lag by milliseconds to seconds, and batch features have a "midnight" inconsistency until upload finishes [34]. Logging served values and backfilling with the same freshness semantics (SNAPSHOT vs TEMPORAL) is the fix [33][34].
3. **Label bias in fast training.** Short waiting windows create false negatives, and fake-negative pipelines duplicate positives. Both need loss corrections [3][5][6].
4. **Operational risk.** An online trainer can be poisoned by a broken stream. Facebook recommends alarms and automatic trainer cutoff [3]. Monolith keeps a batch path and snapshots for recovery [1].
5. **Cost.** TransAct's real-time encoder raised CPU latency more than 20x and forced a move to GPU serving [10]. Untiled streaming aggregation can force 432,000-event folds per request [32].
6. **Engagement decay.** Real-time models degrade quickly if not retrained. TransAct names engagement decay as a reason for frequent retraining [9].

**How designs balance it.**

- **Pair slow and fast components.** PinnerFormer (daily, trained to be robust to staleness) plus TransAct (real-time) [9][13]. SIM/TWIN lifelong search plus short-term DIN features [45][46]. Instagram's offline item tower plus online user tower [37].
- **Split by parameter type.** Sparse embeddings fast, dense weights slow [1][47].
- **Bound the window.** GraphJet and UTEG keep 24-48 hours, and Thunder trims by retention [20][26].
- **Measure the decay curve first.** Facebook's NE-versus-delay curve [3] and PinnerFormer's real-time-versus-daily Recall@10 gap (13.9% vs 8.3%) [13] are examples of quantifying the value of freshness before paying for it.

---

## 37.10 Comparison of public real-time designs

| System | Org / year | Freshness layer | Stream stack | Window / latency | Scale (public) |
|---|---|---|---|---|---|
| Online joiner [3] | Facebook 2014 | Parameters | Custom joiner | Waiting window | 750M DAU, 1M advertisers |
| Photon [4] | Google 2013 | Labels | Paxos IdRegistry | Seconds (approx.) | Millions of events/min |
| Google News [49] | Google 2007 | Candidates | Bigtable counts | Minutes | News scale |
| Netflix blueprint [27] | Netflix 2013 | Nearline compute | Pub/sub (Hermes) | Seconds | n/a |
| GraphJet / UTEG [18][20] | Twitter 2016-23 | Candidates | Kafka + recos-injector | 24-48 h in RAM | 1M edges/s per server |
| Aggregation fw [24] | Twitter 2023 | Features | Heron + memcached | ~30 min RT windows | ~6k features/candidate |
| Thunder [26] | X 2026 | Candidates | Kafka | Sub-ms lookups | n/a |
| Pixie [15] | Pinterest 2018 | Query-time walks | Real-time seeds | p99 < 60 ms | 100k QPS, 17B edges |
| Manas realtime [16] | Pinterest | ANN index | Incremental HNSW | Seconds (approx.) | Billions of Pins |
| TransAct [9] | Pinterest 2023 | Representation | Real-time action seq | ~100 actions | GPU serving |
| TransAct V2 [11] | Pinterest 2025 | Representation | RT + lifelong | ~16k actions | 103-338x kernel gains |
| Airbnb [30] | Airbnb 2018 | Features | Kafka | ~2 weeks of actions | ~4.5M listings (approx.) |
| Chronon [32] | Airbnb/Stripe | Features | Kafka + Flink tiles | ms-s lag | -33% serving latency |
| Monolith [1] | ByteDance 2022 | Parameters | Kafka + Flink joiner | Minute-level sparse sync | TB-scale tables |
| PERSIA [47] | Kuaishou 2022 | Parameters | Async sparse PS | n/a | Up to 100T params |
| TWIN [46] | Kuaishou 2023 | Representation | Cached item keys | ~10^4 behaviours | 99.3% compute cut |

---

## 37.11 Key takeaways for system designers

1. **Decide which clock you are shortening.** Feature, candidate, representation and parameter freshness have different costs. Many products (Airbnb [30], Pixie [15]) get most of the value from fresh features or fresh query seeds without online training.
2. **Build one canonical action stream first.** A deduplicated Kafka stream with stable request and impression IDs (UUA [22], Recos-Injector [21]) is the foundation for features, graphs and label joins.
3. **Log features at serving time and join by request ID.** This is how Facebook, Photon and Monolith make streaming labels trustworthy [1][3][4]. Make the join idempotent.
4. **Declare features once and compile to batch and stream.** Twitter's AggregateGroup and Chronon's GroupBy are the templates [24][33]. Pre-aggregate into tiles on the write path for hot keys [32].
5. **Use decayed counters and bounded windows.** Exponential half-lives give O(1) state per key [24]. Sliding-window in-memory graphs (GraphJet, UTEG, Thunder) keep fresh candidates cheap [18][20][26].
6. **Put a real-time sequence encoder in the ranker, next to a batch user embedding.** TransAct plus PinnerFormer is the reference hybrid [9][13]. Always include the most recent actions even when you also search a lifelong history [11][45].
7. **Train the stable component to be robust to staleness.** PinnerFormer's dense all-action loss shrank the real-time-vs-daily gap from 13.9% to 8.3% [13].
8. **For online training, sync sparse fast and dense slow.** Use collisionless dynamic tables with admission and expiry for churning ID spaces [1][47].
9. **Handle delayed labels explicitly.** Prefer fake-negative ingestion with FN-weighted or calibrated loss, or elapsed-time sampling, over long waiting windows [5][6][7].
10. **Guard stability.** Use time-window masking or recency dropout to avoid over-reaction [9], diversity rules on the final slate [51], stream anomaly alarms with trainer cutoff [3], and a batch fallback path [1][27].
11. **Respect simple session baselines.** Session kNN and co-visitation counts remain strong for short anonymous sessions [39][49]. Fix losses and negative sampling before adding architecture [42].
12. **Budget the serving cost of freshness.** Real-time encoders can multiply latency (20x at Pinterest) and may require GPUs and fused kernels to ship [10][11].

---

## References

1. Zhuoran Liu et al. (ByteDance). "Monolith: Real Time Recommendation System With Collisionless Embedding Table." arXiv 2209.07663, 2022. https://arxiv.org/abs/2209.07663
2. ByteDance. "Monolith: A Lightweight Recommendation System" (GitHub repository, archived October 2025). Accessed 2026. https://github.com/bytedance/monolith
3. He, X., Pan, J., Jin, O., et al. (Facebook). "Practical Lessons from Predicting Clicks on Ads at Facebook." ADKDD 2014. https://dl.acm.org/doi/10.1145/2648584.2648589
4. Ananthanarayanan, R., et al. (Google). "Photon: Fault-tolerant and Scalable Joining of Continuous Data Streams." SIGMOD 2013. https://research.google/pubs/photon-fault-tolerant-and-scalable-joining-of-continuous-data-streams/
5. Ktena, S. I., Tejani, A., Theis, L., et al. (Twitter). "Addressing Delayed Feedback for Continuous Training with Neural Networks in CTR Prediction." RecSys 2019. https://arxiv.org/abs/1907.06558
6. Yang, J.-Q., et al. (Alibaba). "Capturing Delayed Feedback in Conversion Rate Prediction via Elapsed-Time Sampling." AAAI 2021. https://arxiv.org/abs/2012.03245
7. Gu, S., et al. (Alibaba). "Real Negatives Matter: Continuous Training with Real Negatives for Delayed Feedback Modeling." KDD 2021. https://arxiv.org/abs/2104.14121
8. Chapelle, O. (Criteo). "Modeling Delayed Feedback in Display Advertising." KDD 2014. https://dl.acm.org/doi/10.1145/2623330.2623634
9. Xia, X., et al. (Pinterest). "TransAct: Transformer-based Realtime User Action Model for Recommendation at Pinterest." KDD 2023. https://arxiv.org/abs/2306.00248
10. Pinterest Engineering. "How Pinterest Leverages Realtime User Actions in Recommendation to Boost Homefeed Engagement Volume." Pinterest Engineering Blog, 2023. https://medium.com/pinterest-engineering/how-pinterest-leverages-realtime-user-actions-in-recommendation-to-boost-homefeed-engagement-volume-165ae2e8cde8
11. Pinterest. "TransAct V2: Lifelong User Action Sequence Modeling on Pinterest Recommendation." 2025. https://arxiv.org/abs/2506.02267
12. Pinterest Engineering. "Next-Level Personalization: How 16k+ Lifelong User Actions Supercharge Pinterest's Recommendations." Pinterest Engineering Blog, 2025. https://medium.com/pinterest-engineering/next-level-personalization-how-16k-lifelong-user-actions-supercharge-pinterests-recommendations-bd5989f8f5d3
13. Pancha, N., Zhai, A., Leskovec, J., Rosenberg, C. (Pinterest). "PinnerFormer: Sequence Modeling for User Representation at Pinterest." KDD 2022. https://arxiv.org/abs/2205.04507
14. Pal, A., Eksombatchai, C., Zhou, Y., Zhao, B., Rosenberg, C., Leskovec, J. (Pinterest). "PinnerSage: Multi-Modal User Embedding Framework for Recommendations at Pinterest." KDD 2020. https://arxiv.org/abs/2007.03634
15. Eksombatchai, C., Jindal, P., Liu, J. Z., et al. (Pinterest / Stanford). "Pixie: A System for Recommending 3+ Billion Items to 200+ Million Users in Real-Time." WWW 2018. https://arxiv.org/abs/1711.07601
16. Pinterest Engineering. "Manas HNSW Realtime: Powering Realtime Embedding-Based Retrieval." Pinterest Engineering Blog. https://medium.com/pinterest-engineering/manas-hnsw-realtime-powering-realtime-embedding-based-retrieval-dc71dfd6afdd
17. Koh, T. (Pinterest). "Manas HNSW Streaming Filters." Pinterest Engineering Blog, 2022. https://medium.com/pinterest-engineering/manas-hnsw-streaming-filters-351adf9ac1c4
18. Sharma, A., Jiang, J., Bommannavar, P., Larson, B., Lin, J. (Twitter). "GraphJet: Real-Time Content Recommendations at Twitter." PVLDB 9(13), 2016. http://www.vldb.org/pvldb/vol9/p1281-sharma.pdf
19. Twitter. "GraphJet" README. GitHub, 2016. https://github.com/twitter/GraphJet
20. Twitter/X. "UserTweetEntityGraph (UTEG)" README, the-algorithm, 2023. https://github.com/twitter/the-algorithm/blob/main/src/scala/com/twitter/recos/user_tweet_entity_graph/README.md
21. Twitter/X. "Recos-Injector" README, the-algorithm, 2023. https://github.com/twitter/the-algorithm/blob/main/recos-injector/README.md
22. Twitter/X. "Unified User Actions (UUA)" README, the-algorithm, 2023. https://github.com/twitter/the-algorithm/blob/main/unified_user_actions/README.md
23. Twitter/X. "User Signal Service" README, the-algorithm, 2023. https://github.com/twitter/the-algorithm/blob/main/user-signal-service/README.md
24. Twitter/X. Aggregation framework docs (aggregation.rst, batch.rst, real-time.rst, joining.rst), the-algorithm, 2023. https://github.com/twitter/the-algorithm/tree/main/timelines/data_processing/ml_util/aggregation_framework/docs
25. Twitter Engineering. "Twitter's Recommendation Algorithm." Twitter (X) Engineering Blog, 31 March 2023. https://blog.x.com/engineering/en_us/topics/open-source/2023/twitter-recommendation-algorithm
26. xAI / X. "Thunder" source (in-memory post store), x-algorithm repository, 2026. https://github.com/xai-org/x-algorithm/tree/main/thunder
27. Amatriain, X., Basilico, J. (Netflix). "System Architectures for Personalization and Recommendation." Netflix Technology Blog, 2013. https://netflixtechblog.com/system-architectures-for-personalization-and-recommendation-e081aa94b5d8
28. Amatriain, X. "Blueprints for recommender system architectures: 10th anniversary edition." Machine Learning Times, 2023. https://www.predictiveanalyticsworld.com/machinelearningtimes/blueprints-for-recommender-system-architectures-10th-anniversary-edition/12879/
29. Bhattacharya, M., Lamkhede, S., et al. (Netflix). "Augmenting Netflix Search with In-Session Adapted Recommendations." RecSys 2022 (identifier not re-verified). https://arxiv.org/abs/2206.02254
30. Grbovic, M., Cheng, H. (Airbnb). "Real-time Personalization using Embeddings for Search Ranking at Airbnb." KDD 2018. https://dl.acm.org/doi/10.1145/3219819.3219885
31. Airbnb Engineering. "Listing Embeddings for Similar Listing Recommendations and Real-time Personalization in Search." Airbnb Engineering Blog, 2018. https://medium.com/airbnb-engineering/listing-embeddings-for-similar-listing-recommendations-and-real-time-personalization-in-search-601172f7603e
32. Airbnb / Stripe, Chronon project. "The Tiled Architecture." 2024-2025. https://github.com/airbnb/chronon/blob/main/docs/source/Tiled_Architecture.md
33. Airbnb, Chronon project. "GroupBy" documentation. 2024. https://github.com/airbnb/chronon/blob/main/docs/source/authoring_features/GroupBy.md
34. Airbnb, Chronon project. "Online Offline Consistency." 2024. https://github.com/airbnb/chronon/blob/main/docs/source/test_deploy_serve/Online_Offline_Consistency.md
35. Airbnb, Chronon project. "Chronon on Flink." 2024-2025. https://github.com/airbnb/chronon/blob/main/docs/source/setup/Flink.md
36. DoorDash Engineering. "Building Riviera: A Declarative Real-Time Feature Engineering Framework." DoorDash Engineering Blog, 2021. https://doordash.engineering/2021/03/04/building-a-declarative-real-time-feature-engineering-framework/
37. Vorotilov, V., Shugaepov, I. (Meta). "Scaling the Instagram Explore recommendations system." Engineering at Meta, 2023. https://engineering.fb.com/2023/08/09/ml-applications/scaling-instagram-explore-recommendations-system/
38. Hidasi, B., Karatzoglou, A., Baltrunas, L., Tikk, D. "Session-based Recommendations with Recurrent Neural Networks" (GRU4Rec). ICLR 2016. https://arxiv.org/abs/1511.06939
39. Jannach, D., Ludewig, M. "When Recurrent Neural Networks meet the Neighborhood for Session-Based Recommendation." RecSys 2017. https://web-ainf.aau.at/pub/jannach/files/Conference_RecSys_2017.pdf
40. Kang, W.-C., McAuley, J. (UC San Diego). "Self-Attentive Sequential Recommendation." ICDM 2018. https://arxiv.org/abs/1808.09781
41. Sun, F., et al. (Alibaba). "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer." CIKM 2019. https://arxiv.org/abs/1904.06690
42. Petrov, A., Macdonald, C. (University of Glasgow). "gSASRec: Reducing Overconfidence in Sequential Recommendation Trained with Negative Sampling." RecSys 2023. https://arxiv.org/abs/2308.07192
43. Moreira, G. de S. P., et al. (NVIDIA). "Transformers4Rec: Bridging the Gap between NLP and Sequential / Session-Based Recommendation." RecSys 2021; repository. https://github.com/NVIDIA-Merlin/Transformers4Rec
44. Quadrana, M., Karatzoglou, A., Hidasi, B., Cremonesi, P. "Personalizing Session-based Recommendations with Hierarchical Recurrent Neural Networks." RecSys 2017. https://arxiv.org/abs/1706.04148
45. Pi, Q., Zhu, X., Zhou, G., et al. (Alibaba). "Search-based User Interest Modeling with Lifelong Sequential Behavior Data for Click-Through Rate Prediction" (SIM). CIKM 2020. https://arxiv.org/abs/2006.05639
46. Chang, J., et al. (Kuaishou). "TWIN: TWo-stage Interest Network for Lifelong User Behavior Modeling in CTR Prediction at Kuaishou." KDD 2023. https://arxiv.org/abs/2302.02352
47. Lian, X., et al. (Kuaishou and ETH Zurich). "Persia: A Hybrid System Scaling Deep Learning Based Recommenders up to 100 Trillion Parameters." KDD 2022. https://arxiv.org/abs/2111.05897
48. Zhai, J., et al. (Meta). "Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations" (HSTU). ICML 2024. https://arxiv.org/abs/2402.17152
49. Das, A., Datar, M., Garg, A., Rajaram, S. (Google). "Google News Personalization: Scalable Online Collaborative Filtering." WWW 2007. https://dl.acm.org/doi/10.1145/1242572.1242610
50. Li, L., Chu, W., Langford, J., Schapire, R. E. (Yahoo! Labs). "A Contextual-Bandit Approach to Personalized News Article Recommendation." WWW 2010. https://arxiv.org/abs/1003.0146
51. TikTok. "How TikTok recommends videos #ForYou." TikTok Newsroom, June 2020. https://newsroom.tiktok.com/en-us/how-tiktok-recommends-videos-for-you
52. Cao, et al. (Kuaishou). "Pantheon: Personalized Multi-objective Ensemble Sort via Iterative Pareto Policy Optimization." arXiv 2505.13894, 2025. https://arxiv.org/abs/2505.13894
