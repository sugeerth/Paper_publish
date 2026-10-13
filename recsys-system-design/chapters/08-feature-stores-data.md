# 08. Feature Engineering and Feature Stores: Point-in-Time Correctness, Streaming Features, and Serving-Time Logging

A ranking model is only as good as the features it sees, and in a production recommender the difficult part is rarely the feature formula. The difficult part is computing the same value twice: once offline for millions of historical (user, item, timestamp) training rows, and once online for a few thousand candidates inside a 50-200 ms request budget. Feature platforms exist to make those two computations agree. This chapter surveys the public designs (Twitter's aggregation framework, Uber Michelangelo and Palette, Airbnb Zipline/Chronon, LinkedIn Feathr and Venice, Feast, Tecton, Hopsworks and Featureform) and the techniques they share: declarative feature definitions, point-in-time-correct (PITC) backfills, streaming windowed aggregation with tiles, and logging features at serving time.

A note on sourcing. Most of this chapter is drawn from primary open-source documentation and code: the Chronon, Feast, Feathr, Venice, Hopsworks and Tecton example repositories and Twitter's `the-algorithm` repository [1-16]. Several corporate blogs could not be reached while this chapter was being researched: Uber, Medium-hosted Airbnb posts, DoorDash, Netflix and Google developer pages. Sections that rely on those sources say so explicitly, and their numbers should be treated as approximate.

---

## 1. Why recommenders need a feature platform

Recommender ranking models are fed by four families of features:

1. **Entity attributes**: slowly changing properties of a user or item (country, account age, item category, price), usually from database snapshots.
2. **Windowed aggregates (counters)**: "user's clicks on video in last 1h/24h/30d", "item CTR last 7d", "user x author engagement count with 50-day half-life". These carry most of the predictive power in classic ranking stacks and are the hardest to keep consistent.
3. **Embeddings**: precomputed user and item vectors from two-tower or graph models. They are stored as feature tables and combined at request time, for example as cosine similarity.
4. **Request/context features**: query text, device, time of day, page position, and the candidate's retrieval source. These are only known at request time.

Twitter's Home Mixer says that its For You pipeline does "Feature Hydration — Fetch the ~6000 features needed for ranking" between candidate generation and scoring [5]. Its follow-recommendation service builds a `DataRecord` per (user, candidate) pair after hydration. `DataRecord` is "the Twitter-standard Machine Learning data format used to represent feature data, labels, and predictions when training or serving" [6]. Several thousand features times hundreds to thousands of candidates per request is the workload a feature store must serve.

### 1.1 The three failure modes

Every system in this chapter is built to avoid the same three problems:

- **Label leakage / time travel errors.** A training row for an impression at time t uses feature values computed with data from after t. Feathr's docs call this feature leakage: "Models should not use the future to predict the present... The model will (usually) perform better during training, but it will not perform as good during online serving" [10].
- **Training/serving skew.** The offline pipeline (SQL in a warehouse) and the online pipeline (Java/Scala in a service) implement "the same" feature differently. They may differ in null handling, time zones, window boundaries, dedup rules or freshness lag.
- **Duplicated, unshared pipelines.** LinkedIn's Feathr talk states the problem directly: "The cost of building and maintaining feature pipelines was borne redundantly across many teams... no common type system, no common feature namespace" [12].

### 1.2 Two naive strategies and their trade-offs

Chronon's documentation frames the design space around two approaches teams take without a platform [3]:

| Approach | How it works | Pros | Cons |
|---|---|---|---|
| Log-and-wait | Compute features in the serving path; log them to the warehouse; train on the logs once enough data has accumulated | Train/serve consistency by construction; can use service-call and request-context features | Weeks to months before training data exists; large windowed aggregations hard in the app layer; cannot use historical warehouse data |
| Replicate offline-online | Train on warehouse SQL; re-implement the features online | Uses all historical data; warehouse is good at heavy aggregation | "Often very error prone, resulting in inconsistent data between training and serving"; complex infra; real-time windowed aggregations are hard |

Every platform below is an attempt to get the benefits of both columns. You write one declarative definition. It compiles into a PITC backfill for training and an online materialization for serving, and serving-time logs are used to verify, or substitute for, the backfill.

---

## 2. Core concepts and vocabulary

Different systems use different names for the same ideas:

| Concept | Chronon | Feast | Feathr | Hopsworks | Tecton | Twitter agg. framework |
|---|---|---|---|---|---|---|
| Unit of feature definition | `GroupBy` | `FeatureView` / `StreamFeatureView` | `FeatureAnchor` | Feature group | Batch/Stream FeatureView | `AggregateGroup` |
| Model-level feature set | `Join` | `FeatureService` | feature query (list of names) | Feature view | `FeatureService` | config of groups |
| Training "left side" | left source / query log | entity dataframe | observation data | spine / label FG | spine dataframe | training DataRecords |
| Request-time features | contextual / external parts | on-demand FV | (derived, UDF) | on-demand transformation | `realtime_feature_view` | (hydration in service) |
| Online store | pluggable KV store | Redis, DynamoDB, Bigtable, Cassandra, etc. | Redis, Cosmos DB, Aerospike | RonDB | managed | Manhattan (batch), memcached (real-time) |

Sources: [1-4, 7-11, 13-16].

**Point-in-time join.** For each left-side row (key, ts), find for each feature the value that would have been observed online at ts. That means the most recent value with event time < ts, and for aggregates, the window ending at ts. Feast scans "backward in time from the entity dataframe timestamp up to a maximum of the TTL". The TTL is "relative to each timestamp within the entity dataframe", not to query time [7]. Hopsworks uses event_time columns "to determine the most recent (but not newer) feature values" [14].

**Event time vs. availability time.** This subtlety is often missed. Feast's default PIT join constrains only the event timestamp. A value that was "backfilled or corrected after an entity dataframe timestamp can still be returned for it" unless you pass `filter_by_created_timestamp=True`, which restricts retrieval to values already available at the row's timestamp [7]. In production, a daily batch feature for day D only lands in the online store some hours after midnight. A truly faithful backfill must model that landing lag, not just the event date.

---

## 3. Twitter: the aggregation framework (batch + real-time counters)

Twitter's open-sourced `the-algorithm` contains one of the clearest public descriptions of a production counter-feature system for recommendation [4, 17].

**Problem.** The Home Timeline heavy ranker, MagicRecs (push) and EmailRecs all need user, user x author and tweet engagement aggregates, conditional on content type and engagement label. The README notes that "these types of engineered aggregate features have proven to be highly impactful across different teams at Twitter" [4].

**Abstraction: `AggregateGroup`.** It behaves like a SQL GROUP BY with ML-specific knobs [17]:

- `keys`: grouping keys such as `USER_ID`, `(USER_ID, AUTHOR_ID)` or `(USER_ID, ENGAGER_ID)`. Sparse-binary keys get "flatMap" semantics. Grouping by `(USER_ID, AUTHOR_INTEREST_IDS)` emits one record per (user, interest) pair.
- `features`: the binary or continuous features to aggregate (for example `HAS_PHOTO`).
- `labels`: the engagement conditioning the count (for example `IS_FAVORITED`). The aggregate is a feature x label cross.
- `metrics`: `CountMetric`, `SumMetric` and others.
- `halfLives`: exponential decay instead of hard windows. The example group computes "the number of times each user has faved a tweet with a photo... decayed with a 50 day halflife" [17].

Decayed counters are a notable design choice. They are monoid-friendly: an old value plus elapsed time is enough to roll forward. Each key stores O(1) state instead of a window of events.

**Architecture.**

```
                 client events (Kafka "julep" client-event topics)
                         |                              |
          daily DataRecord dumps (HDFS)        Storm/Heron (summingbird)
                         |                              |
   summingbird-scalding batch job          real-time aggregate job
   (DataRecordAggregationMonoid)           (same AggregateGroup config)
                         |                              |
        offline aggregate store (HDFS)         memcached (RTA cache)
                         |                              |
            upload -> Manhattan KV                      |
                         \______________  ______________/
                                        \/
                    Home Mixer / pushservice feature hydration
                         (~6000 features per ranked candidate)
```

The same `AggregateGroup` definitions drive a daily summingbird-scalding batch job (outputs uploaded "to Manhattan for online hydration") and a summingbird-storm real-time job backed by memcached [4, 17]. The real-time job subscribes to client-event Kafka topics, converts events to `DataRecord`s and can enrich them mid-stream. MagicRecs left-joins each engagement with a user-history store before aggregating [17]. Offline, training data joins aggregates with `joinWithLarger((USER_ID, AUTHOR_ID), ...)`, and `AggregatesV2MostRecentFeatureSourceBeforeDate` reads "the most recent aggregates before a provided date". This is a day-granularity form of point-in-time correctness [17].

**Feature identity by hash.** Twitter predefines feature names and types so that "when features are being generated, scribed or used to score they can be identified with only a hash of their name" [18]. This keeps logged and served payloads compact. Feature logging ("scribing") is treated as a first-class path.

**Event backbone.** Unified User Actions (UUA) is "a centralized, real-time stream of user actions". It merges client- and server-side events into one Kafka stream, which is replicated to HDFS, Pub/Sub, GCS and BigQuery [19]. The User Signal Service standardizes explicit and implicit signals for retrieval and ranking features [20]. Graph Feature Service answers pairwise graph features such as "how many of A's followings have favorited C" [21].

**Lessons.** (1) One definition, two engines (batch Scalding and streaming Storm) is the oldest form of the "write once" idea. (2) Decayed counters avoid storing windows. (3) Day-boundary PITC ("most recent before date") is cheap but gives up intra-day freshness. Chronon later made that trade-off configurable.

---

## 4. Uber Michelangelo and Palette

*Primary pages for this section (Uber engineering blog and InfoQ) could not be fetched during research. The description follows the widely cited 2017 Michelangelo post [22] and the 2019 Palette talk [23]. Treat specifics as approximate.*

**Problem.** Uber's 2017 Michelangelo post describes an end-to-end ML platform: manage data, train, evaluate, deploy, predict, monitor. Feature management was singled out as the hardest part to make reusable [22].

**Architecture (as publicly described).**

```
  Offline (batch)                           Online / near-real-time
  ----------------                          -----------------------
  HDFS / Hive data lake                     Kafka event streams
        |                                         |
  Spark / Hive SQL jobs                     streaming jobs (Samza, later Flink)
        |                                         |
  Feature tables (Hive) --- batch upload ---> Cassandra online feature store
        |                                         |
  training data generation               prediction service fetches features
        \___________________   ___________________/
                            \ /
               Feature DSL (select + transform, applied
                identically at training and serving)
```

**Key design elements.**

- **Shared feature store.** Teams publish features to a central store and reference them by name in model configs. The 2017 post reported on the order of 10,000 features in the store (approximate).
- **Two pipelines.** A batch path precomputes features from the data lake and periodically bulk-loads them into an online KV store (Cassandra). A near-real-time path computes streaming metrics, for example a restaurant's recent average meal-prep time, from Kafka and writes them to the same store. The streaming path also logs values back to HDFS for training.
- **Feature DSL.** Feature selection and lightweight transformations (for example normalization or day-of-week extraction) are expressed in a restricted DSL. The DSL is part of the model configuration and is applied identically at training and prediction time. This is the main defense against transformation skew.
- **Palette (2019).** Palette is the evolution of the Michelangelo feature store into a feature engineering platform. It formalizes feature definitions over batch, near-real-time and RPC/external sources, and adds a catalog, ownership and orchestration [23].

**Lessons.** Michelangelo popularized two ideas that every later system copies: the dual offline/online store, and putting transformations inside the model artifact so they are applied the same way everywhere. Hopsworks later called these model-dependent transformations (Section 9). Tecton grew directly out of this lineage.

---

## 5. Airbnb Zipline to Chronon

Zipline was Airbnb's internal feature platform, presented publicly in 2018 and 2020 [24, 25]. It was rebuilt and open-sourced as **Chronon**, which the project says powers "all major ML applications within Airbnb, as well as major use cases at Stripe", jointly maintained by both companies [2]. Chronon's internal design docs still use the name "zipline" [1].

### 5.1 Data model

Chronon distinguishes sources by their temporal semantics [1, 26]:

- **EventSource**: an immutable log, for example purchases. It needs a date-partitioned Hive table, where each partition holds only that day's events, plus optionally a stream topic.
- **EntitySource**: a mutable table, for example users or listings. It needs a daily snapshot table, and for intra-day accuracy a mutation (CDC) table plus a mutation topic. In the docs' terms these are "fact/measure" sources versus "dimension" sources [1].
- Request context and external service calls ("contextual" and "external" parts of a `Join`), which cannot be backfilled natively [27].

A **GroupBy** aggregates a source by keys over windows (SUM, COUNT, AVERAGE, LAST_K, approx percentiles via KLL sketches, and others) [2, 28]. A **Join** assembles many GroupBys onto a left source. That left source is either the events that drive predictions (for example checkouts or searches) or the logged online queries. One definition produces three things: training backfills, serving endpoints and monitoring pipelines [2].

```python
v1 = GroupBy(
    sources=[Source(events=EventSource(table="data.purchases",
                                       topic="events/purchases",
                                       query=Query(selects=..., time_column="ts")))],
    keys=["user_id"], online=True,
    aggregations=[Aggregation(input_column="price", operation=Operation.SUM,
                              windows=[Window(d, TimeUnit.DAYS) for d in [3, 14, 30]]),
                  Aggregation(input_column="price", operation=Operation.LAST_K(10))])
```

(abridged from [2])

### 5.2 Accuracy: SNAPSHOT vs TEMPORAL

`accuracy` is an explicit GroupBy toggle [28]:

- `SNAPSHOT`: values computed as of midnight and refreshed daily. The backfill for a row at ts uses the value as of the prior midnight. That is exactly what the online store would have served if updates land only in batch.
- `TEMPORAL`: real-time updates online, and millisecond-accurate point-in-time backfills offline. This is the default when a `topic` or `mutationTopic` is set.

The Join docs give the accuracy for each (left source type, right source type) combination. For example, left events with right batch-only events default to midnight-accurate unless TEMPORAL is requested [27]. Making the online freshness model explicit in the backfill semantics is the key idea. The backfill does not compute the "true" value at ts. It computes the value the online system would have served at ts.

### 5.3 The backfill algorithm: from quadratic joins to sawtooth windows

Chronon's "Window Tiling Algorithm" doc is the most detailed public treatment of how to make PITC backfills scale to recommender training sets [1]:

1. **Naive SQL.** Join queries to events on key with `event.ts in [query.ts - W, query.ts)` and group by query. For skewed keys this is quadratic, and daily rollups only reduce "trillions" of operations to "billions".
2. **Sort-and-scan.** Two cursors give O(n log n) time, but only for subtractable aggregates such as sum and count, not min, max or last. It also requires holding all events in memory.
3. **Interval tree with broadcast join.** Labels (queries) are usually far fewer than events, "at least two orders of magnitude". Queries are broadcast to all workers, events are streamed through an interval tree over sorted query timestamps with O(log n) updates per event, and the results are reduced. Row orientation (one tree traversal updating hundreds of features) beats column orientation.
4. **But recommenders break the assumption.** "There is a class of very impactful models that this is not true for - search ranking & recommendation systems... the labelled training data can be as large as all the searches on the platform." Both sides are now too big for memory [1].
5. **Sawtooth windows.** Sliding windows are fresh but hard to reuse. Hopping windows are reusable but stale. Chronon combines them: the window is `[round(query_ts - W, hop), query_ts)`. The *tail* is hop-aligned and precomputed once per hop bucket, shared by every query in that 5-minute bucket. The *head* (from the last hop boundary to query_ts) is computed with a sorted merging scan. This avoids quadratic compute while keeping the head exact to the millisecond.

```
 window W, hop h:
 |<------------- tail: precomputed per hop bucket ------------->|<- head ->|
 [round(ts - W, h) ............................... round(ts, h)) [.. ts)
   shared by all queries whose ts falls in the same h-bucket      exact, per query
```

The online store mirrors these semantics. Batch uploads hold hop-aligned partial aggregates as of the batch end. Streaming updates cover everything after that end. "Streaming fetches replay everything after the batch end date" [28].

### 5.4 Online architecture: untiled vs tiled (Flink)

The Tiled Architecture doc quantifies the serving problem [29]:

- **Untiled.** Spark Streaming or Flink writes individual events to the KV store, and the fetcher reads O(events) and aggregates at request time. With 10 events/sec for one key and a 12-hour window, a single request must fetch and fold 432,000 events.
- **Tiled.** A stateful Flink job pre-aggregates events into tiles of intermediate representations (IRs), for example (sum, count) for AVG, and writes the tiles. The fetcher reads O(tiles). The same request reads 12 or 13 one-hour tiles. Stripe reported a 33% reduction in feature-serving latency after the initial implementation. Tiling also helps hot keys (for example big merchants) and stores without range scans. "If your hottest keys don't exceed a few thousand events per day, the untiled approach may still be sufficient" [29].

```
 Kafka -> Flink: [Source] -> [Spark-SQL eval (CatalystUtil)] -> [Window/tiling op]
                                                                  |  IRs per tile,
                                                                  v  emitted per event
                                         [Avro encode] -> [async KV sink]
 Fetcher: get(key, tiles in [now-W, now)) -> merge IRs -> finalize -> feature values
 e.g. COUNT/LAST over 3h, 1h tiles: [2,"B"] + [9,"A"] + [3,"C"] => [14, "C"]
```

The Flink job evaluates the same Spark SQL selects as the batch engine by embedding Catalyst inside Flink [30]. This is how one expression language is guaranteed across batch and streaming.

### 5.5 Measuring online/offline consistency from serving logs

Chronon logs every fetch, as Avro-encoded key bytes, value bytes, ts and caller, to a `query_logs` table partitioned by ds and join name [31]. A daily Spark job then:

1. Decodes the logs into `<join>_logged`.
2. Re-runs the join offline using the logged table as the left side.
3. Compares logged values against backfilled values per feature. It computes mismatch, missing and extra rates; SMAPE and delta distributions for numerics; and Levenshtein edit distance for sequences such as LAST_K. Distributions are summarized with ReqSketch, which was chosen over t-digest because it is more accurate at all ranges and more than 2x faster to update and serialize [31].

Logging sampling is key-based: `hash(key_bytes) % 100 * 1000 <= sample_percent * 1000`. This captures all queries for a subset of keys rather than a random subset of queries, which keeps the join's bloom filter small [31]. The doc names the inherent sources of skew. Kafka plus KV write latency means streaming features lag by milliseconds to seconds. Batch features are inconsistent "centered around midnight until the bulk upload of new features to kv store finishes". Proposed future work is a per-GroupBy write-lag that shifts offline query timestamps to match [31].

### 5.6 Bootstrap: log stitching

Chronon's **Bootstrap** primitive lets a Join take precomputed values from tables, including serving logs, for any (row, column) they cover. It backfills only the rest [32]. Listed scenarios include "Log stitching: Use online serving log as the source for production features instead of continuously recomputing it offline to ensure online/offline consistency and reduce computation cost"; backfilling external or contextual features before they have been logged; sharing data between production and experimental Joins; and recovering from logging outages [32]. To make logs joinable, row IDs such as an `event_id` are registered as contextual features. They are then logged with the served features [27].

Chronon also supports **chaining**: the output of a Join can feed another GroupBy, for example "average price of the last 10 items that a user purchased" (LAST_K followed by AVERAGE), served in real time [33].

**Lessons from Chronon.** (1) Make the online freshness model part of the offline semantics (SNAPSHOT vs TEMPORAL). (2) Recommender-scale PITC needs specialized algorithms such as sawtooth windows, not generic SQL. (3) Pre-aggregate on the write path (tiles of IRs) when keys are hot. (4) Logging is part of the platform, used both to measure skew and to replace recomputation.

---

## 6. LinkedIn: Feathr (and Venice as the serving substrate)

**History and scale.** According to the Feathr community talk, LinkedIn began Feathr development in 2017, saw broad internal adoption in 2018, and had the "majority of LinkedIn ML applications onboarded" by 2020. It was open-sourced with Azure integration in 2022 and joined LF AI & Data [12]. At LinkedIn it supports "hundreds of models", "thousands of features", "many kinds of entities (economic graph)" at "petabyte scale" [12]. Reported impact: experimentation with new features went "from weeks to days", running time improved "as much as 50%" over custom pipelines, and features became shareable across teams [12]. The model deployment service uses Feathr to ensure "a model's feature dependencies are deployed, before deploying the model" [12]. Use cases named in the talk include LinkedIn Jobs hiring preferences, LinkedIn Learning recommendations, and near-real-time personalization [12].

**Abstractions** [9, 10]:

- **Observation data**: the training "left side" of keys and timestamps, for example a click stream. Other stores call this the entity dataframe.
- **Source**: batch (HDFS, Delta, Parquet, etc.) or streaming (Kafka, Event Hubs).
- **Anchor** (`FeatureAnchor`): a set of features defined on one source with row-level transforms (`"cast_float(trip_distance)>30"`) or sliding-window aggregations.
- **Derived feature**: a feature computed from other features. The docs' recommender example is `user_item_similarity` built from a user embedding and an item embedding owned by different teammates [9].
- **Registry**: named features shared across teams, "like a package manager for feature engineering". Model code requests `FeatureQuery(feature_list=[...], key=item_id)` [12].

**Point-in-time join.** For each observation row, Feathr looks up the feature value "closest from the past" for the same key. This includes repeated keys at different times, for example the same user with labels on 05/01 and 05/03 getting different feature snapshots [10]. Feathr's docs contrast this with simple "time travel" to a single global timestamp. The per-event join wastes no data and is what Feathr implements at scale, with built-in optimizations "such as bloom filters and salted joins" for billions of rows [8].

**Serving substrate: Venice.** LinkedIn's open-source Venice is a "derived data storage platform" with high-throughput asynchronous ingestion from batch (Hadoop) and streaming (Samza) sources, low-latency reads "via remote queries or in-process caching", active-active cross-region replication with CRDT-based conflict resolution, and multi-tenancy. The project describes it as "particularly suitable as the stateful component backing a Feature Store, such as Feathr" [13]. The pattern is a write-optimized bulk/stream ingestion path plus a read-optimized, possibly in-process, store. It reflects the reality that recommendation feature stores are read-dominated. Every request fans out to thousands of candidate keys.

```
  HDFS / Spark (offline features) ----bulk push----\
                                                    >---> Venice (derived data store)
  Kafka -> Samza (nearline features) ---stream-----/        |  remote reads or
                                                             |  Da Vinci in-process cache
  Feathr definitions (anchors, derived, registry)            v
        |-- offline: PIT join onto observation data --> training sets
        `-- online: feature fetch for inference <------ ranking services
```

---

## 7. Feast: the open-source reference design

Feast (**Fea**ture **St**ore) originated at Gojek with Google Cloud in 2019 [34]. Its stated purpose is to make "features consistently available for training and serving" through an offline store, an online store and a feature server, and to "avoid data leakage by generating point-in-time correct feature sets" [35].

**Architecture choices** [36, 37, 38]:

- **Push model.** Producers push feature values to Feast, which stores them in the online store. Feast does not pull from producers at request time, because that latency "would be at least as long as your slowest call". The trade-off is stated plainly: "strong consistency is not guaranteed out of the box" [36].
- **Write patterns.** Sync vs async writes crossed with precomputed vs on-demand vs hybrid transformations. Hybrid is recommended for "time since last" features [37].
- **Transformation engines.** The feature server, the offline store (Snowflake, BigQuery, Spark, DuckDB...) or a compute engine. The docs warn that "different feature transformation code may be used under different transformation engines". This is a candid admission that the multi-engine design can itself cause skew [38].
- **Language.** Python for the feature server, on the argument that serving should be "a lightweight database lookup" over precomputed values. Java and Go clients are also available [39].
- **Online store backends.** As of 2026 the list includes Redis, Valkey, Dragonfly, DynamoDB, Bigtable, Datastore, Cassandra, ScyllaDB, HBase, Aerospike, Postgres, MySQL, MongoDB, Couchbase, Hazelcast, SingleStore, Snowflake, Elasticsearch, and vector stores such as Milvus, Qdrant and FAISS. There is also a Chronon bridge in which Chronon owns materialization and serving and Feast routes `get_online_features` to Chronon's join or groupby endpoints [40, 41].

**Point-in-time joins.** `get_historical_features(entity_df, features)` scans backward from each entity row's timestamp up to the feature view's TTL. Rows older than the earliest feature or outside the TTL get nulls. `created_timestamp` dedups ties, and `filter_by_created_timestamp=True` enforces availability-time correctness [7].

**Streaming and tiling.** `StreamFeatureView` supports time-windowed aggregations over Kafka, Kinesis or push sources (Local and Spark compute engines; Flink is listed as not yet supported) [42]. Feast's tiling docs make the same IR argument Chronon does: avg(tile1, tile2) is not (avg1 + avg2)/2. In their example, tiles [10, 20, 30] and [100] give a correct average of 40 but a naive one of 60. So tiles must store (sum, count) for avg, and analogous sufficient statistics for std and var. The docs claim reuse of "90%+ of tiles between updates" [43].

**Serving-time logging and monitoring.** A `FeatureService` can declare `logging_config=LoggingConfig(destination=..., sample_rate=...)`. Logged serving data can then feed data-quality monitoring (`feast monitor run --source-type log`) next to batch baselines [44].

**Labels as first-class objects (2026).** Recent Feast adds a `LabelView` that stores mutable "judgments" (rewards, human reviews, LLM-judge scores) separately from append-only feature observations. It supports multiple labelers and conflict policies, and is composed with feature views in a `FeatureService` for PIT-correct training retrieval [45]. This matters for recommenders whose labels arrive late or get corrected, for example delayed conversions or moderation outcomes.

**Inference patterns.** Feast's docs list four ways to serve predictions: online inference with online features; precomputed predictions served as a feature; online inference with cached predictions that are recomputed when producers write; and inference without features [46]. The third pattern resembles the "score on write" caches used for some recommendation surfaces.

---

## 8. Tecton: managed real-time feature platform

*Tecton's own blog was unreachable during research. This section uses its public example repository [15].*

Tecton grew out of the Michelangelo lineage described above. Its public examples show the declarative surface for recommender and ads features [15]:

- **Batch feature views**, for example `user_distinct_ad_count_7d`.
- **Stream feature views** with built-in windowed `Aggregate`s, for example ad-click counts over 1h, 24h and 72h windows fed by a push (ingest API) source. Flags `online=True, offline=True` materialize to both stores.
- **Continuous stream processing mode** (`StreamProcessingMode.CONTINUOUS`) for low-latency counters such as 1-minute and 5-minute keyword-click counts, with `AggregationLeadingEdge.LATEST_EVENT_TIME` controlling how the window's leading edge is defined.
- **Realtime (on-demand) feature views** that combine a `RequestSource` with precomputed feature tables at request time. The example computes cosine similarity between a request-time query embedding and a stored user embedding. This is the standard way to add "user-query affinity" to a ranker without materializing every pair.
- **Feature tables** for embeddings (`user_embeddings`, `ad_embeddings`) written by external jobs.
- **Feature services** that bundle features per model, for example `ad_ctr_feature_service`.
- **Two compute options**: Spark/Databricks, or "Rift", a Python-native engine for batch, streaming and real-time features without a Spark cluster [15].

The design lesson matches Chronon and Feast. Aggregation semantics (window, function, leading edge) are declared, not coded. That lets the platform compute them identically in backfill, streaming and serving.

---

## 9. Hopsworks: FTI pipelines and a transformation taxonomy

Hopsworks contributes a useful conceptual frame [14, 16, 47, 48]:

- **FTI architecture.** Every ML system splits into **F**eature pipelines (data to feature data), **T**raining pipelines (features to model) and **I**nference pipelines (features plus model to predictions plus prediction logs). These connect through a feature store with a row-oriented online store, a columnar offline store and a vector index, plus a model registry [47].
- **Transformation taxonomy.**
  - *Model-independent* transformations (aggregations, windowed counts, binning) produce reusable features and belong in the feature store.
  - *Model-dependent* transformations (scaling, one-hot, tokenization) are parameterized by a training set. They should not be stored, because storing them causes write amplification (every new write would force re-encoding of existing rows). They must be applied identically at training and inference.
  - *On-demand* transformations need request-time inputs, such as `time_since_last_transaction`. They run online, and the same function can be used to backfill them.
  - The taxonomy is meant to locate exactly where skew can arise [48].
- **Write path for consistency.** The Stream API writes to Kafka first. Hopsworks then syncs to the online store (RonDB, latest values only, idempotent upserts) and the offline store (Hudi with dedup). This upgrades Kafka's at-least-once delivery to effectively exactly-once and keeps online and offline consistent [16].
- **PIT training data with spines.** Feature views join feature groups updated at different cadences (hourly vs daily) with a transparent PIT join keyed on event_time. "Spine groups" define precisely which entities and timestamps appear on the left, which avoids duplicate-entity bias [14].

---

## 10. Featureform: the virtual feature store

Featureform describes itself as a **virtual feature store**. It "sits atop your existing infrastructure and orchestrates it to work like a traditional feature store". It manages definitions, versions ("variants"), lineage and immutability, and leaves computation to existing engines and storage [49]. Its README cites a taxonomy of "three common architectures" for feature stores. In this chapter's terms, the space looks like this:

| Archetype | Examples | Who computes features | Pros | Cons |
|---|---|---|---|---|
| Literal store | early Feast | external pipelines; store only holds values | simple, infra-agnostic | skew risk stays with users |
| Physical platform | Chronon, Tecton, Feathr, Hopsworks, Michelangelo/Palette | platform engines (Spark/Flink/Rift) from declarative defs | strongest consistency guarantees; PITC backfills built in | heavier infra; DSL limits |
| Virtual store | Featureform | existing infra, orchestrated | reuse existing warehouse and streams | consistency depends on underlying engines |

---

## 11. Netflix, DoorDash and others (secondary pointers)

*These primary posts could not be fetched during research. They are listed for completeness, without detailed claims.*

- **Netflix: "Distributed Time Travel for Feature Generation" (2016)** [50]. A snapshot-based approach: periodically snapshot the online services' state for sampled contexts, then regenerate features "as of" those snapshots, so new features can be evaluated on historical contexts. It sits between log-and-wait and full replication.
- **DoorDash Fabricator (2022) and Riviera (2021)** [51, 52]. Declarative batch and real-time (Flink-based) feature frameworks.
- **DoorDash gigascale feature store with Redis (2020)** [53]. Covers Redis benchmarking, binary serialization and compression for very large online feature sets.
- **LinkedIn near-real-time features (2022)** [54]. Nearline feature computation for personalization, also cited in the Feathr talk [12].
- **Twitter's feature store talk (2019)** [55], Condé Nast [56], Monzo [57], QuintoAndar Butterfree [58] and Constructor.io [59] are further industrial case studies.

---

## 12. Cross-cutting design patterns

### 12.1 Serving-time feature logging ("log what you serve")

Logging the exact feature vector used at inference is the strongest skew defense available. Google's *Rules of ML* recommends saving the features used at serving time and piping them to training (Rule #29; source not re-fetched for this chapter) [60]. The public systems implement it in different ways:

- **Twitter** scribes `DataRecord`s with hashed feature IDs [18].
- **Chronon** logs fetch requests and responses, uses them for consistency metrics, and can *bootstrap* training rows directly from logs [31, 32].
- **Feast** offers `LoggingConfig` per feature service for monitoring [44].
- **Hopsworks** puts "prediction logs" in the inference pipeline's output contract [47].

A practical design that combines these patterns:

```
 request --> retrieval --> feature fetch (KV + request ctx + on-demand fns)
                              |                         |
                              |                   ranker scores
                              v                         v
                 log {request_id, ts, keys,      impressions shown
                      feature_vector, model_ver}        |
                              |                         v
                              |             engagement events (Kafka)
                              v                         |
                   warehouse: served_features     labels (delayed join
                              \                    with attribution window)
                               \_______  __________/
                                       \/
                     training rows = logged features + labels
                     (+ bootstrap/backfill only for NEW features via PITC)
```

Trade-offs of pure logging. You cannot add a new feature to past data, which needs a PITC backfill (hence Chronon's bootstrap mix of logged and backfilled columns). Logs for thousands of features times every scored candidate are enormous. Sampling is needed, and Chronon samples by key hash so that whole key histories are kept [31]. Logging also captures only what was scored, so retrieval-stage counterfactuals are missing.

### 12.2 Streaming feature computation

- **Ingestion backbone.** A unified, deduplicated user-action stream such as Twitter UUA [19] or Kafka at Airbnb, Stripe and LinkedIn. Feature jobs should consume a canonical event schema rather than raw client logs.
- **Engines.** Storm/Heron (Twitter, historical) [17], Samza (LinkedIn) [13], Spark Streaming and Flink (Chronon) [30], Spark (Feast's stream processor) [42], Rift (Tecton) [15].
- **State model.** Choose decayed counters (O(1) state, Twitter [17]), raw-event storage with read-side aggregation (simple, but O(events) per read), or tiles of IRs (write-side pre-aggregation; Chronon and Feast [29, 43]).
- **Mergeability.** Only algebraic aggregates (sum, count, min, max, last, and avg via sum and count) tile exactly. Holistic ones (percentiles, distinct counts) need sketches such as KLL or HLL. Chronon exposes a memory/accuracy knob for approximate aggregations [28].
- **Hot keys.** Popular items and large merchants dominate traffic. Tiling and write-side aggregation are the standard fix, and Chronon cites "very hot key entities" as a primary motivation [29].
- **Skew from lag.** Streaming lag (ms to seconds) and batch landing lag (hours) are inherent. Either model them in the backfill (Chronon's proposed write-lag shift [31], Feast's created-timestamp filter [7]) or log what was served.

### 12.3 Point-in-time correctness checklist

1. Every source carries an event time, and if possible an availability or ingestion time.
2. Backfills reproduce the online freshness model (midnight snapshot vs real-time).
3. Windows are defined identically in both engines, including inclusive/exclusive bounds. Chronon uses `[ts - W, ts)` [1].
4. Mutable entities use snapshots plus CDC mutations, not the current table state [1, 26].
5. TTL and staleness semantics match. Feast's TTL is relative to each row [7].
6. Labels are joined with an explicit attribution window, and label corrections are versioned (Feast `LabelView` [45]).
7. Spines define the training population explicitly so it is not inferred from label tables [14].

### 12.4 Online store engineering for recommenders

- Ranking is read-heavy fan-out: one request reads features for 1 user plus N candidates (N in the hundreds to thousands). Batch multi-get, co-locate features for the same entity, and cache item features in-process. Venice's Da Vinci in-process client is one example [13].
- Encode compactly: hashed feature IDs (Twitter [18]), Avro (Chronon [31]), binary serialization and compression (DoorDash [53]).
- Store embeddings as feature tables and compute pairwise similarities on demand (Tecton [15], Feathr derived features [9]) instead of materializing user x item.

---

## 13. Comparison of public feature platforms

| System | Org / year public | Definition API | Batch engine | Streaming engine | Online store | PITC backfill | Serving-log support | Notable numbers (public) |
|---|---|---|---|---|---|---|---|---|
| Aggregation framework | Twitter, 2023 (open-sourced code) | Scala `AggregateGroup` | summingbird-scalding | summingbird-storm/Heron | Manhattan (batch), memcached (RT) | day-boundary ("most recent before date") | features scribed by hashed ID | ~6000 features hydrated for For You ranking [5] |
| Michelangelo / Palette | Uber, 2017 / 2019 | config + feature DSL | Spark/Hive | Kafka + Samza (later Flink) | Cassandra | yes (batch + logged streaming) | streaming features logged to HDFS | ~10k features in store (2017, approx.) [22] |
| Zipline / Chronon | Airbnb (+Stripe), 2018 / OSS 2024 | Python `GroupBy` / `Join` | Spark (sawtooth tiling) | Spark Streaming or Flink (tiled) | pluggable KV | yes, SNAPSHOT or TEMPORAL | query logs, consistency metrics, bootstrap | 12h window: 432k events vs 12-13 tiles; Stripe -33% latency [29] |
| Feathr | LinkedIn, 2017 internal / OSS 2022 | Python anchors + derived | Spark (bloom filters, salted joins) | Kafka/Event Hubs streaming | Redis, Cosmos DB, Aerospike; Venice at LinkedIn | yes (per-event "closest from past") | (not documented) | hundreds of models, thousands of features, PB scale; up to 50% runtime gain [12] |
| Feast | Gojek + Google Cloud 2019; OSS | Python `FeatureView` | offline store / Spark / DuckDB etc. | Spark stream processor; push sources | 20+ backends | yes (TTL, created-ts filter) | `LoggingConfig` + monitoring | tile reuse "90%+" claim [43] |
| Tecton | commercial | Python decorators | Spark or Rift | Spark or Rift (continuous mode) | managed | yes | (not covered in sources) | -- |
| Hopsworks | commercial / OSS core | Python feature groups / views | Spark, Python | Kafka + OnlineFS; Flink/Spark | RonDB | yes (event_time + spines) | prediction logs in FTI | -- |
| Featureform | OSS | Python | orchestrates existing | orchestrates existing | existing | via underlying infra | -- | -- |

---

## Key takeaways for system designers

1. **Declare features once, compile everywhere.** Every successful platform (Twitter's AggregateGroup, Michelangelo's DSL, Chronon's GroupBy, Feathr's anchors, Feast and Tecton feature views) makes a single declarative definition the source of truth for backfill, streaming and serving. Hand-written duplicates are the main source of skew.
2. **Point-in-time correctness means "what would online have served at ts".** It does not mean "the true value at ts". Model batch landing lag and streaming lag explicitly, using Chronon's SNAPSHOT vs TEMPORAL and Feast's created-timestamp filter.
3. **Recommender training sets break generic PITC joins.** When labels are as numerous as impressions or searches, use specialized algorithms (sawtooth windows with hop-aligned shared tails and exact heads) and broadcast or bloom-filter tricks, not naive range joins.
4. **Pre-aggregate on the write path for hot keys.** Tiles of intermediate representations (sum and count, not avg; sketches for percentiles) turn O(events) reads into O(tiles). Decayed counters give O(1) state when hard windows are not required.
5. **Log what you serve.** Serving-time feature logs are the ground truth for consistency measurement and the cheapest source of training data for existing features. Use PITC backfills only for new features, and stitch the two together (Chronon bootstrap). Sample by key, not by request.
6. **Classify transformations.** Store model-independent features. Bundle model-dependent transforms with the model. Run on-demand features at request time with the same code used for backfill.
7. **Design the online store for fan-out reads.** Batch multi-gets, compact encodings (hashed IDs, Avro, compression), in-process caches for item features, and on-demand similarity over stored embeddings.
8. **Measure skew continuously.** Mismatch, missing and extra rates and value-distribution deltas per feature, computed daily from logs (Chronon's consistency job), catch silent breakages that offline AUC never will.

---

## References

1. Airbnb / Chronon project, "Window Tiling Algorithm" (docs/source/window_tiling.md), Chronon repository, 2023-2024. https://github.com/airbnb/chronon/blob/main/docs/source/window_tiling.md
2. Airbnb / Chronon project, "What is Chronon?" (Introduction) and README, Chronon repository, 2024. https://github.com/airbnb/chronon
3. Airbnb / Chronon project, "Benefits of Chronon over other approaches" (docs/source/getting_started/Benefits.md), 2024. https://github.com/airbnb/chronon/blob/main/docs/source/getting_started/Benefits.md
4. Twitter/X, "Aggregation framework" README (timelines/data_processing/ml_util/aggregation_framework), the-algorithm repository, 2023. https://github.com/twitter/the-algorithm/tree/main/timelines/data_processing/ml_util/aggregation_framework
5. Twitter/X, "Home Mixer" README, the-algorithm repository, 2023. https://github.com/twitter/the-algorithm/blob/main/home-mixer/README.md
6. Twitter/X, "Follow Recommendations Service" README, the-algorithm repository, 2023. https://github.com/twitter/the-algorithm/blob/main/follow-recommendations-service/README.md
7. Feast project, "Point-in-time joins" (docs/getting-started/concepts/point-in-time-joins.md), 2026. https://github.com/feast-dev/feast/blob/master/docs/getting-started/concepts/point-in-time-joins.md
8. Feathr project (LinkedIn / LF AI & Data), Feathr documentation README, 2022-2023. https://github.com/feathr-ai/feathr/blob/main/docs/README.md
9. Feathr project, "Concepts for Beginners" (docs/concepts/feathr-concepts-for-beginners.md), 2022. https://github.com/feathr-ai/feathr/blob/main/docs/concepts/feathr-concepts-for-beginners.md
10. Feathr project, "Point-in-time Correctness and Point-in-time Join" (docs/concepts/point-in-time-join.md), 2022. https://github.com/feathr-ai/feathr/blob/main/docs/concepts/point-in-time-join.md
11. Feathr project, "Comparison between Feature Stores" (docs/concepts/feature-store-comparison.md), 2022. https://github.com/feathr-ai/feathr/blob/main/docs/concepts/feature-store-comparison.md
12. Feathr project (LinkedIn and Microsoft), "Feathr Community Talk - An Enterprise-Grade High Performance Feature Store" and "Feathr Feature Store Talk" slide decks (docs/talks), 2022. https://github.com/feathr-ai/feathr/tree/main/docs/talks
13. LinkedIn, "Venice: Derived Data Platform for Planet-Scale Workloads" (docs/README.md), Venice repository, 2022-2026. https://github.com/linkedin/venice
14. Hopsworks (Logical Clocks), "Offline API" / point-in-time correct training data and spine groups (docs/concepts/fs/feature_view/offline_api.md), 2026. https://github.com/logicalclocks/logicalclocks.github.io/blob/main/docs/concepts/fs/feature_view/offline_api.md
15. Tecton, "Tecton Examples Repository" (rift/ads and rift/fraud feature definitions), 2024-2025. https://github.com/tecton-ai/tecton-sample-repo
16. Hopsworks, "Write APIs" (docs/concepts/fs/feature_group/write_apis.md) and "Features and Feature Groups" (fg_overview.md), 2026. https://github.com/logicalclocks/logicalclocks.github.io/blob/main/docs/concepts/fs/feature_group/write_apis.md
17. Twitter/X, Aggregation framework docs: aggregation.rst, batch.rst, real-time.rst, joining.rst, the-algorithm repository, 2023. https://github.com/twitter/the-algorithm/tree/main/timelines/data_processing/ml_util/aggregation_framework/docs
18. Twitter/X, "Prediction Features" README (src/scala/com/twitter/timelines/prediction/features), the-algorithm repository, 2023. https://github.com/twitter/the-algorithm/blob/main/src/scala/com/twitter/timelines/prediction/features/README.md
19. Twitter/X, "Unified User Actions (UUA)" README, the-algorithm repository, 2023. https://github.com/twitter/the-algorithm/blob/main/unified_user_actions/README.md
20. Twitter/X, "User Signal Service" README, the-algorithm repository, 2023. https://github.com/twitter/the-algorithm/blob/main/user-signal-service/README.md
21. Twitter/X, "Graph Feature Service" README, the-algorithm repository, 2023. https://github.com/twitter/the-algorithm/blob/main/graph-feature-service/README.md
22. J. Hermann, M. Del Balso, "Meet Michelangelo: Uber's Machine Learning Platform", Uber Engineering, 2017 (not re-fetched; content as widely cited). https://eng.uber.com/michelangelo-machine-learning-platform/
23. Uber, "Michelangelo Palette: A Feature Engineering Platform at Uber", InfoQ presentation, 2019 (not re-fetched). https://www.infoq.com/presentations/michelangelo-palette-uber/
24. Airbnb, "Zipline: Airbnb's Machine Learning Data Management Platform", talk (YouTube), 2018 (not re-fetched). https://www.youtube.com/watch?v=Tg5VEMEsC-0
25. Airbnb, "Zipline - A Declarative Feature Engineering Framework", talk (YouTube), 2020 (not re-fetched). https://www.youtube.com/watch?v=LjcKCm0G_OY
26. Airbnb / Chronon project, "Source" (docs/source/authoring_features/Source.md), 2024. https://github.com/airbnb/chronon/blob/main/docs/source/authoring_features/Source.md
27. Airbnb / Chronon project, "Join" (docs/source/authoring_features/Join.md), 2024. https://github.com/airbnb/chronon/blob/main/docs/source/authoring_features/Join.md
28. Airbnb / Chronon project, "GroupBy" (docs/source/authoring_features/GroupBy.md), 2024. https://github.com/airbnb/chronon/blob/main/docs/source/authoring_features/GroupBy.md
29. Airbnb / Stripe, Chronon project, "The Tiled Architecture" (docs/source/Tiled_Architecture.md), 2024-2025. https://github.com/airbnb/chronon/blob/main/docs/source/Tiled_Architecture.md
30. Airbnb / Chronon project, "Chronon on Flink" (docs/source/setup/Flink.md), 2024-2025. https://github.com/airbnb/chronon/blob/main/docs/source/setup/Flink.md
31. Airbnb / Chronon project, "Online Offline Consistency" (docs/source/test_deploy_serve/Online_Offline_Consistency.md), 2024. https://github.com/airbnb/chronon/blob/main/docs/source/test_deploy_serve/Online_Offline_Consistency.md
32. Airbnb / Chronon project, "Bootstrap" (docs/source/authoring_features/Bootstrap.md), 2024. https://github.com/airbnb/chronon/blob/main/docs/source/authoring_features/Bootstrap.md
33. Airbnb / Chronon project, "Chaining Features" (docs/source/authoring_features/ChainingFeatures.md), 2024. https://github.com/airbnb/chronon/blob/main/docs/source/authoring_features/ChainingFeatures.md
34. Gojek / Google Cloud, "Introducing Feast: An Open Source Feature Store for Machine Learning", Google Cloud Blog, 2019 (not re-fetched). https://cloud.google.com/blog/products/ai-machine-learning/introducing-feast-an-open-source-feature-store-for-machine-learning
35. Feast project, README, 2026. https://github.com/feast-dev/feast
36. Feast project, "Push vs Pull Model" (docs/getting-started/architecture/push-vs-pull-model.md), 2026. https://github.com/feast-dev/feast/blob/master/docs/getting-started/architecture/push-vs-pull-model.md
37. Feast project, "Writing Data to Feast" (docs/getting-started/architecture/write-patterns.md), 2026. https://github.com/feast-dev/feast/blob/master/docs/getting-started/architecture/write-patterns.md
38. Feast project, "Feature Transformation" (docs/getting-started/architecture/feature-transformation.md), 2026. https://github.com/feast-dev/feast/blob/master/docs/getting-started/architecture/feature-transformation.md
39. Feast project, "Architecture Overview" (docs/getting-started/architecture/overview.md), 2026. https://github.com/feast-dev/feast/blob/master/docs/getting-started/architecture/overview.md
40. Feast project, online store reference index (docs/reference/online-stores), 2026. https://github.com/feast-dev/feast/tree/master/docs/reference/online-stores
41. Feast project, "Chronon online store (contrib)" (docs/reference/online-stores/chronon.md), 2026. https://github.com/feast-dev/feast/blob/master/docs/reference/online-stores/chronon.md
42. Feast project, "Stream Feature View" and "Stream Processor" docs, 2026. https://github.com/feast-dev/feast/blob/master/docs/getting-started/concepts/stream-feature-view.md
43. Feast project, "Tiling with Intermediate Representations in Feast" (docs/getting-started/concepts/tiling.md), 2026. https://github.com/feast-dev/feast/blob/master/docs/getting-started/concepts/tiling.md
44. Feast project, "Feature monitoring" how-to (feature service logging and log metrics), 2026. https://github.com/feast-dev/feast/blob/master/docs/how-to-guides/feature-monitoring.md
45. Feast project, "Label View" (docs/getting-started/concepts/label-view.md), 2026. https://github.com/feast-dev/feast/blob/master/docs/getting-started/concepts/label-view.md
46. Feast project, "Feature Serving and Model Inference" (docs/getting-started/architecture/model-inference.md), 2026. https://github.com/feast-dev/feast/blob/master/docs/getting-started/architecture/model-inference.md
47. Hopsworks, "FTI Pipeline Architecture" (docs/concepts/fti.md), 2026. https://github.com/logicalclocks/logicalclocks.github.io/blob/main/docs/concepts/fti.md
48. Hopsworks, "Data Transformations" taxonomy (docs/concepts/mlops/data_transformations.md), 2025-2026. https://github.com/logicalclocks/logicalclocks.github.io/blob/main/docs/concepts/mlops/data_transformations.md
49. Featureform, "What is Featureform?" README, 2023-2025. https://github.com/featureform/featureform
50. Netflix Technology Blog, "Distributed Time Travel for Feature Generation", 2016 (not re-fetched). https://netflixtechblog.com/distributed-time-travel-for-feature-generation-389cccdd3907
51. DoorDash Engineering, "Introducing Fabricator: A Declarative Feature Engineering Framework", 2022 (not re-fetched). https://doordash.engineering/2022/01/11/introducing-fabricator-a-declarative-feature-engineering-framework/
52. DoorDash Engineering, "Building Riviera: A Declarative Real-Time Feature Engineering Framework", 2021 (not re-fetched). https://doordash.engineering/2021/03/04/building-a-declarative-real-time-feature-engineering-framework/
53. DoorDash Engineering, "Building a Scalable ML Feature Store with Redis, Binary Serialization, and Compression", 2020 (not re-fetched). https://doordash.engineering/2020/11/19/building-a-gigascale-ml-feature-store-with-redis/
54. LinkedIn Engineering, "Near real-time features for near real-time personalization", 2022 (not re-fetched). https://engineering.linkedin.com/blog/2022/near-real-time-features-for-near-real-time-personalization
55. Twitter, "The Architecture That Powers Twitter's Feature Store", talk (YouTube), 2019 (not re-fetched). https://www.youtube.com/watch?v=UNailXoiIrY
56. Conde Nast Technology, "Accelerating Machine Learning with the Feature Store Service", 2019 (not re-fetched). https://technology.condenast.com/story/accelerating-machine-learning-with-the-feature-store-service
57. N. Lathia (Monzo), "Building a Feature Store", 2020 (not re-fetched). https://nlathia.github.io/2020/12/Building-a-feature-store.html
58. QuintoAndar, "Butterfree: A Spark-based Framework for Feature Store Building", 2020 (not re-fetched). https://medium.com/quintoandar-tech-blog/butterfree-a-spark-based-framework-for-feature-store-building-48c3640522c7
59. Constructor.io, "Feature Store Design at Constructor", 2023 (not re-fetched). https://medium.com/constructor-engineering/feature-store-design-at-constructor-330b65f64b18
60. M. Zinkevich (Google), "Rules of Machine Learning: Best Practices for ML Engineering", Google Developers (not re-fetched). https://developers.google.com/machine-learning/guides/rules-of-ml
61. E. Yan et al., "applied-ml: Papers and tech blogs by companies sharing their work on data science and machine learning in production" (Feature Stores section; used to locate and verify URLs for refs 22-25, 34, 50-59). https://github.com/eugeneyan/applied-ml
