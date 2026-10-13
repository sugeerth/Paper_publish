# 21. Twitter/X Case Study: From WTF and GraphJet to the 2023 the-algorithm and the 2026 Phoenix Transformer

Twitter (renamed X in 2023) is the most thoroughly documented large-scale recommender in public. Over about thirteen years the company published or open-sourced each major generation of its stack:

- the Who-to-Follow (WTF) system and its in-memory graph engine Cassovary (WWW 2013) [14];
- GraphJet, a real-time bipartite-graph engine for content recommendation (VLDB 2016) [12][13];
- SimClusters community embeddings (KDD 2020) [8][9] and TwHIN heterogeneous-graph embeddings (KDD 2022) [15][16];
- the March 2023 release of `twitter/the-algorithm` and `twitter/the-algorithm-ml`, which exposed the full For You pipeline: Home Mixer, candidate sources, the Earlybird light ranker, the MaskNet heavy ranker, heuristics and visibility filtering [1][2][3][4][6];
- the January 2026 release of `xai-org/x-algorithm`, a Rust and JAX rebuild. It drops almost all hand-engineered features and ranks with a transformer derived from Grok-1, called Phoenix. Later updates in May, August and September 2026 published production model code, production scoring weights, and the trust-and-safety labeling stack [28][29][30][31].

The 2023 and 2026 systems are close to a controlled experiment in recommender design. The product, the candidate volumes and the multi-action value model barely change between them. The 2023 system is a feature-engineered ensemble of about 6,000 features feeding a parallel MaskNet [4][6]. The 2026 system is an ID- and sequence-based transformer that reads the viewer's raw engagement history [28][29]. This chapter covers both, plus the graph systems that came before them, and ends with lessons for system designers.

A note on sourcing: everything below comes from Twitter/X's own repositories and READMEs, read directly for this chapter. Figures that come only from the 2023 engineering blog post [2] are marked as approximate. That post and the academic papers [9][13][14][15][17] could not be re-fetched from this research environment, so they are cited by their canonical URLs.

---

## 21.1 Pre-history: Who-to-Follow, Cassovary and RealGraph

### Problem

Twitter's first large recommendation problem was account recommendation rather than tweet recommendation. A new user follows few accounts, so the home timeline is empty, and the follow graph is what makes the product work. WTF (Gupta, Goel, Lin, Sharma, Wang, Zadeh; WWW 2013) [14] describes the first production system.

### Architecture

The main design bet in WTF was to keep the entire follow graph in memory on a single server, using an in-house graph library called Cassovary, instead of partitioning it across a cluster [14]. The authors argued that the graph fit in the RAM of a large commodity machine. Single-machine random-walk algorithms were far simpler and faster than distributed graph processing, and the system could be replicated for throughput.

The core algorithm runs in two stages [14]:

1. **Circle of trust.** For user u, run an egocentric random walk (personalized PageRank) from u and keep the top-k nodes. These are the accounts most "socially near" u.
2. **SALSA on a bipartite graph.** Put the circle of trust on the left ("hubs") and the accounts they follow on the right ("authorities"). Run SALSA, a HITS-like random walk, over this bipartite graph. High-scoring authorities become follow recommendations, and high-scoring hubs are users similar to u.

```
            follow graph (in memory, Cassovary)
                         |
   user u --> personalized PageRank --> circle of trust (hubs, top-k)
                         |
            hubs  ==follow edges==>  authorities
                         |
                    SALSA iterations
                         |
      authorities ranked --> "Who to Follow" candidates --> filters --> UI
```

### Lessons from WTF

The paper is candid about the system's limits. A single-server design requires periodic graph snapshots, so it is not real-time. Later work moved toward real-time engines; GraphJet is the direct descendant [13]. WTF also established three ideas that every later Twitter system keeps: interaction graphs as the main signal, random walks for candidate generation, and SALSA-style bipartite scoring.

### RealGraph and Tweepcred

Two user-level graph models from that period still existed in the 2023 release.

- **RealGraph** [10] predicts how likely user A is to interact with user B. A daily Dataflow (Scio) job aggregates pairwise interactions. These include public ones (favorites, retweets, follows) and private ones (profile views, tweet clicks, and, if the user opted in, address-book membership). A rollup job keeps a decayed sum. A gradient-boosted tree classifier, trained on "edges active in a period, labelled by whether they interact the next day", outputs a pairwise interaction score. In the 2023 For You pipeline, RealGraph decides which followed accounts matter most. It weights in-network candidates and serves as a source signal for CR-Mixer and UTEG [2][20].
- **Tweepcred** [25] is a weighted or unweighted PageRank over the user interaction graph (mentions, retweets and so on), implemented as an iterative Hadoop/Scalding job. It produces a per-user reputation score that the Earlybird light ranker uses as a feature [18][25]. Its 2026 successor, `user-cred-v2/`, still "runs PageRank over the follow graph and engagement edges" [28].

---

## 21.2 GraphJet: Real-Time Collaborative Filtering on a Single Server

### Problem

Tweets have a very short useful life, often hours. Collaborative filtering over a nightly batch graph is too stale for them. Twitter needed "users who engaged with X also engaged with Y" recommendations computed over the last day or so of activity, with fresh edges visible within seconds.

### Architecture

GraphJet (Sharma, Jiang, Bommannavar, Larson, Lin; VLDB 2016) [13] is a Java library that "maintains a full graph index over a sliding time window in memory on a single server" [12]. Its key features:

- **Bipartite graph.** Users are on the left and tweets (or URLs, hashtags) on the right. Edges are interactions such as posts, retweets and likes [12].
- **Temporal partitioning.** The index is a sequence of time-sliced segments. New edges go into the newest segment, and whole segments are dropped as they leave the window. This avoids per-edge deletion and garbage-collection pressure [13].
- **Compact edge encoding and dynamic memory allocation.** These let the engine ingest edges and serve queries concurrently [12].
- **Random-walk algorithms.** SALSA-style personalized walks are the main recommendation primitive, as in WTF [13].

Throughput as stated in the README [12]:

| Metric | Value per GraphJet server |
|---|---|
| Edge ingestion | up to 1,000,000 edges/s |
| Recommendations (steady state) | up to 500/s |
| Edge reads | "several million" per second |

### Production descendants

In the 2023 open-source release, GraphJet powers several services under `src/scala/com/twitter/recos`. The most important for Home is **UTEG (User-Tweet-Entity-Graph)** [11]:

- It is "a Finagle thrift service built on the GraphJet framework" that produces the "XXX Liked" out-of-network tweets on the home timeline [11].
- It takes as input the user's **weighted follow graph** (a list of weighted user IDs, with weights from RealGraph). It traverses the graph, aggregates, and returns tweets ranked by how many weighted followees engaged with them [11].
- It is stateful and fed by Kafka. It keeps **in-memory engagements for the past 24-48 hours** and drops older events [11].
- **Recos-Injector** [26] is the streaming preprocessor for all GraphJet services. It consumes favorites, retweets, follows and client events, filters and combines them, and publishes them to a dedicated Kafka topic per GraphJet service.

```
 Event Bus / Kafka (fav, RT, follow, client events)
            |
      recos-injector  (filter, combine, per-service topic)
            |
   +--------+---------------------+
   |                              |
  UTEG (GraphJet, 24-48h window)  other GraphJet recos services
   |
  input: viewer's RealGraph-weighted followees
  traverse user->tweet edges, aggregate engagement weight
   |
  top tweets with social proof "<followee> liked"
```

### Lessons from GraphJet

A sliding window kept entirely in RAM is cheap and very fresh for collaborative filtering on short-lived content. The price is that it only sees about the last 1-2 days, which is acceptable because tweets decay quickly. Single-server replication is again preferred to sharding. The same "in-memory recent-content store per server" idea comes back in 2026 as **Thunder** (Section 21.6).

---

## 21.3 Representation Layer: SimClusters and TwHIN

Two embedding systems give the 2023 stack its out-of-network reach. They are sparse community vectors (SimClusters) and dense knowledge-graph vectors (TwHIN). Both are served through `representation-manager` and scored by `representation-scorer` [1].

### 21.3.1 SimClusters (KDD 2020)

**Problem.** Twitter needed one general-purpose representation of users, tweets, topics and other entities that was interpretable and cheap to compute and update [8][9].

**Algorithm** (from the official README [8]):

1. Treat the follow graph as a bipartite **producer-consumer** matrix A (consumers follow producers).
2. Compute **producer-producer cosine similarity** over follower sets, and drop edges below a threshold.
3. Run **Metropolis-Hastings sampling-based community detection** on the producer similarity graph with k communities. The result is a maximally sparse **KnownFor** matrix V (n x k), with each producer in at most one community. In production KnownFor covers **the top 20M producers and k is about 145,000 communities**.
4. **InterestedIn** (consumer embedding) U = A x V, followed by noise removal. This captures long-term consumer interest and is "one of our major source[s] for consumer-based tweet recommendations".
5. **Producer embeddings** are the cosine similarity between a producer's follower vector and each community's InterestedIn vector. A producer can belong to many communities.
6. **Tweet embeddings** start empty when the tweet is created. **Each time a user favorites the tweet, that user's InterestedIn vector is added** to the tweet vector. A Heron/Summingbird streaming job does this in real time, and a separate job persists the embeddings from Memcache to Manhattan [8].
7. **Topic embeddings** are built from fav-based Topic-Follow-Graph aggregates with time decay [8].

```
 follow graph A (consumers x producers)
        |  cosine sim over followers, threshold
        v
 producer-producer graph --MH community detection (k~145K)--> KnownFor V (20M x 145K)
        |
   U = A . V  ---> InterestedIn (consumer, long-term)
        |                        \
        |                         +--> producer embeddings (multi-community)
   fav stream (Heron) --> tweet vec += InterestedIn(faver)   [real-time]
        |
 SimClusters ANN: cluster -> top tweets index; user vec -> candidate tweets
```

**Serving.** SimClusters ANN (SANN) indexes cluster -> top tweets. BigQuery/GCP jobs build engagement-specific indexes, for example a push-open-based index for Notifications and a video-view-based index for Home video [8]. Because the vectors are sparse, retrieval is an inverted lookup: for each of the user's top clusters, fetch that cluster's top tweets, then score them by dot product.

**Durability.** SimClusters survives into the 2026 X stack. The August 2026 update added `simclusters/` as "an additional source of posts from accounts the viewer does not follow that is called in retrieval alongside Thunder and Phoenix retrieval". It has a 48-hour maximum candidate age and an `OONNsfwSimclustersFilter` [28][30].

### 21.3.2 TwHIN (KDD 2022)

**Problem.** SimClusters is built mainly from follow and favorite graphs. Twitter has many more relation types (follows, favorites, retweets, replies, ad clicks) and wanted one dense embedding space covering users, tweets, advertisers and other entities [15][16].

**Approach.** TwHIN (El-Kishky et al.) [15] trains knowledge-graph embeddings over a heterogeneous information network. Edges are (lhs, relation, rhs) triples, and each relation is modelled as a **translation operator** in the embedding space, in the TransE family. The open-sourced TorchRec version [16] expects parquet edges with `lhs, rel, rhs` columns. Its sample config defines `fav`, `reply`, `retweet` and `magic_recs` relations, all user -> tweet with `operator: translation`, and trains with in-batch negatives. Twitter also released heavily subsampled, anonymized follow and favorite graphs on Hugging Face [16].

**Uses.** TwHIN embeddings are used "for candidate retrieval and as model features in a variety of recommender system models" [16]. In For You they feed ANN out-of-network candidates through CR-Mixer and are features in the heavy ranker [2]. Later Home Mixer parameters also show TwHIN used for diversity: `twhin_diversity_rescoring_weight` and `twhin_diversity_rescoring_ratio` [7].

### SimClusters vs TwHIN

| Property | SimClusters [8] | TwHIN [15][16] |
|---|---|---|
| Vector type | Sparse, ~145K dimensions, interpretable communities | Dense, low-dimensional |
| Input graph | Follow graph (+ fav for tweets/topics) | Multi-relation heterogeneous graph |
| Training | Community detection + matrix products | Translation-based KG embedding (SGD) |
| Tweet update | Real time: add InterestedIn of each faver | Batch-trained |
| Retrieval | Inverted cluster -> top-tweets index | ANN over dense vectors |
| Still used in 2026? | Yes (`simclusters/` source) [28] | Not in the 2026 public repo |

---

## 21.4 The 2023 For You Pipeline (the-algorithm)

### 21.4.1 Overview and scale

On 31 March 2023 Twitter published the code for the For You timeline and Recommended Notifications [1][2]. The engineering blog [2] describes the funnel as follows. These numbers are approximate and as reported in the blog:

- About **500 million tweets** posted per day.
- For each request, about **1,500 candidate tweets** are fetched, around **50% in-network and 50% out-of-network** on average (the repo README also says "~50% of posts come from" the Earlybird search index [1]).
- Ranking uses a neural network of about **48M parameters** trained on tweet interactions, with **thousands of features**, predicting **10 labels**.
- The pipeline runs about **5 billion times a day** and completes in **under 1.5 s on average**. One execution takes about **220 s of CPU time**, roughly 57x its wall-clock latency.

The home-mixer README [6] confirms the stages and the feature count: "Fetch the ~6000 features needed for ranking."

### 21.4.2 Architecture

```
 client request (For You)
        |
  HOME MIXER (Scala, Product Mixer framework)
  ForYouProductPipelineConfig
   +-- ForYouScoredTweetsMixerPipeline  (mix tweets + ads + WTF users)
   |     +-- ScoredTweetsRecommendationPipeline
   |     |     candidate pipelines (fetch caps from ScoredTweetsParam):
   |     |       InNetwork (Earlybird)   <= 600
   |     |       TweetMixer / CR-Mixer   <= 400   (SimClusters, TwHIN ANN, ...)
   |     |       UTEG (GraphJet)         <= 300
   |     |       FRS (FutureGraph)       <= 100
   |     |     feature hydration (~6000 features)
   |     |     ScoringPipeline: heavy ranker (MaskNet, Navi)
   |     |     heuristics: author diversity, in/out balance, fatigue, dedup
   |     |     visibility filtering
   |     +-- ConversationService pipeline  (reverse-chron backup on failure)
   |     +-- Ads pipeline
   |     +-- WhoToFollow pipeline (FRS)
   +-- marshal -> timeline (modules, social context, cursors) -> client
```

The candidate caps come from the open-sourced `ScoredTweetsParam.scala` [7]. They are `in_network_max_tweets_to_fetch = 600`, `tweet_mixer_max_tweets_to_fetch = 400`, `uteg_max_tweets_to_fetch = 300`, `frs_max_tweets_to_fetch = 100` and `server_max_results = 50`. Together that is about 1,400 candidates, close to the blog's "~1,500". The result is a page of about 50 posts. The same file has `out_of_network_scale_factor = 0.75`, so out-of-network scores are multiplied by 0.75 [7]. These are code defaults at the last snapshot of the repo and may have been overridden by feature switches in production.

### 21.4.3 Product Mixer: the framework

Home Mixer is built on **Product Mixer**, a Scala framework of composable pipelines [6][24]:

- **Product Pipelines** choose which Mixer or Recommendation Pipeline to run for a request.
- **Mixer Pipelines** combine heterogeneous candidates (tweets, ads, users).
- **Recommendation Pipelines** score and rank homogeneous candidates through **Scoring Pipelines**.
- **Candidate Pipelines** fetch from Candidate Sources, filter, decorate and hydrate features.

This split between configuration and business logic carries straight into the 2026 Rust `candidate-pipeline` crate (Section 21.6). It is one of the most durable patterns in the codebase.

### 21.4.4 In-network source: Earlybird and the light ranker

**Earlybird** [19] is Twitter's Lucene-based real-time search system. It has three clusters:

- a **realtime** cluster with public tweets from roughly the last 7 days;
- a **protected** cluster with protected tweets over the same window;
- an **archive** cluster with all tweets up to about two days ago.

Each cluster is sharded into partitions with a single-writer, multiple-reader inverted index. Roots fan out requests and merge the results. For Home, the query is essentially "recent tweets authored by this list of followees", so retrieval is a user-ID search [19].

Inside Earlybird, the **light ranker** [18] scores candidates before they leave the index:

- It is a **logistic regression** predicting engagement. There are two variants: `recap_earlybird` for in-network and `rectweet_earlybird` for UTEG out-of-network tweets.
- Static features (computed at ingestion) include is-retweet, has-link, has-trend-words, is-reply, a **text quality score** (offensiveness, entropy, "shout" score, length, readability) and **tweepcred**.
- Realtime features (updated through a Heron signal ingester) include like, reply and retweet counts, plus **pToxicity and pBlock** health-model scores.
- The README's own warning reads: "the light ranker is an old part of the stack which we are currently in the process of replacing. The current model was last trained several years ago, and uses some very strange features" [18].

RealGraph scores are used to choose and weight the followee list that Earlybird searches [2].

### 21.4.5 Out-of-network sources

- **CR-Mixer / Tweet Mixer** [20] is a coordination layer for out-of-network candidate generation. It runs four steps: source-signal extraction (from UserProfileService, RealGraph and similar), candidate generation from compute services (SimClusters ANN, TwHIN ANN, and so on), filtering, and light ranking. Its stated aim is "a 1-stop-shop" that speeds up iteration on candidate sources [20].
- **UTEG** [11] is GraphJet collaborative filtering (Section 21.2). The blog attributes about 15% of For You tweets to social-graph traversal of this kind (approximate) [2].
- **FRS** [21] is the Follow Recommendations Service. Besides Who-to-Follow modules, it supplies "FutureGraph" tweets from accounts the user may follow in future. FRS ranks account candidates by a weighted sum of p(follow | recommendation) and p(positive engagement | follow). Its flow is: candidate generation, light filters, ranking, heavy filters, transforms (social proof such as "followed by X"), then truncation [21].

### 21.4.6 The heavy ranker (MaskNet)

The heavy ranker is open-sourced in `the-algorithm-ml` (PyTorch/TorchRec) [3][4].

**Inputs.** Features are documented in FEATURES.md [5]. Most of them are **aggregate features** from the Timelines Aggregation Framework. These are rolling counts over scopes such as user, author and user-author, crossed with engagement types. They are kept over long windows (50 days) and real-time windows (under 3 days, typically 30 minutes), for example `user_aggregate_v2.pair.recap.engagement.is_favorited.....50.days.count`. Other inputs include RealGraph, SimClusters and TwHIN features, author features and tweet content features [5].

**Architecture.** The backbone is a **parallel MaskNet** (Wang et al., 2021) [17]. From the public config `local_prod.yaml` and `mask_net.py` [36][37]:

- Four MaskBlocks run in parallel (`use_parallel: true`), each with `aggregation_size 1024` and `output_size 1024`.
- Each MaskBlock computes `LayerNorm(W_h (x * MLP_mask(x)))`. Here `MLP_mask` is Linear -> ReLU -> Linear back to the input dimension. This is an instance-guided multiplicative mask over the feature embedding, a learned feature-wise gating.
- The four block outputs (4 x 1024) are concatenated and passed through a shared MLP of width 2048.
- The multi-task type is `share_all`. Each task has its own MLP head `256 -> 128 -> 1` with batch norm, and dropout 0.1 on the negative-feedback head.
- Input featurization uses a `double_norm_log` transform: log, then batch norm and layer norm with magnitude clipping at 5.0.

```
  ~6000 features (aggregates, graph, embeddings, content)
        |  double-norm-log featurization
        v
  +-----------+-----------+-----------+-----------+
  | MaskBlock | MaskBlock | MaskBlock | MaskBlock |   parallel
  | x*mask(x) | x*mask(x) | x*mask(x) | x*mask(x) |   (1024 each)
  +-----------+-----------+-----------+-----------+
        | concat (4096) -> MLP 2048 (shared)
        v
  10 task heads: MLP 256-128-1, sigmoid  -> P(engagement_i)
        |
  score = sum_i w_i * P_i   (weights from config)
```

**Outputs and value model.** There are 10 heads, each a probability. The weights published on 5 April 2023 [4] are:

| Head (predicted event) | Weight |
|---|---|
| fav (like) | 0.5 |
| retweet | 1.0 |
| reply | 13.5 |
| good_profile_click (open profile and like/reply) | 12.0 |
| video_playback50 (watch >= 50%) | 0.005 |
| reply_engaged_by_author | 75.0 |
| good_click (open convo and reply/like) | 11.0 |
| good_click_v2 (open convo, stay >= 2 min) | 10.0 |
| negative_feedback_v2 (show less, block, mute) | -74.0 |
| report | -369.0 |

The README explains that the weights "were originally set so that, on average, each weighted engagement probability contributes a near-equal amount to the score", and were later tuned against platform metrics [4]. The large weight on a reply that the author engages with (75) shows a deliberate preference for conversation over passive likes.

**Serving.** Models are served by **Navi**, Twitter's Rust ML-serving server. It is gRPC-compatible with TensorFlow Serving and has TensorFlow and ONNX runtimes; the ONNX runtime is used for "home recommendation with a proprietary BatchPredictRequest format" [23].

### 21.4.7 Heuristics, filters and mixing

After scoring, the home-mixer README lists the following steps [6]:

- **Author diversity**: demote consecutive tweets by the same author.
- **Content balance**: keep a target in-network vs out-of-network mix (see also `out_of_network_scale_factor`) [7].
- **Feedback fatigue**: demote content similar to things the user gave negative feedback on.
- **Deduplication**: remove previously seen tweets.
- **Social proof**: out-of-network tweets need a second-degree connection (the blog describes requiring that someone you follow engaged with the tweet or follows the author) [2].
- **Visibility filtering**: blocked or muted authors, NSFW settings and similar.
- **Mixing**: ads, Who-to-Follow modules, prompts, conversation modules for replies, edited tweets, and pagination.

Later snapshots of `ScoredTweetsParam.scala` also show newer features [7]:

- "Control AI" show-less and show-more scale factors (0.05 and 20.0) with an embedding-similarity threshold of 0.67;
- content-exploration pipelines for cold posts;
- category and TwHIN diversity rescoring;
- CLIP image clusters and multimodal embeddings as features.

### 21.4.8 Visibility filtering (VF)

`visibilitylib` [22] is "a centralized rule engine that instructs clients how to alter the display of certain Twitter content on read time". Its concepts are:

- **SafetyLevel**: the product context, such as Timeline or Profile.
- **Features**: safety labels, user flags, user-user relationships and user-content relationships.
- **Conditions** combined into **Rules**, which form a **Policy**. Rules are in priority order, with one policy per SafetyLevel.
- **Actions**: Drop (hard filter), Labels and Interstitials (soft filter), and "coarse-grained downranking".

The 2023 release explicitly omitted part of this code as "being reviewed and rebuilt" [22]. The 2026 release filled that gap (Section 21.7).

### 21.4.9 Recommended notifications

`pushservice` [27] follows the same funnel: build the target and check eligibility, fetch candidates (including SimClusters ANN indexes built on push opens [8]), hydrate, apply light filters, rank, then take the top candidate that survives heavy filtering. Ranking uses a **light ranker**, which pre-selects from the initial huge pool, and a **multi-task heavy ranker** that predicts open and engagement probabilities. Because the result is a single notification rather than a page, "take step" walks down the ranked list one candidate at a time, running expensive filters until one candidate passes [27].

---

## 21.5 Why X rebuilt it: limits of the 2023 design

The 2023 READMEs point to the weaknesses that motivated the 2026 rewrite:

1. **Feature sprawl.** There were about 6,000 features [6], mostly cross-product aggregates [5], spread over many offline jobs, streaming jobs and feature-serving services.
2. **Stale components.** The light ranker was "last trained several years ago" with "very strange features" [18]. `twml` was a "legacy" TensorFlow v1 framework, and `timelineranker` was legacy [1].
3. **Many heterogeneous candidate sources.** Earlybird, CR-Mixer, UTEG, FRS, SimClusters ANN and TwHIN ANN each had its own models and infrastructure.
4. **Heuristics layered on top.** Diversity, balance, fatigue and social-proof rules all altered the model's output.

---

## 21.6 The 2026 xAI Rebuild: x-algorithm and Phoenix

### 21.6.1 Release history

`xai-org/x-algorithm` was first published on **20 January 2026**, with further drops on **15 May**, **13-18 August**, and later notes dated **18 September 2026** (from the repository's git history and README) [28]. The first README called Phoenix "a Grok-based transformer model" and stated: "We have eliminated every single hand-engineered feature and most heuristics from the system. The Grok-based transformer does all the heavy lifting by understanding your engagement history" [28].

The first Phoenix release shipped "a sample transformer ported from the Grok-1 open source release" [34]. The current release ships "the production implementation itself: the real model code, the real training step, and the real Rust serving engine", with synthetic data generators in place of production data [29]. Everything is Apache 2.0 [28].

### 21.6.2 Request path

Home Mixer is now a Rust service on a generic `candidate-pipeline` crate. The crate's stage types are Source, Hydrator, Filter, Scorer, Selector and SideEffect, and independent stages run in parallel [28].

```
 FOR YOU REQUEST
   |
 HOME MIXER (Rust) -- Post Pipeline (PhoenixCandidatePipeline)
   1 query hydration: user action sequence (recent engagements = main model input),
     following list, blocks/mutes, muted keywords, seen/served posts, topics
   2 candidate sources (parallel)
       in-network : Thunder (in-memory recent posts of followees)   cap 1200
       out-of-net : Phoenix retrieval (two-tower ANN)               cap 1000
                    SimClusters (cluster similarity, <=48h)
                    Popular posts (shared list)                      cap 500
   3 candidate hydration: text/media, author + labels, quoted post, lang, counts
   4 pre-scoring filters (dedup, 48h age, self, OON replies/RTs, seen/served,
     muted kw, block/mute, subscription, election filter, holdouts ...)
   5 scoring: PhoenixScorer -> RankingScorer (weighted sum, author decay,
     OON discount, new-author boost) -> VMRanker (DPP rerank)
   6 selection: TopKScoreSelector
   7 post-selection: VFCandidateHydrator -> VFFilter -> AncillaryVFFilter
     -> DedupConversationFilter
   |
 Blending Pipeline (ForYouCandidatePipeline): ads blender, WTF, prompts
   |
 side effects (after response): record served posts, refresh cache, log events
```

Source caps are the defaults in `param.rs`: `PhoenixMaxResults = 1000`, `ThunderMaxResults = 1200`, `PopularPostsMaxResults = 500`, cold-start Phoenix 200, and an optional semantic-ID source of 800. The legacy `TweetMixer` source is still present but disabled (`EnableTweetMixerSource = false`) [30]. The file header says these values are "mirrored from config feature-switch defaults" by a cron job, so the repo tracks primary production values [28][30].

### 21.6.3 Thunder: the in-network store

Thunder replaces Earlybird for in-network retrieval. It is "an in-memory post store and realtime ingestion pipeline" that consumes post create and delete events from Kafka, serves posts by the viewer's followees, and trims posts older than a retention period. The README promises "sub-millisecond lookups for in-network content without hitting an external database" [28]. It is the GraphJet idea again: a time-windowed in-memory index replaces a general-purpose search engine. Thunder also receives the already-seen list so it can exclude those posts at the source [28][35].

### 21.6.4 Phoenix retrieval (two-tower)

From the Phoenix README [29]:

- **User tower.** A transformer encodes the user's engagement history plus a single user-features token (country, language and similar) into a normalized embedding. Production retrieval uses **no learned per-user ID embedding** (`use_user_embedding=False`). The user is represented by what they interacted with.
- **Candidate tower.** Normalized post embeddings. Since a "semantic-ID migration", posts are represented by **semantic IDs (SIDs)**: residual-quantized codes, **6 levels x 256 codes**, derived from each post's multimodal embedding, plus hashed author IDs. Same-topic posts share SID prefixes, which helps the model generalize to unseen posts.
- **Index inside the checkpoint.** At every checkpoint save, the trainer runs the candidate tower over the corpus and stores `post_embeddings` in the checkpoint. Serving loads the index from there, so "nothing embeds a corpus at boot time". The production index (`max_posts`) holds **10.24M posts (28.67M for the combined config)**.
- **Training.** Contrastive, with **favorites as positives**, using in-batch plus **64 sampled global negatives per example with log-Q correction**.
- **Index maintenance.** `phoenix-rankall` keeps the index updated as events arrive. `phoenix-rankall-strato` decides which index a post belongs in, "consulting visibility filtering first" [28].

### 21.6.5 Phoenix ranking (transformer with candidate isolation)

The ranking model is a transformer over the sequence `[user tokens | history | candidates]` [29]:

- **User tokens** ([B, 2]): hashed IDs plus profile features (country, language, location, gender, age bracket, installed apps).
- **History** ([B, S, D]): each engagement event carries post, author, action type, semantic IDs, and context (timezone, local hour, product surface, post age), plus dwell time.
- **Candidates** ([B, C, D]): post, author, SIDs, context.
- **Attention mask (candidate isolation).** User and history tokens attend to each other bidirectionally. Each candidate attends to user and history tokens and to itself, never to other candidates.
- **Output.** `[B, num_candidates, num_actions]` logits over a **64-action discrete taxonomy**, trained as multi-label targets, plus **8 continuous-action regression slots** (dwell time) [29].

```
  keys ->      | user | history (S) | candidates (C) |
  query user   |  v   |   v v v v   |   x x x x      |
  query hist   |  v   |   v v v v   |   x x x x      |
  query cand_1 |  v   |   v v v v   |   v x x x      |  <- self only
  query cand_2 |  v   |   v v v v   |   x v x x      |
  ...
  => score(c) depends only on (user, history, c): batch-invariant, cacheable
```

**Published production configs** (`xrex/configs`) [29]:

| Parameter | Ranking (prod) | Retrieval (prod) |
|---|---|---|
| Embedding dim | 2560 | 1024 |
| Transformer layers | 8 | 8 |
| Query / KV heads (GQA) | 20 / 4 | 16 / 4 |
| Attention key size | 128 | 128 |
| Embedding-table width | 1024 | 1024 |
| FFN widening factor | 2 | 2 |
| History length | 1022 | 1023 |
| Candidates per sequence | 64 | 64 |
| User / item / author hash vocab | 100M / 100M / 30M | - / 100M / 30M |
| IP-address vocab | 10M | - |
| Hashes per entity | 2 | 2 |
| Semantic IDs | 6 x 256 (input feature) | 6 x 256 (candidate identity) |
| Action taxonomy | 64 + 8 dwell slots | 64 (positive = favorite) |
| Per-device batch | 512 (GB300) / 256 (H100) | 480 |

Further implementation details:

- **Hash embeddings.** Every entity is looked up with multiple hash functions (2 per entity), so "there is no vocabulary to maintain and a new post is representable immediately" [28][29].
- **Sequence packing.** Production training packs several variable-length sessions per row and uses a variable-length attention kernel [29].
- **Optimizers.** Dense parameters in the flagship ranker use **Muon** ("consistent-RMS scaling with decoupled weight decay"). Embedding tables use **sparse rowwise AdaGrad** with accumulator half-life decay, lazy per-row decay and decoupled weight decay. Steps with non-finite gradients are skipped [32].
- **Parameterization.** μP is used: `emb_size = 512` is the μP base width, and single-GPU "nano" twins share production losses and code paths [29].
- **Serving.** A Rust gRPC engine is used for serving. It links `libibverbs` as "the production embedding transport" (RDMA), with kernels tuned for Hopper and Blackwell GPUs (H100 to GB300) [29].

Compared with the 2023 heavy ranker, this is roughly a move from 48M dense parameters on 6,000 handcrafted features to a 2560-wide, 8-layer transformer. Most of its capacity sits in hashed embedding tables with about 100M rows for each major entity type, and it reads a history of about 1,000 events [2][29].

### 21.6.6 Value model, adjustments and reranking

`RankingScorer` computes `score = sum_i w_i * P(action_i)`. The weights are in `home-mixer/params/param.rs` and the arithmetic is in `xai-value-model/scoring.rs` [28][30]. Production defaults as of the October 2026 sync [30]:

| Action | Weight | | Action | Weight |
|---|---|---|---|---|
| favorite | 0.5 | | share via copy link | 20.0 |
| reply | 5.0 | | follow author | 4.0 |
| repost | 1.0 | | dwell (binary) | 0.05 |
| quote | 5.0 | | continuous dwell time | 0.004 |
| click | 0.3 | | continuous click-dwell time | 0.4 |
| open link | 0.2 | | not interested | -47.52 |
| photo expand | 0.05 | | mute author | -58.8 |
| video open | 0.07 | | block author | -31.2 |
| share | 2.0 | | report | -234.0 |
| share via DM | 5.0 | | not dwelled | -0.02 |

The code comments push back on a common misreading of these numbers [30]. "The weights do not multiply raw engagement counts", so it is wrong to say "one report cancels 468 likes". The baseline probability of a report "is more than 1000x lower than a Like", which is why its weight is so large. Mass-reporting by bad actors mostly affects predictions for users similar to those actors, and only actions on posts served in Home Timeline count [30].

After the weighted sum, three adjustments are applied [28]:

- **Author diversity**: each post after an author's first is multiplied by a decaying factor, down to a floor.
- **Out-of-network discount**: posts from non-followed accounts are multiplied by a factor below 1, and so are replies and reposts from followed accounts. This continues the 2023 `out_of_network_scale_factor = 0.75` [7].
- **New-author boost / cold start**: posts by authors below an impression threshold are lifted toward a target position. The defaults [30] are an impression threshold of 200, a follower cap of 50,000, maximum post age 2 h, boost slots 15-16, and Thompson sampling enabled with Beta prior alpha0 = 0.75 and beta0 = 49.25 (a prior mean of about 1.5%) choosing the top 2.

**VMRanker / vm-ranker** is a separate service that reorders scored posts with a **determinantal point process (DPP)** over post embeddings, "giving up a little score for less similarity between neighbours" [28]. The service's defaults [33] are theta 0.5, a pool of the top 100 by score, 1024-dimensional fp16 embeddings, and greedy log-determinant selection. This replaces several of 2023's hand-written diversity heuristics with a single principled reranker.

### 21.6.7 Transparency features unique to the 2026 release

- **Experiment visibility.** X says it aims to make experiments that reach "a notable share of traffic — e.g. 10% or more" visible in the repo, and it syncs production parameter defaults into code [28].
- **Worked algorithm diff.** `docs/BIDIRECTIONAL_BOOST_CHANGE.md` [31] walks through a real launch. A *bidirectional-follow reply boost* adds weight to P(reply) for posts by mutual follows. It was A/B tested from 10 July 2026 with values 5, 10, 15 and 20, rolled out at 20 on 13 July, and cut to 15 on 24 July after feedback, for example that World Cup discussion from non-followed accounts was being crowded out. The current default is `BidirectionalFollowReplyWeightBoost = 15.0`, and a dwell-boost variant was tested but not shipped (`0.0`) [30][31].
- **Under the Hood** gives each account a report of visibility-limiting labels on its account and posts. Since September 2026 it also shows legal withholding by country [28].
- **Deterministic holdouts.** `InventoryHoldoutFilter` and `FavHoldoutFilter` remove a configured share of posts, chosen deterministically per (post, viewer), to measure the effect of inventory [28].

---

## 21.7 Visibility and Trust & Safety in 2026

The August 2026 drop published the labeling path that 2023 had kept out [28]:

```
 CONTENT UNDERSTANDING (async)            LABELING RULES             SERVE
 grox/ (LLM + classifiers: spam, adult,   scarecrow/ (event rules,   visibility-filtering/
   violent media, text/image embeddings)    botmaker rule engine)      per (post, viewer):
 media-model-proxy/, clip/, pnsfwmedia/   abuse-enforcement-svc/       ALLOW | NOTICE |
 agatha/ (blocks+reports vs favs)           (+ abuse-ledger appeals)   INTERSTITIAL | DROP
 bdsm/ (behaviour sequences)              safety-label-user-agg/       |
 user-cred-v2/ (PageRank cred)                     |                   VFFilter,
                \______________ labels -> storage _/                   AncillaryVFFilter
```

Design points [28]:

- **Ranking and visibility are separate systems.** Ranking sets the order. VF decides whether a post can be shown at all, using "different services, different inputs, different rules".
- **VF runs after selection.** `VFCandidateHydrator`, `VFFilter` and `AncillaryVFFilter` run after top-K is fixed. This avoids per-candidate VF calls on the whole pool, although pre-scoring filters already handle blocks, mutes and keywords cheaply. `AncillaryVFFilter` also drops a post when its parent, quoted post or reposted post was dropped.
- **Rule evaluation.** The first rule that answers DROP ends evaluation. A further set of out-of-network-only rules can only drop. For example, high-recall spam classifiers suppress a post as a *recommendation* while still showing it to followers.
- **Withheld rules.** Some rules and the Grox LLM prompts are deliberately not published "to reduce the risk of gaming". The Under the Hood tool is the compensating transparency [28].
- **Jurisdiction filters.** These appear in code, for example `Brazil2026ElectionFilter` [28].

---

## 21.8 Side-by-side: 2013 to 2026

| Dimension | WTF (2013) [14] | For You 2023 [1][2][4] | For You 2026 [28][29][30] |
|---|---|---|---|
| Language / stack | Scala, Cassovary | Scala (Product Mixer), TF v1 / PyTorch, Navi (Rust) | Rust (candidate-pipeline), JAX, Rust gRPC engine |
| In-network source | n/a | Earlybird search (Lucene) + LR light ranker | Thunder in-memory store |
| Out-of-network | PPR + SALSA | CR-Mixer (SimClusters/TwHIN ANN), UTEG (GraphJet), FRS | Phoenix two-tower ANN, SimClusters, popular posts |
| Candidates / request | n/a | ~1,500 (caps 600/400/300/100) | up to ~2,700 (caps 1200/1000/500 + SimClusters) |
| Ranker | n/a | Parallel MaskNet, ~48M params (approx), ~6,000 features | 8-layer transformer, d = 2560, ~1K-event history, hash + SID embeddings |
| Heads | n/a | 10 engagement probabilities | 64 discrete actions + 8 dwell regressions |
| Value model | n/a | weighted sum, 10 weights | weighted sum, ~20 weights + boosts |
| Diversity | n/a | author diversity, balance, fatigue heuristics | author decay + OON discount + DPP rerank |
| Safety | n/a | visibilitylib (partially published) | full labeling path + VF + Under the Hood |

---

## 21.9 Key takeaways for system designers

1. **Keep the funnel; change what is inside it.** In both generations about 1-3K candidates per request come from a few sources, are scored by one multi-head model, combined by an explicit weighted value model, diversified, safety-filtered and then mixed with ads. A clean pipeline framework (Product Mixer, then `candidate-pipeline`) lets you swap models without rewriting the product [6][24][28].
2. **Predict many actions, combine them separately.** Both generations predict per-action probabilities and keep the business trade-off in a weight vector held in configuration. This makes the policy auditable, explains why rare negative events need large weights, and lets product teams tune without retraining [4][30].
3. **Calibrate weights to base rates.** The 2023 weights were set so that each term contributes about equally on average [4]. The 2026 comments make the same point: a report is more than 1000x rarer than a like [30]. Weight magnitudes are not exchange rates between raw counts.
4. **In-memory, time-windowed stores win for fresh content.** Cassovary, GraphJet (24-48 h, 1M edges/s per server) and Thunder all trade completeness for latency and freshness on a single machine [11][12][28].
5. **Sequence models can replace feature factories.** X deleted about 6,000 handcrafted features and the aggregate-feature infrastructure behind them. It now feeds a transformer the raw engagement history, with hashed IDs and semantic IDs [6][28][29]. The costs move to GPU serving, RDMA embedding transport and large hash tables.
6. **Candidate isolation makes transformer ranking work in production.** If candidates cannot attend to each other, scores are batch-invariant, cacheable and comparable across requests and sources [28][29].
7. **Semantic IDs plus hashing solve cold start for content.** Hash embeddings need no vocabulary. Residual-quantized SIDs over multimodal embeddings let the model place an unseen post next to similar ones [29].
8. **Put the index in the checkpoint.** Baking the candidate-tower index into each retrieval checkpoint removes the version skew between user tower and item index [29].
9. **Use principled diversity and exploration.** Replace stacks of heuristics with a DPP reranker over embeddings and with Thompson sampling for new authors under an explicit Beta prior [30][33].
10. **Separate relevance from eligibility.** Ranking and visibility filtering are different services with different inputs. Run expensive VF after top-K. Apply stricter rules to out-of-network recommendations than to followed content [28].
11. **Some components last a long time.** SimClusters (2020) remains a 2026 candidate source, and PageRank-based credibility runs from Tweepcred to user-cred-v2 [8][25][28].
12. **Make open source part of operations.** Syncing production defaults into the repo, publishing experiment diffs such as the bidirectional boost, and pairing code with per-account label reports is a model for transparency in algorithmic feeds [28][31].

---

## References

1. "the-algorithm: Source code for Twitter's Recommendation Algorithm" (README), Twitter/X, 2023. https://github.com/twitter/the-algorithm
2. "Twitter's Recommendation Algorithm", Twitter Engineering Blog, 2023. https://blog.x.com/engineering/en_us/topics/open-source/2023/twitter-recommendation-algorithm
3. "the-algorithm-ml: Source code for Twitter's Recommendation Algorithm (ML)", Twitter/X, 2023. https://github.com/twitter/the-algorithm-ml
4. "Heavy Ranker" README (projects/home/recap), Twitter/X, 2023. https://github.com/twitter/the-algorithm-ml/blob/main/projects/home/recap/README.md
5. "Heavy Ranker FEATURES.md", Twitter/X, 2023. https://github.com/twitter/the-algorithm-ml/blob/main/projects/home/recap/FEATURES.md
6. "Home Mixer" README, Twitter/X, 2023. https://github.com/twitter/the-algorithm/blob/main/home-mixer/README.md
7. "ScoredTweetsParam.scala" (Home Mixer scored-tweets parameters), Twitter/X, 2023-2025. https://github.com/twitter/the-algorithm/blob/main/home-mixer/server/src/main/scala/com/twitter/home_mixer/product/scored_tweets/param/ScoredTweetsParam.scala
8. "SimClusters: Community-based Representations for Heterogeneous Recommendations at Twitter" (README), Twitter/X, 2023. https://github.com/twitter/the-algorithm/blob/main/src/scala/com/twitter/simclusters_v2/README.md
9. Satuluri et al., "SimClusters: Community-Based Representations for Heterogeneous Recommendations at Twitter", KDD 2020 (ADS track). https://www.kdd.org/kdd2020/accepted-papers/view/simclusters-community-based-representations-for-heterogeneous-recommendatio
10. "Real Graph" README (interaction_graph), Twitter/X, 2023. https://github.com/twitter/the-algorithm/blob/main/src/scala/com/twitter/interaction_graph/README.md
11. "UserTweetEntityGraph (UTEG)" README, Twitter/X, 2023. https://github.com/twitter/the-algorithm/blob/main/src/scala/com/twitter/recos/user_tweet_entity_graph/README.md
12. "GraphJet" README, Twitter, 2016. https://github.com/twitter/GraphJet
13. Sharma, Jiang, Bommannavar, Larson, Lin, "GraphJet: Real-Time Content Recommendations at Twitter", PVLDB 9(13), 2016. http://www.vldb.org/pvldb/vol9/p1281-sharma.pdf
14. Gupta, Goel, Lin, Sharma, Wang, Zadeh, "WTF: The Who to Follow Service at Twitter", WWW 2013. https://web.stanford.edu/~rezab/papers/wtf_overview.pdf
15. El-Kishky et al., "TwHIN: Embedding the Twitter Heterogeneous Information Network for Personalized Recommendation", KDD 2022 (arXiv 2202.05387). https://arxiv.org/abs/2202.05387
16. "TwHIN in torchrec" README, Twitter/X, 2023. https://github.com/twitter/the-algorithm-ml/blob/main/projects/twhin/README.md
17. Wang, She, Zhang, "MaskNet: Introducing Feature-Wise Multiplication to CTR Ranking Models by Instance-Guided Mask", arXiv 2102.07619, 2021. https://arxiv.org/abs/2102.07619
18. "Earlybird Light Ranker" README, Twitter/X, 2023. https://github.com/twitter/the-algorithm/blob/main/src/python/twitter/deepbird/projects/timelines/scripts/models/earlybird/README.md
19. "Tweet Search System (Earlybird)" README, Twitter/X, 2023. https://github.com/twitter/the-algorithm/blob/main/src/java/com/twitter/search/README.md ; and Busch et al., "Earlybird: Real-Time Search at Twitter", ICDE 2012. http://notes.stephenholiday.com/Earlybird.pdf
20. "CR-Mixer" README, Twitter/X, 2023. https://github.com/twitter/the-algorithm/blob/main/cr-mixer/README.md
21. "Follow Recommendations Service" README, Twitter/X, 2023. https://github.com/twitter/the-algorithm/blob/main/follow-recommendations-service/README.md
22. "Visibility Filtering" README (visibilitylib), Twitter/X, 2023. https://github.com/twitter/the-algorithm/blob/main/visibilitylib/README.md
23. "Navi: High-Performance Machine Learning Serving Server in Rust" README, Twitter/X, 2023. https://github.com/twitter/the-algorithm/blob/main/navi/README.md
24. "Product Mixer" README, Twitter/X, 2023. https://github.com/twitter/the-algorithm/blob/main/product-mixer/README.md
25. "Tweepcred" README, Twitter/X, 2023. https://github.com/twitter/the-algorithm/blob/main/src/scala/com/twitter/graph/batch/job/tweepcred/README
26. "Recos-Injector" README, Twitter/X, 2023. https://github.com/twitter/the-algorithm/blob/main/recos-injector/README.md
27. "Pushservice" README, Twitter/X, 2023. https://github.com/twitter/the-algorithm/blob/main/pushservice/README.md
28. "X For You Feed Algorithm" (x-algorithm README and git history; initial release 2026-01-20, updates through 2026-09), xAI / X, 2026. https://github.com/xai-org/x-algorithm
29. "Phoenix: Recommendation System" README, xAI / X, 2026. https://github.com/xai-org/x-algorithm/blob/main/phoenix/README.md
30. "home-mixer/params/param.rs" (production default parameters and scoring weights, synced 2026-10-08), xAI / X, 2026. https://github.com/xai-org/x-algorithm/blob/main/home-mixer/params/param.rs
31. "Example algorithm diff: bidirectional follow boost" (docs/BIDIRECTIONAL_BOOST_CHANGE.md), xAI / X, 2026. https://github.com/xai-org/x-algorithm/blob/main/docs/BIDIRECTIONAL_BOOST_CHANGE.md
32. "Training the Phoenix recsys model" (phoenix/TRAINING.md), xAI / X, 2026. https://github.com/xai-org/x-algorithm/blob/main/phoenix/TRAINING.md
33. "vm-ranker" source (dpp.rs, args.rs), xAI / X, 2026. https://github.com/xai-org/x-algorithm/tree/main/vm-ranker
34. "grok-1" open-source release, xAI, 2024. https://github.com/xai-org/grok-1
35. "Thunder" source (in-memory post store), xAI / X, 2026. https://github.com/xai-org/x-algorithm/tree/main/thunder
36. "local_prod.yaml" (heavy ranker MaskNet config), Twitter/X, 2023. https://github.com/twitter/the-algorithm-ml/blob/main/projects/home/recap/config/local_prod.yaml
37. "mask_net.py" (MaskNet implementation), Twitter/X, 2023. https://github.com/twitter/the-algorithm-ml/blob/main/projects/home/recap/model/mask_net.py
