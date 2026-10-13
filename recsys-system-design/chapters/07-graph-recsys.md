# 07. Graph-Based Recommendation in Production: Random Walks, Graph Embeddings and GNNs at Web Scale

Graphs are the oldest structure in recommendation. "People who bought X also bought Y" is a two-hop walk on a user-item bipartite graph. "People you may know" is a triangle-closing walk on a social graph. Most of the large consumer platforms (Pinterest, Twitter/X, LinkedIn, Alibaba, Uber, Meta) have published at least one production system that treats recommendation as a graph problem. This chapter collects those designs and groups them into four families. Each family makes a different trade-off between freshness, expressiveness and serving cost:

1. **Online random-walk serving.** An in-memory graph answers each query with a short random walk or a SALSA-style traversal, all at request time. Examples: Pinterest Pixie, Twitter GraphJet (UTEG/UTG/UUG/UVG), Twitter's WTF/SALSA follow recommendations.
2. **Community or sparse factorisation representations.** Offline graph clustering produces sparse, interpretable vectors that streaming jobs keep fresh. Example: Twitter SimClusters and its SimClusters-ANN serving layer.
3. **Shallow (transductive) graph embeddings.** Large-scale node-embedding training (DeepWalk, TransE-style knowledge-graph embeddings) produces dense vectors for ANN retrieval and for use as ranking features. Examples: Alibaba EGES, Twitter TwHIN, Meta PyTorch-BigGraph.
4. **Inductive GNNs.** GraphSAGE-style aggregators compute embeddings from features plus sampled neighbourhoods, usually offline in batch and sometimes nearline. Examples: Pinterest PinSage (and PinnerSage built on it), Uber Eats GraphSAGE, LinkedIn LiGNN, Alibaba AliGraph/Graph-Learn and its Dynamic Graph Service.

A note on sourcing. Facts about Twitter/X systems (SimClusters, GraphJet services, TwHIN training code, Recos-Injector, the SALSA and two-hop candidate sources) and about Alibaba Graph-Learn/DGS were read directly from the open-source repositories and their documentation [1][2][3][4][5][6][7]. Numbers from the academic papers (Pixie, PinSage, PinnerSage, EGES, TwHIN, LiGNN, Uber Eats) are given as reported by their authors and should be treated as approximate. Check them against the cited paper before you quote them.

---

## 1. A taxonomy of graph recommenders in production

```
                    where is the graph computation done?
                 request time                  offline / nearline
              +-----------------------+    +------------------------------+
  no learned  | Pixie, GraphJet,      |    | SimClusters (community       |
  params      | SALSA/WTF, 2-hop walk |    | detection + sparse vectors), |
              | (walks over in-memory |    | Tweepcred PageRank,          |
              |  graph)               |    | RealGraph aggregates         |
              +-----------------------+    +------------------------------+
  learned     | AliGraph DGS (sample  |    | PinSage, LiGNN, Uber Eats    |
  params      |  subgraph online, GNN |    | GraphSAGE (inductive GNN);   |
              |  forward in TF-Serve) |    | EGES, TwHIN, PBG (shallow    |
              |                       |    | embeddings) -> ANN index     |
              +-----------------------+    +------------------------------+
```

Request-time walks give the freshest results, at the cost of a large RAM-resident graph per replica. Offline embeddings are cheap to serve because retrieval becomes ANN lookup, but they are stale between refreshes, and transductive models do not cover new nodes. Inductive GNNs plus nearline inference are the current compromise. Every large platform in this chapter runs more than one family at once, usually as several candidate sources feeding one ranker.

---

## 2. Pinterest Pixie: real-time random walks on a 3-billion-node graph

### Problem

Pinterest needed to recommend pins (images) in real time for Homefeed, Related Pins and other surfaces. The recommendations had to be personalised to a user's recent actions, which are a handful of "query pins". The earlier pipeline was Hadoop-based and precomputed results offline, so it could not react within a session [8].

### Architecture

Pixie keeps the whole pin-board bipartite graph in RAM on each server. Pins are saved into boards, so every save is an edge. The paper reports a graph of roughly 3 billion nodes and 17 billion edges, compacted to fit in the memory of a single large machine. Each server is a full replica, which removes any cross-machine communication from the walk [8].

```
 user actions (saves, clicks)      offline graph build (Hadoop, daily)
          |                                   |
          v                                   v
  query pins Q={q1..qk}, weights w_q    pruned pin<->board graph
          |                          (CSR arrays, loaded into RAM)
          v                                   |
  +--------------------------- Pixie server --+-----------------+
  | for each q in Q: budget N_q ~ w_q * f(degree(q))            |
  |   repeat: walk q -> board -> pin -> board ... (biased       |
  |           by user language/topic), restart w.p. alpha       |
  |   count visits V_q[p]; early-stop when n_p pins have >= n_v |
  | combine: V[p] = (sum_q sqrt(V_q[p]))^2   (multi-hit boost)  |
  +------------------------------------------+------------------+
                                             v
                            top-K pins -> ranking model -> surface
```

### Algorithm and key tricks

- **Random walk with restarts on the bipartite graph.** Each step goes pin to board to pin. Visit counts approximate personalised PageRank from the query pin [8].
- **Biased walks.** Edge selection is biased towards boards and pins that match user features such as language or topic. This personalises the walk without storing per-user graphs [8].
- **Multiple weighted query pins.** Each query pin gets its own step budget. The budget scales with the pin's weight (recency, action type) and is adjusted by the pin's degree, so popular pins do not take over the walk. Visit counts are then combined with a sum of square roots, squared, which rewards candidates that several query pins reach over candidates that one query pin reaches many times [8].
- **Early stopping.** A walk stops once a target number of pins have each been visited at least a minimum number of times. This cuts latency substantially with little loss in quality [8].
- **Graph pruning.** Boards that are too diverse, which the paper measures as high topic entropy over their pins, are removed. Very popular pins have their edges pruned down to the most relevant boards. The paper reports that pruning made the graph smaller and also improved recommendation quality, so it was not only a memory-saving measure [8].

### Scale and results (as reported, approximate)

The paper reports that Pixie served on the order of a billion-plus users' worth of requests with per-query latencies in the tens of milliseconds at high percentiles. It also reports large engagement lifts over the previous Hadoop-based system, on the order of +50% in some surfaces [8]. Treat the exact figures as approximate.

### Lessons

- If the graph fits in RAM on one machine, **replicate it rather than partition it**. Walks are dominated by pointer chasing, and network hops would ruin them.
- Pruning the graph is a modelling decision as much as a memory decision. Removing noisy hub nodes improves walk quality.
- Random walks give a candidate *distribution*. A learned ranker still has to sit on top.

---

## 3. Pinterest PinSage: GraphSAGE at 3 billion nodes

### Problem

Pixie needs graph connectivity at query time and cannot use content features. New pins with few saves get poor recommendations. Pinterest wanted one embedding per pin that combined visual and text features with graph structure. The embedding had to be computed for billions of items and served through nearest-neighbour search [9][10].

### Architecture

PinSage is a GraphSAGE-family inductive GCN [11] adapted to web scale. The paper reports a graph of about 3 billion nodes (pins and boards) and about 18 billion edges, trained on about 7.5 billion examples [9].

```
            TRAINING (GPU, producer-consumer)
  CPU producers: sample minibatch of (query, positive) pin pairs
                 -> random-walk neighbourhoods (top-T by visit count)
                 -> gather features (visual, text, ...)  ---+
                                                            v
  GPU consumer:  K=2 conv layers; importance pooling (weights =
                 normalised visit counts); max-margin loss with
                 random + curriculum "hard" negatives
                                                            |
            INFERENCE (MapReduce, no repeated computation)  v
  layer-1 embeddings for all nodes -> join -> layer-2 embeddings
                  -> pin embeddings -> ANN index (related pins, homefeed)
```

### Key tricks

1. **Random-walk-defined neighbourhoods ("importance pooling").** PinSage does not use the raw adjacency. It runs short random walks from each node and keeps the top-T nodes by visit count as the node's neighbourhood. The normalised counts become aggregation weights. This bounds fan-out (degree can be huge on Pinterest) and makes the neighbourhood relevance-weighted [9].
2. **Producer-consumer minibatching.** CPUs sample neighbourhoods and fetch features while GPUs run the convolutions, so the GPUs stay busy [9].
3. **Curriculum training with hard negatives.** Random negatives are too easy among billions of items. Later epochs add "hard" negatives: items ranked moderately high, but not top, by personalised PageRank from the query. The paper reports this was important for fine-grained relevance [9].
4. **MapReduce inference.** If the GCN ran per node with its sampled neighbourhood, overlapping neighbourhoods would be recomputed many times. PinSage instead computes each layer once for all nodes and joins the results, which removes the redundant work [9].
5. **Serving.** Embeddings go into an approximate nearest-neighbour index for candidate generation [9][10].

### Results (as reported, approximate)

In offline evaluation the paper reports large gains in hit-rate and MRR over content-only and earlier graph baselines. User studies and A/B tests showed engagement improvements for related-pin and homefeed recommendations [9][10].

### Descendant: PinnerSage (multi-modal user embeddings)

A single user vector averages over unrelated interests. PinnerSage (KDD 2020) represents each user as **several** PinSage-space embeddings. It clusters the user's recent action pins, the paper describes roughly 90 days of history, using Ward hierarchical clustering. Each cluster is represented by its **medoid**, an actual pin, not a centroid. Medoids are weighted by time-decayed frequency and a few are sampled per request for ANN retrieval [12]. Because PinSage embeddings stay fixed, PinnerSage only has to cluster the user's own actions. It never retrains the item model. Pinterest later moved part of this role to PinnerFormer, a transformer over action sequences [13]. That system is covered in the sequential-models chapter.

### Lessons

- Use random walks for *neighbourhood selection*, not only for retrieval. This gives bounded fan-out and relevance weights in one step.
- Batch inference must avoid redundant computation. Layer-wise MapReduce costs O(layers x nodes) instead of O(nodes x fanout^layers).
- Item embeddings that combine content and graph give a stable "coordinate system" that many downstream systems can reuse (PinnerSage, ads, ranking features).

---

## 4. Twitter/X: a full graph recommendation stack

Twitter has published the most graph-based recommendation infrastructure of any company. Most of it can be read in the open-sourced `the-algorithm` repository and its launch blog post [1][23]. That repository says candidate sourcing narrows about 1 billion posts to a few thousand per request [2], and in-network search supplies about 50% of For You posts [1]. Out-of-network candidates come largely from graph systems: GraphJet services, SimClusters, TwHIN and the Follow Recommendations Service.

```
 unified-user-actions / event bus (fav, RT, follow, reply, click ...)
        |                 |                          |
        v                 v                          v
  Recos-Injector     SimClusters Heron jobs     batch (Scalding/GCP)
  (filter, publish   (tweet embeddings updated   KnownFor, InterestedIn,
   per-service Kafka  on every fav)              producer/topic embeds,
   topics)                |                      TwHIN, RealGraph,
        |                 v                      Tweepcred PageRank
        v           SimClusters-ANN                    |
  GraphJet services  (cluster->top tweets index)       v
  UTEG UTG UUG UVG        |                  Representation Manager
        |                 |                  / Representation Scorer
        +--------+--------+-------------------------+
                 v
          CR-Mixer / tweet-mixer (fetch, mix, filter, light rank)
                 v
          Home Mixer -> heavy ranker -> For You timeline
```

### 4.1 GraphJet: in-memory, real-time bipartite graph engine

**Problem.** Tweets are short-lived. Recommendations such as "X liked this tweet" have to reflect engagement from the past hours, not yesterday's batch.

**Design.** GraphJet is a Java library that "maintains a full graph index over a sliding time window in memory on a single server" [3]. Edges are ingested from streams (Kafka), and lookups are served concurrently. It uses compact edge encoding and dynamic memory allocation: the graph is held as a series of time-ordered segments, and the oldest segment is dropped as the window slides. Each server can ingest up to **one million edges per second** and, in steady state, compute up to **500 recommendations per second**, which amounts to several million edge reads per second [3][14].

**Production services built on GraphJet** (from the repo READMEs [1]):

| Service | Graph | Window | Use |
|---|---|---|---|
| UTEG (UserTweetEntityGraph) | user -> tweet engagements (one-directional) | 24-48 h | "X liked" out-of-network tweets on Home; input is a weighted follow list |
| UTG (UserTweetGraph) | user <-> tweet (bi-directional) | 24-48 h | 1-, 2-, 3+-hop CF and random walks from seed users/tweets; more memory than UTEG |
| UUG (UserUserGraph) | user -> user follows | ~1 week | "who your network recently followed" account recommendations |
| UVG (UserVideoGraph) | user <-> video tweets | 24-48 h | video recommendations from seed tweets |

All four are stateful. They rebuild state from a dedicated Kafka topic that **Recos-Injector** produces. Recos-Injector is a streaming processor that consumes arbitrary event streams (favs, retweets, follows, client events), filters them, and publishes a cleaned stream to each GraphJet service [1].

**Algorithms.** UTEG's core is collaborative filtering by aggregation. It takes the requesting user's weighted follow list (weights from RealGraph), walks out to tweets those users recently engaged with, and scores each tweet by how many and which of those users engaged with it. The engagers are returned as social proof [1]. The VLDB paper describes SALSA-style random walks on the bipartite graph as the main personalised algorithm [14].

**Lessons.** A sliding window bounds memory and also acts as a freshness prior, because old engagement is forgotten automatically. Storing edges in only one direction (UTEG) halves memory but limits the traversals you can run (contrast with UTG).

### 4.2 Follow recommendations: SALSA, two-hop walks and graph features

The Follow Recommendations Service (FRS) collects many graph candidate sources [1]:

- **SALSA PYMK.** For each target user it builds a local bipartite graph. The left side holds the user plus top candidates from the forward and reverse address book, mutual follows, recent followings and recent followers, each source weighted differently. The right side holds those candidates plus the mutual follows of the left-side entries. It then runs a **personalised random walk with restarts** from the target user. This lineage goes back to Twitter's 2013 "WTF: Who to Follow" system [15].
- **Two-hop random walk.** Take the user's top K first-degree connections by RealGraph weight, expand each to its top L, drop first-degree nodes, score by weighted sum, and keep the top M.
- **Graph Feature Service.** Computes pairwise graph features for ranking, such as "how many of A's followings have favourited C" or "how many of A's followings follow C" [1].
- **RealGraph.** A gradient-boosted classifier, trained on daily-aggregated interaction graphs, that predicts whether user A will interact with user B. Its scores weight the edges for most of the walks above [1].
- **Tweepcred.** A PageRank-based user reputation score [1].

FRS ranks the union of candidates by a weighted sum of p(follow | rec) and p(positive engagement | follow) [1].

### 4.3 SimClusters: sparse community embeddings

**Problem.** Twitter wanted one representation that could cover users, tweets, topics and entities, could be updated in real time for tweets, and could be interpreted by people [16][1].

**Algorithm** (from the repo documentation [1]):

1. Treat follows as a **bipartite graph** of consumers (followers) and producers (followees).
2. Compute **producer-producer cosine similarity** over follower sets, and drop edges below a threshold.
3. Run **Metropolis-Hastings sampling-based community detection** with k communities on the similarity graph. The output is **KnownFor** (V), a maximally sparse matrix in which each producer belongs to at most one community. In production it covers the **top 20M producers** with **k of about 145,000 communities**.
4. **InterestedIn** U = A x V (follow matrix times KnownFor), with noise removal. These are long-term consumer embeddings.
5. **Producer embeddings**: cosine similarity between each producer's follower set and each community's InterestedIn vector. This lets a producer belong to many communities.
6. **Tweet embeddings** start empty. On every favourite, the faver's InterestedIn vector is added to the tweet's vector. A Heron (Summingbird) streaming job does this in real time and persists from Memcache to Manhattan.
7. **Topic embeddings** come from time-decayed favourites on topic-annotated tweets, or (in the TFG variant) from the sum of the topic's followers' fav-based InterestedIn vectors.

**Serving: SimClusters-ANN and approximate cosine.** The streaming job maintains two indexes: **top 400 tweets per cluster** and **top 100 clusters per tweet**. Exact cosine was too expensive. One home-timeline candidate path (TWISTLY) scanned up to 15,000 candidates per request (6 source tweets x 25 clusters x 100 tweets) at more than 6,000 RPS. So SimClusters-ANN uses a two-stage approximation [1]:

```
 source embedding SV (top N=50 clusters)
      |  fetch top M=400 tweets per cluster   (cluster->tweet index)
      v
 sparse approx matrix R (M*N tweets x N clusters)
      |  score = R . SV^T  (dot product on truncated vectors)
      v
 top X=200 tweets -> fetch full tweet embeddings (only ~6% of M*N)
      |  exact cosine vs SV, keep top Y above threshold Z
      v
 candidates (cosine >= ~0.7 treated as highly related in prod)
```

The second-stage cosine filter is there because raw dot products favour popular tweets [1]. The same index pattern is built in BigQuery for other engagement types, for example push-open-based indexes for notifications and video-view-based indexes for Home video [1].

**Lessons.** Sparse, interpretable embeddings with roughly 145K dimensions can be updated *additively* in streaming, which dense embeddings cannot easily do. They also give an inverted index (cluster to items) at no extra cost. The cost is a separate rescoring pass to remove popularity bias.

### 4.4 TwHIN: dense heterogeneous-graph embeddings

**Problem.** SimClusters is derived from the follow graph alone. Twitter wanted dense embeddings that combine many relation types (follows, favs, retweets, replies, ad clicks) across many entity types, for use in candidate retrieval and as model features [2][17].

**Model.** TwHIN is a knowledge-graph embedding over a heterogeneous information network. It uses translation-style scoring (TransE-like): each relation has a learned translation vector, and an edge (lhs, rel, rhs) scores highly when lhs + r is close to rhs. The paper describes training at billions of nodes and on the order of 10^11 edges with PyTorch-BigGraph-style partitioned training [17][18]. The open-source version reimplements the training in **TorchRec**. It reads parquet edge lists with `lhs`, `rel` and `rhs` columns, defines one embedding table per entity type and a `translation` operator per relation (example relations: fav, reply, retweet, magic_recs), and uses **in-batch negatives** [4]. Subsampled, anonymised follow and fav graphs were released as public datasets [4].

**Tricks reported in the paper (approximate):**

- **Multi-modal user embeddings.** Users can have several embeddings, made by clustering the TwHIN embeddings of the accounts they follow. This mirrors PinnerSage [17].
- The same embeddings are reused across tasks: candidate generation (ANN), features in heavy rankers, ads, and harmful-content detection [17].

Twitter's retrieval-signals table shows TwHIN uses follows, favs, retweets, quotes and replies as features or labels, and notification opens as features [2].

### 4.5 How the Twitter graph pieces fit together

| Component | Family | Freshness | Representation | Serving |
|---|---|---|---|---|
| UTEG/UTG/UVG | online walk / CF | seconds (Kafka) | none (raw graph) | in-memory GraphJet replicas |
| UUG, SALSA, 2-hop | online walk | seconds-days | none | GraphJet / FRS |
| SimClusters | community + sparse vectors | KnownFor batch; tweets real time | ~145K-dim sparse | cluster->tweet inverted index |
| TwHIN | shallow KG embedding | batch | dense | ANN + ranking features |
| RealGraph | GBDT edge scorer | daily | edge weights | feature store |

---

## 5. LinkedIn LiGNN: production GNNs on a temporal, heterogeneous graph

### Problem

LinkedIn's graph links members, jobs, companies, posts, ads, groups and skills. It runs recommendation across Feed, Jobs, People You May Know (PYMK) and Ads. Static shallow embeddings did not handle cold-start entities (new jobs, new posts), temporal drift, or the cost of training on very large graphs [19].

### Architecture and techniques (as reported in [19])

```
 heterogeneous member/job/post/ad/company graph (multi-edge-type, timestamps)
        |
        v
 graph densification for cold start (add edges from low-degree nodes
        to similar/content-matched nodes)
        |
        v
 GNN training: temporal neighbourhood sampling (multi-hop),
   learnable ID embeddings + content features, long-term losses,
   adaptive neighbour sampling, grouped/sliced batches,
   shared-memory sampling queue, local gradient optimisation
        |
        v
 embeddings refreshed (batch + nearline) -> feature store / ANN
        |
        v
 downstream rankers & retrievers: Feed, Jobs, PYMK, Ads
```

- **Temporal graph architectures with long-term losses.** Sampling and training use edge timestamps, so the model predicts future engagement from past neighbourhoods. This avoids leakage and captures drift.
- **Graph densification for cold start.** Low-degree nodes get extra edges, for example to content-similar entities, so the GNN has neighbours to aggregate.
- **ID embeddings plus multi-hop neighbour sampling.** This combines transductive capacity with inductive generalisation.
- **Training-systems work.** The paper reports about a **7x training speedup** from adaptive neighbour sampling, grouping and slicing of training batches, a specialised shared-memory queue, and local gradient optimisation [19].
- **Nearline inference** keeps embeddings fresh for new entities.

### Reported online impact (approximate)

The paper reports A/B lifts that include, approximately: about +1% in job-application hearing-back rate, about +2% Ads CTR, about +0.5% feed engaged daily active users, about +0.2% sessions, and about +0.1% weekly active users from people recommendations [19]. The individual lifts are small, but LiGNN runs as an embedding platform, so they add up across many products.

### Lessons

- In production, **temporal correctness** matters more than architecture novelty. Sampling neighbours from the future inflates offline metrics.
- Cold start is a graph-construction problem (densification) as much as a model problem.
- The biggest wins were in the **systems layer**: sampling throughput, data layout and queueing.

---

## 6. Uber Eats: GraphSAGE for dish and restaurant recommendations

### Problem

Uber Eats wanted to recommend dishes and restaurants. Ordering is sparse and strongly local: a user orders from a small set of nearby restaurants [20].

### Design (as reported in [20])

- **Graphs.** Bipartite user-dish and user-restaurant graphs, built **per city**, because the delivery radius makes cross-city edges almost meaningless. Edge weights encode order frequency.
- **Model.** GraphSAGE [11] with sampled neighbourhoods and node features (for example dish and restaurant descriptors).
- **Loss.** Max-margin ranking loss, extended so that heavily ordered (high-weight) positives are pushed above lightly ordered "low-rank positives", and those above negatives. The graph's edge weights are therefore used in the loss as well as in sampling.
- **Pipeline.** Daily offline training and embedding generation on Uber's ML platform. User-dish and user-restaurant similarity from the embeddings becomes **features** in the downstream recommendation ranking model, and also feeds candidate retrieval.
- **Results (approximate).** The blog reports a notable offline AUC improvement for the ranker when graph features were added, with the graph-based similarity among the most important features. Online tests in initial cities showed CTR uplift [20].

```
 orders (per city) -> user-dish / user-restaurant weighted bipartite graphs
        -> GraphSAGE (2-hop sampled, max-margin + low-rank positives)
        -> user, dish, restaurant embeddings (daily)
        -> similarity features -> ranking model -> Eats homefeed carousels
```

### Lessons

Partitioning the graph by geography gives scalability (each city is trained independently) and also encodes a hard constraint. A GNN does not have to be the recommender: GNN embeddings used as *features* in an existing ranker are often the lowest-risk way to ship.

---

## 7. Alibaba: session graphs, side information and GNN platforms

### 7.1 EGES: billion-scale commodity embedding (Taobao)

**Problem.** Taobao matches about a billion users to about two billion items in candidate generation. Item-item similarity from co-purchase CF misses long-tail and new items [21].

**Pipeline (as reported in [21]):**

1. **Build a weighted directed item graph from user sessions.** Behaviour sequences are split into sessions (the paper uses a time window of about one hour). Consecutive items in a session become a directed edge, weighted by co-occurrence count across all users. The paper also describes filtering noise such as very short clicks and over-active users.
2. **BGE (base graph embedding).** DeepWalk-style weighted random walks plus skip-gram.
3. **GES.** Adds side information (category, shop, brand, price level and similar) with separate embeddings that are averaged with the item ID embedding, which helps cold-start items.
4. **EGES.** Learns per-item **attention weights over side-information embeddings** (softmax of learned scalars), because the importance of brand or shop differs by item.

**Deployment.** Item embeddings drive item-to-item similarity in the matching stage. Each item's nearest neighbours by embedding become its candidate list. The paper reports CTR gains over the previous CF-based matching in online A/B tests [21].

**Lesson.** A session graph combined with random walks turns ordered co-occurrence into multi-hop transitive similarity, and side-information embeddings close the cold-start gap at almost no serving cost.

### 7.2 AliGraph / Graph-Learn and the Dynamic Graph Service

Alibaba open-sourced its GNN platform, **Graph-Learn (formerly AliGraph)**, which is used "in many scenarios within Alibaba, such as search recommendation, network security, and knowledge graph" [5][22]. It has two parts:

- **GraphLearn-Training.** Distributed sampling over batch graphs, a gremlin-like Graph Sampling Language, TensorFlow and PyTorch support, and offline or incremental GNN training. The documented production loop retrains incrementally every hour on a window of graph data and pushes the model to TF Serving [5].
- **Dynamic Graph Service (DGS).** Online inference on dynamic graphs [6]. DGS does not query a general graph store at request time. Instead, it **pre-samples the update stream according to a pre-registered sampling query** and caches only the sampled results, so each request becomes a local memory lookup.

```
 graph updates (stream/bulk)
    -> Data Loader (build Vertex/EdgeUpdateRecord, partition)
    -> Sampling Workers (decompose n-hop query into 1-hop samplers,
       Sample Store + Subscription Table, Subgraph Router)
    -> Serving Workers (all n-hop samples for a vertex co-located;
       SubGraph Store)                      <- client request (Java GSL)
    -> sampled subgraph -> TF model service -> prediction
 (all hops connected via fault-tolerant queues)
```

Documented performance: on one node (64 cores, 256 GB RAM), DGS completes a **2-hop sampling query within 20 ms P99**, at **20,000 QPS**, while ingesting **110 MB/s** of graph updates, and it scales out roughly linearly [6]. Alibaba also maintains **Euler**, a separate graph-learning framework from its advertising (Alimama) teams [7].

**Lesson.** The DGS design applies "materialise the query at write time" to GNN inference. If the sampling pattern is known in advance, keeping the sample current incrementally is cheaper than running the traversal on every read.

---

## 8. Meta PyTorch-BigGraph: the shallow-embedding workhorse

PyTorch-BigGraph (PBG) is Meta's open-source system for learning embeddings on graphs "with up to billions of entities and trillions of edges" [18]. It supports TransE, RESCAL, DistMult and ComplEx with per-relation operators over shared entity embeddings. Its scaling techniques are:

- **Graph partitioning.** Entities are split into P partitions and edges into P x P buckets, so only two partitions need to be in memory at once.
- **Multi-threaded and distributed training** with a shared parameter server for relation parameters.
- **Batched negative sampling.** Negatives are reused across a chunk of positives, which turns scoring into a matrix multiply [18].

TwHIN's original training followed this pattern, and its TorchRec reimplementation keeps the same edge-list (lhs, rel, rhs) contract [4][17]. PBG is the reference design when you need *transductive* embeddings for every node of a huge multi-relational graph on commodity hardware.

---

## 9. Cross-system comparison

| System (org, year) | Family | Graph | Scale (reported, approx.) | Serving path | Freshness |
|---|---|---|---|---|---|
| Pixie (Pinterest, 2018) [8] | online random walk | pin-board bipartite | ~3B nodes, ~17B edges, RAM per server | walk per request, tens of ms | graph rebuilt periodically; query pins real time |
| PinSage (Pinterest, 2018) [9] | inductive GCN | pin-board + features | ~3B nodes, ~18B edges, ~7.5B examples | MapReduce inference -> ANN | batch |
| PinnerSage (Pinterest, 2020) [12] | clustering on GNN embeds | user action history | per-user multi-medoid | medoid ANN (HNSW-style) | daily + real-time actions |
| GraphJet / UTEG (Twitter, 2016) [3][14] | online CF / SALSA | user-tweet bipartite, sliding window | 1M edges/s ingest, 500 recs/s per server | in-memory per replica | seconds; 24-48 h window |
| SimClusters (Twitter, 2020) [1][16] | community detection, sparse vectors | follow bipartite | 20M producers, ~145K clusters | cluster->top-400-tweets index; >6K RPS path | KnownFor batch; tweets streaming |
| TwHIN (Twitter, 2022) [4][17] | KG embedding (translation) | multi-relation HIN | billions of nodes, ~10^11 edges | ANN + features | batch |
| LiGNN (LinkedIn, 2024) [19] | temporal GNN | member/job/post/ad HIN | very large; 7x training speedup | features + retrieval, nearline | batch + nearline |
| Uber Eats GNN (Uber, 2019) [20] | GraphSAGE | user-dish/restaurant per city | per-city graphs | ranking features + retrieval | daily |
| EGES (Alibaba, 2018) [21] | random walk + skip-gram + side info | session item graph | ~1B users, ~2B items | item-item ANN in matching | batch |
| Graph-Learn DGS (Alibaba) [5][6] | online GNN sampling | dynamic HIN | 2-hop P99 20 ms, 20K QPS/node | pre-sampled cache -> TF Serving | streaming |
| PBG (Meta, 2019) [18] | KG embedding | multi-relation | billions of entities, trillions of edges | offline embeddings | batch |

---

## 10. Recurring design patterns

### 10.1 Random walks show up at every layer

The same primitive does different jobs in different systems:

- As the **retriever** (Pixie, UTG, SALSA).
- As a **neighbourhood sampler** for GNNs (PinSage importance pooling).
- As the **training corpus** for skip-gram (EGES/DeepWalk).
- As a **negative-mining oracle**: PinSage uses PPR ranks to select hard negatives.

### 10.2 Bipartite thinking

Pinterest (pin-board), Twitter (consumer-producer, user-tweet) and Uber (user-dish) all project their data onto bipartite graphs. The reasons are practical:

- Two-hop walks give item-item similarity directly.
- SALSA's hub/authority split matches consumer and producer roles.
- Storage can be one-directional (UTEG) when only one traversal direction is needed.

### 10.3 Popularity control

Hubs dominate every graph method, so each system controls popularity explicitly:

- Pixie prunes edges on popular pins and scales walk budgets by degree.
- SimClusters re-scores by cosine rather than dot product.
- PinSage uses top-T neighbour truncation.
- EGES filters over-active users from sessions.

### 10.4 Multi-interest users

When an item embedding space is good, users are usually represented as *several* points in it. PinnerSage uses medoids of action clusters, and TwHIN builds multi-modal user embeddings from clusters of followed accounts.

### 10.5 Freshness tiers

Production systems almost always combine three tiers:

1. A slow batch backbone: KnownFor, TwHIN, PinSage.
2. A streaming update of item vectors: SimClusters tweet embeddings, LiGNN nearline.
3. An online traversal over a short window: GraphJet, Pixie.

### 10.6 Sampling infrastructure is the bottleneck

PinSage's producer-consumer loop, LiGNN's shared-memory queues and 7x speedup, and DGS's pre-sampling all make the same point. The engineering effort in a production GNN goes into **neighbourhood sampling and feature fetching**, not into the convolution.

---

## 11. Key takeaways for system designers

1. **Start with walks before GNNs.** A replicated in-memory bipartite graph with random walks with restart (Pixie) or SALSA (GraphJet) is a strong, explainable, real-time candidate source. It has no training loop, and it returns social proof ("liked by X") for free.
2. **Replicate the graph if it fits in RAM.** Pixie and GraphJet both keep the whole working graph per server. Distributed walks are latency poison. Use sliding windows (24-48 h in GraphJet services) or pruning to make the graph fit.
3. **Separate representation from serving.** Twitter's Representation Manager/Scorer and Pinterest's PinSage-as-coordinate-system let many products share one embedding space. Treat graph embeddings as a platform.
4. **Choose transductive or inductive deliberately.** Shallow embeddings (TwHIN, PBG, EGES) are cheap and strong for established nodes. Inductive GNNs (PinSage, LiGNN, Uber) cover cold-start items through features. EGES-style side-information embeddings are a cheap halfway point.
5. **Sparse, additive representations stream well.** The SimClusters approach (add the InterestedIn vector of each user who favourites a tweet) gives real-time tweet embeddings plus an inverted index. Plan a cosine rescoring pass to undo popularity bias.
6. **Use two-stage approximate scoring to save I/O.** SimClusters-ANN fetches only about 6% of full embeddings by first scoring truncated vectors. The same idea (coarse score, then exact rescoring of a short list) applies to any embedding-retrieval path.
7. **Be strict about time.** Use temporal neighbourhood sampling, label windows after feature windows (RealGraph labels the day after the candidate period [1]), and long-horizon losses (LiGNN).
8. **Engineer for hubs.** Use degree-aware walk budgets, neighbour truncation, entropy-based pruning and cosine normalisation. Without them, graph recommenders turn into popularity recommenders.
9. **Ship GNN output as features first.** Uber Eats and LinkedIn both report most of their wins from adding graph-derived embeddings or similarities to existing rankers and retrievers. This is lower risk than replacing the stack.
10. **Precompute the query when the pattern is fixed.** Alibaba DGS keeps the n-hop sample current incrementally as the graph changes, which reaches 20 ms P99 at 20K QPS per node. Prefer this design when every request runs the same sampling pattern.

---

## References

1. Twitter/X, "the-algorithm" repository: README files for SimClusters v2, SimClusters ANN, user_tweet_entity_graph, user_tweet_graph, user_user_graph, user_video_graph, recos-injector, follow-recommendations-service (SALSA, two-hop random walk), graph-feature-service, interaction_graph (RealGraph), representation-manager, representation-scorer, cr-mixer. X Corp., 2023. https://github.com/twitter/the-algorithm
2. Twitter/X, "Signals for Candidate Sources" (RETREIVAL_SIGNALS.md), the-algorithm repository, 2023. https://github.com/twitter/the-algorithm/blob/main/RETREIVAL_SIGNALS.md
3. Twitter, "GraphJet" open-source repository README, 2016. https://github.com/twitter/GraphJet
4. Twitter/X, "TwHIN in torchrec" project README and configs, the-algorithm-ml repository, 2023. https://github.com/twitter/the-algorithm-ml/blob/main/projects/twhin/README.md
5. Alibaba, "Graph-Learn (formerly AliGraph)" repository README, 2020-2023. https://github.com/alibaba/graph-learn
6. Alibaba, "Dynamic Graph Service (DGS) Introduction", Graph-Learn documentation (docs/en/dgs/intro.md). https://github.com/alibaba/graph-learn/blob/master/docs/en/dgs/intro.md
7. Alibaba (Alimama), "Euler" graph learning framework repository. https://github.com/alibaba/euler
8. C. Eksombatchai, P. Jindal, J. Z. Liu, Y. Liu, R. Sharma, C. Sugnet, M. Ulrich, J. Leskovec, "Pixie: A System for Recommending 3+ Billion Items to 200+ Million Users in Real-Time", Pinterest / WWW 2018. https://arxiv.org/abs/1711.07601
9. R. Ying, R. He, K. Chen, P. Eksombatchai, W. L. Hamilton, J. Leskovec, "Graph Convolutional Neural Networks for Web-Scale Recommender Systems" (PinSage), Pinterest / KDD 2018. https://arxiv.org/abs/1806.01973
10. Pinterest Engineering, "PinSage: A new graph convolutional neural network for web-scale recommender systems", Pinterest Engineering Blog, 2018. https://medium.com/pinterest-engineering/pinsage-a-new-graph-convolutional-neural-network-for-web-scale-recommender-systems-88795a107f48
11. W. L. Hamilton, R. Ying, J. Leskovec, "Inductive Representation Learning on Large Graphs" (GraphSAGE), Stanford / NeurIPS 2017. https://arxiv.org/abs/1706.02216
12. A. Pal, C. Eksombatchai, Y. Zhou, B. Zhao, C. Rosenberg, J. Leskovec, "PinnerSage: Multi-Modal User Embedding Framework for Recommendations at Pinterest", Pinterest / KDD 2020. https://arxiv.org/abs/2007.03634
13. N. Pancha, A. Zhai, J. Leskovec, C. Rosenberg, "PinnerFormer: Sequence Modeling for User Representation at Pinterest", Pinterest / KDD 2022. https://arxiv.org/abs/2205.04507
14. A. Sharma, J. Jiang, P. Bommannavar, B. Larson, J. Lin, "GraphJet: Real-Time Content Recommendations at Twitter", Twitter / PVLDB 9(13), 2016. http://www.vldb.org/pvldb/vol9/p1281-sharma.pdf
15. P. Gupta, A. Goel, J. Lin, A. Sharma, D. Wang, R. Zadeh, "WTF: The Who to Follow Service at Twitter", Twitter / WWW 2013. https://web.stanford.edu/~rezab/papers/wtf_overview.pdf
16. V. Satuluri, Y. Wu, X. Zheng, Y. Qian, B. Wichers, Q. Dai, G. M. Tang, J. Jiang, J. Lin, "SimClusters: Community-Based Representations for Heterogeneous Recommendations at Twitter", Twitter / KDD 2020. https://www.kdd.org/kdd2020/accepted-papers/view/simclusters-community-based-representations-for-heterogeneous-recommendatio
17. A. El-Kishky, T. Markovich, S. Park, C. Verma, B. Kim, R. Eskander, Y. Malkov, F. Portman, S. Samaniego, Y. Xiao, A. Haghighi, "TwHIN: Embedding the Twitter Heterogeneous Information Network for Personalized Recommendation", Twitter / KDD 2022. https://arxiv.org/abs/2202.05387
18. A. Lerer, L. Wu, J. Shen, T. Lacroix, L. Wehrstedt, A. Bose, A. Peysakhovich, "PyTorch-BigGraph: A Large-scale Graph Embedding System", Facebook AI / SysML 2019 (paper and repository). https://mlsys.org/Conferences/2019/doc/2019/71.pdf and https://github.com/facebookresearch/PyTorch-BigGraph
19. F. Borisyuk et al., "LiGNN: Graph Neural Networks at LinkedIn", LinkedIn / KDD 2024. https://arxiv.org/abs/2402.11139
20. A. Jain, I. Liu, A. Sarda, P. Molino, "Food Discovery with Uber Eats: Using Graph Learning to Power Recommendations", Uber Engineering Blog, 2019. https://www.uber.com/blog/uber-eats-graph-learning/
21. J. Wang, P. Huang, H. Zhao, Z. Zhang, B. Zhao, D. L. Lee, "Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba" (EGES), Alibaba / KDD 2018. https://arxiv.org/abs/1803.02349
22. R. Zhu, K. Zhao, H. Yang, W. Lin, C. Zhou, B. Ai, Y. Li, J. Zhou, "AliGraph: A Comprehensive Graph Neural Network Platform", Alibaba / PVLDB 12(12), 2019. https://arxiv.org/abs/1902.08730
23. Twitter/X Engineering, "Twitter's Recommendation Algorithm", Twitter Engineering Blog, 2023. https://blog.x.com/engineering/en_us/topics/open-source/2023/twitter-recommendation-algorithm
