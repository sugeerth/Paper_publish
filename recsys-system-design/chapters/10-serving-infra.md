# 10. Online Serving and Inference: Latency Budgets, Model Servers, Caches and Cascades

A recommender is only as good as what it can compute inside one request. Every system in this
atlas, from a two-tower retriever to a trillion-parameter generative recommender, eventually
has to answer one question within a few hundred milliseconds: given this viewer, right now,
which K items do we return? This chapter covers the serving side of that question. It looks at
how production systems split a latency budget across stages, how they run model inference on
CPUs, GPUs and custom accelerators, how they fan out feature and embedding fetches, what they
cache, and how they shrink models through cascades, distillation, quantization and early exit.

The chapter builds on primary sources: open-source production code (X/Twitter's
`the-algorithm` and xAI's `x-algorithm` repositories, Meta's `generative-recommenders`,
TorchRec, FBGEMM), official documentation (Vespa, Triton Inference Server, NVIDIA Merlin
HugeCTR), and engineering papers and posts (DeepRecSys, Earlybird, Instagram Explore, Meta
Andromeda, Pinterest GPU inference).

> Source note. In this research pass, outbound access was limited to GitHub-hosted content.
> Facts drawn from repositories and documentation were read in full. Sections on Meta
> (Instagram Explore, Andromeda) and Pinterest summarize their published engineering posts.
> Those posts could not be re-fetched here, so their numbers are marked approximate and
> should be checked against the linked originals before anyone quotes them.

---

## 10.1 The shape of the problem: funnels and latency budgets

Almost every large recommender shares the same request-time skeleton, popularized by YouTube's
candidate-generation-then-ranking design [33]. A cheap stage looks at a huge corpus. Then progressively more expensive stages look at progressively fewer items:

```
 corpus (10^6 - 10^10 items)
   |
   |  RETRIEVAL / candidate generation   ANN, inverted index, graph walks, in-mem
   v                                     follow-graph stores; cost ~O(log N) per source
 10^3 - 10^4 candidates
   |
   |  FEATURE HYDRATION                  fan-out to feature stores, embedding servers,
   v                                     counters; often the latency long pole
   |
   |  LIGHT / EARLY-STAGE RANKING        LR, GBDT, two-tower dot product, distilled nets
   v
 10^2 - 10^3 candidates
   |
   |  HEAVY RANKING                      large MTML DNN / transformer, GPU or accelerator
   v
 10^1 - 10^2 candidates
   |
   |  RE-RANKING / POLICY                diversity (DPP, author decay), filters, ads mix
   v
 response (page of K items)
```

X's For You timeline makes the funnel explicit. Home Mixer fetches candidates from several
sources, hydrates "the ~6000 features needed for ranking", scores them with an ML model, and
then applies filters and heuristics such as author diversity, in-network/out-of-network
balance, feedback fatigue, dedup and visibility filtering. Only after that does it mix in ads
and who-to-follow modules [2]. xAI's 2026 release of the X algorithm keeps the same skeleton.
Its parts are query hydration, parallel candidate sources (Thunder in-network, Phoenix
retrieval, SimClusters, popular posts), candidate hydration, pre-scoring filters, Phoenix
scoring, a weighted value model, DPP re-ranking (`vm-ranker`), top-K selection and post-selection
visibility filters [11].

Why a cascade at all? The Vespa documentation puts the economics plainly: "A good quality
ranking expression will for most applications consume too much CPU to be runnable on all
retrieved or matched documents within the latency budget/SLA." The engineering job therefore
becomes finding "a first-phase function, which correlates sufficiently well with the
second-phase function" [13]. Each stage is a cheaper approximation of the next, and the
rerank counts at the stage boundaries are the main knobs for latency and cost.

A typical budget breakdown, written here as a design heuristic rather than a number from any
one source, looks like this:

| Stage | Typical share of p99 budget | Dominant cost |
|---|---|---|
| Query/user hydration | 5-15% | KV lookups, user sequence fetch |
| Candidate generation (parallel) | 15-25% | ANN / index traversal, slowest source |
| Feature hydration | 20-35% | fan-out RPCs, tail latency of slowest shard |
| Light ranking | 5-10% | CPU, simple models over many items |
| Heavy ranking | 20-35% | GPU/accelerator compute, batching delay |
| Re-rank, filters, mixing | 5-10% | rule evaluation, DPP, ads auction |

Two structural lessons follow from production systems:

1. Parallelize sources and fetches, then bound the tail. In X's Product Mixer framework,
   Candidate Pipelines run under Mixer/Recommendation Pipelines, and stages run "in parallel
   where it can" [8][11]. The response time is set by the slowest parallel branch, so each
   branch needs its own timeout and a fallback. Home Mixer keeps a reverse-chronological
   "backup ... pipeline in case Scored Tweets fails" [2].
2. Side effects belong off the critical path. xAI's Home Mixer records served posts,
   refreshes the post cache and logs client and ad events "after the response is sent" [11].

---

## 10.2 X / Twitter: an open-source reference serving stack

X has published more of its serving stack than any other large company [10]: the 2023
`the-algorithm` release and the 2025-2026 `x-algorithm` (xAI) release. That makes it the best
public case study for end-to-end serving.

### 10.2.1 Architecture (2023 Twitter stack)

```
                       +-------------------------------+
  client request ----> | Home Mixer (Product Mixer,     |
                       | Scala)                         |
                       +---+-----------+-----------+---+
                           |           |           |
              in-network   |   OON     |  graph     |  follow recs
           +---------------+  +--------+------+  +--+------------------+
           | Earlybird     |  | CR-Mixer /    |  | UTEG (GraphJet,     |
           | search index  |  | tweet-mixer   |  | in-memory graph)    |
           | + LR light    |  | + light rank  |  | FRS                 |
           |   ranker      |  | + result cache|  |                     |
           +-------+-------+  +-------+-------+  +----------+----------+
                   \                  |                    /
                    v                 v                   v
              +-------------------------------------------------+
              | Feature hydration (~6000 features): user-signal |
              | service, graph-feature-service, real-graph,     |
              | representation-manager (SimClusters/TwHIN emb.) |
              +--------------------------+----------------------+
                                         v
              +-------------------------------------------------+
              | Heavy ranker (parallel MaskNet, multi-task)     |
              | served by Navi (Rust; TF / ONNX runtimes)       |
              +--------------------------+----------------------+
                                         v
              filters / heuristics / visibility / ads + WTF mixing
```

Candidate sources. Earlybird, a Lucene-based real-time search system, does in-network
retrieval. It supplies roughly half of For You posts [1]. The index is split into a realtime
cluster (about the last 7 days of public tweets), a protected cluster, and an archive cluster.
Each cluster is partitioned into a sharded, replicated inverted index with a
single-writer/multi-reader design. Earlybird "roots" fan requests out to partitions and merge
the responses [6]. Out-of-network candidates come through CR-Mixer, which "fetches source
signals externally ... calls external candidate generation services, and caches results",
then applies filters and a light ranking step [7]. A second out-of-network source is the
in-memory UTEG graph built on GraphJet [1].

Light ranking at the index. The Earlybird light ranker is a logistic regression model,
"a simplified version of the heavy ranker which can run on a greater amount of tweets". It
runs inside the search index itself and uses static features computed at ingestion (has URL,
is reply, text-quality score, tweepcred) plus real-time engagement counters pushed by a
signal ingester [5]. This is an example of compute pushed to the data, the same idea as
Vespa's content-node first phase (Section 10.5). The README also admits the model "was last
trained several years ago" and was due to be replaced [5], a reminder that cascade stages
tend to rot when nobody owns them.

Heavy ranking. The Recap heavy ranker is a parallel MaskNet that outputs about ten
engagement probabilities (fav, retweet, reply, good click, negative feedback, report and
others). They are combined by a configurable weighted sum. The April 2023 weights included
reply-engaged-by-author 75.0, report -369.0 and fav 0.5. The weights were first set "so that,
on average, each weighted engagement probability contributes a near-equal amount to the
score" [9][28].

### 10.2.2 Navi: a minimalist Rust model server

Twitter replaced generic model serving for this path with Navi, "a high-performance,
versatile machine learning serving server implemented in Rust". It speaks the TensorFlow
Serving gRPC API and has pluggable runtimes: TensorFlow (most complete), ONNX Runtime (used
for home recommendation with a proprietary `BatchPredictRequest` Thrift input) and
experimental PyTorch [3]. Its CLI defaults show the core batching levers of any recsys model
server [4]:

| Navi flag | Default | Meaning |
|---|---|---|
| `max_batch_size` | 16 | max requests coalesced into one batch |
| `batch_time_out_millis` | 2 | max wait to accumulate a batch |
| `batch_drop_millis` | 90 | threshold to start dropping batches under stress |
| `model_check_interval_secs` | 300 | polling for new model versions |
| `onnx_gpu_ep` | (none) | choose CUDA or TensorRT execution provider |
| `max_warmup_records` | - | warm-up records before serving (TF) |

Lessons from Navi. (a) A server that does fewer things can be much faster. Navi trades
features for performance [3]. (b) Load shedding is a first-class setting. Dropping a batch
that is already 90 ms late beats letting a queue build. (c) Feature transformation from the
wire format (`dr_transform`, `segdense`) lives inside the server, so the client ships raw
records once [3]. (d) Hot model reloading by version directory gives continuous deployment
without restarts [3].

### 10.2.3 xAI x-algorithm (2026): transformer ranking built to be cacheable

The 2026 release replaces MaskNet with Phoenix, a transformer used both as a two-tower
retriever and as the ranker. It is trained in JAX and served by "a Rust gRPC serving engine"
[11][12]. Several serving-driven design decisions stand out:

- Candidate isolation. In the ranking transformer, "candidates cannot attend to each other",
  only to the user and history tokens. The attention mask is full bidirectional over
  user+history, and diagonal-only among candidates. The stated reason is serving: "the score
  for a post doesn't depend on which other posts are in the batch, making scores consistent
  and cacheable" [11][12]. It also means the user/history prefix can be computed once and
  shared by every candidate in the request. This is the same amortization idea as Meta's
  M-FALCON (Section 10.6).
- Batch-invariant serving. Together with a retrieval index baked "inside the checkpoint"
  (the trainer runs the candidate tower over the corpus at every checkpoint save), "a
  candidate's score depends only on the user and that candidate" [12]. Batch invariance
  makes caching, A/B debugging and replay possible.
- Hash-based embeddings plus semantic IDs. Multiple hash functions per entity mean "no
  dictionary service", so no ID-to-row lookup RPC sits on the serving path. Semantic IDs
  (residual-quantized codes, 6 levels x 256 codes) give new posts usable representations
  immediately [11][12].
- Multi-action outputs plus a separate value model. Phoenix outputs
  `[B, num_candidates, num_actions]` plus dwell regression. `RankingScorer` computes
  `sum(weight_i * P(action_i))`, then applies author-diversity decay, an out-of-network
  discount and a new-author boost. `vm-ranker` reorders by a determinantal point process
  over embeddings [11].
- Hardware. The open-source quickstart detects A100/H100/H200/GB200/GB300 and ships
  Hopper- and Blackwell-tuned attention kernels [12]. The ranker is a GPU workload.

---

## 10.3 Meta: Instagram Explore and the Andromeda ads retrieval engine

### 10.3.1 Instagram Explore (2023 post)

Meta's "Scaling the Instagram Explore recommendations system" (2023) describes a four-stage
funnel: retrieval, first-stage (early) ranking, second-stage (late) ranking, and final
re-ranking [31]. As summarized from the post (approximate; check the original):

```
  sources: two-tower retrieval, user-interaction histories, heuristics
            |   (item embeddings precomputed offline, ANN at request time)
            v
  first-stage ranking: lightweight two-tower-style model, trained partly to
            |          reproduce the late-stage ranker's output (distillation)
            v
  second-stage ranking: multi-task multi-label (MTML) DNN predicting many
            |           engagement events
            v
  value model: weighted combination of MTML outputs (weights tuned online)
            |
            v
  final re-ranking: integrity filters, diversity / business rules
```

Serving-relevant points in the post:

- Two-tower retrieval lets item embeddings be computed and indexed offline, so only the user
  tower runs at request time.
- The first-stage ranker is cheap because it is a two-tower-style model whose item side is
  precomputed. It is trained to mimic the heavier second stage, a form of ranking
  distillation (Section 10.8).
- The final score is a value model: a formula over predicted event probabilities with
  tunable weights. Product trade-offs then become configuration changes, not retraining.
  The post describes tuning these weights online (Bayesian-optimization-style
  experimentation).
- Caching and precomputation are used heavily, because Explore serves a very large user
  base under tight latency.

### 10.3.2 Andromeda (2024): GPU-native ads retrieval

Meta's Andromeda post (December 2024) describes a personalized ads retrieval engine built for
NVIDIA Grace Hopper Superchips and Meta's MTIA accelerator [32]. As reported in the post
(numbers approximate):

- It replaces a retrieval stage that had been constrained to lightweight models with a
  deep neural retrieval model co-designed with the hardware. The post cites about a 10,000x
  increase in retrieval model complexity.
- A hierarchical index makes retrieval sub-linear in the number of ads. The index structure
  is trained jointly with the model rather than being a separate offline ANN build.
- Reported results include a recall improvement of several percent and ads-quality gains
  on selected segments. The post attributes this to applying deep models at a stage that
  previously used simple scoring.
- The key lesson is that co-designing model, index and hardware (high-bandwidth unified
  CPU-GPU memory on Grace Hopper) makes "heavy" computation affordable at the stage that
  sees the most items. That moves the cascade's quality bottleneck upstream.

### 10.3.3 The Meta inference software substrate

Meta's open-source stack shows what production recsys inference is built on:

- FBGEMM, "a low-precision, high-performance matrix-matrix multiplications and
  convolution library for server-side inference". It supports small batch sizes and
  "row-wise quantization and outlier-aware quantization", and it fuses operators to beat
  bandwidth limits [18]. FBGEMM_GPU adds recsys-focused GPU operators (embedding bags,
  table-batched embeddings). FBGEMM GenAI adds FP8 row-wise quantization [18].
- TorchRec Inference, a C++ library for GPU inference. It captures the model with `torch.fx`
  and runs it as TorchScript in a C++ gRPC server. The reference example serves a DLRM on one
  GPU [17]. The README calls it "a simple reference ... not a fully scaled out solution"
  [17]. Production adds sharding, request batching and feature pipelines on top.
- DeepRecSys (Harvard + Facebook, ISCA 2020) characterized eight industry-representative
  models (DLRM-RMC1/2/3, NCF, Wide&Deep, MT-WnD, DIN, DIEN). It built DeepRecSched, a
  scheduler that chooses query batch size and splits queries between CPU cores and
  accelerators based on query-size and arrival distributions. It reports roughly 2x
  latency-bounded throughput and "over 30% latency reduction" in an at-scale production
  datacenter [22]. The core insight is that recsys queries vary widely in size (number of
  candidates), so a fixed batch policy wastes either latency or throughput.

---

## 10.4 GPU inference for ranking: Pinterest and the general recipe

### 10.4.1 Pinterest (2022 post)

Pinterest's "GPU-accelerated ML Inference at Pinterest" (2022) describes moving ranking
models from CPU to GPU serving [30]. In outline (approximate, per the original post):

- Motivation: CPU serving costs grew roughly linearly with model size, which blocked
  larger ranking models (bigger embeddings, transformer-style user-sequence modules).
- Result: GPU serving let Pinterest serve much larger models, on the order of 100x more
  compute per the post, at comparable or lower cost and latency.
- Techniques the post discusses: aggressive batching of candidates per request, CUDA Graphs
  to cut kernel-launch overhead, merging many small embedding-table lookups into fewer
  larger ops, mixed precision, and moving feature preprocessing onto the GPU so the host
  is not the bottleneck.

### 10.4.2 Why recsys GPU inference is different from vision/NLP inference

| Aspect | Typical CV/NLP inference | Recsys ranking inference |
|---|---|---|
| Work per request | 1 input | 100s-1000s of candidates x same user |
| Parameter mix | dense weights | mostly sparse embeddings (GB-TB), small dense MLP |
| Bottleneck | FLOPs | memory bandwidth (embedding gathers), launch overhead |
| Shared computation | none | user tower / user sequence shared across candidates |
| Batching | across requests | within request (candidates) and across requests |
| Model freshness | days-weeks | minutes-hours (online / streaming updates) |

Recipe distilled from the sources:

1. Batch inside the request first. Score all candidates of one request as one batch, so
   user-side work is computed once (Phoenix candidate isolation [12], M-FALCON [21]).
2. Batch across requests with a bounded delay. Triton's dynamic batcher holds requests up to
   `max_queue_delay_microseconds` (e.g. 100 us) to form preferred or maximum batch sizes.
   Its docs advise starting from defaults and raising the delay only "until the latency
   budget is exceeded" [16]. Navi's 2 ms timeout and 16-request batches play the same role
   [4].
3. Cut launch overhead with CUDA Graphs, fused kernels, Triton/CUTLASS custom kernels. The
   HSTU repository ships Triton kernels with roughly 2-2.3x speedups over PyTorch for the
   attention block on H100 [21].
4. Lower the precision. HSTU-Ultra FP8 projection GEMMs give 1.13-1.25x end-to-end
   forward speedups over BF16 on one H100 (e.g. 9.375 ms to 7.768 ms for a 12-layer,
   1024-length stack) [21]. FBGEMM provides INT8/row-wise quantization for CPU paths [18].
5. Keep embeddings near compute. Use GPU embedding caches and hierarchical parameter
   servers (Section 10.7).

---

## 10.5 Search-engine-based serving: Vespa and Elasticsearch

Many companies serve recommendations from a search engine. Retrieval and first-phase
ranking then run where the data lives, so candidate features never leave the shard.

### 10.5.1 Vespa phased ranking

Vespa splits the work into content nodes (stateful, hold documents and indexes) and a
stateless container (scatter-gather, global logic) [13]:

```
               query (user vector, filters, terms)
                              |
                    +---------v---------+
                    | stateless         |  scatter
                    | container         |-----------+-------------+
                    +---------^---------+           |             |
                              | gather         +----v----+   +----v----+
                              |                | content |   | content |
                              |                | node 1  |   | node N  |
                              |                |         |   |         |
                              |                | match:  |   | match:  |
                              |                | WAND /  |   | WAND /  |
                              |                | ANN     |   | ANN     |
                              |                | 1st ph. |   | 1st ph. |
                              |                | (all    |   | (all    |
                              |                |  hits)  |   |  hits)  |
                              |                | 2nd ph. |   | 2nd ph. |
                              |                | (top    |   | (top    |
                              |                |  ~100)  |   |  ~100)  |
                              |                +----+----+   +----+----+
                              +---------------------+-------------+
                    global-phase (top ~100 merged, ONNX, optional GPU)
```

- Retrieval. Top-k operators (`weakAnd`/WAND, nearest-neighbor) are sublinear. They expose
  only the top-k hits to ranking, and several retrievers can be combined in one query [13].
- First phase runs on every retrieved hit on each content node and must be cheap.
  `rank-score-drop-limit` can drop hits early [13].
- Second phase re-ranks, by default, the 100 best first-phase hits per content node.
  `total-rerank-count` caps the total across nodes. The documentation example uses
  `bm25 + freshness` for the first phase and an XGBoost model for the second [13].
- Global phase runs in the container on the merged global top hits (default 100,
  `rerank-count`), is "optimized for inference with ONNX models", and can use a GPU via
  `gpu-device`. It supports cross-hit normalization (`normalize_linear`, `reciprocal_rank`,
  `reciprocal_rank_fusion` with k = 60 by default) [13].
- Placement advice. Large GBDTs (XGBoost/LightGBM) and expressions that need lots of vector
  data should stay in the second phase on content nodes, because "moving vector data across
  the network is expensive". Expensive functions computed on content nodes can be passed up
  via `match-features`, so the global phase reuses them instead of recomputing [13].
- Observability. Mutable attributes can count how often each document survives each phase
  (`on-match`, `on-first-phase`, `on-second-phase`, `on-summary`). This makes cascade recall
  measurable per document [13].

Vespa's sample apps include an album-recommendation intro and a news recommendation tutorial
that uses ANN search in a shared user/item embedding space for retrieval [14]. That is the
canonical two-tower-in-a-search-engine pattern.

### 10.5.2 Elasticsearch / OpenSearch with Learning to Rank

The OpenSource Connections LTR plugin stores features as query templates, logs feature values
for training, and stores linear, XGBoost or RankLib models that rank search results [15]. In
recommender settings this works as a rescoring window. The engine's own scoring retrieves
candidates, and the LTR model re-scores the top window, a two-stage cascade inside the
engine. Its limits compared with Vespa are coarser control over per-shard phases and no
native tensor/ONNX global phase. Teams often add an external re-ranking service for deep
models.

### 10.5.3 Earlybird as a search-engine recommender

Twitter's Earlybird (Section 10.2) belongs here too. It is an inverted-index search system
whose main recsys role is UserID-based retrieval (tweets from accounts you follow). It is
ranked in-index by a light model, and features stay fresh through a separate Feature Update
Service [5][6]. Twitter reported reducing search indexing latency to about one second (see
the blog cited in [6]). Freshness of the index directly bounds the freshness of
recommendations.

---

## 10.6 Generative recommenders at serving time: HSTU and M-FALCON

Meta's Generative Recommenders work (ICML 2024) restates ranking and retrieval as
sequential transduction over long user action histories, with HSTU as the encoder. Serving
cost is the obvious objection, since attention over thousands of history events per
candidate is expensive. The paper's answer is M-FALCON (Microbatched-Fast Attention
Leveraging Cacheable OperatioNs) [21]:

```
 naive: for each candidate c_j:  encode([history ; c_j])     cost ~ m * O(n^2)

 M-FALCON:
   encode history once ---------------> cached keys/values (n tokens)
   split m candidates into microbatches of size b
   for each microbatch: attend [b candidates] -> cached history,
                        candidates masked from each other
                                                              cost ~ O(n^2) + m*O(n)
```

- Candidates in a microbatch attend to the shared history but not to each other. The xAI
  Phoenix "candidate isolation" mask is the same contract (Section 10.2.3) [12].
- The repository describes HSTU and M-FALCON as accelerating training and inference "by
  10x-1000x" and reports deployment in "billion-user scale recommendation systems" [21].
- The same repository includes a DLRM-v3 (HSTU-based) inference benchmark wired to MLPerf
  LoadGen. It reports H100 forward latencies of roughly 5-9 ms for 8-12-layer stacks at
  sequence lengths 512-1536 in BF16, and roughly 5-8 ms in FP8 [21]. The README notes that
  cached incremental decoding was "not included yet" in the open-source release [21].

Design takeaway: once a ranker has a large shared user-side computation, the right serving
unit is (user, many candidates), not (user, candidate). Caching user-side state across
requests (for example a KV cache updated incrementally as new actions arrive) is the next
step.

---

## 10.7 Embedding serving and caching

### 10.7.1 The embedding problem

DLRM-style models (DLRM, arXiv 1906.03109 [23]) put most parameters in sparse embedding
tables. ByteDance's Monolith argues for collisionless embedding tables and real-time training,
so that tables grow and change continuously [24]. Serving therefore has to handle tables that
(a) exceed GPU memory, (b) are updated at minute granularity, and (c) are accessed with a
heavily skewed (power-law) distribution. Skew is what makes caching work.

### 10.7.2 NVIDIA Merlin HugeCTR Hierarchical Parameter Server

HugeCTR's Hierarchical Parameter Server (HPS) was "a native C++ library that provides caching
and hierarchical storage for embeddings". It combined a GPU embedding cache with a database
backend, for embeddings "that exceed the capacity of GPU and host memory". It shipped as a
TensorFlow plugin and as a Triton backend [19]. NVIDIA also blogged about scaling inference with it [20]. It was published in RecSys 2022 ("A GPU-specialized
Inference Parameter Server for Large-Scale Deep Recommendation Models") [19]:

```
  request keys
      |
      v
  +-------------------+   hit (hot IDs, majority of lookups under skew)
  | GPU embedding     |------------------------------------------> GPU dense model
  | cache (HBM)       |
  +---------+---------+
            | miss
            v
  +-------------------+   CPU-memory tier (e.g. in-process hash map / Redis cluster)
  | volatile DB tier  |
  +---------+---------+
            | miss
            v
  +-------------------+   SSD-backed persistent tier (e.g. RocksDB), full table
  | persistent tier   |
  +-------------------+
      ^
      | streaming updates from training (online model refresh)
```

Note that HPS is marked deprecated since HugeCTR v25.03 [19]. The hierarchy itself (HBM
cache, then DRAM, then SSD, with streaming updates) remains the standard design.

### 10.7.3 Other caching layers in production recsys

| Cache | What is cached | Invalidation | Example |
|---|---|---|---|
| Result / candidate cache | candidate lists per source per user | TTL, session | CR-Mixer caches candidate generation results [7] |
| Served/seen cache | IDs already shown | session, rolling window | X `PreviouslyServedPostsFilter`, post cache refresh [11] |
| Item embedding index | precomputed item tower outputs | per checkpoint | Phoenix index baked into checkpoint [12] |
| Embedding cache | hot embedding rows | streaming updates | HPS GPU cache [19] |
| Score cache | (user, item) scores | model version, TTL | enabled by batch-invariant scoring [11][12] |
| User-state cache | user tower / sequence KV | new user action | M-FALCON-style history encoding [21] |
| Feature cache | user/item features | TTL / write-through | feature stores (Ch. 8) [25] |

Caching is only safe when the model's output is a function of a stable key. That is why xAI
treats candidate isolation as a serving invariant, not a modeling detail: listwise models
whose scores depend on the other candidates cannot reuse cached per-item scores.

---

## 10.8 Feature fetching fan-out

Feature hydration is often the latency long pole. X's Home Mixer hydrates about 6,000
features per request from many services (user-signal service, graph-feature-service,
real-graph, representation-manager, engagement counters) [1][2]. Earlybird co-locates
features with the index and updates them through a separate stream [5][6].

Common patterns:

- Co-locate or precompute. Push static and slow-moving item features into the index or the
  model server, as Earlybird and Vespa do [5][13]. Hash-based embeddings remove the ID-to-row
  dictionary lookup service entirely [11].
- Batch-get by entity. Fetch features per entity type in bulk (one call per store for all
  candidates), not per candidate.
- Online feature stores. Feast describes an offline store for training, "a low-latency
  online store" for real-time prediction, and a feature server, with Redis, DynamoDB,
  Bigtable and Dragonfly among online backends [25]. DoorDash's 2020 post on building a
  "gigascale" feature store on Redis is a widely cited case study [38]. It covers Redis
  hashes, hashed feature names and compressed embedding values to cut memory and
  latency (details approximate here; see the original). Lyft and Etsy describe similar
  feature-serving and recommendation-serving platforms [36][37], and Eugene Yan's applied-ml
  index collects many more such posts [29].
- Tail-latency control. With fan-out across F services, request p99 approaches the max of F
  service latencies. Hedged requests, per-dependency timeouts with default feature values,
  and degrade-to-cached-features keep the tail bounded. Navi's batch-drop threshold applies
  the same thinking at the model server [4].
- Train-serve consistency. Feature logging at serving time (log what you served, train on
  it) avoids skew. The Elasticsearch LTR plugin's feature-logging mode is a small example
  [15].

---

## 10.9 Making models cheaper: cascades, early exit, distillation, quantization

### 10.9.1 Cascades and pre-ranking

- Rule-of-thumb cascade. A cheap model on everything, then an expensive model on the top
  slice: Vespa first/second/global phases [13], the Earlybird LR light ranker before the
  heavy ranker [5], the X pushservice light ranker that "bridges candidate generation and
  heavy ranking by pre-selecting highly-relevant candidates from the initial huge candidate
  pool" [1].
- Pre-ranking with compute awareness. Alibaba's COLD (Computing power cost-aware Online
  and Lightweight Deep pre-ranking system, 2020) argued that pre-ranking can be a real deep
  model with cross features if its cost is co-optimized with the serving system. Feature
  selection should account for compute cost, and engineering should cover low precision and
  parallelism [34].
- Stage alignment. A cascade stage should optimize agreement with the downstream stage,
  not only the click label. Otherwise the early stage discards items the late ranker would
  have liked. Vespa's per-document phase-survival counters are a practical way to measure
  this [13].

### 10.9.2 Early exit

Early exit means stopping computation for items or requests once the answer is clear. In
cascades it shows up as:

- score thresholds that drop hits after the first phase (`rank-score-drop-limit` in Vespa
  [13]);
- top-k operators (WAND, ANN) that never fully score most documents [13];
- adaptive rerank counts per query (`ranking.globalPhase.rerankCount` can be set per query
  [13]) or per load level, which is graceful degradation under stress;
- load shedding at the server (Navi `batch_drop_millis` [4]).

### 10.9.3 Distillation

- Ranking distillation. Tang and Wang (KDD 2018) train a small student ranker on the
  teacher's top-K ranking as extra supervision, and report students at a fraction of the
  teacher's size with comparable ranking quality [35].
- Cross-stage distillation. Instagram Explore's first-stage ranker is trained to reproduce
  late-stage outputs [31]. That is distillation used to align a cascade, not only to
  compress it.
- Shared trunk. xAI's Phoenix retrieval user tower "uses the same transformer trunk and
  input machinery as the ranking model". The two models "differ in their heads" [12]. This
  shares representation learning across stages while keeping the retrieval-time item side
  precomputable.

### 10.9.4 Quantization and low precision

- Embedding tables: row-wise INT8/INT4 quantization (FBGEMM) cuts memory and bandwidth
  for the gather-heavy part of the model [18].
- Dense compute: FP8 row-wise GEMMs on H100 (HSTU-Ultra) give 1.13-1.25x forward speedups
  [21]. BF16 is the common baseline.
- ANN indexes: compressed codes (PQ and variants) let FAISS scale to "billions of vectors
  in main memory on a single server", with GPU implementations [26]. ScaNN uses anisotropic
  vector quantization for maximum inner product search [27]. Semantic IDs (residual
  quantization) are now also used as model inputs [12].

---

## 10.10 Comparison of serving designs

| System | Retrieval serving | Ranking compute | Model server / runtime | Notable serving trick |
|---|---|---|---|---|
| X (2023) | Earlybird index, GraphJet UTEG, CR-Mixer | parallel MaskNet MTL | Navi (Rust; TF/ONNX) | LR light ranker in index; ~6k features; batch-drop |
| X/xAI (2026) | Thunder in-memory, Phoenix two-tower, SimClusters | Phoenix transformer | Rust gRPC engine, JAX-trained | candidate isolation; hash embeddings; index in checkpoint |
| Instagram Explore | two-tower + ANN, multiple sources | 2-stage: light then MTML | Meta internal | distilled first stage; value model |
| Meta Andromeda (ads) | deep neural retrieval + hierarchical index | n/a (retrieval stage) | Grace Hopper, MTIA | model-index-hardware co-design |
| Meta GR (HSTU) | generative retrieval/ranking | HSTU transformer | GPU, Triton/CUDA kernels | M-FALCON microbatching + caching; FP8 |
| Pinterest | (various) | DNN rankers on GPU | GPU serving | batching, CUDA graphs, fused embeddings |
| Vespa apps | WAND / ANN in content nodes | 1st, 2nd phase on nodes; global on container | Vespa + ONNX (opt. GPU) | compute near data; rerank-count knobs |
| Elasticsearch LTR | BM25 / kNN | LTR rescoring window | ES plugin (linear/XGBoost) | in-engine 2-stage cascade |
| NVIDIA Merlin | ANN (FAISS) | DLRM-style | Triton + HPS backend | GPU embedding cache hierarchy |

---

## 10.11 Key takeaways for system designers

1. Budget per stage, not per system. Write down p99 budgets per stage (Section 10.1) and
   enforce them with per-branch timeouts and fallbacks. X keeps a reverse-chron backup
   pipeline for when the scored pipeline fails [2].
2. The serving unit is (user, many candidates). Compute user-side work once per request:
   two-tower precomputation, M-FALCON microbatching, Phoenix candidate isolation
   [12][21][31].
3. Make scores batch-invariant if you want to cache them. Listwise cross-candidate
   attention costs you score caching and reproducibility [11][12].
4. Push cheap compute to the data. Index-side light rankers (Earlybird), Vespa first and
   second phases on content nodes, and `match-features` to avoid moving vectors over the
   network [5][13].
5. Batch with a bounded delay and shed load. Dynamic batching (Triton
   `max_queue_delay_microseconds`, Navi 2 ms / 16-request defaults) plus explicit drop
   thresholds [4][16].
6. Embeddings are a memory-hierarchy problem. GPU cache, then DRAM, then SSD, with
   streaming updates. Skew makes small caches effective [19][24].
7. Feature hydration is usually the long pole. Co-locate, batch-get, use defaults on
   timeout, and log served features for training [2][5][25].
8. Cascades need alignment. Train early stages to agree with late stages (distillation)
   and measure per-stage survival, or the cascade silently caps quality [13][31][35].
9. Spend the precision budget wisely. Low precision first for embeddings (INT8/INT4) and
   ANN (PQ), then FP8 for dense GEMMs on modern GPUs [18][21][26].
10. Co-design moves the frontier. Andromeda (hardware + index + model) and Pinterest GPU
    serving show that bigger models upstream can be cheaper than more stages
    downstream [30][32].
11. Keep a separate, explicit value model. Combining multi-task predictions with
    configurable weights lets product trade-offs ship without retraining [9][11][31].
12. Simpler servers win on the hot path. Navi's minimalist Rust design and TorchRec's
    TorchScript C++ path show that the critical path benefits from a small,
    purpose-built runtime [3][17].

---

## References

1. "X's Recommendation Algorithm" (repository README), X Corp / Twitter, 2023.
   https://github.com/twitter/the-algorithm
2. "Home Mixer" README, Twitter, 2023.
   https://github.com/twitter/the-algorithm/blob/main/home-mixer/README.md
3. "Navi: High-Performance Machine Learning Serving Server in Rust", Twitter, 2023.
   https://github.com/twitter/the-algorithm/blob/main/navi/README.md
4. Navi `cli_args.rs` (batching defaults), Twitter, 2023.
   https://github.com/twitter/the-algorithm/blob/main/navi/navi/src/cli_args.rs
5. "Earlybird Light Ranker" README, Twitter, 2023.
   https://github.com/twitter/the-algorithm/blob/main/src/python/twitter/deepbird/projects/timelines/scripts/models/earlybird/README.md
6. "Tweet Search System (Earlybird)" README, Twitter, 2023 (cites "Earlybird: Real-Time
   Search at Twitter", http://notes.stephenholiday.com/Earlybird.pdf, and "Reducing search
   indexing latency to one second", https://blog.twitter.com/engineering/en_us/topics/infrastructure/2020/reducing-search-indexing-latency-to-one-second).
   https://github.com/twitter/the-algorithm/blob/main/src/java/com/twitter/search/README.md
7. "CR-Mixer" README, Twitter, 2023.
   https://github.com/twitter/the-algorithm/blob/main/cr-mixer/README.md
8. "Product Mixer" README, Twitter, 2023.
   https://github.com/twitter/the-algorithm/blob/main/product-mixer/README.md
9. "Heavy Ranker" (Recap) README, Twitter, 2023.
   https://github.com/twitter/the-algorithm-ml/blob/main/projects/home/recap/README.md
10. "Twitter's Recommendation Algorithm", Twitter Engineering Blog, 2023.
    https://blog.x.com/engineering/en_us/topics/open-source/2023/twitter-recommendation-algorithm
11. "X For You Feed Algorithm" (x-algorithm README), xAI / X, 2025-2026.
    https://github.com/xai-org/x-algorithm
12. "Phoenix" README (retrieval and ranking models, Rust serving engine), xAI, 2026.
    https://github.com/xai-org/x-algorithm/blob/main/phoenix/README.md
13. "Phased Ranking", Vespa.ai documentation, 2025-2026.
    https://docs.vespa.ai/en/ranking/phased-ranking.html
    (source: https://github.com/vespa-engine/documentation/blob/master/en/ranking/phased-ranking.html)
14. "Vespa sample applications" README, Vespa.ai.
    https://github.com/vespa-engine/sample-apps
15. "Elasticsearch Learning to Rank" plugin README, OpenSource Connections.
    https://github.com/o19s/elasticsearch-learning-to-rank
16. "Batchers" (dynamic batching), Triton Inference Server user guide, NVIDIA.
    https://github.com/triton-inference-server/server/blob/main/docs/user_guide/batcher.md
17. "TorchRec Inference Library", PyTorch / Meta.
    https://github.com/pytorch/torchrec/blob/main/torchrec/inference/README.md
18. "The FBGEMM Project", PyTorch / Meta.
    https://github.com/pytorch/FBGEMM
19. "Merlin HugeCTR" README and "Hierarchical Parameter Server" docs, NVIDIA; Wei et al.,
    "A GPU-specialized Inference Parameter Server for Large-Scale Deep Recommendation
    Models", RecSys 2022.
    https://github.com/NVIDIA-Merlin/HugeCTR ;
    https://github.com/NVIDIA-Merlin/HugeCTR/blob/main/docs/source/hierarchical_parameter_server/index.md ;
    https://dl.acm.org/doi/10.1145/3523227.3546765
20. Verma et al., "Scaling Recommendation System Inference with Merlin Hierarchical Parameter
    Server", NVIDIA Developer Blog, 2022.
    https://developer.nvidia.com/zh-cn/blog/scaling-recommendation-system-inference-with-merlin-hierarchical-parameter-server/
21. Zhai et al., "Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers
    for Generative Recommendations", ICML 2024, and the generative-recommenders repository
    (HSTU, M-FALCON, DLRM-v3 inference, FP8 benchmarks), Meta.
    https://proceedings.mlr.press/v235/zhai24a.html ;
    https://github.com/facebookresearch/generative-recommenders
22. Gupta et al., "DeepRecSys: A System for Optimizing End-To-End At-scale Neural
    Recommendation Inference", ISCA 2020, Harvard / Facebook.
    http://vlsiarch.eecs.harvard.edu/wp-content/uploads/2020/05/DeepRecSys_Gupta_ISCA2020.pdf ;
    https://github.com/harvard-acc/DeepRecSys
23. Naumov et al., "Deep Learning Recommendation Model for Personalization and
    Recommendation Systems", Facebook, 2019. https://arxiv.org/abs/1906.03109
24. Liu et al., "Monolith: Real Time Recommendation System With Collisionless Embedding
    Table", ByteDance, 2022. https://arxiv.org/abs/2209.07663 ;
    https://github.com/bytedance/monolith
25. "Feast: Feature Store for Machine Learning" README, Feast project.
    https://github.com/feast-dev/feast
26. Johnson, Douze, Jegou, "Billion-scale similarity search with GPUs", 2017, and the
    FAISS README, Meta FAIR. https://arxiv.org/abs/1702.08734 ;
    https://github.com/facebookresearch/faiss
27. Guo et al., "Accelerating Large-Scale Inference with Anisotropic Vector Quantization"
    (ScaNN), Google, ICML 2020.
    https://github.com/google-research/google-research/tree/master/scann
28. Wang, She, Zhang, "MaskNet: Introducing Feature-Wise Multiplication to CTR Ranking
    Models by Instance-Guided Mask", 2021. https://arxiv.org/abs/2102.07619
29. Eugene Yan, "applied-ml: Papers & tech blogs by companies sharing their work on data
    science & machine learning in production" (curated index used for source discovery).
    https://github.com/eugeneyan/applied-ml
30. Pinterest Engineering, "GPU-accelerated ML Inference at Pinterest", 2022.
    https://medium.com/@Pinterest_Engineering/gpu-accelerated-ml-inference-at-pinterest-ad1b6a03a16d
31. Meta Engineering, "Scaling the Instagram Explore recommendations system", 2023.
    https://engineering.fb.com/2023/08/09/ml-applications/scaling-instagram-explore-recommendations-system/
32. Meta Engineering, "Meta Andromeda: Supercharging Advantage+ automation with the next-gen
    personalized ads retrieval engine", 2024.
    https://engineering.fb.com/2024/12/02/production-engineering/meta-andromeda-advantage-automation-next-gen-personalized-ads-retrieval-engine/
33. Covington, Adams, Sargin, "Deep Neural Networks for YouTube Recommendations", Google,
    RecSys 2016 (canonical two-stage candidate-generation/ranking serving design).
    https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf
34. Wang et al., "COLD: Towards the Next Generation of Pre-Ranking System", Alibaba, 2020.
    https://arxiv.org/abs/2007.16122
35. Tang, Wang, "Ranking Distillation: Learning Compact Ranking Models With High
    Performance for Recommender System", KDD 2018. https://arxiv.org/abs/1809.07428
36. Lyft Engineering, "ML Feature Serving Infrastructure at Lyft", 2021 (further reading).
    https://eng.lyft.com/ml-feature-serving-infrastructure-at-lyft-d30bf2d3c32a
37. Etsy Code as Craft, "Building a Platform for Serving Recommendations at Etsy", 2022
    (further reading).
    https://www.etsy.com/codeascraft/building-a-platform-for-serving-recommendations-at-etsy
38. DoorDash Engineering, "Building a Gigascale ML Feature Store with Redis", 2020.
    https://doordash.engineering/2020/11/19/building-a-gigascale-ml-feature-store-with-redis/
