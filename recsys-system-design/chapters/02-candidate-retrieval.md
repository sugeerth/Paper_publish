# 02. Candidate Generation and Retrieval: Two-Tower Models, Negative Sampling, ANN Indexes and Multi-Source Retrieval

## Overview

Every large recommender is a funnel. A corpus of millions to billions of items (videos, Pins, posts, ads, listings, stores, users) must be narrowed to a few hundred or a few thousand candidates within a budget of tens of milliseconds. Only then can a heavy ranking model with rich cross features score them. The first stage of that funnel is called candidate generation, retrieval, recall or matching. It is the subject of this chapter.

Retrieval has three properties that set it apart from ranking:

1. **It must be sub-linear in corpus size.** You cannot score a billion items per request with a deep model. Most production designs therefore factor the score into a user (query) vector and an item vector, precompute and index the item side, and use approximate nearest neighbor (ANN) search. Alternatives replace the vector index with a learned tree or path structure (TDM, Deep Retrieval, Andromeda) or with graph walks (Pixie, GraphJet).
2. **It optimizes recall, not precision.** Its job is to make sure good items reach the ranker. The ranker decides their order.
3. **It is almost always multi-source.** Production systems run a portfolio of retrievers, such as learned embeddings, item-to-item co-occurrence, graph walks, social-graph expansion and heuristic "trending" sources. They merge the outputs with quotas, weights or round-robin interleaving.

The chapter has six parts:

- the foundational modeling work: the YouTube DNN, sampled softmax, in-batch negatives and the logQ correction;
- company-by-company production designs: Facebook Search, Instagram, Meta Ads Andromeda, Pinterest, LinkedIn, Twitter/X, Snap, Uber, Etsy, Airbnb and Alibaba/ByteDance;
- the ANN index families: IVF-PQ in FAISS, ScaNN, HNSW and DiskANN;
- multi-source merging, graph retrievers and heuristic retrievers;
- a comparison table;
- takeaways for system designers.

```
                    THE CANONICAL RETRIEVAL FUNNEL
 corpus: 1e6 .. 1e10 items
   |
   |  [Retrieval / candidate generation]   many sources in parallel
   |    - two-tower EBR (ANN over item embeddings)
   |    - item-to-item CF / co-engagement tables
   |    - graph walks (Pixie, GraphJet/UTEG, PPR)
   |    - social graph (friends-of-friends, in-network)
   |    - heuristic (trending, fresh, location)
   v
 ~1e3 .. 1e4 candidates (dedup + merge + filter)
   |
   |  [Pre-rank / lightweight scoring]  often a two-tower too
   v
 ~1e2 .. 1e3
   |
   |  [Heavy ranking]  multi-task DNN with cross features
   v
 ~1e1 .. 1e2 shown, after re-ranking for diversity, policy, ads
```

---

## 1. Foundations: from extreme multiclass to two towers

### 1.1 YouTube DNN candidate generation (Google, 2016)

**Problem.** Recommend videos from a corpus of millions to a very large user base. The model must narrow those millions down to a few hundred candidates per user [1].

**Architecture.** Covington, Adams and Sargin split the system into two neural networks, following the classic information-retrieval two-stage dichotomy. A candidate-generation network reads the user's activity history and context and produces a user vector. A separate ranking network then scores the few hundred survivors [1]. Candidate generation is framed as **extreme multiclass classification**: predict which video, out of millions of classes, the user will watch next.

**Training trick.** A full softmax over millions of classes is too expensive. The model is trained with **sampled softmax**, which draws a few thousand negatives per example and corrects for the sampling distribution. Secondary summaries of the paper report a speedup of more than 100x over the full softmax [1].

**Serving.** The learned output-layer video vectors become the item embeddings. At serving time the softmax is replaced by a nearest-neighbor search over those vectors in dot-product space [1]. This "train as a classifier, serve as a nearest-neighbor lookup" pattern is the template for almost everything that follows.

```
  YouTube DNN candidate generation (2016)
  user history + context features --> MLP (ReLU stack) --> user vector u
                                                              |
  training:  softmax over all videos (approximated by sampled softmax)
  serving :  top-N by dot(u, v_j) via nearest-neighbor index over v_j
```

**Lesson.** Phrase retrieval as next-item classification and keep the scoring function a dot product. That turns serving into maximum-inner-product search (MIPS), which can be solved approximately at very large scale.

### 1.2 The two-tower model with in-batch negatives and logQ correction (YouTube, 2019)

**Problem.** The 2016 output layer had one learned vector per video ID. That makes cold-start and content features awkward. Google's RecSys 2019 paper "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations" (Yi et al.) moved to a **two-tower** design. A query tower encodes user and context features, and a candidate tower encodes item features. Both produce embeddings that are compared by inner product [2].

**In-batch negatives.** Instead of sampling negatives from a fixed vocabulary, every other positive item in the mini-batch serves as a negative for each example. This is cheap, because the item embeddings are already computed. It is also streaming-friendly, because no global vocabulary is needed. It has a bias, though. An item's chance of appearing as a negative is proportional to its frequency in the training stream. Popular items get over-penalized, and under a power-law item distribution this "can severely restrict model performance" [2].

**logQ correction.** The fix subtracts log(p_j) from each logit s(x, y_j), where p_j is the estimated probability that item j appears in a batch. The corrected softmax then approximates the full softmax [2][5]. The paper's other contribution is a **streaming frequency estimator** that needs no fixed vocabulary. A secondary write-up describes it as two hashed arrays. One array stores the last global step at which each item was seen. The other stores a running average of the gap between occurrences, updated like SGD with a fixed learning rate. Both arrays live on parameter servers during distributed training [2]. The estimated probability is roughly 1 / (average gap).

**Deployment.** The model was deployed as "Neural Deep Retrieval" (NDR) for YouTube, retrieving personalized suggestions from a corpus of tens of millions of videos. Live A/B tests showed improved recommendation quality [2].

```
  Two-tower training with in-batch softmax + logQ   (batch of B pairs)

  query tower  q_i = f(user_i, context_i)          i = 1..B
  item tower   v_j = g(item_j features)            j = 1..B
  logits       s_ij = <q_i, v_j> / temperature  -  log p_j     (logQ)
  loss         -sum_i log softmax_j(s_ij)[j = i]
  p_j          streaming estimate from hashed "last-seen step" arrays
```

### 1.3 Mixed Negative Sampling (Google, 2020)

In-batch negatives have a second problem: **selection bias**. Items that never received engagement never show up as positives, so they are never seen as negatives either. A fresh or long-tail item can therefore drift into a region of embedding space where it scores high for everyone. Yang et al.'s Mixed Negative Sampling (MNS) adds a second stream of negatives sampled **uniformly from the whole corpus** (an "index data stream"). These are shared by every batch and concatenated with the in-batch negatives. The method was evaluated on Google Play app recommendations and outperformed batch-only sampling [3]. A later paper that compared against MNS used 128 in-batch negatives plus a large shared set of uniform negatives [3]. That gives a sense of the typical ratio.

### 1.4 Cross-batch negatives and memory banks (Huawei, 2021)

The quality of in-batch sampling depends on batch size, and very large batches are expensive. Cross-Batch Negative Sampling (CBNS, SIGIR 2021) keeps a FIFO memory bank of item embeddings from recent mini-batches and reuses them as extra negatives [4]. The justification is empirical. After warm-up, encoders produce fairly stable embeddings for the same input, so stale embeddings remain useful negatives. The same "slow drift" observation underlies Cross-Batch Memory in metric learning [4]. The paper reports many more negatives at a training cost close to plain in-batch sampling [4].

### 1.5 Revisiting the logQ correction (Yandex, RecSys 2025)

Khrylchenko et al. point out that the standard logQ derivation assumes positives are drawn from the sampling distribution. In practice the positive item is in its own batch with probability 1 [5]. Their corrected formulation does two things:

- It applies importance weighting only to the negatives and removes the positive from the corrected denominator.
- It adds a confidence-based weight that shrinks the loss when the model already scores the positive highly.

The authors report consistent gains over the standard logQ correction on public and proprietary datasets [5]. The practical takeaway is that sampled-softmax retrieval training is still being refined. Small correctness details in the loss can move recall.

### 1.6 Facebook Search EBR: triplet loss, hard negatives and ANN tuning (KDD 2020)

**Problem.** Facebook Search historically relied on Boolean term matching in **Unicorn**, the inverted-index retrieval engine behind most Facebook search products. Social search also needs personalization from the searcher's own graph, which text matching alone cannot provide [6].

**Architecture.** A **unified embedding** two-tower model encodes the query side (query text plus searcher context, such as location and social connections) and the document side (text plus document context). Each document's vector is quantized with FAISS and stored inside the existing inverted index:

- the coarse cluster ID becomes a **term**;
- the product-quantized residual becomes the **payload**.

This lets a single Unicorn query mix Boolean constraints with nearest-neighbor matching [6].

```
  Facebook EBR inside Unicorn (2020)

  query + searcher context --> query tower --> q (e.g. 128-d)
                                                   |
            Unicorn query:  (nn  q :radius ...)  AND  (term filters)
                                                   |
  inverted index:  term = coarse cluster id (IVF centroid)
                   payload = PQ code of residual
                                                   |
             candidates --> later-stage ranking (embedding as a feature)
```

**Loss and sampling details:**

- *Triplet loss with a margin.* The best margin varied across tasks, and different margins moved KNN recall by 5-10% [6].
- *Number of negatives.* The authors argue that sampling n random negatives per positive approximately optimizes recall at the top N/n of a serving pool of size N [6].
- *Impressed-but-not-clicked items are bad negatives.* Using non-click impressions as negatives caused about a 55% absolute recall regression for the people embedding model compared with random negatives. Those negatives are too hard and not representative of the retrieval task [6].
- *Online hard negative mining.* Within the batch, the other positives with the highest similarity to the query are used as hard negatives. This gave +8.38% recall for people search, +7% for groups and +5.33% for events [6]. A reading note also describes an offline variant that takes documents ranked roughly 101-500 by the current model as hard negatives (approximate; from a secondary source).
- *Unified vs. text embedding.* Unified embeddings beat text-only embeddings by +18% recall for events search and +16% for groups search [6].

**ANN tuning lessons:**

- The authors measure 1-Recall@10, the fraction of exact-KNN top-1 results that appear in the ANN top-10.
- They suggest sizing PQ bytes at about d/4, with diminishing returns beyond that.
- In one 128-d experiment, PCA+PQ16 reached 51.62% 1-Recall@10 versus 74.81% for the flat index at a similar fraction of the index scanned. Quantization is not free.
- The number of coarse clusters and nprobe trade off performance against recall.
- Re-tune ANN parameters whenever the model changes significantly.
- Use an offline tuning pipeline to shortlist settings, then pick the final configuration with online experiments [6].

---

## 2. Meta: Instagram Explore and Andromeda

### 2.1 Instagram Explore retrieval (2023)

Explore is a four-stage funnel: **retrieval, first-stage ranking, second-stage ranking and final re-ranking** [7]. Retrieval tries to predict what the later stages would rank highly, starting from the general media distribution. Meta's design relies on **caching and precomputation** combined with **Two Towers networks**. The post's figure shows two-tower inference with caching on both the user side and the item side [7].

**Sources.** Explore runs many retrieval sources at once. They include heuristics, such as trending posts, and ML strategies that capture both real-time and pre-generated interests. Sources are mixed with tunable weights, so different user groups get different blends [7]. Meta's transparency documentation names concrete source types [8]:

- author-based sources: media from accounts you engaged with;
- media-based sources: media similar to media you engaged with;
- item collaborative filtering;
- personalized PageRank;
- two-tower sparse-network sourcing.

Each source selects hundreds of items from a pool of billions [8]. Meta notes that an earlier generation used Word2Vec-style ID embeddings for users, media and authors.

**Counts.** Up to **1,500** media survive retrieval. A two-tower early-stage ranker then passes the top **100** to the next stage [8]. The early-stage two-tower model scores user features against media features. Its objective is reportedly trained to predict how the second-stage ranker would rank the item. That is a distillation-style alignment between stages, reported by a secondary source [7].

```
  Instagram Explore (2023, per Meta)
   sources (weighted mix): heuristic | item-CF | PPR | two-tower | ...
        |  hundreds each, from billions
        v
   retrieval output: <= ~1500 media
        |  first-stage ranker = two-tower (cached user + item embeddings)
        v
   top 100 --> second-stage ranker (heavy MTML) --> final re-rank
```

### 2.2 Meta Andromeda: a learned hierarchical index for ads (2024)

**Problem.** Generative-AI ad tools sharply increased the number of eligible ad creatives. Meta said more than a million advertisers created more than 15 million ads in one month with its GenAI tools. Two-tower plus ANN retrieval caps how much user-ad interaction a model can express [9].

**Architecture.** Andromeda is a personalized ads retrieval engine. It is a highly customized deep neural network with **sub-linear inference cost** that runs with heavy parallelism on NVIDIA Grace Hopper Superchips [9]. Ads are organized into a **hierarchical index with multiple layers**, so inference only expands the most relevant nodes. The **index and the retrieval model are trained jointly**, which Meta says gives better precision and recall than two-tower or ANN approaches [9].

**Results.** Deployed on Facebook and Instagram, Andromeda gave **+6% recall** to the retrieval system and **+8% ads quality** on selected segments [9]. Third-party explainers cite a "10,000x" increase in model capacity and a narrowing from tens of millions of ads to a few thousand. These figures could not be verified against the excerpts of Meta's post and should be treated as approximate.

**Lesson.** At the top end of scale the industry is moving beyond dot-product-only retrieval. The options are learned trees or hierarchies (Andromeda, TDM, Deep Retrieval) and GPU brute force with richer scoring (LiNR). In both cases the retrieval model can capture interactions that a factorized dot product cannot.

---

## 3. Pinterest: learned retrieval, multi-embedding, graph retrievers and Manas

Pinterest has one of the best-documented retrieval stacks. It has graph retrievers (Pixie), graph neural network embeddings (PinSage), learned two-tower retrieval and its own ANN platform (Manas).

### 3.1 Pixie: real-time random walks (2017-2018)

**Problem.** Recommend from 3+ billion items to 200+ million users in real time [18].

**Design.** Pixie loads a pruned bipartite **Pin-board graph** into the RAM of a single machine. The graph has 3 billion nodes and 17 billion edges and takes about 120 GB of memory. The unpruned graph had 7 billion nodes and more than 100 billion edges. Every walk therefore stays on one machine, and the service scales by replicating machines [18]. A query is a set of Pins with weights. Pixie runs many short random walks with restarts from those Pins, counts visits, and returns the most-visited Pins. The walk is biased toward the user's interests and boosts smaller boards so that high-degree nodes do not dominate [18].

**Scale.**

- The deployment used AWS r3.8xlarge machines with 244 GB of RAM.
- 99th-percentile latency was under 60 ms.
- One server handled about 1,200 requests per second.
- The cluster served about 100,000 requests per second.
- Pixie gave up to 50% higher user engagement than the previous Hadoop-based system [18].

```
  Pixie (graph retriever)
   query pins (weighted) --> N short random walks w/ restart on Pin<->Board graph
                              (single-machine RAM, ~120 GB pruned graph)
                         --> visit counts (early stop when top-k stabilizes)
                         --> top-k pins as candidates   (p99 < 60 ms)
```

### 3.2 PinSage: GCN embeddings at web scale (2018)

PinSage is a graph convolutional network trained on a graph of **3 billion nodes and 18 billion edges**, using 7.5 billion training examples [17]. Its key tricks are:

- **Random-walk neighborhoods.** Short random walks select the top-T most-visited neighbors instead of using the full k-hop neighborhood. This bounds memory per node.
- **Importance pooling.** Neighbors are weighted by visit counts. The paper credits this with a 46% gain in offline metrics.
- **Curriculum training.** Harder negatives are introduced progressively, for a 12% gain [17].

The resulting Pin embeddings are served with nearest-neighbor search for related-Pin and homefeed candidate generation. This is a good example of graph structure being "compiled" into vectors so that standard ANN serving can be used.

### 3.3 Learned two-tower retrieval for Homefeed

**Problem.** Pinterest's older candidate generators were heuristic and graph-based. A learned retriever lets the model optimize the engagement objective directly [10].

**Architecture.**

- A standard two-tower model with no explicit user-Pin interaction features.
- The **Pin tower** encodes visual, text and statistical features. Its embeddings are precomputed offline and pushed to Pinterest's in-house ANN system, **Manas**.
- The **user tower** encodes engagement sequences and context. It runs once per Homefeed request [10][11].
- Training uses in-batch negatives with a popularity-bias (logQ-style) correction [10]. The logQ detail is reported by a secondary summary.

**Operations.** Each two-tower model is split into **two artifacts deployed to separate services**: the user-embedding service and the index or Pin-embedding pipeline. Auto-retraining must keep the model versions of both services in sync. Otherwise user vectors from model version k get compared with Pin vectors from version k-1, which silently breaks retrieval [10].

**Results.** Secondary summaries say the learned retriever became one of the top-performing candidate generators, with the **highest user coverage** and a top-three save rate. It was deployed for Homefeed and notifications [10].

### 3.4 Advancements: feature crossing, ID embeddings and multi-embedding retrieval

The 2025 "Advancements in EBR at Pinterest Homefeed" post describes several upgrades [11]:

- **Feature crossing inside each tower.** MaskNet and DHEN blocks recover some of the expressiveness lost by the dot-product bottleneck. DHEN reportedly gave +0.1-0.2% engagement and more than 1% more homefeed saves (approximate; from a secondary summary).
- **Pre-trained ID embeddings.** User and Pin ID embeddings are pre-trained contrastively on a cross-surface dataset to reduce overfitting.
- **Multi-embedding retrieval.** Several user embeddings are generated per request. Implicit interests come from a differentiable clustering module, and explicit interests come from conditional retrieval [12].
- **Merging.** Each embedding's ANN results are merged by **round-robin with de-duplication**, which balances interest clusters. In the companion paper's experiments, conditional retrieval with post-filtering performed best. Serving p90 latency rose from 150 ms to 205 ms, which stayed within budget [12].
- **Reported gains.** Engagement rose 0.1% to 1.2% across metrics such as sessions, saves and clicks (approximate; secondary summary) [11].

### 3.5 Conditional learned retrieval

Conditional retrieval feeds an explicit condition, such as a topic or interest, into the user tower. One user can then issue several conditioned queries that still share the same overall behavior representation [13]. Pinterest's paper on bootstrapping conditional retrieval reports that a topic-based notification feed built this way gave **+0.26% weekly active users** and beat two-tower models with filters [19]. A 2026 post on scaling conditional learned retrieval for Home Feed describes redesigning the serving infrastructure to support many conditions per request [13].

### 3.6 Manas: Pinterest's ANN platform

Manas started as an HNSW-based serving system:

- **Real-time HNSW.** Pins can be inserted into the HNSW graph incrementally for real-time EBR. Filtering was initially done by over-fetching K neighbors and then post-filtering [16].
- **Streaming filters.** The inverted index was reused as the filter store. Skiplist posting lists were replaced with bitmap and array-backed posting lists for faster lookups [15].
- **Quantization and SSD serving.** The 2026 "Evolving Pinterest's Embedding Retrieval Platform" post benchmarks quantization on a 100-million-embedding GraphSage dataset [14]:
  - baseline HNSW: 121 GB at 93.72% Recall@100;
  - HNSW with scalar quantization (SQ): 50 GB at 92.92%;
  - baseline IVF: 97 GB at 91.69%;
  - IVF with SQ: 25 GB at 95.71%.

  PQ compressed more but lost much more recall. Manas also implemented **SPANN** for SSD-resident indexes, with an in-memory HNSW over the cluster centroids [14].

**Lesson.** Scalar quantization is often the best first compression step for recommendation embeddings. Production ANN platforms tend to move from all-in-RAM HNSW toward IVF or SPANN-style designs with quantization once cost and update rates matter.

---

## 4. LinkedIn: GPU retrieval and LLM-based dual encoders

### 4.1 LiNR: model-based neural retrieval on GPUs (2024)

**Problem.** LinkedIn needed out-of-network post recommendations for the Feed, meaning posts from people the member does not follow. Standard KNN with post-filtering lost quality when attribute filters, such as language or freshness, removed many results [20].

**Architecture.** LiNR stores the **item vectors and the model weights in the same model binary**, so index construction becomes part of model training and export. It supports a **billion-sized index on GPUs** [20]. Key features:

- **Exhaustive GPU search with attribute-based pre-filtering.** Filters run before scoring, which avoids the recall loss of post-filtering. The authors note that PyTorch and TensorFlow lacked efficient pre-filtering, so they built custom indexing and filtering ops.
- **Multi-embedding retrieval** and cold-start strategies.
- **Quantization for larger indexes.** A secondary review describes 1-bit Sign-OPORP random-projection compression.
- **Near-real-time updates.** Per a secondary summary, a change-data-capture stream updates the GPU-resident index without hurting inference latency [20].

**Results.** About a **3% relative increase in professional daily active users** [20]. The authors present LiNR as a step toward merging retrieval and ranking into one GPU model.

```
  LiNR (GPU "retrieval as a model")
   model binary = { user/query tower weights , item matrix [N x d] , attrs }
   request:  q = tower(member)          (GPU)
             mask = prefilter(attrs)    (GPU, exact)
             scores = q . items^T  over masked rows  (dense matmul / quantized)
             top-k --> ranking
```

A later LinkedIn paper (2026) calls this pattern **"GPU Retrieval as Ranking"**: the entire candidate set sits in GPU memory and every candidate is scored with dense matrix operations. It also mentions **SilverTorch**, which implements ANN search and filtering as tensor operations, using a fused int8 IVF kernel and a GPU bloom-filter index [22].

### 4.2 LLM dual-encoder retrieval for the Feed (2025)

LinkedIn fine-tuned **LLaMA 3**, a decoder-only causal LM, as a dual encoder. Member and post texts are encoded separately and pooled into embeddings [21]. The scale is about **2,000 candidates from a pool of hundreds of millions**, within a latency budget of a few milliseconds, at several thousand queries per second. Two findings stand out:

- Numerical features written into the prompt work better when they are **quantized** into tokens. This improved alignment between retrieval and ranking.
- The largest online gains were for **newer members** with sparse networks. This is the cold-start regime, where content-based LLM embeddings beat ID or graph signals [21].

---

## 5. Twitter/X: a portfolio of graph and embedding retrievers

Twitter's 2023 open-source release documents a canonical multi-source retrieval design. The For You timeline extracts about **1,500 candidate Tweets from a pool of hundreds of millions** per request. On average it is split **50% in-network and 50% out-of-network** [23].

- **In-network.** Tweets from accounts the user follows, ranked by **RealGraph**, a model of the likelihood of engagement between two users. A logistic-regression model is used for relevance [23].
- **Out-of-network:**
  - **GraphJet / UTEG.** An in-memory user-Tweet interaction graph that generates candidates through graph traversals, such as "Tweets engaged with by people you follow" [23][24].
  - **SimClusters.** Sparse, interpretable embeddings over overlapping communities, discovered by a custom matrix-factorization and community-detection step over the producer-producer similarity graph [25]. A secondary source quoting Twitter puts the count at about 145k communities, updated every three weeks (approximate).
  - **TwHIN.** Dense knowledge-graph embeddings for users and Tweets, served with ANN [24].

```
  Twitter "For You" candidate sourcing (2023)
   in-network  : follow graph --> RealGraph-weighted (LR) ---------\
   out-network : UTEG/GraphJet traversal ---------------------------+--> ~1500
                 SimClusters sparse community vectors -------------|     merged
                 TwHIN dense KG embeddings (ANN) ------------------/     to ranker
```

**Lesson.** The interpretable sparse community vectors in SimClusters serve several products at once: Tweets, accounts and Communities [25]. One shared representation layer can feed many retrievers.

---

## 6. Snap: EBR for friend recommendation

Friend recommendation traditionally uses **friends-of-friends (FoF)** graph expansion. That expansion reaches little in cold-start settings and becomes expensive beyond one or two hops [26]. Snap added an **EBR source** built on GNN embeddings for billions of users. It retrieves candidates beyond two hops and also provides a natural way to rank FoF candidates. Online A/B tests showed significant gains in friendships made, in both low-density and high-density network markets [26].

Snap's GiGL paper reports that a **sampled-softmax retrieval loss** suited the friend-recommendation EBR use case. It was combined with feature-reconstruction and whitening-decorrelation losses in multi-task training [28]. A follow-up added **ANN query expansion**, which issues several queries per user. It improved recall and gave an 11% relative gain in friend reciprocated-communication metrics, serving more than 800 million monthly active users [27].

---

## 7. Marketplaces: Uber Eats, Etsy, Airbnb and Amazon

### 7.1 Uber Eats Two-Tower Embeddings (TTE)

**Problem.** The earlier Deep MF model was trained per city for the first-pass ranking stage. It needed thousands of weekly Spark jobs and did not scale as Uber expanded to more cities [29].

**Design.** A single two-tower model. Store embeddings are computed offline and indexed in Uber's search platform (SIA). The eater (query) tower runs in real time per Homefeed request, followed by ANN search over store embeddings. This cuts cost from O(q x M) real-time inference to O(M) offline plus O(q) online, where q is queries and M is stores [29]. A Feature Store Summit talk gives the latency target: retrieve the best stores out of thousands in about **50 ms**. It also notes that about 95% of users start their journey on the Homefeed [30].

### 7.2 Etsy Unified Embedding

Etsy's two-tower model has a **product encoder** and a **joint query-user encoder**, scored with cosine similarity, so products can be indexed offline for ANN [31]. A notable trick is **ANN-based boosting**. Quality signals, such as price or seller quality, are appended to the vectors as extra dimensions, so the inner product adds a tuned quality boost inside the ANN search. The weights are tuned with black-box optimization. Live A/B tests showed **+5.58% search purchase rate** and **+2.63% site-wide conversion rate** [31].

### 7.3 Airbnb

- **2018 listing embeddings.** Airbnb learned 32-dimensional skip-gram listing embeddings for about 4.5 million active listings, from more than 800 million search click sessions. Sessions were split at 30-minute gaps, and the objective added market-level negatives and the booked listing as global context. The embeddings power similar-listing recommendations and real-time personalization features. Similar-listing carousel CTR rose 21% [32] (figures from secondary summaries of the KDD 2018 paper).
- **2025-2026 EBR for search.** A two-tower model with a listing tower computed offline daily and a query tower online. It was trained with **session-based hard negatives** rather than random negatives [33][34]. HNSW scored slightly better on recall, but Airbnb chose **IVF**. Frequent price and availability updates caused memory overhead with HNSW. With IVF, cluster assignments become an ordinary search filter in the existing search system [33]. The paper reports significant booking gains [34].

### 7.4 Amazon item-to-item collaborative filtering (2003)

This is the classic heuristic retriever. An offline job builds a **similar-items table** from co-purchase statistics. Item B is related to item A when buyers of A are unusually likely to buy B. Online retrieval is just a lookup over the items the user interacted with [35][36]. The 2003 paper was later chosen by IEEE Internet Computing as the paper that best withstood the "test of time" [36]. Co-engagement tables like this remain a standard source in most retrieval portfolios today. Instagram's item-CF source and Pinterest's related-Pins candidates are direct descendants.

---

## 8. Beyond dot products: multi-interest, tree, path and generative retrieval

| Design | Org / year | Structure | Retrieval op | Key claim |
|---|---|---|---|---|
| MIND [37] | Alibaba / 2019 | K user vectors via capsule routing | K ANN queries | Multi-interest; Tmall homepage |
| TDM [38] | Alibaba / 2018 | Max-heap-like item tree + DNN | Layer-wise beam search | Log-cost with deep models |
| Deep Retrieval [39] | ByteDance / 2020 | D x K discrete path codes | Beam search on paths | Near brute-force accuracy |
| TIGER [40] | Google / 2023 | RQ-VAE semantic IDs | Autoregressive decoding | Beats random IDs; cold start |
| Andromeda [9] | Meta / 2024 | Learned hierarchical index | Hierarchical expansion | +6% recall vs prior |

- **MIND** (Multi-Interest Network with Dynamic Routing) uses capsule routing to group a user's behaviors into several interest vectors, trained with label-aware attention. Each vector issues its own ANN query. It was deployed for major traffic on the Mobile Tmall homepage [37]. Pinterest's multi-embedding work follows the same pattern [12].
- **TDM** (Tree-based Deep Model) attaches items to the leaves of a tree and trains a deep model to score tree nodes, using a max-heap-like formulation. Retrieval is a layer-wise top-down beam search, so even a model that is not an inner product achieves cost logarithmic in corpus size. The follow-up JTM learns the tree jointly with the model [38].
- **Deep Retrieval** (ByteDance) maps each item to discrete latent paths. The path codes are learned as model parameters with the same objective, and training uses EM. Beam search over paths retrieves candidates. With sub-linear cost it matched brute-force accuracy on public data and beat a well-tuned ANN baseline in production engagement [39].
- **TIGER** (generative retrieval) quantizes content embeddings, such as Sentence-T5 vectors, into hierarchical **semantic IDs** with an RQ-VAE. A seq2seq Transformer then decodes the next item's ID tokens [40]. Because similar items share code prefixes, this helps cold start. Generative retrieval is covered in more depth in a later chapter.

---

## 9. ANN index families

Every embedding retriever above needs a MIPS or cosine nearest-neighbor index. The main families follow.

### 9.1 Inverted file plus product quantization (IVF-PQ, FAISS)

**Product quantization** (Jegou, Douze and Schmid, TPAMI 2011) splits a d-dimensional vector into m sub-vectors and quantizes each with its own small codebook. A vector becomes m bytes, and **asymmetric distance computation** compares an uncompressed query against the compressed codes using lookup tables. The method was validated on a two-billion-vector dataset [41]. **IVF-ADC** adds a coarse quantizer (k-means centroids). Each vector goes into the inverted list of its nearest centroid, with PQ encoding the residual. A query scans only the `nprobe` nearest lists [41][42].

**FAISS** (Meta, open source since 2017) is the reference implementation [44][43]. The GPU paper by Johnson, Douze and Jegou (2017) introduced a k-selection algorithm reaching about 55% of theoretical peak and an 8.5x speedup over the prior GPU state of the art. It built a k-NN graph over 95 million YFCC100M images in 35 minutes, and over 1 billion vectors in under 12 hours on 4 Maxwell Titan X GPUs [42]. The 2024 "The Faiss library" paper documents the design space: flat, IVF, PQ, OPQ, HNSW and their compositions [43]. Facebook EBR and LiNR's successors both use IVF with quantized codes [6][22].

### 9.2 ScaNN and SOAR (Google)

**ScaNN** introduced **anisotropic vector quantization**. Quantization error parallel to the data point is penalized more than orthogonal error, because parallel error distorts the inner products that matter for top-k MIPS. On ann-benchmarks glove-100-angular at launch, it handled about twice the QPS of the next-best library at equal recall [45]. **SOAR** (NeurIPS 2023) adds **spilling**: each vector is assigned to multiple clusters, and a modified loss encourages the secondary assignment's residual to be orthogonal to the primary one. Google reports the best speed-memory-recall trade-off on glove-100. Libraries close in query speed needed more than 10x the memory and 50x the indexing time [46].

### 9.3 HNSW

**Hierarchical Navigable Small World graphs** (Malkov and Yashunin, 2016/TPAMI 2018) build a multi-layer proximity graph. Each element's top layer is drawn from an exponentially decaying distribution. Search starts greedily at the sparse top layer and descends layer by layer, giving roughly logarithmic scaling [47]. Later work notes that the logarithmic behavior is hard to achieve consistently in high dimensions [47].

HNSW gives excellent recall and latency in RAM. Its costs are memory overhead, slow builds and awkward deletes and updates. Pinterest's Manas started on HNSW [16], and Airbnb rejected it because of its update-memory overhead [33]. **Spotify's Voyager** (2023), built on hnswlib, replaced Spotify's tree-based Annoy. Spotify reports up to 10x the speed of Annoy at similar recall, or up to 50% higher accuracy at the same speed. It has Python and Java bindings and is queried hundreds of millions of times per day [48].

### 9.4 DiskANN and SSD-resident indexes

**DiskANN** (Microsoft, NeurIPS 2019) builds a **Vamana** graph with a smaller search radius than HNSW or NSG, which keeps SSD reads per query low. Compressed PQ vectors stay in RAM, and full-precision vectors and the graph live on SSD. On SIFT1B, a single 64 GB-RAM, 16-core machine served more than 5,000 QPS at under 3 ms mean latency with at least 95% 1-recall@1. It indexes 5-10x more points per node than in-memory graphs in the high-recall regime [49]. **FreshDiskANN** adds concurrent real-time inserts, deletes and searches, with thousands of each per second, on a billion-point index. It reports a 5-10x lower cost of maintaining freshness [50]. Pinterest's adoption of SPANN, which keeps an in-memory centroid index and SSD posting lists, follows the same economics [14].

### 9.5 Comparison of ANN families

| Family | Example impl. | Memory | Updates | Filtering | Strength | Weakness |
|---|---|---|---|---|---|---|
| Flat / brute force | FAISS Flat, LiNR GPU | Full vectors | Trivial | Exact pre-filter | Exact; GPU-friendly | O(N) per query |
| IVF-Flat / IVF-SQ | FAISS, Airbnb, Manas | Medium | Easy (append) | Cluster ID as filter | Simple; index-friendly | Recall depends on nprobe |
| IVF-PQ | FAISS, FB EBR | Very low (m B/vec) | Easy | Terms + payloads | Billions in RAM | Quantization recall loss |
| Anisotropic VQ + spill | ScaNN / SOAR | Low | Rebuild-oriented | Limited | Top MIPS benchmarks | Google-centric tooling |
| HNSW | hnswlib, Voyager, Manas | High (graph + vectors) | Inserts OK, deletes hard | Post-filter typical | Best in-RAM recall/latency | Memory; churn |
| Vamana / DiskANN | DiskANN, FreshDiskANN | RAM: PQ; SSD: graph | Fresh variant | Filtered variant | Billions per node on SSD | SSD I/O tuning |
| SPANN | Manas (2026) | RAM centroids + SSD | Moderate | Via postings | Cheap at scale | Long-tail latency |

---

## 10. Multi-source retrieval and merging

Almost every system above combines several retrievers. The recurring patterns are:

1. **Quota or weight mixing.** Instagram Explore mixes heuristic and ML sources with tunable weights per user group [7]. Twitter targets roughly a 50/50 in-network/out-of-network split [23].
2. **Round-robin with de-duplication.** Pinterest's multi-embedding retrieval interleaves results from each interest embedding and de-duplicates them, to balance interest coverage [12].
3. **A learned pre-ranker as the merger.** Instead of hand-tuned quotas, all candidates go through a cheap shared scorer. Instagram's first-stage two-tower ranker cuts about 1,500 candidates to 100 [8]. Pinterest has also written about a unified lightweight two-tower scoring stage for Home Feed.
4. **Source-specific strengths:**
   - graph walks give freshness and serendipity (Pixie, GraphJet);
   - item-to-item CF gives precise, explainable "more like this" (Amazon, Instagram);
   - EBR gives generalization and long-tail coverage (Facebook, Pinterest);
   - social-graph expansion gives explicit relationships (Twitter in-network, Snap FoF);
   - heuristics give guarantees for trending, fresh or local content (Instagram trending).
5. **Pre-filter vs. post-filter.** Post-filtering ANN results, such as by language, availability or policy, wastes recall when filters are selective. Systems respond in different ways:
   - pushing filters into the index (Facebook's Unicorn terms; Airbnb's IVF cluster as a filter);
   - streaming bitmap filters (Manas);
   - over-fetching (early Manas);
   - exact pre-filtering under GPU brute force (LiNR) [6][15][20][33].

```
  Generic multi-source retrieval service
  request --> fan-out (parallel, per-source timeout)
     |-- EBR source(s): user tower -> ANN (k per embedding) -----\
     |-- item-CF: recent items -> co-engagement table lookups ---|
     |-- graph walk: seeds -> random walks / traversals ---------+--> union
     |-- social: follow / FoF expansion ---------------------------|   dedup
     |-- heuristic: trending / fresh / local ----------------------/   filter
                                                                        |
                       quotas / weights / round-robin / pre-ranker ------+
                                                                        v
                                                         1e2..1e3 to ranking
```

---

## 11. System-level comparison of production retrieval designs

| System | Retrieval type | Corpus scale | Output size | Index / engine | Notable trick |
|---|---|---|---|---|---|
| YouTube DNN 2016 [1] | Softmax-trained embeddings | Millions of videos | Hundreds | NN search | Sampled softmax |
| YouTube NDR 2019 [2] | Two-tower | Tens of millions | n/a | ANN | In-batch + streaming logQ |
| Facebook Search EBR [6] | Two-tower, triplet loss | Billions of docs | n/a | FAISS IVF-PQ in Unicorn | Hard negatives; ANN in inverted index |
| Instagram Explore [7][8] | Multi-source + two-tower | Billions of media | <= ~1,500 -> 100 | Cached towers | Weighted source mix |
| Meta Andromeda [9] | Learned hierarchical index | Very large ad pool | Few thousand (approx.) | Grace Hopper GPUs | Joint index + model training |
| Pinterest Pixie [18] | Random walk | 3B nodes / 17B edges | Top-k pins | Single-machine RAM | p99 < 60 ms, ~100k QPS cluster |
| Pinterest learned retrieval [10][11] | Two-tower, multi-embedding | Billions of Pins | n/a | Manas (HNSW, IVF, SPANN) | Round-robin merge; version sync |
| LinkedIn LiNR [20] | GPU exhaustive + pre-filter | ~1B items | n/a | Model binary on GPU | Pre-filtering; CDC updates |
| LinkedIn LLM retrieval [21] | LLaMA 3 dual encoder | Hundreds of millions | ~2,000 | ANN | Quantized numeric tokens |
| Twitter/X For You [23] | Graph + embeddings | Hundreds of millions | ~1,500 | GraphJet, SimClusters, TwHIN | 50/50 in/out-of-network |
| Snap friend EBR [26][27] | GNN embeddings | Billions of users | n/a | ANN | Query expansion |
| Uber Eats TTE [29] | Two-tower | Thousands per city | n/a | SIA ANN | One model replaces per-city models |
| Etsy UEPPR [31] | Two-tower | Large catalog | n/a | ANN | Quality boost as extra dims |
| Airbnb EBR [33] | Two-tower | Millions of listings | n/a | IVF as filter | Session hard negatives |

---

## Key takeaways for system designers

1. **Factor for retrieval, interact for ranking.** Keep the retrieval score decomposable, such as a dot product, a tree or a path code. Push feature interactions into the towers (MaskNet or DHEN crossing at Pinterest [11]) or into the next stage. Learned indexes (Andromeda [9], TDM [38], Deep Retrieval [39]) and GPU brute force (LiNR [20]) are the escape routes when the dot product becomes the bottleneck.
2. **Negative sampling is the model.**
   - In-batch negatives are cheap but popularity-biased. Always apply a logQ-style correction [2], and check how it treats the positive [5].
   - Add uniform corpus negatives so unseen items are not left unconstrained [3].
   - Use memory banks to get more negatives cheaply [4].
   - Mine hard negatives carefully. Shown-but-not-clicked items can destroy recall when used as retrieval negatives [6].
3. **Measure the ANN gap separately from the model gap.** Track exact-KNN recall and ANN recall such as 1-Recall@10 [6]. Re-tune nprobe, cluster count and quantization whenever the model changes. Scalar quantization is often almost free, and PQ often is not [14].
4. **Choose the index for your update pattern, not only for benchmark recall.** HNSW wins in-RAM benchmarks but struggles with heavy churn, which is why Airbnb chose IVF [33]. SSD-resident designs (DiskANN, SPANN) cut cost for billion-scale corpora [49][14]. Pushing filters into the index beats post-filtering for selective constraints [6][20].
5. **Version both towers atomically.** User-tower serving and item-index builds must use the same model version. Make that a deployment invariant [10].
6. **Run a portfolio.** Combine EBR with item-CF, graph walks, social expansion and heuristics, then merge them with quotas, round-robin or a learned pre-ranker [7][12][23]. Each source covers a different failure mode: cold start, freshness, diversity or explicit relationships.
7. **One user vector is not enough for diverse users.** Multi-interest embeddings (MIND [37], Pinterest [12]) and query expansion (Snap [27]) raise recall at a moderate latency cost. Pinterest reported p90 going from 150 to 205 ms [12].
8. **Content and LLM encoders fix cold start.** Text-based LLM dual encoders helped LinkedIn's newest members most [21]. Semantic IDs give generative retrieval a shared structure for new items [40].
9. **Precompute and cache aggressively.** Item embeddings are computed offline. User embeddings are computed once per request or cached (Instagram caches both sides [7]). Retrieval latency budgets are tens of milliseconds, for example Uber's 50 ms [30].
10. **Align retrieval with ranking.** Train early stages to predict what later stages would rank highly (Instagram [7]), and feed retrieval embeddings into rankers as features (Facebook [6], Airbnb [32]). Retrieval that returns items the ranker never picks is wasted capacity.

---

## References

1. Covington, P., Adams, J., Sargin, E. "Deep Neural Networks for YouTube Recommendations." Google, RecSys 2016. https://research.google/pubs/pub45530
2. Yi, X., et al. "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations." Google, RecSys 2019. https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/
3. Yang, J., Yi, X., et al. "Mixed Negative Sampling for Learning Two-tower Neural Networks in Recommendations." Google, WWW 2020 Companion. https://research.google/pubs/mixed-negative-sampling-for-learning-two-tower-neural-networks-in-recommendations/
4. Wang, J., Zhu, J., He, X. "Cross-Batch Negative Sampling for Training Two-Tower Recommenders." Huawei Noah's Ark Lab, SIGIR 2021. https://arxiv.org/abs/2110.15154v1
5. Khrylchenko, K., et al. "Correcting the LogQ Correction: Revisiting Sampled Softmax for Large-Scale Retrieval." Yandex, RecSys 2025. https://arxiv.org/pdf/2507.09331
6. Huang, J.-T., et al. "Embedding-based Retrieval in Facebook Search." Facebook, KDD 2020. https://arxiv.org/pdf/2006.11632
7. Meta Engineering. "Scaling the Instagram Explore recommendations system." Meta, 2023. https://engineering.fb.com/2023/08/09/ml-applications/scaling-instagram-explore-recommendations-system/
8. Meta Transparency Center. "Instagram Explore AI system." Meta, accessed 2026. https://transparency.meta.com/features/explaining-ranking/ig-explore/
9. Meta Engineering. "Meta Andromeda: Supercharging Advantage+ automation with the next-gen personalized ads retrieval engine." Meta, 2024. https://engineering.fb.com/2024/12/02/production-engineering/meta-andromeda-advantage-automation-next-gen-personalized-ads-retrieval-engine/
10. Pinterest Engineering. "Establishing a Large Scale Learned Retrieval System at Pinterest." Pinterest, 2025. https://medium.com/pinterest-engineering/establishing-a-large-scale-learned-retrieval-system-at-pinterest-eb0eaf7b92c5
11. Pinterest Engineering. "Advancements in Embedding-Based Retrieval at Pinterest Homefeed." Pinterest, 2025. https://medium.com/pinterest-engineering/advancements-in-embedding-based-retrieval-at-pinterest-homefeed-d7d7971a409e
12. Pinterest. "Synergizing Implicit and Explicit User Interests: A Multi-Embedding Retrieval Framework at Pinterest." KDD 2025. https://arxiv.org/pdf/2506.23060
13. Pinterest Engineering. "Scaling Conditional Learned Retrieval for Pinterest Home Feed." Pinterest, 2026. https://medium.com/pinterest-engineering/scaling-conditional-learned-retrieval-for-pinterest-home-feed-ecfba7e5a426
14. Pinterest Engineering. "Evolving Pinterest's Embedding Retrieval Platform." Pinterest, 2026 (summarized by InfoQ, https://www.infoq.com/news/2026/09/pinterest-search/). https://medium.com/pinterest-engineering/evolving-pinterests-embedding-retrieval-platform-aede4e831e01
15. Koh, T. "Manas HNSW Streaming Filters." Pinterest Engineering, 2022. https://medium.com/pinterest-engineering/manas-hnsw-streaming-filters-351adf9ac1c4
16. Pinterest Engineering. "Manas HNSW Realtime: Powering Realtime Embedding-Based Retrieval." Pinterest. https://medium.com/pinterest-engineering/manas-hnsw-realtime-powering-realtime-embedding-based-retrieval-dc71dfd6afdd
17. Ying, R., et al. "Graph Convolutional Neural Networks for Web-Scale Recommender Systems" (PinSage). Pinterest/Stanford, KDD 2018. https://arxiv.org/pdf/1806.01973 ; blog: https://medium.com/pinterest-engineering/pinsage-a-new-graph-convolutional-neural-network-for-web-scale-recommender-systems-88795a107f48
18. Eksombatchai, C., et al. "Pixie: A System for Recommending 3+ Billion Items to 200+ Million Users in Real-Time." Pinterest/Stanford, WWW 2018. https://arxiv.org/pdf/1711.07601
19. Pinterest. "Bootstrapping Conditional Retrieval for User-to-Item Recommendations." 2024/2025. https://arxiv.org/html/2508.16793
20. Borisyuk, F., et al. "LiNR: Model Based Neural Retrieval on GPUs at LinkedIn." LinkedIn, 2024. https://arxiv.org/pdf/2407.13218
21. LinkedIn. "Large Scale Retrieval for the LinkedIn Feed using Causal Language Models." 2025 (AAAI 2026). https://arxiv.org/abs/2510.14223
22. LinkedIn. "Connected Content Retriever: Dense Graph Edge Features Powering Pre-Ranking at LinkedIn." 2026. https://arxiv.org/pdf/2609.22441
23. Twitter Engineering. "Twitter's Recommendation Algorithm." Twitter, 2023. https://blog.x.com/engineering/en_us/topics/open-source/2023/twitter-recommendation-algorithm
24. Twitter/X. "the-algorithm" source repository. 2023. https://github.com/twitter/the-algorithm
25. X Help Center. "Communities Recommendations" (SimClusters). X Corp. https://help.x.com/en/resources/recommender-systems/communities-recommendations
26. Snap Research. "Embedding Based Retrieval in Friend Recommendation." Snap, SIGIR 2023. https://research.snap.com/publications/embedding-based-retrieval-in-friend-recommendation.html
27. Snap Research. "Improving Embedding-Based Retrieval in Friend Recommendation with ANN Query Expansion." Snap. https://research.snap.com/publications/improving-embedding-based-retrieval-in-friend-recommendation-with-ann-query-expansion.html
28. Snap. "GiGL: Large-Scale Graph Neural Networks at Snapchat." 2025. https://arxiv.org/pdf/2502.15054
29. Uber Engineering. "Innovative Recommendation Applications Using Two Tower Embeddings at Uber." Uber. https://eng.uber.com/blog/innovative-recommendation-applications-using-two-tower-embeddings/
30. Kurra, D. D. "Uber - Embeddings" (Feature Store Summit 2024 talk). Uber, 2024. https://content.hopsworks.ai/hubfs/Feature%20Store%20Summit%202024/FS%20Summit%2024%20-%20Uber%20-%20Embeddings.pdf
31. Jha, R., et al. "Unified Embedding Based Personalized Retrieval in Etsy Search." Etsy, 2023/2024. https://arxiv.org/abs/2306.04833v2
32. BestPractice.AI case study of Grbovic and Cheng, "Real-time Personalization using Embeddings for Search Ranking at Airbnb" (KDD 2018), secondary summary. https://bestpractice.ai/ai-use-cases/case-studies/consumer-retail/airbnb-increased-similar-property-listing-recommendations-click-through-rates-by-21-with-machine-learning-embeddings
33. Airbnb Engineering. "Embedding-Based Retrieval for Airbnb Search." Airbnb, 2025. https://airbnb.tech/ai-ml/embedding-based-retrieval-for-airbnb-search/
34. Airbnb. "Applying Embedding-Based Retrieval to Airbnb Search." 2026. https://arxiv.org/pdf/2601.06873
35. Linden, G., Smith, B., York, J. "Amazon.com Recommendations: Item-to-Item Collaborative Filtering." IEEE Internet Computing, 2003. https://www.cs.umd.edu/%7Esamir/498/Amazon-Recommendations.pdf
36. Amazon Science. "The history of Amazon's recommendation algorithm." Amazon. https://www.amazon.science/the-history-of-amazons-recommendation-algorithm
37. Li, C., et al. "Multi-Interest Network with Dynamic Routing for Recommendation at Tmall" (MIND). Alibaba, 2019. https://arxiv.org/abs/1904.08030v1
38. Zhu, H., et al. "Learning Tree-based Deep Model for Recommender Systems" (TDM). Alibaba, KDD 2018. https://arxiv.org/pdf/1801.02294 ; follow-up JTM: https://arxiv.org/pdf/1902.07565
39. Gao, W., et al. "Deep Retrieval: Learning A Retrievable Structure for Large-Scale Recommendations." ByteDance, 2020/2021. https://arxiv.org/abs/2007.07203v1
40. Rajput, S., et al. "Recommender Systems with Generative Retrieval" (TIGER). Google, NeurIPS 2023. https://ar5iv.labs.arxiv.org/html/2305.05065
41. Jegou, H., Douze, M., Schmid, C. "Product Quantization for Nearest Neighbor Search." INRIA, IEEE TPAMI 2011. https://hal-univ-rennes1.archives-ouvertes.fr/inria-00514462
42. Johnson, J., Douze, M., Jegou, H. "Billion-scale similarity search with GPUs." Facebook AI Research, 2017. https://arxiv.org/pdf/1702.08734
43. Douze, M., et al. "The Faiss library." Meta FAIR, 2024. https://arxiv.org/abs/2401.08281
44. Meta Engineering. "Faiss: A library for efficient similarity search." Facebook, 2017. https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/
45. Google Research. "Announcing ScaNN: Efficient Vector Similarity Search." Google, 2020. https://research.google/blog/announcing-scann-efficient-vector-similarity-search/
46. Google Research. "SOAR: New algorithms for even faster vector search with ScaNN." Google, 2024. https://research.google/blog/soar-new-algorithms-for-even-faster-vector-search-with-scann/
47. Malkov, Y. A., Yashunin, D. A. "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs." arXiv 2016 / TPAMI 2018. https://arxiv.org/pdf/1603.09320
48. Spotify Engineering. "Introducing Voyager: Spotify's New Nearest-Neighbor Search Library." Spotify, 2023. https://engineering.atspotify.com/introducing-voyager-spotifys-new-nearest-neighbor-search-library
49. Subramanya, S. J., et al. "DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node." Microsoft Research, NeurIPS 2019. https://papers.neurips.cc/paper/9527-rand-nsg-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node.pdf
50. Singh, A., et al. "FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming Similarity Search." Microsoft Research, 2021. https://www.microsoft.com/en-us/research/?p=905277
