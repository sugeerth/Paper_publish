# 34. Open-Source Frameworks, Tooling, Benchmarks and Public Datasets

Most of the systems in this atlas are proprietary. Almost all of them, though, are built from a small set of open-source parts, or have open-source versions of their core ideas. Meta's sharded embedding training comes out as TorchRec and FBGEMM. Google's two-tower retrieval and quantized maximum-inner-product search come out as TensorFlow Recommenders and ScaNN. NVIDIA's GPU-resident pipelines come out as Merlin. Academic groups maintain RecBole, Microsoft Recommenders, LightFM and implicit as testbeds. The serving layer is served by search engines and vector databases such as Vespa, Milvus, Weaviate, Qdrant and FAISS. Feature stores (Feast) and workflow and compute frameworks (Ray, Metaflow) hold the system together.

This chapter maps that ecosystem layer by layer. It then covers the public datasets and benchmarks that nearly every paper in the field reports on: MovieLens, Criteo, Avazu, Amazon Reviews, KuaiRand, MIND and Yambda. For each one it gives the problem it targets, its architecture, the scale numbers the maintainers publish, and the lessons a system designer should take away.

A note on sourcing. Live web search was unavailable while this chapter was being researched, and several hosts (arxiv.org, grouplens.org, huggingface.co, kuairand.com, docs.vespa.ai) could not be fetched. The chapter therefore relies mainly on primary GitHub repositories, READMEs, source files and notebooks, which were read directly. Facts that come from other chapters of this atlas, or from the author's prior knowledge of well-known papers, are marked **(approximate / not re-verified)**. Treat those figures as pointers to check, not as authoritative.

```
              The open-source recsys stack (typical composition)
+------------------------------------------------------------------------------+
| Orchestration / compute     Metaflow (flows, versioning)   Ray (Data/Train/Serve)|
+------------------------------------------------------------------------------+
| Features                    Feast: offline store <-> registry <-> online store   |
+------------------------------------------------------------------------------+
| Training                    TorchRec+FBGEMM | TFRS (+TFRA) | Merlin HugeCTR/Models|
|                             RecBole | MS Recommenders | LightFM | implicit       |
+------------------------------------------------------------------------------+
| Retrieval index             FAISS | ScaNN | Milvus | Weaviate | Qdrant | Vespa   |
+------------------------------------------------------------------------------+
| Serving / ranking           Triton (Merlin Systems) | Vespa phased ranking |     |
|                             TorchRec inference (C++/TorchScript)               |
+------------------------------------------------------------------------------+
| Benchmarks / data           MovieLens Criteo Avazu Amazon KuaiRand MIND Yambda |
|                             BARS / FuxiCTR / RecBole protocols                 |
+------------------------------------------------------------------------------+
```

---

## 34.1 Industrial-scale training frameworks

### 34.1.1 TorchRec and FBGEMM (Meta / PyTorch)

**Problem.** Deep learning recommendation models (DLRMs) are dominated by sparse embedding tables. These can reach terabytes, far beyond one GPU's HBM. The dense MLPs on top are comparatively small. Plain data parallelism (replicating everything) fails on memory, and plain model parallelism leaves dense compute underused. Meta built TorchRec as "a PyTorch domain library" to make hybrid parallelism a first-class primitive [1][2].

**Architecture.** According to the README and the RecSys 2022 paper [1][2], TorchRec provides:

- **Modeling primitives:** embedding bags and jagged tensors (the `KeyedJaggedTensor` abstraction holds variable-length ID lists per feature), plus common recsys modules and an end-to-end DLRM example [1][2].
- **Sharding types:** data-parallel, table-wise, row-wise, table-wise-row-wise, column-wise and table-wise-column-wise sharding of embedding tables [1].
- **A planner** that "automatically generates optimized sharding plans" [1]. It estimates compute, memory and communication cost and assigns a scheme per table.
- **Kernels from FBGEMM** [1][3]. FBGEMM is a low-precision, high-performance GEMM library that supports row-wise and outlier-aware quantization. FBGEMM_GPU adds recsys GPU operators for training and inference, and a newer FBGEMM_GPU GenAI package adds FP8 row-wise quantization and collectives [3].
- **Train pipelining** that overlaps data loading, host-to-device copy, inter-device communication (ID and embedding all-to-all) and compute [1].
- **Reduced-precision training and inference**, and export to C++ inference [1]. The TorchRec inference library captures the model with `torch.fx` and runs it as TorchScript in a C++ gRPC server. Its README calls it "a simple reference ... not a fully scaled out solution" [5].

```
        TorchRec sparse/dense hybrid step (per batch, conceptual)
  GPU0..GPUn each hold: replicated dense MLPs (DDP) + a SHARD of embedding tables
  1. input KJT (ids per feature) --all-to-all--> owner shard of each table/row range
  2. owner shard: fused table-batched lookup (FBGEMM TBE) -> pooled embeddings
  3. pooled embeddings --all-to-all--> back to the sample's GPU
  4. dense forward/backward (interaction + MLPs), allreduce dense grads
  5. embedding grads --all-to-all--> owner shard; fused optimizer update in-kernel
  Pipeline: step 1 of batch i+1 overlaps steps 2-5 of batch i
```

**Scale.** TorchRec "powers many production RecSys models at Meta" [1]. Its paper abstract describes Meta's move from CPU asynchronous training to accelerator-based synchronous training [2]. The README links a follow-up on "Toward 100TB model with Embedding Offloading" [1]. The Neo/ZionEX system that TorchRec productionizes reported models of up to 12 trillion parameters **(approximate, from the Neo paper as summarized in Chapter 9)** [37]. The repository ships loaders for Criteo 1TB (24 days, 13 integer and 26 categorical features, 4,195,197,692 training rows across days 0-22, with day 23 for validation and test) and for MovieLens 20M/25M [4].

**Key tricks and lessons.**

- Choose sharding *per table*. Tiny tables go data-parallel, most tables go table-wise, and huge tables go row-wise or column-wise. A cost-model planner beats hand placement.
- Fuse many small embedding lookups into one table-batched kernel, and fuse the sparse optimizer into the backward pass so dense gradient tensors for sparse tables are never materialized [3].
- Quantize both storage (row-wise INT8/INT4) and communication [3].
- Treat the C++ inference path as a reference. Production adds request batching, feature pipelines and sharded serving [5].

### 34.1.2 TensorFlow Recommenders (Google)

**Problem.** TFRS packages the Google two-tower and ranking recipes as a Keras-native library. It covers "data preparation, model formulation, training, evaluation, and deployment" [6].

**Architecture.** Models subclass `tfrs.Model` and implement `compute_loss`. Two task objects carry most of the logic [6]:

- `tfrs.tasks.Retrieval` computes an in-batch softmax loss for a query tower and a candidate tower. The `tfrs.metrics.FactorizedTopK` metric computes top-K recall against the full candidate corpus.
- `tfrs.tasks.Ranking` handles pointwise or listwise scoring of the retrieved candidates (cross layers such as DCN are offered as layers, **not re-verified in this pass**).

For serving, TFRS offers `factorized_top_k.BruteForce` and `factorized_top_k.ScaNN` layers with the same interface. Both can be exported as a SavedModel [7].

**Scale numbers (tutorial).** The efficient-serving tutorial [7] starts from MovieLens 100K (fewer than 1,000 movies). It pads the corpus with perturbed dummy embeddings to about 1 million candidates. Brute-force latency "grows linearly" with corpus size and is projected at about 250 ms per query at 10 million candidates. At about 1M candidates, ScaNN returns results in about 2 ms, roughly 15x faster than brute force, with top-100 accuracy "almost the same." Tuning `num_leaves` and `num_leaves_to_search` (for example 1000/100 rather than the 100/10 default), or adding reordering candidates (for example 70 leaves searched and 400 reordering candidates), trades latency for about 3 points of absolute recall [7].

**Lessons.** TFRS is the most readable reference for two-tower training with in-batch negatives. It is also a good teaching tool for the shift in evaluation from brute force to ANN. Its activity is modest (about 2k stars, 389 commits) [6]. Large-scale dynamic embeddings in the TensorFlow world live in the separate TensorFlow Recommenders-Addons (TFRA) project **(see Chapter 9)**.

### 34.1.3 NVIDIA Merlin (NVTabular, HugeCTR, Models, Transformers4Rec, Systems)

**Problem.** NVIDIA's goal is to keep the entire recommender lifecycle on GPUs: feature engineering, training and inference. Merlin is an umbrella of libraries with a shared schema [8].

**Components** [8][9][12][13][14][15]:

| Component | Role | Notes |
|---|---|---|
| NVTabular | GPU ETL for tabular data | Handles TB-scale data beyond GPU and CPU memory |
| HugeCTR | CUDA/C++ training framework | Model-parallel embeddings, data-parallel dense layers, multi-node |
| Sparse Operation Kit (SOK) | GPU sparse ops for TF users | For teams who cannot adopt HugeCTR directly |
| HierarchicalKV | Header-only GPU hash table for embeddings | HBM + host memory, score-based eviction |
| Merlin Models | Reference retrieval and ranking models | MF, two-tower, YouTube DNN, DLRM, DCN-v2, DeepFM; TF first |
| Transformers4Rec | Session-based and sequential recommendation | Wraps 64+ Hugging Face architectures |
| Merlin Systems | Serving DAGs on Triton | Feast, FAISS and model operators in one ensemble |
| Merlin Core | Dataset abstraction, schema, DAG API | Shared by the above |

**HugeCTR training.** Embedding layers are model-parallel across GPUs and nodes, and the dense model is data-parallel. Communication uses NCCL with GPUDirect RDMA. Embedding variants include localized-slot (each slot on one GPU) and distributed-slot (keys spread across GPUs) [11]. The RecSys 2022 paper describes a **hybrid embedding**: hot categories are replicated and cold categories are sharded, so frequent IDs avoid all-to-all. This was a key optimization in NVIDIA's MLPerf DLRM submissions [10]. HugeCTR documentation states that two nodes of 8x A100 80 GB can train models of about 1 TB entirely in GPU memory [11]. The repository's quick-start is a DCN model on Criteo-style input (26 sparse slots, 13 dense features, embedding size 16) [9].

**Deprecations.** The Hierarchical Parameter Server (HPS) and Embedding Cache are listed as deprecated as of v25.03 [9]. HierarchicalKV is the surviving general-purpose piece. It keeps keys in HBM, lets values spill to host memory, and evicts only when a bucket is full. Eviction is driven by a per-key score under LRU, LFU, EpochLRU, EpochLFU or custom policies [12]. Chapter 9 summarizes HKV's published A100 benchmarks: about 1-2.5 billion KV/s for pure-HBM operations, dropping by an order of magnitude or more in hybrid HBM+host mode **(approximate)**.

**Merlin Systems** expresses an end-to-end recommender as a Triton ensemble [14]:

```
  request(user_id)
     |
  QueryFeast(user features) --> PredictTensorflow(query tower) --> QueryFaiss(top-100)
     |                                                               |
     +--> FilterCandidates(remove already-seen) <--------------------+
                |
          QueryFeast(item features) -> UnrollFeatures -> PredictTensorflow(ranker)
                |
          SoftmaxSampling(temperature, top-10)  --> response
  Ensemble(...).export(dir);  tritonserver --model-repository=dir
```

This is the canonical four-stage design (retrieval, filtering, ranking, ordering) shown in Chapter 1, packaged as code. Softmax sampling at the ordering stage adds explicit exploration [14].

**Transformers4Rec** adapts Hugging Face transformer architectures, which natively take only token-ID sequences, to sequences of multiple tabular features. It supports next-item prediction, classification and regression heads [15]. Transformer-based approaches from the team won the 2021 WSDM WebTour (Booking.com) and SIGIR eCommerce (Coveo) session-based challenges [15].

**Lessons.** Merlin's main contribution is architectural: one schema flows from ETL through training to the serving DAG, so training/serving skew is reduced by construction. Its maintenance signals are mixed, though. The Merlin meta-repository shows about 200 open issues, Merlin Models states its PyTorch API is "initiated, but incomplete", and HPS is deprecated [8][9][13]. Adopters should treat it as a set of components to cherry-pick (NVTabular, HKV, Triton ensembles) rather than as a single supported platform.

---

## 34.2 Research and prototyping libraries

### 34.2.1 RecBole (RUC AI Box)

RecBole is the de facto unified academic benchmark library in PyTorch. Its README currently lists **94 algorithms** across four families (general/collaborative filtering, sequential, context-aware/CTR, and knowledge-based). It also cites 44 benchmark datasets in the introduction, while its comparison table lists 43 [16]. All datasets are converted to a single atomic-file format (`.inter`, `.user`, `.item`, `.kg`). Features include multi-GPU and mixed-precision training, three hyperparameter search methods, standardized evaluation protocols, and time and memory benchmarks [16]. Releases run from v0.1.1 (November 2020) to v1.2.1 (February 2025). The project is described in three papers: RecBole 1.0 (CIKM 2021), RecBole 2.0 (CIKM 2022, eight extension packages) and a SIGIR 2023 resource paper [16].

**Design lesson.** RecBole separates the data, model, trainer and evaluator modules behind a configuration system. A new model therefore only implements `calculate_loss` and `predict`/`full_sort_predict`. That separation is what makes fair comparison possible: same splits, same negative sampling, same full-ranking metrics.

### 34.2.2 Microsoft Recommenders (Linux Foundation AI & Data)

Microsoft Recommenders, now under the Linux Foundation AI & Data, is a best-practices repository. It passed 20,000 GitHub stars in April 2025 [17]. It is organized as notebooks for five tasks (prepare data, model, evaluate, model select and optimize, operationalize), plus a `recommenders` Python package with data loaders, splitters (random, chronological, stratified) and metrics (MAP, nDCG@k, Precision@k, Recall@k, RMSE, MAE, R2, explained variance) [17].

The algorithm list spans classic and deep CF: ALS, BPR (via Cornac), BiVAE, LightGCN, NCF, SAR, SASRec, xDeepFM, Wide & Deep, Multinomial VAE, RBM and more. It also includes Microsoft's news models (DKN, LSTUR, NAML, NPA, NRMS), which are evaluated on MIND [17]. SARplus provides a Spark implementation of the Simple Algorithm for Recommendation (SAR) [17].

Its MovieLens 100K benchmark notebook (a single 75/25 split, explicitly "illustrative") is a useful sanity table [18]:

| Algorithm | Train time (s) | MAP@10 | nDCG@10 | Precision@10 | Recall@10 |
|---|---|---|---|---|---|
| ALS (Spark) | 13.4 | 0.011 | 0.037 | 0.043 | 0.015 |
| SAR | 0.6 | 0.258 | 0.394 | 0.341 | 0.185 |
| NCF | 70.3 | 0.277 | 0.411 | 0.360 | 0.189 |
| BPR (Cornac) | 3.8 | 0.229 | 0.362 | 0.285 | 0.252 |
| BiVAE | 31.5 | 0.330 | 0.472 | 0.411 | 0.224 |
| LightGCN | 28.9 | 0.281 | 0.420 | 0.362 | 0.193 |

The lesson is clear. SAR, a co-occurrence item-similarity method that trains in under a second, comes within a few points of nDCG of much heavier neural models on small data [18]. The ALS row is low because it optimizes rating RMSE rather than ranking. A simple, well-tuned baseline belongs in every system's offline evaluation.

### 34.2.3 LightFM (Lyst)

LightFM implements Maciej Kula's hybrid matrix factorization [19]. Each user and item is represented "as the sum of the latent representations of their features." Identity features recover pure MF, and metadata features give cold-start coverage for new items and new users [19]. It ships efficient BPR and WARP ranking losses with multithreaded (Hogwild-style) fitting [19]. Its WARP loss, which samples negatives until one violates the margin and weights the update by the estimated rank, is a widely copied recipe for optimizing precision at the top of the list. LightFM is a good production baseline for small and medium catalogs with rich metadata and frequent cold start. It is not designed for large-scale distributed training.

### 34.2.4 implicit (Ben Frederickson)

`implicit` is the pragmatic workhorse for implicit-feedback CF on a single machine [20]. It implements ALS (Hu-Koren-Volinsky confidence weighting, with a conjugate-gradient solver), BPR, logistic MF, and item-item nearest neighbours (cosine, TF-IDF, BM25). ALS and BPR have custom CUDA kernels, and pip wheels include GPU support on Linux [20]. Recommendation generation can use Annoy, NMSLIB or FAISS for approximate top-K [20]. Two operational notes from the README matter in practice. Set `OPENBLAS_NUM_THREADS=1` / `MKL_NUM_THREADS=1` to avoid oversubscription with the library's own OpenMP threading, which gives "substantial speedups." The project also benchmarks ALS fitting against Spark MLlib [20].

**Design lesson.** For a first-version recommender, or a candidate generator in a larger cascade, `implicit` ALS plus an ANN index over item factors is often the strongest cost/quality point. A GPU ALS sweep over millions of users completes in minutes on one machine.

### 34.2.5 CTR benchmarking libraries: FuxiCTR and BARS

Huawei Noah's Ark and collaborators maintain **FuxiCTR**, a configurable CTR library with a model zoo of 57 entries. The entries span feature interaction (DeepFM, DCN, AutoInt and others), behavior sequence, long sequence, dynamic weight and multi-task models, mostly in PyTorch with some TensorFlow ports [44]. **BARS** ("open BenchmArking for Recommender Systems", SIGIR 2022) builds standardized benchmarks on top of it: BARS-CTR and BARS-Match, with BARS-Rerank and BARS-MTL in progress [43]. The project's stated motivation is that without a recognized benchmark, studies are "hard to reproduce" and results conflict across papers [43]. The companion `reczoo/Datasets` repository publishes frozen splits with versioned IDs (Criteo_x1/x2/x4, Avazu_x1/x2/x4, KKBox_x1, Frappe_x1, MIND_small_x1, AmazonBooks_m1, Yelp18_m1, Gowalla_m1 and more) [42]. A paper can then report "Avazu_x4" and mean exactly one preprocessing and split.

---

## 34.3 Retrieval indexes: ANN libraries and vector databases

Every two-tower retrieval stage needs an index that returns approximate top-K by inner product or cosine over millions to billions of item vectors. The open-source options fall into **libraries** (FAISS, ScaNN), which give you a data structure, and **engines** (Milvus, Weaviate, Qdrant, Vespa), which add storage, replication, filtering and APIs.

### 34.3.1 FAISS (Meta FAIR)

FAISS is the reference ANN library [27][28]. Its index families are [27]:

- Exact (flat) search.
- Compressed-code indexes (product quantization and its variants, binary codes), which do not need the original vectors and "can scale to billions of vectors in main memory on a single server."
- Graph indexes on raw vectors (HNSW, NSG).

Inverted-file (IVF) coarse partitioning composes with these. Index factory strings such as `IVF65536,PQ32` combine coarse partition and code. Metrics are L2, inner product, and cosine (inner product on normalized vectors) [27]. The GPU implementation can replace CPU indexes directly (for example `GpuIndexFlatL2`), supports multi-GPU, runs on CUDA or ROCm, and can optionally use NVIDIA cuVS as a backend [27]. The GPU paper (Johnson, Douze, Jégou) claimed the fastest exact and approximate k-NN for high-dimensional vectors as of 2017 [29]. The 2024 "The Faiss library" paper is now the recommended citation [28].

### 34.3.2 ScaNN (Google Research)

ScaNN targets maximum inner product search (MIPS), which is what dot-product two-tower models need, and it is optimized for x86 AVX/FMA (and ARM NEON) [30]. Search runs in three phases [31]:

```
 query q
   |
   v
 [1] partition (tree, num_leaves ~ sqrt(n)) -> pick num_leaves_to_search leaves
   |
   v
 [2] score candidates with asymmetric hashing (AH), anisotropic quantization,
     dimensions_per_block = 2   (or brute force, optionally int8-quantized)
   |
   v
 [3] rescore top k' (reordering_num_neighbors > k) with exact / int8 dot product
   |
   v
 top-k
```

The library's sizing guidance [31] is: brute force below about 20k points; AH plus rescoring below about 100k; and partitioning plus AH plus rescoring above that, with `num_leaves` around sqrt(n). Int8 brute-force scoring cuts memory 4x and helps bandwidth-bound workloads, with "generally negligible" accuracy loss [31]. The core algorithmic contribution is **anisotropic vector quantization** (ICML 2020). It penalizes quantization error parallel to the data point more than orthogonal error, because parallel error dominates inner-product ranking errors [32]. **SOAR** (NeurIPS 2023) adds spilled assignments with orthogonality-amplified residuals for better partition recall [33]. ScaNN claims state-of-the-art results on ann-benchmarks (glove-100-angular) [30]. It is integrated into TFRS [7], and Milvus offers it as an index type [24].

### 34.3.3 Milvus, Weaviate, Qdrant

| Engine | Language | Index types (as documented) | Distinctive features for recsys | Scale claim (vendor) |
|---|---|---|---|---|
| Milvus [24] | Go/C++ | HNSW, IVF, FLAT, SCANN, DiskANN, quantized variants, mmap; GPU (CAGRA) | Compute/storage disaggregation; query nodes scale for reads, data nodes for writes; Kubernetes-native; streaming ingest | "tens of thousands of search queries on billions of vectors" |
| Weaviate [25] | Go | HNSW (plus others, not re-verified) | Hybrid BM25 + vector in one call; built-in vectorizer modules; multi-tenancy; quantization and multi-vector | "billions of vectors in milliseconds" |
| Qdrant [26] | Rust | HNSW | Rich payload filters (keyword, full-text, range, geo; must/should/must_not); **recommend API with positive and negative examples**; Discovery; sparse vectors; quantization claimed to cut RAM up to 97% | Horizontal sharding and replication, zero-downtime resizing |

All three scale claims are vendor statements without benchmarks in the READMEs [24][25][26].

**Recsys-relevant observations.**

- **Filtering is the hard part.** Recommendation queries almost always carry constraints such as in-stock, region, language, already-seen and safety. Pre-filtering versus post-filtering versus filter-aware graph traversal determines recall under selective filters. Qdrant's payload filter model and Milvus's metadata filtering exist for this reason [24][26].
- **Qdrant's recommend API** turns "more like these, less like those" into a single query using positive and negative example points [26]. That is effectively a lightweight session-based recommender without a trained query tower.
- **Freshness.** Milvus's separation of data nodes from query nodes [24] is a direct answer to the recsys need for continuous item insertion (new videos, new listings) without rebuilding the index.

### 34.3.4 Vespa

Vespa (Apache 2.0, originally from Yahoo) is not just a vector database. It is a serving engine that runs **retrieval and machine-learned ranking next to the data** [21]. Content nodes hold documents, inverted indexes and HNSW tensors, and evaluate first- and second-phase ranking expressions locally. Container nodes scatter queries, merge results and run a global phase. Models (ONNX, GBDT, tensor expressions) execute in the ranking phases [22]. The README states that Vespa is used by large internet services serving "hundreds of thousands of queries" per second, typically within about 100 ms [21]. The sample-apps repository includes album-recommendation and news-recommendation (MIND-based) tutorials that implement two-tower retrieval with `nearestNeighbor` plus ranking profiles [23].

```
  query ---> container (stateless)
               | scatter
     +---------+---------+
     v         v         v
  content   content   content       per node: retrieve (WAND / HNSW / filters)
  node 1    node 2    node N          -> first-phase (cheap expr, all hits)
     |         |         |            -> second-phase (ML model, top rerank-count)
     +---------+---------+
               | gather top hits
               v
        global-phase rerank (e.g. ONNX cross-model on GPU) -> response
```

**Lesson.** Moving compute to the data avoids shipping thousands of candidates' features over the network. That is the same insight as Twitter Earlybird's index-side light ranker (Chapter 10). Vespa is the main open-source system where retrieval, filtering, feature storage (attributes) and multi-phase ranking all live in one engine.

---

## 34.4 Feature stores and ML platforms

### 34.4.1 Feast

Feast is the most widely used open-source feature store [34]. Its architecture has four parts:

- An **offline store** for historical features (Snowflake, BigQuery, Redshift, Spark, DuckDB, Postgres and community plugins such as Hive and Trino).
- An **online store** for low-latency lookup (Redis, DynamoDB, Bigtable, Cassandra/AstraDB, Postgres, SQLite).
- A **registry** of feature definitions, populated by `feast apply`.
- **Feature servers**: Python, with Go and Java servers in alpha [34].

Two operations carry the recsys value [34]:

1. **Point-in-time joins** (`get_historical_features`). Each training row gets feature values as of its event timestamp, which prevents future-data leakage into training labels.
2. **Materialization** (`feast materialize` / `materialize-incremental`) copies the latest values to the online store for serving.

```
  event log (user_id, item_id, ts, label) --+
                                           |  point-in-time join
  feature tables (entity, ts, values) -----+--> training set (no leakage)
            |
            | materialize(-incremental)
            v
  online store (Redis/Dynamo/...) <-- get_online_features(entity rows) <-- ranker
```

Feast also lists vector search (alpha), with Milvus, Qdrant and FAISS as online-store backends [34]. This blurs the line between feature store and retrieval index. Merlin Systems' `QueryFeast` operator shows the standard integration in a serving DAG [14].

**Lesson.** Feast does not compute streaming features itself. It stores and serves them. Teams that need sub-minute counters (clicks in the last 5 minutes) pair it with a stream processor, or choose a computation-oriented platform (Chronon, Tecton, Feathr; see Chapter 8).

### 34.4.2 Ray

Ray is a general distributed runtime. **Core** provides tasks (stateless functions), actors (stateful workers) and an immutable object store. On top sit **Data** (scalable datasets for ML), **Train** (distributed training), **Tune** (hyperparameter search), **Serve** (programmable model serving) and **RLlib** (reinforcement learning) [35]. In recsys stacks, Ray typically does one of three jobs:

- Ray Data streams preprocessing to GPU trainers (replacing Spark-to-files-to-trainer hops).
- Ray Tune sweeps embedding dimensions and learning rates.
- Ray Serve composes multi-model inference graphs (retrieval, ranking and business-logic stages as separate deployments with independent autoscaling).

RLlib is a natural home for bandit and RL slate experiments (Chapter 13).

### 34.4.3 Metaflow (Netflix, now Outerbounds)

Metaflow came out of Netflix's ML platform. It models ML work as a DAG of Python steps with automatic artifact versioning, local-to-cloud scaling on CPUs and GPUs, gang-scheduled distributed jobs, dependency management, and deployment to production orchestrators with event-triggered ("reactive") flows [36]. Netflix reports more than 3,000 AI/ML projects on it, "hundreds of millions of compute jobs", petabytes of data processed and tens of petabytes of models and artifacts [36]. Other named users include Amazon, DoorDash, Dyson, Goldman Sachs and Ramp [36]. For recommender teams, its value is reproducible daily retraining DAGs (extract, features, train, evaluate, publish embeddings, refresh index) with every intermediate artifact addressable for debugging.

### 34.4.4 Framework comparison

| Tool | Layer | Primary language / runtime | Best fit | Scale evidence (public) |
|---|---|---|---|---|
| TorchRec + FBGEMM [1][3] | Training / inference | PyTorch, CUDA | Terabyte embedding DLRMs, sync GPU training | Meta production; 100TB offloading work |
| TFRS [6] | Training | TF / Keras | Two-tower and ranking prototypes, education | Tutorials to ~1M candidates |
| Merlin HugeCTR / HKV [9][12] | Training / embeddings | CUDA / C++ | GPU-resident CTR training, GPU hash tables | ~1 TB on 2x8 A100 |
| Merlin Systems [14] | Serving | Triton | Multi-stage serving DAGs | Reference examples |
| RecBole [16] | Research | PyTorch | Reproducible comparison of 94 models | 44 datasets |
| MS Recommenders [17] | Research / best practices | Python, Spark | Notebooks, metrics, splitters, MIND models | 20k+ stars |
| LightFM [19] | Model | Cython | Hybrid cold-start MF | Single machine |
| implicit [20] | Model | Cython / CUDA | Fast implicit ALS/BPR candidate generation | Single machine, GPU |
| FAISS [27] | ANN library | C++ / CUDA | Billion-scale ANN on CPU or GPU | Billions per server |
| ScaNN [30] | ANN library | C++ (AVX/NEON) | MIPS for dot-product towers | ann-benchmarks SOTA claim |
| Milvus / Weaviate / Qdrant [24-26] | Vector DB | Go-C++ / Go / Rust | Managed ANN with filters and updates | Vendor claims to billions |
| Vespa [21] | Search + ranking engine | C++ / Java | Retrieval and ML ranking on content nodes | Hundreds of thousands of QPS |
| Feast [34] | Feature store | Python (+Go/Java) | Point-in-time training sets, online lookup | Many backends |
| Ray [35] | Compute | Python | Data, train, tune, serve pipelines | General |
| Metaflow [36] | Workflow | Python | Versioned retraining DAGs | 3,000+ projects at Netflix |

---

## 34.5 Public datasets

A dataset is a design decision. It fixes the feedback type (explicit ratings versus implicit events versus impressions), whether negatives are observed, whether the exposure policy is known, the time span, and the split protocol. Those choices decide which conclusions transfer to production.

### 34.5.1 MovieLens (GroupLens, University of Minnesota)

MovieLens is the most-cited recommendation dataset family. It consists of explicit 1-5 star ratings (half stars in later versions) from the MovieLens research site, with tags and genre metadata. The history and appropriate uses are documented by Harper and Konstan (ACM TiiS 2015) [39]. Common variants **(approximate / not re-verified; GroupLens could not be fetched)**: ML-100K (about 100k ratings, 943 users, 1,682 movies), ML-1M (about 1M ratings, about 6k users), ML-20M, ML-25M (about 25M ratings, about 162k users, about 62k movies), and the 2024 ML-32M release. TorchRec ships loaders for 20M and 25M with columns `userId, movieId, rating, timestamp` [4]. TFRS and Microsoft Recommenders use 100K in their tutorials and benchmarks [6][18].

**Caveats for designers.** Ratings are explicit and filtered (every user has at least 20 ratings). Users rate in bursts at sign-up, so timestamps poorly reflect consumption order. The data comes from a recommender-mediated site with no exposure logs. MovieLens is good for debugging pipelines and teaching. It is weak as evidence that a sequential or implicit-feedback model will win in production.

### 34.5.2 Criteo (display advertising CTR)

- **Criteo Kaggle / Display Advertising Challenge.** About one week of display-ad click data, with 13 numeric and 26 categorical (hashed) features. The BARS Criteo_x1 version has 45,840,617 samples, split randomly 7:2:1 into 33,003,326 train, 8,250,124 validation and 4,587,167 test rows [40].
- **Criteo 1TB Click Logs.** 24 days with the same 13+26 schema. TorchRec counts 4,195,197,692 training rows across days 0-22 and uses day 23 for validation and test, with a frequency threshold of 3 for remapping rare IDs [4]. This is the dataset behind the DLRM reference implementation and MLPerf recommendation training [37][38]. A multi-hot synthetic extension is used for the newer DLRM-DCNv2 MLPerf benchmark **(approximate / not re-verified)**.

The original DLRM repository [37] documents the pipeline: raw `train.txt`, then a preprocessed `.npz`, then training with `--memory-map` and sub-sampling options for the Terabyte data. DLRM itself is a bottom MLP for dense features plus embedding lookups for sparse features, combined by a dot, sum or concatenation interaction, followed by a top MLP and a sigmoid [37][38].

**Caveats.** Features are anonymized and hashed, so no feature engineering transfers. Criteo AUC gains of 0.001 are conventionally treated as significant, which makes split and preprocessing standardization (BARS) essential [43].

### 34.5.3 Avazu (mobile ads CTR)

Avazu comes from the Kaggle Avazu CTR Prediction competition. It covers about 10 days of mobile-ad click logs with 22 fields (ad, site, app, device attributes and anonymized categorical features). The BARS Avazu_x1 version has 40,428,967 samples, split randomly 7:1:2 into 28,300,276 train, 4,042,897 validation and 8,085,794 test rows [41]. Avazu and Criteo are the standard pair for feature-interaction models (FM, DeepFM, DCN, xDeepFM, AutoInt, FinalMLP) in FuxiCTR [44].

### 34.5.4 Amazon Reviews (McAuley lab, UCSD)

The Amazon reviews corpora have been released in several versions (2013, 2014, 2018, and the 2023 release accompanying BLaIR). They provide user-item-rating-timestamp-review data with rich item metadata (title, description, categories, price, images), split by product category [45]. The 2023 release is reported at roughly 570M reviews and about 48M items across 33 categories, spanning 1996 to 2023 **(approximate / not re-verified)**. Its repository provides scripts that build "standard train/validation/test splits" for benchmarking [45]. The 2023 data is the substrate for text-based and LLM-based recommenders (BLaIR pairs language and items for retrieval) [45]. Older category subsets (Beauty, Sports, Toys, Books, CDs) remain the standard sequential-recommendation benchmarks, and they appear as AmazonBooks_m1 and others in reczoo [42].

### 34.5.5 KuaiRand (Kuaishou, CIKM 2022)

KuaiRand is the key public dataset for **unbiased and off-policy** recommendation research [46]. It is built from Kuaishou short-video logs in which each recommended video had an equal chance of being replaced by a random video from a candidate pool. About 0.37% of final interactions are these random exposures, giving "millions of intervened interactions" [46].

| Version | Users | Items | Interactions | Size |
|---|---|---|---|---|
| KuaiRand-27K | 27,285 | ~32 million | 322,278,385 | ~46 GB |
| KuaiRand-1K | 1,000 (sampled) | ~4.37 million | 11,713,045 | ~4.3 GB |
| KuaiRand-Pure | 27,285 | 7,551 | 1,436,609 | ~194 MB |

Each interaction carries **12 feedback signals** (click, like, follow, comment, forward, hate, long view, view time and others), and the logs span 15 recommendation policies [46]. That makes the dataset suitable for multi-task learning, debiasing (random exposures act as an unbiased test set), and off-policy evaluation. Its sibling KuaiRec provides a nearly fully observed user-item matrix **(not re-verified in this pass)**.

### 34.5.6 MIND (Microsoft News Dataset, ACL 2020)

MIND was collected from anonymized Microsoft News logs over six weeks (12 October to 22 November 2019) [47]. It covers about 1M users (sampled from users with at least 5 news clicks), about 160k English articles and more than 15M impression logs [47][48]. MIND-small has 50,000 users and train/validation splits only. The full dataset trains on week five (with histories from weeks one to four), validates on the last day of week five, and tests on week six [47]. Each article has title, abstract, body, category and Wikidata-linked entities, with 100-dimensional TransE entity embeddings [47].

The design-relevant property is the **impression log**. Each record lists exactly what was shown and which items were clicked, so models can train on true shown-but-not-clicked negatives and be evaluated with per-impression AUC, MRR and nDCG@5/10 [47][48]. Microsoft Recommenders ships DKN, NPA, NAML, LSTUR and NRMS on MIND [17], and Vespa's news tutorial uses it [23].

### 34.5.7 Yambda (Yandex, 2025)

Yambda ("Yandex Music Billion-interactions Dataset") is, as of 2025-2026, the largest open recommendation dataset. All figures below are **approximate / not re-verified**, because Hugging Face and arXiv could not be fetched [49]:

- **Size and versions.** About 4.79 billion user-item interactions from roughly 1M Yandex Music users over about 9.4M tracks, released in 50M, 500M and 5B-event versions.
- **Feedback.** Implicit listens (with played-ratio information) and explicit likes, dislikes, unlikes and undislikes.
- **Flags and content.** An `is_organic` flag distinguishes organic listens from recommendation-driven ones, and precomputed audio embeddings are included for many tracks.
- **Evaluation.** A *global temporal split* (train strictly before a time cutoff, test after), rather than leave-one-out, to mimic production deployment.
- **Baselines.** Popularity variants, ItemKNN, iALS, BPR, SANSA and SASRec.

Yambda matters because it closes the scale gap. Most academic results are on datasets with 10^5-10^8 events, while production systems train on 10^10 or more. The organic/recommended split is a cheap exposure signal similar in spirit to KuaiRand's random-exposure slice. It lets researchers separate "what users chose" from "what the system pushed."

### 34.5.8 Dataset comparison

| Dataset | Domain | Feedback | Negatives / exposure | Scale (approx.) | Typical use |
|---|---|---|---|---|---|
| MovieLens [39] | Movies | Explicit ratings | None | 100K-32M ratings | Teaching, CF baselines |
| Criteo Kaggle / 1TB [4][40] | Display ads | Click label | Shown, non-clicked rows | 45.8M / 4.2B rows | CTR, DLRM, MLPerf |
| Avazu [41] | Mobile ads | Click label | Shown, non-clicked rows | 40.4M rows | CTR feature interaction |
| Amazon Reviews [45] | E-commerce | Ratings + text | None | Up to hundreds of millions of reviews | Sequential, text/LLM recsys |
| KuaiRand [46] | Short video | 12 signals | Random-exposure slice | 322M interactions | Debiasing, OPE, MTL |
| MIND [47][48] | News | Clicks | Impression slates | 15M+ impressions | News recsys, NLP encoders |
| Yambda [49] | Music | Listens + likes/dislikes | Organic flag | ~4.8B events | Scale, temporal-split evaluation |

---

## 34.6 Benchmarking practice

Tooling has converged on a few protocol choices that strongly affect reported numbers:

1. **Split.** Random splits (the BARS Criteo/Avazu convention [40][41]) leak future information. Leave-one-out (common in sequential recsys) ignores global time. Global temporal splits (MIND [47], Yambda [49]) best match deployment. Report which one you used.
2. **Candidate set at evaluation.** Full ranking over all items (RecBole's `full_sort_predict` [16], TFRS `FactorizedTopK` [6]) versus ranking against sampled negatives. Sampled metrics can reorder models, so prefer full ranking or report both.
3. **Frozen preprocessing.** Frequency thresholds (TorchRec uses 3 for Criteo [4]), hashing, and the definition of a "positive" (rating >= 4, play ratio above a threshold) all move metrics. BARS dataset IDs exist to freeze them [42][43].
4. **Strong simple baselines.** The Microsoft Recommenders benchmark shows SAR and BPR within reach of neural models on MovieLens 100K at a fraction of the training time [18]. iALS and ItemKNN sit in the Yambda baseline set for the same reason [49].
5. **Systems metrics.** Throughput, latency at a recall target (ScaNN/TFRS tuning curves [7][31]) and memory footprint belong next to quality metrics. A 0.1% AUC gain that doubles embedding memory is rarely worth it.

```
  Reproducible offline benchmark loop
  raw logs -> frozen preprocessing (versioned ID, e.g. Avazu_x4) -> temporal split
         -> library (RecBole / FuxiCTR / TorchRec) with fixed config + seeds
         -> full-ranking metrics (nDCG@k, Recall@k, AUC/logloss) + latency/memory
         -> compare vs simple baselines (popularity, ItemKNN, iALS, SAR)
         -> only then: online A/B (Chapter 12)
```

---

## 34.7 Key takeaways for system designers

1. **Assemble, don't invent.** A credible production stack can be built entirely from open source: Feast for features, `implicit` or a TFRS/TorchRec two-tower for candidates, FAISS/ScaNN or a vector DB for the index, TorchRec/HugeCTR for the ranker, Triton or Vespa for serving, and Metaflow or Ray for orchestration. The proprietary edge is in data, features and objectives, not in the plumbing.
2. **Embedding tables are the system.** TorchRec's sharding planner, FBGEMM's fused table-batched kernels, HugeCTR's hybrid hot/cold embedding and HierarchicalKV's score-based eviction all solve the same problem: skewed, huge, sparse parameters. Choose a framework by how it shards, caches and evicts embeddings [1][3][10][12].
3. **Pick the ANN layer by filtering and freshness needs, not peak QPS.** Libraries (FAISS, ScaNN) are fastest when you control rebuilds. Engines (Milvus, Qdrant, Weaviate, Vespa) earn their cost when you need filtered queries, streaming inserts and replication [24]-[27][30].
4. **Use MIPS-aware quantization for dot-product towers.** ScaNN's anisotropic quantization targets inner-product ranking error directly. Always add a rescoring stage over a few hundred candidates [31][32].
5. **Compute near the data when candidate sets are large.** Vespa's phased ranking on content nodes shows the payoff of not shipping thousands of candidate features across the network [21][22].
6. **Point-in-time correctness is non-negotiable.** Use a feature store's point-in-time join (or an equivalent) for every training set, and materialize the same definitions to the online store [34].
7. **Package the four-stage pipeline as one versioned DAG.** Merlin Systems' retrieval, filter, rank and sample ensemble is a concrete template that keeps exploration (softmax sampling) explicit [14].
8. **Match the dataset to the claim.** MovieLens for debugging. Criteo/Avazu for CTR feature interaction. KuaiRand for debiasing and off-policy work. MIND for impression-aware news ranking. Amazon 2023 for text/LLM methods. Yambda when scale and temporal realism matter.
9. **Freeze protocols and report systems costs.** Use versioned splits (BARS), full-ranking metrics, temporal splits and strong simple baselines. Report latency and memory alongside accuracy [16][18][43].
10. **Watch maintenance signals.** Several well-known projects have deprecated components (HugeCTR HPS), incomplete APIs (Merlin Models PyTorch) or modest activity (TFRS). Check commit activity and deprecation notes before you build on a component [6][9][13].

---

## References

1. PyTorch / Meta. "TorchRec" GitHub repository and README. Accessed 2026. https://github.com/pytorch/torchrec
2. Ivchenko, D., Van Der Staay, D., Taylor, C., Liu, X., Feng, W., Kindi, R., Sudarshan, A., Sefati, S. "TorchRec: a PyTorch Domain Library for Recommendation Systems." Meta, RecSys 2022. https://doi.org/10.1145/3523227.3547387
3. PyTorch / Meta. "FBGEMM and FBGEMM_GPU" GitHub repository. Accessed 2026. https://github.com/pytorch/FBGEMM
4. PyTorch / Meta. TorchRec dataset loaders `criteo.py` and `movielens.py`. Accessed 2026. https://github.com/pytorch/torchrec/blob/main/torchrec/datasets/criteo.py ; https://github.com/pytorch/torchrec/blob/main/torchrec/datasets/movielens.py
5. PyTorch / Meta. "TorchRec Inference Library" README. Accessed 2026. https://github.com/pytorch/torchrec/blob/main/torchrec/inference/README.md
6. TensorFlow. "TensorFlow Recommenders" GitHub repository. Google, accessed 2026. https://github.com/tensorflow/recommenders
7. TensorFlow. "Efficient serving" tutorial notebook (BruteForce vs ScaNN). Google, accessed 2026. https://github.com/tensorflow/recommenders/blob/main/docs/examples/efficient_serving.ipynb
8. NVIDIA. "NVIDIA Merlin" GitHub repository. Accessed 2026. https://github.com/NVIDIA-Merlin/Merlin
9. NVIDIA. "Merlin HugeCTR" GitHub repository and README (v25.03 deprecation notes). Accessed 2026. https://github.com/NVIDIA-Merlin/HugeCTR
10. Wei, Y. et al. "Merlin HugeCTR: GPU-accelerated Recommender System Training and Inference." NVIDIA, RecSys 2022. https://arxiv.org/abs/2210.08803
11. NVIDIA. "HugeCTR Core Features" documentation. Accessed 2026. https://github.com/NVIDIA-Merlin/HugeCTR/blob/main/docs/source/hugectr_core_features.md
12. NVIDIA. "HierarchicalKV" GitHub repository. Accessed 2026. https://github.com/NVIDIA-Merlin/HierarchicalKV
13. NVIDIA. "Merlin Models" GitHub repository. Accessed 2026. https://github.com/NVIDIA-Merlin/models
14. NVIDIA. "Merlin Systems" GitHub repository. Accessed 2026. https://github.com/NVIDIA-Merlin/systems
15. NVIDIA. "Transformers4Rec" GitHub repository; Moreira, G. de S. P. et al., "Transformers4Rec: Bridging the Gap between NLP and Sequential / Session-Based Recommendation," RecSys 2021. https://github.com/NVIDIA-Merlin/Transformers4Rec ; https://dl.acm.org/doi/10.1145/3460231.3474255
16. RUCAIBox. "RecBole" GitHub repository (RecBole 1.0 CIKM 2021, 2.0 CIKM 2022, SIGIR 2023). Accessed 2026. https://github.com/RUCAIBox/RecBole
17. Recommenders team (Microsoft, Linux Foundation AI & Data). "Recommenders" GitHub repository. Accessed 2026. https://github.com/recommenders-team/recommenders
18. Recommenders team. "MovieLens benchmark" notebook (examples/06_benchmarks). Accessed 2026. https://github.com/recommenders-team/recommenders/blob/main/examples/06_benchmarks/movielens.ipynb
19. Lyst / Kula, M. "LightFM" GitHub repository; Kula, M., "Metadata Embeddings for User and Item Cold-start Recommendations," CBRecSys workshop at RecSys 2015. https://github.com/lyst/lightfm ; https://arxiv.org/abs/1507.08439
20. Frederickson, B. "implicit: Fast Python Collaborative Filtering for Implicit Datasets" GitHub repository. Accessed 2026. https://github.com/benfred/implicit
21. Vespa.ai. "Vespa" GitHub repository and README. Accessed 2026. https://github.com/vespa-engine/vespa
22. Vespa.ai. "Phased Ranking" documentation. 2025-2026. https://docs.vespa.ai/en/ranking/phased-ranking.html
23. Vespa.ai. "Vespa sample applications" (album recommendation, news recommendation on MIND). Accessed 2026. https://github.com/vespa-engine/sample-apps
24. Zilliz / LF AI & Data. "Milvus" GitHub repository. Accessed 2026. https://github.com/milvus-io/milvus
25. Weaviate. "Weaviate" GitHub repository. Accessed 2026. https://github.com/weaviate/weaviate
26. Qdrant. "Qdrant" GitHub repository. Accessed 2026. https://github.com/qdrant/qdrant
27. Meta FAIR. "Faiss" GitHub repository. Accessed 2026. https://github.com/facebookresearch/faiss
28. Douze, M. et al. "The Faiss library." Meta FAIR, 2024. https://arxiv.org/abs/2401.08281
29. Johnson, J., Douze, M., Jégou, H. "Billion-scale similarity search with GPUs." Facebook AI Research, 2017 (IEEE Transactions on Big Data, 2019). https://arxiv.org/abs/1702.08734
30. Google Research. "ScaNN: Scalable Nearest Neighbors" README. Accessed 2026. https://github.com/google-research/google-research/tree/master/scann
31. Google Research. "ScaNN algorithms and configuration" (docs/algorithms.md). Accessed 2026. https://github.com/google-research/google-research/blob/master/scann/docs/algorithms.md
32. Guo, R., Sun, P., Lindgren, E., Geng, Q., Simcha, D., Chern, F., Kumar, S. "Accelerating Large-Scale Inference with Anisotropic Vector Quantization." Google, ICML 2020. https://arxiv.org/abs/1908.10396
33. Sun, P. et al. "SOAR: Improved Indexing for Approximate Nearest Neighbor Search." Google, NeurIPS 2023. https://arxiv.org/abs/2404.00774
34. Feast project. "Feast: Feature Store for Machine Learning" GitHub repository. Accessed 2026. https://github.com/feast-dev/feast
35. Anyscale / Ray project. "Ray" GitHub repository. Accessed 2026. https://github.com/ray-project/ray
36. Netflix / Outerbounds. "Metaflow" GitHub repository. Accessed 2026. https://github.com/Netflix/metaflow
37. Meta (Facebook Research). "DLRM: Deep Learning Recommendation Model" GitHub repository. Accessed 2026. https://github.com/facebookresearch/dlrm ; Mudigere, D. et al., "Software-Hardware Co-design for Fast and Scalable Training of Deep Learning Recommendation Models," ISCA 2022: https://arxiv.org/abs/2104.05158
38. Naumov, M. et al. "Deep Learning Recommendation Model for Personalization and Recommendation Systems." Facebook, 2019. https://arxiv.org/abs/1906.00091
39. Harper, F. M., Konstan, J. A. "The MovieLens Datasets: History and Context." GroupLens, ACM TiiS, 2015. https://doi.org/10.1145/2827872
40. reczoo. "Criteo_x1" dataset card. Accessed 2026. https://github.com/reczoo/Datasets/tree/main/Criteo/Criteo_x1
41. reczoo. "Avazu_x1" dataset card. Accessed 2026. https://github.com/reczoo/Datasets/tree/main/Avazu/Avazu_x1
42. reczoo. "Datasets for recommender systems benchmarking." Accessed 2026. https://github.com/reczoo/Datasets
43. Zhu, J., Dai, Q., Su, L., Ma, R., Liu, J., Cai, G., Xiao, X., Zhang, R. "BARS: Towards Open Benchmarking for Recommender Systems." Huawei Noah's Ark et al., SIGIR 2022. https://github.com/reczoo/BARS ; https://arxiv.org/abs/2205.09626
44. reczoo. "FuxiCTR: A configurable, tunable, and reproducible library for CTR prediction." Accessed 2026. https://github.com/reczoo/FuxiCTR
45. Hou, Y., Li, J., He, Z., Yan, A., Chen, X., McAuley, J. "Bridging Language and Items for Retrieval and Recommendation" (BLaIR; Amazon Reviews 2023). UCSD, 2024. https://github.com/hyp1231/AmazonReviews2023 ; https://arxiv.org/abs/2403.03952
46. Gao, C. et al. "KuaiRand: An Unbiased Sequential Recommendation Dataset with Randomly Exposed Videos." Kuaishou / USTC, CIKM 2022. https://github.com/chongminggao/KuaiRand ; https://arxiv.org/abs/2208.08696
47. Microsoft. "MIND: MIcrosoft News Dataset" homepage and introduction document. 2020. https://msnews.github.io/ ; https://github.com/msnews/msnews.github.io/blob/master/assets/doc/introduction.md
48. Wu, F. et al. "MIND: A Large-scale Dataset for News Recommendation." Microsoft, ACL 2020. https://aclanthology.org/2020.acl-main.331/
49. Yandex. "Yambda-5B: A Large-Scale Multi-modal Dataset for Ranking and Retrieval" (dataset card and paper). 2025. https://huggingface.co/datasets/yandex/yambda ; https://arxiv.org/abs/2505.22238
