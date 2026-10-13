# 09. Training Infrastructure at Scale: Sparse Embeddings, Sharding, Parameter Servers and Online Learning

## Overview

Industrial recommenders differ from most other deep learning workloads in one important way. Nearly all of their parameters sit in **sparse embedding tables** keyed by categorical IDs (users, items, creators, queries, ad IDs, crossed features). Only a small fraction sit in the dense neural network. In the DLRM family of models [1], each categorical feature looks up one or more rows in a table, pools them (sum or mean), and feeds the result into feature interaction and an MLP. Production tables have billions of rows, and their total size runs from hundreds of gigabytes to tens or hundreds of terabytes. Meta's ZionEX/Neo paper describes models of up to 12 trillion parameters [3]. Kuaishou's PERSIA reports a system that scales to 100 trillion parameters [4]. Alibaba's DeepRec claims support for "over ten trillion parameters" [10].

This shape of workload produces a set of system problems that look quite different from training a large language model:

- **Memory capacity, not FLOPs, is the binding constraint.** The tables do not fit on one accelerator, and often not on one host.
- **Embedding lookups are memory-bandwidth and network bound.** Each lookup is a small random read. Each step needs an all-to-all exchange of pooled embeddings.
- **The ID space is open and keeps changing.** New users, items and videos arrive every second, so a fixed-size table either collides (hashing) or has to grow (dynamic hash tables).
- **Freshness matters.** User interest and the item corpus drift within hours, which pushes systems toward online or continual training and toward frequent parameter sync to serving.
- **Data access is very skewed.** A small fraction of IDs accounts for most lookups, which makes caching, tiered storage and frequency-based admission effective.

This chapter covers the main publicly documented designs for these problems:

- Meta's DLRM, Neo/ZionEX, TorchRec and FBGEMM.
- ByteDance's Monolith.
- NVIDIA's HugeCTR, Merlin and HierarchicalKV.
- Parameter-server systems: Alibaba XDL and DeepRec, Kuaishou PERSIA, HET, and 4Paradigm OpenEmbedding.
- TensorFlow Recommenders-Addons, which Tencent and others contribute to.
- Checkpointing for continual training.
- Embedding compression: the hashing trick, hash embeddings, QR compositional embeddings, mixed-dimension embeddings, TT-Rec, ROBE-style and DHE-style methods.
- Hardware directions such as Google's TPU SparseCore.

A note on sourcing: the research for this chapter hit a web-search budget limit and DNS failures on several sites (arxiv.org, pytorch.org, developer.nvidia.com and engineering.fb.com could not be fetched). The facts here come from the project repositories that could be read (GitHub READMEs and docs) and from the primary papers cited by ID. Where a number could not be re-verified against a fetched page, it is marked approximate.

## 1. The Shape of the Problem: DLRM as Reference Workload

### 1.1 Model anatomy

DLRM [1] is the reference model that most training-infrastructure papers assume. Dense features go through a bottom MLP. Each sparse feature indexes an embedding table. A pairwise interaction operator (dot product, or concatenation, or sum) combines the dense output with the pooled embeddings, and a top MLP predicts a click probability [11]. The open-source DLRM code supports synchronous distributed training over gloo, NCCL and MPI backends. It also includes MLPerf benchmark tooling with AUC-based early stopping [11].

```
            dense features            sparse features (multi-hot IDs)
                 |                      |        |        |
            +---------+             +------+ +------+ +------+
            | bottom  |             | EMB  | | EMB  | | EMB  |   <- 99%+ of params
            |  MLP    |             | T1   | | T2   | | Tk   |      (sharded)
            +---------+             +------+ +------+ +------+
                 |                      | pool   | pool   | pool
                 +----------+-----------+--------+--------+
                            |  feature interaction (dot / cat)
                       +---------+
                       | top MLP |   <- small, data-parallel
                       +---------+
                            |
                         p(click)
```

### 1.2 Why it is hard to train efficiently

Meta's study "Understanding Training Efficiency of Deep Learning Recommendation Models at Scale" [12] characterized production DLRM training. Its main points:

- Embedding operators are memory-bound and dominated by communication.
- The large embedding tables and the varied sparse-feature profiles across models make one hardware configuration a poor fit for all models.
- In the CPU parameter-server era, trainers spent much of each step waiting on embedding pulls and pushes.

The paper argued for co-designing hardware and software around embedding traffic. That argument led to the Zion and ZionEX GPU platforms [3].

Two kinds of parallelism are the standard solution:

- **Model parallelism for embeddings.** Tables, or slices of them, are placed on different devices. Each device does the lookups for the IDs it owns.
- **Data parallelism for the dense MLPs.** The network is replicated and gradients are all-reduced.

The two worlds meet at an **all-to-all** collective. In the forward pass it moves pooled embeddings from the devices that own the tables to the devices that own the samples. In the backward pass it moves the gradients the other way.

```
 forward:  sample-sharded batch -> [input dist: all-to-all of IDs]
           -> table-sharded lookup + pooling
           -> [output dist: all-to-all of pooled embeddings]
           -> data-parallel interaction + MLP on each GPU
 backward: MLP grads -> all-reduce (dense)
           pooled-emb grads -> all-to-all back to owners
           -> fused sparse optimizer update in place on owner GPU
```

## 2. Meta: Neo, ZionEX, TorchRec and FBGEMM

### 2.1 Problem

Before 2020, Meta trained recommendation models asynchronously on CPU clusters with parameter servers. That approach scaled in capacity, but it had two drawbacks. Asynchrony hurt model quality and reproducibility. Throughput was also limited by network traffic to the parameter servers [12]. The goal of Neo/ZionEX [3] was synchronous, GPU-based training of models with trillions of parameters.

### 2.2 Architecture

**ZionEX hardware** [3]:

- Each node has 8 GPUs (A100-class in the paper) connected internally by NVLink.
- Each node also has a dedicated RDMA over Converged Ethernet (RoCE) **scale-out network**: several high-bandwidth NICs per node, so every GPU has its own path for inter-node all-to-all traffic.
- The paper's evaluation covers clusters of up to 16 nodes (128 GPUs).
- The emphasis is on all-to-all bandwidth, because embedding exchange dominates the step time.

**Neo software** [3] introduced **4D parallelism** for embeddings, chosen per table:

| Sharding scheme | What is split | When it fits |
|---|---|---|
| Table-wise (TW) | Whole tables placed on different GPUs | Many small/medium tables; low comm |
| Row-wise (RW) | Rows of one table split across GPUs | Huge tables beyond one GPU's HBM |
| Column-wise (CW) | Embedding dimension split across GPUs | Wide tables; balances load per GPU |
| Data-parallel (DP) | Table replicated everywhere | Tiny tables; avoids all-to-all entirely |

Neo also contributed the following [3]:

- **Hybrid kernel fusion.** All tables on a device are batched into one fused kernel, and the sparse optimizer update is fused into the backward pass, so no full gradient tensor is ever materialized.
- **A sharding planner** that balances compute, memory and communication cost across devices.
- **Software-managed caching** of embedding rows across HBM, host DRAM and UVM.
- **Quantized collectives.** Embedding all-to-all traffic is sent in reduced precision (FP16/BF16) to save network bandwidth, while model quality is preserved.

The paper reports training models of up to **12 trillion parameters**. It also reports roughly order-of-magnitude (approximately up to 40x, per the paper's headline claim) time-to-solution speedups over Meta's earlier CPU-based asynchronous training [3].

### 2.3 TorchRec and FBGEMM_GPU

TorchRec [2][13] is the open-source productionization of these ideas as a PyTorch domain library. According to its README [13], it provides:

- Sharders for **data-parallel, table-wise, row-wise, table-wise-row-wise, column-wise and table-wise-column-wise** sharding.
- A **planner** that automatically produces an optimized sharding plan.
- RecSys kernels from **FBGEMM** [14].
- A **train pipeline** that overlaps data loading, host-to-device copies, input distribution (all-to-all of IDs) and forward/backward compute.
- Reduced-precision training and inference, with C++ inference export.

The project states that TorchRec "powers many production RecSys models at Meta." Its citation abstract describes how Meta moved from CPU asynchronous training to accelerator-based full-sync training with it [13]. The README also links later work on **"Toward 100TB model with Embedding Offloading"** [13], which keeps hot rows in HBM and offloads cold rows to host or SSD tiers.

FBGEMM_GPU supplies the kernels [14]. Its **table-batched embedding (TBE)** operator runs many embedding bags in one launch, and it also supports quantized embeddings and data-layout transforms. Two ideas carry over to any design here:

1. **Table batching and fused optimizers.** Thousands of small kernel launches become one. The optimizer state (for example rowwise Adagrad) is updated in the same pass as the backward lookup.
2. **Pipelining across stages.** The ID all-to-all for batch i+1 overlaps with the dense compute for batch i.

```
TorchRec TrainPipelineSparseDist (conceptual)
 time ->
 batch i   : [H2D copy][ID a2a][lookup][emb a2a][fwd/bwd MLP][emb grad a2a+update]
 batch i+1 :           [H2D copy][ID a2a]......(overlapped with batch i compute)
 batch i+2 :                     [H2D copy]....
```

### 2.4 Lessons

- **Synchronous training at scale is achievable.** It is preferable for reproducibility, but it needs dedicated all-to-all bandwidth [3].
- **Use a heterogeneous sharding plan.** No single scheme is best. The planner choosing per table (TW for most tables, RW for the huge ones, DP for the tiny ones) is where much of the performance comes from [3][13].
- **Kernel fusion and fused optimizers** avoid materializing dense gradients for sparse tables.

## 3. ByteDance Monolith: Collisionless Tables and Real-Time Training

### 3.1 Problem

ByteDance's feed products (Douyin/TikTok and others) have three properties that clash with standard TensorFlow training:

- The ID space is **non-stationary and unbounded**.
- Feature popularity is **long-tailed**.
- **Concept drift** happens within hours.

TensorFlow's built-in variables are fixed-size dense tensors. Fitting unbounded IDs into them means hashing, and hashing collisions degrade quality. Daily batch retraining is also too slow to keep up with drift [5][15].

### 3.2 Collisionless embedding table

Monolith [5] replaces fixed `tf.Variable` tables with a **key-value hash table on the parameter servers**, built on **Cuckoo hashing**. Cuckoo hashing uses two hash functions and two arrays. An insert that hits an occupied slot evicts the occupant to its other location, and this repeats until every key has a slot. The result is guaranteed O(1) lookup and no collisions between distinct IDs [5]. The repository describes the goal directly: "each ID feature gets its own unique representation" [15].

Unbounded growth is controlled with two mechanisms [5]:

- **Frequency filtering (admission).** An ID gets an embedding only after it has appeared some threshold number of times. A probabilistic filter keeps the memory needed to count rare IDs low. Long-tail IDs seen once or twice add memory cost without adding quality.
- **Expirable embeddings (eviction).** Each feature has an expiry time. IDs that go inactive, such as stale users or deleted videos, are evicted.

### 3.3 Real-time (online) training pipeline

```
 user actions (clicks, plays, ...) ---> Kafka (action log)
 serving features ------------------>  Kafka (feature log)
                                           |
                                    Flink online joiner
                     (join by request key, in-memory cache + on-disk KV
                      for late actions; negative sampling)
                                           |
                     +---------------------+-------------------+
                     |                                         |
           Kafka (training examples)                   HDFS (batch dumps)
                     |                                         |
            online training workers  <--- same model --->  batch training
                     |
            training parameter servers (Cuckoo hash tables)
                     |  periodic sync: sparse incremental (minutes), dense (slower)
                     v
            serving parameter servers -> online serving
```

How the loop works, per the paper [5]:

1. **One framework, two modes.** Batch training runs on historical HDFS data to bootstrap a model. Online training then consumes the streaming Kafka example queue indefinitely.
2. **Online joiner.** A Flink job joins user actions with the features logged at serving time. Features are cached because the corresponding action can arrive minutes later. Negative sampling reduces the share of negatives, and a **log-odds correction** at serving time compensates for the changed label distribution.
3. **Parameter synchronization.** Moving terabytes of parameters to serving every minute is impractical. Monolith therefore makes the update asymmetric:
   - **Sparse parameters** are synced **incrementally**, sending only the IDs that changed since the last sync, at minute-level intervals.
   - **Dense parameters** change slowly under momentum-style optimizers, so they are synced much less often (the paper describes a daily-scale cadence).
4. **Fault tolerance.** The paper studies how snapshot frequency affects model quality. It concludes that **daily snapshots** of parameter servers lose very little quality on PS failure, because one shard holds only a small fraction of IDs and the model recovers quickly with online training. Infrequent snapshots save large amounts of compute and I/O [5].

### 3.4 Results and lessons

The Monolith paper reports [5]:

- **Collisionless** embeddings beat hashed (colliding) baselines on both MovieLens-style public data and internal production data.
- **Shorter parameter-sync intervals** (moving from hours toward ~30 minutes in their experiments) give consistently higher online AUC. Real-time training measurably beats batch training for ads.

Lessons:

- Freshness is a model-quality feature, not only an operations concern.
- The **asymmetric sync** of sparse and dense parameters is the key trick that makes minute-level updates affordable.
- Admission and eviction policies matter as much as the table data structure.

The Monolith GitHub repository is now archived and read-only. It is built on TensorFlow with Bazel [15]. The same ideas carried over to BytePlus's commercial recommendation offering.

## 4. Parameter-Server Lineage: XDL, DeepRec, PERSIA, HET, OpenEmbedding, TFRA

### 4.1 Alibaba XDL and DeepRec

**XDL (X-DeepLearning)** [16] is Alibaba Mama's framework for high-dimensional sparse data in advertising, recommendation and search. It came with:

- the **Blaze** serving engine,
- the **TDM** (tree-based deep matching) retrieval engine.

The XDL paper (DLP-KDD 2019) describes a "bridge" design. XDL handles the high-dimensional sparse part on a distributed parameter server, and the dense sub-network runs on a pluggable backend (TensorFlow/MXNet).

Version 1.2 added automatic global parameter allocation, request merging, and removal of PS compute, storage and communication hotspots. It reports 50–100% gains for large-batch, low-concurrency workloads [16]. Its **streaming-training features** are the same set that later frameworks standardized on [16]:

- **feature admission**,
- **feature eviction**,
- **incremental model export**,
- **feature frequency statistics**.

**DeepRec** [10] is Alibaba's successor: a heavily modified TensorFlow. Its features include:

- **EmbeddingVariable**, a dynamic hash-backed embedding, in dynamic-dimension, adaptive and multi-hash variants, plus group embedding.
- **Multi-tier hybrid embedding storage** across HBM, DRAM, PMem and SSD.
- **Delta (incremental) checkpoint** loading and export.
- Two training modes:
  - **Asynchronous PS training** over gRPC+Seastar, FuseRecv and StarServer.
  - **Synchronous collective training** via HybridBackend and NVIDIA's Sparse Operation Kit.
- Graph and runtime optimizations: SmartStage pipelining, AutoGraphFusion, MicroBatch, a graph-aware memory allocator, and CUDA Graph support.

DeepRec claims support for "trillion samples and over ten trillion parameters" [10].

### 4.2 Kuaishou PERSIA: hybrid synchronous/asynchronous training

**Problem.** Fully synchronous training is slow, because embedding communication is large. Fully asynchronous training hurts the accuracy of the dense network. PERSIA [4][17] (Kuaishou with ETH Zurich, KDD 2022) observes that the two parts behave differently:

- The **embedding layer** is huge, but each step touches very few rows. Staleness on sparse rows is therefore rare and harmless.
- The **dense network** is small, but every step updates all of it, so staleness there does hurt.

**Design.** PERSIA trains the embeddings **asynchronously** and the dense NN **synchronously** (all-reduce, via the Bagua communication library). The implementation is in Rust and Python. Its components:

```
   data loaders --> embedding workers <--> embedding parameter servers (CPU, huge)
                          |   (async: get/put embeddings, stale-tolerant)
                          v
                  NN workers (GPU) -- sync all-reduce of dense grads (Bagua)
```

**Scale.** PERSIA reports training models of up to **100 trillion parameters**, described at the time as the largest recommendation model reported [17]. It ran in production at Kuaishou for applications with roughly hundreds of millions of daily active users. The paper reports several-fold throughput gains (approximately up to 7x) over contemporary systems at similar accuracy [4]. The project README now notes that PERSIA is unmaintained after a reorganization [17].

**Lesson.** Matching the consistency model to each parameter's access pattern gives most of the speed of asynchrony without most of its accuracy cost. TFRA uses the same idea (section 4.4).

### 4.3 HET: cache-enabled embedding training

HET [18] (Peking University, VLDB 2022) targets the PS communication bottleneck by keeping an **embedding cache on each worker**. Because accesses are skewed, hot embeddings are cached locally. A **bounded-staleness consistency** protocol, with per-embedding clocks, decides when a cached copy must be refreshed or written back. This cuts PS traffic sharply while keeping convergence close to synchronous training. The design point to note is that caching combined with **explicit staleness bounds** is a principled middle ground between fully synchronous and fully asynchronous training.

### 4.4 TensorFlow Recommenders-Addons (TFRA)

TFRA [19] is the SIG-Recommenders project that adds a trainable **dynamic embedding** to TensorFlow. It is a key-value structure with no hash conflicts, and it works with native TF optimizers, initializers, checkpoints and SavedModel. Its storage backends are pluggable:

- **cuckoohash_map** (from CMU) on CPU,
- NVIDIA **HierarchicalKV** on GPU,
- **Redis**, for distributed inference.

TFRA also supports **half-synchronous training with Horovod**: synchronous for dense weights, asynchronous for sparse weights. Listed contributors and supporters include Tencent, Alibaba, VIP.com and BOSS Zhipin, plus the NVIDIA Merlin team [19].

### 4.5 4Paradigm OpenEmbedding

OpenEmbedding [20] keeps embedding layers on a parameter server and trains dense layers with Horovod. It exports standalone SavedModels for TF Serving. It also has an experimental **persistent-memory (PMem) PS**, where PMem acts as a lightweight checkpointing mechanism (see the associated ICDE 2023 paper). It reports 3–8x speedups over pure all-reduce (Horovod) on a single 8-GPU machine for sparse models [20].

## 5. NVIDIA HugeCTR, Merlin and HierarchicalKV

### 5.1 Problem

The goal is to keep the whole training loop on GPUs: data reading, embedding lookup, communication and dense compute. The system should still handle tables larger than aggregate HBM, and serve them with low latency [6][21].

### 5.2 Training architecture

HugeCTR [6][21] is a CUDA/C++ framework with a Python front end. Its embedding layers are model-parallel across GPUs and nodes, while the dense model is data-parallel. Inter-node communication uses NCCL with GPUDirect RDMA [21]. Its main embedding types:

- **LocalizedSlotSparseEmbeddingHash.** Each feature slot lives on one GPU, with no global reduction. This suits embeddings that fit on a single GPU [21].
- **DistributedSlotSparseEmbeddingHash.** Keys are spread across GPUs regardless of slot, so a single slot can span devices. This suits tables larger than one GPU at the cost of more communication [21].
- **Hybrid embedding** (described in the RecSys 2022 paper [6]). Hot categories are replicated data-parallel and cold categories are model-parallel. Frequent IDs then need no all-to-all, which was a key optimization in NVIDIA's MLPerf DLRM submissions.

Other features [21]:

- Mixed precision, with FP16 used on embeddings to save bandwidth and loss scaling to prevent underflow.
- Learning-rate warmup and decay schedules.
- HDFS model I/O.
- A `hugectr2onnx` converter.

Two nodes with 8x A100 80 GB each can train models of up to about **1 TB entirely in GPU memory** [21].

For TensorFlow users, NVIDIA ships the **Sparse Operation Kit (SOK)**, a model-parallel GPU embedding layer [21]. Alibaba DeepRec uses SOK for its synchronous mode [10].

### 5.3 Beyond HBM: ETC, HPS and HierarchicalKV

- **Embedding Training Cache (ETC)** [6] trains models larger than HBM by keeping a working set in GPU memory. The full table lives in host or SSD storage, and the working set is swapped in per data pass.
- **Hierarchical Parameter Server (HPS)** [6] is the inference counterpart:

  ```
   request -> GPU embedding cache (hot, HBM)
           -> miss -> CPU-memory volatile DB (e.g., Redis cluster / hash maps)
           -> miss -> persistent DB (e.g., RocksDB on SSD)
   online-training updates streamed (e.g., via Kafka) into the tiers
  ```

  HPS and the training cache are listed as **deprecated as of v25.03** in the current HugeCTR repository [21]. NVIDIA's investment has moved toward the general-purpose library below and toward framework plugins.

- **HierarchicalKV (HKV)** [22] is a header-only GPU hash table:
  - Keys always reside in HBM. Values can spill to host memory.
  - Buckets are locally ordered (128 slots by default), and the default maximum load factor is 0.5 before rehash.
  - Eviction happens only when a bucket is full, using a per-key 64-bit **score**. Built-in policies: LRU (device clock), LFU, EpochLRU, EpochLFU, or fully custom scores.

  HKV published benchmarks on a single A100 80 GB [22]:

  | Configuration | insert_or_assign (B KV/s) | find (B KV/s) |
  |---|---|---|
  | Pure HBM, dim 8, 128M keys, load 0.5 | ~1.09 | ~2.47 |
  | Pure HBM, dim 64, 64M keys, load 0.5 | ~0.83 | ~1.98 |
  | HBM+host, dim 64, 512M keys (32 GB HBM + 96 GB host) | ~0.049 | ~0.069 |

  The pointer-returning `find*` variant stays at about 3.5–4 billion KV/s even in hybrid mode, because it avoids copying values [22]. HKV is the GPU backend for TFRA's dynamic embedding [19]. It is effectively the GPU counterpart of Monolith's CPU Cuckoo tables.

### 5.4 Lessons

- **Replicate the hot head, shard the long tail.** Hybrid embedding exploits frequency skew to remove most all-to-all traffic for the most common IDs [6].
- **Put score-based eviction inside the hash table** (HKV), so that admission and eviction policy is a pluggable function rather than a separate service [22].
- **Use a tiered store with a GPU cache** for inference. Skew makes high hit rates possible, but cache consistency with online training must be designed in from the start [6].

## 6. Continual / Incremental Training and Checkpointing

### 6.1 Patterns

Across Monolith, XDL, DeepRec, TFRA and HugeCTR the same continual-training pattern recurs:

```
  stream (Kafka) -> joiner (Flink/Storm-like) -> online trainer
       |                                            |
       |                          incremental export (delta of touched IDs)
       v                                            v
  offline log (HDFS) -> periodic full retrain   serving PS / embedding cache
                         (bootstrap / recovery)
```

Its components:

- **Warm start plus streaming.** A batch model trained on history is the starting point, and streaming updates follow.
- **Delta export.** Only rows touched since the last export are shipped. Dense weights are shipped less often [5][10][16].
- **Admission and eviction** keep table size bounded as the ID space churns [5][16][22].
- **Sampling corrections** (for example log-odds correction under negative downsampling) keep calibrated scores consistent across training and serving [5].

### 6.2 Check-N-Run: checkpointing trillion-parameter tables

Training jobs on terabyte-scale models must checkpoint often to survive failures, but full checkpoints are huge. Meta's **Check-N-Run** [9] (NSDI 2022) uses two techniques:

1. **Differential checkpointing.** In any interval only a small fraction of embedding rows change, so only the modified rows are written. This is tracked per row.
2. **Checkpoint quantization.** Rows are written in reduced precision (for example 4-bit or 8-bit with adaptive schemes). The paper shows this is acceptable for recovery when applied with care, for example by limiting how many consecutive quantized restores are allowed.

The paper reports roughly 6–17x reductions in checkpoint write bandwidth and roughly 2.5–8x reductions in storage capacity (approximate, from the paper's abstract) with no meaningful accuracy loss [9]. The idea is the same as Monolith's incremental sparse sync: **write only what changed, and write it compactly**.

## 7. Hashing Tricks and Embedding Compression

When a table cannot hold one row per ID, or when memory, bandwidth or serving cost matters more than the last fraction of a percent of quality, compression techniques apply. They form a spectrum from "fewer rows" to "no table at all".

### 7.1 The hashing trick

Weinberger et al. [7] introduced **feature hashing**: map a huge or unbounded feature space into a fixed number of buckets with a hash function, and use a sign hash to make inner products unbiased. Applied to embeddings, an ID is mapped to `row = h(id) mod M`. This is simple and stateless and handles unseen IDs, but distinct IDs collide. Monolith's motivation was precisely that collisions measurably hurt ranking quality at ByteDance scale [5].

**Hash embeddings** [23] (Svenstrup et al., NeurIPS 2017) reduce the damage from collisions:

- Each token picks k vectors from a shared pool using k hash functions.
- It combines them with **learned per-token importance weights**.
- Two colliding tokens are then unlikely to share the same combination.

### 7.2 Compositional embeddings: the QR trick

Shi et al. [8] (Meta, KDD 2020) proposed **compositional embeddings using complementary partitions**. In the quotient-remainder (QR) case, an ID `x` uses two smaller tables:

```
 q = x // m      -> row in quotient table   (size ceil(|S|/m) x d)
 r = x  % m      -> row in remainder table  (size m x d)
 emb(x) = combine(Wq[q], Wr[r])     combine in {multiply, add, concat}
```

Every ID still gets a **unique** embedding, because the pair (q, r) is unique. Memory, however, drops from |S| x d to about (|S|/m + m) x d. With m near sqrt(|S|) this is O(sqrt(|S|)) rows. The paper generalizes the idea to multiple complementary partitions and to path-based composition. It shows large memory reductions on Criteo-style DLRM with small accuracy loss [8]. The open-source DLRM cites this work as the "reduce the number of vectors" approach [11].

### 7.3 Mixed-dimension embeddings

Ginart et al. [24] observe that popular IDs need more capacity than rare ones. They assign **embedding dimensions in proportion to frequency (popularity)**, with blocks of rows at different widths, and project each block to a common dimension. This is the "reduce vector dimensions" approach cited by DLRM [11]. In practice it composes well with QR, with low-precision storage, and with frequency admission.

### 7.4 TT-Rec: tensor-train decomposition

TT-Rec [25][26] (Meta, MLSys 2021) factorizes each large table into a sequence of small **Tensor-Train cores**. The table shape is factored as products of `p_shapes` x `q_shapes` with TT ranks. Only the requested rows are reconstructed on the fly. According to the FBTT-Embedding repository [26]:

- It is a drop-in replacement for `EmbeddingBag`, with a fused-backward kernel for SGD and Adagrad.
- It achieves **up to 100x smaller models** on Meta's open-source DLRM with the same model quality, and up to 10,000x lower runtime memory per table.
- A **32-way set-associative LFU software cache** keeps uncompressed hot rows, so the frequent IDs skip TT reconstruction.

This is the same skew-exploiting pattern as hybrid embedding and HKV: compress the long tail, and keep the head in full precision.

### 7.5 Hashing-based "shared memory" compression (ROBE-style)

Random-offset block embedding methods (ROBE, MLSys 2022) push hashing further. All tables share **one small parameter array**, and each embedding is read as a contiguous block at a hashed offset. That keeps memory access cache-friendly while allowing extreme compression ratios. These methods are surveyed widely, but the primary paper could not be fetched for this chapter, so specific ratios are omitted here.

### 7.6 Deep Hash Embedding: no table at all

Google's **DHE** [27] (KDD 2021) removes the table entirely. Each ID is encoded by many (on the order of 1,000) **dense hash functions** into a real-valued vector. A **deep MLP** then decodes that vector into an embedding. The parameter count is independent of the vocabulary, every ID gets a unique code, and unseen IDs are handled naturally. The cost moves from memory to compute, which makes DHE attractive on compute-rich, memory-poor accelerators. The trade-off is that serving cost per lookup rises.

### 7.7 Comparison

| Technique | Unique per ID? | Memory scaling | Extra compute | Handles new IDs | Typical use |
|---|---|---|---|---|---|
| Full table (dense array) | Yes (if pre-indexed) | O(N d) | None | Needs re-indexing | Small/stable vocabularies |
| Dynamic hash table (Monolith Cuckoo, HKV, TFRA, DeepRec EV) | Yes | O(active N d) | Hashing | Yes, with admission | Streaming, open vocab |
| Hashing trick [7] | No (collisions) | O(M d), M fixed | Hash | Yes | Cheap baseline, long tail |
| Hash embeddings [23] | Mostly | O(B d + N k) | k lookups + weights | Yes | NLP / long-tail IDs |
| QR compositional [8] | Yes | O((N/m + m) d) | 2 lookups + op | Within range | Huge ID tables |
| Mixed dimension [24] | Yes | Sum over blocks of N_i d_i | Projection | Needs re-indexing | Skewed popularity |
| TT-Rec [25][26] | Yes | O(sum of core sizes) | TT reconstruct | Within shape | 10–100x compression |
| ROBE-style shared array | No (shared) | O(M), one array | Hash + block read | Yes | Extreme compression |
| DHE [27] | Yes | O(MLP params) | Deep MLP per ID | Yes | Memory-bound accelerators |
| Low-precision rows (FP16/INT8/INT4) | Yes | O(N d bits) | Dequantize | n/a | Serving, checkpoints [9] |

## 8. Hardware Direction: Accelerating Sparse Lookups

Google's TPU v4 [28] includes **SparseCores**. These are dataflow processors dedicated to embedding lookups and the related all-to-all traffic. The paper reports that they accelerate embedding-heavy models by about 5–7x while using only about 5% of die area and power (approximate figures from the TPU v4 paper's abstract) [28]. Together with ZionEX's dedicated RoCE fabric [3], this shows the industry treating embedding traffic as a first-class hardware concern rather than a software afterthought.

Newer **generative recommenders** shift the balance. Meta's HSTU work [29] (ICML 2024) reports sequential transduction models of up to **1.5 trillion parameters**, with much more dense compute per example (long user sequences and attention over jagged tensors). The embedding tables are still huge, but these models move ranking training closer to LLM training in compute intensity. The TorchRec and FBGEMM stack (jagged tensors, sharded embeddings, offloading [13]) therefore has to serve both regimes at once. See Chapter 06 for the modeling side.

## 9. Cross-System Comparison

| System (org) | Sparse storage | Consistency | Hardware | Online/continual | Notable scale claim |
|---|---|---|---|---|---|
| Neo / ZionEX (Meta) [3] | Sharded HBM tables + caching | Fully sync | GPU, RoCE fabric | Batch-oriented | Up to 12T params |
| TorchRec + FBGEMM (Meta) [2][13][14] | TW/RW/CW/DP sharded, TBE kernels, offload | Sync | GPU | Via pipelines | 100TB offloading work |
| Monolith (ByteDance) [5][15] | Cuckoo hash on PS, filter + expiry | Async PS (TF) | CPU PS + workers | Yes, minute-level sync | Production feeds/ads |
| PERSIA (Kuaishou/ETH) [4][17] | CPU embedding PS | Hybrid: async emb, sync dense | GPU NN + CPU PS | Supported | Up to 100T params |
| XDL (Alibaba) [16] | Distributed PS | Async PS | CPU/GPU | Admission, eviction, incremental export | Ads/search/rec |
| DeepRec (Alibaba) [10] | EmbeddingVariable, multi-tier | Async PS or sync collective | CPU/GPU | Delta checkpoints | ">10T params" |
| HugeCTR / SOK (NVIDIA) [6][21] | GPU hash, hybrid embedding | Sync | Multi-node GPU | Kafka into HPS | ~1 TB on 2x8 A100 |
| HierarchicalKV (NVIDIA) [22] | GPU hash, HBM+host, scored eviction | Library | GPU | Eviction policies | ~1–4 B KV/s on A100 |
| TFRA (SIG-Rec) [19] | Dynamic embedding (cuckoo/HKV/Redis) | Half-sync | CPU/GPU | Yes | Tencent et al. |
| HET (PKU) [18] | Worker cache + PS | Bounded staleness | CPU/GPU | n/a | Comm reduction |
| OpenEmbedding (4Paradigm) [20] | PS (DRAM/PMem) | Sync dense + PS | GPU + PS | PMem checkpoint | 3–8x vs Horovod |

## 10. Key Takeaways for System Designers

1. **Design around the embeddings.** For DLRM-style models, the capacity, bandwidth and communication of embeddings set the cost. Size the cluster by table bytes and all-to-all bandwidth first, and by FLOPs second [3][12].
2. **Use model parallelism for sparse parameters and data parallelism for dense parameters.** Pick the sharding scheme per table: table-wise, row-wise, column-wise or replicated. Use a cost-model planner rather than hand placement [3][13].
3. **Exploit skew everywhere.** Replicate or cache the hot head (hybrid embedding, HET caches, TT-Rec LFU cache, HPS GPU cache) and shard, compress or offload the tail [6][18][26].
4. **Use dynamic hash tables when the vocabulary is open.** Collision-free embeddings with frequency admission and time- or score-based eviction (Monolith, HKV, TFRA, DeepRec) beat fixed hashing in quality for fast-churning ID spaces [5][19][22].
5. **Match consistency to access pattern.** Sparse rows tolerate staleness, and dense weights do not. Hybrid sync/async (PERSIA, TFRA half-sync) or bounded staleness (HET) captures most of the throughput of asynchrony [4][18][19].
6. **Freshness pays.** Streaming training plus incremental sync of sparse parameters at minute-level, with dense weights synced less often, gives measurable AUC gains in production [5].
7. **Make everything incremental.** Delta exports, delta checkpoints and quantized checkpoints keep trillion-parameter continual training affordable and recoverable [5][9][10].
8. **Fuse kernels and pipeline aggressively.** Table-batched lookups, fused sparse optimizers, and overlap of input all-to-all with compute are table stakes [13][14].
9. **Compression is a spectrum.** Choose by constraint:
   - QR or mixed dimensions for memory with near-zero quality loss [8][24],
   - TT-Rec for 10–100x [26],
   - DHE when memory is scarce and compute is cheap [27],
   - quantization for serving and checkpoints [9].
10. **Co-design hardware.** Dedicated scale-out networks (ZionEX) and on-chip sparse engines (TPU SparseCore) show that embedding traffic justifies specialized silicon and fabric [3][28].

## References

1. Naumov, M. et al. "Deep Learning Recommendation Model for Personalization and Recommendation Systems." Meta (Facebook), 2019. https://arxiv.org/abs/1906.00091
2. Ivchenko, D. et al. "TorchRec: a PyTorch Domain Library for Recommendation Systems." Meta, RecSys 2022. https://doi.org/10.1145/3523227.3547387
3. Mudigere, D. et al. "Software-Hardware Co-design for Fast and Scalable Training of Deep Learning Recommendation Models" (Neo / ZionEX). Meta, ISCA 2022 (arXiv 2021). https://arxiv.org/abs/2104.05158
4. Lian, X., Yuan, B., Zhu, X. et al. "Persia: A Hybrid System Scaling Deep Learning Based Recommenders up to 100 Trillion Parameters." Kuaishou and ETH Zurich, 2021 (KDD 2022). https://arxiv.org/abs/2111.05897
5. Liu, Z. et al. "Monolith: Real Time Recommendation System With Collisionless Embedding Table." ByteDance, 2022. https://arxiv.org/abs/2209.07663
6. Wei, Y. et al. "Merlin HugeCTR: GPU-accelerated Recommender System Training and Inference." NVIDIA, RecSys 2022. https://arxiv.org/abs/2210.08803
7. Weinberger, K. et al. "Feature Hashing for Large Scale Multitask Learning." Yahoo! Research, ICML 2009. https://arxiv.org/abs/0902.2206
8. Shi, H.-J. M. et al. "Compositional Embeddings Using Complementary Partitions for Memory-Efficient Recommendation Systems." Meta, KDD 2020 (arXiv 2019). https://arxiv.org/abs/1909.02107
9. Eisenman, A. et al. "Check-N-Run: A Checkpointing System for Training Deep Learning Recommendation Models." Meta, NSDI 2022 (arXiv 2020). https://arxiv.org/abs/2010.08679
10. Alibaba. "DeepRec" (GitHub repository and README). Alibaba, accessed 2026. https://github.com/alibaba/DeepRec
11. Meta Research. "DLRM: Deep Learning Recommendation Model" (GitHub repository and README). Meta, accessed 2026. https://github.com/facebookresearch/dlrm
12. Acun, B. et al. "Understanding Training Efficiency of Deep Learning Recommendation Models at Scale." Meta, HPCA 2021 (arXiv 2020). https://arxiv.org/abs/2011.05497
13. PyTorch. "TorchRec" (GitHub repository and README). Meta / PyTorch, accessed 2026. https://github.com/pytorch/torchrec
14. PyTorch. "FBGEMM / FBGEMM_GPU" (GitHub repository and README). Meta / PyTorch, accessed 2026. https://github.com/pytorch/FBGEMM
15. ByteDance. "Monolith" (GitHub repository, archived). ByteDance, accessed 2026. https://github.com/bytedance/monolith
16. Alibaba. "X-DeepLearning (XDL)" (GitHub repository and README). Alibaba Mama, accessed 2026. https://github.com/alibaba/x-deeplearning
17. PersiaML. "PERSIA" (GitHub repository and README). Kuaishou and ETH Zurich, accessed 2026. https://github.com/PersiaML/PERSIA
18. Miao, X. et al. "HET: Scaling out Huge Embedding Model Training via Cache-enabled Distributed Framework." Peking University, VLDB 2022 (arXiv 2021). https://arxiv.org/abs/2112.07221
19. TensorFlow SIG Recommenders. "TensorFlow Recommenders-Addons (TFRA)" (GitHub repository and README). accessed 2026. https://github.com/tensorflow/recommenders-addons
20. 4Paradigm. "OpenEmbedding" (GitHub repository and README). 4Paradigm, accessed 2026. https://github.com/4paradigm/OpenEmbedding
21. NVIDIA. "HugeCTR Core Features" (repository documentation). NVIDIA Merlin, accessed 2026. https://github.com/NVIDIA-Merlin/HugeCTR/blob/main/docs/source/hugectr_core_features.md
22. NVIDIA. "HierarchicalKV" (GitHub repository and README, benchmarks). NVIDIA Merlin, accessed 2026. https://github.com/NVIDIA-Merlin/HierarchicalKV
23. Svenstrup, D., Hansen, J., Winther, O. "Hash Embeddings for Efficient Word Representations." NeurIPS 2017. https://arxiv.org/abs/1709.03933
24. Ginart, A. et al. "Mixed Dimension Embeddings with Application to Memory-Efficient Recommendation Systems." Meta / Stanford, 2019. https://arxiv.org/abs/1909.11810
25. Yin, C. et al. "TT-Rec: Tensor Train Compression for Deep Learning Recommendation Models." Meta, MLSys 2021. https://arxiv.org/abs/2101.11714
26. Meta Research. "FBTT-Embedding" (GitHub repository for TT-Rec, archived). Meta, accessed 2026. https://github.com/facebookresearch/FBTT-Embedding
27. Kang, W.-C. et al. "Learning to Embed Categorical Features without Embedding Tables for Recommendation" (DHE). Google, KDD 2021 (arXiv 2020). https://arxiv.org/abs/2010.10784
28. Jouppi, N. et al. "TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings." Google, ISCA 2023. https://arxiv.org/abs/2304.01433
29. Zhai, J. et al. "Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations" (HSTU). Meta, ICML 2024. https://arxiv.org/abs/2402.17152
