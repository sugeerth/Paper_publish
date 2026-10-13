# 30. Scaling Laws and Hardware for Recommenders: Compute, Embeddings, Accelerators and Cost

> Source note. This chapter was written while live web search and fetching were unavailable.
> The shared search budget for the run was used up and the egress proxy refused arXiv and
> Meta domains. Facts tagged with references that other chapters of this Atlas fetched and
> checked (HSTU, Wukong, GEM, Andromeda, Lattice, RankMixer, OneRec, TPU v4, Neo/ZionEX,
> TorchRec, HugeCTR, PERSIA, DeepRecSys, Pinterest GPU serving) are reused from those
> chapters. A few primary papers could not be re-fetched in this run: Ardalani et al. on
> DLRM scaling laws, Climber, the MTIA posts, Gupta et al. HPCA 2020, Sustainable AI and
> RecNMP. Their figures come from the author's prior reading and are marked
> "(approx., not re-fetched)". Check them against the linked primary source before you
> quote them.

## Overview

For about a decade, industrial recommenders scaled in a way that differed sharply from
vision and language models. Parameter counts grew enormously, into the trillions and
eventually a claimed 100 trillion [4][26]. Almost all of those parameters sat in sparse
embedding tables, though. Each example touched only a few kilobytes of them and did very
little arithmetic on what it read. The models were memory-capacity-bound and
memory-bandwidth-bound, not FLOP-bound. Hardware utilization was poor, and secondary
analyses put DLRM model FLOPs utilization (MFU) at roughly 4-5% [11][31]. Adding
embedding rows or dimensions also stopped improving quality early [5][14].

Three shifts between 2022 and 2026 changed this picture:

1. **Scaling laws for dense compute.** Wukong [6], HSTU [7], Climber [10], RankMixer [11],
   OneRec-V2 [12] and ULTRA-HSTU [9] showed that recommendation quality follows predictable
   power laws when the scaled axis is dense interaction or sequence compute, not table size.
2. **Hardware co-design.** Embedding traffic received dedicated silicon and fabrics: Google
   TPU SparseCore [19], Meta ZionEX's RoCE scale-out network [4], Grace Hopper's unified
   memory used by Meta Andromeda [18], and Meta's own MTIA accelerators [20][21].
3. **MFU as the north-star metric.** Architectures are now judged partly by how well they
   use GPUs. RankMixer raised ranking MFU from 4.5% to 45% [11], OneRec reported about
   24%/29% training/inference MFU [13], and GEM about 20-25% [16]. That headroom is what
   makes LLM-scale recommenders affordable.

This chapter covers the scaling-law papers, the embedding-versus-dense compute trade-off,
the accelerators (TPU SparseCore, MTIA, GPUs and CPUs), and what is publicly known about
power and cost in recommendation datacenters.

```
             parameters (log)                     dense FLOPs / example (log)
                 ^                                      ^
   2019-2022     |  ######## embeddings                 |  #  small MLPs
   "sparse era"  |  #        dense                      |     (memory-bound)
                 +------------------------>             +-------------------->
   2024-2026     |  ######   embeddings (still huge)    |  ########  attention /
   "dense era"   |  ####     dense (HSTU, GEM, OneRec)  |  ########  token mixing
                 +------------------------>             +-------------------->
```

---

## 1. Why recommender compute is different

### 1.1 The DLRM workload profile

DLRM [1] is the reference model. Dense features pass through a bottom MLP. Each sparse
feature indexes an embedding table, and the looked-up rows are pooled (sum or mean). Pairwise
dot-product interactions are then concatenated and fed to a top MLP. Its compute
characteristics come from this structure:

- **Embedding lookups** are gathers of random rows from tables that can reach many GB or TB.
  They do almost no arithmetic per byte moved, so they are limited by memory bandwidth and,
  once tables are sharded, by all-to-all network bandwidth [3][4].
- **MLPs and interactions** are ordinary GEMMs. In classic DLRMs they were small, so they
  could not saturate an accelerator's math units.

Meta's characterization "Understanding Training Efficiency of Deep Learning Recommendation
Models at Scale" [3] found three things. Embedding operators are memory-bound and dominated
by communication. Production models differ widely in sparse-feature profile, so no single
hardware configuration fits all of them. CPU parameter-server trainers also spent much of
each step waiting on embedding pulls and pushes. Gupta et al. (HPCA 2020) made a similar case
for inference [2]. They reported that recommendation models consumed the large majority of
AI inference cycles in Facebook datacenters, about 79% (approx., not re-fetched). They also
showed that model classes split into embedding-dominated and MLP-dominated types with very
different bottlenecks.

### 1.2 A rough roofline for the two halves

| Component | Typical arithmetic intensity | Bound by | Scales with |
|---|---|---|---|
| Embedding bag (gather + pool) | very low (well under 1 FLOP/byte, approx.) | HBM/DRAM bandwidth, all-to-all | lookups x dim x bytes |
| Feature interaction (dot/FM/DCN) | moderate | mixed | features^2 x dim |
| MLP / token-mixing / FFN | high at large batch | tensor-core FLOPs | width^2 x tokens |
| Self-attention over sequences | high (with fused kernels) | FLOPs and HBM for KV | seq^2 x dim |

The practical consequence is that in a DLRM, buying more FLOPs barely helps. You need memory
capacity, memory bandwidth and network bisection bandwidth. Generative and sequential models
(Section 3) move most of the cost into the bottom two rows, where GPUs and TPUs work well.

---

## 2. Scaling laws for classic DLRMs: data, parameters, compute

### 2.1 Understanding Scaling Laws for Recommendation Models (Meta, 2022)

Ardalani et al. [5] (approx., not re-fetched) ran the first systematic scaling study of
production-style DLRMs on Meta click-through-rate data. They looked at three axes: data,
embedding parameters and dense (MLP) parameters. They fit power laws plus an irreducible
constant. Their main findings, as commonly summarized, were:

- **Data scaling** gave the most reliable gains for a given compute budget.
- **Parameter scaling**, especially of embedding tables, was "running out of steam": loss
  improvements flattened quickly as tables grew.
- Under a compute budget, the best mix favored more data over larger tables. This motivated
  efforts to find architectures whose dense compute could scale usefully.

### 2.2 Embedding collapse: why bigger tables do not help

Guo et al. (Tsinghua and Tencent, ICML 2024) [14] give a mechanistic explanation. When
embedding size is scaled up, the embedding matrices tend to **collapse** into a
low-dimensional subspace, so the extra dimensions carry little information. Feature
interaction modules make the collapse worse even as they limit overfitting. Their
multi-embedding design uses several independent embedding sets, each with its own
interaction module. It kept improving quality up to 10x embedding size [14]. The lesson for
system designers is not to buy memory capacity on the assumption that wider embeddings will
pay off. Scale through compute in the interaction and sequence layers instead.

---

## 3. Dense-compute scaling laws (2024-2026)

### 3.1 Wukong (Meta, ICML 2024)

**Problem.** Existing interaction architectures (DLRM, DCNv2, AutoInt, and others) stopped
improving as they were scaled up in compute. A recommender with a predictable scaling law,
like an LLM's, needs an interaction backbone whose quality keeps rising with FLOPs [6].

**Architecture.** Wukong is a stack of identical layers. Each layer has two parallel blocks
whose outputs are concatenated:

- A **Factorization Machine Block (FMB)**, which computes FM-style pairwise interactions of
  the previous layer's embeddings and projects them with an MLP.
- A **Linear Compress Block (LCB)**, which linearly recombines the embeddings and preserves
  lower-order terms.

Each layer interacts the outputs of the previous one, so layer i covers interaction orders
1 to 2^i. Order grows exponentially with depth, whereas DCN-style crosses grow linearly.
Scaling knobs are the number of layers, layer width, number of FM-projected embeddings and
MLP sizes [6].

```
 embeddings X0 (n x d)
      |
 +----v-------------------------------------------+
 | Wukong layer i                                  |
 |   FMB: X_{i-1} X_{i-1}^T -> flatten -> MLP --+  |
 |   LCB: W * X_{i-1} (linear recombine) -------+--> concat -> LN + residual
 +------------------------------------------------+
      |  (repeat L times: orders 1 .. 2^L)
      v
 top MLP -> task heads
```

**Scale.** On an internal Meta dataset, Wukong held a scaling law across about two orders of
magnitude of model complexity, beyond about 100 GFLOP per example. Prior architectures had
stopped improving well before that point. Wukong also beat state-of-the-art models on six
public datasets [6]. Secondary coverage connects Wukong-style stacked factorization to Meta's
GEM ads foundation model [15].

**Lesson.** The interaction operator determines whether extra FLOPs turn into quality.
Designs that raise interaction order cheaply, and that run as dense GEMMs, are the ones that
scale.

### 3.2 HSTU and Generative Recommenders (Meta, ICML 2024)

**Problem.** DLRMs depend on heterogeneous, hand-built features and fail to scale with
compute. Meta recast retrieval and ranking as **sequential transduction** over a user's
chronological stream of (item, action) tokens [7].

**Architecture.** The HSTU block uses pointwise SiLU attention without a softmax, plus a
relative position and time bias. A gating term `LayerNorm(attn) * U` replaces the wide FFN.
Per-layer activation memory falls from about 33x hidden dimension to about 14x (secondary
estimate) [7][31]. Training uses **Stochastic Length**, sparsely subsampling long histories.
Inference uses **M-FALCON**: candidates are appended to the user sequence with a mask that
lets them attend only to history, and they are processed in micro-batches that share the
history's KV cache [7].

**Scale numbers (as reported).**

- Models of about **1.5 trillion parameters** were deployed on multiple surfaces of a
  billion-user platform, with **+12.4%** on online A/B metrics [7].
- HSTU runs **5.3x-15.2x faster** than FlashAttention2-based transformers at sequence length
  8,192 [7].
- Quality follows a **power law in training compute across three orders of magnitude**, up
  to roughly GPT-3 or LLaMa-2 scale training compute [7]. Secondary notes give the fits as
  HR@100 = 0.15 + 0.0195 ln C for retrieval and NE = 0.549 - 0.0053 ln C for ranking. The
  sweep covered sequence lengths 512-8192, hidden dimensions 256-1024 and up to 24 layers
  [31].
- The open-source repository adds Triton kernels, an FA3-based HSTU attention kernel for
  H100, row-wise FP8 GEMMs, a "DLRM-v3" HSTU ranker with an MLCommons LoadGen harness, and
  "HSTU Ultra" [8].

**Lesson.** Restructuring training data, from one example per impression to one causal
sequence per user, amortizes user-side compute across all of that user's events. This
efficiency gain arrives before any parameters are added, and it is what makes the scaling
law economical.

### 3.3 ULTRA-HSTU: "Bending the Scaling Law Curve" (Meta, 2026)

Meta's 2026 follow-up [9] targets the slope and offset of the curve rather than just moving
along it. The repository's HSTU Ultra variant [8] combines three techniques:

- semi-local attention (local windows plus selected global context),
- target-aware attention,
- FP8 projections, reported at 1.13-1.25x end-to-end speedups.

The framing is that once a scaling law exists, architecture and systems work should aim to
get the same quality for fewer FLOPs, which shifts the whole curve.

### 3.4 Climber (NetEase Cloud Music, 2025)

Climber [10] (approx., not re-fetched) targets the cost of HSTU-style scaling in a
mid-sized company. Its main components, as summarized from the abstract:

- **ASTRO** (Adaptive Scalable Transformer for RecOmmendation). It uses multi-scale sequence
  partitioning: the user sequence is split into sub-sequences that are processed separately,
  which cuts attention complexity from O(n^2 d) to roughly O(n^2 d / Nb) for Nb blocks. It
  also uses dynamic temperature modulation to adapt attention distributions across
  scenarios and behaviors.
- **TURBO**, an acceleration framework that co-designs training and serving, including
  "encode once, decode many" style shared computation and fused operators.

The paper reports that this design gives more favorable scaling curves than HSTU-style
baselines. It also reports roughly a 5x training-throughput improvement (approx.) and an
online metric gain of about 12% in NetEase Cloud Music deployment (approx.). Treat the exact
numbers as unverified.

**Lesson.** Scaling laws are only actionable if their compute cost is affordable. Block-wise
sequence partitioning and serving-side KV reuse are the usual levers.

### 3.5 RankMixer and TokenMixer-Large (ByteDance, 2025-2026)

**Problem.** Douyin's industrial rankers had accumulated many handcrafted interaction
modules that ran at about **4.5% MFU** [11].

**Architecture.** Features are grouped into tokens. Each block applies **multi-head token
mixing**, a parameter-free recombination of token heads that replaces quadratic
self-attention over heterogeneous feature spaces. It then applies a **per-token FFN** with
separate parameters per feature subspace, optionally extended to a sparse MoE that is dense
in training and sparse at inference [11].

**Results.**

- MFU rose from **4.5% to 45%**.
- Ranking-model parameters scaled about **100x** at roughly constant inference latency.
- Full Douyin deployment gave +0.2% active days and +0.5% app duration [11].
- TokenMixer-Large reports offline scaling to 7B and 15B parameters [33].

**Lesson.** Hardware efficiency is itself an architecture objective. Making the model
"GEMM-shaped" let ByteDance scale parameters 100x within the same latency envelope.

### 3.6 OneRec and OneRec-V2 (Kuaishou, 2025)

OneRec replaces the retrieval-and-ranking cascade with a generative model. Its technical
report claims about **24% training MFU and 29% inference MFU** and operating cost of about
**10%** of the previous cascade (approx.) [13].

OneRec-V2 introduced a lazy decoder-only design. At 1B parameters it matched encoder-decoder
quality (loss 3.27 vs 3.28) with **18.89 vs 296.36 GFLOPs**, about 94% less compute [12].
Its dense scaling law, fit from 0.1B to 8B parameters, is L(N) = 3.13 + 3660 / N^0.489. A
4B-total, 0.5B-active MoE reached loss 3.22, better than a 2B dense model [12].

### 3.7 Other scaling studies

- **Large sequential recommendation scaling** (RUC, 2023-2024) [30] (approx., not
  re-fetched). Decoder-only ID-based sequential recommenders scaled to about 0.8B parameters
  followed power laws. The gains were largest for cold-start and long-tail users.
- **MTGR** (Meituan, 2025) [32]. Pure generative ranking without cross features
  underperformed. The hybrid design reached about 65x FLOPs per sample versus DLRM (approx.).
  The point is that scaling laws hold within an architecture family, but feature
  information can matter more than a 10x compute change.
- **LUM** (Alibaba, 2025) [34]. A generative pre-trained user model is queried by
  discriminative downstream models to "unlock" scaling in industrial settings.

### 3.8 Scaling-law comparison

| Work (org, year) | Scaled axis | Reported law / range | Hardware-efficiency note |
|---|---|---|---|
| Ardalani et al. (Meta, 2022) [5] | data, emb params, MLP params | power law + constant; emb scaling saturates (approx.) | DLRM, memory-bound |
| Embedding collapse (Tsinghua/Tencent, 2024) [14] | embedding size | gains stall unless multi-embedding | n/a |
| Wukong (Meta, 2024) [6] | interaction layers/width | ~2 orders of magnitude, >100 GFLOP/example | dense FM GEMMs |
| HSTU (Meta, 2024) [7] | seq len, depth, width | power law over 3 orders of compute; 1.5T params | 5.3-15.2x vs FA2 |
| Climber (NetEase, 2025) [10] | seq blocks, depth | improved curve vs HSTU (approx.) | ~5x training speedup (approx.) |
| RankMixer (ByteDance, 2025) [11] | params (dense + MoE) | ~100x params at constant latency | MFU 4.5% -> 45% |
| OneRec-V2 (Kuaishou, 2025) [12] | params 0.1B-8B | L = 3.13 + 3660/N^0.489 | 94% fewer FLOPs than enc-dec |
| GEM (Meta, 2025-26) [15][16] | training FLOPs | 4x FLOPs in 12 months | MFU 20-25% |
| ULTRA-HSTU (Meta, 2026) [9] | architecture efficiency | "bends" the curve | FP8, semi-local attention |

---

## 4. Embedding compute versus dense compute: systems consequences

### 4.1 The sparse side: sharding, caching, offload

When embeddings dominate, the training system has to be designed around them.

- **Neo / ZionEX (Meta)** [4]. This is a fully synchronous GPU training system. Each node
  has 8 GPUs on NVLink, and each node also has a **dedicated RoCE scale-out network**, with
  several NICs per node so every GPU has its own path for all-to-all embedding traffic.
  "4D parallelism" chooses table-wise, row-wise, column-wise or data-parallel placement per
  table using a cost-model planner. The paper reports training models of up to
  **12 trillion parameters** and roughly order-of-magnitude (up to about 40x) speedups over
  CPU-based asynchronous training [4].
- **TorchRec + FBGEMM_GPU** [24] is the open-source production version of these ideas.
  It provides sharded embedding collections, table-batched embedding (TBE) kernels that
  run many tables in one launch, quantized tables, and pipelined input distribution.
- **PERSIA (Kuaishou/ETH)** [26] keeps embeddings on a CPU parameter server trained
  asynchronously and the dense network on GPUs trained synchronously. It reports models of
  up to **100 trillion parameters**.
- **HugeCTR (NVIDIA)** [25] keeps the whole loop on GPU. Its documentation states that
  two 8x A100-80GB nodes can hold models of about **1 TB** entirely in GPU memory.
- **Check-N-Run (Meta)** [28] checkpoints only modified rows and quantizes them. It reports
  roughly 6-17x lower checkpoint write bandwidth and 2.5-8x lower storage.

```
 sparse half (model-parallel)                dense half (data-parallel)
 +-------------------------------+           +----------------------------+
 | ID features -> sharded tables  |  all-to-  | interaction + MLP/attn     |
 | (TW/RW/CW, HBM + host offload)  |==> all ==>| replicated on every GPU    |
 | bandwidth- & capacity-bound    |  pooled   | FLOP-bound, all-reduce     |
 +-------------------------------+   embs    +----------------------------+
          ^ scale: memory, NICs                     ^ scale: tensor cores
```

### 4.2 The dense side: MFU becomes the metric

Once dense compute is the scaled axis, the important question is what fraction of peak FLOPs
the model actually uses:

| System | Reported MFU | Main tricks |
|---|---|---|
| Typical legacy DLRM ranker (per ByteDance) [11] | ~4.5% | (baseline) |
| RankMixer [11] | ~45% | token mixing, per-token FFN, MoE |
| OneRec V1 [13] | ~24% train / ~29% infer (approx.) | MoE decoder, compute in generator |
| GEM training [16] | 20-25% | kernels, precision, parallelism, networking, memory co-design |
| LLM pre-training (for context, secondary) [31] | ~40-60% | dense transformer |

The HSTU kernels [7][8] and fused attention for jagged sequences show what it takes to close
the gap to LLMs. Recommendation data comes in variable-length ("jagged") batches, many tables
and small per-feature tensors. Without padding-free kernels and op fusion, GPUs sit idle.

### 4.3 Where the parameters versus the FLOPs live

| Model family | Params mostly in | FLOPs mostly in | Hardware fit |
|---|---|---|---|
| Classic DLRM/DCN (2019-2022) | embeddings (often >90%) [1][31] | small MLPs | memory/network-bound; CPUs and SparseCore |
| Wukong / RankMixer (2024-25) | embeddings + large dense stacks | dense interaction GEMMs | GPU-friendly if fused |
| HSTU / GEM / OneRec (2024-26) | embeddings + transformer blocks | attention + FFN over long sequences | GPU/TPU, LLM-like |

---

## 5. Google TPU SparseCore: embeddings in silicon

**Problem.** In Google's fleet, embedding-heavy models (DLRMs for ads, search and YouTube)
are a large workload. On TensorCores alone, embedding lookups and the all-to-all traffic
around them run poorly [19].

**Design (TPU v4, ISCA 2023).** Each TPU v4 chip includes **SparseCores**. These are
dataflow processors dedicated to embedding lookups, which are gathers with variable-length
pooling and scatter-add updates, and to the related all-to-all communication over the
inter-chip interconnect. Embedding tables are sharded across the HBM of many chips in a
slice. SparseCores perform the lookup, the reduction and the cross-chip exchange while the
TensorCores run the dense layers [19].

```
 TPU v4 chip
 +--------------------------------------------------------------+
 |  TensorCores (MXU systolic arrays) <-- dense MLP / attention |
 |  SparseCores (dataflow tiles)      <-- embedding gather,     |
 |        |                                 pool, scatter-add   |
 |        +--> ICI (3D torus, OCS-reconfigurable) all-to-all     |
 |  HBM: shard of every large embedding table                   |
 +--------------------------------------------------------------+
   x 4,096 chips per pod (optically switched)
```

**Reported numbers [19].**

- SparseCores accelerate embedding-heavy models by about **5-7x**.
- They use only about **5% of die area and power** (approx., from the abstract).
- At the system level, TPU v4 pods use **optical circuit switches** to reconfigure the
  interconnect topology.
- The paper compares TPU v4 with contemporary GPUs and reports better performance and
  performance per watt on its benchmarks (approx.).

**Later generations.** Google has said that later TPUs continue the SparseCore line.
Trillium (TPU v6e) is described as having a third-generation SparseCore for "ultra-large
embeddings" in ranking and recommendation workloads [37] (approx., not re-fetched).

**Lesson.** About 5% of the silicon budget, spent on the operator that dominates
recommendation runtime, bought several-fold speedups. That is the clearest public case for
specialized sparse hardware. YouTube's later move toward semantic-ID generative retrieval
(see Chapter 17) shifts the balance back toward dense compute, so a chip needs both kinds of
engine.

---

## 6. Meta MTIA: a recommendation-first accelerator

> Figures in this section are from Meta's MTIA posts and ISCA papers as previously read.
> They could not be re-fetched in this run and are marked approximate.

**Problem.** Ranking and recommendation inference is a large share of Meta's AI fleet [2].
GPUs were seen as over-provisioned in FLOPs and expensive in power for these
memory-intensive, small-batch models. Meta wanted better performance per watt and per TCO
dollar for its own workloads [20][21].

### 6.1 MTIA v1 (announced May 2023)

Reported design (approx.) [20]:

- TSMC 7nm at 800 MHz, with about 102 TOPS INT8 and 51 TFLOPS FP16/BF16 at about a 25 W TDP.
- A grid of 64 processing elements (8x8), each with RISC-V cores, plus fixed-function units
  for matrix multiply, SIMD and data movement.
- About 128 MB of on-chip SRAM and 64 GB of off-chip LPDDR5.

The design philosophy was to keep hot embeddings and activations in large on-chip SRAM and
to use cheap, high-capacity LPDDR instead of HBM for the long tail. The co-designed software
stack is PyTorch-native and integrates with FBGEMM-style embedding kernels.

### 6.2 MTIA v2 ("next-generation MTIA", April 2024)

Reported design (approx.) [21]:

- TSMC 5nm at 1.35 GHz, with about 354 TOPS INT8 dense (about 708 with sparsity) at about a
  90 W TDP.
- About 256 MB of on-chip SRAM and 128 GB of LPDDR5.
- A rack design holding 72 accelerators (three chassis of 12 boards with two chips each).

Meta reported roughly 3x performance over the first generation on four key models and
roughly 6x model-serving throughput at the platform level. It also reported roughly 1.5x
performance per watt. The chip was deployed for ranking and recommendation ads models in
production.

### 6.3 Andromeda and the mixed fleet

Meta's Andromeda ads retrieval engine (December 2024) is co-designed with **NVIDIA Grace
Hopper Superchips**, and the post names **MTIA** as part of the hardware plan [18]. Andromeda
replaced lightweight retrieval with a deep network over a jointly learned hierarchical index.
It reported **+6% recall** and **+8% ads quality** on selected segments. Secondary coverage
cites about a **10,000x** increase in retrieval model complexity (approx.) [18]. Grace
Hopper's unified CPU-GPU memory over NVLink-C2C lets large embedding and index structures sit
in CPU-attached memory while the GPU computes on them. This directly addresses the capacity
problem from Section 4.

```
 Meta recsys fleet (as publicly described, 2024-2026)
   training: GPU clusters (ZionEX lineage -> large GPU clusters for GEM)
   retrieval: Andromeda DNN on Grace Hopper (unified memory) [18]
   ranking:  Lattice / GEM-distilled students on GPUs and MTIA [17][15][21]
   teacher:  GEM, LLM-scale, thousands of GPUs, MFU 20-25% [15][16]
```

**Lesson.** Hyperscalers run heterogeneous fleets. Custom ASICs (MTIA, SparseCore) win on
perf/W for stable, high-volume operators. GPUs are used for the fast-moving frontier
(generative, LLM-scale teachers) and for retrieval stages co-designed with unified memory.

---

## 7. GPU versus CPU (and accelerator) trade-offs

### 7.1 The historical CPU case

Until roughly 2020, most industrial recommendation inference ran on CPUs, and much training
ran on CPU parameter-server clusters [3][4]. CPUs had three advantages. First, large and
cheap DRAM holds TB-scale tables. Second, request handling is flexible, with variable
candidate counts and many small ops. Third, the fleet already existed. DeepRecSys
(Harvard/Facebook, ISCA 2020) [22] showed that query sizes (candidates per request) vary
widely. Its scheduler, DeepRecSched, picks batch sizes and splits queries between CPU cores
and accelerators: small queries stay on CPU and large ones go to the accelerator. This gave
about 2x latency-bounded throughput and over 30% latency reduction in production [22].
Near-memory processing proposals such as RecNMP [36] (approx., not re-fetched) went further
and put embedding pooling in DIMM-side logic.

### 7.2 The GPU transition

- **Training.** Meta moved from asynchronous CPU training to synchronous GPU training with
  Neo/ZionEX and TorchRec. It reported up to about 40x faster time-to-solution and better
  reproducibility and quality from synchronous updates [4][24]. NVIDIA HugeCTR and Merlin
  make the same case [25].
- **Inference.** Pinterest's 2022 post describes moving ranking from CPU to GPU serving
  because CPU cost grew about linearly with model size. GPUs let it serve models with on the
  order of 100x more compute at comparable or lower cost and latency [23]. The techniques
  were batching all candidates of a request, CUDA Graphs, merging embedding lookups, mixed
  precision, and moving feature preprocessing onto the GPU [23].

### 7.3 Decision table

| Criterion | CPU | GPU | Custom ASIC (MTIA, TPU+SparseCore) |
|---|---|---|---|
| Embedding capacity | best (TB DRAM, cheap) | limited HBM; needs sharding, caching, or Grace Hopper | large SRAM + LPDDR (MTIA); sharded HBM (TPU) |
| Embedding bandwidth | moderate | high HBM; NVLink/RoCE all-to-all | dedicated (SparseCore) |
| Dense FLOPs | poor | best | good |
| Small/variable batches | good | needs batching, CUDA graphs | designed for it (MTIA) |
| Perf/W on stable DLRMs | low | medium | best (vendor claims) |
| Flexibility for new archs | high | high | lower; compiler-dependent |
| Best fit | light rankers, feature logic, retrieval glue | heavy ranking, generative models, training | high-volume stable ranking inference |

### 7.4 Precision as a cost lever

Low precision is applied unevenly:

- **Embeddings** are quantized first, to FP16/INT8/INT4 rows with row-wise scales, for
  serving and checkpoints [24][28].
- **Dense layers** follow. HSTU Ultra uses row-wise FP8 GEMMs [8], and GEM training
  efficiency gains cite numeric-precision co-design [16].

Most of the bytes are embedding bytes, so cutting embedding precision first gives the
largest memory and bandwidth savings per unit of quality risk.

---

## 8. Power, cost and datacenter footprint

Public, precise numbers on recommendation power are rare. What is documented:

- **Share of cycles.** Gupta et al. [2] (approx., not re-fetched) reported that
  recommendation models accounted for most AI inference cycles in Facebook's datacenters, on
  the order of 79%.
- **Sustainable AI (Wu et al., Meta, MLSys 2022)** [27] (approx., not re-fetched)
  characterized the carbon footprint of Meta's production models, including recommendation
  models (RM1-RM5). It reported that power capacity for AI was split roughly 10:20:70 across
  experimentation, training and inference. Recommendation data and model sizes grew several
  times over in about two years. Most of the gain came from efficiency work: hardware-software
  co-design, caching and quantization.
- **TPU v4 SparseCore** [19] delivered about 5-7x embedding speedups for about 5% of die
  power, so for Google's embedding-heavy models the efficiency gain is several-fold.
- **MTIA v2** [21] targets better perf/W and TCO than GPUs for ranking models. Meta reported
  roughly 1.5x perf/W over MTIA v1 (approx.).
- **Model consolidation as a cost tool.** Meta Lattice merged many ads models into one
  multi-domain, multi-task model. It reported **+10% topline** and **about 20% capacity
  savings** [17]. OneRec reported operating cost of about 10% of its prior cascade
  (approx.) [13]. GEM is kept offline as a teacher and distilled into cheap serving students
  [15].

```
 cost per served recommendation  ~=  (training $ / requests served over model life)
                                   + (serving FLOPs x $/FLOP at achieved MFU)
                                   + (embedding bytes x $/GB-month of the memory tier)
                                   + (network all-to-all at training scale)
 levers: MFU up | precision down | distill teacher->student | consolidate models |
         tier embeddings (SRAM/HBM -> DRAM/LPDDR -> SSD) | cache hot rows
```

**Lesson.** The biggest documented cost wins came from structural changes, not from
low-level tuning alone. Examples are consolidating model fleets, distilling large teachers,
replacing cascades, and moving to architectures that use accelerators well.

---

## 9. Putting it together: a reference scaling stack (2026)

```
                    +-------------------------------------------+
  offline teacher   | Foundation recsys model (GEM / HSTU-scale) |  GPU cluster,
  (LLM-scale)       | long sequences, MoE, FP8, MFU 20-45%       |  power-law scaling
                    +---------------------+---------------------+
                                          | distillation, embeddings
            +-----------------------------+------------------------------+
            v                             v                              v
  retrieval (GPU + unified mem)   light ranking (CPU/GPU)       heavy ranking (GPU/ASIC)
  learned index, sub-linear       cheap towers, caching          RankMixer/Wukong-style,
  (Andromeda-like)                                               M-FALCON KV sharing
            \_____________________________|______________________________/
                                          v
             sparse substrate: sharded embeddings (TorchRec/TPU SparseCore),
             tiered memory (HBM -> DRAM/LPDDR -> SSD), INT8/INT4 rows,
             incremental checkpoints, row admission/eviction
```

---

## Key takeaways for system designers

1. **Scale dense compute, not table size.** The evidence of saturating embedding scaling
   [5], embedding collapse [14], and dense-compute power laws (Wukong, HSTU, RankMixer,
   OneRec-V2) [6][7][11][12] all point the same way. Budget extra FLOPs into interaction and
   sequence modules.
2. **Expect logarithmic returns.** Reported laws are linear in ln(compute) [7][31] or power
   laws in N [12]. Each fixed quality step costs about 10x compute, so efficiency work that
   bends the curve [9][10] is worth as much as raw scale.
3. **Track MFU as a first-class metric.** Going from about 5% to 20-45% MFU [11][13][16] is
   an order of magnitude of free model capacity. Make architectures GEMM-shaped, use
   padding-free jagged kernels, and fuse operators.
4. **Restructure training data before scaling.** One causal sequence per user (HSTU) and
   encode-once/score-many inference (M-FALCON, Climber's TURBO) amortize user compute [7][10].
5. **Embeddings are a memory-hierarchy and network problem.** Size clusters by table bytes
   and all-to-all bandwidth first. Shard per table with a cost-model planner. Cache the hot
   head, and quantize and offload the tail [4][24][25].
6. **Specialized sparse hardware pays off.** SparseCore's about 5-7x for about 5% of die
   area [19] and ZionEX's dedicated RoCE fabric [4] show that embedding traffic justifies its
   own silicon and network.
7. **Run a heterogeneous fleet deliberately.** Use CPUs for feature logic and light stages,
   GPUs for frontier models and training, and ASICs (MTIA, TPU) for stable, high-volume
   inference. Grace Hopper-style unified memory changes what retrieval can afford [18][21].
8. **Train big, serve small.** LLM-scale teachers distilled into fleet students (GEM) and
   model consolidation (Lattice) were the largest documented cost and quality levers
   [15][17].
9. **Quantize embeddings before dense weights.** Most bytes are in tables, so INT8/INT4 rows
   and quantized incremental checkpoints save the most for the least risk [24][28].
10. **Treat vendor and paper numbers as approximate.** Scaling-law constants are
    dataset-specific and do not transfer between companies. Re-fit them on your own data at
    small scale before committing a hardware budget.

---

## References

1. Naumov, M. et al. "Deep Learning Recommendation Model for Personalization and Recommendation Systems" (DLRM). Meta (Facebook), 2019. https://arxiv.org/abs/1906.00091
2. Gupta, U. et al. "The Architectural Implications of Facebook's DNN-based Personalized Recommendation." Facebook / Harvard, HPCA 2020 (not re-fetched in this run). https://arxiv.org/abs/1906.03109
3. Acun, B. et al. "Understanding Training Efficiency of Deep Learning Recommendation Models at Scale." Meta, HPCA 2021. https://arxiv.org/abs/2011.05497
4. Mudigere, D. et al. "Software-Hardware Co-design for Fast and Scalable Training of Deep Learning Recommendation Models" (Neo / ZionEX). Meta, ISCA 2022. https://arxiv.org/abs/2104.05158
5. Ardalani, N. et al. "Understanding Scaling Laws for Recommendation Models." Meta, 2022 (not re-fetched in this run). https://arxiv.org/abs/2208.08489
6. Zhang, B., Luo, L., Chen, Y. et al. "Wukong: Towards a Scaling Law for Large-Scale Recommendation." Meta, ICML 2024. https://arxiv.org/abs/2403.02545
7. Zhai, J. et al. "Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations" (HSTU). Meta, ICML 2024. https://arxiv.org/abs/2402.17152
8. meta-recsys. "generative-recommenders" repository (HSTU, M-FALCON, DLRM-v3, HSTU Ultra, Triton/FP8 kernels). Meta, 2024-2026. https://github.com/meta-recsys/generative-recommenders
9. Meta. "Bending the Scaling Law Curve in Large-Scale Recommendation Systems" (ULTRA-HSTU). Meta, 2026. https://arxiv.org/abs/2602.16986
10. NetEase Cloud Music. "Climber: Toward Efficient Scaling Laws for Large Recommendation Models." 2025 (not re-fetched in this run). https://arxiv.org/abs/2502.09888
11. ByteDance. "RankMixer: Scaling Up Ranking Models in Industrial Recommenders." 2025. https://arxiv.org/abs/2507.15551
12. Kuaishou OneRec Team. "OneRec-V2 Technical Report." Kuaishou, 2025. https://arxiv.org/abs/2508.20900
13. Kuaishou OneRec Team. "OneRec Technical Report." Kuaishou, 2025. https://arxiv.org/abs/2506.13695
14. Guo, X. et al. "On the Embedding Collapse When Scaling up Recommendation Models." Tsinghua / Tencent, ICML 2024. https://arxiv.org/abs/2310.04400
15. Engineering at Meta. "Meta's Generative Ads Model (GEM): The central brain accelerating ads recommendation AI innovation." Meta, November 2025. https://engineering.fb.com/2025/11/10/ml-applications/metas-generative-ads-model-gem-the-central-brain-accelerating-ads-recommendation-ai-innovation/
16. Engineering at Meta. "GEM Training: How Meta Doubled the Efficiency of Its LLM-Scale Ads Foundation Model." Meta, August 2026. https://engineering.fb.com/2026/08/03/ml-applications/training-gem-at-llm-scale-meta-ads-recommendation-foundation-model/
17. Meta. "Meta Lattice: Model Space Redesign for Cost-Effective Industry-Scale Ads Recommendations." Meta, 2025. https://arxiv.org/abs/2512.09200
18. Engineering at Meta. "Meta Andromeda: Supercharging Advantage+ automation with the next-gen personalized ads retrieval engine." Meta, December 2024. https://engineering.fb.com/2024/12/02/production-engineering/meta-andromeda-advantage-automation-next-gen-personalized-ads-retrieval-engine/
19. Jouppi, N. et al. "TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings." Google, ISCA 2023. https://arxiv.org/abs/2304.01433
20. Meta AI. "MTIA v1: Meta's first-generation AI inference accelerator." Meta, May 2023 (not re-fetched in this run). https://ai.meta.com/blog/meta-training-inference-accelerator-AI-MTIA/
21. Meta AI. "Our next-generation Meta Training and Inference Accelerator." Meta, April 2024 (not re-fetched in this run). https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/
22. Gupta, U. et al. "DeepRecSys: A System for Optimizing End-To-End At-scale Neural Recommendation Inference." Harvard / Facebook, ISCA 2020. http://vlsiarch.eecs.harvard.edu/wp-content/uploads/2020/05/DeepRecSys_Gupta_ISCA2020.pdf
23. Pinterest Engineering. "GPU-accelerated ML Inference at Pinterest." Pinterest, 2022. https://medium.com/@Pinterest_Engineering/gpu-accelerated-ml-inference-at-pinterest-ad1b6a03a16d
24. PyTorch / Meta. "TorchRec" and "FBGEMM / FBGEMM_GPU" repositories. Accessed 2026. https://github.com/pytorch/torchrec ; https://github.com/pytorch/FBGEMM
25. Wei, Y. et al. "Merlin HugeCTR: GPU-accelerated Recommender System Training and Inference." NVIDIA, RecSys 2022. https://arxiv.org/abs/2210.08803
26. Lian, X. et al. "Persia: A Hybrid System Scaling Deep Learning Based Recommenders up to 100 Trillion Parameters." Kuaishou / ETH Zurich, KDD 2022. https://arxiv.org/abs/2111.05897
27. Wu, C.-J. et al. "Sustainable AI: Environmental Implications, Challenges and Opportunities." Meta, MLSys 2022 (not re-fetched in this run). https://arxiv.org/abs/2111.00364
28. Eisenman, A. et al. "Check-N-Run: A Checkpointing System for Training Deep Learning Recommendation Models." Meta, NSDI 2022. https://arxiv.org/abs/2010.08679
29. Liu, Z. et al. "Monolith: Real Time Recommendation System With Collisionless Embedding Table." ByteDance, 2022 (background reading). https://arxiv.org/abs/2209.07663
30. Zhang, G. et al. "Scaling Law of Large Sequential Recommendation Models." Renmin University of China, 2023 / RecSys 2024 (not re-fetched in this run). https://arxiv.org/abs/2311.11351
31. Xing, H. "HSTU: The Cornerstone of Scaling Laws" (fun-rec-mdbook technical notes, secondary). 2026. https://github.com/Haozhe-Xing/fun-rec-mdbook/blob/main/src/en/part7-scaling/hstu.md
32. Han, R. et al. "MTGR: Industrial-Scale Generative Recommendation Framework in Meituan." Meituan, 2025. https://arxiv.org/abs/2505.18654
33. ByteDance. "TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders." 2026. https://arxiv.org/abs/2602.06563
34. Alibaba. "Unlocking Scaling Law in Industrial Recommendation Systems with a Three-step Paradigm based Large User Model" (LUM). 2025. https://arxiv.org/abs/2502.08309
35. Meta Research. "DLRM" repository (reference implementation and MLPerf DLRM lineage). Accessed 2026. https://github.com/facebookresearch/dlrm
36. Ke, L. et al. "RecNMP: Accelerating Personalized Recommendation with Near-Memory Processing." Facebook / Harvard, ISCA 2020 (not re-fetched in this run). https://arxiv.org/abs/1912.12953
37. Google Cloud. "Introducing Trillium, sixth-generation TPUs." Google, May 2024 (not re-fetched in this run). https://cloud.google.com/blog/products/compute/introducing-trillium-6th-gen-tpus
