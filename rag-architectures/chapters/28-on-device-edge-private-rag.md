# 28. On-Device, Edge and Air-Gapped RAG: Small Models, Embedded Indexes and Private Compute

## Overview

Most RAG chapters assume a data center. The retriever runs next to a GPU-backed vector database, the generator is a frontier model behind an API, and the corpus lives in an object store that the operator controls. This chapter turns that around. On phones, laptops, vehicles, factory gateways and air-gapped networks, the corpus is usually *personal or regulated* (messages, email, screen history, medical or financial files, classified documents). The hardware is fixed and shared with other workloads. Memory, flash wear, thermals and battery are the binding constraints, not GPU-hours.

The design space has five layers:

1. **Embedders** small enough to index a whole personal corpus in the background (EmbeddingGemma 308M, static embeddings, NPU-resident OS embedders).
2. **Generators**: small language models (SLMs) at 1-4B effective parameters, such as Llama 3.2 1B/3B, Phi-3/Phi-4-mini, Phi Silica, Gemma 3n and Gemma 4 E2B/E4B, and Apple's AFM on-device models. They are quantized to roughly 2-4 bits per weight, limited to short contexts, and run on portable runtimes such as llama.cpp, MLX, ExecuTorch and LiteRT-LM.
3. **Embedded vector stores** that run in-process with no server: sqlite-vec, LanceDB (embedded mode), ObjectBox, Qdrant Edge, PGlite and DuckDB VSS, and OS-level indexes such as Apple's Spotlight/Core Spotlight and Windows' AppContentIndexer.
4. **Edge-specific retrieval algorithms** that trade compute for memory (EdgeRAG) or swap LLM-heavy graph construction for structure an SLM can handle (MiniRAG).
5. **Hybrid trust boundaries**: when the device is not enough, requests escalate to a cloud whose privacy properties can be verified (Apple Private Cloud Compute). The cloud index may also be synced to or from the device.

The chapter covers each layer, then two flagship products that combine them: Apple Intelligence's on-device semantic index plus PCC, and Microsoft's Recall on Copilot+ PCs. It closes with memory and battery budgets, sync patterns, and takeaways.

```
              ON-DEVICE / EDGE RAG REFERENCE STACK
 +--------------------------------------------------------------------+
 |  App / OS assistant (Siri, Recall, in-app chat, field tool)        |
 +--------------------------------------------------------------------+
 |  Orchestrator: query rewrite, tool routing, escalation policy      |
 +-----------------+--------------------------+-----------------------+
 |  SLM generator  |  Retriever               |  Escalation           |
 |  1-4B, 2-4 bit  |  lexical + vector (+KG)  |  verifiable cloud     |
 |  ~4K-8K ctx     |  rerank optional         |  (PCC-like) or none   |
 +-----------------+--------------------------+-----------------------+
 |  Embedded index: SQLite/vec0, Lance files, ObjectBox HNSW,         |
 |  OS index (Spotlight / AppContentIndexer); encrypted at rest       |
 +--------------------------------------------------------------------+
 |  Background indexer: OCR, chunking, small embedder (NPU/CPU),      |
 |  scheduled when charging / idle; incremental updates               |
 +--------------------------------------------------------------------+
 |  Hardware: shared DRAM (8-16 GB), NPU/GPU, flash, battery, TPM/SE  |
 +--------------------------------------------------------------------+
```

---

## 1. Why on-device RAG is a different problem

### 1.1 The hard constraints

**DRAM is shared and small.** Martin Alderson's 2026 analysis puts it this way: most new laptops ship with 16 GB, 8 GB is still common, and many phones ship with 8 GB (12 GB on Pro models). After the OS and apps, only about 4 GB (non-Pro phones) to 8 GB (Pro phones, base laptops) is realistically left for AI [36]. He estimates a highly quantized 3B model at roughly 2 GB and a 7B model at roughly 5 GB. He also notes that Apple and Google appear to limit on-device context to about 4K tokens on phones [36]. Microsoft documents Phi Silica's context window as "approximately 3.5K tokens" and warns that this limits "large sized knowledge retrieval when using Retrieval-Augmented Generation" [12].

**The KV cache, not the weights, often sets the RAG budget.** Every retrieved passage costs KV-cache memory and prefill compute. That is why on-device systems retrieve fewer, shorter chunks than cloud RAG. Killingback et al. (UMass/Google, 2026) say it directly: on device, "the context size provided to the generative model must be restricted to manage KV cache and attention memory usage, while the size of stored embeddings must be minimized to preserve disk space" [35].

**The OS can evict you.** Phi Silica's platform card notes that "under sustained system memory pressure, Windows may temporarily unload the model," and that this raises time-to-first-token on the next load [12]. Design for cold starts.

**Energy and thermals.** A background embedding job over tens of thousands of documents can drain a phone or overheat it. Production systems therefore schedule indexing opportunistically and keep inference on NPUs where possible. Microsoft notes that Phi Silica "reduces power usage compared to models deployed via the cloud" [12]. Hardware gating is common: Recall requires a 40 TOPS NPU, 16 GB RAM and 256 GB of storage [14].

**Storage is cheap but not free.** Recall needs at least 50 GB free to enable, and pauses snapshots below 25 GB free [14]. Embedding size matters: a 768-dim float32 vector is 3 KB, and binary quantization cuts it 32x [32].

### 1.2 What you gain

There are no per-query API costs, it works offline, latency is predictable, and privacy is the default. That last point decides it for personal data. App Content Search, for example, "does not send indexed content to a cloud search or embedding service" [13].

---

## 2. Small embedders

### 2.1 EmbeddingGemma (Google DeepMind, Sept 2025)

**Problem.** Index a personal or app corpus in 100+ languages on a phone without a server.

**Architecture.** EmbeddingGemma is a 308M-parameter model built on the Gemma 3 backbone with *bidirectional* attention, so the decoder works as an encoder. It has a 2,048-token input window, mean pooling, and two dense projection layers that produce a 768-dimensional vector [4]. It was trained on about 320B tokens of multilingual web text, code, technical documentation and synthetic task data [4].

**Key techniques.**
- **Matryoshka Representation Learning (MRL).** You can truncate the 768-dim output to 512, 256 or 128 dimensions on demand. The trade is quality against index size and scan cost [4].
- **Quantization.** It runs in under 200 MB of RAM when quantized [4].
- **Task prompts.** The model expects prefixes such as `task: search result | query: ` for queries and `title: none | text: ` for documents. Leaving them out is a silent quality bug [4].
- **Runtimes.** Sentence Transformers, Transformers.js (in-browser), ONNX Runtime, TEI, LangChain, LlamaIndex, Haystack and txtai are all supported [4].

**Quality.** At release it was the highest-ranking text-only multilingual embedding model under 500M parameters on MTEB [4]. Hugging Face's fine-tuning example on the MIRIAD medical retrieval dataset beat models twice its size on that task [4]. This is the standard lesson for edge embedders: domain fine-tuning closes much of the gap to larger models.

**Successor.** Google's LiteRT-LM runtime (v0.18.0) ships a multimodal **EmbeddingGemma 2** that embeds text, images and audio, keeps Matryoshka truncation, and is exposed through Python, Kotlin, Swift, JavaScript and C++ APIs plus an OpenAI-compatible `/v1/embeddings` endpoint [45]. For screenshot or voice-note corpora (Section 7), one embedding space across modalities removes the OCR-then-embed step. Google had not published detailed benchmark numbers in the sources reviewed here.

**Lessons.** Pair EmbeddingGemma's MRL truncation with an index that supports int8 or binary vectors (Section 4). Then storage per chunk drops from 3 KB to 16 bytes at 128 binary dimensions, before rescoring.

### 2.2 Static embeddings (Hugging Face, Jan 2025)

For the weakest hardware (microcontroller-class gateways, low-end Android, browser extensions), Sentence Transformers' static embedding recipe drops attention completely. Each token embedding is precomputed, in the spirit of word2vec and Model2Vec. The resulting models run **100x to 400x faster on CPU** than all-mpnet-base-v2 or multilingual-e5-small, while reaching **at least 85%** of their quality on the reported benchmarks [5]. The released `static-retrieval-mrl-en-v1` is also MRL-trained [5]. A common pattern: a static embedder for first-pass recall over a large local corpus, then a small cross-encoder or the SLM itself to rerank a handful of candidates.

### 2.3 OS-managed embedders

Apple and Microsoft increasingly hide the embedder behind an OS API. Windows' `AppContentIndexer` (App Content Search) "manages the text index, embedding model, and vector storage" and chunks long text automatically. Semantic matching runs only on NPU-equipped devices, with lexical matching as the fallback [13]. On Apple platforms, Core Spotlight added semantic search over app-donated items in 2024 [27]. The 2026 Siri relies on "the Spotlight index ... which work[s] entirely on device" [24]. The trade-off is that you cannot choose or fine-tune the model, but you inherit OS-level scheduling, power management and protection.

---

## 3. SLM generators

### 3.1 Llama 3.2 1B / 3B (Meta, Sept 2024)

**Design.** Meta built the 1B (1.23B) and 3B (3.21B) text models by **pruning and distillation**. Logits from Llama 3.1 8B and 70B served as token-level targets during pretraining, on up to 9T tokens [6][7]. Hugging Face shipped day-one Transformers, llama.cpp and Transformers.js support [8]. The model card lists knowledge retrieval, summarization, and query and prompt rewriting among the intended uses [6]. Those are the jobs an on-device RAG pipeline needs.

**Quantization for phones.** Meta's on-device scheme targets ExecuTorch on Arm CPUs. It uses 4-bit groupwise weights (group size 32) with 8-bit per-token dynamic activations in transformer blocks, and 8-bit per-channel quantization for the embedding and classification layers [6]. Two variants were released:
- **SpinQuant**: rotation-based post-training quantization plus GPTQ.
- **QLoRA**: quantization-aware training with LoRA adapters, followed by DPO.

The quantized variants drop context from 128K to 8K [6].

**Measured on a OnePlus 12 (ExecuTorch, Arm CPU)** [6]:

| Model | Decode tok/s | TTFT (s, 64-tok prompt) | Prefill tok/s | File MB | RSS MB |
|---|---|---|---|---|---|
| 1B BF16 | 19.2 | 1.0 | 60.3 | 2358 | 3185 |
| 1B SpinQuant | 50.2 | 0.3 | 260.5 | 1083 | 1921 |
| 1B QLoRA | 45.8 | 0.3 | 252.0 | 1127 | 2255 |
| 3B BF16 | 7.6 | 3.0 | 21.2 | 6129 | 7419 |
| 3B SpinQuant | 19.7 | 0.7 | 89.7 | 2435 | 3726 |
| 3B QLoRA | 18.5 | 0.7 | 88.8 | 2529 | 4060 |

**RAG implication.** At roughly 90 prefill tok/s, a 3B quantized model needs about 22 seconds to read 2,000 tokens of retrieved context on this phone CPU. That is why mobile RAG keeps top-k small, compresses context, or uses an NPU/GPU backend. Quality loss from quantization is modest but real. On the 1B model, MMLU goes from 49.3 (BF16) to 47.3 (SpinQuant) and 49.0 (QLoRA), against 43.3 for vanilla PTQ [6]. The cost of quantization shows up most in tool use. BFCL V2 on the 1B model falls from 25.7 (BF16) to 14.3 with vanilla PTQ and is barely rescued by SpinQuant (15.9). Only the quantization-aware QLoRA variant holds most of it (23.7). The 3B model shows the same ordering: 67.0 BF16, 53.4 PTQ, 60.1 SpinQuant, 63.5 QLoRA [6]. If an on-device RAG agent depends on tool calls, prefer QAT-style checkpoints over post-training rotation methods.

### 3.2 Phi family and Phi Silica (Microsoft)

Phi-3-mini (3.8B) was introduced as small enough to run on a phone when quantized to 4 bits [11]. Phi-4-mini continued the line at the same size class [40]. On Windows, Microsoft ships **Phi Silica**, an NPU-tuned local model exposed through the Windows App SDK as a Limited Access Feature. It uses **speculative decoding**, in which a smaller draft model proposes tokens that the main model verifies in parallel. It supports LoRA adapters and built-in "Text Intelligence Skills" such as summarize, rewrite and text-to-table [12]. The documented constraints read like a list of on-device RAG design rules [12]:
- About a 3.5K-token context window.
- Built-in input and output content moderation, which can over-block, and which adds overhead in streaming mode.
- Possible unloading under memory pressure.
- Availability only on Copilot+ NPUs.

### 3.3 Gemma 3n and Gemma 4 E-series (Google)

Gemma 3n (June 2025) introduced "effective" sizes. E2B and E4B have 5B and 8B real parameters but need only about 2 GB and 3 GB of accelerator memory [9]. Three mechanisms get them there [9]:
- **Per-Layer Embeddings (PLE).** Large per-layer embedding tables are offloaded to the CPU and used only for lookups.
- **MatFormer.** E2B is trained as a nested sub-model of E4B, so developers can "mix and match" layers to fit a memory budget.
- **KV-cache sharing.** This gives 2x faster prefill than Gemma 3 4B for long audio and video.

Gemma 3n also bundles a 300M MobileNet-V5 vision encoder and a USM-based audio encoder [9]. That makes multimodal local RAG possible, for example over screenshots or voice notes. Gemma 4 (2026) keeps the E2B/E4B on-device tier (2.3B/4.5B effective, 5.1B/8B with embeddings) with a 128K context. It alternates local sliding-window layers (512 tokens in the small dense models) with global full-context layers, and the last layers reuse K/V tensors from earlier layers (shared KV cache) [10][41]. The same family adds a 12B dense "Unified" model, a 31B dense model and a 26B-total / ~4B-active mixture-of-experts model, all with 256K context [41]. These are laptop-class rather than phone-class. Whether a phone can actually hold a 128K KV cache is a separate question (Section 8).

### 3.4 Apple Foundation Models on device

Apple's 2024 on-device model was about 3B parameters. It used a mixed 2-bit/4-bit palettization averaging about 3.5-3.7 bits per weight, with LoRA adapters swapped per feature. Apple reported roughly 0.6 ms per prompt token time-to-first-token and about 30 tok/s generation on iPhone 15 Pro (approximate figures from Apple's announcement) [21]. The 2025 update split the model into two blocks, with the second block reusing the first block's KV cache, which cut KV memory by about 37.5%. It also used 2-bit quantization-aware training, and it exposed the model to developers through the Foundation Models framework with guided generation and tool calling [22].

The third generation (WWDC, June 2026) adds **AFM 3 Core**, a 3B dense model, and **AFM 3 Core Advanced**. Core Advanced is a 20B sparse model that activates 1-4B parameters per request. It uses "Instruction-Following Pruning" (IFP): the full model is stored in NAND flash, and a dense block selects a fixed set of experts *per prompt*, reselecting periodically during generation. Apple's reason is that flash-to-DRAM bandwidth is too slow for per-token expert swapping [23].

The open-source **TurboFieldfare** runtime (2026) takes the opposite route for Gemma 4 26B-A4B. It keeps a 1.35 GB shared core plus the FP16 KV cache in RAM and streams routed experts from SSD *per token*, using an expert cache and parallel `pread`. It reports 5-6 tok/s on an 8 GB M2 MacBook Air and 31-35 tok/s on an M5 Pro [37]. Both show that **flash-streamed sparse models** are becoming the way to fit "bigger than DRAM" generators on device. The routing granularity follows directly from flash bandwidth.

### 3.5 Local inference runtimes

Outside the Apple and Windows OS models, the generator and embedder run on an open runtime. The runtime choice fixes the file format, the accelerators you can reach, and often the quantization menu.

| Runtime | Format / scope | Accelerators | Notes for RAG |
|---|---|---|---|
| llama.cpp [42] | GGUF; plain C/C++, no dependencies | CPU (NEON, AVX), Metal, CUDA, Vulkan, SYCL; Ascend NPU, OpenVINO (in progress) | 1.5- to 8-bit integer quantization; the default for laptops, edge boxes and air-gapped servers |
| MLX [43] | Python/Swift/C++ array framework | Apple silicon GPU | Unified memory, so weights, KV cache and index can share one pool without copies |
| ExecuTorch [44] | `.pte` programs from `torch.export` | XNNPACK CPU, vendor NPU/GPU/DSP backends, Cortex-M and Ethos-U | Runs Meta's Llama 3.2 quantized checkpoints (Section 3.1); reaches microcontroller-class targets |
| LiteRT-LM [45] | `.litertlm` bundles | CPU, GPU, NPU on Android, iOS, web, desktop, Raspberry Pi | Powers on-device GenAI in Chrome, Chromebook Plus and Pixel Watch; function calling, speculative decoding with MTP drafters, on-demand KV-cache growth on NPU |
| MLC LLM [46] | Compiled models via TVM | WebGPU/WASM, Metal, Vulkan, CUDA, iOS, Android | One OpenAI-compatible engine across browser and mobile |

**Lessons.** Pick the runtime per target class, not per project. A common split is llama.cpp or MLX on laptops and gateways, LiteRT-LM or ExecuTorch on phones, and MLC/WebGPU or Transformers.js in the browser. Keep the same base checkpoints across all of them so the evaluation set stays comparable. LiteRT-LM's on-demand KV-cache growth is a direct answer to the budget problem in Section 8: memory is reserved for the context a request actually uses, not for the model's maximum.

---

## 4. Embedded vector stores

### 4.1 sqlite-vec

**Problem.** Bring vector search to the database already inside nearly every app, browser and embedded Linux image, with no dependencies.

**Architecture.** sqlite-vec is a pure-C SQLite extension, a Mozilla Builders project and successor to sqlite-vss. It runs "anywhere SQLite runs": Linux, macOS, Windows, WASM in the browser, Raspberry Pi, and Android/iOS through prebuilt loadable libraries [29][31]. Vectors live in a `vec0` virtual table that stores float32, int8 or bit vectors in chunked shadow tables (`_chunks`, `_rowids`, `_vector_chunksNN`). It also supports metadata, auxiliary and partition-key columns, so KNN queries can filter by user, folder or tenant inside SQL [29]. A KNN query is ordinary SQL:

```
create virtual table vec_docs using vec0(
  doc_id integer primary key,
  user_id integer partition key,      -- shard per user / tenant
  embedding float[768],
  embedding_coarse bit[768]           -- 96 bytes per row
);
select doc_id, distance from vec_docs
 where embedding match :q and user_id = :u
 order by distance limit 10;
```

**Key techniques.**
- **Brute force by design (v0.1 stable, 2024).** The v0.1.0 release [30] was "brute-force only and meant to run on small devices" [32]. It relied on SIMD distance kernels and quantization to stay fast enough for personal-scale corpora.
- **Binary quantization with rescoring.** `vec_quantize_binary()` cuts storage 32x (a 1M x 8-dim float32 index goes from 32 MB to 1 MB). The documented pattern oversamples (for example 8x top-k) over the bit column, then rescores with exact float distance [32]. Some models, such as nomic-embed-text-v1.5 and mxbai-embed-large-v1, are trained to survive BQ [32].
- **ANN indexes (2025-2026 pre-releases).** The 0.1.10 alpha line adds `INDEXED BY rescore(quantizer=bit)`, an **IVF** index (default nlist 128, nprobe 10, packed cells capped at 64 vectors, about 200 KB per cell at 768-dim float32, sized to avoid SQLite overflow-page traversal on insert), and a **DiskANN** index with quantized neighbor vectors (`neighbor_quantizer=binary`, `n_neighbors`, `search_list_size`). Benchmarks on Cohere 768-dim 1M and 10M datasets ship in the repo [29]. The project is still pre-v1 and warns about breaking changes [29].

**Failure modes.** Brute-force latency grows linearly, so plan to switch to IVF/DiskANN or binary-plus-rescore past roughly a few hundred thousand vectors on a phone (an engineering estimate, not a published threshold). Distribution can also bite. Since v0.1.2 the project publishes precompiled loadable libraries for Android and iOS, but you still drop them into the Android Studio or Xcode project yourself; `.aar` and xcframework packaging were still listed as future work [29]. Some React Native SQLite wrappers bundle the extension, which avoids the manual step.

**Sync.** Because the index is "just SQLite," it can travel with existing SQLite replication and sync tools. The docs show it loaded into rqlite, a Raft-replicated SQLite, and the project is sponsored by Turso and SQLite Cloud, among others [29].

### 4.2 LanceDB (embedded mode)

LanceDB is built on the Lance columnar format. It has "one storage layer and two ways to put compute on top of it" [33]. In **embedded** mode, `lancedb.connect("./data")` returns a *library*, not a client: "your process opens the tables, plans the queries, runs the scans, builds the indexes, and writes the new versions," with no server, no background compaction and no server-side jobs [33]. LanceDB says the embedded mode "comfortably handles millions of vectors on one node" with disk-based indexes [33]. It supports vector search, full-text search, SQL, automatic versioning and multimodal columns [33].

**The edge-relevant property is format compatibility.** "A table written embedded is opened by a deployment, and the reverse, without conversion" [33]. That makes it a natural fit for **edge-cloud sync by file**. Index on a laptop or edge box, then push the Lance dataset to object storage for a central deployment. Or pull a cloud-built index down to an air-gapped box. Versioned manifests give you snapshot semantics for free. One caveat: embedded mode does not compact itself, so the application must schedule compaction and index rebuilds, ideally while charging or idle.

### 4.3 ObjectBox

ObjectBox bills itself as the "first on-device vector database." It is an object database for Java/Kotlin/Android (with Swift, Dart/Flutter, C/C++ and Python bindings) that added HNSW vector search in version 4.0 (2024) [34][39]. Its `@HnswIndex` annotation exposes the usual knobs [34]:
- `neighborsPerNode` (M, default 30).
- `indexingSearchCount` (efConstruction, default 100; 200+ recommended when indexing time does not matter).
- Distance type.
- A graph-repair `reparationBacklinkProbability` used after deletes.
- `vectorCacheHintSizeKB`, a non-binding cache size defaulting to 2 GB per index and allocated on demand.

The docs carry a memory warning that matters on phones: consumption "can temporarily exceed the cache size, e.g. for large changes, it can double due to multi-version transactions" [34]. Changing dimensions or HNSW parameters triggers a full re-index [34]. **ObjectBox Sync** is the differentiator. It is an offline-first replication layer that "keeps data in sync between devices and servers," so vectors and objects replicate together [34].

### 4.4 OS indexes: Core Spotlight and AppContentIndexer

Apple's **Core Spotlight** is the system-wide, on-device index that apps donate `CSSearchableItem`s to. Since 2024 it supports semantic search through `CSUserQuery` [27]. In the 2026 Siri, "personal context understanding extends to third-party apps when developers integrate with Spotlight" [24]. Apple has effectively made Spotlight the *retrieval API* for its assistant.

Windows' **AppContentIndexer** (Windows App SDK 2.5.1, a Limited Access Feature) makes a similar offer. Apps add text and images under app-defined content IDs. The system runs OCR on images, chunks text, embeds content on the NPU, and returns **one fused lexical plus semantic result set** of content IDs only [13]. The contract is clearly drawn. The app must keep the original content, detect adds, edits, moves and deletes, and re-submit failures, because the indexer "does not watch your app's files" [13]. On security, the index is protected only by "the signed-in Windows user account and the device." There is no extra encryption layer, so apps with stricter needs must add their own [13].

### 4.5 Qdrant Edge

Qdrant, otherwise a client-server vector database, added **Qdrant Edge**, a lightweight build that "runs inside the application process" instead of behind a server [47]. The unit is an `EdgeShard`, created on a local directory, with methods to upsert, query and **restore snapshots**. Data "can be synchronized with a Qdrant server" [47]. That gives a third sync shape beside LanceDB's file portability and ObjectBox's record replication. A natural deployment is for a central Qdrant cluster to build or filter a shard and for the device to restore it as a snapshot. Check the current docs for how local writes flow back to the server. The appeal is one query model and payload-filter syntax on both sides. The cost is that the edge build is tied to Qdrant's server versioning, and Python and Rust are the documented bindings [47]. Mobile coverage is thinner than ObjectBox's.

### 4.6 Browser and analytical embedded stores: PGlite and DuckDB VSS

Two more in-process options are worth knowing, each with a sharp limitation.

- **PGlite** is Postgres compiled to WASM, about 3 MB gzipped. It runs in the browser, Node, Bun and Deno with no Linux VM, and supports extensions including **pgvector**. It persists to IndexedDB in the browser or to the file system [48]. Because it is single-process Postgres in single-user mode, it suits one user's local data. Its value for RAG is schema parity: the same pgvector SQL can run in a local-first web app and in the cloud Postgres it syncs with.
- **DuckDB VSS** adds an HNSW index (built on usearch) over DuckDB's fixed-size `ARRAY` type. It is explicitly experimental. The index "is not buffer managed and must be able to fit into RAM." It has no incremental persistence, so every checkpoint re-serializes the whole index, and a restart deserializes it all back into memory [49]. It is a good fit for analytics-style local RAG over a laptop's files. It is a poor fit for a memory-capped phone or a constantly changing index.

---

## 5. Edge-specific retrieval algorithms

### 5.1 EdgeRAG (Seemakhupt, Liu, Khan, 2024)

**Problem.** On memory-constrained edge devices, specifically an NVIDIA Jetson Orin Nano class board, a flat or IVF index over a large corpus cannot keep all embeddings in memory. Swapping them in slows retrieval so much that time-to-first-token is dominated by retrieval, not generation [1].

**Architecture.** EdgeRAG starts from a two-level **IVF** index. It keeps the first-level centroids resident but **prunes the second-level embeddings**, the per-chunk vectors inside each cluster. At query time it **generates them on the fly** for only the clusters being probed, re-embedding those clusters' chunks with the local embedder. Two refinements address the long tail [1]:
- **Selective precomputation.** For clusters so large that on-the-fly embedding would blow the latency budget, embeddings are precomputed and stored.
- **Cost-aware adaptive caching.** Recently generated cluster embeddings are cached, prioritized by how expensive they are to regenerate.

```
           EdgeRAG retrieval path (after [1])
 query --> embed --> compare to centroids (resident) --> top-n clusters
                                                          |
             +--------------------------------------------+
             v
   cluster large?  --yes-->  load precomputed embeddings from storage
       | no
       v
   in cache?       --yes-->  use cached embeddings
       | no
       v
   re-embed cluster chunks on device --> insert into cost-aware cache
             |
             v
   exact search within clusters --> top-k chunks --> SLM generator
```

**Results.** The authors report about 1.8x faster TTFT on average versus a baseline IVF RAG on the edge platform, and about 3.82x on large datasets, with generation quality preserved (approximate, from the paper's abstract) [1].

**Lessons.** EdgeRAG reverses the cloud assumption that embeddings are computed once and stored forever. On a device whose NPU is idle most of the time but whose RAM is full, **recomputation can be cheaper than residency**. The pattern generalizes: keep a compact routing layer in memory, and materialize the detailed layer lazily.

### 5.2 MiniRAG (HKU Data Intelligence Lab, 2025)

**Problem.** Graph RAG systems such as GraphRAG and LightRAG rely on a strong LLM to extract entities and relations, write community summaries, and reason over long, structured contexts. With SLMs (Phi-3.5-mini, GLM-Edge-1.5B, Qwen2.5-3B, MiniCPM3-4B), these pipelines degrade badly. GraphRAG often "struggles to generate effective responses" at all [2][3].

**Architecture.** MiniRAG makes two changes [2][3]:
1. **Semantic-aware heterogeneous graph indexing.** Text chunks *and* named entities are nodes in a single graph. The SLM only has to do the easy part, extracting entity names and types. It does not have to write rich relationship descriptions or summaries.
2. **Lightweight topology-enhanced retrieval.** At query time the SLM extracts up to five query entities and "answer-type" keywords, constrained to the graph's type pool. Each entity is matched against an entity-name vector index. The system then expands each match to neighbors within 2 hops, scores candidate reasoning paths, and maps them back to chunks. The node context passed to the SLM is capped at about 500 tokens, because "if too long, SLM may fail to generate any response" (code comment) [3].

```
      MiniRAG query flow (after [2][3])
 query -> SLM: {entities<=5, answer_type_keywords}
            |                         |
            v                         v
   entity-name vector index      type pool filter
            |                         |
            +---> graph: k-hop (k=2) neighbors, path scoring
                         |
                         v
           chunk nodes linked to top paths + direct chunk hits
                         |
                         v
          compact context (~500 tok node ctx) -> SLM answer
```

**Results.** On the authors' LiHua-World benchmark (a year of synthetic chat logs for one user, with single-hop, multi-hop and summary questions), MiniRAG with Phi-3.5-mini reaches 53.29% accuracy. NaiveRAG reaches 41.22% and LightRAG 39.81% [3]. On MultiHop-RAG with Qwen2.5-3B, the figures are 48.55% against 39.48% (NaiveRAG) and 21.91% (LightRAG) [3]. The paper reports comparable performance to LLM-based methods while using **only 25% of the storage** [2]. With gpt-4o-mini, LightRAG still wins on LiHua-World (56.90% vs 54.08%) [3]. The structural simplification mainly helps *weak* generators.

**Lessons.** For SLMs, put the "intelligence" into **index topology**, not into generated text. Pick extraction tasks an SLM does reliably (NER, typing) and keep generated context very short.

### 5.3 Unified retrieve-compress-generate (ECG, 2026)

Killingback et al. propose a single small decoder (tested on SmolLM-v2 135M and Gemma 3 1B). The same multi-vector representations serve **both** retrieval (MaxSim) **and** compressed context for generation [35]. With a budget of only 16-32 representations, ECG gets over 3x the exact-match score of standard RAG on Natural Questions (for example 0.343 vs 0.106 with SmolLM). Standard ColBERT-plus-reader RAG needs 5x-16x the context to match it. ECG also halves disk use compared with storing separate retrieval and compression representations [35]. This line of work, along with xRAG, PISCO and similar compressors, targets the KV-cache bottleneck directly.

### 5.4 Framework support

Google's **AI Edge RAG SDK** (MediaPipe LLM Inference on Android) packages chunking, an on-device embedder, a local vector store and a Gemma-class generator into one pipeline [38]. It is the most direct "reference stack" from a platform vendor for app developers who do not want to write their own.

---

## 6. Apple Intelligence: semantic index plus Private Cloud Compute

### 6.1 Problem

Apple wants an assistant that answers questions over a user's messages, mail, photos, calendar and on-screen content ("personal context"). It also wants *no* server ever to retain that data, and outside researchers to be able to *verify* that claim.

### 6.2 Architecture

```
                         APPLE INTELLIGENCE (simplified)
  +------------------------------- device -------------------------------+
  | apps --donate--> Spotlight / Core Spotlight index (lexical+semantic) |
  | screen --> onscreen awareness        App Intents / "App Toolbox"     |
  |                 \                     /                              |
  |                  v                   v                               |
  |               system orchestrator (Siri)                             |
  |                 |               |                                    |
  |       on-device AFM (3B /   escalate? ---- yes ----+                 |
  |       20B sparse, adapters)                        |                 |
  +----------------------------------------------------|-----------------+
                                                       | encrypted to keys
                                                       | of attested nodes,
                                                       v via 3rd-party relay
                         +------------- Private Cloud Compute -------------+
                         | stateless inference nodes (Apple silicon, and   |
                         | from 2026 NVIDIA GPUs in Google Cloud);         |
                         | no persistent storage of request data;          |
                         | published binaries + transparency log           |
                         +-------------------------------------------------+
```

Apple's 2026 press release states that Siri "uses the system orchestrator to tap into core capabilities like the Spotlight index and App Toolbox, which work entirely on device" [24]. The retrieval side of personal RAG stays local. Only the generation step escalates, and only when needed. Third-party apps join the personal-context corpus by indexing into Spotlight [24][27]. Apple's 2024 announcement of the "semantic index" and personal Siri ran into a long delay. The "more personalized Siri" announced at WWDC 2024 slipped. That led to a class action that Apple settled for $250 million without admitting wrongdoing [28]. It is a reminder that personal-context RAG over heterogeneous app data is much harder to ship than to demo.

### 6.3 Private Cloud Compute design

Apple's June 2024 PCC design sets out five core requirements [18]:
1. **Stateless computation on personal user data.** Data is used only to fulfill the request and is not retained.
2. **Enforceable guarantees.** The guarantees come from technical mechanisms, not policy.
3. **No privileged runtime access.** There are no remote shells or debug interfaces, even for Apple SREs.
4. **Non-targetability.** An attacker cannot steer a specific user's requests to compromised nodes.
5. **Verifiable transparency.** Researchers can inspect the software images that production nodes run.

Mechanisms described in the PCC blog and Security Guide include [18][20]:
- Custom Apple-silicon servers with Secure Enclave and Secure Boot.
- A hardened, minimal OS with code signing enforced through trust caches.
- Node keys bound to attested measurements, with clients encrypting requests only to nodes whose attestations match published releases.
- Ephemeral data volumes whose encryption keys are rotated on reboot.
- Request routing through third-party OHTTP-style relays that hide the client IP.
- A public transparency log of software releases.

Apple publishes PCC source for the security-critical components: `CloudAttestation`, `CloudBoard`, `Thimble`/`privatecloudcomputed` (the on-device client daemon), `EnsembleWarden`, `CloudMetrics` and research tools [19]. That code lets researchers check the claims against the implementation [19].

**2026 evolution.** For AFM 3 Cloud Pro, Apple extended PCC to Google Cloud systems with NVIDIA GPUs. According to Apple's security blog as quoted by Simon Willison, it keeps the same patterns [25]:
- Per-request network parsing in a dedicated process and namespace.
- Shared inference software recycled with a short time-to-live.
- Attested keys held in a separate confidential VM.
- Publication of all binaries.

Apple also opened PCC-hosted Foundation Models to small developers (App Store Small Business Program, under 2M first-time downloads) at no cloud API cost [26].

### 6.4 Lessons

- **Split retrieval and generation across trust boundaries.** The index (the most sensitive, longest-lived artifact) never leaves the device. The cloud sees only the per-request prompt plus retrieved snippets, and holds nothing afterward.
- **Make privacy verifiable, not promised.** Attestation plus published binaries are what let the trust boundary extend to third-party hardware.
- **The OS index is the moat.** Spotlight donation is a retrieval API that every app already feeds.

---

## 7. Copilot+ PC Recall

### 7.1 Problem

Recall aims to let users find anything they have seen on their PC by natural-language description. It is effectively RAG over a continuously captured, screenshot-based personal corpus.

### 7.2 Architecture

With the user's opt-in, a **snapshot** of the active screen is saved "every few seconds and when the content of your active window changes" [14]. Snapshots are analyzed locally. Microsoft names an OCR model local to the PC [14]. Text and visual content become searchable, with "text and visual matches ... automatically sorted by how closely the results match your search" [14]. Results deep-link back into the source app through the UserActivity API ("relaunch"). **Click to Do** runs on top of a snapshot to act on recognized text and images [14][15].

```
             RECALL PIPELINE (simplified, after [14][15][16])
 screen change --> snapshot capture --> filters (apps, sites, private
                                         browsing, sensitive info, DRM)
                         |
                         v
           local OCR + semantic/visual embedding (NPU)
                         |
                         v
     encrypted snapshot store + vector index (local disk only)
                         |  unlocked just-in-time by Windows Hello ESS
                         v
 user query (text/voice) --> semantic + lexical search --> timeline hits
                         --> relaunch via UserActivity deep link
```

**Security architecture.** Microsoft reworked Recall in its September 2024 redesign after criticism of the June 2024 preview. In the redesign, snapshot and index data are encrypted, keys are protected by the TPM, and processing is isolated in VBS (virtualization-based security) enclaves. Access requires Windows Hello Enhanced Sign-in Security, and the feature is opt-in and removable [16]. The documented requirements back this up [14]:
- A Secured-core PC.
- Device Encryption or BitLocker.
- ESS enrollment with at least one biometric.
- Data that is never shared with Microsoft or across Windows users on the same device.

### 7.3 Operational numbers

The hardware floor is a 40 TOPS NPU, 16 GB RAM and 256 GB of storage. Recall needs 50 GB free to enable and auto-pauses below 25 GB free [14]. It is optimized for English, Simplified Chinese, French, German, Japanese and Spanish [14]. It reached general availability on Copilot+ PCs in April 2025 [15].

### 7.4 Privacy controls as retrieval filters

Privacy filters are applied at ingestion time, which is better than redacting at query time [14]:
- App and website exclusion lists.
- Private-browsing filtering in supported browsers.
- **Sensitive information filtering**, on by default, for passwords, credit cards and similar data.
- No DRM content, no audio, no continuous video.

Users can bulk-delete every snapshot from an app or site. In the EEA, users can export snapshots encrypted under a 32-character export code shown at setup. Microsoft publishes sample decryption code for third-party tools [17].

### 7.5 Failure modes and lessons

- **Continuous capture is a high-value target.** The June 2024 preview stored data in a form that drew heavy public criticism. That pushed Microsoft to enclave isolation and just-in-time decryption [16]. If you build ambient-capture RAG, treat the index like a credential store.
- **Filter at ingest.** Data you never index cannot leak through retrieval.
- **Not everything gets protected the same way.** App Content Search, the developer-facing sibling, explicitly adds *no* extra encryption beyond the user account boundary [13]. Pick the protection level that fits the data.

---

## 8. Memory and battery budgets

### 8.1 A worked budget

A typical 8 GB phone with about 4 GB available for AI [36], using sourced component sizes:

| Component | Example | Approximate resident memory |
|---|---|---|
| Generator | Llama 3.2 3B SpinQuant | ~3.7 GB RSS [6] |
| Generator (alt) | Llama 3.2 1B SpinQuant | ~1.9 GB RSS [6] |
| Generator (alt) | Gemma 3n E2B | ~2 GB accelerator memory [9] |
| Embedder | EmbeddingGemma quantized | < 200 MB [4] |
| Vector index (100K chunks) | 768-d float32 | ~300 MB (computed) |
| Vector index (100K chunks) | 768-d binary | ~9.6 MB (computed) |
| Vector index (100K chunks) | 256-d MRL int8 | ~25 MB (computed) |
| KV cache | grows with context | often the limiting term [35][36] |

A 3B generator plus embedder already uses about 4 GB. A 1B or E2B-class generator leaves room for KV cache and a mostly mmap'd index. Quantize vectors aggressively, use binary or int8 with rescoring, and memory-map the index file instead of loading it. Keep retrieved context to a few hundred tokens per chunk with small top-k.

### 8.2 Latency budget

On a phone CPU, prefill dominates. Llama 3.2 3B SpinQuant prefills at about 90 tok/s and 1B at about 260 tok/s on a OnePlus 12 [6], so every 1,000 retrieved tokens costs roughly 4-11 seconds before the first output token. Apple's reported ~0.6 ms per prompt token on iPhone 15 Pro [21] (about 0.6 s per 1,000 tokens) is roughly 18x faster than the 3B CPU path (about 11 s per 1,000 tokens). The two models and devices differ, so treat the ratio as an order of magnitude. That is why vendor NPU stacks matter so much for RAG specifically. Context compression (ECG [35]), KV sharing (Gemma 3n [9], AFM 2025 [22]) and speculative decoding (Phi Silica [12]) all attack this term.

### 8.3 Battery and scheduling

None of the vendors publish per-query joule figures in the sources reviewed here, so treat energy as something to measure yourself. The documented practices are:
- NPU-first execution (Phi Silica, Recall, AppContentIndexer semantic mode) [12][13][14].
- Hardware gating, with lexical fallback on devices without an NPU [13].
- Storage-aware throttling (Recall's 25 GB auto-pause) [14].
- OS-managed model unloading under memory pressure [12].

Standard mobile practice, not specific to these sources, is to run bulk embedding only while charging and idle, and to process changes incrementally.

---

## 9. Sync between device and cloud indexes

| Pattern | Example | What syncs | Trade-offs |
|---|---|---|---|
| Local-only, no sync | Recall [14], AppContentIndexer [13] | Nothing; per-device, per-user index | Strongest privacy; re-index per device; no cross-device recall |
| Retrieval local, generation escalates | Apple Intelligence + PCC [18][24] | Per-request prompt + snippets, never the index | Needs verifiable stateless cloud; network dependency for hard queries |
| Object/record replication | ObjectBox Sync [34] | Objects + vectors, bidirectional | Conflict resolution; vectors must be re-embedded if model changes |
| File-format portability | LanceDB embedded <-> distributed [33] | Whole versioned table/manifest | Simple for edge boxes, air-gapped imports; coarse-grained |
| SQLite replication | sqlite-vec in SQLite/libSQL replicas, rqlite [29] | Database pages/rows incl. vec0 shadow tables | Reuses mature tooling; index rebuild cost on schema change |
| Shard snapshots | Qdrant Edge `EdgeShard` [47] | Snapshot of a shard, restored on device; sync with a Qdrant server | Same query model edge and cloud; tied to one vendor's versioning |
| Schema-parity SQL | PGlite + pgvector [48] | Rows via the app's own Postgres sync layer | Same SQL locally and in the cloud; single-user, browser-scale |
| Lazy materialization | EdgeRAG [1] | Raw chunks only; embeddings regenerated on demand | Minimal storage/transfer; compute at query time |

**Design rules for sync.**
1. **Version the embedding model with the index.** A cloud index built with model A cannot be queried by a device embedder B. Ship model ID and MRL dimension as index metadata, and plan for re-embedding on upgrade.
2. **Sync source content, not just vectors, when devices differ.** Different devices may run different embedders (NPU vs. no NPU), so it is often cheaper to sync chunks and re-embed locally. That is the EdgeRAG insight applied to sync.
3. **Keep personal and shared corpora separate.** A common enterprise edge pattern runs two retrievers: a synced, read-mostly *shared* corpus (manuals, policies) pulled as a versioned snapshot, and a never-synced *personal* corpus. Merge their results at query time.
4. **Air-gapped deployments** reverse the flow. Indexes are built centrally, signed, and imported as immutable versioned artifacts. The Lance format, or a SQLite file, is a convenient unit, and integrity is checked before mounting.

---

## 10. Comparison of designs

| System | Layer | Core idea | Hardware / scale (public) | Main limitation |
|---|---|---|---|---|
| EmbeddingGemma [4] | Embedder | Gemma 3 bidirectional encoder, MRL 768->128 | 308M params, <200 MB quantized, 2K ctx | Prompt prefixes required; 2K input cap |
| Static embeddings [5] | Embedder | Attention-free token embeddings | 100-400x faster on CPU, >=85% quality | Lower ceiling on hard queries |
| Llama 3.2 1B/3B [6] | Generator | Prune + distill, SpinQuant/QLoRA 4-bit | 1B: 50 tok/s, 1.9 GB RSS (OnePlus 12) | Quantized ctx 8K; tool-use drops with PTQ |
| Phi Silica [12] | Generator | NPU-tuned SLM, speculative decoding, LoRA | Copilot+ NPUs only | ~3.5K-token context |
| Gemma 3n / 4 E-series [9][10] | Generator | PLE, MatFormer, KV sharing | E2B ~2 GB, E4B ~3 GB | Long ctx limited by device KV memory |
| AFM 3 Core Advanced [23] | Generator | 20B sparse, per-prompt experts from NAND | 1-4B active | Apple-only; coarse routing |
| sqlite-vec [29][32] | Store | vec0 virtual table, BQ+rescore, IVF/DiskANN (alpha) | Runs anywhere SQLite runs incl. WASM/mobile | Pre-v1; brute-force default |
| LanceDB embedded [33] | Store | In-process library on Lance columnar files | Millions of vectors per node | No auto-compaction in embedded mode |
| ObjectBox [34] | Store | Object DB + HNSW + Sync | 2 GB default vector cache hint | Memory can double during large txns |
| Qdrant Edge [47] | Store | In-process Qdrant shard with snapshots and server sync | Python/Rust bindings | Thinner mobile coverage |
| PGlite + pgvector [48] | Store | Postgres in WASM with pgvector | ~3 MB gzipped; browser/Node | Single-user, single-process |
| DuckDB VSS [49] | Store | Experimental HNSW over ARRAY columns | Index must fit in RAM | Full re-serialize on checkpoint |
| llama.cpp / LiteRT-LM / ExecuTorch [42][44][45] | Runtime | Portable quantized inference | CPU/GPU/NPU, laptop to microcontroller | Per-target formats (GGUF, .litertlm, .pte) |
| AppContentIndexer [13] | OS index | Managed chunk/embed/fuse, returns IDs | NPU for semantic; lexical fallback | App owns change tracking; no extra encryption |
| Core Spotlight [24][27] | OS index | App-donated items, semantic query, feeds Siri | On device | Apple-controlled model and ranking |
| EdgeRAG [1] | Algorithm | Prune IVF L2 embeddings, regenerate + cache | Jetson-class; ~1.8x avg TTFT gain | Query-time compute spikes |
| MiniRAG [2][3] | Algorithm | Heterogeneous chunk-entity graph, topology retrieval | 25% storage of LLM graph RAG | Gains shrink with strong LLMs |
| ECG [35] | Algorithm | Shared retrieval + compressed context vectors | 135M-1B decoders, 1/10 context | Research-stage; needs custom training |
| Apple PCC [18][19] | Trust boundary | Stateless, attested, transparent cloud | Apple silicon + NVIDIA/Google Cloud (2026) | Only Apple can operate it |
| Recall [14][16] | Product | Screenshot capture -> OCR/embed -> encrypted index | 40 TOPS NPU, 16 GB RAM, 50 GB free | High-value target; heavy storage |

---

## Key takeaways for system designers

1. **Budget the KV cache first, then the weights.** On 8-16 GB devices, retrieved context is the most expensive thing you add. Use small top-k, short chunks and context compression. Pick the generator after you know how much context you can afford [6][35][36].
2. **Quantize everything, and keep a rescoring path.** 4-bit weights with QAT or SpinQuant beat vanilla PTQ on general benchmarks, but on tool use only QAT (QLoRA) keeps most of the BF16 score [6]. Binary or int8 vectors with float rescoring cut index size 4-32x [32]. MRL lets one embedder serve several budgets [4].
3. **Put intelligence in index structure when the generator is weak.** MiniRAG's chunk-entity graph and type-constrained keyword extraction beat LLM-centric graph RAG when paired with SLMs [2][3].
4. **Compute can stand in for memory.** EdgeRAG's on-demand re-embedding and the per-prompt or per-token flash streaming of sparse experts (AFM 3, TurboFieldfare) show that idle accelerators plus flash bandwidth can replace DRAM residency [1][23][37].
5. **Use the OS index when you can.** Spotlight and AppContentIndexer give you NPU-scheduled embedding, hybrid lexical-semantic fusion and assistant integration. In return you own change tracking and lose model control [13][24][27].
6. **Keep the index on the device and escalate only the request.** If you need cloud generation, follow PCC's model: stateless nodes, attested keys, a relay that hides identity, published binaries [18][19][25].
7. **Filter at ingestion and encrypt at rest with hardware-bound keys.** Recall's history shows that ambient-capture RAG is judged on its worst-case leak [14][16].
8. **Design for cold starts and hardware variance.** Models get unloaded under memory pressure [12], NPUs may be missing [13], and storage runs low [14]. Every feature needs a lexical-only or smaller-model fallback.
9. **Version embedder and index together, and prefer syncing content over syncing vectors** across heterogeneous devices. Use immutable, signed index snapshots for air-gapped imports [29][33][34].
10. **Personal-context RAG is harder than it looks.** Apple's delayed personal Siri and the resulting settlement show the gap between a semantic-index demo and a dependable product [28].

---

## References

1. Korakit Seemakhupt, Sihang Liu, Samira Khan. "EdgeRAG: Online-Indexed RAG for Edge Devices." arXiv, 2024. https://arxiv.org/abs/2412.21023
2. Tianyu Fan, Jingyuan Wang, Xubin Ren, Chao Huang (HKU). "MiniRAG: Towards Extremely Simple Retrieval-Augmented Generation." arXiv, 2025. https://arxiv.org/abs/2501.06713
3. HKUDS. "MiniRAG" code repository and README (results tables, LiHua-World dataset). GitHub, 2025. https://github.com/HKUDS/MiniRAG
4. Tom Aarsen, Joshua Lochner, et al. (Hugging Face). "Welcome EmbeddingGemma, Google's new efficient embedding model." Hugging Face Blog, 2025. https://huggingface.co/blog/embeddinggemma
5. Tom Aarsen (Hugging Face). "Train 400x faster Static Embedding Models with Sentence Transformers." Hugging Face Blog, 2025. https://huggingface.co/blog/static-embeddings
6. Meta. "Llama 3.2 Model Card" (incl. quantization scheme and OnePlus 12 inference table). GitHub, 2024. https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md
7. Meta. "Llama 3.2: Revolutionizing edge AI and vision with open, customizable models." Meta AI Blog, 2024. https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/
8. Hugging Face. "Llama can now see and run on your device - welcome Llama 3.2." Hugging Face Blog, 2024. https://huggingface.co/blog/llama32
9. Hugging Face. "Gemma 3n fully available in the open-source ecosystem!" Hugging Face Blog, 2025. https://huggingface.co/blog/gemma3n
10. Google DeepMind. "google/gemma-4-E4B" model card. Hugging Face, 2026. https://huggingface.co/google/gemma-4-E4B
11. Microsoft. "Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone." arXiv, 2024. https://arxiv.org/abs/2404.14219
12. Microsoft. "Get started with Phi Silica" and "Platform card: Phi Silica." Microsoft Learn, 2025-2026. https://learn.microsoft.com/en-us/windows/ai/apis/phi-silica and https://learn.microsoft.com/en-us/windows/ai/cards/phi-silica-platform-card
13. Microsoft. "App Content Search Overview" (AppContentIndexer API). Microsoft Learn, 2026. https://learn.microsoft.com/en-us/windows/ai/apis/app-content-search
14. Microsoft. "Application card: Recall." Microsoft Learn, 2026. https://learn.microsoft.com/en-us/windows/ai/cards/recall-application-card
15. Microsoft. "Recall overview." Microsoft Learn, 2025. https://learn.microsoft.com/en-us/windows/ai/recall/
16. Microsoft (Windows Experience Blog). "Update on Recall security and privacy architecture." 2024. https://blogs.windows.com/windowsexperience/2024/09/27/update-on-recall-security-and-privacy-architecture/
17. Microsoft. "RecallSnapshotsExport" sample (decrypting EEA snapshot exports). GitHub, 2025. https://github.com/microsoft/RecallSnapshotsExport
18. Apple Security Engineering and Architecture. "Private Cloud Compute: A new frontier for AI privacy in the cloud." Apple Security Research Blog, 2024. https://security.apple.com/blog/private-cloud-compute/
19. Apple. "security-pcc" (Private Cloud Compute source code release). GitHub, 2024. https://github.com/apple/security-pcc
20. Apple. "Private Cloud Compute Security Guide." Apple Security Research, 2024. https://security.apple.com/documentation/private-cloud-compute/
21. Apple Machine Learning Research. "Introducing Apple's On-Device and Server Foundation Models." 2024. https://machinelearning.apple.com/research/introducing-apple-foundation-models
22. Apple Machine Learning Research. "Updates to Apple's On-Device and Server Foundation Language Models." 2025. https://machinelearning.apple.com/research/apple-foundation-models-2025-updates
23. Apple Machine Learning Research. "Introducing the Third Generation of Apple's Foundation Models." 2026. https://machinelearning.apple.com/research/introducing-third-generation-of-apple-foundation-models
24. Apple Newsroom. "Apple introduces Siri AI, a profoundly more capable and personal assistant." 2026. https://www.apple.com/newsroom/2026/06/apple-introduces-siri-ai-a-profoundly-more-capable-and-personal-assistant/
25. Simon Willison. "Siri AI at WWDC 2026" (quoting Apple's "Expanding Private Cloud Compute"). 2026. https://simonwillison.net/2026/Jun/8/wwdc/
26. Apple Developer. "Private Cloud Compute" (developer access and eligibility). 2026. https://developer.apple.com/private-cloud-compute/
27. Apple. "Support semantic search with Core Spotlight." WWDC24 session 10131, 2024. https://developer.apple.com/videos/play/wwdc2024/10131/
28. 9to5Mac. "iPhone users could get up to $95 per device as Apple reaches $250M settlement over Siri delays." 2026. https://9to5mac.com/2026/05/05/apple-reaches-250m-settlement-over-siri-delays-users-could-get-up-to-95-per-device/
29. Alex Garcia. "sqlite-vec" repository (README, ARCHITECTURE.md, IVF/DiskANN/rescore sources, benchmarks-ann). GitHub, 2024-2026. https://github.com/asg017/sqlite-vec
30. Alex Garcia. "Introducing sqlite-vec v0.1.0: a vector search SQLite extension that runs everywhere." 2024. https://alexgarcia.xyz/blog/2024/sqlite-vec-stable-release/index.html
31. Mozilla Hacks. "Sponsoring sqlite-vec to enable more powerful Local AI applications." 2024. https://hacks.mozilla.org/2024/06/sponsoring-sqlite-vec-to-enable-more-powerful-local-ai-applications/
32. Alex Garcia. "Binary Quantization" guide, sqlite-vec documentation. 2024. https://alexgarcia.xyz/sqlite-vec/guides/binary-quant.html
33. LanceDB. "lancedb" repository and Architecture documentation (`docs/web/architecture.mdx`, embedded vs distributed). GitHub, 2026. https://github.com/lancedb/lancedb
34. ObjectBox. "objectbox-java" repository (HnswIndex API, ObjectBox Sync). GitHub, 2024-2026. https://github.com/objectbox/objectbox-java
35. Julian Killingback, Ofer Meshi, Henry Li, Hamed Zamani, Maryam Karimzadehgan. "A Unified Model and Document Representation for On-Device Retrieval-Augmented Generation." arXiv, 2026. https://arxiv.org/abs/2604.14403
36. Martin Alderson. "Why on-device agentic AI can't keep up." 2026. https://martinalderson.com/posts/why-on-device-agentic-ai-cant-keep-up/
37. Andrey Mikhaylov (drumih). "TurboFieldfare: Gemma 4 26B-A4B in ~2 GB RAM on Apple Silicon." GitHub, 2026. https://github.com/drumih/turbo-fieldfare
38. Google AI Edge. "AI Edge RAG guide for Android." Google AI for Developers, 2025. https://ai.google.dev/edge/mediapipe/solutions/genai/rag
39. ObjectBox. "The first On-Device Vector Database: ObjectBox 4.0." ObjectBox Blog, 2024. https://objectbox.io/the-first-on-device-vector-database-objectbox-4-0/
40. Microsoft. "Phi-4-Mini Technical Report: Compact yet Powerful Multimodal Language Models via Mixture-of-LoRAs." arXiv, 2025. https://arxiv.org/abs/2503.01743
41. Hugging Face. "Welcome Gemma 4: Frontier multimodal intelligence on device." Hugging Face Blog, 2026. https://huggingface.co/blog/gemma4
42. ggml-org. "llama.cpp" repository (GGUF, 1.5- to 8-bit quantization, backends). GitHub, 2023-2026. https://github.com/ggml-org/llama.cpp
43. Apple Machine Learning Research. "MLX: An array framework for Apple silicon" repository. GitHub, 2023-2026. https://github.com/ml-explore/mlx
44. PyTorch. "ExecuTorch" repository (on-device and embedded inference, backends). GitHub, 2024-2026. https://github.com/pytorch/executorch
45. Google AI Edge. "LiteRT-LM" repository (v0.18.0 release notes, EmbeddingGemma 2, NPU KV-cache growth). GitHub, 2025-2026. https://github.com/google-ai-edge/LiteRT-LM
46. MLC AI. "MLC LLM: Universal LLM deployment engine" repository. GitHub, 2023-2026. https://github.com/mlc-ai/mlc-llm
47. Qdrant. "Qdrant Edge" (README section and documentation). 2025-2026. https://github.com/qdrant/qdrant and https://qdrant.tech/documentation/edge/
48. ElectricSQL. "PGlite: Postgres in WASM" repository. GitHub, 2024-2026. https://github.com/electric-sql/pglite
49. DuckDB. "duckdb-vss: Vector Similarity Search for DuckDB" repository. GitHub, 2024-2026. https://github.com/duckdb/duckdb-vss
