# 30. Algorithm-System Co-Design for RAG Inference: Overlap, Reuse, Compression, and Placement

## Overview

Most RAG chapters stop at the question "does the answer get better?" This one is about what the answer costs to produce and how long it takes. A RAG request runs a chain of heterogeneous stages:

- query embedding;
- approximate nearest neighbor (ANN) search, usually on CPU-resident indexes;
- optional reranking or rewriting;
- a prefill over thousands of retrieved tokens;
- autoregressive decoding, sometimes with further retrievals during it.

Each stage has a different bottleneck: memory bandwidth, PCIe, GPU FLOPs, or CPU cores. The retrieved context is also mostly redundant across requests. In a production trace characterized by the RAGCache authors, 2% of documents accounted for 60% of requests [4].

Since 2023 a body of work has treated this chain as one system to co-design rather than a vector database bolted onto an LLM server. The designs fall into five families:

1. **Overlap retrieval with computation.** PipeRAG [2], RaLMSpec [3], TeleRAG [10], Stream2LLM [13], and the speculative pipelining inside RAGCache [4] hide retrieval latency behind generation or prefill.
2. **Reuse the KV cache of retrieved documents.** RAGCache [4], CacheBlend [5], Prompt Cache [34], APE [35], KVLink [36], TurboRAG [8], Block-Attention [9], CacheGen [16], and LMCache [6], on substrates such as Mooncake [38] and Dynamo [39], avoid recomputing prefill for documents seen before, including when they appear at non-prefix positions.
3. **Shrink or restructure what the generator reads.** Speculative RAG [1] splits documents across parallel small drafters. REFRAG [7] replaces chunk tokens with precomputed chunk embeddings, and xRAG [37] compresses each document to a single token.
4. **Place and partition hardware jointly.** Chameleon [11], VectorLiteRAG [12], HedraRAG [14], and RAGO [15] split work across CPU, GPU, and FPGA, decide how much of the ANN index lives in HBM next to the KV cache, and size batches per stage.
5. **Adapt configuration per query.** METIS [17] picks the number of chunks and the synthesis method per query, jointly with scheduling.

The chapter covers each design in turn: the problem, the architecture, the key techniques, the published numbers, failure modes, and lessons. It ends with a latency and cost model for TTFT and throughput, a comparison table, and design takeaways. Unless stated otherwise, numbers are the authors' own reported results on their own testbeds and are not directly comparable across papers.

A note on sourcing: arXiv, conference sites, and vendor blogs could not be reached directly while this chapter was compiled or during its fact-check pass. Facts were drawn from the full text of the Speculative RAG paper [1], from extracted full texts and proceedings mirrors of RAGCache [4], TeleRAG [10], Stream2LLM [13], and RAGO [15], from the official code repositories, and from detailed paper notes hosted on GitHub. During fact-checking, numbers for RaLMSpec, TurboRAG, Block-Attention, CacheBlend, CacheGen, Chameleon, VectorLiteRAG, METIS, REFRAG, Stream2LLM, and the Speculative RAG reproduction were re-checked against those repositories and notes. Where a number comes only from a secondary summary, it is marked approximate or attributed to that summary.

```
            Where RAG latency comes from (one request, non-iterative)

 query --> [embed] --> [ANN search] --> [rerank] --> [prefill: Q + k docs] --> [decode]
             ~ms        10s ms to s      ~10s ms     dominant for long ctx   per token
                         (CPU/GPU/       (small LM)   O(L) FLOPs, O(L^2)     memory-bw
                          FPGA)                       attention               bound
   overlap targets:   PipeRAG, RaLMSpec, TeleRAG, Stream2LLM, RAGCache-DSP
   reuse targets:                                    RAGCache, CacheBlend, TurboRAG,
                                                     Block-Attn, CacheGen, LMCache
   shrink targets:                                   REFRAG, Speculative RAG
   placement targets: Chameleon, VectorLiteRAG, HedraRAG, RAGO (all stages)
```

## 1. Where the time goes: characterization results

Several papers start by profiling. Their findings set up the rest of the chapter.

- **Retrieval latency is not negligible, and it varies widely.** RAGCache reports that vector search latency ranges from tens of milliseconds to a few seconds, and can match or exceed generation time when high recall is required [4]. RAGO finds that the bottleneck shifts by paradigm. Hyperscale retrieval can spend over 80% of end-to-end time in retrieval, while in long-context configurations retrieval is under 1% of latency [15].
- **Prefill dominates generation for long retrieved contexts.** On Llama2-7B with a single A10G, generation reaches about one second at roughly 4,000 input tokens, dominated by prefill [4]. The CacheBlend authors cite about 3 s (34B-class) and about 6 s (70B) for a 4K-token prefill on an A40 [5].
- **Small models can become the bottleneck.** In long-context processing, a database encoder 100x smaller than the main LLM can dominate because it must process many more tokens [15].
- **Document access is highly skewed.** On MMLU-based RAG, 60% of requests reference 3% of documents. On a production dataset ("Dataset-X"), 2% of documents account for 60% of requests [4]. VectorLiteRAG observes that the top 20% of IVF clusters receive roughly 60% of accesses on Wiki-All and over 93% on ORCAS [12].
- **ANN search converges early.** About 95% of requests already have their final top-k after 13% of the search [4]. This makes speculative early-start possible.
- **CPU retrieval inflates TTFT at scale.** With a 128M-vector index, CPU retrieval raised TTFT for Llama3-8B from 197 ms to 606 ms in VectorLiteRAG's measurements [12].

These observations motivate the techniques that follow. Skew suggests caching, early convergence suggests speculation, prefill dominance suggests KV reuse or compression, and variable bottlenecks suggest per-workload scheduling.

## 2. Speculative RAG: small drafters, one big verifier

### Problem

Standard RAG concatenates all retrieved documents into one long prompt for one large model. That is slow, because prefill grows with the number of documents. It is also error-prone, because of position bias over long contexts. Self-reflective methods such as Self-RAG and CRAG improve quality but add instruction tuning and extra generation steps, which adds latency [1].

### Architecture

Speculative RAG (Google Cloud AI Research, Google DeepMind and UC San Diego; ICLR 2025) borrows the shape of speculative decoding, but at the level of whole answers rather than tokens [1].

```
                   retrieved docs D (n docs)
                            |
            embed (instruction-aware encoder) + K-Means into k clusters
                            |
           sample 1 doc per cluster -> m subsets delta_1..delta_m (k docs each)
          /                 |                  \
   [Drafter 7B #1]   [Drafter 7B #2]  ...  [Drafter 7B #m]     (parallel endpoints)
     alpha_1, beta_1   alpha_2, beta_2        alpha_m, beta_m   (answer, rationale)
          \                 |                  /
           [Generalist verifier: one forward pass per draft, no docs in prompt]
            rho_j = rho_draft * rho_self-consistency * rho_self-reflection
                            |
                    answer = argmax_j rho_j
```

- **Multi-perspective sampling.** Retrieved documents are embedded with InBedder-RoBERTa, clustered with K-Means into k topics, and one document per cluster forms each subset. This minimizes redundancy within a subset and diversifies across subsets [1].
- **RAG drafter.** A Mistral-7B instruction-tuned to produce an answer draft plus a rationale from its subset. Drafts run in parallel on separate endpoints [1].
- **RAG verifier.** An off-the-shelf generalist (Mistral-7B or Mixtral-8x7B, not fine-tuned). The verifier never reads the documents. It scores each (answer, rationale) pair with a self-consistency probability P(alpha, beta | Q) and a self-reflection probability P("Yes" | Q, alpha, beta, R), both computed in a single forward pass. The final score multiplies these with the drafter's own generation probability [1].

### Numbers

Experiments ran on 16 A100-40GB GPUs with vLLM, greedy decoding, and no batching. The verifier ran at tensor parallelism 4. There were 5 drafter endpoints, or 10 for MuSiQue. Against the strongest standard-RAG baseline (Mixtral-Instruct-8x7B), Speculative RAG cut latency by 11.90% on TriviaQA, 15.07% on MuSiQue, 44.31% on PopQA, 50.83% on PubHealth, and 22.77% on ARC-Challenge. Accuracy improved by up to 12.97% on PubHealth [1].

### Failure modes and lessons

- The speedup depends on **parallel drafter capacity**. The paper runs 5–10 drafter endpoints. An independent single-A100 reproduction that did not use the paper's fine-tuned drafter or multi-GPU parallelism found Speculative RAG slower and less accurate than standard RAG: 62% vs 80% exact match and 9.18 s vs 6.02 s p50 latency on TriviaQA. Once drafting was accelerated with vLLM, retrieval became the dominant cost, at about 7.05 s of 9.13 s p50 [18].
- Quality hinges on a **distilled drafter**. Without instruction tuning for rationale generation, the verifier has little signal to work with [1][18].
- The lesson is that the design trades **aggregate compute** (m drafts) for **critical-path latency** (shorter prompts per draft plus one verification pass). It wins when GPUs are idle and per-request latency matters, and loses in throughput-bound fleets.

## 3. Overlapping retrieval with generation

### 3.1 PipeRAG (Amazon; KDD 2025)

**Problem.** In RETRO-style models that retrieve periodically during generation (every m tokens), generation stalls for each retrieval. The retrieval interval and the search quality (IVF nprobe) are fixed independently of hardware [2][19].

**Architecture and techniques** [2][19][20]:

- **Pipeline parallelism using stale queries.** The retrieval for the next chunk is issued with a query window that lags the current generation context by s tokens. Retrieval therefore runs concurrently with generating the current chunk, instead of after it.
- **Flexible retrieval intervals.** The attention mechanism of RETRO is modified so the model can integrate retrieved content at shorter intervals than the 64-token chunks it was trained on. This keeps the pipeline full.
- **Performance-model-driven retrieval.** A profiled performance model of the generator, the retriever, and the SBERT encoder predicts the largest nprobe whose retrieval latency still hides behind generating the next token chunk. Search quality expands to fill the overlap window rather than adding latency.
- **Implementation.** Generator and retriever communicate over gRPC. The code base is built on an open-source RETRO implementation [20].

```
 time ->
 RETRO (serial):  [gen chunk i][retrieve i+1][gen chunk i+1][retrieve i+2] ...
 PipeRAG:         [gen chunk i      ][gen chunk i+1    ][gen chunk i+2 ] ...
                  [retrieve(stale q_i - s)][retrieve(stale)] ...  nprobe sized to fit
```

**Numbers.** Up to 2.6x end-to-end generation speedup over RETRO with improved generation quality [2][19]. Some secondary summaries also cite a perplexity reduction of about 0.93 at the same latency budget. The digest used for this chapter does not repeat that figure, so treat it as unconfirmed.

**Failure modes.** Stale queries trade a little relevance for overlap. The design targets encoder-decoder or cross-attention retrieval (RETRO). Stream2LLM's authors argue it does not transfer directly to causal decoder-only serving stacks [13].

**Lesson.** Treat retrieval quality (nprobe) as a variable to fit the latency hidden behind generation, not a fixed hyperparameter.

### 3.2 RaLMSpec (ICML 2024)

**Problem.** Iterative RaLMs (for example In-Context RALM, FLARE, Iter-RetGen) and KNN-LMs retrieve every few tokens, so retrieval sits on the critical path many times per request [3][21].

**Architecture** [3][21]:

- **Speculative retrieval from a request-local cache.** Each request keeps a small cache of previously retrieved documents. Thanks to temporal and spatial locality, the next retrieval is guessed from the cache and generation proceeds immediately.
- **Batched verification.** Every few speculative steps (the stride), the real retriever runs once on the batched queries. If any speculated document differs, the system rolls back to the first mismatch and regenerates. Outputs are therefore identical to the non-speculative model.
- **Optional optimizations.** Cache prefetching (pulling extra top-k candidates into the cache), an Optimal Speculation Stride Scheduler (OS3) that adapts the stride to observed acceptance rates and latencies, and asynchronous verification that overlaps verification with the next speculative step.

```
 spec step 1   spec step 2   spec step 3   | verify (batch 3 queries) | accept or rollback
 cache->doc    cache->doc    cache->doc    | KB retrieval            |
 gen tokens    gen tokens    gen tokens    |                         |
```

**Numbers** (repository README) [21]. Document-level RaLM speedups over the baseline were 1.75–2.39x with an exact dense retriever, 1.04–1.39x with an approximate dense retriever, and 1.31–1.77x with a sparse retriever. KNN-LM speedups reached up to 7.59x (exact) and 2.45x (approximate). Models included GPT-2, OPT-1.3B, and Llama-2 7B/13B/70B.

**Failure modes.** Gains shrink when the retriever is already fast (approximate dense retrieval). Rollbacks waste compute when locality is low, for example on topic-shifting generations.

**Lesson.** Speculation plus verification gives exactness guarantees. That makes it the "safe" overlap technique when outputs must not change.

### 3.3 RAGCache's dynamic speculative pipelining

RAGCache splits vector search into stages: by cluster for IVF, by time slice for HNSW. When an intermediate top-k is available, it starts LLM prefill speculatively on those documents. If the final top-k differs, the speculative run is terminated and restarted. Because 95% of searches converge within 13% of the search, most speculation succeeds. The system turns speculation on or off depending on load. The reported gains are up to 1.6x TTFT reduction and a 1.5–4.3x cut in non-overlapped search time [4].

### 3.4 TeleRAG (UW; MLSys 2026)

**Problem.** Keeping a large IVF index in GPU memory is too expensive, because it competes with weights and KV cache. Searching on the CPU is slow, and copying clusters on demand over PCIe stalls the query [10].

**Architecture.** TeleRAG uses **lookahead retrieval**, a prefetching mechanism that predicts which index data (IVF clusters) a query will need and moves them from CPU to GPU in parallel with LLM generation. In pipelines with a query rewrite or pre-retrieval generation step, the transfer hides behind that step. A **prefetching scheduler** and a **cache-aware scheduler** extend this to multi-GPU serving with low overhead. The library is released as `ragacc` [10][22].

**Numbers.** Up to 1.98x average end-to-end latency reduction for single-query inference, and up to 1.83x higher average throughput for batched inference, with minimal GPU memory [10].

**Lesson.** PCIe transfer of the hot part of an index is a schedulable resource. Overlap it with the LLM steps that precede retrieval.

### 3.5 Stream2LLM (Georgia Tech; MLSys 2026)

**Problem.** Retrieval often produces context incrementally. Crawlers append pages over seconds, and ANN refinement updates results. Waiting for the full context before prefill wastes time [13].

**Architecture** [13]:

- Built on vLLM v1 for the prefill tier of a prefill/decode-disaggregated deployment.
- **Append mode** (context grows) and **update mode** (context replaced as ANN refines).
- **LCP-based invalidation.** On an update, it keeps KV for the longest common prefix of old and new inputs and invalidates the rest.
- **Two-phase scheduler.** Phase 1 ranks requests and checks feasibility without allocating anything. Phase 2 acquires blocks and preempts the lowest-priority requests.
- **Cost-based preemption.** Recompute or swap to CPU is chosen per request from offline hardware profiles.

**Numbers.** Setup: Llama-3.1-8B on H100/H200 with TP=2. Median TTFT improved 3.9–4.3x over non-streaming baselines on the crawler workload at low load (QPS 0.5–1.0), and 10.8–11.0x at QPS 4.0 (mean retrieval latency 9.9 s). On ANNS, P95 TTFT improved 2.49–2.63x at QPS 1.0 (mean retrieval latency 4.5 s). Under memory pressure, naive vLLM streaming hurts the tail: P99 falls to 0.19x of baseline on ANNS and 0.71x on the crawler. FCFS or last-chunk-arrival scheduling with cost-based eviction instead keeps P99 gains of about 2x (ANNS) to 8.6–9.1x (crawler). In ANNS more than 10% of requests invalidate over 10,000 tokens, so update mode can waste substantial prefill [13].

**Lesson.** Streaming prefill needs **scheduler awareness**. Without it, early-started requests hog KV blocks and hurt the tail.

## 4. Reusing KV caches of retrieved knowledge

The single largest lever on TTFT is not recomputing prefill for documents that were already processed. The hard part is that standard prefix caching (vLLM automatic prefix caching, SGLang RadixAttention [23]) only matches **exact prefixes**. In RAG, the same document appears at different positions, after different system prompts and different preceding documents.

```
 prefix cache hit requires identical token prefix:
   req A: [sys][doc7][doc3][q_A]      req B: [sys][doc3][doc7][q_B]
           ^^^^^^^^^^^ hit for A only once [sys][doc7] cached; B misses on doc3
 non-prefix reuse (CacheBlend/TurboRAG/Block-Attn): reuse doc3 and doc7 KV anywhere,
   then repair cross-attention (selective recompute) or train model to not need it
```

### 4.1 RAGCache (Peking University and ByteDance; ACM TOCS)

**Architecture** [4]:

- **Knowledge tree.** A prefix tree keyed by document IDs. Because KV tensors depend on the order of preceding documents, each root-to-node path is one ordered document sequence. Nodes point to vLLM-style paged KV blocks, so inserts and evictions copy no data.
- **Two-tier memory.** Hot nodes live in GPU memory and warm nodes in host memory. Each node is swapped out to host at most once, and the host copy is retained.
- **PGDSF replacement.** Prefix-aware Greedy-Dual-Size-Frequency weighs frequency, size, recency, and a prefix-aware recomputation cost, with separate logical clocks for GPU and host tiers.
- **Cache-aware request reordering.** Requests are prioritized by the ratio of cached length to compute length, within a 32-request window to bound starvation.
- **Dynamic speculative pipelining** with vector search (Section 3.3).

```
                     root (system prompt)
                    /        |         \
                 doc12      doc3       doc40          <- GPU tier (hot)
                /    \        \
            doc3    doc8     doc12                    <- host tier (warm, swapped once)
            |
          doc8                                         path = ordered doc sequence
```

**Numbers.** Main testbed: one A10G with 192 GiB host cache. RAGCache cut TTFT by up to 4x (1.2–4x) and raised throughput up to 2.1x versus vLLM plus Faiss. Against SGLang it cut TTFT up to 3.5x and raised throughput up to 1.8x. On the production dataset, TTFT fell 1.9–2.6x versus vLLM. For Mixtral-8x7B and Llama2-70B on two H800s, TTFT fell 1.4–2.1x versus vLLM. PGDSF raised hit rate 1.06–1.62x over LRU. Scheduling overhead was under 1 ms [4].

**Failure modes.** Order sensitivity fragments the tree: the same document set in a different order is a different path. Host-memory bandwidth (PCIe) bounds how fast warm nodes return. RAGCache keeps exact-prefix semantics, so it gains nothing for a hot document behind a cold one.

### 4.2 CacheBlend (University of Chicago and collaborators; EuroSys 2025)

**Problem.** Full KV reuse of independently precomputed chunks (PromptCache-style) ignores cross-attention between chunks, which degrades quality on multi-hop questions. Prefix caching only helps the first chunk [5][24].

**Architecture** [5][24]:

- **Per-chunk KV store.** Each reusable chunk's KV is stored independently and looked up by chunk hash.
- **Selective KV recompute.** In each layer, only a small set of **high-KV-deviation (HKVD)** tokens is recomputed. These are the tokens whose KV would differ most from the cached version under the full context. The rest reuse cached KV. Compute scales with the recompute ratio.
- **Gradual filtering.** The first layer selects a somewhat larger candidate set. Later layers compute deviation only within the previous layer's candidates and narrow it.
- **Loading controller.** It picks the recompute ratio and the storage tier (GPU, CPU RAM, SSD) so that recomputing layer i overlaps with loading layer i+1's KV. The default floor is about 15%, since roughly 10–15% of tokens show markedly higher deviation.
- **Implementation.** About 3K lines on vLLM. It is now part of LMCache as "non-prefix KV reuse" [6][33].

**Numbers** (per the paper, as summarized in [24]). TTFT fell 2.2–3.3x versus both full recompute and prefix caching. At similar TTFT, throughput rose 2.8–5x versus full recompute and up to 3.3x versus prefix caching. Quality stayed within about 0.01–0.03 (F1 or Rouge-L) of full recompute, and was 0.15–0.35 higher than full KV reuse. For a 7B model with a 4K context at 15% recompute, recompute takes about 3 ms per layer while loading a layer's KV from NVMe (measured at 4.8 GB/s) takes about 16 ms. At 7B scale, SSD loading is therefore the binding constraint. For 70B the figures flip (about 7 ms recompute vs 4 ms load), so recompute dominates. Models tested: Mistral-7B, Yi-34B, and Llama-70B (the latter two 8-bit quantized) on two A40s.

**Failure modes.** The ratio floor is empirical. Queries needing heavy cross-chunk reasoning may need more recompute. Baselines in the paper assume zero loading latency for prefix caching, which flatters the baseline, but secondary reviewers note averages only, with no P99 under I/O jitter [24]. Caches for multiple models or versions multiply storage.

**Lesson.** Position-independent reuse plus a small, attention-guided repair recovers nearly all quality. Design the storage tier and the recompute ratio together.

### 4.3 TurboRAG and Block-Attention: train the model to not need cross-chunk attention

Both take the opposite approach to CacheBlend. They change the model so that independently computed chunk KVs are valid as they stand.

- **TurboRAG** (Moore Threads) precomputes chunk KV caches offline (512-token chunks in the reference code). It modifies attention masks so chunks attend only within themselves, rearranges position IDs, and fine-tunes the LLM to handle this layout. It reports up to 9.4x (average 8.6x) TTFT reduction with comparable multi-document QA accuracy, and about 98.5% less inference compute for the retrieved part [8][25]. The README's own example shows 0.65 s vs 4.13 s TTFT for 10 documents [8].
- **Block-Attention** splits the prompt into blocks that compute KV independently, except the last block (the query), which attends to all. It re-encodes rotary positions when blocks move, and fine-tunes on mixed block and full attention for about 500–1,000 steps. It reports 98.7% TTFT reduction and 99.8% FLOPs-to-first-token reduction at 32K tokens, with accuracy comparable to full attention [9][26].

**Trade-off.** No runtime repair cost, but a model fine-tune per base model. Every model upgrade invalidates all cached KVs and needs re-training. Static corpora suit this best [25].

### 4.4 CacheGen and LMCache: KV as a network object

When KV lives off-GPU (CPU, SSD, remote store), **bytes transferred** become the TTFT term. CacheGen (SIGCOMM 2024) encodes KV with token-group delta coding, layer-wise quantization (finer in shallow layers), and arithmetic coding. Chunks are stored at several compression levels, and the level is adapted per chunk to measured bandwidth, falling back to text plus recompute when bandwidth is poor. It reports 3.5–4.3x smaller KV than default quantization and 3.1–4.7x lower TTFT than sending text, at a 3 Gbps setting [16].

LMCache is the open-source engine that productizes this line. It is a KV management layer, running as a standalone daemon, that offloads to CPU, disk, and remote backends (Redis/Valkey, S3-compatible). Its README lists CPU RAM, local SSD, Redis/Valkey, Mooncake, InfiniStore, S3-compatible object storage, NIXL, and GDS as backends. It supports non-prefix reuse by applying CacheBlend's selective recompute, integrates with vLLM, and was announced as integrated into NVIDIA Dynamo in September 2025. It is described in the paper "LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference" (Cheng et al., arXiv:2510.09665) [6].

### 4.5 Prompt Cache, APE, and KVLink: the middle ground

Between CacheBlend (runtime repair, no model change) and TurboRAG or Block-Attention (full fine-tune), several designs reuse independently encoded segments with little or no training.

- **Prompt Cache** (Yale; MLSys 2024) is the "PromptCache-style" baseline that CacheBlend argues against. A schema language declares reusable prompt modules, such as documents, system prompts, and templates, along with their position IDs. Their attention states are precomputed once and spliced into new prompts without recomputing cross-module attention. The repository's single persona demo shows TTFT falling from 286.9 ms to 78.2 ms on an RTX 4090 (about 3.7x) and from about 17.9 s to 0.51 s on an i9-13900K CPU (about 35x). These are demo figures, not benchmark averages [34].
- **APE: Adaptive Parallel Encoding** (ICLR 2025) encodes retrieved contexts in parallel and in a training-free way. It realigns the attention distribution of the parallel-encoded segments toward what sequential encoding would produce, using a shared prefix, an attention temperature, and a scaling factor. The README reports that the default settings (temperature and scale of 0.9) keep over 90% of sequential-encoding performance on few-shot tasks [35]. It is a good fit when the base model cannot be fine-tuned and no recompute budget is available.
- **KVLink** (UCSB; arXiv 2025) precomputes each document's KV independently. At inference it re-encodes positions for the concatenated layout and appends a few trainable "link tokens" per document, which restore cross-document attention. Only a light fine-tune is needed, and the released KVLink5 checkpoints are on Hugging Face [36]. Conceptually it is a learned version of CacheBlend's repair step: a few new tokens are added rather than a subset of existing ones recomputed.

**Lesson.** The design space for non-prefix reuse is a continuum. One axis is how much cross-chunk attention is restored. The other is where the cost of restoring it is paid: never (Prompt Cache), with a fixed analytic correction (APE), with a few learned tokens (KVLink), with selective runtime recompute (CacheBlend), or with a full fine-tune (TurboRAG and Block-Attention).

## 5. Compressing what the decoder reads: REFRAG

### Problem

Even with KV reuse, decoding still attends over every retrieved token. Attention cost and KV memory scale with context length. Retrieved passages are also mostly unrelated to each other, so the attention matrix is close to block-diagonal [7][27].

### Architecture (Meta Superintelligence Labs, NUS and Rice; 2025)

```
 retrieved passages --> split into k-token chunks
        |
   [light encoder: RoBERTa-base/large]  -> 1 embedding per chunk   (query-independent,
        |                                                    precomputable/cacheable)
   [2-layer MLP projection] -> decoder token-embedding space
        |
   RL policy: which chunks to EXPAND back to full tokens (selective compression)
        |
 decoder input = [question tokens] + [chunk embeddings ... | expanded chunk tokens]
        |
   [LLaMA-2-7B decoder]  -> answer          input length shrinks ~k-fold
```

- **Compress-sense-expand.** Each k-token chunk becomes one projected embedding that the decoder consumes in place of the k tokens [7][27].
- **Training recipe.** First a reconstruction task (decoder frozen, encoder and projection learn to reconstruct the chunk), with a curriculum from one chunk to many. Then continual pre-training on next-paragraph prediction, with the decoder unfrozen, on about 20B Slimpajama tokens. Skipping reconstruction or the curriculum makes training fail [27].
- **RL selective compression.** A GRPO/PPO-style policy chooses which chunks stay uncompressed, with negative perplexity as the reward. It beats heuristic and random selection [27].

### Numbers

With k=32, TTFT is 30.85x faster than LLaMA and 3.75x faster than the prior compressor CEPE, with matching perplexity. With k=16, TTFT is about 16.5x faster than LLaMA with cached chunk embeddings and about 2x faster than CEPE, with better perplexity than CEPE. Compression rates of 8–32 are practical, while k=64 degrades noticeably. The paper describes extending the usable context of LLaMA-2-7B from 4k to 16k+ tokens. At equal latency (8 passages vs 1), average accuracy over 16 RAG tasks improves 1.22% with a strong retriever and 1.93% with a weak one [7][27][28].

### Failure modes and lessons

- **Requires continual pre-training and fine-tuning** of the decoder. It cannot be applied off the shelf, and results are mostly on LLaMA-2-7B [27].
- Chunk embeddings are tied to the encoder/decoder pair. A model update means re-embedding the corpus.
- The lesson: KV reuse removes prefill compute but not decode-time context length. Compression removes both, at the cost of model coupling. The two are composable, since precomputed chunk embeddings are themselves a cache.

### Related: one-token compression with xRAG

xRAG (NeurIPS 2024) takes compression to the extreme. Rather than one embedding per k-token chunk, it feeds the frozen retriever's document embedding through a small trained projector (a "modality bridge") as one token. The retriever and the LLM both stay frozen [37]. The paper's abstract reports an average improvement of over 10% across six knowledge-intensive tasks relative to no retrieval, with about 3.5x fewer FLOPs than uncompressed RAG. These figures could not be re-checked against the paper for this edition, so treat them as approximate. The trade-off sits at the opposite end from REFRAG. Reusing the existing dense-index vectors means no separate chunk encoder and no corpus re-embedding, but one token cannot carry fine-grained facts as reliably as one embedding per 16–32 tokens. REFRAG's selective expansion is a way of buying that detail back where it matters.

## 6. GPU-resident ANN, co-location, and heterogeneous placement

### 6.1 Chameleon (ETH Zurich; VLDB 2025)

A disaggregated, heterogeneous accelerator system for retrieval-augmented LMs [11][29][32]:

- **ChamVS.** IVF index scanning on GPU, then broadcast to FPGA near-memory nodes that scan PQ codes using BRAM lookup tables and a systolic priority queue (one element every two cycles), followed by aggregation. An approximate hierarchical queue shrinks the level-one queue from K=100 to 20 while returning identical results for 99% of queries.
- **ChamLM.** Multi-GPU LM inference that consumes ChamVS results.
- **Numbers.** The FPGA retrieval accelerator is up to 23.72x faster and up to 26.2x more energy-efficient than CPU-based search. End-to-end, latency falls up to 2.16x and throughput rises up to 3.18x versus hybrid CPU-GPU [29].
- **Lesson.** Retrieval and LM inference scale differently. Retrieval is memory-capacity- and bandwidth-bound, while LM inference is compute- and HBM-bound. Disaggregation lets each scale on the right silicon.

### 6.2 VectorLiteRAG (Georgia Tech; HPCA 2026)

**Problem.** GPU ANN (for example NVIDIA cuVS/CAGRA [30]) is fast, but HBM spent on the index is HBM not spent on KV cache, which caps LLM batch size [12].

**Architecture** [12][31]:

- Hybrid IVF. The coarse quantizer and cold clusters stay on the CPU. A minimal set of **hot clusters** is cached in sharded GPU sub-indexes.
- A **latency-bounded partitioner** finds the smallest GPU-resident share of the index that meets the search SLO. It models per-batch hit rate with a Beta distribution, and adjusts the LLM batch size to balance the pipeline.
- Per-shard routing tables and a **dynamic dispatcher** that releases early-finishing queries, cutting search latency up to 16%.
- Online re-partitioning as access patterns drift, in under a minute.

**Numbers.** Up to 1.5x higher SLO-bound throughput with no extra hardware. Under a 400 ms search SLO it placed 31.5% of the index on GPU, versus 73% for HedraRAG's policy. Index sizes ranged from about 18 GB (Wiki-All, 88M vectors) to 80 GB (ORCAS 2K). The abstract also claims 2x better vector search responsiveness [12][31].

### 6.3 HedraRAG (SOSP 2025)

HedraRAG targets heterogeneous workflows (multi-round, branching, iterative) whose CPU retrieval and GPU generation stages interleave unpredictably [14]:

- **RAGraph.** A graph representation of each request's workflow. Graph transforms (node split, edge addition, dependency rewiring) are applied across concurrent requests.
- **Fine-grained sub-stage partitioning** with dynamic batching, so long stages do not stall short ones.
- **Semantic-aware reordering with speculative execution.** It uses the similarity of consecutive query embeddings within a request to start dependent stages early.
- **Partial GPU index caching** with asynchronous updates, exploiting hot clusters.
- **Wavefront scheduling** that aligns CPU vector search batches with GPU decode iterations.
- **Numbers.** Over 1.5x and up to 5x throughput versus existing RAG frameworks [14].

### 6.4 RAGO (ETH Zurich and Google; ISCA 2025)

**RAGSchema** describes a RAG workload: the pipeline (document encoder, query rewriter, reranker, generator) plus configuration (model sizes, database size, query vectors per retrieval, retrieval frequency). **RAGO** searches the scheduling space for it [15]:

- **Task placement.** Collocate or disaggregate components across accelerators.
- **Resource allocation.** Accelerator type and count per component, with retrieval on CPU servers.
- **Batching policy.** A separate batch size for retrieval and for each inference stage, trading throughput against latency.

It uses an analytical cost model of a generic systolic-array accelerator ("XPU") to trace the Pareto frontier. The authors report up to 2x QPS per chip and 55% lower TTFT versus RAG systems built as extensions of LLM-only serving [15].

### 6.5 Mooncake and KV-aware serving stacks: the substrate for document KV reuse

The KV-reuse designs in Section 4 assume a place to keep many gigabytes of document KV and a way to route requests to the worker that already holds it. Two production systems now provide this substrate.

- **Mooncake** (Moonshot AI and Tsinghua; FAST 2025 Best Paper) is the serving platform behind Kimi. It is KV-cache-centric: prefill and decode clusters are disaggregated, and the otherwise underused CPU, DRAM, and SSD across the GPU cluster are pooled into a distributed KV cache. A scheduler places requests to maximize cache reuse within TTFT and TPOT SLOs. The project reports that Kimi handles 75% more requests under SLOs on real workloads. Its Transfer Engine reaches up to 87 GB/s on 4x200 Gbps RoCE and 190 GB/s on 8x400 Gbps RoCE [38]. For RAG, a pooled cache of this kind makes RAGCache's host tier or CacheBlend's per-chunk store a cluster-wide resource rather than a per-node one, and LMCache can use Mooncake as a backend [6].
- **NVIDIA Dynamo** is an orchestration layer above vLLM, SGLang, and TensorRT-LLM. It provides **KV-aware routing**, which sends a request to the worker whose cache overlaps most with its prompt, weighted by load (the project claims about 2x faster TTFT). It also provides a **KV Block Manager** that offloads KV from GPU to CPU, SSD, and remote or object storage, plus disaggregated prefill and decode [39]. For RAG, KV-aware routing is the cluster-level version of RAGCache's cache-aware reordering: put a request where its documents' KV already lives.

**Lesson.** Document KV reuse at fleet scale is a distributed-storage and routing problem as much as an attention problem. Hit ratio h in Section 8 depends on routing locality, and T_kv_load depends on the interconnect (RDMA vs PCIe vs object storage).

### 6.6 Retrieval-aware batching, consolidated

The batching lessons scattered across these systems are:

- Batch sizes should be **per stage**, because retrieval and LLM stages have different throughput-latency curves [15].
- **Align CPU search batches with GPU decode iterations** (wavefronts), so that neither side waits [14].
- **Reorder for cache hits**, prioritizing by cached-to-compute ratio within a bounded window [4], or using radix-tree-aware scheduling [23].
- **Batch verification** in speculative schemes, amortizing one real retrieval over several speculative steps [3].
- Make batches **config-aware**. METIS chooses each query's chunk count and synthesis method jointly with GPU memory, avoiding configurations that look fast but would queue behind out-of-memory limits. It reports 1.64–2.54x lower response delay than an AdaptiveRAG-style baseline with no F1 loss, and 1.8–4.5x higher serving capacity than Parrot- and vLLM-based baselines at a fixed 1.8 s delay [17].

## 7. Comparison table

| System (venue/year) | Stage targeted | Core mechanism | Model change? | Exact outputs? | Headline reported result |
|---|---|---|---|---|---|
| Speculative RAG (ICLR'25) [1] | Prefill + generation | Parallel small drafters on doc subsets; generalist verifies | Fine-tuned drafter | No | Up to -50.83% latency, +12.97% acc (PubHealth) |
| PipeRAG (KDD'25) [2] | Iterative retrieval | Stale-query prefetch, flexible intervals, perf-model nprobe | Attention tweak (RETRO) | No | Up to 2.6x vs RETRO |
| RaLMSpec (ICML'24) [3] | Iterative retrieval | Local-cache speculation + batched verification + rollback | No | Yes | 1.04–2.39x RaLM; up to 7.59x KNN-LM |
| RAGCache (TOCS) [4] | Prefill + retrieval | Knowledge tree, PGDSF, reordering, speculative pipelining | No | Yes (prefix) | Up to 4x TTFT, 2.1x thpt vs vLLM+Faiss |
| CacheBlend (EuroSys'25) [5] | Prefill | Non-prefix chunk KV reuse + ~15% selective recompute | No | Approx. | 2.2–3.3x TTFT, 2.8–5x thpt |
| TurboRAG (2024) [8] | Prefill | Precomputed chunk KV, independent attention, fine-tune | Fine-tune | No | Up to 9.4x TTFT (avg 8.6x) |
| Block-Attention (2024) [9] | Prefill | Blockwise independent KV, position re-encoding | Fine-tune | No | -98.7% TTFT at 32K |
| CacheGen (SIGCOMM'24) [16] | KV transfer | KV codec + bandwidth-adaptive streaming | No | Approx. | 3.1–4.7x TTFT vs text |
| REFRAG (2025) [7] | Prefill + decode | Chunk embeddings replace tokens; RL selective expansion | CPT + fine-tune | No | 30.85x TTFT (k=32) |
| TeleRAG (MLSys'26) [10] | Retrieval | Lookahead CPU-to-GPU IVF prefetch | No | Yes | 1.98x latency, 1.83x thpt |
| Stream2LLM (MLSys'26) [13] | Retrieval + prefill | Streaming prefill, LCP invalidation, cost-based preempt | No | Yes | 3.9–11x TTFT (crawler) |
| Chameleon (VLDB'25) [11] | Retrieval | FPGA near-memory PQ scan, disaggregated GPUs | No | Yes | 2.16x latency, 3.18x thpt |
| VectorLiteRAG (HPCA'26) [12] | Retrieval + KV HBM | Hot-cluster GPU partition sized to SLO | No | Yes | 1.5x SLO throughput |
| HedraRAG (SOSP'25) [14] | Whole workflow | RAGraph transforms, wavefront batching, GPU index cache | No | Mostly | 1.5–5x throughput |
| RAGO (ISCA'25) [15] | Whole pipeline | Placement, allocation, per-stage batching search | No | Yes | 2x QPS/chip, -55% TTFT |
| METIS (SOSP'25) [17] | Config + scheduling | Per-query chunk count and synthesis selection | No | No | 1.64–2.54x lower delay |
| Prompt Cache (MLSys'24) [34] | Prefill | Schema-declared reusable prompt modules | No | Approx. | ~3.7x GPU, ~35x CPU TTFT (demo) |
| APE (ICLR'25) [35] | Prefill | Training-free parallel encoding with attention rescaling | No | Approx. | >90% few-shot quality retained |
| KVLink (2025) [36] | Prefill | Per-document KV + link tokens, positional re-encoding | Fine-tune | No | Training-light non-prefix reuse |
| xRAG (NeurIPS'24) [37] | Prefill | One-token projected document embedding | Projector only | No | ~3.5x fewer FLOPs (abstract) |
| Mooncake (FAST'25) [38] | KV storage + P/D | KV-centric disaggregated pool over CPU/DRAM/SSD | No | Yes | +75% requests under SLO (Kimi) |

"Exact outputs" means the system reproduces the baseline model's output given the same retrieved documents. "Approx." means bounded quality loss.

## 8. End-to-end latency and cost models for TTFT and throughput

The papers above each use internal performance models: PipeRAG's nprobe predictor [2], CacheBlend's loading controller [5], RAGO's XPU model [15], Stream2LLM's recompute-vs-swap profiles [13], and VectorLiteRAG's Beta hit-rate model [12]. The synthesis below is this chapter's own framing. It combines their terms into a first-order model a designer can fill in with measured constants.

### 8.1 TTFT for one non-iterative request

```
TTFT ~= T_embed(q)
      + T_search(index, nprobe, placement)          # CPU / GPU / FPGA
      + T_rerank(k)                                  # small model, many tokens
      + T_kv_load(bytes_hit / BW_tier)               # 0 if no reuse; PCIe/NVMe/net
      + T_prefill((1 - h) * L_ctx + r * h * L_ctx + L_q)
      + T_queue
      - T_overlap                                    # hidden by pipelining
where
  L_ctx   = retrieved tokens (k chunks x chunk length)
  h       = fraction of context tokens with reusable KV (cache hit ratio)
  r       = recompute ratio on hits (0 for exact prefix; ~0.15 CacheBlend; 0 TurboRAG)
  T_overlap = min(T_search, T_pre-retrieval-LLM) (TeleRAG), or speculative prefill
              on partial results (RAGCache), or streaming chunks (Stream2LLM)
```

Reading the terms against published data:

- **T_prefill** is roughly linear in tokens for short-to-medium contexts and superlinear at long lengths. A 4K-token prefill is about 1 s on Llama2-7B with an A10G [4] and about 3–6 s for 34B–70B on an A40 [5]. Cutting effective tokens by (1-h)+rh is why KV reuse dominates TTFT gains (2–4x for RAGCache and CacheBlend [4][5]). Cutting tokens by a factor of k is why REFRAG gets about 16–31x [7].
- **T_kv_load** can exceed the savings. In CacheBlend's own estimate for a 7B model, loading a layer's KV from NVMe (about 16 ms) exceeds recomputing 15% of it (about 3 ms) [5][24]. Rule of thumb: KV per token is about 2 x layers x kv_heads x head_dim x bytes. Small models have cheap prefill, so reuse from slow tiers may not pay. Large models have expensive prefill, so reuse from SSD pays. CacheGen-style compression reduces bytes 3.5–4.3x [16].
- **T_search** depends on placement and nprobe. CPU retrieval over 128M vectors tripled TTFT in VectorLiteRAG's setup (197 ms to 606 ms) [12]. Its share ranges from under 1% to over 80% of latency depending on paradigm [15].
- **T_overlap** is bounded by the shortest concurrent stage. Overlap helps most when retrieval and the preceding LLM step are of similar duration [2][10].

### 8.2 Iterative retrieval during decoding

For a generation of N tokens with retrieval every m tokens:

```
T_total ~= N * TPOT + (N/m) * T_retrieve * (1 - overlap_fraction)
           + rollback_cost   (speculative: (1 - accept_rate) * stride * TPOT)
```

PipeRAG drives overlap_fraction toward 1 by choosing nprobe so that T_retrieve is at most the time to generate one interval [2]. RaLMSpec replaces most T_retrieve terms with cache lookups and pays a rollback term proportional to misses [3]. KNN-LM (retrieval per token, m=1) is where speculation pays most, with up to 7.59x [21].

### 8.3 Throughput and cost per query

At steady state, throughput is the minimum over pipeline stages:

```
QPS ~= min( QPS_search(cores or GPU share, batch_s),
            QPS_prefill(GPU FLOPs, effective tokens),
            QPS_decode(HBM bandwidth, batch_d limited by free KV memory) )
cost/query ~= sum_over_resources( price_per_hour / 3600 / QPS_at_that_resource )
```

- **HBM is shared.** Every GB of ANN index, or of cached document KV, held in HBM shrinks the decode batch. VectorLiteRAG and RAGCache both solve this jointly: the minimal hot-cluster footprint in one case [12], GPU/host tiers with PGDSF in the other [4].
- **Reuse raises prefill throughput** (CacheBlend: 2.8–5x at similar TTFT [5]) but adds storage cost (RAM/SSD per cached chunk per model version) and I/O bandwidth.
- **Per-stage batching** lets each stage sit at its own knee of the latency-throughput curve. RAGO's 2x QPS per chip comes from this kind of placement and batching search [15].
- **Speculative RAG** moves cost the other way: m drafter passes plus one verifier pass, versus one long-context pass. It lowers latency when drafter GPUs would otherwise idle, and raises cost per query when the fleet is throughput-bound [1][18].

### 8.4 Measurement checklist

Report TTFT at P50/P95/P99 under load, not single-request averages. Stream2LLM shows that naive designs can win P50 and lose P99 by 5x under memory pressure [13]. Also report TPOT (time per output token), hit ratio h, the recompute ratio r, bytes moved per tier, index HBM footprint, and quality deltas on multi-hop tasks, where cross-chunk attention matters most [5].

## Key takeaways for system designers

1. **Profile before optimizing.** Retrieval can be under 1% or over 80% of latency depending on corpus size, retrieval frequency, and context length [15]. Small auxiliary models (encoders, rerankers) can be the bottleneck [15].
2. **Exploit skew everywhere.** A few percent of documents or clusters serve most requests [4][12]. That justifies caching document KV (RAGCache) and caching hot IVF clusters in HBM (VectorLiteRAG, HedraRAG, TeleRAG).
3. **Prefix caching is not enough for RAG.** Document order varies per query. Choose between runtime repair (CacheBlend, about 15% recompute, no model change) [5] and training the model for position-independent chunks (TurboRAG, Block-Attention) [8][9]. Repair keeps model freedom; training removes runtime cost but couples caches to a model version. Training-free parallel encoding (APE) and learned link tokens (KVLink) sit between the two [35][36].
4. **Overlap is cheap latency.** Start prefill on partial or streamed results [4][13], prefetch index data during query rewriting [10], or issue retrievals with stale queries [2]. Use speculation with verification when outputs must stay exact [3].
5. **Treat retrieval quality as an elastic knob.** Fit nprobe or chunk count to the latency budget that overlap hides [2], or adapt it per query [17].
6. **HBM is the contested resource.** Index shards, document KV caches, and decode KV all compete for it. Size them jointly against SLOs [12][4].
7. **Tiered KV needs bandwidth math.** Loading from SSD or network can cost more than recomputing for small models [5]. Compress KV for transfer [16] and pipeline layer-wise loading with compute [5].
8. **Compression changes the asymptotics.** REFRAG-style chunk embeddings shrink both prefill and decode context by a factor of k [7], but require continual pre-training and lock the corpus embeddings to a model pair [27].
9. **Speculative RAG trades compute for latency.** It needs parallel drafter capacity and a well-distilled drafter. Without them it can lose on both speed and accuracy [1][18].
10. **Schedule for the tail.** Cache-aware reordering [4], bounded windows, and FCFS-style priorities under memory pressure [13] matter as much as the core mechanism.
11. **Expect composition.** The families stack: per-stage batching (RAGO) + hot-cluster GPU ANN (VectorLiteRAG) + streamed prefill (Stream2LLM) + non-prefix KV reuse (CacheBlend/LMCache) + compressed chunks (REFRAG). Production stacks such as LMCache with vLLM, Mooncake's pooled KV store, and Dynamo's KV-aware routing are where these are converging [6][38][39].

## References

1. Speculative RAG: Enhancing Retrieval Augmented Generation through Drafting. Zilong Wang, Zifeng Wang, Long T. Le, et al. (UC San Diego, Google Cloud AI Research, Google DeepMind). ICLR 2025 (arXiv 2024). https://arxiv.org/abs/2407.08223
2. PipeRAG: Fast Retrieval-Augmented Generation via Algorithm-System Co-design. Wenqi Jiang, Shuai Zhang, Boran Han, Jie Wang, Bernie Wang, Tim Kraska (Amazon). KDD 2025 (arXiv 2024). https://arxiv.org/abs/2403.05676
3. Accelerating Retrieval-Augmented Language Model Serving with Speculation (RaLMSpec). Zhang et al. ICML 2024 (arXiv 2024). https://arxiv.org/abs/2401.14021
4. RAGCache: Efficient Knowledge Caching for Retrieval-Augmented Generation. Chao Jin, Zili Zhang, Xuanlin Jiang, Fangyue Liu, Shufan Liu, Xuanzhe Liu, Xin Jin (Peking University, ByteDance). arXiv 2024; ACM TOCS. https://arxiv.org/abs/2404.12457
5. CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion. Jiayi Yao, Hanchen Li, Yuhan Liu, Siddhant Ray, Yihua Cheng, Qizheng Zhang, Kuntai Du, Shan Lu, Junchen Jiang. EuroSys 2025 (arXiv 2024). https://arxiv.org/abs/2405.16444
6. LMCache: KV cache management layer (repository; paper arXiv:2510.09665). LMCache project. 2024–2026. https://github.com/LMCache/LMCache
7. REFRAG: Rethinking RAG based Decoding. Xiaoqiang Lin, Aritra Ghosh, Bryan Kian Hsiang Low, Anshumali Shrivastava, Vijai Mohan (Meta Superintelligence Labs, NUS, Rice). arXiv 2025. https://arxiv.org/abs/2509.01092
8. TurboRAG: Accelerating Retrieval-Augmented Generation with Precomputed KV Caches for Chunked Text (code and README). Songshuo Lu, Hua Wang, Yutian Rong, Zhi Chen, Yaohua Tang (Moore Threads). 2024. https://github.com/MooreThreads/TurboRAG
9. Block-Attention for Efficient Prefilling. Dongyang Ma, Yan Wang, Lan Tian. arXiv 2024. https://arxiv.org/abs/2409.15355
10. TeleRAG: Efficient Retrieval-Augmented Generation Inference with Lookahead Retrieval. Chien-Yu Lin, Keisuke Kamahori, et al., Baris Kasikci (University of Washington). MLSys 2026 (arXiv 2025). https://arxiv.org/abs/2502.20969
11. Chameleon: a Heterogeneous and Disaggregated Accelerator System for Retrieval-Augmented Language Models. Wenqi Jiang, Marco Zeller, Roger Waleffe, Torsten Hoefler, Gustavo Alonso (ETH Zurich). VLDB 2025 (arXiv 2023). https://arxiv.org/abs/2310.09949
12. VectorLiteRAG: Latency-Aware and Fine-Grained Resource Partitioning for Efficient RAG. Junkyum Kim, Divya Mahajan (Georgia Tech). HPCA 2026 (arXiv 2025). https://arxiv.org/abs/2504.08930
13. Stream2LLM: Overlap Context Streaming and Prefill for Reduced Time-to-First-Token. Rajveer Bachkaniwala, Chengqi Luo, Richard So, Divya Mahajan, Kexin Rong (Georgia Tech). MLSys 2026 (proceedings text mirror). https://github.com/bojieli/ai-infra-book/blob/main/references/proceedings/MLSys/2026/text/mlsys2026-3b3889d313ba9476c12c2d77ea66b24f.txt
14. HedraRAG: Co-Optimizing Generation and Retrieval for Heterogeneous RAG Workflows. Zhengding Hu, Vibha Murthy, Zaifeng Pan, Wanlu Li, Xiaoyi Fang, Yufei Ding, Yuke Wang. SOSP 2025 (paper notes). https://github.com/lqhl/awesome-system-papers/blob/main/wiki/papers/HedraRAG-SOSP25.md
15. RAGO: Systematic Performance Optimization for Retrieval-Augmented Generation Serving. Wenqi Jiang, Suvinay Subramanian, Cat Graves, Gustavo Alonso, Amir Yazdanbakhsh, Vidushi Dadu (ETH Zurich, Google, Google DeepMind). ISCA 2025 (arXiv 2025). https://arxiv.org/abs/2503.14649
16. CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving. Yuhan Liu, Hanchen Li, et al., Junchen Jiang. SIGCOMM 2024 (paper notes). https://github.com/lqhl/awesome-system-papers/blob/main/wiki/papers/CacheGen-SIGCOMM24.md
17. METIS: Fast Quality-Aware RAG Systems with Configuration Adaptation. Siddhant Ray, Rui Pan, Zhuohan Gu, Kuntai Du, Shaoting Feng, Ganesh Ananthanarayanan, Ravi Netravali, Junchen Jiang. SOSP 2025 (paper notes). https://github.com/lqhl/awesome-system-papers/blob/main/wiki/papers/METIS-SOSP25.md
18. speculative-rag: single-A100 reproduction of Speculative RAG. soyoon-cu (independent). 2026. https://github.com/soyoon-cu/speculative-rag
19. PipeRAG paper digest (2403.05676). memgrafter/research-digests. 2026. https://github.com/memgrafter/research-digests/blob/main/ml_research_analysis_2024/2403.05676_piperag-fast-retrieval-augmented-generation-via-algorithm-system-co-design_20260212_051629.md
20. PipeRAG official code. Amazon Science. 2024. https://github.com/amazon-science/piperag
21. RaLMSpec official code (ralm-sys). JackFram et al. 2024. https://github.com/JackFram/ralm-sys
22. TeleRAG official code. UW SyFI Lab. 2026. https://github.com/uw-syfi/TeleRAG
23. SGLang: Efficient Execution of Structured Language Model Programs (RadixAttention). Lianmin Zheng et al. arXiv 2023. https://arxiv.org/abs/2312.07104
24. CacheBlend (EuroSys'25) paper notes, with design and numbers. lqhl/awesome-system-papers. 2025. https://github.com/lqhl/awesome-system-papers/blob/main/wiki/papers/CacheBlend-EuroSys25.md
25. TurboRAG paper digest (2410.07590). memgrafter/research-digests. 2026. https://github.com/memgrafter/research-digests/blob/main/ml_research_analysis_2024/2410.07590_turborag-accelerating-retrieval-augmented-generation-with-precomputed-kv-caches-for-chunked-text_20260212_050452.md
26. Block-Attention paper digest (2409.15355). memgrafter/research-digests. 2026. https://github.com/memgrafter/research-digests/blob/main/ml_research_analysis_2024/2409.15355_block-attention-for-efficient-prefilling_20260212_122742.md
27. REFRAG literature summary. datanooblol/claude-literature-review. 2025. https://github.com/datanooblol/claude-literature-review/blob/main/summary/REFRAG%20Rethinking%20RAG%20based%20Decoding.md
28. REFRAG-style reference implementation (compress, sense/select, expand). simulanics. 2025. https://github.com/simulanics/REFRAG
29. Chameleon paper digest (2310.09949). memgrafter/research-digests. 2026. https://github.com/memgrafter/research-digests/blob/main/ml_research_analysis_2023/2310.09949_chameleon-a-heterogeneous-and-disaggregated-accelerator-system-for-retrieval-augmented-language-models_20260215_233652.md
30. cuVS: GPU-accelerated vector search and clustering (CAGRA). NVIDIA RAPIDS. 2024–2026. https://github.com/rapidsai/cuvs
31. VectorLiteRAG paper notes (2504.08930). Crew-Scaler/multi_agent_system_risks. 2025. https://github.com/Crew-Scaler/multi_agent_system_risks/blob/main/papers/2504.08930.md
32. Chameleon RAG acceleration official code. fpgasystems (ETH Zurich). 2025. https://github.com/fpgasystems/Chameleon-RAG-Acceleration
33. CacheBlend official code. Jiayi Yao et al. 2024. https://github.com/YaoJiayi/CacheBlend
34. Prompt Cache: Modular Attention Reuse for Low-Latency Inference. In Gim, Guojun Chen, Seung-seob Lee, Nikhil Sarda, Anurag Khandelwal, Lin Zhong (Yale). MLSys 2024 (arXiv:2311.04934). https://github.com/yale-sys/prompt-cache
35. APE: Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding. Xinyu Yang, Tianqi Chen, Beidi Chen (CMU). ICLR 2025 (arXiv:2502.05431). https://github.com/Infini-AI-Lab/APE
36. KVLink: Accelerating Large Language Models via Efficient KV Cache Reuse. Jingbo Yang, Bairu Hou, Wei Wei, Yujia Bao, Shiyu Chang. arXiv 2025 (arXiv:2502.16002). https://github.com/UCSB-NLP-Chang/KVLink
37. xRAG: Extreme Context Compression for Retrieval-augmented Generation with One Token. Xin Cheng et al. NeurIPS 2024 (arXiv:2405.13792). https://github.com/Hannibal046/xRAG
38. Mooncake: Trading More Storage for Less Computation, A KVCache-centric Architecture for Serving LLM Chatbot. Ruoyu Qin et al. (Moonshot AI, Tsinghua). FAST 2025 (Best Paper). https://github.com/kvcache-ai/Mooncake
39. NVIDIA Dynamo: datacenter-scale distributed inference framework (KV-aware routing, KV Block Manager, disaggregated serving). NVIDIA. 2025–2026. https://github.com/ai-dynamo/dynamo
