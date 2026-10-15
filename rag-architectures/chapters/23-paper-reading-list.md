# 23. Canonical RAG and Retrieval Paper Reading List: 100+ Papers (2017-2026) by Theme, with Venue, Link and System-Design Takeaway

## Overview

This chapter is a curated reading list for engineers who design retrieval-augmented generation (RAG) systems. Each paper is here for one of three reasons. It introduced a component that is now standard (dense bi-encoders, late interaction, HNSW, cross-encoder rerankers). It established a failure model that production systems design against (lost in the middle, distractor sensitivity, insufficient context). Or it defined a benchmark or evaluation method that teams still use. The list runs from retrieve-then-read open-domain QA in 2017 (DrQA) to RL-trained search agents and the 2026 work that systematizes agentic RAG.

Each entry gives the venue (or "arXiv" for preprints with no confirmed venue), the year, a citation number that resolves to a full URL in the References section, and a one-line system-design takeaway. Where a paper's abstract, repository or the authors' own blog reports headline numbers, they are quoted. Numbers taken from secondary sources are marked approximate. arXiv identifiers are given inline, so `arXiv:2005.11401` is `https://arxiv.org/abs/2005.11401`.

The chapter has fifteen themes:

1. Foundations: retrieve-then-read and retrieval-augmented language models
2. Retrievers: dense, sparse, late-interaction and general-purpose embedders
3. Approximate nearest-neighbour (ANN) indexing
4. Reranking
5. Query transformation and synthetic training data
6. Indexing granularity, chunking and hierarchy
7. Iterative, adaptive and corrective retrieval
8. RL-trained search agents and deep research (2025-2026)
9. Graph and structured RAG
10. Generator training and context utilization
11. Compression, caching and serving efficiency
12. Multimodal (visual document) retrieval
13. Evaluation and benchmarks
14. Surveys and engineering experience reports
15. Security, privacy and adversarial robustness

The map below places these themes on the canonical pipeline. Most production designs can be described as choices made at each box.

```
                         OFFLINE (indexing)
  docs --> parse --> chunk/propositions --> embed ------> ANN index (HNSW/DiskANN/IVF)
             |        (Dense X, late      (DPR, E5,     (HNSW, FAISS, ScaNN, SPANN)
             |         chunking, RAPTOR,   ColBERT,
             |         contextual retr.)   SPLADE)  \--> sparse index (BM25/SPLADE)
             \--> entity/relation extraction --> graph + communities (GraphRAG, HippoRAG)

                         ONLINE (query time)
  query --> rewrite/expand --> retrieve (hybrid) --> rerank --> compress --> generate
            (HyDE, Query2doc,   (RRF fusion)     (monoT5,   (RECOMP,     (FiD, RAG,
             step-back)                           RankGPT)   LLMLingua)   Self-RAG)
     ^                                                                        |
     |____ iterate / route / abstain (IRCoT, FLARE, CRAG, Search-R1, CoRAG) __|

  serving: KV reuse (RAGCache, CacheBlend), drafting (Speculative RAG)
  evaluation: BEIR, MTEB, KILT, RAGAS, ARES, RGB, BRIGHT, FRAMES, TREC RAG
```

## 1. Foundations: retrieve-then-read and retrieval-augmented LMs

**Problem.** Parametric language models store knowledge in their weights. That knowledge is hard to update, hard to attribute and costly to scale. The foundational papers show that a non-parametric memory (an index over text) can be queried at inference time and, in several designs, trained jointly with the reader.

**Architecture lineage.** The line runs from a TF-IDF retriever plus a reader (DrQA) to learned latent retrieval (ORQA, REALM), then to bi-encoder dense retrieval (DPR), then to end-to-end retrieve-and-generate (RAG, FiD), and finally to retrieval baked into pretraining at trillion-token scale (RETRO, Atlas). In 2023 the field shifted to retrieval as an in-context plug-in for frozen black-box LLMs (In-Context RALM, REPLUG). That last shift is the architecture almost every production system uses today.

| Paper | Venue / year | Ref | System-design takeaway |
|---|---|---|---|
| DrQA: Reading Wikipedia to Answer Open-Domain Questions | ACL 2017 | [1] | The two-stage retriever + reader split is the root of every RAG pipeline. Retrieval recall caps end-to-end accuracy. |
| ORQA: Latent Retrieval for Weakly Supervised Open Domain QA | ACL 2019 | [2] | A retriever can be learned from QA pairs alone, with the evidence as a latent variable. Pretraining the retriever (inverse cloze task) matters. |
| REALM: Retrieval-Augmented LM Pre-Training | ICML 2020 | [3] | Retrieval can be part of pretraining. Refreshing the index asynchronously while the encoder trains is the core systems problem. |
| Dense Passage Retrieval (DPR) | EMNLP 2020 | [4] | A plain dual encoder trained with in-batch negatives beat Lucene BM25 by 9-19 points absolute on top-20 passage accuracy. A later replication found BM25 was stronger than first reported, so the gap is smaller [4]. |
| RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP | NeurIPS 2020 | [5] | The paper that named the field. DPR retriever + BART generator over a Wikipedia index (about 21M passages), with RAG-Sequence and RAG-Token marginalization. It showed that an index swap updates world knowledge without retraining. |
| Fusion-in-Decoder (FiD) | EACL 2021 | [6] | Encode each passage separately and fuse in the decoder. Cost grows linearly with passage count, so readers can use many more passages. This is the template for "many chunks, one answer". |
| kNN-LM: Generalization through Memorization | ICLR 2020 | [7] | Interpolating next-token distributions with nearest neighbours from a datastore improves perplexity. It is the token-level ancestor of datastore-backed LMs. |
| RETRO: Improving LMs by Retrieving from Trillions of Tokens | ICML 2022 | [8] | With a 2-trillion-token database, RETRO matched GPT-3 and Jurassic-1 on the Pile with 25x fewer parameters. It uses a frozen BERT retriever and chunked cross-attention. Retrieval substitutes for parameters. |
| Atlas: Few-shot Learning with Retrieval Augmented LMs | JMLR 2023 | [9] | Contriever retriever + FiD reader, jointly pretrained. Over 42% on Natural Questions with 64 examples, beating a 540B model by 3 points with 50x fewer parameters [9]. |
| In-Context Retrieval-Augmented LMs (In-Context RALM) | TACL 2023 | [10] | Prepend retrieved text to the prompt of an unmodified LM. No architecture change is needed. This is the design most deployed RAG systems use. |
| REPLUG: Retrieval-Augmented Black-Box LMs | NAACL 2024 | [11] | Treat the LLM as a black box. Ensemble over retrieved documents, and train the retriever from LM feedback (LSR). The retriever can be tuned when the generator cannot. |
| KILT: a Benchmark for Knowledge Intensive Language Tasks | NAACL 2021 | [12] | One Wikipedia snapshot shared by 11 datasets spanning five task types (fact checking, entity linking, slot filling, open QA, dialogue), with provenance scoring. It made the evaluation of retrieval and generation joint. |

**Failure modes these papers exposed.** Joint training (REALM, Atlas) requires re-embedding the corpus when the encoder changes. That cost is why production systems freeze the document encoder and adapt the query side or the reranker instead. RETRO's gains depend partly on test-set leakage into the retrieval database, which the paper analyzes. Treat any "smaller model plus retrieval" comparison with a contamination check.

## 2. Retrievers: dense, sparse, late-interaction and general embedders

**Problem.** Single-vector dense retrievers compress a passage into one point. They are fast but lose term-level precision and generalize unevenly out of domain. Lexical BM25 is robust but misses paraphrase. This research line explores three trade-offs: expressiveness against storage (late interaction), interpretability against recall (learned sparse), and general-purpose against domain-tuned (instruction embedders).

```
  single-vector bi-encoder     late interaction (ColBERT)      learned sparse (SPLADE)
  q -> [v]  d -> [v]           q -> [v1..vn] d -> [u1..um]     q,d -> |V|-dim sparse
  score = <v_q, v_d>           score = sum_i max_j <vi,uj>     score = <w_q, w_d>
  1 vector / passage           1 vector / token (compressed)   inverted index, BM25-like
  cheapest, weakest OOD        best quality/storage trade-off  exact-match + expansion
```

| Paper | Venue / year | Ref | System-design takeaway |
|---|---|---|---|
| ColBERT: Contextualized Late Interaction over BERT | SIGIR 2020 | [13] | Per-token embeddings with MaxSim scoring approach cross-encoder quality while document embeddings are still precomputed at index time. The README cites search "in tens of milliseconds" [13]. |
| ColBERTv2: Lightweight Late Interaction | NAACL 2022 | [14] | Residual compression cut the MS MARCO index from 154 GiB to 16-25 GiB, a 6-10x reduction. Denoised distillation from a cross-encoder improved quality. |
| PLAID: Efficient Engine for Late Interaction | CIKM 2022 | [15] | Centroid interaction prunes candidates before residual decompression. Retrieval engines, not only models, decide late-interaction latency. |
| ANCE: Approximate Nearest Neighbor Negative Contrastive Learning | ICLR 2021 | [16] | Hard negatives mined from an asynchronously refreshed ANN index substantially improve dense retrievers over in-batch or BM25 negatives. |
| SPLADE: Sparse Lexical and Expansion Model | SIGIR 2021 | [17] | Learned term weights plus expansion run on an inverted index. Existing Lucene/Elastic infrastructure gets neural recall. |
| SPLADE v2 | arXiv 2021 | [18] | Distillation and FLOPS regularization tune the sparsity/latency trade-off explicitly. |
| Contriever: Unsupervised Dense IR with Contrastive Learning | TMLR 2022 | [19] | Contrastive pretraining without labels gives a strong zero-shot retriever. Self-RAG and Atlas use it as their default retriever. |
| E5: Text Embeddings by Weakly-Supervised Contrastive Pre-training | arXiv 2022 | [20] | Large-scale weakly supervised pairs plus "query:"/"passage:" prefixes. It became the default retriever in RL search-agent work (Search-R1, CoRAG). |
| GTE: General Text Embeddings with Multi-stage Contrastive Learning | arXiv 2023 | [21] | Multi-stage contrastive training on mixed sources gives general-purpose embedders that compete with much larger models. |
| BGE M3-Embedding | Findings ACL 2024 | [22] | One model emits dense, sparse and multi-vector outputs. Hybrid retrieval no longer needs three separate models. |
| Matryoshka Representation Learning | NeurIPS 2022 | [23] | Nested embeddings can be truncated to fewer dimensions with graceful quality loss. This enables adaptive storage tiers and coarse-to-fine search. |
| Promptriever: Instruction-Trained Retrievers | ICLR 2025 | [24] | Retrievers can follow per-query natural-language instructions, which moves "what counts as relevant" into the query. |
| ReasonIR: Training Retrievers for Reasoning Tasks | arXiv 2025 | [25] | Synthetic reasoning-heavy training data targets BRIGHT-style queries where semantic similarity fails. |
| Agentic-R: Learning to Retrieve for Agentic Search | arXiv 2026 | [26] | It trains a retriever for multi-turn agents, scoring passage utility by local relevance plus global answer correctness. Retrievers for agents need different supervision than retrievers for one-shot QA. |
| On the Theoretical Limitations of Embedding-Based Retrieval (LIMIT) | arXiv 2025 | [115] | Shows that the number of top-k document combinations a single-vector embedder can represent is bounded by its dimension, and builds a simple dataset (LIMIT) on which strong embedders fail while BM25 and multi-vector models do far better. A theoretical argument for keeping sparse or late-interaction paths in hybrid stacks. |

## 3. Approximate nearest-neighbour indexing

**Problem.** Exact maximum-inner-product search over millions to billions of vectors is too slow. Every vector database is a set of engineering choices over the algorithms below: graph-based (HNSW, Vamana/DiskANN), quantization-based (IVF-PQ, ScaNN's anisotropic quantization) and hybrid memory/disk (SPANN, DiskANN).

| Paper | Venue / year | Ref | System-design takeaway |
|---|---|---|---|
| HNSW: Hierarchical Navigable Small World graphs | TPAMI (arXiv 2016) | [27] | A multi-layer proximity graph gives log-like search. It is the default in-memory index in most vector DBs, but its RAM footprint grows with the graph degree. |
| Billion-scale similarity search with GPUs (FAISS) | IEEE TBD 2019 (arXiv 2017) | [28] | GPU IVF-PQ with k-selection made billion-scale brute force and IVF practical. |
| The Faiss library | arXiv 2024 | [29] | A design-space paper for index composition (IVF, PQ, HNSW, refinement). Read it before choosing vector DB parameters. |
| ScaNN: Accelerating Large-Scale Inference with Anisotropic Vector Quantization | ICML 2020 | [30] | Quantization loss should be weighted by its effect on inner product, not reconstruction error. |
| DiskANN: Billion-point search on a single node | NeurIPS 2019 | [31] | SSD-resident Vamana graph: a billion points on a 64 GB RAM workstation, over 5,000 QPS at under 3 ms mean latency with 95%+ 1-recall@1 on SIFT1B (16 cores). It holds 5-10x more points per node than HNSW/NSG at high recall [31]. |
| SPANN: Highly-efficient Billion-scale ANN Search | NeurIPS 2021 | [32] | Posting lists on disk with centroids in memory. Balanced partitions and boundary duplication control tail latency. |
| FreshDiskANN: Fast and Accurate Graph-Based ANN Index for Streaming Similarity Search | arXiv 2021 | [114] | Supports concurrent inserts and deletes on a Vamana graph with background merges into the SSD index. Index freshness is an ANN design property, not only a pipeline concern. |
| Filtered-DiskANN: Graph Algorithms for ANN Search with Filters | WWW 2023 | [113] | Builds label-aware graphs so that metadata-filtered queries (tenant, ACL, date) keep high recall. Naive post-filtering of an unfiltered graph collapses recall for selective filters. |

**Lesson.** ANN recall is a retrieval-quality hyperparameter, not just a latency knob. The DiskANN paper reports that competing billion-scale methods with similar memory plateau near 50% 1-recall@1 [31]. A RAG system on such an index silently loses half its true nearest neighbours before any reranker runs.

## 4. Reranking

**Problem.** First-stage retrievers optimize recall@100-1000 cheaply. Precision at the top-k that fits the context window needs a cross-encoder or LLM that reads query and passage jointly.

| Paper | Venue / year | Ref | System-design takeaway |
|---|---|---|---|
| Passage Re-ranking with BERT (monoBERT) | arXiv 2019 | [33] | A cross-encoder reranker over BM25 candidates gave large MS MARCO gains. This is the two-stage pattern still in production. |
| Document Ranking with a Pretrained Seq2Seq Model (monoT5) | Findings EMNLP 2020 | [34] | Seq2seq relevance scoring ("true"/"false" token) generalizes well zero-shot. It is the backbone of many open rerankers. |
| Is ChatGPT Good at Search? (RankGPT) | EMNLP 2023 | [35] | LLMs as listwise rerankers with a sliding window, distillable into small models. LLM reranking trades latency for quality. |
| Document Expansion by Query Prediction (doc2query) | arXiv 2019 | [36] | Index-time expansion with predicted queries moves cost offline. It complements query-time reranking. |
| Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods (RRF) | SIGIR 2009 | [112] | Fuse ranked lists with score 1/(k + rank), k = 60 in the paper. It needs no score calibration across BM25 and dense retrievers, which is why it is the default hybrid-fusion operator in most vector databases. |

In production, Anthropic's contextual retrieval study retrieved the top 150 candidates, reranked them to the top 20 with a Cohere reranker, and found 20 chunks beat 5 and 10 [46]. That is a concrete operating point for the reranker funnel.

## 5. Query transformation and synthetic training data

**Problem.** User queries are short, underspecified and phrased unlike documents. Two families respond. Query-time rewriting uses an LLM to reshape the query. Train-time synthesis uses an LLM to generate queries that train or adapt retrievers.

| Paper | Venue / year | Ref | System-design takeaway |
|---|---|---|---|
| HyDE: Precise Zero-Shot Dense Retrieval without Relevance Labels | ACL 2023 | [37] | Generate a hypothetical answer document and embed that instead of the query. It turns query-document asymmetry into document-document similarity. |
| Query2doc: Query Expansion with LLMs | EMNLP 2023 | [38] | Append an LLM pseudo-document to the query. It helps both BM25 and dense retrieval, at the cost of one LLM call per query. |
| Query Rewriting for Retrieval-Augmented LLMs | EMNLP 2023 | [39] | A small trainable rewriter, tuned with RL from reader feedback, sits in front of a frozen retriever and reader. |
| Take a Step Back (step-back prompting) | ICLR 2024 | [40] | Retrieve on an abstracted "higher-level" question as well as the original. This helps reasoning-heavy queries. |
| InPars: Data Augmentation for IR using LLMs | arXiv 2022 | [41] | Few-shot LLM generation of synthetic queries for in-domain reranker training. |
| Promptagator: Few-shot Dense Retrieval from 8 Examples | ICLR 2023 | [42] | Task-specific prompts plus consistency filtering produce retrievers adapted to a new domain from a handful of examples. |

**Failure mode.** Every rewrite step adds an LLM call to the critical path. Rewrites can also drift away from the user's intent, for example by inventing entities. Production systems typically run the rewrite in parallel with the raw query and fuse the results, rather than replacing the query.

## 6. Indexing granularity, chunking and hierarchy

**Problem.** The retrieval unit determines what can be found. A fixed chunk loses document context, and a whole document overwhelms the embedder. This theme is the most direct bridge from research to production practice.

```
  granularity spectrum
  proposition --- sentence --- chunk (100-800 tok) --- section --- doc --- summary tree
  (Dense X)                   (+context prefix:                   (LongRAG) (RAPTOR)
                               Anthropic contextual retrieval;
                               +late chunking: embed whole doc,
                               pool per chunk)
```

| Paper | Venue / year | Ref | System-design takeaway |
|---|---|---|---|
| Dense X Retrieval: What Retrieval Granularity Should We Use? | EMNLP 2024 | [43] | Index atomic propositions. The first version reports average Recall@20 gains of +10.1 for unsupervised and +2.2 for supervised dense retrievers over passage indexing [43]. Finer units help weaker retrievers most. |
| Late Chunking: Contextual Chunk Embeddings | arXiv 2024 | [44] | Run a long-context embedder over the full document, then mean-pool token embeddings per chunk. No retraining is needed. Gains of about 2.7-3.6% nDCG@10 on BEIR are reported (approximate, from a secondary summary) [44]. |
| RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval | ICLR 2024 | [45] | Recursively embed, cluster and summarize chunks into a tree, and retrieve across levels. With GPT-4 it raised the best QuALITY accuracy by 20 points absolute [45]. |
| Introducing Contextual Retrieval (Anthropic) | Engineering blog 2024 | [46] | Prepend LLM-written chunk context before embedding and BM25. Failed retrievals (1 - recall@20) fell 35% with contextual embeddings, 49% with contextual BM25 added (5.7% to 2.9%), and 67% with reranking (to 1.9%). The one-time cost is about $1.02 per million document tokens using prompt caching [46]. |
| LongRAG: Enhancing RAG with Long-context LLMs | arXiv 2024 | [47] | Use larger retrieval units (grouped documents) so the retriever's job is easier and the long-context reader does more of the work. |

**Lesson.** The numbers above favour context-preserving indexing over chunk-size tuning. Contextual prefixes, late pooling and summary hierarchies each attack the same root cause: a chunk embedded in isolation is ambiguous.

## 7. Iterative, adaptive and corrective retrieval

**Problem.** One retrieval per query fails on multi-hop questions, long-form generation and queries where the first retrieval is wrong. Always retrieving also wastes cost and adds distractors for queries the model can answer alone. This theme covers when to retrieve, how many times, and what to do when retrieval is bad.

```
  ReAct / IRCoT style loop                    Self-RAG / CRAG style gates
  +---------+  thought   +----------+         retrieve? --no--> generate
  |  LLM    |----------->| search   |            | yes
  | reason  |<-----------| tool     |         retrieve -> grade docs
  +---------+  evidence  +----------+            | correct    | ambiguous | incorrect
      | stop when answer is supported         refine     refine+web    web search
      v                                          \____________|___________/
   answer (+ citations)                                generate -> critique (IsSup)
```

| Paper | Venue / year | Ref | System-design takeaway |
|---|---|---|---|
| WebGPT: Browser-assisted QA with Human Feedback | arXiv 2021 | [48] | A text browser as an action space, plus quoting evidence for citations. The template for answer engines. |
| ReAct: Synergizing Reasoning and Acting | ICLR 2023 | [49] | Interleave thought, action and observation traces. It is the control loop behind most agent frameworks. |
| Self-Ask: Measuring and Narrowing the Compositionality Gap | Findings EMNLP 2023 | [50] | Explicit follow-up sub-questions, each answerable by search. A minimal decomposition pattern. |
| IRCoT: Interleaving Retrieval with Chain-of-Thought | ACL 2023 | [51] | Use each reasoning step as the next retrieval query. Large gains on multi-hop QA without training. |
| FLARE: Active Retrieval Augmented Generation | EMNLP 2023 | [52] | Retrieve when upcoming tokens are low-confidence, using the tentative sentence as the query. Retrieval is triggered by generator uncertainty. |
| Self-RAG: Learning to Retrieve, Generate, and Critique | ICLR 2024 (oral) | [53] | Reflection tokens (retrieve, relevance, support, utility) are trained into the generator. Inference-time weights and thresholds tune precision against recall without retraining. The released setup uses Contriever over DPR Wikipedia, and full-Wikipedia retrieval needs about 100 GB of RAM [53]. |
| CRAG: Corrective Retrieval Augmented Generation | arXiv 2024 | [54] | A lightweight retrieval evaluator maps results to Correct, Ambiguous or Incorrect. These trigger refinement ("decompose-then-recompose"), web-search fallback, or both. Plug-and-play on top of any RAG. |
| Adaptive-RAG: Learning to Adapt Retrieval-Augmented LLMs | NAACL 2024 | [55] | A small classifier routes each query by predicted complexity to no-retrieval, single-step or multi-step pipelines. Cost-aware routing. |
| Speculative RAG: Enhancing RAG through Drafting | ICLR 2025 | [56] | A small specialist drafts answers in parallel from disjoint document subsets, and a large generalist verifies them. Up to 12.97% higher accuracy and 50.83% lower latency on PubHealth [56]. |
| Chain-of-Retrieval Augmented Generation (CoRAG) | NeurIPS 2025 | [57] | Retrieval chains are synthesized by rejection sampling and used for training. More than 10 EM points on multi-hop QA, and state of the art on KILT, with E5-large over 36M passages [57]. Test-time compute can be spent on retrieval steps. |

**Failure modes.** Iterative loops multiply latency and cost, and they can loop on unproductive queries. Evaluators (CRAG's grader, Self-RAG's critique tokens) are new components that can themselves be miscalibrated. Adaptive-RAG and Speculative RAG make this explicit: the design target is accuracy per unit of latency, not accuracy alone.

## 8. RL-trained search agents and deep research (2025-2026)

**Problem.** Prompted agents (ReAct, IRCoT) do not learn how to search well. Starting in early 2025, a cluster of papers applied outcome-reward RL (PPO, GRPO), in the style of DeepSeek-R1, to train LLMs that interleave reasoning with search calls.

```
  rollout:  <think> ... </think> <search> q1 </search> <information> d1..dk </information>
            <think> ... </think> <search> q2 </search> <information> ...  </information>
            <answer> a </answer>
  reward:   rule-based outcome match(a, gold)       (no process reward)
  trick:    mask retrieved tokens from the policy-gradient loss (not model-generated)
  infra:    retriever served as an HTTP tool (e5 + Wikipedia in Search-R1's quick start)
```

| Paper | Venue / year | Ref | System-design takeaway |
|---|---|---|---|
| Search-o1: Agentic Search-Enhanced Large Reasoning Models | EMNLP 2025 | [58] | Agentic RAG inside a reasoning model's chain of thought. A "Reason-in-Documents" module condenses retrieved pages before they enter the reasoning trace. |
| Search-R1: Training LLMs to Reason and Leverage Search Engines with RL | COLM 2025 | [59] | Multi-turn search inside RL rollouts with retrieved-token masking and an outcome reward. The abstract reports double-digit relative improvements over RAG baselines on seven QA datasets (about 20% for Qwen2.5-3B; the 7B figure differs between arXiv versions, so check the version you cite). The repository supports PPO, GRPO and REINFORCE and uses an e5 retriever over Wikipedia in its quick start [59]. |
| R1-Searcher | arXiv 2025 | [60] | Two-stage outcome-based RL that first teaches the format for invoking search, then answer quality. |
| ReSearch: Learning to Reason with Search via RL | NeurIPS 2025 | [61] | Search operations are treated as part of the reasoning chain and trained without supervised reasoning steps. |
| DeepResearcher: Scaling Deep Research via RL in Real-world Environments | arXiv 2025 | [62] | RL against the live web rather than a static Wikipedia index. It exposes noise, rate limits and page-fetch failures that a local corpus hides. |
| ZeroSearch: Incentivize Search Capability without Searching | arXiv 2025 | [63] | An LLM simulates the search engine during RL to cut API cost and control document quality via a curriculum. |
| GRASP: Granularity-Aware Search Policy for Agentic RAG | arXiv 2026 | [64] | Gives the agent semantic search, keyword search and paragraph-reading actions, so it fetches sentence-level evidence and expands context only when needed. Granularity becomes a learned action. |
| AutoSearch: Adaptive Search Depth for Efficient Agentic RAG via RL | arXiv 2026 | [111] | Trains the agent to stop searching once evidence suffices, targeting redundant search calls. Search depth becomes a cost-aware learned policy. |
| SoK: Agentic RAG (taxonomy, architectures, evaluation) | arXiv 2026 | [65] | Formalizes agentic retrieval loops as finite-horizon POMDPs. Argues for trajectory-level evaluation and flags memory poisoning and compounding hallucination as systemic risks. |

**Lessons.** The RL line confirms that the retriever, not just the policy, bounds performance; Agentic-R [26] trains the retriever for agents directly. Simulated or local search environments make training cheap and reproducible but hide real-web failure modes [62][63]. AutoSearch [111] targets the obvious cost problem of redundant search calls by learning adaptive search depth.

## 9. Graph and structured RAG

**Problem.** Vector RAG retrieves locally similar text. It fails on global "sensemaking" questions (what are the main themes across this corpus?) and on multi-hop associations whose entities never co-occur in one chunk. Graph RAG builds an entity/relation structure offline and retrieves through it.

```
  GraphRAG (Microsoft)                            HippoRAG
  chunks -> LLM entity/relation extraction        chunks -> OpenIE triples -> KG
        -> KG -> hierarchical Leiden communities   query -> entities -> seed nodes
        -> LLM community summaries (C0..C3)        Personalized PageRank over KG
  global query: map over summaries -> reduce       -> passage scores -> top-k passages
  cost: LLM pass over the whole corpus at index    cost: OpenIE pass, cheap queries
```

| Paper | Venue / year | Ref | System-design takeaway |
|---|---|---|---|
| From Local to Global: A Graph RAG Approach to Query-Focused Summarization | arXiv 2024 | [67] | LLM-extracted KG, hierarchical Leiden communities and pre-generated community summaries. Map-reduce answering for global questions over corpora of about 1M tokens, judged by LLM on comprehensiveness and diversity [67][68]. |
| GraphRAG: Unlocking LLM discovery on narrative private data (MSR blog) | Blog 2024 | [68] | Answers link back to source text for provenance. Faithfulness was similar to baseline RAG, and the gains are in comprehensiveness. The repository warns that indexing "can be an expensive operation" and recommends prompt tuning [69]. |
| LazyGraphRAG: Setting a new standard for quality and cost | MSR blog 2024 | [70] | NLP noun-phrase extraction replaces LLM summarization at index time, so indexing costs the same as vector RAG and 0.1% of full GraphRAG. Global-query quality comparable to GraphRAG costs more than 700x less at query time. A single "relevance test budget" (100-1,500) sets the trade-off [70]. |
| LightRAG: Simple and Fast RAG | Findings EMNLP 2025 | [71] | Dual-level (entity-level and theme-level) retrieval over graph + vectors, with incremental updates that avoid a full rebuild. |
| HippoRAG: Neurobiologically Inspired Long-Term Memory | NeurIPS 2024 | [72] | OpenIE knowledge graph plus Personalized PageRank gives single-step multi-hop retrieval, cheaper than iterative LLM loops. |
| HippoRAG 2: From RAG to Memory | ICML 2025 | [73] | Deeper passage integration and online LLM filtering fix the factual-recall regression of earlier structure-augmented RAG. The authors claim significantly fewer offline indexing resources than other graph-based approaches [73][74]. |
| Think-on-Graph: Deep and Responsible Reasoning on KGs | ICLR 2024 | [75] | The LLM agent beam-searches over an existing KG (Freebase/Wikidata). It suits curated graphs, not extracted ones. |
| G-Retriever: Retrieval-Augmented Generation for Textual Graph QA | NeurIPS 2024 | [76] | Retrieve a subgraph (prize-collecting Steiner tree) and feed it via a GNN plus a text prompt. Subgraph retrieval controls context size. |

**Lesson.** The graph line's own trajectory, from GraphRAG to LazyGraphRAG to HippoRAG 2, is a cost correction. LLM-heavy indexing produces good global answers but is expensive and hard to keep fresh. Later work defers LLM calls to query time, or uses cheap extraction plus graph algorithms.

## 10. Generator training and context utilization

**Problem.** Retrieval quality is not answer quality. LLMs ignore relevant context in some positions, are misled by distractors, and answer confidently when the retrieved context is insufficient. This theme studies and trains the reader side.

| Paper | Venue / year | Ref | System-design takeaway |
|---|---|---|---|
| Lost in the Middle: How LMs Use Long Contexts | TACL 2024 | [77] | Accuracy is U-shaped in the position of the relevant passage, best at the start or end. Moving it to the middle cost GPT-3.5 roughly 20 points (approximate, secondary). Order reranked chunks with the best at the edges. |
| The Power of Noise: Redefining Retrieval for RAG Systems | SIGIR 2024 | [78] | Related-but-irrelevant distractors hurt more than random documents. Reranking precision matters more than raw k. |
| Retrieval meets Long Context LLMs | ICLR 2024 | [79] | Retrieval helps even long-context models, and a 4K-context model with simple retrieval can match a 16K-context model extended by positional interpolation and fine-tuning on long-context tasks (per the authors). RAG and long context are complements. |
| Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG | ICLR 2025 | [116] | With long-context readers, quality first rises then falls as more passages are retrieved, because hard negatives from strong retrievers mislead the model. Retrieval reordering and RAG-specific fine-tuning recover the loss. Do not assume a bigger context window lets you raise k for free. |
| RAG or Long-Context LLMs? (Self-Route) | EMNLP 2024 Industry | [80] | Long context beat RAG on average when resourced sufficiently, but RAG was much cheaper. Self-Route lets the model decide per query, keeping LC-level quality at much lower cost [80]. |
| RAFT: Adapting LMs to Domain-Specific RAG | COLM 2024 | [81] | Fine-tune with oracle plus distractor documents and chain-of-thought citing the oracle. Teaches the generator to ignore distractors in-domain. |
| RankRAG: Unifying Context Ranking with RAG in LLMs | NeurIPS 2024 | [82] | One instruction-tuned LLM both reranks and generates, so a separate reranker service can be removed. |
| Sufficient Context: A New Lens on RAG Systems | ICLR 2025 | [83] | Separates "context lacks the answer" from "model failed to use it". Strong models guess rather than abstain on insufficient context. Sufficiency-guided selective generation raised correct answers among responses by 2-10% [83]. |
| ALCE: Enabling LLMs to Generate Text with Citations | EMNLP 2023 | [84] | Benchmarks and automatic metrics for citation recall and precision. Citation quality must be measured separately from fluency. |

## 11. Compression, caching and serving efficiency

**Problem.** RAG prompts are long, and the retrieved documents repeat across queries, yet the standard prefix cache rarely hits because document order and combination vary. Two responses exist. Compress the context, or reuse the KV states of document chunks.

```
  standard prefix cache:  [sys][docA][docB][q]  hit only if identical prefix
  RAGCache:               knowledge tree of doc KV states across GPU/host memory,
                          prefix-aware eviction, overlap retrieval with prefill
  CacheBlend:             reuse non-prefix chunk KV caches, recompute a small subset
                          of tokens to restore cross-attention; overlap with KV load
```

| Paper | Venue / year | Ref | System-design takeaway |
|---|---|---|---|
| RECOMP: Improving RAG with Compression and Selective Augmentation | ICLR 2024 | [85] | Extractive and abstractive compressors that can output an empty summary, turning compression into selective augmentation. |
| LongLLMLingua: Prompt Compression in Long Context Scenarios | ACL 2024 | [86] | Question-aware token dropping plus document reordering lowers cost and mitigates lost-in-the-middle. |
| xRAG: Extreme Context Compression for RAG with One Token | NeurIPS 2024 | [87] | Project the retriever embedding into a single soft token, reusing the index embedding as the context. |
| RAGCache: Efficient Knowledge Caching for RAG | arXiv 2024 | [88] | Multilevel KV cache of retrieved documents with retrieval-aware replacement. Up to 4x lower TTFT and 2.1x throughput against vLLM + Faiss [88]. |
| CacheBlend: Fast LLM Serving for RAG with Cached Knowledge Fusion | EuroSys 2025 (best paper) | [89] | Selective recomputation on reused non-prefix KV caches gives 2.2-3.3x lower TTFT and 2.8-5x throughput against full recompute without quality loss. It ships in LMCache and the vLLM production stack [89]. |
| TurboRAG: Precomputed KV Caches for Chunked Text | arXiv 2024 | [90] | Precompute per-chunk KV offline, with position and attention-mask adjustments. Prefill cost moves to indexing time. |
| PipeRAG: Algorithm-System Co-design for Iterative Retrieval | arXiv 2024 | [91] | Pipeline retrieval with generation (stale-query prefetch, flexible intervals) to hide retrieval latency in retrieval-during-decoding designs. |

## 12. Multimodal (visual document) retrieval

**Problem.** PDFs, slides and scanned forms lose tables, figures and layout when converted to text. The OCR, layout and chunking pipeline is slow and error-prone.

| Paper | Venue / year | Ref | System-design takeaway |
|---|---|---|---|
| ColPali: Efficient Document Retrieval with Vision Language Models | ICLR 2025 | [92] | Embed page images directly with a VLM into multi-vector, ColBERT-style patch embeddings. It outperforms parse-then-embed pipelines while being simpler. It also introduces the ViDoRe benchmark [92]. |
| VisRAG: Vision-based RAG on Multi-modality Documents | ICLR 2025 | [93] | Image-based retrieval and a VLM generator end-to-end, with no text parsing at any stage. |
| M3DocRAG: Multi-modal Retrieval for Multi-page Multi-document Understanding | arXiv 2024 | [94] | ColPali retrieval plus a multimodal LLM over many-page, many-document collections, with an open-domain benchmark. |

**Trade-off.** Multi-vector page embeddings move cost from the parsing pipeline into index storage (one vector per patch). The ColBERTv2 and PLAID compression and pruning techniques [14][15] therefore apply directly to visual RAG.

## 13. Evaluation and benchmarks

**Problem.** Retrieval metrics (nDCG, recall@k) and end-to-end metrics (answer correctness, faithfulness, citation quality) diverge. Benchmarks also saturate or leak into training data. The papers below are the tools most teams use, grouped by what they measure.

| Paper | Venue / year | Ref | Measures | System-design takeaway |
|---|---|---|---|---|
| MS MARCO | arXiv 2016 | [95] | Passage ranking | The training set behind most retrievers and rerankers. In-domain results there overstate zero-shot quality. |
| BEIR: Heterogeneous Zero-shot IR Benchmark | NeurIPS 2021 D&B | [96] | Zero-shot retrieval | Many datasets across domains. Dense models trained on MS MARCO often failed to beat BM25 out of domain. Always test on your own domain. |
| MTEB: Massive Text Embedding Benchmark | EACL 2023 | [97] | Embedders | Leaderboard rank is averaged across tasks. Select embedders on the retrieval subset and your own data. |
| BRIGHT: Reasoning-Intensive Retrieval | ICLR 2025 | [98] | Reasoning retrieval | The MTEB leader at 59.0 nDCG@10 scored only about 18 nDCG@10 on BRIGHT. Adding explicit LLM reasoning to queries gave up to 12.2 points [98]. |
| RGB: Benchmarking LLMs in RAG | AAAI 2024 | [99] | Generator robustness | Tests noise robustness, negative rejection, information integration and counterfactual robustness. |
| RAGAS: Automated Evaluation of RAG | EACL 2024 demo | [100] | Reference-free RAG | LLM-judged faithfulness, answer relevance and context precision/recall. A cheap regression signal, not ground truth. |
| ARES: Automated RAG Evaluation System | NAACL 2024 | [101] | Judges + PPI | Fine-tuned judges plus prediction-powered inference from a small human-labeled set give confidence intervals. |
| MultiHop-RAG | COLM 2024 | [102] | Multi-hop RAG | A news-based multi-hop query set. Standard RAG pipelines struggle on cross-document evidence. |
| FRAMES (Fact, Fetch, and Reason) | NAACL 2025 (arXiv 2024) | [103] | End-to-end RAG | Joint test of factuality, retrieval and multi-step reasoning on Wikipedia-based questions. |
| Ragnarok: Reusable RAG Framework for TREC 2024 RAG Track | ECIR 2025 | [104] | Shared-task eval | MS MARCO V2.1 segmented corpus, non-factoid topics and open baselines. A standardized arena for comparing RAG systems. |
| BrowseComp: A Simple Yet Challenging Benchmark for Browsing Agents | arXiv 2025 | [117] | Deep-research agents | 1,266 hard-to-find but easy-to-verify questions that require persistent web browsing. Short verifiable answers make agentic search measurable without LLM judges of long reports. |

**Lesson.** Run retrieval evaluation (BEIR- or BRIGHT-style on your own queries) and generation evaluation (RAGAS/ARES-style plus citation metrics from ALCE [84]) as separate gates. The sufficient-context framing [83] tells you which side to fix.

## 14. Surveys and engineering experience reports

| Paper | Venue / year | Ref | System-design takeaway |
|---|---|---|---|
| Retrieval-Augmented Generation for LLMs: A Survey | arXiv 2023 | [105] | The Naive, Advanced and Modular RAG taxonomy that most vendor docs reuse. |
| Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG | arXiv 2025 | [106] | A taxonomy of single-agent, multi-agent, hierarchical and graph-based agentic RAG. |
| Seven Failure Points When Engineering a RAG System | CAIN 2024 | [107] | Experience report from three deployments. Failures include missing content, missed top-ranked documents, documents not in context, extraction failures, wrong format, wrong specificity and incomplete answers. Robustness evolves rather than being designed in. |
| STORM: Writing Wikipedia-like Articles from Scratch | NAACL 2024 | [108] | Perspective-guided question asking plus retrieval-driven outlining. An early "deep research" pipeline for long-form, grounded writing. |
| PaperQA: Retrieval-Augmented Generative Agent for Scientific Research | arXiv 2023 | [109] | An agentic loop of search, gather evidence and answer over full-text papers. A domain template for scientific RAG. |
| MemoRAG: Memory-Inspired Knowledge Discovery for Long-Context RAG | arXiv 2024 | [110] | A light long-context "memory" model drafts clue answers that guide retrieval for ambiguous or global queries. |
| RL-based Agentic Search paper list (GitHub) | Living list 2025-2026 | [66] | A maintained index of Search-R1-style papers and 2026 surveys. Useful for tracking a fast-moving line. |

## 15. Security, privacy and adversarial robustness

**Problem.** A RAG system executes text it did not write. Anything an attacker can get into the index, such as a web page, a shared drive file, an email or a support ticket, becomes part of the prompt. The retrieval layer is therefore both an attack surface (injection, poisoning) and a leakage channel (the generator can be induced to reproduce private chunks). Agentic RAG makes this worse, because retrieved text can now trigger tool calls.

| Paper | Venue / year | Ref | System-design takeaway |
|---|---|---|---|
| Not What You've Signed Up For: Indirect Prompt Injection in LLM-Integrated Applications | AISec (CCS workshop) 2023 | [118] | Instructions hidden in retrieved content can hijack retrieval-augmented and tool-using LLMs. Treat retrieved text as untrusted data: separate it from instructions and gate tool calls that it could trigger. |
| PoisonedRAG: Knowledge Corruption Attacks to RAG | USENIX Security 2025 | [119] | Injecting a handful of crafted passages per target question into a large knowledge base steers answers with high success rates. Ingestion provenance and write access control matter as much as retrieval quality. |
| The Good and the Bad: Exploring Privacy Issues in RAG | Findings ACL 2024 | [120] | Targeted prompts can make a RAG system leak verbatim chunks of its private datastore, while retrieval can also reduce leakage of the model's own training data. Enforce document-level ACLs at retrieval time, before the generator sees anything. |
| RobustRAG: Certifiably Robust RAG against Retrieval Corruption | arXiv 2024 | [121] | Isolate-then-aggregate: answer from each passage separately and aggregate securely (keyword or decoding aggregation), so a bounded number of corrupted passages cannot flip the answer. Robustness costs extra generator calls. |

**Lesson.** Design defenses at the layers this chapter already covers. Use provenance and access control at ingestion and retrieval, isolation of passages at generation time (as in RobustRAG and, for efficiency reasons, FiD [6] and Speculative RAG [56]), and evaluation that includes adversarial documents, extending RGB-style robustness tests [99].

## Comparison: which paper family solves which production problem

| Production symptom | Root cause | Papers to read first | Typical fix and its cost |
|---|---|---|---|
| Right doc exists but is not in top-k | Embedding capacity, domain shift | [4][13][14][17][22][46] | Hybrid BM25 + dense + reranker. One more index and 50-200 ms of reranking |
| Chunk retrieved but ambiguous | Chunk lacks document context | [43][44][45][46] | Contextual prefix or late chunking. A one-time LLM or long-context embed pass |
| Multi-hop question fails | Evidence spread across documents | [51][57][59][72][73] | Iterative retrieval or graph PPR. 2-10x more retrieval calls, or an OpenIE index |
| "What are the themes?" fails | Global question, local retriever | [67][70][45] | Community summaries or lazy graph. Index cost from 0.1% (Lazy) to 100% (GraphRAG) |
| Answer ignores retrieved fact | Position bias, distractors | [77][78][81][82] | Reorder, tighten k, RAFT-style fine-tuning |
| Confident wrong answer | Insufficient context, no abstention | [53][54][83] | Sufficiency or relevance grader plus abstain path. One extra classifier call |
| High TTFT, GPU cost | Long prompts, no prefix reuse | [85][86][88][89] | Compression or chunk KV reuse. Cache memory tiers |
| Scanned PDFs, tables | Lossy parsing | [92][93][94] | Page-image multi-vector retrieval. Larger index |
| Index exceeds RAM | Billion-scale vectors | [27][31][32] | SSD-resident graph or partitioned index. Some tail latency |
| Reasoning-heavy queries | Semantic similarity is not relevance | [24][25][40][98] | Query reasoning expansion, reasoning-trained retrievers |
| Poisoned or injected content changes answers | Untrusted text in the index | [118][119][121] | Provenance, ingestion access control, passage isolation. Extra generator calls for certified aggregation |
| Filtered queries lose recall | Metadata filter applied after ANN search | [113][112] | Filter-aware index or pre-filtered partitions. More index variants |

## A suggested reading order (two weeks)

```
  week 1  (core)                          week 2  (frontier)
  d1  DrQA [1], DPR [4], RAG [5]          d6  Self-RAG [53], CRAG [54], Adaptive-RAG [55]
  d2  FiD [6], RETRO [8], In-Ctx RALM[10] d7  Search-o1 [58], Search-R1 [59], CoRAG [57]
  d3  ColBERTv2 [14], SPLADE [17],        d8  GraphRAG [67], LazyGraphRAG [70],
      HNSW [27], DiskANN [31]                 HippoRAG 2 [73]
  d4  monoT5 [34], RankGPT [35],          d9  RAGCache [88], CacheBlend [89],
      HyDE [37], contextual retr. [46]        Speculative RAG [56], ColPali [92]
  d5  Lost in Middle [77], Power of       d10 BEIR [96], BRIGHT [98], RAGAS [100],
      Noise [78], Sufficient Ctx [83]         ARES [101], Seven Failure Points [107]
```

## Key takeaways for system designers

1. **Retrieval recall is the ceiling.** From DrQA [1] through Search-R1 [59] and Agentic-R [26], the reader cannot recover evidence the retriever missed. Budget most effort on the retriever, index and reranker, and measure recall@k separately from answer quality.
2. **Hybrid beats pure dense in practice.** DPR's margin over BM25 shrank under replication [4], BEIR showed dense models' out-of-domain fragility [96], and Anthropic's ablation got its biggest gains by adding BM25 and a reranker to contextual embeddings [46].
3. **Context-preserving indexing is the cheapest big win.** Contextual prefixes [46], late chunking [44], propositions [43] and summary trees [45] all fix chunk ambiguity at index time, where the cost is paid once.
4. **ANN recall is a quality knob.** Billion-scale indexes can plateau near 50% recall at a given memory budget [31]. Check ANN recall against exact search on a sample before tuning prompts.
5. **Retrieve adaptively, not always.** Self-RAG [53], FLARE [52], Adaptive-RAG [55] and Self-Route [80] show that deciding whether, how often and where to retrieve improves both cost and quality.
6. **Design an abstain path.** Strong models guess rather than abstain when the context is insufficient [83]. Graders (CRAG [54]) and critique tokens (Self-RAG [53]) are the architectural fix.
7. **Fewer, better chunks beat many.** Distractors hurt [78], position matters [77], and Anthropic's operating point was 150 candidates reranked to 20 [46]. Tune k with an evaluation set, not intuition.
8. **Graph RAG is for global and associative questions, and indexing cost decides adoption.** Use it where vector RAG demonstrably fails. Prefer lazy or cheap-extraction variants [70][73] unless corpus-wide summaries are the product.
9. **Agentic search is now trained, not just prompted.** RL with outcome rewards and retrieved-token masking [59] is the 2025-2026 default for research agents. The retriever [26], the simulated or real search environment [62][63], evidence granularity [64] and search-depth budgets [111] are the new design surfaces.
10. **Serve RAG as a caching problem.** Chunk-level KV reuse cut TTFT by roughly 2-4x in the reported settings [88][89], and speculative drafting cut end-to-end latency by about half on PubHealth [56]. These gains are larger than most model-side optimizations.
11. **Evaluate in layers.** Use retrieval benchmarks (BEIR, MTEB, BRIGHT), generator robustness (RGB), reference-free judges (RAGAS, ARES) and citation metrics (ALCE). Diagnose with the sufficient-context split [83] to know whether to fix retrieval or generation.

## References

1. Chen, D., Fisch, A., Weston, J., Bordes, A. "Reading Wikipedia to Answer Open-Domain Questions." ACL 2017. https://arxiv.org/abs/1704.00051
2. Lee, K., Chang, M.-W., Toutanova, K. "Latent Retrieval for Weakly Supervised Open Domain Question Answering." ACL 2019. https://arxiv.org/abs/1906.00300
3. Guu, K., Lee, K., Tung, Z., Pasupat, P., Chang, M.-W. "REALM: Retrieval-Augmented Language Model Pre-Training." ICML 2020. https://arxiv.org/abs/2002.08909
4. Karpukhin, V., et al. "Dense Passage Retrieval for Open-Domain Question Answering." Meta AI, EMNLP 2020. https://aclanthology.org/2020.emnlp-main.550 (replication: Ma, X., et al. "A Replication Study of Dense Passage Retriever," 2021. https://arxiv.org/abs/2104.05740)
5. Lewis, P., et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." Meta AI / UCL, NeurIPS 2020. https://arxiv.org/abs/2005.11401
6. Izacard, G., Grave, E. "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering." EACL 2021. https://arxiv.org/abs/2007.01282
7. Khandelwal, U., et al. "Generalization through Memorization: Nearest Neighbor Language Models." ICLR 2020. https://arxiv.org/abs/1911.00172
8. Borgeaud, S., et al. "Improving Language Models by Retrieving from Trillions of Tokens." DeepMind, ICML 2022. https://proceedings.mlr.press/v162/borgeaud22a.html
9. Izacard, G., et al. "Atlas: Few-shot Learning with Retrieval Augmented Language Models." Meta AI, JMLR 2023. https://www.jmlr.org/beta/papers/v24/23-0037.html
10. Ram, O., et al. "In-Context Retrieval-Augmented Language Models." AI21 Labs, TACL 2023. https://arxiv.org/abs/2302.00083
11. Shi, W., et al. "REPLUG: Retrieval-Augmented Black-Box Language Models." NAACL 2024. https://arxiv.org/abs/2301.12652
12. Petroni, F., et al. "KILT: a Benchmark for Knowledge Intensive Language Tasks." NAACL 2021. https://arxiv.org/abs/2009.02252
13. Khattab, O., Zaharia, M. "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT." SIGIR 2020. https://arxiv.org/abs/2004.12832 (code: https://github.com/stanford-futuredata/ColBERT)
14. Santhanam, K., et al. "ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction." NAACL 2022. https://arxiv.org/abs/2112.01488
15. Santhanam, K., et al. "PLAID: An Efficient Engine for Late Interaction Retrieval." CIKM 2022. https://arxiv.org/abs/2205.09707
16. Xiong, L., et al. "Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval." ICLR 2021. https://arxiv.org/abs/2007.00808
17. Formal, T., Piwowarski, B., Clinchant, S. "SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking." SIGIR 2021. https://arxiv.org/abs/2107.05720
18. Formal, T., et al. "SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval." 2021. https://arxiv.org/abs/2109.10086
19. Izacard, G., et al. "Unsupervised Dense Information Retrieval with Contrastive Learning." TMLR 2022. https://arxiv.org/abs/2112.09118
20. Wang, L., et al. "Text Embeddings by Weakly-Supervised Contrastive Pre-training (E5)." Microsoft, 2022. https://arxiv.org/abs/2212.03533
21. Li, Z., et al. "Towards General Text Embeddings with Multi-stage Contrastive Learning (GTE)." Alibaba, 2023. https://arxiv.org/abs/2308.03281
22. Chen, J., et al. "BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings." BAAI, 2024. https://arxiv.org/abs/2402.03216
23. Kusupati, A., et al. "Matryoshka Representation Learning." NeurIPS 2022. https://arxiv.org/abs/2205.13147
24. Weller, O., et al. "Promptriever: Instruction-Trained Retrievers Can Be Prompted Like Language Models." 2024. https://arxiv.org/abs/2409.11136
25. Shao, R., et al. "ReasonIR: Training Retrievers for Reasoning Tasks." 2025. https://arxiv.org/abs/2504.20595
26. "Agentic-R: Learning to Retrieve for Agentic Search." 2026. https://arxiv.org/abs/2601.11888 (code: https://github.com/8421BCD/Agentic-R)
27. Malkov, Y., Yashunin, D. "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs." IEEE TPAMI. https://arxiv.org/abs/1603.09320
28. Johnson, J., Douze, M., Jegou, H. "Billion-scale Similarity Search with GPUs." Meta AI, 2017. https://arxiv.org/abs/1702.08734
29. Douze, M., et al. "The Faiss Library." Meta AI, 2024. https://arxiv.org/abs/2401.08281
30. Guo, R., et al. "Accelerating Large-Scale Inference with Anisotropic Vector Quantization (ScaNN)." Google, ICML 2020. https://arxiv.org/abs/1908.10396
31. Jayaram Subramanya, S., et al. "DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node." Microsoft Research, NeurIPS 2019. https://www.microsoft.com/en-us/research/publication/diskann/ (proceedings: https://papers.nips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html)
32. Chen, Q., et al. "SPANN: Highly-efficient Billion-scale Approximate Nearest Neighbor Search." Microsoft, NeurIPS 2021. https://arxiv.org/abs/2111.08566
33. Nogueira, R., Cho, K. "Passage Re-ranking with BERT." 2019. https://arxiv.org/abs/1901.04085
34. Nogueira, R., et al. "Document Ranking with a Pretrained Sequence-to-Sequence Model." Findings of EMNLP 2020. https://arxiv.org/abs/2003.06713
35. Sun, W., et al. "Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents." EMNLP 2023. https://arxiv.org/abs/2304.09542
36. Nogueira, R., et al. "Document Expansion by Query Prediction." 2019. https://arxiv.org/abs/1904.08375
37. Gao, L., Ma, X., Lin, J., Callan, J. "Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)." ACL 2023. https://arxiv.org/abs/2212.10496
38. Wang, L., Yang, N., Wei, F. "Query2doc: Query Expansion with Large Language Models." EMNLP 2023. https://arxiv.org/abs/2303.07678
39. Ma, X., et al. "Query Rewriting for Retrieval-Augmented Large Language Models." EMNLP 2023. https://arxiv.org/abs/2305.14283
40. Zheng, H. S., et al. "Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models." Google DeepMind, ICLR 2024. https://arxiv.org/abs/2310.06117
41. Bonifacio, L., et al. "InPars: Data Augmentation for Information Retrieval using Large Language Models." 2022. https://arxiv.org/abs/2202.05144
42. Dai, Z., et al. "Promptagator: Few-shot Dense Retrieval From 8 Examples." Google, ICLR 2023. https://arxiv.org/abs/2209.11755
43. Chen, T., et al. "Dense X Retrieval: What Retrieval Granularity Should We Use?" EMNLP 2024. https://arxiv.org/abs/2312.06648
44. Gunther, M., Mohr, I., Williams, D. J., Wang, B., Xiao, H. "Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models." Jina AI / Weaviate, 2024. https://arxiv.org/abs/2409.04701
45. Sarthi, P., et al. "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval." Stanford, ICLR 2024. https://arxiv.org/abs/2401.18059
46. Anthropic. "Introducing Contextual Retrieval." September 19, 2024. https://www.anthropic.com/news/contextual-retrieval
47. Jiang, Z., Ma, X., Chen, W. "LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs." 2024. https://arxiv.org/abs/2406.15319
48. Nakano, R., et al. "WebGPT: Browser-assisted Question-answering with Human Feedback." OpenAI, 2021. https://arxiv.org/abs/2112.09332
49. Yao, S., et al. "ReAct: Synergizing Reasoning and Acting in Language Models." ICLR 2023. https://arxiv.org/abs/2210.03629
50. Press, O., et al. "Measuring and Narrowing the Compositionality Gap in Language Models (Self-Ask)." Findings of EMNLP 2023. https://arxiv.org/abs/2210.03350
51. Trivedi, H., et al. "Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions (IRCoT)." ACL 2023. https://arxiv.org/abs/2212.10509
52. Jiang, Z., et al. "Active Retrieval Augmented Generation (FLARE)." EMNLP 2023. https://arxiv.org/abs/2305.06983
53. Asai, A., Wu, Z., Wang, Y., Sil, A., Hajishirzi, H. "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection." ICLR 2024 (oral). https://arxiv.org/abs/2310.11511 (code: https://github.com/AkariAsai/self-rag)
54. Yan, S.-Q., Gu, J.-C., Zhu, Y., Ling, Z.-H. "Corrective Retrieval Augmented Generation." 2024. https://arxiv.org/abs/2401.15884
55. Jeong, S., et al. "Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity." NAACL 2024. https://arxiv.org/abs/2403.14403
56. Wang, Z., et al. "Speculative RAG: Enhancing Retrieval Augmented Generation through Drafting." Google, ICLR 2025. https://arxiv.org/abs/2407.08223
57. Wang, L., Chen, H., Yang, N., Huang, X., Dou, Z., Wei, F. "Chain-of-Retrieval Augmented Generation." Microsoft / Renmin University, NeurIPS 2025. https://arxiv.org/abs/2501.14342
58. Li, X., Dong, G., et al. "Search-o1: Agentic Search-Enhanced Large Reasoning Models." Renmin University / Tsinghua, EMNLP 2025. https://arxiv.org/abs/2501.05366
59. Jin, B., et al. "Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning." UIUC et al., COLM 2025. https://arxiv.org/abs/2503.09516 (code: https://github.com/PeterGriffinJin/Search-R1)
60. Song, H., et al. "R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning." 2025. https://arxiv.org/abs/2503.05592
61. Chen, M., et al. "ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning." NeurIPS 2025. https://arxiv.org/abs/2503.19470
62. Zheng, Y., et al. "DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments." 2025. https://arxiv.org/abs/2504.03160
63. Sun, H., et al. "ZeroSearch: Incentivize the Search Capability of LLMs without Searching." Alibaba, 2025. https://arxiv.org/abs/2505.04588
64. "GRASP: GRanularity-Aware Search Policy for Agentic RAG." 2026. https://arxiv.org/abs/2607.10463
65. Mishra, S., Niroula, S., Yadav, U., Thakur, D., Gyawali, S., Gaire, S. "SoK: Agentic Retrieval-Augmented Generation (RAG): Taxonomy, Architectures, Evaluation, and Research Directions." 2026. https://arxiv.org/abs/2603.07379
66. ventr1c. "Awesome RL-based Agentic Search Papers." GitHub, 2025-2026. https://github.com/ventr1c/Awesome-RL-based-Agentic-Search-Papers
67. Edge, D., et al. "From Local to Global: A Graph RAG Approach to Query-Focused Summarization." Microsoft Research, 2024. https://arxiv.org/abs/2404.16130
68. Microsoft Research. "GraphRAG: Unlocking LLM discovery on narrative private data." 2024. https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/
69. Microsoft. "graphrag" repository README. GitHub. https://github.com/microsoft/graphrag
70. Microsoft Research. "LazyGraphRAG: Setting a new standard for quality and cost." November 25, 2024. https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/
71. Guo, Z., Xia, L., Yu, Y., Ao, T., Huang, C. "LightRAG: Simple and Fast Retrieval-Augmented Generation." HKU, Findings of EMNLP 2025. https://arxiv.org/abs/2410.05779
72. Gutierrez, B. J., et al. "HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models." OSU, NeurIPS 2024. https://arxiv.org/abs/2405.14831
73. Gutierrez, B. J., Shu, Y., Qi, W., Zhou, S., Su, Y. "From RAG to Memory: Non-Parametric Continual Learning for Large Language Models (HippoRAG 2)." ICML 2025. https://arxiv.org/abs/2502.14802
74. OSU NLP Group. "HippoRAG" repository README. GitHub. https://github.com/OSU-NLP-Group/HippoRAG
75. Sun, J., et al. "Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph." ICLR 2024. https://arxiv.org/abs/2307.07697
76. He, X., et al. "G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering." NeurIPS 2024. https://arxiv.org/abs/2402.07630
77. Liu, N. F., et al. "Lost in the Middle: How Language Models Use Long Contexts." Stanford, TACL 2024. https://arxiv.org/abs/2307.03172
78. Cuconasu, F., et al. "The Power of Noise: Redefining Retrieval for RAG Systems." SIGIR 2024. https://arxiv.org/abs/2401.14887
79. Xu, P., et al. "Retrieval meets Long Context Large Language Models." NVIDIA, ICLR 2024. https://arxiv.org/abs/2310.03025
80. Li, Z., et al. "Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study and Hybrid Approach." Google DeepMind / U. Michigan, EMNLP 2024 Industry. https://arxiv.org/abs/2407.16833
81. Zhang, T., et al. "RAFT: Adapting Language Model to Domain Specific RAG." UC Berkeley, COLM 2024. https://arxiv.org/abs/2403.10131
82. Yu, Y., et al. "RankRAG: Unifying Context Ranking with Retrieval-Augmented Generation in LLMs." NVIDIA, NeurIPS 2024. https://arxiv.org/abs/2407.02485
83. Joren, H., Zhang, J., Ferng, C.-S., Juan, D.-C., Taly, A., Rashtchian, C. "Sufficient Context: A New Lens on Retrieval Augmented Generation Systems." Google, ICLR 2025. https://arxiv.org/abs/2411.06037
84. Gao, T., Yen, H., Yu, J., Chen, D. "Enabling Large Language Models to Generate Text with Citations (ALCE)." Princeton, EMNLP 2023. https://arxiv.org/abs/2305.14627
85. Xu, F., Shi, W., Choi, E. "RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation." ICLR 2024. https://arxiv.org/abs/2310.04408
86. Jiang, H., et al. "LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression." Microsoft, ACL 2024. https://arxiv.org/abs/2310.06839
87. Cheng, X., et al. "xRAG: Extreme Context Compression for Retrieval-augmented Generation with One Token." NeurIPS 2024. https://arxiv.org/abs/2405.13792
88. Jin, C., et al. "RAGCache: Efficient Knowledge Caching for Retrieval-Augmented Generation." Peking University / ByteDance, 2024. https://arxiv.org/abs/2404.12457
89. Yao, J., et al. "CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion." EuroSys 2025. https://arxiv.org/abs/2405.16444
90. Lu, S., et al. "TurboRAG: Accelerating Retrieval-Augmented Generation with Precomputed KV Caches for Chunked Text." 2024. https://arxiv.org/abs/2410.07590
91. Jiang, W., et al. "PipeRAG: Fast Retrieval-Augmented Generation via Algorithm-System Co-design." 2024. https://arxiv.org/abs/2403.05676
92. Faysse, M., et al. "ColPali: Efficient Document Retrieval with Vision Language Models." ICLR 2025. https://arxiv.org/abs/2407.01449
93. Yu, S., et al. "VisRAG: Vision-based Retrieval-augmented Generation on Multi-modality Documents." ICLR 2025. https://arxiv.org/abs/2410.10594
94. Cho, J., et al. "M3DocRAG: Multi-modal Retrieval is What You Need for Multi-page Multi-document Understanding." 2024. https://arxiv.org/abs/2411.04952
95. Bajaj, P., et al. "MS MARCO: A Human Generated MAchine Reading COmprehension Dataset." Microsoft, 2016. https://arxiv.org/abs/1611.09268
96. Thakur, N., et al. "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models." NeurIPS 2021 Datasets and Benchmarks. https://arxiv.org/abs/2104.08663 (code: https://github.com/beir-cellar/beir)
97. Muennighoff, N., et al. "MTEB: Massive Text Embedding Benchmark." EACL 2023. https://arxiv.org/abs/2210.07316
98. Su, H., et al. "BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval." ICLR 2025. https://arxiv.org/abs/2407.12883
99. Chen, J., Lin, H., Han, X., Sun, L. "Benchmarking Large Language Models in Retrieval-Augmented Generation (RGB)." AAAI 2024. https://arxiv.org/abs/2309.01431
100. Es, S., James, J., Espinosa-Anke, L., Schockaert, S. "RAGAS: Automated Evaluation of Retrieval Augmented Generation." EACL 2024 demo. https://arxiv.org/abs/2309.15217 (code: https://github.com/explodinggradients/ragas, now redirecting to https://github.com/vibrantlabsai/ragas)
101. Saad-Falcon, J., et al. "ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems." Stanford, NAACL 2024. https://arxiv.org/abs/2311.09476
102. Tang, Y., Yang, Y. "MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries." COLM 2024. https://arxiv.org/abs/2401.15391
103. Krishna, S., et al. "Fact, Fetch, and Reason: A Unified Evaluation of Retrieval-Augmented Generation (FRAMES)." Google, NAACL 2025. https://arxiv.org/abs/2409.12941
104. Pradeep, R., Thakur, N., et al. "Ragnarok: A Reusable RAG Framework and Baselines for TREC 2024 Retrieval-Augmented Generation Track." University of Waterloo / Microsoft, ECIR 2025. https://arxiv.org/abs/2406.16828
105. Gao, Y., et al. "Retrieval-Augmented Generation for Large Language Models: A Survey." Tongji / Fudan, 2023. https://arxiv.org/abs/2312.10997
106. Singh, A., Ehtesham, A., Kumar, S., Talaei Khoei, T. "Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG." 2025. https://arxiv.org/abs/2501.09136
107. Barnett, S., et al. "Seven Failure Points When Engineering a Retrieval Augmented Generation System." Deakin University, CAIN 2024. https://arxiv.org/abs/2401.05856
108. Shao, Y., et al. "Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models (STORM)." Stanford, NAACL 2024. https://arxiv.org/abs/2402.14207
109. Lala, J., et al. "PaperQA: Retrieval-Augmented Generative Agent for Scientific Research." FutureHouse, 2023. https://arxiv.org/abs/2312.07559
110. Qian, H., et al. "MemoRAG: Moving towards Next-Gen RAG via Memory-Inspired Knowledge Discovery." BAAI, 2024. https://arxiv.org/abs/2409.05591
111. "AutoSearch: Adaptive Search Depth for Efficient Agentic RAG via Reinforcement Learning." 2026. https://arxiv.org/abs/2604.17337 (code: https://github.com/bofusun/AutoSearch)
112. Cormack, G. V., Clarke, C. L. A., Buettcher, S. "Reciprocal Rank Fusion outperforms Condorcet and Individual Rank Learning Methods." SIGIR 2009. https://doi.org/10.1145/1571941.1572114
113. Gollapudi, S., et al. "Filtered-DiskANN: Graph Algorithms for Approximate Nearest Neighbor Search with Filters." Microsoft Research, WWW 2023. https://doi.org/10.1145/3543507.3583552
114. Singh, A., Subramanya, S. J., Krishnaswamy, R., Simhadri, H. V. "FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming Similarity Search." Microsoft Research, 2021. https://arxiv.org/abs/2105.09613 (code: https://github.com/microsoft/DiskANN)
115. Weller, O., Boratko, M., Naim, I., Lee, J. "On the Theoretical Limitations of Embedding-Based Retrieval." Google DeepMind / JHU, 2025. https://arxiv.org/abs/2508.21038
116. Jin, B., Yoon, J., Han, J., Arik, S. O. "Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG." Google / UIUC, ICLR 2025. https://arxiv.org/abs/2410.05983
117. Wei, J., et al. "BrowseComp: A Simple Yet Challenging Benchmark for Browsing Agents." OpenAI, 2025. https://arxiv.org/abs/2504.12516
118. Greshake, K., Abdelnabi, S., Mishra, S., Endres, C., Holz, T., Fritz, M. "Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection." AISec 2023. https://arxiv.org/abs/2302.12173
119. Zou, W., Geng, R., Wang, B., Jia, J. "PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models." USENIX Security 2025. https://arxiv.org/abs/2402.07867
120. Zeng, S., et al. "The Good and The Bad: Exploring Privacy Issues in Retrieval-Augmented Generation (RAG)." Findings of ACL 2024. https://arxiv.org/abs/2402.16893
121. Xiang, C., Wu, T., Zhong, Z., Wagner, D., Chen, D., Mittal, P. "Certifiably Robust RAG against Retrieval Corruption (RobustRAG)." Princeton / UC Berkeley, 2024. https://arxiv.org/abs/2405.15556
