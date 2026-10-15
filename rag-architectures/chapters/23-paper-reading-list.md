# 23. Canonical RAG and Retrieval Paper Reading List: 120+ Papers (2017-2026) by Theme, with System-Design Takeaways

## Overview

This chapter is a curated reading list. It is not a bibliography dump. Each entry was chosen
because it changed how production retrieval-augmented generation (RAG) systems are built:
it introduced a component that is now standard (dense retrievers, rerankers, late
interaction, HNSW), set a failure model that engineers design against (lost in the middle,
distractor sensitivity, single-vector capacity limits), or defined a benchmark that teams
still use. The list covers 2017 to 2026. It begins with retrieve-then-read open-domain QA
(DrQA) and ends with RL-trained search agents, KV-cache-level RAG decoding and the 2026
agentic RAG systematizations.

The papers fall into fourteen themes:

1. Foundations: retrieve-then-read and retrieval-augmented language models
2. Dense, sparse and late-interaction retrievers, and general-purpose embedders
3. Approximate nearest neighbour (ANN) indexing
4. Reranking
5. Query transformation and synthetic training data
6. Context construction: chunking, ordering, compression, hierarchy
7. Adaptive, iterative and agentic retrieval
8. Training generators for RAG
9. Graph and structured RAG
10. Long context, KV caching and RAG decoding
11. Multimodal (visual document) retrieval
12. Evaluation and benchmarks
13. Security and privacy
14. Surveys and engineering experience reports

Each theme has a short design narrative and a table with venue, year, a reference link
(cited as [n] and listed in full under References) and a single takeaway for system design.
Deeper notes follow for the papers whose numbers matter most. Every figure below comes from
the cited paper or from the primary-source analyses in earlier chapters of this atlas.
Approximate figures are marked "about" or "approx.".

### A suggested reading order

Most readers do not need all 120 papers. The following path gives a working mental model in
about 25 papers. The rest of the list is for going deeper on one theme.

```
  2017-2020 FOUNDATIONS          2020-2023 RETRIEVER STACK       2023-2024 CONTEXT + CONTROL
  DrQA [1] -> ORQA [2]           DPR [4] / BM25 baseline         Lost in the Middle [53]
     -> REALM [3] -> RAG [5]     ColBERT/ColBERTv2 [14][15]      RAPTOR [57], Contextual [59]
     -> FiD [6] -> RETRO [9]     SPLADE [20], Contriever [21]    IRCoT [63] -> FLARE [64]
     -> Atlas [10]               BEIR [102], MTEB [103]          Self-RAG [65] -> CRAG [66]
                                 HNSW [35], DiskANN [39]         GraphRAG [81], HippoRAG [83]
                                 monoT5 [44] -> RankGPT [45]

  2024-2026 SCALE + AGENTS                                         EVALUATE EVERYTHING
  Self-Route [90], CAG [93], CacheBlend [95], REFRAG [97]          RAGAS [107], ARES [108]
  Search-o1 [68] -> Search-R1 [69] -> DeepResearcher [72]          CRAG bench [111]
  LIMIT [34] (why single vectors are not enough)                   BrowseComp-Plus [115]
```

## 1. Foundations: retrieve-then-read and retrieval-augmented LMs

**Problem.** Language models store knowledge in their weights. That knowledge is hard to
update or attribute, and it is weak on rare facts. The 2017-2022 papers separated *where
knowledge lives* (an external, swappable corpus) from *how it is used* (a reader or
generator).

**Architecture lineage.** DrQA [1] fixed the two-stage template: a TF-IDF/bigram-hash
retriever over Wikipedia feeds an RNN reader. ORQA [2] made the retriever learned and
treated the retrieved block as a latent variable. REALM [3] pushed that latent retrieval
into pre-training with an asynchronously refreshed MIPS index. DPR [4] showed that a
supervised BERT dual encoder alone beats BM25 by a wide margin. RAG [5] coined the name: a
DPR retriever plus a BART generator, with marginalisation over documents per sequence or per
token. FiD [6] encodes each passage independently and fuses them in the decoder, so reader
cost grows linearly with passage count. RETRO [9] and Atlas [10] scaled retrieval to
trillion-token datastores and few-shot settings. In-Context RALM [11] and REPLUG [12]
marked the API era: retrieval works with a frozen, black-box LM through the prompt alone.

```
 Query --> [Retriever: sparse | dense | latent] --> top-k passages
                                                       |
            +------------------------------------------+-------------------+
            | concat in prompt     | FiD: encode each,  | RETRO: chunked    |
            | (RAG, RALM, REPLUG)  | fuse in decoder    | cross-attention   |
            +----------------------+--------------------+-------------------+
                                                       |
                                                  Generator --> answer
```

| # | Paper | Venue, year | Key system-design takeaway |
|---|-------|-------------|----------------------------|
| 1 | DrQA: Reading Wikipedia to Answer Open-Domain Questions [1] | ACL 2017 | A cheap sparse retriever plus a strong reader is a viable baseline; retriever recall caps end-to-end accuracy. |
| 2 | ORQA: Latent Retrieval for Weakly Supervised Open-Domain QA [2] | ACL 2019 | The retriever can be trained from answer-only supervision; Inverse Cloze Task pre-training bootstraps it. |
| 3 | REALM [3] | ICML 2020 | End-to-end retriever training requires periodic index refresh; stale indexes hurt. |
| 4 | DPR: Dense Passage Retrieval [4] | EMNLP 2020 | In-batch negatives plus one BM25 hard negative are enough for a strong dual encoder. |
| 5 | RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP [5] | NeurIPS 2020 | Swapping the document index updates model knowledge without retraining. |
| 6 | FiD: Leveraging Passage Retrieval with Generative Models [6] | EACL 2021 | Accuracy keeps rising with more passages (up to 100) when the reader fuses them in the decoder. |
| 7 | FiD-KD: Distilling Knowledge from Reader to Retriever [7] | ICLR 2021 | Reader cross-attention scores are a free relevance label for retriever training. |
| 8 | kNN-LM: Generalization through Memorization [8] | ICLR 2020 | Interpolating a token-level datastore lowers perplexity with no retraining; the datastore can be domain-swapped. |
| 9 | RETRO: Improving LMs by Retrieving from Trillions of Tokens [9] | ICML 2022 | Retrieval can substitute for parameters: a 7.5B model competes with models about 25x larger. |
| 10 | Atlas: Few-shot Learning with Retrieval-Augmented LMs [10] | JMLR 2023 | Jointly pre-training retriever and reader gives strong few-shot QA (64-shot NQ 42.4 EM). |
| 11 | In-Context RALM [11] | TACL 2023 | Prepending retrieved text to a frozen LM is a large, free win; retrieve often, with short query windows. |
| 12 | REPLUG [12] | NAACL 2024 | Treat the LM as a black box and tune only the retriever from LM likelihood feedback. |
| 13 | WebGPT [13] | arXiv 2021 | Browsing plus quoted references makes answers verifiable and lets humans judge factuality. |

**Deeper notes.** DPR's headline figure is top-20 retrieval accuracy on Natural Questions of
78.4% against 59.1% for BM25 [4]. That 19-point retrieval gain carried through to end-to-end
QA. It was the result that moved industry from pure lexical search to dense retrieval. REALM
used a 330M-parameter model to reach 40.4 EM on NQ-Open. Its ablations showed that a much
staler index (refreshed far less often) fell *below* the baseline [3]. That is the first
published evidence that index freshness is a correctness issue, not only an ops issue.
RETRO splits text into 64-token chunks and retrieves neighbours from a 2-trillion-token
database with a frozen BERT plus SCaNN. Neighbours enter through chunked cross-attention.
DeepMind summarised the result as GPT-3-level performance with 25x fewer parameters [9].
RETRO also warned that test-set leakage into the retrieval corpus inflates scores, which
matters for anyone benchmarking a RAG system on public data.

**Lessons.** (a) Freeze the document encoder whenever you can. REALM, RAG, Atlas and RETRO
all did, at some stage, to avoid costly re-indexing. (b) Reader architecture decides how
many passages you can use. (c) The API era made retrieval a prompt-construction problem.
Most later work optimises what goes *into* the prompt.

## 2. Retrievers and embedding models

**Problem.** Dense retrievers trained on one dataset generalise poorly. BEIR [102] showed
BM25 beating many neural retrievers zero-shot. Single-vector models also compress a whole
passage into one point. Three answers emerged: better negatives and distillation (ANCE,
RocketQA, TAS-B), different representations (late interaction, learned sparse), and
massive weakly supervised or LLM-based pre-training (Contriever, E5, GTE, BGE, Gecko,
NV-Embed, Qwen3, Gemini Embedding).

| # | Paper | Venue, year | Key system-design takeaway |
|---|-------|-------------|----------------------------|
| 14 | ColBERT [14] | SIGIR 2020 | Late interaction (per-token vectors, MaxSim) keeps cross-encoder-like quality with precomputed documents. |
| 15 | ColBERTv2 [15] | NAACL 2022 | Residual compression shrinks multi-vector indexes by about 6-10x; distillation from a cross-encoder adds quality. |
| 16 | PLAID [16] | CIKM 2022 | Centroid pruning before full MaxSim scoring makes late interaction fast enough to serve. |
| 17 | ANCE [17] | ICLR 2021 | Hard negatives mined from an asynchronously refreshed ANN index beat random/BM25 negatives. |
| 18 | RocketQA [18] | NAACL 2021 | Cross-batch negatives, denoised hard negatives and cross-encoder-labelled data improve dual encoders. |
| 19 | TAS-B [19] | SIGIR 2021 | Topic-aware batch sampling plus balanced margin distillation trains strong retrievers cheaply. |
| 20 | SPLADE [20] | SIGIR 2021 | Learned sparse expansion runs on inverted indexes and gives exact-term matching with semantic recall. |
| 21 | Contriever [21] | TMLR 2022 | Unsupervised contrastive pre-training gives a robust zero-shot dense retriever. |
| 22 | E5 [22] | arXiv 2022 | Weakly supervised pairs mined from the web (CCPairs) produce general embedders; use query/passage prefixes. |
| 23 | E5-Mistral: Improving Text Embeddings with LLMs [23] | ACL 2024 | A decoder LLM fine-tuned on synthetic multilingual data becomes a top embedder. |
| 24 | GTE [24] | arXiv 2023 | Multi-stage contrastive learning (pre-train, then fine-tune) is the standard recipe. |
| 25 | C-Pack / BGE [25] | SIGIR 2024 | Open data, models and training recipes made strong open embedders the default. |
| 26 | BGE-M3 [26] | arXiv 2024 | One model can emit dense, sparse and multi-vector outputs; hybrid retrieval from one forward pass. |
| 27 | Matryoshka Representation Learning [27] | NeurIPS 2022 | Nested embeddings let you truncate dimensions for cheap first-pass search and rerank at full size. |
| 28 | Gecko [28] | arXiv 2024 | Distil LLM knowledge into a compact embedder by having the LLM generate and relabel training pairs. |
| 29 | NV-Embed [29] | ICLR 2025 | Latent-attention pooling and two-stage instruction tuning make decoder LLMs strong embedders. |
| 30 | Qwen3 Embedding [30] | arXiv 2025 | Open embedding and reranking models built from one foundation family; synthetic data at scale. |
| 31 | Gemini Embedding [31] | arXiv 2025 | Initialising from a frontier LLM gives broad multilingual and code retrieval quality. |
| 32 | Promptriever [32] | ICLR 2025 | Retrievers can follow per-query natural-language instructions like an LM. |
| 33 | ReasonIR [33] | arXiv 2025 | Reasoning-intensive retrieval needs purpose-built synthetic training data, not only web pairs. |
| 34 | LIMIT: Theoretical Limitations of Embedding-Based Retrieval [34] | arXiv 2025 | The number of top-k subsets a single-vector embedder can represent is bounded by its dimension. |

**Deeper notes.** LIMIT (Google DeepMind) is the most consequential 2025 result for
architecture choices [34]. It proves a dimension-dependent bound on which combinations of
documents can be returned as top-k. It confirms this with "free embedding" optimisation and
a deliberately simple dataset on which state-of-the-art single-vector embedders fail. BM25
and multi-vector models do much better there. The practical reading: single-vector dense
retrieval is a recall layer, not a complete solution. Keep a lexical or multi-vector path and
a reranker for combinatorial and exact-match queries. ColBERTv2's residual compression [15]
and PLAID's centroid-interaction pruning [16] are the reason late interaction is deployable
at all. Matryoshka embeddings [27] are the reason "truncate to 256 dims for ANN, rescore at
full width" became a common storage pattern.

## 3. Approximate nearest neighbour indexing

**Problem.** Exact maximum inner product search over 10^8 to 10^10 vectors is too slow and
too memory-hungry. Each index family makes a different trade-off between memory, update
cost, filtering and recall.

| # | Paper | Venue, year | Key system-design takeaway |
|---|-------|-------------|----------------------------|
| 35 | HNSW [35] | TPAMI 2018 (arXiv 2016) | A layered proximity graph gives log-like search and excellent recall; RAM-heavy, and deletes are awkward. |
| 36 | Billion-scale similarity search with GPUs (Faiss) [36] | IEEE TBD 2019 (arXiv 2017) | GPU IVF-PQ with k-selection makes billion-scale brute-force-like search practical. |
| 37 | The Faiss library [37] | arXiv 2024 | A design-space map of IVF, PQ, graph and refine stages; index choice is a memory/recall/latency trade-off. |
| 38 | ScaNN: Anisotropic Vector Quantization [38] | ICML 2020 | Quantisation loss should weight the error component parallel to the query; better MIPS recall per byte. |
| 39 | DiskANN [39] | NeurIPS 2019 | A Vamana graph on SSD with compressed vectors in RAM serves a billion points from one node. |
| 40 | FreshDiskANN [40] | arXiv 2021 | Graph indexes can absorb streaming inserts and deletes with background merges. |
| 41 | SPFresh [41] | SOSP 2023 | In-place incremental rebalancing of cluster-based indexes avoids costly global rebuilds. |
| 42 | ACORN [42] | SIGMOD 2024 | Predicate-agnostic filtered vector search by denser graph construction and filtered traversal. |

**Deeper notes.** DiskANN reported serving about 1 billion points on a 64 GB RAM workstation
with high recall and millisecond-scale latency (approx.) [39]. That result is why SSD-resident
graph indexes now underpin several managed vector databases. Filtered search (ACORN [42])
is the most common production pain point: post-filtering an HNSW result list collapses
recall when the filter is selective. Chapter 04 covers this in depth.

## 4. Reranking

**Problem.** First-stage retrievers optimise recall at low cost. Precision at the top of the
list, which is what the LLM actually reads, needs a more expensive model over a short list.

| # | Paper | Venue, year | Key system-design takeaway |
|---|-------|-------------|----------------------------|
| 43 | monoBERT: Passage Re-ranking with BERT [43] | arXiv 2019 | A cross-encoder over (query, passage) pairs gives large precision gains over BM25 lists. |
| 44 | monoT5: Document Ranking with a Pretrained Seq2Seq Model [44] | Findings EMNLP 2020 | Scoring relevance as the probability of a "true" token transfers well zero-shot. |
| 45 | RankGPT: Is ChatGPT Good at Search? [45] | EMNLP 2023 | LLMs rerank well listwise with sliding windows; distil into small rankers to cut cost. |
| 46 | RankZephyr [46] | arXiv 2023 | An open 7B listwise reranker can match proprietary LLM reranking. |
| 47 | Rank1: Test-Time Compute for Reranking [47] | arXiv 2025 | Reasoning traces before a relevance judgment help on reasoning-heavy retrieval, at a latency cost. |

**Lesson.** The two-stage cascade (cheap recall, then expensive precision on 50-200
candidates) has been stable from monoBERT to Rank1. What changes is the scorer's cost per
pair. Budget reranking latency explicitly; it is often the largest retrieval-side cost.

## 5. Query transformation and synthetic training data

| # | Paper | Venue, year | Key system-design takeaway |
|---|-------|-------------|----------------------------|
| 48 | HyDE: Precise Zero-Shot Dense Retrieval without Relevance Labels [48] | ACL 2023 | Embed an LLM-written hypothetical answer instead of the query; helps zero-shot, risks hallucinated anchors. |
| 49 | Query2doc [49] | EMNLP 2023 | Expand the query with a pseudo-document; helps both BM25 and dense retrieval. |
| 50 | InPars [50] | arXiv 2022 | Few-shot LLM prompting generates synthetic queries to train domain rerankers. |
| 51 | Promptagator [51] | ICLR 2023 | Eight task examples plus LLM query generation train task-specific retrievers. |
| 52 | GPL: Generative Pseudo Labeling [52] | NAACL 2022 | Generated queries plus cross-encoder pseudo-labels adapt dense retrievers to new domains without labels. |

**Lesson.** HyDE and Query2doc add one LLM call to the query path, so they trade latency for
recall. InPars, Promptagator and GPL move the LLM cost offline into training-data generation.
For high-QPS systems the offline route usually wins. Chapters 06 and 17 compare the two.

## 6. Context construction: chunking, ordering, compression and hierarchy

**Problem.** Retrieval quality is necessary but not sufficient. How retrieved text is cut,
ordered and compressed decides whether the generator uses it.

| # | Paper | Venue, year | Key system-design takeaway |
|---|-------|-------------|----------------------------|
| 53 | Lost in the Middle [53] | TACL 2024 | Accuracy is U-shaped in evidence position; put the best evidence at the start or end. |
| 54 | The Power of Noise [54] | SIGIR 2024 | Related-but-irrelevant distractors hurt more than random noise; rerank to remove near-misses. |
| 55 | RECOMP [55] | ICLR 2024 | Extractive and abstractive compressors cut prompt length a lot with little accuracy loss, and may return nothing. |
| 56 | LongLLMLingua [56] | ACL 2024 | Question-aware token pruning and reordering reduce cost and mitigate position bias. |
| 57 | RAPTOR [57] | ICLR 2024 | Recursive cluster-and-summarise trees let one query retrieve at several levels of abstraction. |
| 58 | Late Chunking [58] | arXiv 2024 | Embed the whole document with a long-context encoder, then pool per chunk to keep cross-chunk context. |
| 59 | Contextual Retrieval (Anthropic) [59] | Engineering blog, 2024 | Prepend an LLM-written 50-100 token context to each chunk before embedding and BM25. |
| 60 | Is Semantic Chunking Worth the Computational Cost? [60] | Findings NAACL 2025 | Semantic chunking's gains over fixed-size chunking are inconsistent and often not worth the cost. |

**Deeper notes.** Anthropic's contextual retrieval write-up is one of the few publicly
quantified chunking interventions [59]. Across its test corpora, the top-20 retrieval failure
rate fell from 5.7% to 3.7% with contextual embeddings (35% lower), to 2.9% with contextual
embeddings plus contextual BM25 (49% lower), and to 1.9% with reranking added (67% lower).
The one-time preprocessing cost was about $1.02 per million document tokens with prompt
caching, assuming 800-token chunks and 8k-token documents. RAPTOR reported a 20% absolute
accuracy gain on the QuALITY benchmark when paired with GPT-4 [57]. Its tree is an index-time
cost that must be rebuilt when documents change. Lost in the Middle [53] is the single most
cited reason for "fewer, better chunks" designs.

```
   doc --> chunk --+--> [contextualize: LLM adds 50-100 tok header] --+--> embed --> ANN
                   |                                                  +--> BM25 index
                   +--> [late chunking: embed whole doc, pool spans] ------> ANN
                   +--> [RAPTOR: cluster -> summarize -> recurse]   -------> tree index
   query --> hybrid recall --> rerank --> order (best at edges) --> compress --> LLM
```

## 7. Adaptive, iterative and agentic retrieval

**Problem.** One retrieval pass fails on multi-hop questions and wastes compute on easy ones.
The field moved through four control strategies. Fixed loops came first (IRCoT). Then came
uncertainty-triggered retrieval (FLARE), self-critique (Self-RAG, CRAG), and complexity
routing (Adaptive-RAG). Since 2025, learned policies have been trained with reinforcement
learning against live or simulated search (Search-R1 and successors).

```
  2022-23 fixed loop      2023 trigger         2023-24 critique       2025-26 learned policy
  IRCoT: CoT step ->      FLARE: retrieve      Self-RAG: reflection   Search-R1: RL on
  query -> retrieve       when next-sentence   tokens; CRAG: T5       <search>/<answer>
  -> next CoT step        confidence is low    evaluator + web        with outcome reward
```

| # | Paper | Venue, year | Key system-design takeaway |
|---|-------|-------------|----------------------------|
| 61 | ReAct [61] | ICLR 2023 | Thought/Action/Observation is the interface nearly every search agent still uses. |
| 62 | Toolformer [62] | NeurIPS 2023 | Models can self-supervise when to call tools (including search) by keeping calls that lower loss. |
| 63 | IRCoT [63] | ACL 2023 | Use each chain-of-thought sentence as the next retrieval query for multi-hop QA. |
| 64 | FLARE [64] | EMNLP 2023 | Retrieve only when the model's predicted next sentence has low-confidence tokens; requires logprobs. |
| 65 | Self-RAG [65] | ICLR 2024 | Train one LM to emit retrieve/relevance/support/utility tokens; controllable at inference. |
| 66 | CRAG: Corrective RAG [66] | arXiv 2024 | A light retrieval evaluator triggers refine, discard or web fallback; quality bounded by the evaluator. |
| 67 | Adaptive-RAG [67] | NAACL 2024 | A small classifier routes no-retrieval / single-step / multi-step, trading accuracy against cost. |
| 68 | Search-o1 [68] | EMNLP 2025 | Reasoning models can call search mid-thought; a Reason-in-Documents step condenses results first. |
| 69 | Search-R1 [69] | COLM 2025 | RL with outcome-only reward and masked retrieved tokens teaches when and what to search. |
| 70 | R1-Searcher [70] | arXiv 2025 | Two-stage outcome-supervised RL elicits search without SFT cold-start data. |
| 71 | ReSearch [71] | arXiv 2025 | Search calls become part of the reasoning chain, trained with RL and no supervised reasoning steps. |
| 72 | DeepResearcher [72] | arXiv 2025 | End-to-end RL against the real web beats prompt-engineered and local-RAG-trained agents. |
| 73 | WebThinker [73] | NeurIPS 2025 | A deep-web explorer that navigates pages lets reasoning models write research reports. |
| 74 | Anthropic multi-agent research system [74] | Engineering blog, 2025 | An orchestrator with parallel subagents scales breadth-first research at a large token cost. |

**Deeper notes.** Self-RAG 13B reported 55.8 accuracy on PopQA and 69.3 on TriviaQA,
beating retrieval-augmented baselines of the time [65]. Its inference cost varies with how
often the model chooses to retrieve. CRAG's lightweight T5-large evaluator classifies
retrievals as correct, incorrect or ambiguous [66]. The authors note that the method is only
as good as that evaluator. Search-R1 [69] defines the 2025-2026 template. The policy LLM
emits `<search>` queries; retrieved passages are inserted and masked out of the RL loss; a
rule-based exact-match reward on the final answer drives PPO or GRPO. Its follow-ups differ
mainly in the environment: local Wikipedia, simulated search (ZeroSearch), or the live web
(DeepResearcher). On Anthropic's internal research evaluation, the multi-agent configuration
(Opus 4 lead, Sonnet 4 subagents) beat single-agent Opus 4 by about 90%. Anthropic also
reported that such systems use roughly 15x the tokens of a chat interaction and that token
usage explained most of the performance variance (approx. figures from the post) [74].

**Failure modes.** Uncertainty triggers depend on logprob access, which hosted APIs often
restrict. Critic-based loops multiply LLM calls. RL agents can over-search or reward-hack
exact-match rewards. Chapter 08 covers each in detail.

## 8. Training generators for RAG

| # | Paper | Venue, year | Key system-design takeaway |
|---|-------|-------------|----------------------------|
| 75 | RA-DIT [75] | ICLR 2024 | Two light fine-tunes (LM learns to use retrieved text; retriever learns LM preferences) retrofit any LLM. |
| 76 | RAFT [76] | COLM 2024 | Fine-tune with oracle plus distractor documents and cited chain-of-thought answers for domain RAG. |
| 77 | ChatQA [77] | NeurIPS 2024 | Two-stage instruction tuning makes an open model competitive on conversational RAG. |
| 78 | RankRAG [78] | NeurIPS 2024 | One instruction-tuned LLM can both rerank contexts and generate the answer. |
| 79 | Fine-Tuning or Retrieval? [79] | EMNLP 2024 | For injecting new facts, RAG consistently beats unsupervised fine-tuning. |
| 80 | Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations? [80] | EMNLP 2024 | Models learn new facts slowly through fine-tuning, and doing so raises hallucination; prefer retrieval. |

**Lesson.** Fine-tuning and RAG are complements. Fine-tune for *behaviour*: ignoring
distractors, citing, abstaining. Retrieve for *facts*. RAFT [76] and RA-DIT [75] are the
two recipes to start from. Chapter 17 expands on this.

## 9. Graph and structured RAG

**Problem.** Chunk-level similarity cannot answer global, corpus-wide questions ("what are
the main themes?"). It also struggles with multi-hop entity paths. Graph approaches extract
entities and relations at index time, then traverse or summarise at query time.

| # | Paper | Venue, year | Key system-design takeaway |
|---|-------|-------------|----------------------------|
| 81 | GraphRAG: From Local to Global [81] | arXiv 2024 | LLM-built entity graphs plus hierarchical community summaries answer global sensemaking queries; indexing is costly. |
| 82 | LightRAG [82] | EMNLP 2025 | Dual-level (entity and theme) keys with incremental graph updates cut GraphRAG's cost. |
| 83 | HippoRAG [83] | NeurIPS 2024 | OpenIE triples plus Personalized PageRank give single-step multi-hop retrieval. |
| 84 | HippoRAG 2: From RAG to Memory [84] | ICML 2025 | Adding passages to the graph and filtering recognised triples improves factual, sense-making and associative memory together. |
| 85 | G-Retriever [85] | NeurIPS 2024 | Retrieve a subgraph via prize-collecting Steiner tree and pass it to the LLM as soft graph tokens plus text. |
| 86 | KAG (Ant Group) [86] | arXiv 2024 | Professional domains need schema-constrained KGs and logical-form reasoning, not only open extraction. |
| 87 | RAG vs. GraphRAG: A Systematic Evaluation [87] | arXiv 2025 | Graph methods win on multi-hop and global queries; plain RAG often wins on single-hop detail. |
| 88 | RAG with KGs for Customer Service QA (LinkedIn) [88] | SIGIR 2024 | Ticket-structure-aware KGs improved retrieval and cut per-issue resolution time in production. |

**Deeper notes.** HippoRAG reported up to 20% better multi-hop QA than prior methods while
being 10-30x cheaper and 6-13x faster than iterative retrieval (IRCoT) [83]. That cost profile
is why graph retrieval is often a replacement for agent loops, not an addition to them.
GraphRAG's community summaries are costly to build. Microsoft's later LazyGraphRAG work
(covered in Chapter 07) reported indexing cost at about 0.1% of full GraphRAG by deferring
LLM work to query time. Treat graph RAG as a query-type-specific tool. Systematic
comparisons [87] show no universal winner.

## 10. Long context, KV caching and RAG decoding

**Problem.** As context windows passed 128K tokens, teams asked whether to retrieve at all.
The research answer: long context and RAG are complementary. When the corpus is small and
stable, preloading wins. Otherwise retrieval wins, and the serving stack should exploit
RAG's structure (independent chunks, sparse relevance).

| # | Paper | Venue, year | Key system-design takeaway |
|---|-------|-------------|----------------------------|
| 89 | Retrieval meets Long Context LLMs [89] | ICLR 2024 | Retrieval with a 4K window can rival a 16K long-context model, and helps long-context models too. |
| 90 | Self-Route: RAG or Long-Context LLMs? [90] | EMNLP 2024 Industry | Let the model judge if retrieved chunks suffice; fall back to long context only when needed. |
| 91 | LongRAG [91] | arXiv 2024 | Retrieve long units (about 4K tokens) to shrink the corpus and shift work to a long-context reader. |
| 92 | Long-Context LLMs Meet RAG [92] | ICLR 2025 | More retrieved passages first help, then hurt, because of hard negatives; reorder and fine-tune for robustness. |
| 93 | CAG: Don't Do RAG [93] | 2025 (short paper) | For a small, bounded knowledge base, precompute the KV cache once and skip retrieval. |
| 94 | RAGCache [94] | arXiv 2024 | Cache KV states of popular retrieved documents in a knowledge tree across GPU and host memory. |
| 95 | CacheBlend [95] | EuroSys 2025 | Reuse per-chunk KV at non-prefix positions by recomputing a small subset of tokens. |
| 96 | TurboRAG [96] | arXiv 2024 | Precompute chunk KV caches offline and fine-tune the model to tolerate independent chunk attention. |
| 97 | REFRAG [97] | arXiv 2025 | Compress retrieved chunks into embeddings and expand only the selected ones; large time-to-first-token gains. |

**Deeper notes.** CacheBlend recomputes KV for only a small fraction of tokens (Chapter 09
cites roughly 10-20%) whose attention deviates most. This recovers cross-chunk attention while
keeping most of the reuse savings [95]. REFRAG (Meta) feeds a lightweight encoder's chunk
embeddings to the decoder in place of full token sequences. An RL policy decides which
chunks to expand. The paper reports large time-to-first-token speedups (around 30x, approx.)
and an extended effective context [97]. The common thread: RAG contexts are block-sparse, and
serving systems that know chunk boundaries can skip most cross-chunk attention.

```
   Classic RAG:   [sys][q][chunk1 tokens][chunk2 tokens]...[chunkK tokens] -> full prefill
   Prefix cache:  [sys cached][q][chunks recomputed]                       (APIs today)
   CacheBlend:    [sys][q][KV1*][KV2*]...  (* reused; ~10-20% tokens recomputed)
   REFRAG:        [sys][q][e1][e2][chunk3 tokens][e4]... (e = one embedding per chunk)
   CAG:           [entire KB KV cached][q]                  (no retriever at all)
```

## 11. Multimodal (visual document) retrieval

| # | Paper | Venue, year | Key system-design takeaway |
|---|-------|-------------|----------------------------|
| 98 | ColPali [98] | ICLR 2025 | Embed page *images* with a VLM and late interaction; skip OCR/layout pipelines for visually rich PDFs. |

ColPali (PaliGemma-3B base) produces about 1,030 vectors per page and introduced the ViDoRe
benchmark. In the colpali-engine listing, v1.0 and v1.3 score about 81.3 and 84.8 [98]. The
storage cost of multi-vector pages is the main deployment concern. Binary quantisation and
token pooling are the usual mitigations (Chapter 10).

## 12. Evaluation and benchmarks

**Problem.** RAG has two failure surfaces: retrieval (did we fetch the evidence?) and
generation (did we use it faithfully?). Benchmarks have moved from extractive QA to
attribution, robustness, multi-hop reasoning and deep-research agents.

| # | Paper | Venue, year | Key system-design takeaway |
|---|-------|-------------|----------------------------|
| 99 | MS MARCO [99] | arXiv 2016 | The training set behind most neural rankers; shallow labels mean many false negatives. |
| 100 | HotpotQA [100] | EMNLP 2018 | Multi-hop questions with supporting-fact labels; the standard testbed for iterative retrieval. |
| 101 | KILT [101] | NAACL 2021 | One Wikipedia snapshot across 11 datasets and 5 tasks; scores provenance as well as output. |
| 102 | BEIR [102] | NeurIPS 2021 D&B | Zero-shot evaluation over 18 datasets; BM25 is a strong out-of-domain baseline. |
| 103 | MTEB [103] | EACL 2023 | The 8-task, 58-dataset embedding leaderboard; retrieval average is not your domain. |
| 104 | MMTEB [104] | ICLR 2025 | Community-built multilingual extension to 500+ tasks; reduced, cheaper benchmark variants. |
| 105 | BRIGHT [105] | ICLR 2025 | Reasoning-intensive retrieval where lexical and semantic overlap fail; reasoning rewrites help. |
| 106 | RGB: Benchmarking LLMs in RAG [106] | AAAI 2024 | Tests noise robustness, negative rejection, information integration and counterfactual robustness. |
| 107 | RAGAS [107] | EACL 2024 Demo | Reference-free faithfulness, answer relevance and context relevance via LLM judges. |
| 108 | ARES [108] | NAACL 2024 | Fine-tuned judges plus prediction-powered inference give confidence intervals from few labels. |
| 109 | ALCE: Generating Text with Citations [109] | EMNLP 2023 | Score citation recall and precision via NLI; citation quality lags fluency. |
| 110 | RAGChecker [110] | NeurIPS 2024 D&B | Claim-level metrics separate retriever errors from generator errors. |
| 111 | CRAG: Comprehensive RAG Benchmark (Meta) [111] | NeurIPS 2024 D&B | Mock web and KG APIs, dynamic and long-tail questions; hallucination is penalised. |
| 112 | FRAMES [112] | arXiv 2024 | Unified test of fact retrieval, multi-hop reasoning and numerical reasoning; multi-step retrieval helps sharply. |
| 113 | MultiHop-RAG [113] | COLM 2024 | Multi-hop queries over news; exposes single-shot retrieval limits. |
| 114 | TREC 2024 RAG Track nugget evaluation [114] | arXiv 2024 | Automatic nugget extraction and matching approximates human RAG judgments at scale. |
| 115 | BrowseComp-Plus [115] | arXiv 2025 | A fixed, curated corpus makes deep-research agent evaluation reproducible and separates retriever from agent. |

**Lesson.** Evaluate the retriever (recall@k against labelled evidence) separately from
the generator (faithfulness, citation precision). Then evaluate end to end on a held-out,
time-split slice. RAGChecker [110] and ARES [108] are the best published patterns for
attributing errors and putting confidence bounds on LLM-judge scores. Chapter 11 has the
full treatment.

## 13. Security and privacy

| # | Paper | Venue, year | Key system-design takeaway |
|---|-------|-------------|----------------------------|
| 116 | Not What You've Signed Up For: Indirect Prompt Injection [116] | AISec 2023 | Retrieved content is an attack channel; treat every document as untrusted input. |
| 117 | PoisonedRAG [117] | USENIX Security 2025 | A few crafted passages injected into a large corpus can steer answers to target questions. |
| 118 | Text Embeddings Reveal (Almost) As Much As Text [118] | EMNLP 2023 | Embeddings can be inverted to recover text; vector stores need the same protection as raw documents. |

**Lesson.** Enforce ACLs at retrieval time. Keep provenance on every chunk. Isolate retrieved
text from instructions. Never treat embeddings as anonymised data (Chapter 16).

## 14. Surveys, frameworks and engineering experience reports

| # | Paper | Venue, year | Key system-design takeaway |
|---|-------|-------------|----------------------------|
| 119 | RAG for LLMs: A Survey [119] | arXiv 2023 | Introduced the Naive / Advanced / Modular RAG taxonomy used across industry. |
| 120 | Seven Failure Points When Engineering a RAG System [120] | CAIN 2024 | Missing content, missed top-k, not in context, not extracted, wrong format, wrong specificity, incomplete. |
| 121 | Agentic RAG survey [121] | arXiv 2025 (rev. 2026) | Taxonomy of single-agent, multi-agent, hierarchical and graph-based agentic RAG. |
| 122 | SoK: Agentic RAG [122] | arXiv 2026 | 2026 systematization of agentic RAG architectures, evaluation and open problems. |
| 123 | FlashRAG [123] | WWW 2025 Companion | One toolkit reimplements dozens of RAG methods under uniform settings; many gains shrink. |
| 124 | DSPy [124] | ICLR 2024 | Declare RAG pipelines as modules and compile prompts/demos against a metric. |

**Deeper notes.** The Seven Failure Points paper [120] came from three real deployments
(research, education and biomedical). It concludes that RAG systems are validated in
operation, not at design time, and that robustness evolves rather than being designed in.
FlashRAG's uniform reproductions [123] give a sobering check. Under identical retrievers and
generators, several celebrated adaptive methods score close to a well-tuned standard RAG
baseline. Always re-baseline before adopting a method from a paper.

## Cross-theme comparison

The table below compresses the list into the decisions a system designer actually makes.
Each row names the representative papers and what they imply for cost, latency and quality.

| Design decision | Representative papers | Quality lever | Cost / latency impact | Main failure mode |
|-----------------|-----------------------|---------------|-----------------------|-------------------|
| Sparse vs dense vs hybrid recall | BM25 in [4][102], DPR [4], SPLADE [20], BGE-M3 [26], LIMIT [34] | Hybrid closes lexical and semantic gaps | Dense needs ANN memory; sparse uses inverted index | Single-vector capacity limit; domain shift |
| Single- vs multi-vector | ColBERTv2 [15], PLAID [16], ColPali [98] | Token-level matching, robust zero-shot | Index often several x larger; extra scoring | Storage blow-up |
| Reranker | monoT5 [44], RankGPT [45], Rank1 [47] | Top-k precision | 10s-100s of ms per query; LLM rerankers more | Latency budget overrun |
| Chunking / contextualisation | Contextual Retrieval [59], Late Chunking [58], RAPTOR [57] | Fewer retrieval misses (-35% to -67% failure in [59]) | One-time index cost (about $1/M tokens in [59]) | Re-index on every doc change |
| Context ordering / compression | Lost in Middle [53], RECOMP [55], LongLLMLingua [56] | Higher use of evidence | Fewer prompt tokens | Compressor drops key facts |
| When to retrieve | FLARE [64], Self-RAG [65], Adaptive-RAG [67], Self-Route [90] | Skip noise on easy queries | Saves calls on easy, adds on hard | Router miscalibration |
| Iterative / agentic | IRCoT [63], Search-R1 [69], DeepResearcher [72], [74] | Multi-hop and open-web research | Many LLM and search calls; about 15x tokens in [74] | Over-searching, loops, cost spikes |
| Graph index | GraphRAG [81], HippoRAG [83], LightRAG [82] | Global and multi-hop questions | High index cost (GraphRAG); cheap query (HippoRAG) | Extraction errors, stale graph |
| No retrieval (preload) | CAG [93], long-context studies [89][92] | No retriever errors | Huge prefill unless cached | Doesn't scale past context window |
| KV-level RAG serving | RAGCache [94], CacheBlend [95], TurboRAG [96], REFRAG [97] | Same quality, lower TTFT | Large TTFT reductions | Model or kernel changes required |
| Train the generator | RA-DIT [75], RAFT [76], ChatQA [77], RankRAG [78] | Distractor robustness, citation | Training cost; ties you to a model | Overfitting to the domain snapshot |

## Key takeaways for system designers

1. **Recall is the ceiling, and lexical recall is not obsolete.** DPR's 19-point top-20 gain
   over BM25 [4] started the dense era. BEIR [102] and LIMIT [34] explain why hybrid
   (lexical plus dense, or multi-vector) remains the safe default.
2. **Always cascade.** Cheap recall over millions of candidates, then a cross-encoder or LLM
   reranker over at most a few hundred [43][44][45]. Spend the latency budget where the
   LLM reads.
3. **Fix chunks before you fix models.** Contextualising chunks cut retrieval failures by
   up to 67% at a one-time cost of about $1 per million tokens [59]. That beats most
   retriever swaps on cost-effectiveness.
4. **Fewer, better passages.** Position bias [53], near-miss distractors [54] and the
   "more passages first help then hurt" curve [92] all argue for aggressive reranking,
   deduplication and edge placement over large top-k.
5. **Make retrieval conditional.** Routers and self-assessment [64][65][67][90] save compute
   on easy queries and spend it on hard ones. Calibrate the router on your own traffic.
6. **Agentic search is now trained, not prompted.** Search-R1 and its successors [69]-[73]
   show that RL against a search tool learns when and what to query. Budget for the token
   multiplier [74] and add hard caps on turns and tool calls.
7. **Pick graph RAG by query type.** It wins for global and multi-hop questions and often
   loses on single-hop detail [81][83][87]. Its index cost must be justified by those
   queries.
8. **Exploit RAG's block structure in serving.** Chunk-level KV reuse and compression
   [94]-[97] are where 2025-2026 latency gains come from. Preloading (CAG [93]) is right only
   for small, stable knowledge bases.
9. **Fine-tune for behaviour, retrieve for facts** [75][76][79][80].
10. **Evaluate in layers and re-baseline.** Separate retriever and generator metrics
    [110], add statistical bounds on judge scores [108], and reproduce methods under
    uniform settings before adopting them [123]. Treat every retrieved document as
    untrusted input [116][117].

## References

1. Chen, D., Fisch, A., Weston, J., Bordes, A. "Reading Wikipedia to Answer Open-Domain Questions" (DrQA). Facebook AI Research, ACL 2017. https://arxiv.org/abs/1704.00051
2. Lee, K., Chang, M.-W., Toutanova, K. "Latent Retrieval for Weakly Supervised Open Domain Question Answering" (ORQA). Google Research, ACL 2019. https://arxiv.org/abs/1906.00300
3. Guu, K., Lee, K., Tung, Z., Pasupat, P., Chang, M.-W. "REALM: Retrieval-Augmented Language Model Pre-Training." Google Research, ICML 2020. https://arxiv.org/abs/2002.08909
4. Karpukhin, V., et al. "Dense Passage Retrieval for Open-Domain Question Answering." Facebook AI, EMNLP 2020. https://arxiv.org/abs/2004.04906
5. Lewis, P., et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." Facebook AI / UCL / NYU, NeurIPS 2020. https://arxiv.org/abs/2005.11401
6. Izacard, G., Grave, E. "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering" (FiD). Facebook AI / Inria, EACL 2021. https://arxiv.org/abs/2007.01282
7. Izacard, G., Grave, E. "Distilling Knowledge from Reader to Retriever for Question Answering." ICLR 2021. https://arxiv.org/abs/2012.04584
8. Khandelwal, U., Levy, O., Jurafsky, D., Zettlemoyer, L., Lewis, M. "Generalization through Memorization: Nearest Neighbor Language Models" (kNN-LM). Stanford / Facebook AI, ICLR 2020. https://arxiv.org/abs/1911.00172
9. Borgeaud, S., et al. "Improving Language Models by Retrieving from Trillions of Tokens" (RETRO). DeepMind, ICML 2022. https://arxiv.org/abs/2112.04426
10. Izacard, G., et al. "Atlas: Few-shot Learning with Retrieval Augmented Language Models." Meta AI, JMLR 2023. https://arxiv.org/abs/2208.03299
11. Ram, O., et al. "In-Context Retrieval-Augmented Language Models." AI21 Labs, TACL 2023. https://arxiv.org/abs/2302.00083
12. Shi, W., et al. "REPLUG: Retrieval-Augmented Black-Box Language Models." UW / Meta, NAACL 2024. https://arxiv.org/abs/2301.12652
13. Nakano, R., et al. "WebGPT: Browser-assisted question-answering with human feedback." OpenAI, 2021. https://arxiv.org/abs/2112.09332
14. Khattab, O., Zaharia, M. "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT." Stanford, SIGIR 2020. https://arxiv.org/abs/2004.12832
15. Santhanam, K., Khattab, O., Saad-Falcon, J., Potts, C., Zaharia, M. "ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction." Stanford, NAACL 2022. https://arxiv.org/abs/2112.01488
16. Santhanam, K., et al. "PLAID: An Efficient Engine for Late Interaction Retrieval." Stanford, CIKM 2022. https://arxiv.org/abs/2205.09707
17. Xiong, L., et al. "Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval" (ANCE). Microsoft, ICLR 2021. https://arxiv.org/abs/2007.00808
18. Qu, Y., et al. "RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering." Baidu, NAACL 2021. https://arxiv.org/abs/2010.08191
19. Hofstätter, S., Lin, S.-C., Yang, J.-H., Lin, J., Hanbury, A. "Efficiently Teaching an Effective Dense Retriever with Balanced Topic Aware Sampling" (TAS-B). SIGIR 2021. https://arxiv.org/abs/2104.06967
20. Formal, T., Piwowarski, B., Clinchant, S. "SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking." Naver Labs Europe, SIGIR 2021. https://arxiv.org/abs/2107.05720
21. Izacard, G., et al. "Unsupervised Dense Information Retrieval with Contrastive Learning" (Contriever). Meta AI, TMLR 2022. https://arxiv.org/abs/2112.09118
22. Wang, L., et al. "Text Embeddings by Weakly-Supervised Contrastive Pre-training" (E5). Microsoft, 2022. https://arxiv.org/abs/2212.03533
23. Wang, L., et al. "Improving Text Embeddings with Large Language Models" (E5-Mistral). Microsoft, 2024. https://arxiv.org/abs/2401.00368
24. Li, Z., et al. "Towards General Text Embeddings with Multi-stage Contrastive Learning" (GTE). Alibaba, 2023. https://arxiv.org/abs/2308.03281
25. Xiao, S., et al. "C-Pack: Packed Resources For General Chinese Embeddings" (BGE). BAAI, 2023. https://arxiv.org/abs/2309.07597
26. Chen, J., et al. "BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation." BAAI, 2024. https://arxiv.org/abs/2402.03216
27. Kusupati, A., et al. "Matryoshka Representation Learning." NeurIPS 2022. https://arxiv.org/abs/2205.13147
28. Lee, J., et al. "Gecko: Versatile Text Embeddings Distilled from Large Language Models." Google DeepMind, 2024. https://arxiv.org/abs/2403.20327
29. Lee, C., et al. "NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models." NVIDIA, 2024. https://arxiv.org/abs/2405.17428
30. Zhang, Y., et al. "Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models." Alibaba Qwen, 2025. https://arxiv.org/abs/2506.05176
31. Lee, J., et al. "Gemini Embedding: Generalizable Embeddings from Gemini." Google, 2025. https://arxiv.org/abs/2503.07891
32. Weller, O., et al. "Promptriever: Instruction-Trained Retrievers Can Be Prompted Like Language Models." 2024. https://arxiv.org/abs/2409.11136
33. Shao, R., et al. "ReasonIR: Training Retrievers for Reasoning Tasks." Meta FAIR / UW, 2025. https://arxiv.org/abs/2504.20595
34. Weller, O., Boratko, M., Naim, I., Lee, J. "On the Theoretical Limitations of Embedding-Based Retrieval." Google DeepMind, 2025. https://arxiv.org/abs/2508.21038
35. Malkov, Y. A., Yashunin, D. A. "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs." IEEE TPAMI 2018. https://arxiv.org/abs/1603.09320
36. Johnson, J., Douze, M., Jégou, H. "Billion-scale similarity search with GPUs." Facebook AI Research, 2017. https://arxiv.org/abs/1702.08734
37. Douze, M., et al. "The Faiss library." Meta, 2024. https://arxiv.org/abs/2401.08281
38. Guo, R., et al. "Accelerating Large-Scale Inference with Anisotropic Vector Quantization" (ScaNN). Google Research, ICML 2020. https://arxiv.org/abs/1908.10396
39. Subramanya, S. J., et al. "DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node." Microsoft Research, NeurIPS 2019. https://harsha-simhadri.org/pubs/DiskANN19.pdf
40. Singh, A., et al. "FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming Similarity Search." Microsoft Research, 2021. https://arxiv.org/abs/2105.09613
41. Xu, Y., et al. "SPFresh: Incremental In-Place Update for Billion-Scale Vector Search." SOSP 2023. https://dl.acm.org/doi/10.1145/3600006.3613166
42. Patel, L., et al. "ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data." SIGMOD 2024. https://arxiv.org/abs/2403.04871
43. Nogueira, R., Cho, K. "Passage Re-ranking with BERT." NYU, 2019. https://arxiv.org/abs/1901.04085
44. Nogueira, R., Jiang, Z., Pradeep, R., Lin, J. "Document Ranking with a Pretrained Sequence-to-Sequence Model" (monoT5). Findings of EMNLP 2020. https://arxiv.org/abs/2003.06713
45. Sun, W., et al. "Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents" (RankGPT). EMNLP 2023. https://arxiv.org/abs/2304.09542
46. Pradeep, R., Sharifymoghaddam, S., Lin, J. "RankZephyr: Effective and Robust Zero-Shot Listwise Reranking is a Breeze!" University of Waterloo, 2023. https://arxiv.org/abs/2312.02724
47. Weller, O., et al. "Rank1: Test-Time Compute for Reranking in Information Retrieval." Johns Hopkins University, 2025. https://arxiv.org/abs/2502.18418
48. Gao, L., Ma, X., Lin, J., Callan, J. "Precise Zero-Shot Dense Retrieval without Relevance Labels" (HyDE). CMU / Waterloo, ACL 2023. https://arxiv.org/abs/2212.10496
49. Wang, L., Yang, N., Wei, F. "Query2doc: Query Expansion with Large Language Models." Microsoft, EMNLP 2023. https://arxiv.org/abs/2303.07678
50. Bonifacio, L., et al. "InPars: Data Augmentation for Information Retrieval using Large Language Models." 2022. https://arxiv.org/abs/2202.05144
51. Dai, Z., et al. "Promptagator: Few-shot Dense Retrieval From 8 Examples." Google Research, 2022. https://arxiv.org/abs/2209.11755
52. Wang, K., Thakur, N., Reimers, N., Gurevych, I. "GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation of Dense Retrieval." NAACL 2022. https://arxiv.org/abs/2112.07577
53. Liu, N. F., et al. "Lost in the Middle: How Language Models Use Long Contexts." Stanford, TACL 2024. https://arxiv.org/abs/2307.03172
54. Cuconasu, F., et al. "The Power of Noise: Redefining Retrieval for RAG Systems." SIGIR 2024. https://arxiv.org/abs/2401.14887
55. Xu, F., Shi, W., Choi, E. "RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation." ICLR 2024. https://arxiv.org/abs/2310.04408
56. Jiang, H., et al. "LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression." Microsoft, 2023. https://arxiv.org/abs/2310.06839
57. Sarthi, P., et al. "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval." Stanford, ICLR 2024. https://arxiv.org/abs/2401.18059
58. Günther, M., Mohr, I., Williams, D. J., Wang, B., Xiao, H. "Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models." Jina AI, 2024. https://arxiv.org/abs/2409.04701
59. Anthropic. "Introducing Contextual Retrieval." Anthropic, 2024. https://www.anthropic.com/news/contextual-retrieval
60. Qu, R., Bao, F., Tu, R. "Is Semantic Chunking Worth the Computational Cost?" Vectara, Findings of NAACL 2025. https://arxiv.org/abs/2410.13070
61. Yao, S., et al. "ReAct: Synergizing Reasoning and Acting in Language Models." Princeton / Google, ICLR 2023. https://arxiv.org/abs/2210.03629
62. Schick, T., et al. "Toolformer: Language Models Can Teach Themselves to Use Tools." Meta AI, NeurIPS 2023. https://arxiv.org/abs/2302.04761
63. Trivedi, H., Balasubramanian, N., Khot, T., Sabharwal, A. "Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions" (IRCoT). Stony Brook / AI2, ACL 2023. https://arxiv.org/abs/2212.10509
64. Jiang, Z., et al. "Active Retrieval Augmented Generation" (FLARE). CMU, EMNLP 2023. https://arxiv.org/abs/2305.06983
65. Asai, A., Wu, Z., Wang, Y., Sil, A., Hajishirzi, H. "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection." UW / AI2 / IBM, ICLR 2024. https://arxiv.org/abs/2310.11511
66. Yan, S.-Q., Gu, J.-C., Zhu, Y., Ling, Z.-H. "Corrective Retrieval Augmented Generation" (CRAG). 2024. https://arxiv.org/abs/2401.15884
67. Jeong, S., Baek, J., Cho, S., Hwang, S. J., Park, J. C. "Adaptive-RAG: Learning to Adapt Retrieval-Augmented LLMs through Question Complexity." KAIST, NAACL 2024. https://arxiv.org/abs/2403.14403
68. Li, X., et al. "Search-o1: Agentic Search-Enhanced Large Reasoning Models." Renmin University, EMNLP 2025. https://arxiv.org/abs/2501.05366
69. Jin, B., Zeng, H., Yue, Z., et al. "Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning." COLM 2025. https://arxiv.org/abs/2503.09516
70. Song, H., et al. "R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning." Renmin University, 2025. https://arxiv.org/abs/2503.05592
71. Chen, M., et al. "ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning." 2025. https://arxiv.org/abs/2503.19470
72. Zheng, Y., et al. "DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments." GAIR-NLP, 2025. https://arxiv.org/abs/2504.03160
73. Li, X., et al. "WebThinker: Empowering Large Reasoning Models with Deep Research Capability." NeurIPS 2025. https://arxiv.org/abs/2504.21776
74. Anthropic. "How we built our multi-agent research system." Anthropic Engineering, June 2025. https://www.anthropic.com/engineering/multi-agent-research-system
75. Lin, X. V., et al. "RA-DIT: Retrieval-Augmented Dual Instruction Tuning." Meta FAIR, ICLR 2024. https://arxiv.org/abs/2310.01352
76. Zhang, T., et al. "RAFT: Adapting Language Model to Domain Specific RAG." UC Berkeley, 2024. https://arxiv.org/abs/2403.10131
77. Liu, Z., et al. "ChatQA: Surpassing GPT-4 on Conversational QA and RAG." NVIDIA, NeurIPS 2024. https://arxiv.org/abs/2401.10225
78. Yu, Y., et al. "RankRAG: Unifying Context Ranking with Retrieval-Augmented Generation in LLMs." NVIDIA / Georgia Tech, NeurIPS 2024. https://arxiv.org/abs/2407.02485
79. Ovadia, O., Brief, M., Mishaeli, M., Elisha, O. "Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs." Microsoft, 2023. https://arxiv.org/abs/2312.05934
80. Gekhman, Z., et al. "Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?" Technion / Google Research, 2024. https://arxiv.org/abs/2405.05904
81. Edge, D., et al. "From Local to Global: A Graph RAG Approach to Query-Focused Summarization." Microsoft Research, 2024. https://arxiv.org/abs/2404.16130
82. Guo, Z., Xia, L., Yu, Y., Ao, T., Huang, C. "LightRAG: Simple and Fast Retrieval-Augmented Generation." HKU, 2024 (EMNLP 2025). https://arxiv.org/abs/2410.05779
83. Gutiérrez, B. J., Shu, Y., Gu, Y., Yasunaga, M., Su, Y. "HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models." Ohio State / Stanford, NeurIPS 2024. https://arxiv.org/abs/2405.14831
84. Gutiérrez, B. J., Shu, Y., Qi, W., Zhou, S., Su, Y. "From RAG to Memory: Non-Parametric Continual Learning for Large Language Models" (HippoRAG 2). ICML 2025. https://arxiv.org/abs/2502.14802
85. He, X., et al. "G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering." NeurIPS 2024. https://arxiv.org/abs/2402.07630
86. Liang, L., et al. "KAG: Boosting LLMs in Professional Domains via Knowledge Augmented Generation." Ant Group, 2024. https://arxiv.org/abs/2409.13731
87. Han, H., et al. "RAG vs. GraphRAG: A Systematic Evaluation and Key Insights." 2025. https://arxiv.org/abs/2502.11371
88. Xu, Z., et al. "Retrieval-Augmented Generation with Knowledge Graphs for Customer Service Question Answering." LinkedIn, SIGIR 2024. https://arxiv.org/abs/2404.17723
89. Xu, P., et al. "Retrieval meets Long Context Large Language Models." NVIDIA, ICLR 2024. https://arxiv.org/abs/2310.03025
90. Li, Z., et al. "Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study and Hybrid Approach" (Self-Route). Google DeepMind / University of Michigan, EMNLP 2024 Industry. https://arxiv.org/abs/2407.16833
91. Jiang, Z., Ma, X., Chen, W. "LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs." University of Waterloo, 2024. https://arxiv.org/abs/2406.15319
92. Jin, B., Yoon, J., Han, J., Arik, S. O. "Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG." Google Cloud AI / UIUC, 2024. https://arxiv.org/abs/2410.05983
93. Chan, B. J., et al. "Don't Do RAG: When Cache-Augmented Generation is All You Need for Knowledge Tasks." 2024 (short paper, 2025). https://arxiv.org/abs/2412.15605
94. Jin, C., et al. "RAGCache: Efficient Knowledge Caching for Retrieval-Augmented Generation." Peking University / ByteDance, 2024. https://arxiv.org/abs/2404.12457
95. Yao, J., et al. "CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion." University of Chicago, EuroSys 2025. https://arxiv.org/abs/2405.16444
96. Lu, S., et al. "TurboRAG: Accelerating Retrieval-Augmented Generation with Precomputed KV Caches for Chunked Text." 2024. https://arxiv.org/abs/2410.07590
97. Lin, X., et al. "REFRAG: Rethinking RAG based Decoding." Meta Superintelligence Labs, 2025. https://arxiv.org/abs/2509.01092
98. Faysse, M., et al. "ColPali: Efficient Document Retrieval with Vision Language Models." ILLUIN Technology / CentraleSupélec, ICLR 2025. https://arxiv.org/abs/2407.01449
99. Bajaj, P., et al. "MS MARCO: A Human Generated MAchine Reading COmprehension Dataset." Microsoft, 2016. https://arxiv.org/abs/1611.09268
100. Yang, Z., et al. "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering." EMNLP 2018. https://arxiv.org/abs/1809.09600
101. Petroni, F., et al. "KILT: a Benchmark for Knowledge Intensive Language Tasks." Facebook AI, NAACL 2021. https://arxiv.org/abs/2009.02252
102. Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., Gurevych, I. "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models." NeurIPS 2021 Datasets and Benchmarks. https://arxiv.org/abs/2104.08663
103. Muennighoff, N., Tazi, N., Magne, L., Reimers, N. "MTEB: Massive Text Embedding Benchmark." Hugging Face, EACL 2023. https://arxiv.org/abs/2210.07316
104. Enevoldsen, K., et al. "MMTEB: Massive Multilingual Text Embedding Benchmark." ICLR 2025. https://arxiv.org/abs/2502.13595
105. Su, H., Yen, H., Xia, M., et al. "BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval." ICLR 2025. https://arxiv.org/abs/2407.12883
106. Chen, J., Lin, H., Han, X., Sun, L. "Benchmarking Large Language Models in Retrieval-Augmented Generation" (RGB). AAAI 2024. https://arxiv.org/abs/2309.01431
107. Es, S., James, J., Espinosa-Anke, L., Schockaert, S. "RAGAS: Automated Evaluation of Retrieval Augmented Generation." EACL 2024 System Demonstrations. https://arxiv.org/abs/2309.15217
108. Saad-Falcon, J., Khattab, O., Potts, C., Zaharia, M. "ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems." Stanford, NAACL 2024. https://arxiv.org/abs/2311.09476
109. Gao, T., Yen, H., Yu, J., Chen, D. "Enabling Large Language Models to Generate Text with Citations" (ALCE). Princeton, EMNLP 2023. https://arxiv.org/abs/2305.14627
110. Ru, D., et al. "RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation." Amazon, NeurIPS 2024 Datasets and Benchmarks. https://arxiv.org/abs/2408.08067
111. Yang, X., et al. "CRAG: Comprehensive RAG Benchmark." Meta, NeurIPS 2024 Datasets and Benchmarks. https://arxiv.org/abs/2406.04744
112. Krishna, S., et al. "Fact, Fetch, and Reason: A Unified Evaluation of Retrieval-Augmented Generation" (FRAMES). Google, 2024. https://arxiv.org/abs/2409.12941
113. Tang, Y., Yang, Y. "MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries." COLM 2024. https://arxiv.org/abs/2401.15391
114. Pradeep, R., et al. "Initial Nugget Evaluation Results for the TREC 2024 RAG Track with the AutoNuggetizer Framework." University of Waterloo, 2024. https://arxiv.org/abs/2411.09607
115. Chen, Z., et al. "BrowseComp-Plus: A More Fair and Transparent Evaluation Benchmark of Deep-Research Agent." University of Waterloo et al., 2025. https://arxiv.org/abs/2508.06600
116. Greshake, K., Abdelnabi, S., Mishra, S., Endres, C., Holz, T., Fritz, M. "Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection." AISec 2023. https://arxiv.org/abs/2302.12173
117. Zou, W., Geng, R., Wang, B., Jia, J. "PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models." USENIX Security 2025. https://arxiv.org/abs/2402.07867
118. Morris, J. X., Kuleshov, V., Shmatikov, V., Rush, A. M. "Text Embeddings Reveal (Almost) As Much As Text." Cornell, EMNLP 2023. https://arxiv.org/abs/2310.06816
119. Gao, Y., et al. "Retrieval-Augmented Generation for Large Language Models: A Survey." Tongji / Fudan, 2023. https://arxiv.org/abs/2312.10997
120. Barnett, S., Kurniawan, S., Thudumu, S., Brannelly, Z., Abdelrazek, M. "Seven Failure Points When Engineering a Retrieval Augmented Generation System." Deakin University, CAIN 2024. https://arxiv.org/abs/2401.05856
121. Singh, A., Ehtesham, A., Kumar, S., Talaei Khoei, T. "Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG." 2025 (revised 2026). https://arxiv.org/abs/2501.09136
122. "SoK: Agentic Retrieval-Augmented Generation (RAG): Taxonomy, Architectures, Evaluation, and Research Directions." 2026. https://arxiv.org/abs/2603.07379
123. Jin, J., Zhu, Y., Dou, Z., et al. "FlashRAG: A Modular Toolkit for Efficient Retrieval-Augmented Generation Research." Renmin University, WWW 2025 Companion. https://arxiv.org/abs/2405.13576
124. Khattab, O., et al. "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines." Stanford, ICLR 2024. https://arxiv.org/abs/2310.03714
