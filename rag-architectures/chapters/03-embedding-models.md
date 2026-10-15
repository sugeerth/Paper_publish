# 03. Embedding Models for Retrieval: Commercial APIs, Open Families, Compression and Adaptation

## Overview

The embedding model sits on the hot path of almost every RAG system twice. At ingestion time it turns each chunk into a vector. At query time it turns the user's question into a vector in the same space. Whatever the embedder cannot tell apart, no vector database, ANN index or prompt can recover later. A reranker (Chapter 06) can reorder candidates, but it only sees what first-stage retrieval returned. The embedder also drives most of the storage bill. Vector count times dimensions times bytes per dimension decides whether an index fits in RAM on one node or needs a cluster.

This chapter covers the design space as it stands in late 2026.

1. **Commercial API families.** OpenAI `text-embedding-3`, Cohere Embed v3/v4/v5, Voyage AI (voyage-3 to voyage-4) and Google Gemini Embedding (001 and the multimodal Embedding 2).
2. **Open-weight families.** BAAI BGE (including BGE-M3), Microsoft E5 (including E5-Mistral), Alibaba GTE and Qwen3-Embedding, Nomic Embed, Jina Embeddings v3 to v5, NVIDIA NV-Embed and Nemotron, Google EmbeddingGemma, and Snowflake Arctic Embed 2.0.
3. **Cross-cutting techniques.** Instruction- and task-conditioned embeddings, LLM-backbone embedders, Matryoshka Representation Learning (MRL), int8 and binary quantization with rescoring, multi-vector and sparse outputs, context-aware chunk embeddings, and the privacy properties of vectors.
4. **Evaluation.** MTEB, MMTEB, the leakage-resistant RTEB and their blind spots, including the theoretical capacity limits of single-vector retrieval.
5. **Domain adaptation.** Contrastive fine-tuning, synthetic query generation, GPL pseudo-labelling, adapters and model merging.

Almost every design in this chapter instantiates the same asymmetric bi-encoder pattern:

```
 INGEST (offline, batch)                         QUERY (online, per request)
+-------------+   +-------------------------+    +-----------+   +----------------------+
| chunk text  |-->| [doc prefix/task tag]   |    | user query|-->| [query instruction]  |
| (+title,    |   | "search_document: ..."  |    +-----------+   | "Instruct: ...       |
|  metadata)  |   +------------+------------+                    |  Query: ..."         |
+-------------+                |                                 +----------+-----------+
                               v                                            v
                  +-------------------------+                  +-------------------------+
                  | encoder (BERT-style or  |   same weights   | encoder                 |
                  | LLM backbone)           |<================>|                         |
                  | pooling: mean/CLS/EOS/  |                  |                         |
                  | latent-attention        |                  |                         |
                  +------------+------------+                  +------------+------------+
                               v                                            v
                  +-------------------------+                  +-------------------------+
                  | truncate to d' (MRL)    |                  | truncate to d' (MRL)    |
                  | L2-normalise            |                  | L2-normalise            |
                  | quantise: fp32/int8/bin |                  | (often kept fp32)       |
                  +------------+------------+                  +------------+------------+
                               v                                            |
                  +-------------------------+   ANN search (cos/dot/hamming)|
                  | vector index            |<------------------------------+
                  +------------+------------+
                               v
                    top-k candidates --> optional fp32/int8 rescoring --> reranker --> LLM
```

The main design choices are which encoder to use, which prefix or instruction to attach on each side, which output dimension d' to keep, which numeric precision to store, and whether to adapt the encoder to the target domain.

## 1. Commercial API families

### 1.1 OpenAI text-embedding-3 (small, large)

**Problem.** `text-embedding-ada-002` (December 2022) gave one fixed 1536-dimensional vector at a single price point. Users with large corpora could not trade quality for storage [3].

**Design.** In January 2024 OpenAI released `text-embedding-3-small` (1536 dimensions by default) and `text-embedding-3-large` (3072 dimensions by default). Both accept up to 8,191 input tokens with the `cl100k_base` tokenizer [1][2][3]. The headline feature is the `dimensions` request parameter, which shortens the embedding at creation time "without the embedding losing its concept-representing properties." OpenAI's documentation reports that `text-embedding-3-large` cut to 256 dimensions still beats `ada-002` at 1536 dimensions on MTEB [1]. This is Matryoshka-style behaviour (Section 3.2), although OpenAI does not describe its training method. Outputs are unit-normalised, so dot product and cosine give the same ranking. If you truncate an embedding yourself instead of using `dimensions`, you must re-normalise it [1].

**Published numbers.** OpenAI's embeddings guide lists these MTEB averages: 62.3% for 3-small, 64.6% for 3-large and 61.0% for ada-002. At roughly 800 tokens per page it also lists approximate cost-efficiency of 62,500 pages per dollar for 3-small, 9,615 for 3-large and 12,500 for ada-002 [1]. Hugging Face's quantization write-up reports that `text-embedding-3-large` keeps about 93.1% of its performance at 12x dimensional compression [39].

**Long inputs.** The OpenAI cookbook gives three strategies for text over the 8,191-token limit: truncate, embed chunks separately, or take a token-count-weighted average of chunk embeddings and re-normalise it [2]. For RAG, embedding chunks separately is nearly always the right choice. Averaged embeddings blur exactly the local detail that retrieval needs.

**Failure modes and lessons.** These models take no task instructions and have no query/document asymmetry flag, so the same text maps to the same vector whatever its role. The 8K context is short next to the 32K to 128K windows of 2025 to 2026 competitors (Sections 1.2 to 1.4). The pricing table also shows a non-obvious point: the large model is roughly 6.5x more expensive per page than the small model, and its quality gap on MTEB is only about 2.3 points [1]. Many teams start with 3-small and spend the savings on a reranker.

### 1.2 Cohere Embed (v3, v4, v5)

**Problem.** Enterprise corpora are multilingual, contain images and scanned pages, and are large enough that float32 storage dominates cost.

**Design.** Cohere's API puts role information in an `input_type` field (`search_query`, `search_document`, `classification`, `clustering`, `image`). It can return several numeric encodings in one call through `embedding_types`: `float`, `int8`, `uint8`, `binary` and `ubinary`. Binary types are bit-packed, so a 1024-dimensional binary vector is 128 bytes [5]. The v3 generation (November 2023) ships as English and multilingual models at 1024 dimensions, plus "light" variants at 384 dimensions, all with a 512-token context [4][52].

`embed-v4.0` was a large step up. According to Cohere's model table it takes text, images, and mixed text-and-image inputs such as PDF pages, has a 128K-token context, and offers Matryoshka output sizes of 256, 512, 1024 and 1536 (default) [4]. As of 2026 the docs list `embed-v5.0-pro` and a lower-latency `embed-v5.0-fast` with the same 128K context and mixed-modality input, and output dimensions of 256 to 2048 (default 2048) [4]. With `embed-v5.0-pro`, a single call can embed up to 96 images. Large images are downsampled above about 2.46 megapixels and small ones are upsampled below 56x56 pixels [5].

**Compression behaviour.** Independent measurement by Hugging Face on MTEB retrieval (15 datasets, rescore multiplier 4) found that `Cohere-embed-english-v3.0` kept 100% of its NDCG@10 at int8 (55.0 vs 55.0) and 94.6% at binary (52.3 vs 55.0) [39]. That is among the best binary retention in that comparison. Models trained with quantization in mind keep much more quality than models quantized after the fact (Section 3.3).

**Lessons.** Cohere was early to treat the storage format as a first-class API output rather than a client-side afterthought. Its 128K-context multimodal models also let a whole slide deck or PDF page be one embedding unit, which affects chunking (Chapter 02). Long context still does not remove the dilution problem: one vector for 100K tokens cannot hold fine-grained facts.

### 1.3 Voyage AI (voyage-3, voyage-3-large, voyage-4 family)

**Problem.** General-purpose embedders underperform on specialised domains (law, finance, code), and storage cost limits index size.

**Design history.** Voyage released domain models early: `voyage-code-2` (January 2024), `voyage-law-2` (April 2024), `voyage-finance-2` (May 2024) and `voyage-multilingual-2` (June 2024). The 2024 general models `voyage-3` and `voyage-3-lite` (September 2024) moved to a 32K-token context [9]. `voyage-code-3` followed in December 2024 [9]. `voyage-3-large` (7 January 2025) brought MRL (2048, 1024, 512 and 256 dimensions) and quantization-aware training (float, signed and unsigned int8, binary) [6], and `voyage-3.5` later in 2025 carried these options to the mid-size tier [9]. In February 2025 MongoDB acquired Voyage AI [61]. The models remain available through Voyage's API, and MTEB now lists them under the `mongodb/` namespace [9].

**Published numbers (vendor-reported).** Across 100 datasets in eight domains (technical documentation, code, law, finance, web reviews, multilingual, long documents, conversations), Voyage reports that voyage-3-large beats OpenAI-v3-large by an average of 9.74% and Cohere-v3-English by 20.71% [6]. Its storage trade-off claims are the most aggressive published by any vendor [6]:

- int8 at 1024 dimensions is 0.31% below float at 2048 dimensions, with 8x less storage.
- int8 at 512 dimensions beats OpenAI-v3-large (float, 3072 dimensions) by 8.56% at 1/24 of the storage.
- binary at 512 dimensions beats OpenAI-v3-large by 1.16% at 1/200 of the storage.
- Binary retrieval followed by rescoring gains up to 5.84% over plain binary retrieval.

**voyage-4 (January 2026).** The voyage-4 family (`voyage-4-large`, `voyage-4`, `voyage-4-lite` and the open-weight `voyage-4-nano`) keeps a 32K context, 256/512/1024/2048 MRL dimensions and float/int8/uint8/binary/ubinary outputs [7][9]. Its notable architectural feature is a **shared embedding space**. Google Cloud's deployment notebook states that embeddings from any Voyage 4 model are interchangeable. It recommends encoding documents with `voyage-4-large` and queries with the cheaper, lower-latency `voyage-4-lite` [7]. `voyage-4-nano` runs locally through sentence-transformers. It returns unit-norm Matryoshka embeddings at every dimension, plus float and binary dtypes. Local int8 is not supported because, in the SDK's words, it needs "fixed calibration ranges we don't ship locally yet" [8].

```
 shared-space asymmetric deployment (voyage-4 style)
 docs (once, offline) --> voyage-4-large  --+
                                            +--> same vector space --> index
 queries (per request) --> voyage-4-lite ---+        (MRL d', int8/binary)
```

**Lessons.** A shared space across model sizes separates ingestion quality from query latency. You can upgrade the query model later without re-indexing. This is the most significant operational change of 2026, because re-embedding a corpus has been the main cost of upgrading an embedder. Treat the percentage claims above as vendor benchmarks on Voyage-chosen datasets.

### 1.4 Google Gemini Embedding (gemini-embedding-001, Gemini Embedding 2)

**Problem.** Pull the multilingual and code ability of a frontier LLM into a general-purpose embedding model.

**Architecture and training (gemini-embedding-001).** The technical report by Lee et al. (arXiv:2503.07891) describes a model initialised from Gemini LLM parameters [10]. As summarised in secondary reading notes of the paper [10], the encoder output is mean-pooled and linearly projected to 3072 dimensions, and training uses a noise-contrastive (NCE) loss with MRL across several output sizes. Training has three stages:

1. **Pre-finetuning** on a large set of noisy web (query, passage) pairs with large batches.
2. **Finetuning** on a mixture of retrieval, classification, clustering, multilingual and code tasks. Each batch draws from one task dataset, and hard negatives are included.
3. **Model soup.** Checkpoints from several fine-tuning runs are averaged.

Gemini itself is used to improve the data: it generates synthetic examples, filters out bad training pairs, and helps mine hard negatives [10]. The abstract reports that the model clearly outperforms prior state of the art on MMTEB (100+ tasks, 250+ languages) and ranks first on multilingual, English and code benchmarks [10].

**API surface.** `gemini-embedding-001` takes a `task_type` (`RETRIEVAL_QUERY`, `RETRIEVAL_DOCUMENT`, `SEMANTIC_SIMILARITY`, `CLASSIFICATION`, `CLUSTERING`, and others) and an `output_dimensionality` parameter that truncates the 3072-dimensional default [11]. MTEB metadata dates its general availability to 14 July 2025 [12].

**Gemini Embedding 2 (2026).** `gemini-embedding-2` is described as the first multimodal embedding model in the Gemini API. It embeds text, images, audio, video and PDFs (processing "the visual and text content of each page") with up to 8,192 input tokens [11]. MTEB metadata lists a preview on 10 March 2026, a stable release on 22 April 2026, and supported dimensions of 768, 1536 and 3072 [12]. One API change matters for RAG code: `task_type` is **not** supported on Gemini Embedding 2 [11]. Task conditioning moves into the text itself. MTEB's integration formats documents as `title: ... | text: ...` and queries with prefixes such as `task: search result | query: ...` [12].

**Lessons.** Applications migrating from 001 to 2 must rewrite their query and document formatting. Dropping the `task_type` flag without adding the textual prefixes silently loses the query/document asymmetry.

## 2. Open-weight families

### 2.1 BAAI BGE: from C-Pack to BGE-M3 and LLM embedders

**BGE v1/v1.5 (2023).** The C-Pack work (arXiv:2309.07597) produced `bge-large-en-v1.5` and its siblings: 335M parameters, 1024 dimensions, 512 tokens [16][53]. When released they ranked first on MTEB and C-MTEB according to BAAI [14]. Queries take the instruction `Represent this sentence for searching relevant passages: ` and passages take none. The v1.5 release fixed a similarity-distribution problem in v1 and improved retrieval when no instruction is given, so the query instruction became optional rather than required [14].

**BGE-M3 (2024).** BGE-M3 (arXiv:2402.03216) is a 568M-parameter, 1024-dimensional model with an 8,192-token context and 100+ languages. One forward pass gives three outputs: a dense vector, a lexical sparse vector (weights only for tokens present in the text), and ColBERT-style multi-vectors [14][15]. BAAI describes it as the first embedding model to support all three retrieval methods [14].

```
                        +--> [CLS] -> dense vector (1024)      --- s_dense
 text -> XLM-R-style    |
         encoder (8192) +--> per-token weights -> sparse lexical --- s_sparse
                        |
                        +--> per-token vectors -> multi-vector  --- s_colbert
 hybrid score = w1*s_dense + w2*s_sparse + w3*s_colbert   (example weights 0.4/0.2/0.4)
```

Training uses **self-knowledge distillation**: the combined score of the three heads serves as a teacher signal for each single head, which mainly helps the sparse and multi-vector heads [15]. **MCLS** is a training-free trick that improves long-document embeddings [15]. BAAI also released MLDR, a long-document retrieval set covering 13 languages [15]. The project notes that BM25 stays a strong baseline for long documents [15].

**LLM embedders.** `bge-en-icl` (7.1B, 4096 dimensions, 32K tokens; arXiv:2409.15700) accepts few-shot query-response examples in context, so users can steer the embedding with demonstrations rather than just an instruction. `bge-multilingual-gemma2` (9.2B, 3584 dimensions) is built on Gemma-2-9B and reported state-of-the-art results on MIRACL, MTEB-fr and MTEB-pl at release [14][53].

**Lessons.** BGE-M3 is the reference design for hybrid retrieval from one model. You get BM25-like lexical matching and late interaction without running separate systems. Its multi-vector output has the usual storage cost of one vector per token (Chapter 05).

### 2.2 Microsoft E5: weak supervision at scale, then LLMs and synthetic data

**E5 (2022).** "Text Embeddings by Weakly-Supervised Contrastive Pre-training" (arXiv:2212.03533) built **CCPairs** from semi-structured web data: Reddit post-comment pairs, Stack Exchange question-answer pairs, Wikipedia section-passage pairs, paper title-abstract pairs, and Common Crawl and news title-passage pairs [18]. Heuristics produced about 1.3B pairs. A **consistency filter** then trained a model on them and kept only pairs whose positive ranked in the model's top 2 among 1M random passages, leaving about 270M pairs [18]. Pre-training used InfoNCE with in-batch negatives at a batch size of 32,768. Fine-tuning used MS MARCO, NQ and NLI with mined hard negatives and cross-encoder distillation. The `query:` / `passage:` prefixes make the shared encoder asymmetric [18]. The paper reports E5 as the first unsupervised model to beat BM25 on zero-shot BEIR (42.9 vs about 41.7 nDCG@10 for the base model). It also reports that larger batches (1K to 32K) and filtering (about 6 points) both help clearly [18].

**Multilingual E5 (2024).** The mE5 family (arXiv:2402.05672) includes `multilingual-e5-large-instruct` (560M, 1024 dimensions, 514 tokens) [20][21]. The MMTEB paper later found it was "the best-performing publicly available model" on its massively multilingual benchmark, beating multi-billion-parameter LLM embedders [41].

**E5-Mistral (2024).** "Improving Text Embeddings with Large Language Models" (arXiv:2401.00368) is the template for the LLM-embedder era [19]:

- **Synthetic data.** GPT-4 (with about 25% GPT-3.5-Turbo) generated about 500K examples across about 150K unique task instructions in 93 languages. A two-step prompt first brainstorms tasks, then writes a query, positive and hard negative for each. The paper reports about 180M tokens consumed [19].
- **Model.** Mistral-7B with LoRA rank 16 (about 42M trainable parameters), last-token ([EOS]) pooling, and the query template `Instruct: {task}\nQuery: {query}`. Documents get no prefix, so the index does not depend on the instruction [19].
- **Training.** One epoch (under 1K steps), batch size 2048, InfoNCE at temperature 0.02, about 18 hours on 32 V100s [19].
- **Results.** MTEB average 66.6 and BEIR 56.9 nDCG@10, against 55.0 to 55.6 for the commercial embedders it compared [19][21].
- **Finding.** Contrastive pre-training added little for Mistral-7B but helped XLM-R-large by 8.2 retrieval points. The authors attribute this to the representations LLMs already learn in autoregressive pre-training [19].
- **Costs the authors acknowledge.** 7B inference cost and 4096-dimensional storage [19].

### 2.3 Alibaba GTE and Qwen3-Embedding

**GTE (2023-2025).** "Towards General Text Embeddings with Multi-stage Contrastive Learning" (arXiv:2308.03281) covers the GTE line [22]. It includes BERT-scale models, long-context `gte-base-en-v1.5` and `gte-multilingual-base` (8,192 tokens; mGTE, arXiv:2407.19669), `gte-modernbert-base`, and LLM-based `gte-Qwen2-7B-instruct` (7B, 3584 dimensions, 32K tokens) and `gte-Qwen2-1.5B-instruct` [23][24]. The Qwen-based models use the E5-Mistral-style `Instruct: ...\nQuery: ` template on queries only [24].

**Qwen3-Embedding (June 2025).** The technical report (arXiv:2506.05176) calls the series "a significant advancement over its predecessor, the GTE-Qwen series" [25]. It is the clearest public recipe for a state-of-the-art open LLM embedder.

```
 Qwen3 LLM (0.6B/4B/8B, causal attention)
          |
          v
 Stage 1: weakly-supervised pre-training on ~150M SYNTHETIC pairs
          (generated by Qwen3-32B: persona from Persona Hub x query type x
           difficulty x language, documents from Qwen3 pre-training corpus)
          |
          v
 Stage 2: supervised fine-tuning on ~7M labelled pairs + ~12M filtered
          synthetic pairs (keep if cosine sim > 0.7); InfoNCE with hard and
          in-batch negatives, false-negative masking
          |
          v
 Stage 3: model merging of SFT checkpoints (slerp)
          |
          v
 embedding = hidden state at final [EOS] token, MRL-capable, 32K context
```

- **Sizes.** 0.6B (28 layers, 1024 dimensions), 4B (36 layers, 2560 dimensions) and 8B (36 layers, 4096 dimensions), all with 32K context, MRL and instruction awareness [25][26].
- **Instructions.** Queries are formatted `Instruct: {task}\nQuery:{query}` and documents take no instruction. The repository says instructions typically add 1% to 5% over none and recommends writing them in English [26].
- **Scores.** MMTEB multilingual / MTEB English v2 / C-MTEB: 64.33 / 70.70 / 66.33 for 0.6B, 69.45 / 74.60 / 72.27 for 4B, and 70.58 / 75.22 / 73.84 for 8B. The 8B model ranked first on the MTEB multilingual leaderboard on 5 June 2025 and scores 80.68 on MTEB Code [25][26].
- **Ablations (0.6B).** Without model merging, MMTEB drops from 64.33 to 62.56. Without the synthetic pre-training stage it falls to 61.21. Training on synthetic data alone gives 58.49 [25].
- **Paired rerankers.** Qwen3-Reranker comes in 0.6B, 4B and 8B. Relevance is read from the probability of a "yes" versus "no" token, and the rerankers were evaluated on the top 100 candidates from Qwen3-Embedding-0.6B [25][26].
- **License.** Apache 2.0 [25].

**Lessons.** Two levers stand out in the ablation table: a large LLM-synthesised weak-supervision stage and checkpoint merging. Both are cheap compared with collecting human labels. They are also exactly the two components that closed proprietary recipes (Gemini's model soup and synthetic data) share [10][25].

### 2.4 Nomic Embed: reproducibility and MoE

**nomic-embed-text v1/v1.5 (2024).** The report (arXiv:2402.01613) presents a "reproducible long context text embedder" [27]. The model has 137M parameters, 768 dimensions and an 8,192-token context [30]. Weights, training code (the `contrastors` library) and training data are all published. The contrastive stage uses about 200M examples [29]. Inputs carry task prefixes: `search_query: `, `search_document: `, `classification: `, `clustering: ` [29][30]. v1.5 added MRL. Hugging Face reports it keeps 95.8% of performance at 3x compression and 90% at 6x [39], and notes that, unlike most Matryoshka models, it needs an explicit layer norm before truncation [38]. In Hugging Face's quantization tests it kept only 87.7% under binary quantization [39]. It is a useful example of a model that tolerates truncation better than binarisation.

**nomic-embed-text-v2-moe (2025).** "Training Sparse Mixture of Experts Text Embedding Models" (arXiv:2502.07972) applies MoE to an embedder (8 experts, top-2 routing): 475M total parameters, of which Nomic's model card says 305M are active at inference, and 768 dimensions [28][63]. (MTEB's metadata records a lower active count of about 142M, which appears to exclude embedding parameters [30].) Its max length is 512 tokens [30][63]. The design point is multilingual quality at small-model inference cost. Nomic also released `nomic-embed-code` (7B, 3584 dimensions, 32K tokens), trained on the CoRNStack data (arXiv:2412.01007) [30].

### 2.5 Jina Embeddings v3, v4 and v5: task adapters and distillation

- **v2 (2023).** One of the first open 8,192-token English embedders (arXiv:2310.19923), at 137M and 33M parameters [34].
- **v3 (September 2024).** jina-embeddings-v3 (arXiv:2409.10173) is "a novel text embedding model with 570 million parameters" with an 8,192-token context [31]. Its main idea is **task-specific LoRA adapters** (`retrieval.query`, `retrieval.passage`, `separation`, `classification`, `text-matching`) on a shared backbone. The task changes the weights, not only the prompt. MRL is built into training [31][34]. The abstract reports that it beats OpenAI and Cohere proprietary embeddings on English MTEB tasks and multilingual-e5-large-instruct on all multilingual tasks [31].
- **v4 (June 2025).** jina-embeddings-v4 (arXiv:2506.18902) is a 3.8B-parameter multimodal model (about 3.93B by MTEB's count, 2048 dimensions, 32K tokens). It produces both single-vector and multi-vector (late-interaction) embeddings for text and images, with LoRA adapters for retrieval, text matching and code. The authors also released Jina-VDR, a benchmark for visually rich documents [32][34].
- **v5 (February to April 2026).** jina-embeddings-v5-text (arXiv:2602.15547) introduces **task-targeted embedding distillation**, which combines distillation from a larger teacher with a task-specific contrastive loss. The authors report this beats either method alone for small models [33]. `v5-text-small` (596M, 32K tokens, MRL from 1024 down to 32 dimensions) and `v5-text-nano` (212M, 8K tokens) are reported to match or beat the state of the art for their size. They are also reported to stay robust under truncation and binary quantization [33][34]. `v5-omni` variants (April 2026) add image, audio and video [34].
- **Ownership.** Elastic completed its acquisition of Jina AI on 9 October 2025, so the v5 generation was released under Elastic [62].

**Lessons.** Jina shows the trade-off between adapter-based and prompt-based task conditioning. LoRA adapters give more capacity per task, but the serving stack must route each request to the right adapter. For RAG this means `retrieval.query` at query time and `retrieval.passage` at index time. Mixing them up is a silent failure mode.

### 2.6 NVIDIA NV-Embed and Nemotron

NV-Embed (arXiv:2405.17428) introduced three ideas for turning a decoder LLM (Mistral-7B) into an embedder [35]:

- a **latent-attention pooling layer**, which outperformed both mean pooling and last-[EOS] pooling;
- **removing the causal attention mask** during contrastive training;
- **two-stage contrastive instruction tuning**: retrieval data with in-batch and curated hard negatives first, then non-retrieval tasks blended in, which also improved retrieval.

It reached 69.32 on MTEB (56 tasks) and 59.36 on the 15 BEIR retrieval tasks, ranking first on 24 May 2024 using only public data [35]. The NVIDIA line continued with `llama-embed-nemotron-8b` (October 2025, arXiv:2511.07025, aimed at multilingual and cross-lingual retrieval) and the `Nemotron-3-Embed` 1B and 8B models (July 2026; 2048 and 4096 dimensions, 32K tokens) [36].

### 2.7 Small and on-device embedders: EmbeddingGemma and static embeddings

**EmbeddingGemma.** Google DeepMind's EmbeddingGemma is about 308M parameters [13]. It modifies the Gemma 3 backbone to use bidirectional attention, then applies mean pooling and two dense layers to produce 768-dimensional outputs. MRL lets those be truncated to 512, 256 or 128. Context is 2K tokens, it covers 100+ languages, and quantized it runs in under 200 MB of RAM. Training used about 320B tokens [13]. It is prompt-conditioned: queries take `task: search result | query: ` and documents take `title: none | text: `. This is the same textual format that Gemini Embedding 2 uses in place of `task_type` (Section 1.4) [12][13]. At launch it was the highest-ranking text-only multilingual model under 500M parameters on MTEB [13]. The launch post's fine-tuning case study is the clearest public example of domain adaptation paying off. On the MIRIAD medical retrieval data, 100K pairs with CachedMultipleNegativesRankingLoss (about 5.5 hours on one RTX 3090) raised NDCG@10 from 0.8340 to 0.8862. That beat the general-purpose Qwen3-Embedding-0.6B (0.8493), a model about twice as large [13].

**Static embeddings.** At the extreme low end, Sentence Transformers' static models (`static-retrieval-mrl-en-v1`, `static-similarity-mrl-multilingual-v1`) drop attention entirely and use per-token static vectors. They run 100x to 400x faster on CPU, and the English retrieval model reaches about 87.4% of all-mpnet-base-v2's NanoBEIR score [50]. They suit first-pass filtering on edge devices or very large re-embedding jobs.

### 2.8 Snowflake Arctic Embed 2.0: multilingual without an English penalty

**Problem.** Multilingual embedders have usually traded away English retrieval quality, which pushes teams to run two models.

**Design.** Arctic-Embed 2.0 (Yu, Merrick, Nuti and Campos, arXiv:2412.04506, December 2024) comes in two sizes [54]. The medium model is built on GTE-multilingual and has 305M parameters, 113M of them non-embedding. The large model is built on a long-context XLM-R-Large and has 568M parameters, 303M of them non-embedding. Both handle 8,192 tokens, are trained with MRL so that vectors can be cut to 256 dimensions, and are released under Apache 2.0 [54]. The paper reports strong English MTEB-Retrieval scores alongside competitive multilingual results (CLEF, MIRACL) [54].

**Evidence outside the vendor.** On Hugging Face's MIRIAD medical retrieval comparison, `snowflake-arctic-embed-m-v2.0` scored 0.8467 NDCG@10 and the large model 0.8433. Both were well ahead of BGE and mE5 models of similar size, and close to Qwen3-Embedding-0.6B (0.8493) [13].

**Lessons.** Arctic 2.0 is a strong Apache-licensed default at encoder scale. It also shows that "medium" encoders with large vocabularies spend most of their parameters on the embedding table, which matters when you compare parameter counts across families.

## 3. Cross-cutting techniques

### 3.1 Instruction- and task-conditioned embeddings

A plain bi-encoder maps a string to one vector regardless of intent. In practice, retrieval is asymmetric: short questions are matched to long passages. Retrieval also differs from clustering or duplicate detection on the same text. Four mechanisms have emerged:

| Mechanism | Examples | Where conditioning lives | Index impact |
|---|---|---|---|
| Fixed role prefixes | E5 `query:`/`passage:`, Nomic `search_query:` | Text prefix | Doc prefix fixed at index time |
| API role flag | Cohere `input_type`, Voyage `input_type`, Gemini-001 `task_type` | Server-side prompt | Must match at index and query time |
| Free-form query instruction | E5-Mistral, GTE-Qwen2, Qwen3, NV-Embed, BGE v1.5 | `Instruct: ...\nQuery: ...` on queries only | None: docs carry no instruction |
| Task LoRA adapter | Jina v3/v4/v5 | Model weights | Adapter must match at index time |
| In-context examples | bge-en-icl | Few-shot demos in query | None on doc side |

Sources: [5][7][11][14][17][19][24][26][29][31][34].

The free-form, query-only design dominates among LLM-based embedders because the document index stays independent of the instruction. One corpus can serve many query intents ("find a counter-argument", "find the code that implements this") by changing only the query instruction [19][26]. Qwen reports instructions are worth 1% to 5% [26]. Common failure modes:

- forgetting the query prefix entirely;
- putting the instruction on documents for a model trained without it;
- for Gemini Embedding 2, relying on a `task_type` flag the model no longer honours [11].

### 3.2 Matryoshka Representation Learning (MRL)

**Mechanism.** MRL (Kusupati et al., NeurIPS 2022, arXiv:2205.13147) trains one model so that every prefix of the embedding is itself a usable embedding [37]. The training loss is the sum (optionally weighted) of the base loss computed on the full vector and on several truncations, for example 768, 512, 256, 128 and 64 dimensions. This adds no notable training overhead [38]. At inference you keep the first d' values and re-normalise [1][38]. Encoding cost does not change, but storage, memory bandwidth and distance computation all scale with d'.

**Evidence.**
- The original paper reports embeddings up to 14x smaller at the same ImageNet-1K accuracy, and up to 14x real-world speed-ups in large-scale retrieval [37].
- Hugging Face's STSBenchmark example keeps 98.37% of performance at 8.3% of the embedding size, against 96.46% for a standard model truncated the same way [38].
- `text-embedding-3-large` keeps about 93.1% at 12x compression, and nomic v1.5 keeps 95.8% at 3x [39].

**Extensions.** Sentence Transformers ships `MatryoshkaLoss`, `Matryoshka2dLoss`, and `AdaptiveLayerLoss`, which trains models that still work with layers removed for faster inference [46].

**Adoption.** By 2026 MRL is effectively standard. OpenAI (`dimensions`), Cohere v4/v5, Voyage 3-large/4, Gemini, Qwen3, Jina v3 to v5, Nomic v1.5, EmbeddingGemma and the static models all expose it [1][4][6][11][13][26][31][34][50]. Jina v5 goes as low as 32 dimensions [34].

**Design pattern: funnel search.** Search a coarse d' = 256 index, then rescore the top candidates with the full-dimensional vector. This is the dimensional version of the rescoring pattern measured for quantization in Section 3.3, and the two can be combined.

### 3.3 Scalar (int8) and binary quantization

**Mechanics.** Hugging Face's "Embedding Quantization" post by Shakir, Aarsen and Lee [39] is the standard reference.

- **Binary.** Each float becomes one bit (threshold at 0), a 32x reduction: 1024 float32 dimensions take 4,096 bytes, and the binary form takes 128 bytes. Search uses Hamming distance, which reduces to XOR plus popcount.
- **Scalar int8.** Each dimension is mapped onto 256 buckets using per-dimension min/max values from a calibration set, a 4x reduction. The calibration set matters, and ranges estimated from more embeddings are more stable [39]. This is also why Voyage's local SDK will not emit int8 without shipped calibration ranges [8].

**Measured trade-offs** (mxbai-embed-large-v1, MTEB retrieval, CPU exact search) [39]:

| Precision | Bytes per 1024-dim vector | Search speed-up vs fp32 | Retained NDCG@10 |
|---|---|---|---|
| float32 | 4,096 | 1x | 100% |
| int8 (no rescoring) | 1,024 | ~3.66x mean (2.99x to 4.8x) | ~97% |
| int8 + rescoring | 1,024 | ~3.66x | ~99.3% |
| binary (no rescoring) | 128 | ~24.76x mean (15x to 45.8x) | ~92.5% |
| binary + rescoring | 128 | ~24.76x | ~96.45% |

Retention depends heavily on the model. Under binary quantization with rescoring, Cohere-v3 kept 94.6%, nomic-v1.5 87.7% and e5-base-v2 only 74.8%. Strangely, all-MiniLM-L6-v2 kept more under binary (93.8%) than under int8 (90.8%) [39]. The lesson: **measure on your own data and model.** Vendors that use quantization-aware training (Voyage-3-large and later, Cohere v3 and later) degrade much more gracefully [6][39].

**Cost at scale.** At $3.8 per GB-month of RAM, 250M 1024-dimensional vectors cost about $3,623/month as float32, $905 as int8 and $113 as binary [39]. The Hugging Face demo over 41M Wikipedia passages illustrates the canonical three-tier design:

```
 query --embed fp32--> q
   |
   +--> binary(q) --Hamming top-40--> [binary index: ~5.2 GB, in RAM]
                                         |
                                         v
            load int8 vectors of the 40 candidates  [int8 store: ~47.5 GB, on disk]
                                         |
                       rescore: dot(q_fp32, doc_int8) --> top-10 --> reranker/LLM
 total ~5.2 GB RAM + ~52 GB disk  vs  ~200 GB RAM + ~200 GB disk for pure fp32
```

Rescoring uses the full-precision *query* against quantised *documents* (asymmetric distance), which recovers most of the lost accuracy. Voyage's measurements point the same way: binary search plus rescoring gains up to 5.84% over plain binary [6].

**Combining MRL and quantization.** The two compressions multiply. Voyage's binary at 512 dimensions against OpenAI float at 3072 dimensions is a 192x size ratio (6x from dimensions times 32x from precision), which matches the "1/200 of storage" claim [6]. Hugging Face also suggests, speculatively, that shrinking MRL embeddings from 1024 to 128 dimensions (usually about a 2% loss) and then quantizing could speed retrieval up by up to 32x for about a 3% quality loss, or up to 256x for about 10% [39].

### 3.4 Pooling and attention choices in LLM embedders

Decoder LLMs use causal attention, so only the final token has seen the whole input. Designs differ in how they handle this:

- **Last-token / [EOS] pooling with causal attention:** E5-Mistral, Qwen3-Embedding [19][25].
- **Removing the causal mask and adding latent-attention pooling:** NV-Embed [35].
- **Bidirectional attention with mean pooling:** EmbeddingGemma, and Gemini Embedding per secondary summaries [10][13].

There is no consensus winner. In practice, use the pooling the model was trained with. Generic wrappers that default to mean pooling for a last-token model produce badly degraded embeddings.

### 3.5 Context-aware chunk embeddings: late chunking, contextual document embeddings, voyage-context-3

A chunk embedded on its own loses whatever its neighbours said ("the city" no longer refers to "Berlin"). Three designs from 2024 to 2025 move that context into the embedder instead of into chunk text (compare contextual chunk headers in Chapter 02):

- **Late chunking** (Günther et al., Jina AI, arXiv:2409.04701) runs a long-context encoder over the whole document, or as much of it as fits, and only then mean-pools token vectors over each chunk's span. Every chunk vector is conditioned on the surrounding text, at the cost of a single forward pass [55]. It needs a mean-pooling, long-context model and no retraining. Jina's repository reports nDCG@10 gains on several BEIR tasks, with the largest effect on chunks that refer to an entity only by anaphora ("its", "the city") [55].
- **Contextual Document Embeddings (CDE)** (Morris and Rush, ICLR 2025, arXiv:2410.02525) condition on the *corpus* rather than the document. Training uses "contextual batches" of clustered neighbouring documents, which act as hard in-batch negatives. A two-stage architecture first embeds a sample of corpus documents and feeds those embeddings as extra input tokens when encoding each document and query. This adapts the embedder to an unseen domain at index time without fine-tuning [56].
- **voyage-context-3** (Voyage AI / MongoDB, July 2025) is a commercial model trained for the job. It takes all of a document's chunks in one call and returns per-chunk vectors that encode document-level context. It offers 256 to 2048 dimensions, int8 and binary outputs and a 32K context. Voyage reports gains of about 14% (chunk-level) and 12.6% (document-level) over OpenAI-v3-large on its own suites [57].

**Trade-off.** Re-embedding one chunk now means re-encoding its document, and a corpus-conditioned model (CDE) must be re-indexed when the corpus distribution shifts. In return, these designs remove most of the need for LLM-written chunk context, which is expensive at ingestion.

### 3.6 Embeddings are not anonymised data: inversion and cross-model translation

It is easy to treat a vector store as safer than the source text. It is not:

- **Inversion.** Vec2Text (Morris, Kuleshov, Shmatikov and Rush, EMNLP 2023, arXiv:2310.06816) iteratively corrects a hypothesis text until its embedding matches a target vector. It recovered 92% of 32-token inputs exactly from a GTR-base embedding and extracted personal names from clinical-note embeddings. Public inverters exist for OpenAI `ada-002` [58].
- **Universal geometry.** vec2vec (Jha, Zhang, Shmatikov and Morris, arXiv:2505.12540, 2025) learns, without paired data, to translate embeddings from an unknown model into the space of a known one. This makes leaked vectors from an undocumented model attackable as well [59]. Constructively, the same idea points to migrating an index between models without full re-embedding, though production-grade quality has not been shown.

**Design implications.** Apply the same access control, encryption and retention rules to embeddings as to the source text, especially in multi-tenant indexes (Chapter 05). Do not treat MRL truncation or quantization as privacy protection. Inversion quality drops with them, but the vectors are not protected.

## 4. Benchmarks: MTEB, MMTEB and their limits

**MTEB (2022).** The Massive Text Embedding Benchmark (Muennighoff et al., arXiv:2210.07316) launched with 56 datasets across 8 task types (retrieval, reranking, classification, clustering, pair classification, STS, summarisation, bitext mining) and up to 112 languages [40][43]. Its original central finding still holds: "Model performance varies a lot depending on the task and dataset". No single model dominated [43]. The retrieval subset is essentially BEIR (15 datasets), which is why papers report both numbers [35].

**MMTEB (2025).** The Massive Multilingual Text Embedding Benchmark (Enevoldsen et al., arXiv:2502.13595) is a community expansion with "over 500 quality-controlled evaluation tasks across 250+ languages". It adds instruction-following, long-document and code retrieval [41]. It introduced inter-task-correlation downsampling and hard-negative-sampled retrieval splits to cut evaluation cost while preserving rankings. This produced the leaner MTEB(eng, v2) benchmark now used for English comparisons [41][42]. The `mteb` toolkit now also evaluates image, audio and video tasks [42].

**RTEB (2025).** The Retrieval Embedding Benchmark, released in beta in October 2025 by MTEB maintainers with MongoDB/Voyage and Hugging Face contributors, responds directly to leaderboard overfitting [60]. It mixes open datasets, where corpus, queries and labels are public, with private datasets that only the MTEB maintainers evaluate. Each private set is described only by statistics and sample triplets. A large drop between a model's open and private scores signals overfitting, and the authors report such drops for some models [60]. The datasets cover enterprise domains (legal, healthcare, code, finance) in about 20 languages, each with at least 1K documents and 50 queries [60]. For RAG model selection, the RTEB retrieval view is a better default than the MTEB overall mean.

**Known pitfalls.**

1. **Train/test leakage.** Many models are fine-tuned on the training splits of MTEB retrieval datasets (MS MARCO, NQ, HotpotQA, FEVER). Qwen3's metadata, for example, lists NQ, MSMARCO, HotpotQA, FEVER and MIRACL among its training data [24]. The Sentence Transformers team warns that competitors trained on (Nano)BEIR training splits get an advantage on that benchmark [50]. Treat leaderboard gaps of 1 to 2 points as noise for your use case. RTEB's private splits were built to expose exactly this effect [60].
2. **Average-of-averages hides retrieval.** A model can gain on the overall mean through classification or STS while losing on retrieval. Read the retrieval column, or better, your own evaluation.
3. **Vendor benchmarks are curated.** Voyage's 100-dataset suite and similar internal suites are informative but chosen by the vendor [6].
4. **Size is not destiny.** MMTEB found a 560M model (multilingual-e5-large-instruct) to be the best public model overall [41]. The EmbeddingGemma fine-tune beat a model twice its size in-domain [13].

**Theoretical limits of single-vector retrieval.** Weller, Boratko, Naim and Lee (Google DeepMind, arXiv:2508.21038) prove that the number of distinct top-k document subsets a single-vector embedder can realise is bounded by its dimension [44]. They then test this empirically with "free embeddings", vectors optimised directly on the test relevance labels with no language model involved. With k = 2, the largest corpus at which every top-2 combination is still reachable extrapolates to roughly 500K documents at 512 dimensions, about 4M at 1024 and about 250M at 4096 [44]. Their LIMIT dataset (50K documents, 1,000 queries, each with exactly 2 relevant documents) is linguistically trivial. Even so, state-of-the-art single-vector models stay below about 20% recall@100. BM25 is near-perfect, a multi-vector model does better than single-vector ones, and a long-context cross-encoder solves the small version [44]. LIMIT is now part of MTEB [44].

The implication for RAG designers: queries that combine attributes (AND/OR/NOT, "all documents that mention both X and Y") are structurally hard for any single-vector embedder, whatever its leaderboard rank. Hybrid lexical retrieval, multi-vector models and rerankers remain necessary (Chapters 04 to 06).

## 5. Domain fine-tuning of embedders

**When it pays.** The general models above are trained on web, Wikipedia and Q&A distributions. Specialised vocabularies (clinical, legal, internal product codes), unusual query styles and low-resource languages can be poorly served. Public evidence that fine-tuning helps:

- **Medical.** EmbeddingGemma gained 5.2 NDCG@10 points (0.834 to 0.886) with 100K pairs [13].
- **Low-resource language.** Navasardyan et al. (Metric AI Lab, arXiv:2603.22290) fine-tuned multilingual-e5-base on only 10K noisy, LLM-translated Armenian pairs. The authors report an average gain of about 11 to 12 points across their benchmark (64.30 to about 76.4) and more than 20 points on manually curated retrieval (58.15 to 79.35). Scaling to about 1M pairs, using stronger translation LLMs, or diversifying domains gave no significant further gain. They also report that the method transfers to another low-resource language with its own script (Georgian) at very low data cost [51].
- **Weaker evidence.** LlamaIndex's financial-filing example (Lyft 10-K to train, Uber 10-K to validate) shows the shape of a typical result. bge-small-en reached a hit rate of 0.793 against 0.878 for ada-002 before tuning. A cheap linear adapter on top of the frozen model improved hit rate only modestly, from 0.787 to 0.799, and MRR from 0.643 to 0.662 [48][49].

**Reference pipeline.**

```
 domain corpus chunks
        |
        v
 (1) synthetic queries: LLM writes 1-3 questions per chunk      [13][48][51]
     (or T5 query generator, as in GPL)                         [47]
        |
        v
 (2) hard-negative mining: retrieve top-k with base embedder/BM25,
     drop the positive and near-duplicates (false-negative risk)
        |
        v
 (3) optional pseudo-labels: cross-encoder scores (q, pos, neg) --> margins (GPL)
        |
        v
 (4) train: MultipleNegativesRankingLoss / Cached-MNRL (big effective batch)
            or MarginMSE (distillation), wrapped in MatryoshkaLoss
        |
        v
 (5) merge: average fine-tuned weights with base weights (soup/slerp)  [10][25][51]
        |
        v
 (6) evaluate on held-out REAL queries (NDCG@10, recall@k) vs base + reranker
        |
        v
 (7) re-embed corpus (or train query-side adapter only to avoid re-indexing)
```

**Losses and tooling.** Sentence Transformers v3+ standardised the workflow: Hugging Face `datasets` in, column order defines the role, and a `SentenceTransformerTrainer` runs the loop [45].

- **MultipleNegativesRankingLoss** (InfoNCE with in-batch negatives) is the default for (anchor, positive) or (anchor, positive, negative) data. Its cached variant is "often used to increase the batch size, resulting in superior performance" [46].
- **GISTEmbedLoss** variants are listed for the same data formats [46].
- **MarginMSELoss** distils a cross-encoder teacher's score margins [46].
- **MatryoshkaLoss** keeps the tuned model truncatable [46].
- Use the `NO_DUPLICATES` batch sampler with in-batch-negative losses, so a batch never contains its own positive as a negative [45].

**GPL (Generative Pseudo Labeling).** GPL (arXiv:2112.07577) is the canonical unsupervised domain-adaptation recipe when no queries exist [47]:

1. A T5 model generates queries for target-domain passages.
2. Dense retrievers or BM25 mine negatives. The miners "do not need to be strong for the target task/domain."
3. A cross-encoder pseudo-labels (query, positive, negative) triples.
4. The student learns the teacher's margins with MarginMSE.

It can be combined with TSDAE pre-training on the target corpus [47].

**Failure modes.**
- **Synthetic-query collapse.** LLM-written questions copy the chunk's wording and over-reward lexical overlap. Validate on real user queries.
- **Catastrophic forgetting.** General-domain retrieval degrades after tuning. Merging with base weights mitigates this [51] and is the same tool Qwen3 and Gemini use at pre-training scale [10][25].
- **False negatives in mining.** Mined "negatives" are often actually relevant. Qwen3 masks candidates scoring too close to the positive [25], and Gemini uses an LLM to vet negatives [10].
- **Re-indexing cost.** Any change to the document tower forces a full re-embed. Query-side-only adapters [49], or shared-space model families (Voyage 4 [7]), reduce this cost.
- **Closed API models.** Most closed API models cannot be fine-tuned by users. Domain gains then come from vendor domain models (Voyage law, finance and code [9]) or from a tuned reranker.

## 6. Comparison of representative models

| Model (release) | Org | Weights | Params | Native dim / MRL options | Max tokens | Quantized output | Task conditioning |
|---|---|---|---|---|---|---|---|
| text-embedding-3-small (2024-01) | OpenAI | Closed | n/a | 1536 / any via `dimensions` | 8,191 | No (client-side) | None |
| text-embedding-3-large (2024-01) | OpenAI | Closed | n/a | 3072 / any via `dimensions` | 8,191 | No (client-side) | None |
| embed-v4.0 | Cohere | Closed | n/a | 1536 / 256-1536 | 128K | int8, uint8, binary, ubinary | `input_type` |
| embed-v5.0-pro / fast | Cohere | Closed | n/a | 2048 / 256-2048 | 128K | int8, uint8, binary, ubinary | `input_type` |
| voyage-3-large (2025-01) | Voyage | Closed | n/a | 1024 / 256-2048 | 32K | int8, uint8, binary | `input_type` |
| voyage-4 family (2026-01) | Voyage | Closed (nano open) | n/a | 1024 / 256-2048, shared space | 32K | int8, uint8, binary | `input_type` |
| gemini-embedding-001 (2025-07) | Google | Closed | n/a | 3072 / truncatable | ~2K (per MTEB meta) | No | `task_type` |
| gemini-embedding-2 (2026-04) | Google | Closed | n/a | 3072 / 768, 1536, 3072 | 8,192 | No | Text prefixes only |
| bge-m3 (2024) | BAAI | Open | 568M | 1024 + sparse + multi-vector | 8,192 | Client-side | None |
| multilingual-e5-large-instruct (2024) | Microsoft | Open | 560M | 1024 | 514 | Client-side | Query instruction |
| e5-mistral-7b-instruct (2024) | Microsoft | Open | 7.1B | 4096 | 32K | Client-side | Query instruction |
| Qwen3-Embedding-0.6B / 4B / 8B (2025-06) | Alibaba Qwen | Open (Apache-2.0) | 0.6 / 4 / 7.6B | 1024 / 2560 / 4096, MRL | 32K | Client-side | Query instruction |
| nomic-embed-text-v1.5 (2024-02) | Nomic | Open (code + data) | 137M | 768, MRL | 8,192 | Client-side | Prefixes |
| nomic-embed-text-v2-moe (2025-02) | Nomic | Open | 475M (305M active) | 768 | 512 | Client-side | Prefixes |
| jina-embeddings-v3 (2024-09) | Jina | Open | 572M | 1024, MRL | 8,192 | Client-side | LoRA adapters |
| jina-embeddings-v4 (2025-06) | Jina | Open | 3.8B | 2048 + multi-vector | 32K | Client-side | LoRA adapters |
| jina-embeddings-v5-text-small (2026-02) | Jina | Open | 596M | 1024 / down to 32 | 32K | Client-side | Adapters + prefixes |
| NV-Embed-v2 (2024) | NVIDIA | Open (check licence) | 7.85B | 4096 | 32K | Client-side | Query instruction |
| EmbeddingGemma (2025) | Google DeepMind | Open | ~308M | 768 / 512, 256, 128 | 2K | Client-side | Text prefixes (`task: ... \| query:`) |
| snowflake-arctic-embed-l-v2.0 (2024-12) | Snowflake | Open (Apache-2.0) | 568M | 1024, MRL to 256 | 8,192 | Client-side | Query prefix |
| voyage-context-3 (2025-07) | Voyage / MongoDB | Closed | n/a | 1024 / 256-2048 | 32K | int8, uint8, binary, ubinary | Contextualized chunks |

Sources: [1][3][4][5][6][7][9][11][12][13][14][21][24][25][26][30][31][32][34][36][53][54][57][63]. Release months come from vendor posts or MTEB model metadata. "n/a" means the vendor does not publish the parameter count. For every open model, check the licence on the model card before commercial use.

## Key takeaways for system designers

1. **Choose on your own retrieval eval, not the MTEB average.** Leaderboard gaps of 1 to 2 points are within noise from training-set overlap and task mix [41][50]. If you need a public signal, prefer retrieval-only, partly private suites such as RTEB [60]. A few hundred real queries with judged relevance will tell you more than any public table.
2. **Get the asymmetry right.** Every modern embedder expects role information: a prefix, an `input_type`, a task adapter or a query instruction. Missing or swapped roles are the most common silent quality bug, and Gemini Embedding 2 changes how roles are passed [11].
3. **Budget storage with MRL times quantization.** Start at a 256 to 1024-dimension int8 index and keep full-precision or int8 vectors on disk for rescoring. Binary-plus-rescore at about 96% retention and 32x compression is mature for models trained for it [6][39]. Test retention per model, because it ranges from about 75% to 100% [39].
4. **Prefer query-only instructions and shared spaces to reduce re-indexing.** Query-side instructions [19][26], query-side adapters [49] and cross-size shared embedding spaces [7] let you change behaviour or cost without re-embedding the corpus.
5. **LLM embedders are a trade, not a free upgrade.** 4B to 8B embedders lead benchmarks [25][35], but they cost 10x or more to run than 0.3B to 0.6B models and emit 2560 to 4096 dimensions. A well-tuned small model (mE5-large-instruct, EmbeddingGemma, Jina v5-small, Qwen3-0.6B) plus a reranker often wins on cost-adjusted quality [13][41].
6. **Domain fine-tuning is cheap and effective when the domain is truly shifted.** 10K to 100K synthetic pairs, an in-batch-negative loss, merging with the base model, and evaluation on real queries is a proven recipe [13][51]. Watch for false negatives and forgetting.
7. **Single vectors have hard capacity limits.** For combinatorial or exact-match queries, add BM25/sparse retrieval (BGE-M3 gives both from one model), multi-vector retrieval or a cross-encoder stage [15][44].
8. **Plan for model churn.** Every major family released a new generation in 2025 to 2026. Store raw chunk text and metadata so you can re-embed. Version embeddings by model ID, dimension and dtype. Never mix vectors from different models in one index unless the vendor guarantees a shared space [7].
9. **Secure vectors like text.** Embeddings can be inverted back to close paraphrases of their inputs [58][59]. Apply the source documents' access controls to the index.

## References

1. "Embeddings" guide (cookbook copy of OpenAI documentation). OpenAI, 2024. https://github.com/openai/openai-cookbook/blob/main/examples/data/oai_docs/embeddings.txt
2. "Embedding texts that are longer than the model's maximum context length." OpenAI Cookbook, 2024. https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_long_inputs.ipynb
3. MTEB model metadata: OpenAI models (openai_models.py). MTEB maintainers, 2024-2026. https://github.com/embeddings-benchmark/mteb/blob/main/mteb/models/model_implementations/openai_models.py
4. "Cohere's Embed Models" (embed.mdx, model table). Cohere, 2024-2026. https://github.com/cohere-ai/cohere-developer-experience/blob/main/fern/pages/models/embed.mdx
5. "Introduction to Embeddings at Cohere" (embeddings.mdx). Cohere, 2024-2026. https://github.com/cohere-ai/cohere-developer-experience/blob/main/fern/pages/text-embeddings/embeddings.mdx
6. "voyage-3-large: the new state-of-the-art general-purpose embedding model." Voyage AI, 2025. https://blog.voyageai.com/2025/01/07/voyage-3-large/
7. "Voyage 4 on Vertex AI" (notebook). Google Cloud Platform, vertex-ai-samples, 2026. https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/generative_ai/voyage-4.ipynb
8. voyageai-python SDK README (local voyage-4-nano). Voyage AI, 2026. https://github.com/voyage-ai/voyageai-python
9. MTEB model metadata: Voyage models (voyage_models.py). MTEB maintainers, 2024-2026. https://github.com/embeddings-benchmark/mteb/blob/main/mteb/models/model_implementations/voyage_models.py
10. Lee, J., Chen, F., Dua, S., Cer, D., Shanbhogue, M., et al. "Gemini Embedding: Generalizable Embeddings from Gemini." Google, arXiv:2503.07891, 2025. https://arxiv.org/abs/2503.07891
11. "Embeddings" quickstart notebook (gemini-embedding-001, gemini-embedding-2). Google, Gemini API Cookbook, 2025-2026. https://github.com/google-gemini/cookbook/blob/main/quickstarts/Embeddings.ipynb
12. MTEB model metadata: Google Gemini models (google_gemini.py). MTEB maintainers, 2025-2026. https://github.com/embeddings-benchmark/mteb/blob/main/mteb/models/model_implementations/google_gemini.py
13. Aarsen, T., et al. "Welcome EmbeddingGemma, Google's new efficient embedding model." Hugging Face, 2025. https://huggingface.co/blog/embeddinggemma
14. FlagEmbedding (BGE) repository README. BAAI, 2023-2025. https://github.com/FlagOpen/FlagEmbedding
15. Chen, J., et al. "BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation." BAAI, arXiv:2402.03216, 2024. https://arxiv.org/abs/2402.03216 (project README: https://github.com/FlagOpen/FlagEmbedding/tree/master/research/BGE_M3)
16. Xiao, S., et al. "C-Pack: Packed Resources For General Chinese Embeddings." BAAI, arXiv:2309.07597, 2023. https://arxiv.org/abs/2309.07597
17. Li, C., et al. "Making Text Embedders Few-Shot Learners" (bge-en-icl). BAAI, arXiv:2409.15700, 2024. https://arxiv.org/abs/2409.15700
18. Wang, L., et al. "Text Embeddings by Weakly-Supervised Contrastive Pre-training" (E5). Microsoft, arXiv:2212.03533, 2022. https://arxiv.org/abs/2212.03533
19. Wang, L., et al. "Improving Text Embeddings with Large Language Models" (E5-Mistral). Microsoft, arXiv:2401.00368, 2024. https://arxiv.org/abs/2401.00368
20. Wang, L., et al. "Multilingual E5 Text Embeddings: A Technical Report." Microsoft, arXiv:2402.05672, 2024. https://arxiv.org/abs/2402.05672
21. E5 model family README (unilm/e5). Microsoft, 2022-2024. https://github.com/microsoft/unilm/tree/master/e5
22. Li, Z., et al. "Towards General Text Embeddings with Multi-stage Contrastive Learning" (GTE). Alibaba, arXiv:2308.03281, 2023. https://arxiv.org/abs/2308.03281
23. Zhang, X., et al. "mGTE: Generalized Long-Context Text Representation and Reranking Models for Multilingual Text Retrieval." Alibaba, arXiv:2407.19669, 2024. https://arxiv.org/abs/2407.19669
24. MTEB model metadata: GTE models (gte_models.py). MTEB maintainers, 2023-2025. https://github.com/embeddings-benchmark/mteb/blob/main/mteb/models/model_implementations/gte_models.py
25. Zhang, Y., Li, M., Long, D., et al. "Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models." Alibaba Qwen, arXiv:2506.05176, 2025. https://arxiv.org/abs/2506.05176
26. Qwen3-Embedding repository README. Alibaba Qwen, 2025. https://github.com/QwenLM/Qwen3-Embedding
27. Nussbaum, Z., Morris, J. X., Duderstadt, B., Mulyar, A. "Nomic Embed: Training a Reproducible Long Context Text Embedder." Nomic AI, arXiv:2402.01613, 2024. https://arxiv.org/abs/2402.01613
28. Nussbaum, Z., Duderstadt, B. "Training Sparse Mixture Of Experts Text Embedding Models." Nomic AI, arXiv:2502.07972, 2025. https://arxiv.org/abs/2502.07972
29. contrastors: contrastive learning toolkit. Nomic AI, 2024-2025. https://github.com/nomic-ai/contrastors
30. MTEB model metadata: Nomic models (nomic_models.py). MTEB maintainers, 2024-2025. https://github.com/embeddings-benchmark/mteb/blob/main/mteb/models/model_implementations/nomic_models.py
31. Sturua, S., et al. "jina-embeddings-v3: Multilingual Embeddings With Task LoRA." Jina AI, arXiv:2409.10173, 2024. https://arxiv.org/abs/2409.10173
32. Günther, M., Sturua, S., Akram, M. K., et al. "jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval." Jina AI, arXiv:2506.18902, 2025. https://arxiv.org/abs/2506.18902
33. Akram, M. K., et al. "jina-embeddings-v5-text: Task-Targeted Embedding Distillation." Jina AI, arXiv:2602.15547, 2026. https://arxiv.org/abs/2602.15547
34. MTEB model metadata: Jina models (jina_models.py). MTEB maintainers, 2023-2026. https://github.com/embeddings-benchmark/mteb/blob/main/mteb/models/model_implementations/jina_models.py
35. Lee, C., et al. "NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models." NVIDIA, arXiv:2405.17428, 2024. https://arxiv.org/abs/2405.17428
36. MTEB model metadata: NVIDIA models (nvidia_models.py; includes llama-embed-nemotron-8b, arXiv:2511.07025, and Nemotron-3-Embed). MTEB maintainers, 2024-2026. https://github.com/embeddings-benchmark/mteb/blob/main/mteb/models/model_implementations/nvidia_models.py
37. Kusupati, A., Bhatt, G., Rege, A., et al. "Matryoshka Representation Learning." NeurIPS 2022, arXiv:2205.13147. https://arxiv.org/abs/2205.13147
38. Aarsen, T., Xenova, Sanseviero, O. "Introduction to Matryoshka Embedding Models." Hugging Face, 2024. https://huggingface.co/blog/matryoshka
39. Shakir, A., Aarsen, T., Lee, S. "Binary and Scalar Embedding Quantization for Significantly Faster & Cheaper Retrieval." Hugging Face, 2024. https://huggingface.co/blog/embedding-quantization
40. Muennighoff, N., Tazi, N., Magne, L., Reimers, N. "MTEB: Massive Text Embedding Benchmark." arXiv:2210.07316, 2022. https://arxiv.org/abs/2210.07316
41. Enevoldsen, K., Chung, I., Kerboua, I., et al. "MMTEB: Massive Multilingual Text Embedding Benchmark." arXiv:2502.13595, 2025. https://arxiv.org/abs/2502.13595
42. mteb: evaluation toolkit and leaderboard. MTEB maintainers, 2022-2026. https://github.com/embeddings-benchmark/mteb
43. Muennighoff, N., et al. "MTEB: Massive Text Embedding Benchmark" (blog). Hugging Face, 2022. https://huggingface.co/blog/mteb
44. Weller, O., Boratko, M., Naim, I., Lee, J. "On the Theoretical Limitations of Embedding-Based Retrieval." Google DeepMind, arXiv:2508.21038, 2025. https://arxiv.org/abs/2508.21038 (data and code: https://github.com/google-deepmind/limit)
45. Aarsen, T. "Training and Finetuning Embedding Models with Sentence Transformers v3." Hugging Face, 2024. https://huggingface.co/blog/train-sentence-transformers
46. "Loss Overview." Sentence Transformers documentation, UKPLab / Hugging Face, 2024-2026. https://github.com/UKPLab/sentence-transformers/blob/master/docs/sentence_transformer/loss_overview.md
47. Wang, K., Thakur, N., Reimers, N., Gurevych, I. "GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation of Dense Retrieval." NAACL 2022, arXiv:2112.07577. https://arxiv.org/abs/2112.07577 (code: https://github.com/UKPLab/gpl)
48. "Finetune Embeddings" (example notebook). LlamaIndex, 2023-2025. https://github.com/run-llama/llama_index/blob/main/docs/examples/finetuning/embeddings/finetune_embedding.ipynb
49. "Finetuning an Adapter on Top of any Black-Box Embedding Model" (example notebook). LlamaIndex, 2023-2025. https://github.com/run-llama/llama_index/blob/main/docs/examples/finetuning/embeddings/finetune_embedding_adapter.ipynb
50. Aarsen, T. "Train 400x faster Static Embedding Models with Sentence Transformers." Hugging Face, 2025. https://huggingface.co/blog/static-embeddings
51. Navasardyan, Z., Bughdaryan, S., Minasyan, B., Davtyan, H. "Less is More: Adapting Text Embeddings for Low-Resource Languages with Small Scale Noisy Synthetic Data." Metric AI Lab, arXiv:2603.22290, 2026. https://arxiv.org/abs/2603.22290
52. MTEB model metadata: Cohere models (cohere_models.py). MTEB maintainers, 2023-2026. https://github.com/embeddings-benchmark/mteb/blob/main/mteb/models/model_implementations/cohere_models.py
53. MTEB model metadata: BGE models (bge_models.py). MTEB maintainers, 2023-2025. https://github.com/embeddings-benchmark/mteb/blob/main/mteb/models/model_implementations/bge_models.py
54. Yu, P., Merrick, L., Nuti, G., Campos, D. "Arctic-Embed 2.0: Multilingual Retrieval Without Compromise." Snowflake, arXiv:2412.04506, 2024. https://arxiv.org/abs/2412.04506 (code: https://github.com/Snowflake-Labs/arctic-embed)
55. Günther, M., Mohr, I., Williams, D. J., Wang, B., Xiao, H. "Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models." Jina AI, arXiv:2409.04701, 2024. https://arxiv.org/abs/2409.04701 (code: https://github.com/jina-ai/late-chunking)
56. Morris, J. X., Rush, A. M. "Contextual Document Embeddings." ICLR 2025, arXiv:2410.02525. https://arxiv.org/abs/2410.02525 (code: https://github.com/jxmorris12/cde)
57. "Introducing voyage-context-3: Focused Chunk-Level Details with Global Document Context." Voyage AI / MongoDB, July 2025. https://www.mongodb.com/company/blog/product-release-announcements/voyage-context-3-focused-chunk-level-details-global-document-context
58. Morris, J. X., Kuleshov, V., Shmatikov, V., Rush, A. M. "Text Embeddings Reveal (Almost) As Much As Text." EMNLP 2023, arXiv:2310.06816. https://arxiv.org/abs/2310.06816 (code: https://github.com/jxmorris12/vec2text)
59. Jha, R., Zhang, C., Shmatikov, V., Morris, J. X. "Harnessing the Universal Geometry of Embeddings." arXiv:2505.12540, 2025. https://arxiv.org/abs/2505.12540 (code: https://github.com/rjha18/vec2vec)
60. Liu, F., Enevoldsen, K., Chung, I., Aarsen, T., et al. (MTEB maintainers and MongoDB). "Introducing RTEB: A New Standard for Retrieval Evaluation." Hugging Face, October 2025. https://huggingface.co/blog/rteb
61. "MongoDB picks up Voyage AI, aims to integrate model reranking." Constellation Research, 24 February 2025. https://www.constellationr.com/blog-news/insights/mongodb-picks-voyage-ai-aims-integrate-model-reranking
62. "Elastic Completes Acquisition of Jina AI, a Leader in Frontier Models for Multimodal and Multilingual Search." Elastic investor relations, 9 October 2025. https://ir.elastic.co/News--Events/news/news-details/2025/Elastic-Completes-Acquisition-of-Jina-AI-a-Leader-in-Frontier-Models-for-Multimodal-and-Multilingual-Search/default.aspx
63. nomic-embed-text-v2-moe model card. Nomic AI, 2025. https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe
