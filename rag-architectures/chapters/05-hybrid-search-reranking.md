# 05. Hybrid Search and Reranking: Fusion, Learned Sparse, Late Interaction and Multi-Stage Cascades

## Overview

Previous chapters covered how documents become chunks (Chapter 02), how chunks become
vectors (Chapter 03) and where those vectors live (Chapter 04). This chapter covers the
retrieval stage that decides which few chunks reach the generator. In nearly every
production RAG system that publishes its design, that stage is not one nearest-neighbour
lookup. It is a cascade. Several cheap, high-recall retrievers run in parallel, typically
lexical BM25 plus dense vectors and sometimes learned sparse or multi-vector indexes. Their
lists are fused, and then progressively more expensive models re-score a shrinking
candidate set: a cross-encoder, a late-interaction scorer, and sometimes an LLM.

The motivation is a simple asymmetry. Models that see the query and the document together
are far more accurate than models that encode them separately, and they cost
proportionally more. On the MS MARCO passage benchmark, tuned BM25 reaches an MRR@10 of
about 0.187 [13]. A 22M-parameter MiniLM cross-encoder reranking candidates reaches about
0.390 [28], and a learned sparse SPLADE++ model used as a first-stage retriever reaches
about 0.383 [14]. The cross-encoder cannot be run against millions of documents, while BM25
and SPLADE can, so the architecture follows from the cost of each model.

Anthropic's Contextual Retrieval study gives the clearest public end-to-end number for a
RAG workload [12]. Across its test corpora, the top-20 retrieval failure rate fell from 5.7%
with plain embeddings to 3.7% with contextual embeddings and 2.9% with contextual
embeddings plus contextual BM25. Adding a reranker brought it to 1.9%, a 67% reduction
overall. Hybrid retrieval and reranking account for most of that gain.

This chapter is organised from the bottom of the cascade upward:

1. **Lexical plus dense hybrid retrieval**, and the fusion problem: reciprocal rank fusion
   (RRF), score normalisation and weighted combination, as implemented in Azure AI Search,
   Elasticsearch, OpenSearch, Weaviate, Vespa and Qdrant, and what the academic comparison
   of fusion functions found.
2. **Learned sparse retrieval**: SPLADE and its ancestors (DeepCT, DeepImpact, uniCOIL),
   SPLADE-v3, plus a cautionary tale (BM42).
3. **Late interaction**: ColBERT, ColBERTv2, PLAID and its successors (XTR, WARP,
   FastPlaid, MUVERA), and visual late interaction (ColPali).
4. **Cross-encoder rerankers**: MS MARCO MiniLM models, the BGE family, Qwen3-Reranker,
   jina-reranker-v3, mxbai-rerank, and managed services (Cohere Rerank, Azure semantic
   ranker, Elastic Rerank, Voyage).
5. **LLM rerankers**: RankGPT's listwise permutation generation, its open distilled
   successors (RankVicuna, RankZephyr, LiT5, FIRST), and 2025 reasoning rerankers (Rank1).
6. **Score calibration and thresholding**, the least documented part of the stack.
7. **Multi-stage cascade design**: candidate counts per stage, where each stage runs, and
   how the stages fit together.

A note on method. This chapter draws on primary sources that could be fetched and verified
in October 2026: vendor documentation (mostly from the vendors' public source
repositories), engineering blogs, and the official code repositories of the research
systems. The Azure AI Search figures are taken from Microsoft Learn pages that could not be
re-fetched during the final check, so treat their defaults as subject to change. Where a number comes from a paper abstract that was not re-read in full for this
chapter, it is marked as approximate.

The canonical cascade that the rest of the chapter refines:

```
                      query
                        |
         +--------------+---------------+------------------+
         v              v               v                  v
   +-----------+  +-----------+  +--------------+  +----------------+
   | BM25 /    |  | dense ANN |  | learned      |  | multi-vector   |   STAGE 1
   | inverted  |  | (HNSW,    |  | sparse       |  | (ColBERT/PLAID)|   recall
   | index     |  |  IVF)     |  | (SPLADE)     |  |  optional      |   1e6-1e9 docs
   +-----+-----+  +-----+-----+  +------+-------+  +-------+--------+   ~1-50 ms
         | top 100-1000 each            |                  |
         +--------------+---------------+------------------+
                        v
              +-------------------+
              | fusion: RRF or    |   STAGE 1.5: merge + dedupe -> 50-150 candidates
              | normalised linear |
              +---------+---------+
                        v
              +-------------------+
              | cross-encoder /   |   STAGE 2: precision, 50-150 pairs, ~50-500 ms
              | late-interaction  |
              +---------+---------+
                        v
              +-------------------+
              | LLM listwise      |   STAGE 3 (optional): top 20-100, seconds, $$
              | reranker          |
              +---------+---------+
                        v
              +-------------------+
              | threshold / top-k |   calibrated cut -> 5-20 chunks into the prompt
              +-------------------+
```

---

## 1. Lexical + dense hybrid retrieval

### 1.1 Problem

Dense bi-encoders capture paraphrase and topical similarity but are weak at exact
identifiers: product codes, error strings, rare proper nouns, and terms that never appeared
in the embedding model's training data. BM25 has the opposite profile. Anthropic's study
states plainly that "Embeddings+BM25 is better than embeddings on their own" [12]. Qdrant's
survey of sparse retrieval names the domains where lexical matching matters most: medicine,
law, and e-commerce with brand names and serial numbers [16]. The same survey notes that
"dense retrievers tend to return many false positives" [16]. The engineering problem is how
to combine two ranked lists whose scores live on incompatible scales. BM25 scores are
unbounded. Cosine similarity in Azure AI Search's HNSW implementation falls in
0.333-1.00 [2].

### 1.2 Reciprocal rank fusion (RRF)

RRF was introduced by Cormack, Clarke and Büttcher at SIGIR 2009 in a paper titled
"Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods" [1].
It ignores raw scores entirely. Each document receives

    RRF(d) = sum over lists L containing d of  1 / (k + rank_L(d))

with rank starting at 1. The constant k damps the influence of top positions so that a
document ranked highly by several retrievers beats one ranked first by only one retriever.
The value k = 60 from the original paper has become an industry default:

- **Azure AI Search** uses RRF for all hybrid and multi-vector queries. Its documentation
  states that "the algorithm performs best when you set k to a small value, such as 60"
  [2]. A document ranked first in two lists scores about 2/61, roughly 0.033.
- **Elasticsearch** exposes an `rrf` retriever with `rank_constant` defaulting to 60
  (minimum 1) and `rank_window_size` defaulting to the request `size`. Each child
  retriever "carries an equal weight" in the base formula, at least two children are
  required, and `scroll`, `sort` and `rescore` are unsupported. Pagination is stable only
  while `rank_window_size` is held fixed [4].
- **Vespa** provides `reciprocal_rank(x, k)` with k defaulting to 60.0 and
  `reciprocal_rank_fusion(a, b, ...)` as global-phase normalisers [7].
- **Weaviate**'s original `rankedFusion` used 1/(rank + 60) [8].

RRF's strengths are that it needs no training data, is robust to score-scale differences,
and is trivially parallel. Its weaknesses follow from discarding score magnitude. A
document that one retriever considers an overwhelming match gets no more credit than a
marginal one at the same rank. The fused score is also not a relevance probability. In
Azure it is bounded by the number of fused queries, with each list contributing at most
about 1/k [2]. That makes it useless for absolute thresholds (see Section 6).

Weighted RRF partly addresses the first problem. Azure lets each vector query carry a
weight multiplier (default 1.0). Its documentation shows a vector query with weight 0.5 and
another with weight 2.0 contributing to the fused ranking alongside an unweighted BM25 list
[2].

### 1.3 Score-normalised linear fusion

The alternative is to normalise each list's scores and take a weighted sum.

- **Weaviate `relativeScoreFusion`** (added in v1.20, default since v1.24) min-max
  normalises each result set, so the top hit scores 1 and the bottom scores 0. It then
  combines the two sets with the `alpha` weight. alpha = 0.5 gives equal weight to vector
  and keyword search. In Weaviate's internal tests on FiQA, relative score fusion improved
  recall by about 6% over rank fusion [8].
- **OpenSearch's normalization processor** offers `min_max` (default), `l2` and `z_score`
  normalisation, with `arithmetic_mean` (default), `geometric_mean` or `harmonic_mean`
  combination. Weights must lie in [0, 1] and sum to 1.0 (the documentation's example uses
  [0.3, 0.7]). `z_score` only works with the arithmetic mean. Min-max can be anchored with
  explicit lower and upper bounds, so that a list whose best score is still poor is not
  inflated to 1.0 [11].
- **Elasticsearch's `linear` retriever** supports `minmax` and `l2_norm` normalisers, plus
  `none`, which the documentation warns leaves scores unnormalised across field groups. In
  its multi-field mode, lexical and semantic field groups are each normalised to contribute
  50% of the final score [5].

The failure mode that bounds options address deserves emphasis. Per-query min-max
normalisation always maps the best candidate in each list to 1.0, even when no candidate is
relevant. When one retriever returns nothing useful, the normalised scores amplify its
noise. Rank-based fusion has the same blind spot, but it does not pretend to carry
magnitude.

### 1.4 Where fusion runs

```
   Azure AI Search                         Elasticsearch / OpenSearch
   ---------------                         --------------------------
   text query --> BM25 (maxTextRecallSize   retriever tree:
                   default 1000)              rrf | linear
   vector q(s) -> HNSW (k each)                 |-- standard (BM25)
          \          /                           |-- knn (dense)
           RRF (k=60, per-query weights)         |-- sparse (ELSER/SPLADE-like)
             |  top 50 by default              text_similarity_reranker
           semantic ranker (top 50 only)          wraps any retriever,
             |  @search.rerankerScore 0..4        rank_window_size default 10
           response

   Vespa                                   Weaviate
   -----                                   --------
   content nodes: first-phase (all hits)   hybrid(query, alpha)
                  second-phase (100/node)    -> BM25F + vector in parallel
   container:     global-phase (100 total)   -> relativeScoreFusion (default)
                  RRF / normalize_linear        or rankedFusion
                  ONNX cross-encoder         -> optional reranker module
```

Azure's pipeline illustrates how product limits shape the cascade. A hybrid query returns
the top 50 fused results by default. The text side recalls up to 1,000 matches
(`maxTextRecallSize`). Only the top 50 results move on to the semantic ranker, "even if
results include more than 50 results" [2][3]. A relevant document at fused rank 51 can
therefore never be promoted. Recall at the fusion stage sets the ceiling for everything
downstream.

### 1.5 What the fusion literature says: convex combination versus RRF

The engine defaults above favour RRF, but the most systematic academic comparison points the
other way when labels exist. Bruch, Gai and Ingber ("An Analysis of Fusion Functions for
Hybrid Retrieval", ACM TOIS 2023) compared a convex combination of normalised lexical and
semantic scores, `alpha * s_dense + (1 - alpha) * s_lexical`, against RRF in-domain and
out-of-domain [44]. Their findings: RRF is sensitive to its parameters (k = 60 is not
universally safe), the convex combination outperforms RRF in both settings, and its single
parameter can be tuned with a small set of labelled queries. The choice of normaliser
mattered less than expected. This is the academic counterpart to Weaviate's switch of
default and OpenSearch's normalisation processor [8][11]. RRF remains the right choice when
there are no labels at all, or when more than two heterogeneous lists are fused.

### 1.6 Lessons

- **Hybrid is the default.** Anthropic, Azure, Elastic, Weaviate, Vespa and Qdrant all
  ship it as a first-class query type [2][4][7][8][12][17].
- **RRF with k = 60 is the zero-configuration baseline.** Move to weighted or normalised
  fusion once you have labelled queries to tune against; the convex combination needs only
  a small set [44]. Weaviate's roughly 6% FiQA recall gain
  from normalised fusion is an internal benchmark, not a universal law [8].
- **Size each child list deliberately.** RRF can only reorder what the children return,
  and Elastic's pagination stability depends on a fixed window [4].
- **Contextualise chunks before indexing them for BM25.** Anthropic prepends 50-100 tokens
  of LLM-generated context to each chunk before both embedding and BM25 indexing. That costs
  about $1.02 per million document tokens one time, with prompt caching [12].

---

## 2. Learned sparse retrieval (SPLADE and relatives)

### 2.1 Problem

BM25 suffers from vocabulary mismatch. A query for "pizza" will not match a document that
only says "Margherita" [16]. Dense retrieval fixes that but gives up the inverted index,
exact-match guarantees and explainability. Learned sparse retrieval trains a transformer to
output term weights over a vocabulary. The output is stored in an ordinary inverted index,
so it keeps BM25-style serving while learning both term importance and expansion terms.

### 2.2 Lineage

Qdrant's 2024 survey traces the evolution [16]:

| Model | Idea | Limitation noted |
|---|---|---|
| DeepCT | Regress an importance weight per word | Ground-truth "importance" is ill-defined |
| DeepImpact | Learn per-word impact from relevance labels | Uses only the first subword piece |
| TILDEv2 | Token-level impacts | One scalar per token cannot separate homonyms |
| COIL | 32-dim vector per term (projected from 768) | Costly storage, not a plain inverted index |
| uniCOIL | Back to scalar impacts | Loses much of COIL's semantics |
| SPARTA | Expansion over BERT's roughly 30k vocabulary | May generalise worse than BM25 off MS MARCO |
| SPLADE / SPLADE++ | MLM-head expansion + sparsity regularisation + distillation | See below |

External document expansion with docT5query generates queries of about 50-60 words per
document. It is slow because it decodes token by token. TILDE cuts expansion time by about
98% relative to docT5query. Qdrant suggests external expansion may be infeasible at
production scale, which pushes designs toward internal expansion [16].

### 2.3 SPLADE architecture

SPLADE (Formal, Piwowarski and Clinchant, SIGIR 2021) "learns query/document sparse
expansion via the BERT MLM head and sparse regularization" [14]. Each input token's MLM
logits produce a weight for every vocabulary term, and these are pooled into one
vocabulary-sized sparse vector. A FLOPS-style regulariser, with separate strengths
`lambda_q` and `lambda_d` for queries and documents, pushes most weights to zero, which
keeps posting lists short [14]. The regularisation strength is the dial between
effectiveness and sparsity.

```
  text --> BERT encoder --> MLM head: logits over |V| ~ 30k terms per token
                                  |
                          pool over tokens + saturate
                                  |
                       sparse vector {term: weight}   (most weights = 0, by FLOPS reg.)
                                  |
            +---------------------+----------------------+
            | documents: indexed offline in an           |
            |   inverted index (Anserini/PISA/Lucene/    |
            |   Qdrant sparse/Elastic, etc.)             |
            | queries: encoded online (or use a          |
            |   lighter/separate query encoder)          |
            +--------------------------------------------+
```

Published MS MARCO dev MRR@10 for the NAVER checkpoints [14]:

| Checkpoint | Generation | MRR@10 |
|---|---|---|
| splade_v2_max | v2 | 34.0 |
| splade_v2_distil | v2 (distillation) | 36.8 |
| splade-cocondenser-selfdistil | SPLADE++ | 37.6 |
| splade-cocondenser-ensembledistil | SPLADE++ | 38.3 |
| efficient-splade-V-large (doc + query) | Efficient SPLADE | 38.8 |

For comparison, tuned BM25 (k1 = 0.82, b = 0.68) scores 18.7 on the same split [13]. SPLADE
v2 [15] added hard-negative mining, cross-encoder distillation and better pre-trained
initialisation. SPLADE++ extended the study of those three ingredients [14]. The Efficient
SPLADE line (Lassance and Clinchant, SIGIR 2022) targets latency. It uses L1 regularisation
on queries, separate query and document encoders, and FLOPS-regularised middle training.
Its abstract claims latency within 4 ms of BM25 under the same compute, with less than a
10% MRR@10 drop from state-of-the-art single-stage neural rankers in-domain [14].

### 2.4 Failure modes and operational notes

- **Licence.** The NAVER code and checkpoints are CC BY-NC-SA 4.0 (non-commercial) [14].
  This matters for commercial deployments and is one reason vendors train their own
  learned-sparse models.
- **Generalisation.** Models trained on MS MARCO can generalise worse than BM25 on other
  domains [16]. BEIR (Thakur et al., NeurIPS 2021 Datasets and Benchmarks) was built to
  measure exactly this zero-shot gap across lexical, sparse, dense and reranking
  architectures [43].
- **Query-time cost.** Unlike BM25, every query needs a transformer forward pass, and
  expanded queries produce longer posting-list traversals. Hence the separate lightweight
  query encoders in Efficient SPLADE [14].

### 2.5 SPLADE-v3 and the 2024 baselines

SPLADE-v3 (Lassance, Déjean, Formal and Clinchant, March 2024 technical report) is the
current NAVER baseline series [45]. The main training changes are many hard negatives per
query and distillation from an ensemble of cross-encoder rerankers. The abstract reports
above 40 MRR@10 on MS MARCO dev, statistically significant gains over BM25 and SPLADE++,
about 2% better out-of-domain BEIR effectiveness, and competitiveness with cross-encoder
rerankers. The v3 checkpoints keep the non-commercial licence noted above, so the same
commercial caveat applies [14][45]. Engine vendors meanwhile ship their own learned-sparse
models under commercial terms (Elastic's ELSER is the sparse leg in Elastic's retriever
trees [5]), which is the practical route for most production deployments.

### 2.6 Cautionary tale: BM42

In July 2024 Qdrant introduced BM42 [17]. It replaced BM25's term frequency with the
attention weights of the `[CLS]` token in a small transformer (all-MiniLM-L6-v2 by
default), kept IDF (computed inside the engine so it stays current under updates), and
produced very compact vectors: about 5.6 non-zero elements per document, roughly a 13 MB
index for about 530k documents in uint8 [17]. The post was later corrected. An evaluation
script error had inflated the results, and the corrected note states that BM42 does not
outperform other vendors' BM25 implementations. The author now calls it experimental and
notes BM25 may remain better for large, unchunked documents [17]. The lesson for readers of
retrieval announcements: check vendor-reported sparse-retrieval gains against independent
BEIR-style runs before you rebuild an index around them.

---

## 3. Late interaction: ColBERT, PLAID and successors

### 3.1 Problem

Single-vector bi-encoders compress a passage into one point. Cross-encoders model
token-to-token interactions but must run once per query-document pair. Late interaction
sits between them. Documents are encoded offline into one vector per token. At query time,
each query token is matched to its most similar document token (MaxSim), and the maxima are
summed.

    score(q, d) = sum over query tokens i of  max over doc tokens j of  (q_i . d_j)

Interactions stay token-level, but document encoding is precomputed, so the method can work
as a retriever or as a cheap reranker.

### 3.2 ColBERT and ColBERTv2

ColBERT (Khattab and Zaharia, SIGIR 2020) introduced the design [18]. Its abstract reports
that it is over two orders of magnitude faster and needs about four orders of magnitude
fewer FLOPs per query than BERT-based cross-encoder rerankers (approximate figures from the
paper's abstract). ColBERTv2 (NAACL 2022) added residual compression and denoised
supervision distilled from a cross-encoder [19]. Each token vector is stored as a centroid
ID plus a quantised residual; the repository's examples use `nbits=2`. The authors found
this more robust than their earlier binarisation scheme [21]. The paper reports roughly a
6-10x reduction in index footprint compared with uncompressed ColBERT (approximate, from the
abstract) [19]. The project README describes search over large collections in "tens of
milliseconds", and a Colab notebook indexes 10,000 passages on a free T4 GPU in about six
minutes [21].

### 3.3 PLAID

PLAID (CIKM 2022) is the retrieval engine for ColBERTv2 [20]. It treats the centroids
themselves as a cheap first approximation of each document:

```
 query tokens (32 x 128-d)
        |
        v
 [1] centroid interaction: score query tokens vs all centroids (K-means codebook)
        |   select candidate passages whose token-centroids score high (IVF probe)
        v
 [2] centroid pruning: drop low-scoring centroids; approximate MaxSim using
        |   centroid scores only (no residual decompression)
        v
 [3] keep top candidates (FastPlaid default n_full_scores = 4096)
        |
        v
 [4] decompress residuals for survivors -> exact MaxSim -> final top-k
```

The paper reports latency reductions of up to about 7x on GPU and 45x on CPU over vanilla
ColBERTv2, with search at the scale of about 140M passages (approximate, from the abstract)
[20]. The open-source descendants publish concrete parameters. LightOn's FastPlaid, a Rust
reimplementation with PyO3 and tch-rs bindings, defaults to `n_ivf_probe = 8` and
`n_full_scores = 4096`. On an H100 it reports 69% to 554% higher QPS than the original
PLAID across eight BEIR datasets, with nearly identical NDCG@10 (for example Quora
43 -> 282 QPS at 0.88 vs 0.87 NDCG@10) [22]. PyLate (LightOn, CIKM 2025) is a Sentence
Transformers-based training and inference library for ColBERT-style models. Its README
builds its PLAID index on FastPlaid and offers alternative WARP and TACHIOM backends [23].

### 3.4 XTR, WARP and the CPU path

Google DeepMind's XTR ("Rethinking the Role of Token Retrieval in Multi-Vector Retrieval",
NeurIPS 2023) changes training so that the token-retrieval stage itself is more
informative. This reduces the need for the expensive gather-and-rescore step [24]. WARP
(SIGIR 2025) is a CPU-optimised engine that combines ideas from ColBERTv2/PLAID and XTR
[25]. PyLate's documentation warns that WARP's approximations assume XTR-trained models and
can degrade results for models not trained that way. It also lists a CPU index, TACHIOM,
which it reports beats WARP on ColBERTv2 across build time, quality, latency and index size
[23].

### 3.5 Late interaction as a reranker, and MUVERA

Many production designs avoid a full PLAID index. They store multi-vectors only as a
rerank payload behind a single-vector or sparse first stage. Qdrant 1.10 added multi-vector
fields with a `MAX_SIM` comparator and a `prefetch` query API for exactly this pattern [26].
Their experiments reused an ordinary dense model's output token embeddings as multi-vectors:
retrieve the top 50 with the pooled vector, then MaxSim-rerank to the top 10. Results:

- SciFact NDCG@10: 0.645 -> 0.703 (MiniLM) and 0.682 -> 0.731 (bge-small).
- TREC-COVID: 0.472 -> 0.638 (MiniLM).
- ArguAna: reranking hurt for both models, 0.502 -> 0.454 (MiniLM) and 0.589 -> 0.573
  (bge-small).

The ArguAna drop shows that late-interaction reranking is not monotonically beneficial
[26]. Storage is the real cost. MiniLM token vectors are 384-dimensional versus 128 for
ColBERT. uint8 scalar quantisation cut memory about 4x with negligible quality loss
(SciFact 0.707 -> 0.703) [26].

MUVERA (Google Research, NeurIPS 2024) goes further [47]. It compresses a multi-vector set into one
fixed-dimensional encoding (FDE) so that a standard ANN index can serve the first stage,
with exact MaxSim as a rerank. On NFCorpus with ColBERTv2, Qdrant measured NDCG@10 of 0.347
for full multi-vector search (1.27 s average), 0.242 for MUVERA alone (0.15 s), and 0.343
for MUVERA plus multi-vector rerank (0.18 s). That is about 7x faster at nearly identical
quality. FDEs can be very large, for example 40,960 dimensions in one configuration [27].

### 3.6 Visual late interaction: ColPali

ColPali (Faysse et al., ICLR 2025) carries late interaction from text tokens to image
patches [46]. A vision-language model (PaliGemma) embeds each page image directly into one
vector per patch, and queries are scored against pages with the same MaxSim operator as
ColBERT. This skips OCR, layout parsing and chunking entirely. The paper also introduced the
ViDoRe benchmark for visually rich page retrieval, where queries need both textual and visual
understanding. The authors report that it largely outperforms text-extraction pipelines
while being simpler and faster to index [46]. For cascade design, ColPali-style indexes
behave like ColBERT ones: each page yields on the order of a thousand patch vectors, so the
storage, pooling and MUVERA-style approximation techniques above apply directly [27][47].
Chapter 02's discussion of document parsing is the natural companion.

### 3.7 Lessons

- **Late interaction is a storage problem first.** Residual compression, product
  quantisation and scalar quantisation are mandatory, not optional [19][22][26].
- **The centroid trick generalises.** Approximate cheaply, prune aggressively, and
  decompress only survivors. PLAID's structure is itself a miniature cascade.
- **Use multi-vectors as a reranking payload when a full PLAID index is too heavy**
  (Qdrant prefetch, MUVERA plus rerank) [26][27].

---

## 4. Cross-encoder rerankers

### 4.1 Architecture

A cross-encoder concatenates `[CLS] query [SEP] passage [SEP]`, runs full self-attention
across both, and maps the output to a relevance score. Because the query is needed at
encoding time, nothing can be precomputed. Cost is one forward pass per candidate, linear in
the candidate count and quadratic in the joint sequence length. The Sentence Transformers
documentation codifies the standard pattern: a bi-encoder or lexical retriever fetches about
100 candidates, and a cross-encoder reranks them [29]. Weaviate's 2022 description is the
same two-stage design. It notes that pretrained MS MARCO cross-encoders work out of the box
and recommends fine-tuning for out-of-domain data [9].

### 4.2 The open small-model baseline: MS MARCO MiniLM cross-encoders

The Sentence Transformers model card table is the most-cited speed/quality frontier for
compact rerankers [28]:

| Model | NDCG@10 TREC DL19 | MRR@10 MS MARCO dev | Docs/sec |
|---|---|---|---|
| ms-marco-TinyBERT-L2-v2 | 69.84 | 32.56 | 9,000 |
| ms-marco-MiniLM-L2-v2 | 71.01 | 34.85 | 4,100 |
| ms-marco-MiniLM-L4-v2 | 73.04 | 37.70 | 2,500 |
| ms-marco-MiniLM-L6-v2 | 74.30 | 39.01 | 1,800 |
| ms-marco-MiniLM-L12-v2 | 74.31 | 39.02 | 960 |
| ms-marco-electra-base | 71.99 | 36.41 | 340 |

Two observations follow. First, quality saturates at six layers: L12 is half the speed of
L6 for 0.01 points. Second, throughput (hardware unspecified in the source) translates
directly into a latency budget. Reranking 100 candidates at 1,800 docs/s is roughly 55 ms of
model time. Elastic ships `cross-encoder/ms-marco-MiniLM-L-6-v2` as its documented example
of a self-hosted `text_similarity` model [6].

### 4.3 Open multilingual and LLM-backbone rerankers

**BGE rerankers (BAAI).** The FlagEmbedding family spans several tiers [30]:
`bge-reranker-base` and `-large` (XLM-RoBERTa, Chinese and English); `bge-reranker-v2-m3`
(built on BGE-M3, multilingual, recommended for efficiency); `bge-reranker-v2-gemma`
(Gemma-2B); and `bge-reranker-v2-minicpm-layerwise` (MiniCPM-2B). The layerwise model lets
callers pick an output layer between 8 and 40 to trade quality for speed, which amounts to
an early-exit cascade inside one model. Scores are raw logits. `normalize=True` applies a
sigmoid, so a raw -5.65 becomes about 0.0035 [30].

**Qwen3-Reranker (Alibaba, June 2025).** Sizes are 0.6B, 4B and 8B, with a 32K context
[31]. The model runs as a causal LM given an instruction, a query and a document. The score
is the softmax probability of "yes" over {"yes", "no"} at the final position. That makes it
instruction-aware: one model can be steered toward, for example, code search or legal
relevance. On reranking the top 100 from Qwen3-Embedding-0.6B, the 4B model scored 69.76
MTEB-R and 81.20 MTEB-Code, versus 57.03 and 41.38 for bge-reranker-v2-m3 and 58.22 and
58.98 for jina-multilingual-reranker-v2-base [31]. On FollowIR (instruction following) the
4B model scored 14.84, while the 8B scored only 8.05, and the 8B was also slightly behind
the 4B on MTEB-R (69.02). Bigger was not uniformly better [31].

**jina-reranker-v3 (Jina AI, September 2025).** A 0.6B multilingual reranker that is
listwise inside a single forward pass [48]. Instead of scoring each pair separately, it puts
the query and many candidate documents in one causal-attention context window and reads
each document's relevance from its final-token embedding, a design the paper calls "last but
not late" interaction. The paper reports about 61.9 BEIR nDCG@10, slightly above the
Qwen3-Reranker-4B figure in its own comparison, at roughly one sixth of the size [48]. It
shows that listwise comparison does not require autoregressive permutation generation
(Section 5).

**mxbai-rerank v2 (Mixedbread).** Sizes are 0.5B and 1.5B, trained with a three-stage recipe
that the README lists as GRPO, contrastive learning and preference learning.
The models support 100+ languages and an 8K default context, extendable to 32K. BEIR average
is 57.49 for large-v2 and 55.57 for base-v2, versus 49.32 for the v1 large model. Latency
on an A100 is 0.89 s and 0.67 s respectively, under a benchmark setting the README does not
fully specify [32].

### 4.4 Managed rerankers

**Cohere Rerank.** Cohere's documentation now lists Rerank 4.0 (`rerank-v4.0-pro`,
`rerank-v4.0-fast`), Rerank 3.5 (`rerank-v3.5`) and the older split English/multilingual
3.0 models [33]. Rerank 3.5 and 4.0 are single multilingual models trained for 100+
languages [33]. Operational parameters [34]:

- **Context:** 4,096 tokens for v3.5 and v3.0 (query capped at 2,048). 32,768 tokens for
  v4.0 (query capped at 16,384).
- **Long documents:** split into chunks that fit the window, each chunk scored, and the
  maximum taken. A 10,000-token document under v3.5 becomes 3 chunks. A 100,000-token
  document under v4.0 becomes 4.
- **Limits:** up to 10,000 documents per request, and documents x max_chunks_per_doc
  must not exceed 10,000.
- **Semi-structured input:** JSON documents with `rank_fields`, where field order matters.

Anthropic's Contextual Retrieval pipeline used Cohere's reranker to cut 150 hybrid
candidates to 20. It was the step that took the failure-rate reduction from 49% to 67% [12].
Weaviate showed Cohere's reranker can be fine-tuned with as few as 256 unique queries (plus
a validation set of at least 64). On a synthetic DSPy-generated dataset of Weaviate blog
content, this gave about +8% Accuracy@1, +5% MRR@10 and +4% NDCG@10 over the base model.
Training took about 2 hours and the demo cost $0.03 [10].

**Azure AI Search semantic ranker.** This is a Bing-derived multilingual model applied only
to the top 50 fused results [3]. It produces `@search.rerankerScore` on a 0-4 scale, where
4 means the document fully answers the query. Inputs are summarised per document to about
2,048 tokens (raised from 256 in November 2024), with `title` and `keywords` fields capped
at 128 tokens each [3]. Field order in the semantic configuration matters because longer
text is trimmed [3].

**Elastic Rerank.** The `text_similarity_reranker` retriever wraps any retriever. It
defaults to the `.rerank-v1-elasticsearch` endpoint and a `rank_window_size` of 10, and it
can also call Cohere, Google Vertex AI or self-hosted Hugging Face models [6]. Its
`chunk_rescorer` (GA in 9.4) scores the best chunks of long documents instead of truncating
them. The documentation calls it "an expert feature" and warns that chunks configured
larger than the reranker's token limit "can degrade relevance significantly" [6].

**Voyage and Jina.** Both sell API rerankers. Anthropic notes Voyage offers one but did not
test it [12], and the `rerankers` library wraps Cohere, Jina, Voyage, MixedBread, Pinecone
and others behind one interface [42]. Voyage's `rerank-2.5` and `rerank-2.5-lite` (August
2025) were announced as its first instruction-following rerankers, with a 32K-token
context; the vendor reports about 7.9% and 7.2% higher accuracy than Cohere Rerank 3.5 on its
own 93-dataset suite [49]. Jina's reranker line is covered under open models above
(jina-reranker-v3) [48]. Both sets of figures are vendor-reported. The independent
comparison point available is Qwen's table, where jina-multilingual-reranker-v2-base (0.3B)
scores 58.22 MTEB-R [31].

### 4.5 Failure modes

- **Truncation.** A 512-token cross-encoder silently drops the end of long chunks. Cohere
  and Elastic handle this with max-over-chunks scoring [6][34]. For open models, chunk
  before reranking.
- **Domain shift.** MS MARCO-trained rerankers inherit web-search notions of relevance.
  Weaviate's fine-tuning results and Qwen's instruction prompts are two remedies [10][31].
- **Window ceilings.** The reranker can only reorder what reaches it. Azure fixes the
  semantic-ranker window at 50 [3], and Elastic's default window is 10 [6]. A default
  window of 10 means the reranker cannot rescue anything ranked 11th or lower. Raise it
  deliberately.
- **Negative transfer.** Qdrant's ArguAna experiment shows that reranking can lower
  NDCG@10 [26]. ArguAna's counter-argument retrieval task is not what most rerankers were
  trained for. Always A/B a reranker against the fused first stage.

---

## 5. LLM rerankers

### 5.1 RankGPT: listwise permutation generation

Sun et al. ("Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking
Agent", EMNLP 2023, Outstanding Paper Award) showed that instruction-following LLMs can
rerank directly [35]. The prompt lists numbered passages and asks the LLM to output a
permutation such as `[2] > [3] > [1]`. This is listwise reranking: the model compares
candidates against each other rather than scoring each in isolation. Because a context
window cannot hold 100 passages, RankGPT uses a sliding window that runs from the back of
the list to the front. In the TREC-DL19 setup, 100 BM25 candidates are reranked with
window size 20 and step 10 [35]. Each pass bubbles strong passages forward, so the top of
the list is resolved last, with the best context.

```
  BM25 top-100:  [ 1 ............................................ 100 ]
  pass 1:                                              [81 ...... 100]  rerank 20
  pass 2:                                       [71 ...... 90]          step 10
  ...                                     (strong docs bubble forward)
  pass 9:        [ 1 ...... 20]  -> final order of the top 20
  => ~9 LLM calls per query, each with 20 passages in context
```

RankGPT also introduced permutation distillation. A DeBERTa-v3-base cross-encoder was
trained with RankNet loss on ChatGPT-generated permutations for 10K and 100K MS MARCO
queries, and the data and a `deberta-10k-rank_net` checkpoint were released [35]. This
pattern, using an expensive LLM as a teacher for a cheap student reranker, recurs throughout
the field.

### 5.2 Open listwise rerankers

The Waterloo `rank_llm` toolkit (SIGIR 2025 resource paper) packages the open successors
and documents pointwise, pairwise and listwise families side by side [36]:

| Family | Example | Reported context/config | Notes |
|---|---|---|---|
| Pointwise | monoT5, monoELECTRA [41] | top 1,000, 512-token context | Scores each pair independently |
| Pairwise | duoT5 [41] | top 50 | O(n^2) comparisons, so only the head of the list |
| Listwise (open LLM) | RankVicuna [37], RankZephyr [38] | top 100, 4,096 context | 7B open models distilled from GPT teachers |
| Listwise (seq2seq) | LiT5 [39] | window 20 for LiT5-Distill | T5-based listwise, small context |
| Listwise (single-token) | FIRST [40] | 4,096 context | Ranks from first-token logits, not full decoding |
| Listwise (proprietary) | RankGPT, RankGemini | API | Teacher models |

The mono-duo split from Pradeep, Nogueira and Lin [41] is itself a cascade. A pointwise
model trims 1,000 candidates. A pairwise model, quadratic in cost, then reorders only the
top 50. FIRST tackles the main cost of listwise LLM reranking, which is generating the
permutation token by token. It reads the ranking from the logits of the first generated
identifier, so one decoding step replaces many [36][40].

### 5.3 Pointwise LLM rerankers blur the line

Qwen3-Reranker's yes/no probability and BGE's Gemma and MiniCPM rerankers are LLM backbones
used pointwise [30][31]. They keep the cross-encoder interface (one score per pair,
trivially batched, compatible with thresholds) while inheriting LLM instruction following.
In 2025-2026 this has become the dominant open design. Full listwise generation is
increasingly reserved for offline teaching and for small, high-value candidate sets.

### 5.4 Reasoning rerankers (2025)

The 2025 wave of reasoning models produced rerankers that think before they judge. Rank1
(Weller et al., COLM 2025) distils reasoning traces from DeepSeek-R1 on MS MARCO, releasing
more than 600,000 of them, into pointwise rerankers from 0.5B to 32B parameters built mostly
on Qwen2.5 [50]. Each model generates a reasoning chain and then a relevance judgment. The
motivation is the BRIGHT benchmark (Su et al., ICLR 2025), whose queries from StackExchange,
coding and maths require reasoning to judge relevance and on which conventional retrievers
and rerankers score poorly [51]. Rank1 reports large gains on BRIGHT over non-reasoning
rerankers, and an ablation shows the reasoning chains, not only the extra training data,
drive much of that gain [50]. The cost is many generated tokens per candidate. Reasoning
rerankers are therefore a stage-3 option for hard, reasoning-intensive queries, not a
default stage 2.

### 5.5 Failure modes and costs

- **Latency and cost.** A 100-candidate sliding window with window 20 and step 10 costs
  about 9 sequential LLM calls per query (see the diagram). That fits research evaluation
  and high-value queries, not interactive search at volume.
- **Output brittleness.** Generated permutations can omit or duplicate IDs, so parsers
  must repair them. The RankGPT repo's `receive_permutation` step exists for this [35].
- **Positional bias.** The initial order affects results, which is why the back-to-front
  window starts from the first-stage ranking.
- **No scores.** A permutation carries no calibrated relevance, so thresholding (Section 6)
  needs a separate signal.

---

## 6. Score calibration and thresholding

Every stage emits a number, and almost none of them are probabilities. Yet RAG systems
routinely need absolute decisions. Should a chunk enter the prompt? Does the query fall
outside the corpus ("I don't know")? Should a dense or sparse result be dropped before
fusion? The public guidance:

| Stage / product | Score semantics | Documented guidance |
|---|---|---|
| BM25 | Unbounded, query-dependent | Not comparable across queries; normalise or use ranks [2] |
| Azure vector (cosine) | 0.333-1.00 | Threshold property exists; avoid overly granular limits [2][3] |
| RRF (any engine) | At most about n_lists/k | Ranking only; never threshold [2] |
| Weaviate relativeScore | Per-query min-max to [0,1] | Top hit always 1.0, so do not threshold blindly [8] |
| OpenSearch min_max | [0,1], optional fixed bounds | Use `lower_bounds`/`upper_bounds` to anchor [11] |
| Cohere Rerank | [0,1], query-dependent | Calibrate the threshold on 30-50 queries with borderline docs [34] |
| Azure semantic ranker | 0-4 (4 = fully answers) | Distribution drifts with infra and model updates [3] |
| BGE reranker | Logit; sigmoid via `normalize=True` | Sigmoid is monotone, not calibrated [30] |
| Qwen3-Reranker | P(yes) over {yes, no} | Probability-shaped, instruction-dependent [31] |
| Elastic text_similarity | Transformed to non-negative | Negative maps to (0,1), positive to [1, inf) [6] |
| LLM listwise | Permutation only | Needs a separate scorer for cut-offs [35] |

Cohere's documentation is the most explicit vendor statement on calibration [34]. The
primary output is the rank. Scores fall in [0, 1] but are query-dependent, and the
documentation warns against reading ratios: a document scoring 0.91 is not "twice as
relevant" as one scoring 0.04. To set a cut-off, pick 30-50 representative queries, pair each with a
borderline-relevant document, score them, and use the mean as the reference threshold.
Microsoft adds a second caution. Semantic ranker scores can shift slightly for the same
query because of infrastructure conditions and model updates, so thresholds should not be
"too granular" [3].

Practical patterns that follow:

1. **Calibrate only at the last scored stage.** Fused RRF scores and min-max-normalised
   scores carry no cross-query meaning.
2. **Learn a per-model mapping on in-domain labels**: Cohere's borderline-mean recipe, or
   Platt/isotonic regression on the raw logits. Re-fit it whenever the reranker version
   changes, given Azure's drift warning [3].
3. **Combine an absolute floor with relative rules**: keep at most k chunks, keep only
   those above the floor, and always keep at least one if the floor is used for answer
   gating rather than filtering.
4. **Anchor normalisation.** OpenSearch-style fixed min/max bounds stop a junk list from
   being stretched to [0, 1] [11].

---

## 7. Multi-stage retrieval cascades

### 7.1 The pattern

A cascade spends compute where it changes the outcome. Each stage has a recall job (do not
lose the relevant item) and a precision job (push it up). Cheap stages see many documents;
expensive stages see few. The public systems share a consistent shape:

| System | Stage 1 | Fusion | Stage 2 | Final to LLM / user |
|---|---|---|---|---|
| Anthropic Contextual Retrieval [12] | Contextual embeddings + contextual BM25 | Rank fusion + dedupe | Cohere rerank of top 150 | Top 20 |
| Azure AI Search [2][3] | BM25 (up to 1,000) + HNSW | RRF k=60, weighted | Semantic ranker, top 50 only | `top` (default 50) |
| Elasticsearch [4][5][6] | standard + knn (+ sparse) | rrf or linear | text_similarity_reranker (window 10 by default) | `size` |
| Vespa [7] | first-phase on every content node (WAND) | global-phase RRF / normalize_linear | second-phase (100/node), global-phase ONNX (100) | Hits |
| Qdrant [26][27] | Prefetch dense / sparse / MUVERA | RRF in Query API | MaxSim multi-vector rerank | Top 10 |
| Sentence Transformers recipe [29] | Bi-encoder or BM25, ~100 | n/a | MiniLM cross-encoder | Top 5 |
| RankLLM / mono-duo [36][41] | BM25 top 1,000 | n/a | monoT5 on 1,000, then duoT5 on 50 or listwise LLM on 100 | Top 10-20 |
| PLAID internal [20][22] | Centroid IVF probe | n/a | Centroid-only approx MaxSim, then exact MaxSim on 4,096 | Top k |

### 7.2 Vespa's phased ranking as a reference design

Vespa makes the cost model explicit [7]:

```
      +--------------------------- content node (x N) ---------------------------+
      | matching: WAND / weakAnd limits candidates to top-K per node             |
      |    |                                                                     |
      |    v                                                                     |
      | first-phase: cheap expression on EVERY matched hit (bm25 + closeness)    |
      |    |  rank-score-drop-limit can discard hits                             |
      |    v                                                                     |
      | second-phase: heavier expression (e.g., GBDT, MaxSim) on best 100/node   |
      |    |  total-rerank-count caps the sum across nodes                       |
      +----+---------------------------------------------------------------------+
           v merged
      +--------------------------- stateless container --------------------------+
      | global-phase: best 100 globally (rerank-count / ranking.globalPhase...)  |
      |   - cross-hit normalisers: normalize_linear, reciprocal_rank(k=60), RRF  |
      |   - ONNX cross-encoder inference, optional GPU (gpu-device)              |
      +--------------------------------------------------------------------------+
```

The placement rules are instructive. First-phase cost scales with hits times expression
cost, so it must stay cheap. Large GBDT models are more efficient in second-phase on the
content nodes, next to the data. Cross-encoders belong in the global phase on stateless
containers, which can use GPUs and see the globally merged list. Normalisers such as RRF
only make sense in the global phase, because they need the whole candidate set [7].

### 7.3 Sizing the stages

The public numbers suggest working defaults:

- **First-stage depth per retriever:** 100-1,000. Azure's text recall defaults to 1,000
  [2]. RankLLM's pointwise examples use 1,000, and its listwise examples use 100 [36].
- **Rerank window:** 50-150. Anthropic retrieved 150 for reranking [12], Azure reranks 50
  [3], and Qdrant and Sentence Transformers use about 50-100 [26][29]. Elastic's default of
  10 is too small for most RAG workloads [6].
- **Chunks passed to the generator:** Anthropic tested 5, 10 and 20 and found 20 best [12].
- **LLM listwise stage:** top 20-100, only when latency and cost allow [35][36].

The latency budget composes additively. At about 1,800 docs/s for MiniLM-L6 [28], a
100-candidate rerank costs tens of milliseconds. A 1.5B mxbai model costs on the order of
a second per batch on an A100 in Mixedbread's benchmark [32]. A sliding-window GPT-class
reranker costs several sequential LLM calls. Each step up must buy measurable quality on
your evaluation set.

### 7.4 Tooling for swapping stages

Because the best reranker is workload-specific, an abstraction layer helps. AnswerDotAI's
`rerankers` library (Clavié, 2024) exposes cross-encoders, ColBERT, T5, LLM pointwise and
listwise rankers (RankGPT and RankLLM), FlashRank ONNX models for CPU, and API rerankers
(Cohere, Jina, Voyage, MixedBread, Pinecone) behind a single `Reranker(...).rank()` call
[42]. Elastic's retriever trees [5], Qdrant's prefetch [26] and Vespa's rank profiles [7]
offer the same flexibility inside the engine, so cascades can be changed per query without
redeploying application code.

### 7.5 Failure modes of cascades

- **Recall leakage.** The final ranking cannot exceed first-stage recall. Measure
  recall@window at each stage boundary, not only NDCG@10 at the end.
- **Window truncation by default settings**, such as Azure's 50 and Elastic's 10 [3][6].
- **Mismatched granularity.** Chunks larger than the reranker's window are truncated, while
  chunks too small lose context. Anthropic's contextualisation and Cohere and Elastic's
  chunk-and-max approach are mitigations [6][12][34].
- **Stage disagreement.** A reranker trained on a different notion of relevance (ArguAna
  counter-arguments, instruction-specific tasks) can undo good first-stage rankings [26][31].
- **Cost creep.** Every stage adds latency, and reranking is "resource-intensive" even in
  managed services [3]. Anthropic explicitly frames reranking as accepting added latency
  and cost for accuracy [12].

---

## 8. Comparison of techniques

| Technique | Representative | Index / state | Query-time cost | Quality signal (public) | Best role |
|---|---|---|---|---|---|
| BM25 | Lucene, Pyserini | Inverted index | Very low | MS MARCO MRR@10 0.187 [13] | Stage-1 recall, identifiers |
| Dense bi-encoder | Any embedding model | ANN (HNSW/IVF) | Low (1 encode + ANN) | Varies by model (Ch. 03) | Stage-1 semantic recall |
| RRF fusion | Azure, Elastic, Vespa | None | Negligible | Strong zero-shot default [1] | Merge heterogeneous lists |
| Normalised linear fusion | Weaviate, OpenSearch | None | Negligible | ~6% recall over RRF on FiQA (internal) [8] | Tuned merge with labels |
| Learned sparse | SPLADE++ | Inverted index | Low-moderate (query encode) | MRR@10 0.383 [14] | Stage-1, replaces or joins BM25 |
| Late interaction (full) | ColBERTv2 + PLAID | Centroids + residuals | Moderate | NDCG@10 parity PLAID vs FastPlaid [22] | High-quality stage 1 |
| Late interaction (rerank) | Qdrant MaxSim, MUVERA | Multi-vector payload | Moderate | ~7x faster than full MV, near parity [27] | Stage 2 over ANN candidates |
| Small cross-encoder | MiniLM-L6 | None | ~1,800 docs/s [28] | MRR@10 0.390 [28] | Stage 2 default |
| LLM-backbone pointwise | Qwen3-Reranker-4B, bge-v2-gemma | None | High | MTEB-R 69.76 [31] | Stage 2, instruction-aware |
| Managed reranker | Cohere 3.5/4.0, Azure semantic | None (API) | API latency | Failure rate 2.9% -> 1.9% [12] | Stage 2 without MLOps |
| LLM listwise | RankGPT, RankZephyr, FIRST | None | Very high (multi-call) | Teacher-quality rankings [35][36] | Stage 3, distillation teacher |
| Reasoning reranker | Rank1 | None | Very high (long generations) | Large gains on BRIGHT [50][51] | Stage 3 for reasoning-heavy queries |

---

## Key takeaways for system designers

1. **Ship hybrid by default.** BM25 plus dense retrieval, fused with RRF at k = 60, is the
   most widely shipped zero-configuration baseline (Azure, Elastic, Vespa, Weaviate) and
   measurably beats dense-only retrieval in Anthropic's study [2][4][7][8][12].
2. **Add a reranker before you change anything else.** In Anthropic's study the reranker
   delivered the largest single step, from 2.9% to 1.9% failure. A six-layer MiniLM
   cross-encoder already more than doubles BM25's MRR@10 on MS MARCO [12][13][28].
3. **Size windows explicitly.** Retrieve 100-1,000 per retriever, rerank 50-150, and send
   5-20 chunks to the generator. Check product defaults: Azure's semantic window is 50, and
   Elastic's reranker window defaults to 10 [3][6][12].
4. **Treat fused scores as ranks, never probabilities.** Calibrate thresholds only on the
   final reranker's scores, per model version, on 30-50 in-domain borderline examples [34].
   Expect drift [3].
5. **Choose learned sparse or late interaction for specific reasons.** Use SPLADE when you
   need inverted-index serving with expansion (and check the non-commercial licence of the
   NAVER checkpoints). Use late interaction when token-level matching matters, and budget
   for quantisation and centroid-pruned engines (PLAID, FastPlaid, WARP) or
   MUVERA-style approximation [14][22][25][27].
6. **Use LLMs as teachers more often than as online rerankers.** Listwise LLM reranking
   costs several sequential calls per query. Distil it into cross-encoders (RankGPT ->
   DeBERTa, RankZephyr, FIRST), or use pointwise LLM-backbone rerankers that keep a
   batchable per-pair score [31][35][36][40].
7. **Expect rerankers to hurt on some tasks.** Qdrant's ArguAna results and Qwen's
   non-monotone FollowIR scores both show that rerankers can regress on tasks unlike their
   training data. Gate each stage with an A/B on your own evaluation set [26][31].
8. **Fine-tune or instruct when the domain is unusual.** A few hundred labelled queries
   fine-tuned Cohere's reranker for measurable gains [10]. Instruction-aware rerankers such
   as Qwen3 can be steered without training [31].
9. **Be sceptical of vendor sparse-retrieval claims.** BM42's correction is a reminder to
   replicate on BEIR-style suites before re-indexing [17][43].
10. **Keep the cascade swappable.** Retriever trees, prefetch pipelines, rank profiles and
    unified reranker APIs let you change stages as models improve each quarter
    [5][7][26][42].

---

## References

1. G. V. Cormack, C. L. A. Clarke, S. Büttcher. "Reciprocal Rank Fusion outperforms
   Condorcet and individual Rank Learning Methods." SIGIR, 2009.
   https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
2. Microsoft. "Relevance scoring in hybrid search using Reciprocal Rank Fusion (RRF)."
   Azure AI Search documentation, 2024-2026.
   https://learn.microsoft.com/azure/search/hybrid-search-ranking
3. Microsoft. "Semantic ranking in Azure AI Search." Azure AI Search documentation,
   2024-2026.
   https://learn.microsoft.com/azure/search/semantic-search-overview
4. Elastic. "Reciprocal rank fusion." Elasticsearch reference, 2025-2026.
   https://github.com/elastic/elasticsearch/blob/main/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion.md
5. Elastic. "Retrievers." Elasticsearch reference, 2025-2026.
   https://github.com/elastic/elasticsearch/blob/main/docs/reference/elasticsearch/rest-apis/retrievers.md
6. Elastic. "Text similarity re-ranker retriever." Elasticsearch reference, 2025-2026.
   https://github.com/elastic/elasticsearch/blob/main/docs/reference/elasticsearch/rest-apis/retrievers/text-similarity-reranker-retriever.md
7. Vespa.ai. "Phased ranking." Vespa documentation, 2024-2026.
   https://github.com/vespa-engine/documentation/blob/master/en/ranking/phased-ranking.html
8. Weaviate (authors listed as "dirk" and "jp"). "Unlocking the Power of Hybrid Search:
   A Deep Dive into Weaviate's Fusion Algorithms." Weaviate blog, 2023.
   https://github.com/weaviate/weaviate-io/blob/main/blog/2023-08-29-hybrid-search-fusion/index.mdx
9. Weaviate (author listed as "laura"). "Using Cross-Encoders as reranker in multistage vector search."
   Weaviate blog, 2022.
   https://github.com/weaviate/weaviate-io/blob/main/blog/2022-08-09-using-cross-encoders-as-reranker-in-multistage-vector-search/index.mdx
10. Weaviate (author listed as "Erika"). "Fine-Tuning Cohere's Reranker." Weaviate blog, 2024.
    https://github.com/weaviate/weaviate-io/blob/main/blog/2024-02-13-fine-tuning-cohere-reranker/index.mdx
11. OpenSearch Project. "Normalization processor." OpenSearch documentation, 2023-2026.
    https://github.com/opensearch-project/documentation-website/blob/main/_search-plugins/search-pipelines/normalization-processor.md
12. Anthropic. "Introducing Contextual Retrieval." 2024.
    https://www.anthropic.com/news/contextual-retrieval
13. Castorini (University of Waterloo). "Pyserini: BM25 Baselines for MS MARCO Passage
    Ranking." Pyserini documentation.
    https://github.com/castorini/pyserini/blob/master/docs/experiments-msmarco-passage.md
14. NAVER Labs Europe (T. Formal, B. Piwowarski, S. Clinchant, C. Lassance et al.).
    "SPLADE" code and model repository, 2021-2026. https://github.com/naver/splade
15. T. Formal, C. Lassance, B. Piwowarski, S. Clinchant. "SPLADE v2: Sparse Lexical and
    Expansion Model for Information Retrieval." arXiv, 2021.
    https://arxiv.org/abs/2109.10086
16. E. Sukhodolskaya (Qdrant). "Modern Sparse Neural Retrieval: From DeepCT to SPLADE++."
    2024.
    https://github.com/qdrant/landing_page/blob/master/qdrant-landing/content/articles/modern-sparse-neural-retrieval.md
17. A. Vasnetsov (Qdrant). "BM42: New Baseline for Hybrid Search." 2024 (with later
    correction).
    https://github.com/qdrant/landing_page/blob/master/qdrant-landing/content/articles/bm42.md
18. O. Khattab, M. Zaharia. "ColBERT: Efficient and Effective Passage Search via
    Contextualized Late Interaction over BERT." SIGIR, 2020.
    https://arxiv.org/abs/2004.12832
19. K. Santhanam, O. Khattab, J. Saad-Falcon, C. Potts, M. Zaharia. "ColBERTv2: Effective
    and Efficient Retrieval via Lightweight Late Interaction." NAACL, 2022.
    https://arxiv.org/abs/2112.01488
20. K. Santhanam, O. Khattab, C. Potts, M. Zaharia. "PLAID: An Efficient Engine for Late
    Interaction Retrieval." CIKM, 2022. https://arxiv.org/abs/2205.09707
21. Stanford FutureData. "ColBERT (ColBERTv2 + PLAID)" repository, 2020-2026.
    https://github.com/stanford-futuredata/ColBERT
22. LightOn. "FastPlaid: high-performance multi-vector search engine." 2025.
    https://github.com/lightonai/fast-plaid
23. LightOn. "PyLate" library for late-interaction models. CIKM, 2025. https://github.com/lightonai/pylate
24. J. Lee, Z. Dai, S. M. K. Duddu, T. Lei, I. Naim, M.-W. Chang, V. Zhao (Google
    DeepMind). "Rethinking the Role of Token Retrieval in Multi-Vector Retrieval (XTR)."
    NeurIPS, 2023. https://github.com/google-deepmind/xtr
25. J. L. Scheerer et al. "WARP: An Efficient Engine for Multi-Vector Retrieval." SIGIR,
    2025. https://github.com/jlscheerer/xtr-warp (paper: https://arxiv.org/abs/2501.17788)
26. K. Łukawski (Qdrant). "Late Interaction Retrieval with Dense Token Embeddings." 2024.
    https://github.com/qdrant/landing_page/blob/master/qdrant-landing/content/articles/late-interaction-models.md
27. K. Łukawski (Qdrant). Article on MUVERA embeddings in Qdrant. 2025.
    https://github.com/qdrant/landing_page/blob/master/qdrant-landing/content/articles/muvera-embeddings.md
28. UKP Lab / Sentence Transformers. "Pretrained Cross-Encoder Models (MS MARCO)."
    https://github.com/UKPLab/sentence-transformers/blob/master/docs/cross_encoder/pretrained_models.md
29. UKP Lab / Sentence Transformers. "Retrieve & Re-Rank."
    https://github.com/UKPLab/sentence-transformers/blob/master/examples/sentence_transformer/applications/retrieve_rerank/README.md
30. BAAI. "FlagEmbedding: BGE Reranker (bge-reranker-v2 family)." 2024.
    https://github.com/FlagOpen/FlagEmbedding/tree/master/research/llm_reranker
31. Qwen Team, Alibaba. "Qwen3 Embedding and Qwen3 Reranker." 2025.
    https://github.com/QwenLM/Qwen3-Embedding (paper: https://arxiv.org/abs/2506.05176)
32. Mixedbread. "mxbai-rerank (v2)." 2025. https://github.com/mixedbread-ai/mxbai-rerank
33. Cohere. "Rerank overview." Cohere documentation, 2024-2026.
    https://github.com/cohere-ai/cohere-developer-experience/blob/main/fern/pages/text-embeddings/reranking/overview.mdx
34. Cohere. "Rerank best practices." Cohere documentation, 2024-2026.
    https://github.com/cohere-ai/cohere-developer-experience/blob/main/fern/pages/text-embeddings/reranking/reranking-best-practices.mdx
35. W. Sun, L. Yan, X. Ma, P. Ren, D. Yin, Z. Ren. "Is ChatGPT Good at Search?
    Investigating Large Language Models as Re-Ranking Agent (RankGPT)." EMNLP, 2023.
    https://github.com/sunnweiwei/RankGPT (paper: https://arxiv.org/abs/2304.09542)
36. S. Sharifymoghaddam et al. (Castorini, University of Waterloo). "RankLLM" reranking
    toolkit (resource paper at SIGIR 2025). https://github.com/castorini/rank_llm
37. R. Pradeep, S. Sharifymoghaddam, J. Lin. "RankVicuna: Zero-Shot Listwise Document
    Reranking with Open-Source Large Language Models." arXiv, 2023.
    https://arxiv.org/abs/2309.15088
38. R. Pradeep, S. Sharifymoghaddam, J. Lin. "RankZephyr: Effective and Robust Zero-Shot
    Listwise Reranking is a Breeze!" arXiv, 2023. https://arxiv.org/abs/2312.02724
39. M. S. Tamber, R. Pradeep, J. Lin. "Scaling Down, LiTting Up: Efficient Zero-Shot
    Listwise Reranking with Seq2seq Encoder-Decoder Models (LiT5)." arXiv, 2023.
    https://arxiv.org/abs/2312.16098
40. R. G. Reddy, J. Doo, Y. Xu, M. A. Sultan, D. Swain, A. Sil, H. Ji. "FIRST: Faster
    Improved Listwise Reranking with Single Token Decoding." arXiv, 2024.
    https://arxiv.org/abs/2406.15657
41. R. Pradeep, R. Nogueira, J. Lin. "The Expando-Mono-Duo Design Pattern for Text Ranking
    with Pretrained Sequence-to-Sequence Models." arXiv, 2021.
    https://arxiv.org/abs/2101.05667
42. B. Clavié (Answer.AI). "rerankers: A Lightweight Python Library to Unify Ranking
    Methods." 2024. https://github.com/AnswerDotAI/rerankers (paper:
    https://arxiv.org/abs/2408.17344)
43. N. Thakur, N. Reimers, A. Rücklé, A. Srivastava, I. Gurevych. "BEIR: A Heterogeneous
    Benchmark for Zero-shot Evaluation of Information Retrieval Models." NeurIPS Datasets
    and Benchmarks, 2021. https://github.com/beir-cellar/beir (paper:
    https://arxiv.org/abs/2104.08663)
44. S. Bruch, S. Gai, A. Ingber. "An Analysis of Fusion Functions for Hybrid Retrieval."
    ACM Transactions on Information Systems, 2023. https://doi.org/10.1145/3596512
    (preprint: https://arxiv.org/abs/2210.11934)
45. C. Lassance, H. Déjean, T. Formal, S. Clinchant. "SPLADE-v3: New baselines for
    SPLADE." arXiv technical report, 2024. https://arxiv.org/abs/2403.06789
46. M. Faysse, H. Sibille, T. Wu, B. Omrani, G. Viaud, C. Hudelot, P. Colombo. "ColPali:
    Efficient Document Retrieval with Vision Language Models." ICLR, 2025.
    https://arxiv.org/abs/2407.01449
47. L. Dhulipala, M. Hadian, R. Jayaram, J. Lee, V. Mirrokni. "MUVERA: Multi-Vector
    Retrieval via Fixed Dimensional Encodings." NeurIPS, 2024.
    https://arxiv.org/abs/2405.19504
48. Jina AI. "jina-reranker-v3: Last but Not Late Interaction for Listwise Document
    Reranking." arXiv, 2025. https://arxiv.org/abs/2509.25085
49. Voyage AI (MongoDB). "rerank-2.5 and rerank-2.5-lite: instruction-following
    rerankers." August 2025.
    https://www.mongodb.com/company/blog/product-release-announcements/rerank-2-5-and-rerank-2-5-lite-instruction-following-rerankers
50. O. Weller, K. Ricci, E. Yang, A. Yates, D. Lawrie, B. Van Durme. "Rank1: Test-Time
    Compute for Reranking in Information Retrieval." COLM, 2025.
    https://github.com/orionw/rank1 (paper: https://arxiv.org/abs/2502.18418)
51. H. Su, H. Yen, M. Xia et al. "BRIGHT: A Realistic and Challenging Benchmark for
    Reasoning-Intensive Retrieval." ICLR, 2025. https://arxiv.org/abs/2407.12883
