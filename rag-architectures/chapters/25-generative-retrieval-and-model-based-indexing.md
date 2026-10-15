# 25. Generative Retrieval and Model-Based Indexing: DSI, NCI, SEAL, TIGER, CorpusBrain and Their Hybrids

## Overview

Every RAG system covered so far in this atlas keeps the corpus *outside* the model. Text goes into an inverted index (BM25, SPLADE) or into a vector index (HNSW, IVF-PQ), and a scorer compares the query against those stored representations. *Generative retrieval* (GR), also called *model-based indexing* or *corpus-in-parameters retrieval*, drops that separation. A sequence-to-sequence model is trained to take a query and **decode the identifier of a relevant document** (a docid), token by token. The "index" is the model's weights, plus a small decoding constraint structure (a prefix trie or an FM-index) that keeps the output to identifiers that actually exist.

The idea took shape in 2020-2022. GENRE generated Wikipedia entity titles under a trie constraint [7]. Google's Differentiable Search Index (DSI) showed that a T5 model could memorize and retrieve whole documents [1]. Microsoft's Neural Corpus Indexer (NCI) won a NeurIPS 2022 Outstanding Paper award for a heavily engineered version of DSI [2][3]. Meta's SEAL replaced opaque docids with *any substring of the corpus*, constrained through an FM-index [4][5]. CorpusBrain pre-trained a generative retriever for the KILT benchmark's eleven knowledge-intensive datasets [8][9]. In recommender systems, Google's TIGER brought the idea into production-style recommendation through *semantic IDs* learned with an RQ-VAE [10]. By 2024-2025 that line had grown into Meta's trillion-parameter-class generative recommenders [24], semantic IDs in YouTube's production ranking models [51], Kuaishou's end-to-end OneRec [52][53] and Snap's open GRID toolkit [25].

Results became more sober once researchers left the 10K-320K document benchmarks. Pradeep et al. scaled GR to the full 8.8M-passage MS MARCO corpus with models up to 11B parameters. They found that synthetic queries matter more than any architectural trick, and that naive scaling of model size does not close the gap to dense retrieval [11]. Follow-up work (RIPOR, PAG, MEVI, GDR) tied docids to dense representations and brought GR roughly to dense-retriever quality on MS MARCO [12][13][17][18]. Related designs such as Ultron mixed URL, title and product-quantization docids [19]. Analytical work showed that much of GR *is* dense retrieval in disguise [15][16]. The 2024-2026 trend is hybridization. GR is now used as a coarse candidate generator or cluster router, as one view among several identifiers, or inside a single LLM that also reranks and answers (Self-Retrieval, CorpusLM). It is paired with dense retrievers and LLM rerankers rather than replacing them [19][20][21][28].

This chapter covers each major design, the docid design space, the index-update problem (GR's main operational weakness), the scaling evidence, and when GR makes sense compared with approximate-nearest-neighbor (ANN) search.

> Research note: the chapter is built from primary-source repositories, proceedings listings and the TOIS survey by Li et al. [6]. Benchmark numbers that could not be read directly from a primary text in this research pass are marked "approximately". In the fact-checking pass, figures for NCI, DSI, RIPOR, PAG, IncDSI, MINDER, DSI-QG, LIMIT, HSTU and DDRO were checked against the official repositories; figures and findings stated only in paper PDFs (notably Pradeep et al. [11] and the 2025 analyses in Sections 7.6-7.7) could not be re-read and should be checked against the cited papers.

---

## 1. The generative retrieval paradigm

### 1.1 Problem statement

A dual-encoder retriever learns `f(q)` and `g(d)` and ranks by `<f(q), g(d)>`. The document side is encoded once offline and stored, and retrieval is maximum inner product search (MIPS). Generative retrieval instead learns the conditional distribution

    P(docid | q) = prod_t P(z_t | z_<t, q)

where `docid = (z_1, ..., z_L)` is a sequence of tokens, and it ranks documents by beam search over that distribution. Two training signals are mixed:

- **Indexing** (memorization): `document text -> docid`. The model learns which identifier belongs to which content.
- **Retrieval** (generalization): `query -> docid`. Supervised pairs come from labeled data or, crucially, from *synthetic queries* produced by a doc2query model.

### 1.2 Canonical architecture

```
                  OFFLINE                                  ONLINE
 +-------------+    +-----------------+          +-----------+
 | corpus D    |--->| docid assignment|          |  query q  |
 +-------------+    | (atomic / k-means|          +-----+-----+
        |           |  / RQ / title /  |                |
        |           |  n-gram / terms) |                v
        |           +--------+--------+       +-------------------+
        |                    |                |  seq2seq / decoder |
        v                    v                |  LM (weights hold  |
 +-------------+    +-----------------+       |  the "index")      |
 | doc2query   |--->| training pairs  |------>+---------+---------+
 | (synthetic  |    | doc->id, q->id  |  train          | logits
 |  queries)   |    +-----------------+                 v
 +-------------+                              +-------------------+
                    +-----------------+       | constrained beam   |
                    | constraint store|------>| search (prefix trie|
                    | trie / FM-index |       | or FM-index)       |
                    +-----------------+       +---------+---------+
                                                        | top-k docids
                                                        v
                                              +-------------------+
                                              | docid -> doc store |
                                              | (+ optional rerank)|
                                              +-------------------+
```

The "index" therefore has three physical parts: (a) model parameters, (b) a constraint structure that lists valid identifiers, and (c) an ordinary key-value store from docid to text for the generator in a RAG pipeline. Only (a) is learned. Changing the corpus means changing (b) and (c) cheaply and (a) expensively. That asymmetry drives most of the deployment discussion later in this chapter.

### 1.3 The docid design space

The survey by Li et al. groups docids into numeric and text-based families [6]. The table below summarizes the main families.

| Docid family | Examples | How built | Pros | Cons |
|---|---|---|---|---|
| Atomic (one token per doc) | DSI atomic [1], Pradeep et al. [11] | New vocab entry per doc | Single decode step, strongest at 8.8M per [11] (approx.) | Output softmax grows with the corpus; equivalent to dense retrieval [15] |
| Naive string | DSI naive [1] | Arbitrary integer as digit tokens | Trivial to build | No semantics, so prefixes carry no meaning |
| Hierarchical semantic (k-means tree) | DSI semantic [1], NCI [2] | Recursive k-means on BERT embeddings | Prefix = topic cluster; good beam pruning | Fixed after training; new docs need re-clustering |
| Residual-quantized (RQ) | TIGER [10], RIPOR [12], MEVI [17], GRID [25] | RQ-VAE / RQ-k-means codebooks | Compact (3-4 tokens), relevance-aware if built from a trained dense encoder | Collisions; codebook drift; quality tied to the embedding |
| Learned / tokenized | GenRet [26], GLEN [27], ASI [6] | Docid learned jointly with retrieval | End-to-end optimized | Training instability; costly reassignment |
| Titles / URLs | GENRE [7], CorpusBrain [8], LLM-URL [6] | Existing unique names | Human-readable; uses LM pretraining | Only for corpora with good unique names |
| Substrings / n-grams | SEAL [4], MINDER [29] | Any span in the corpus via FM-index | No assignment step; adding docs = adding text | Many spans map to many docs; needs score aggregation |
| Term sets | AutoTSG [40], PAG lexical side [13] | Unordered salient terms | Robust to left-to-right pruning errors | Needs set-style decoding |

---

## 2. Differentiable Search Index (DSI) - Google, 2022

### 2.1 Problem and idea

Tay et al. asked whether "a single Transformer" could act as the whole search index, with everything about the corpus encoded in its parameters [1]. DSI frames retrieval as text-to-text generation with T5. Indexing is fine-tuning on `doc tokens -> docid`, and retrieval is fine-tuning on `query -> docid`, both in one multi-task run.

### 2.2 Key techniques

- **Document representation during indexing.** The paper compared several input strategies and found that indexing on the first L tokens of the document (L = 32 or 64) worked well. That is a very lossy view of each document.
- **Docid types.** Unstructured atomic docids, naive string docids, and *semantically structured* docids. The last are built by recursive k-means over BERT embeddings, so that a docid prefix encodes a cluster path.
- **Multi-task ratio.** The ratio of indexing examples to retrieval examples was a sensitive hyperparameter.
- **Model scaling.** T5-Base through T5-XXL (11B). Gains grew with model size, especially for semantic docids.

### 2.3 Results (as reported)

DSI was evaluated on Natural Questions subsets of 10K, 100K and 320K documents (NQ10K/NQ100K/NQ320K). On these scales the paper reported that DSI beat a T5-based dual-encoder baseline and, in the zero-shot setting, beat BM25. An independent re-implementation lists the DSI-Large naive-string target on NQ10K as Hits@1 = 0.347 and Hits@10 = 0.605 (Table 3 of the original paper) [30]. The NCI repository reports DSI (T5-Large) on NQ320k at R@1 35.6 and R@10 62.6 [3].

### 2.4 Failure modes and lessons

- **Memorization is not retrieval.** The re-implementer observed training Hits@1 near 99.99% quickly, while test Hits kept improving slowly. The "indexing" step is effectively memorization, and generalization to real queries is a separate problem [30].
- **Indexing-retrieval gap.** Indexing sees long document text, while retrieval sees short queries. DSI-QG fixed this by *representing each document with generated queries* instead of raw text: the reference setup generates 10 queries per document, using an off-the-shelf T5-large doc2query model for MS MARCO and a cross-lingual mT5-large query generator for XOR QA [22][23]. That became the standard recipe.
- **What DSI actually learns.** An ACL 2023 Findings analysis probed DSI's exclusivity, completeness and relevance-ordering behavior and found weaknesses in its ability to rank relevant documents, motivating multi-task training refinements [32].
- **Reproducibility.** Community re-implementations struggled to match the paper's numbers; the gap was attributed to example ratios, step counts and document text construction [30].
- **Scale.** All results were at 320K documents or fewer, which is far from production corpora.

---

## 3. Neural Corpus Indexer (NCI) - Microsoft Research Asia et al., 2022

### 3.1 Architecture

NCI is "an end-to-end, sequence-to-sequence differentiable document retrieval model" that generates docids from queries [3]. It keeps DSI's framing and adds four engineering pieces:

```
 doc --BERT--> embedding --hierarchical k-means--> semantic docid  (e.g. 3-7-1-4)
 doc --docT5query--> synthetic queries --+
 real queries ---------------------------+--> T5 encoder
                                                    |
                                    PAWA decoder: prefix-aware,
                                    weight-adaptive output layer
                                    (same digit has different
                                    embedding at each position/prefix)
                                                    |
                                   consistency regularization
                                   (two dropout passes must agree)
                                                    |
                                   constrained beam search -> docids
```

1. **Semantic hierarchical docids** from BERT embeddings plus hierarchical k-means. The repository also builds several docid sets for ensembling [3].
2. **Query generation** with a docTTTTTquery checkpoint, which the authors say helps most on long-tail queries [3].
3. **Prefix-Aware Weight-Adaptive (PAWA) decoder.** The meaning of a token like "7" depends on its position and on the prefix before it, so the output projection is made conditional on that prefix [2].
4. **Consistency-based regularization** between forward passes, to reduce overfitting to the indexing examples.

### 3.2 Results

The NCI repository reports this comparison (on NQ320k, the main benchmark of the paper) [3]:

| Model | R@1 | R@10 | R@100 | MRR@100 |
|---|---|---|---|---|
| NCI w/ qg-ft (Ensemble) | 72.78 | 91.76 | 96.22 | 80.12 |
| NCI (Large) | 66.23 | 85.27 | 92.49 | 73.37 |
| NCI (Base) | 65.86 | 85.20 | 92.42 | 73.12 |
| SEAL (Large) | 59.93 | 81.24 | 90.93 | 67.70 |
| ANCE (MaxP) | 52.63 | 80.38 | 91.31 | 62.84 |
| DSI (T5-Large) | 35.60 | 62.60 | -- | -- |
| BM25 + DocT5Query | 35.43 | 61.83 | 76.92 | 44.47 |

The paper also reports large relative recall@1 gains on TriviaQA [2].

### 3.3 Lessons

NCI's headline numbers made GR look clearly better than dense retrieval, but the comparison has three caveats. First, NQ320k is small and its document set is closely matched to its queries. Second, NCI trains on many synthetic queries per document, which dense baselines like ANCE did not get; later analyses identified query generation as the main source of gains [11]. Third, the best row is an ensemble. The lasting lesson is that **doc2query augmentation is the single most important ingredient of a GR system**, and architectural changes to the decoder are second-order.

---

## 4. SEAL and FM-index constrained decoding - Meta AI, 2022

### 4.1 Problem

Assigned docids, whether atomic or clustered, have no meaning the language model already knows, so the model must learn a new "language" of identifiers. Titles (GENRE) only work for entity-like corpora. SEAL (Search Engines with Autoregressive LMs) asks: what if *every n-gram in the corpus* is a valid identifier [4][5]?

### 4.2 Architecture

```
 query --> BART-large --> constrained decoding over FM-index
                               |
             FM-index (compressed suffix array over all passages):
             - next-token lookup: which tokens can follow prefix p?
             - occurrence count: how many times does p appear?
             - locate: which documents contain p?
                               |
               generated n-grams {g1, g2, ...} with LM scores
                               |
          document score = aggregate over n-grams it contains
          (LM prob x corpus-frequency-based weight; intersective
           scoring combines multiple non-overlapping n-grams)
                               |
                        ranked passages
```

The FM-index (Ferragina-Manzini) is a compressed suffix array. SEAL uses it as the constraint structure for decoding, as the retrieval structure, and as full-text storage, all at once. It can be seen as "a trie over every substring of every document", which lets decoding start anywhere in a document, not only at a title prefix [5]. The implementation builds on sdsl-lite [5].

### 4.3 Key techniques

- **Unconstrained start position.** Unlike a title trie, any span is reachable.
- **Frequency-aware scoring.** An n-gram that is likely under the LM but rare in the corpus is more informative, in the spirit of tf-idf.
- **Intersective scoring.** A document's score aggregates several distinct generated n-grams, so one lucky phrase does not dominate.

### 4.4 Results and costs

SEAL was evaluated on NQ (over the DPR Wikipedia passage collection) and on KILT, with pre-built FM-indices released for both [5]. On NQ320k it reaches R@1 of about 60 in NCI's table [3]. The paper reported that SEAL matched or exceeded strong retrievers on several KILT tasks, with an index much smaller than a dense passage index. Treat the exact memory ratio as approximate and see [4].

### 4.5 Descendants

- **MINDER** (Li et al., ACL 2023) uses *multiview* identifiers: titles, substrings, and pseudo-queries, all decoded against FM-index structures for Wikipedia (psgs_w100) and MS MARCO. Training took about 4 days on 8 V100s for NQ and about 13 hours for MS MARCO [29].
- **LTRGR** (AAAI 2024) continues training from MINDER with a learning-to-rank loss on passages, not just on identifier likelihood [14][31].
- **DGR** (2024) distills ranking knowledge from a stronger teacher, such as a cross-encoder reranker, into the generative retriever [6][33]. This is an early sign of the GR plus reranker coupling covered in Section 9.3.

### 4.6 Lessons

SEAL is the most *operationally friendly* GR design. Adding a document means adding its text to the FM-index (which itself is rebuilt or merged offline). The model needs no new identifier vocabulary, because it already generates natural-language n-grams. It also behaves more like a learned lexical retriever than a memorizing one, which makes it more robust to corpus change. The cost is decoding latency, since there are FM-index lookups at every beam step, and weaker semantic matching when the query's wording does not appear in the passage.

---

## 5. TIGER and semantic IDs - Google, 2023

### 5.1 Problem

Sequential recommenders traditionally embed every item ID in a giant table and score candidates with ANN over item embeddings. That has three problems: cold-start for new items, a table that grows linearly with the catalog, and no sharing between similar items. Rajput et al., in "Recommender Systems with Generative Retrieval", apply the GR paradigm to recommendation [10].

### 5.2 Architecture

```
  item content (title, brand, category, description)
            |
     sentence encoder (e.g. Sentence-T5)  -> dense item vector x
            |
     RQ-VAE: level 1 codebook -> c1, residual r1 = x - e(c1)
             level 2 codebook -> c2, residual r2 = r1 - e(c2)
             level 3 codebook -> c3
             (+ extra token to disambiguate collisions)
            |
     semantic ID = (c1, c2, c3, c4)   e.g. (12, 201, 7, 0)
            |
  user history: [SID(item_1)] [SID(item_2)] ... [SID(item_n)]
            |
     Transformer encoder-decoder  --> autoregressively decode SID(item_n+1)
            |
     constrained to valid SIDs (prefix trie)  -> next items
```

The semantic ID is coarse-to-fine: items that share `c1` are broadly similar, and items that share `(c1, c2)` are closer still. Collisions, where two items get the same code tuple, are resolved with an extra disambiguation token. Generation is restricted to a prefix trie of valid IDs [10][34].

### 5.3 Why it matters for RAG architects

TIGER showed three things that transfer back to document retrieval. First, **content-derived IDs allow cold-start**: a new item gets an ID from its content embedding, with no retraining needed to *name* it, although the model has never been trained to *generate* that ID. Second, ID tables become small: a few codebooks of a few hundred entries each instead of one row per item. Third, decoding *invalid* or *new* IDs can be used deliberately for diversity and exploration.

### 5.4 2024-2026 industrial descendants

- **Meta HSTU / Generative Recommenders (ICML 2024)** recasts DLRM ranking and retrieval as sequential transduction and reports scaling-law behavior "in deployed, billion-user scale recommendation systems". Its HSTU and M-FALCON components are reported to speed up training and inference by 10x-1000x [24].
- **Snap GRID (CIKM 2025)** is an open-source "practitioner's handbook" pipeline. It covers LLM-based item embedding, semantic-ID learning with RQ-k-means, RQ-VAE or RVQ, and TIGER-style generation [25].
- **Open re-implementations** such as RQ-VAE-Recommender add collision disambiguation, corpus-prefix-constrained generation and rotation-trick quantizer gradients. They have been used for later work on collision-aware semantic-ID learning "at industrial scale" [34].

### 5.5 Semantic IDs in production ranking and end-to-end generative recommenders

Two 2024-2025 developments show semantic IDs moving from a research retrieval trick into production stacks:

- **Semantic IDs as feature IDs (YouTube, RecSys 2024).** Singh et al. replaced random, hashed video IDs in a production-scale ranking model with RQ-VAE semantic IDs derived from content embeddings, and tokenized them with n-gram and SentencePiece-style schemes. The reported effect is better generalization to new and long-tail items without losing memorization of popular ones [51]. Here the semantic ID is not decoded at all; it is a compact, content-sharing embedding key, which is a lesson for document retrieval too.
- **End-to-end generative recommendation (Kuaishou OneRec, 2025).** OneRec replaces the retrieve-then-rank cascade with a single encoder-decoder generator that emits sessions of semantic-ID items, and aligns it with a reward model through DPO-style iterative preference optimization. The authors report deployment on Kuaishou's main short-video feed with an online watch-time gain on the order of 1-2% (approximately; see [52]). The later OpenOneRec release provides 1.7B and 8B "OneRec-Foundation" models on a Qwen3 backbone with hierarchical-quantization "itemic tokens" [53].

OneRec is the clearest case of a generative model *absorbing* the candidate-retrieval stage in production. It also makes plain that it works because the "corpus" is a catalog with dense interaction data, not open text.

Recommendation is where GR has had its most convincing production success. Catalogs are large but structured, queries are user histories rather than free text, and the generative model already exists for ranking.

---

## 6. CorpusBrain - ICT/CAS, 2022

### 6.1 Problem

KILT bundles eleven datasets spanning five knowledge-intensive task families (fact checking, entity linking, slot filling, open-domain QA, dialogue) over one Wikipedia snapshot (the 2019-08-01 dump, about 5M pages). A pipeline retriever needs task-specific training data. CorpusBrain asks whether one *pre-trained* generative retriever, with no explicit index, can serve all of them [8][9].

### 6.2 Architecture and pre-training

CorpusBrain uses the BART-Large architecture. Its checkpoint is a drop-in replacement for BART's, and the docids are Wikipedia page titles constrained by a KILT title trie, as in GENRE [9]. The idea is to pre-train on self-supervised `pseudo-query -> title` pairs built from Wikipedia structure. The paper describes three pre-training tasks: Inner Sentence Selection (a sentence from a page acts as a query for that page), Lead Paragraph Selection (lead paragraphs as queries) and Hyperlink Identifier Prediction (anchor contexts as queries for the linked page). Only after this does fine-tuning on downstream KILT tasks begin [8].

```
  Wikipedia page P  --sample sentence / lead para / anchor ctx-->  pseudo-query
                                                                       |
                                   BART-large  <---- pretrain: pseudo-query -> title(P)
                                       |
                    fine-tune (few or many labels) on KILT tasks
                                       |
          query --> constrained beam search over KILT title trie --> pages
```

### 6.3 Results and follow-ups

The paper reports strong results on KILT page-level retrieval, including in low-resource fine-tuning settings, compared with both pipeline retrievers and GENRE [8]. **CorpusBrain++** (2024) adds continual generative pre-training, so that the model can absorb new Wikipedia documents without full retraining [35]. **CorpusLM** (SIGIR 2024) merges generative retrieval, closed-book generation and RAG into one LM [36].

### 6.4 Lessons

Title-based docids plus structure-derived pseudo-queries are a cheap and effective form of *retrieval pre-training*. The approach depends on the corpus having clean unique names and dense hyperlink structure. Wikipedia has both; most enterprise corpora have neither.

---

## 7. Scaling limits: does corpus-in-parameters scale?

### 7.1 Pradeep et al., EMNLP 2023

"How Does Generative Retrieval Scale to Millions of Passages?" (Google Research and Waterloo) is the key empirical study [11]. It ran GR on MS MARCO passages from 100K up to the full 8.8M, with T5 models up to 11B parameters. Its findings, paraphrased:

1. **Synthetic queries are the critical ingredient.** Training on doc2query-generated queries as the document representation mattered more than any other technique tested. Many proposed architecture tweaks (PAWA-style decoders, prefix-aware tricks) gave little benefit once synthetic queries were used.
2. **Docid choice changes with scale.** At 8.8M passages, simple atomic docids did best. Semantic and naive string IDs fell behind, but atomic IDs need an output layer with one row per passage, which makes the model behave like a dense retriever with a learned document matrix.
3. **Naive parameter scaling stalls.** At full scale, larger models did not reliably help. The study reported that moving from T5-XL (3B) to T5-XXL (11B) did not reliably improve full-scale results (approximately; see [11]). Later work with decoder-only LLM backbones and n-gram identifiers reports cleaner scaling behavior (Section 7.6).
4. **A gap to dense retrieval remained.** At 8.8M, the best GR MRR@10 sat in the mid-20s (approximately). That is above BM25 (about 18.4 MRR@10 on MS MARCO dev) but well below strong dual encoders of the time (low-to-high 30s).

### 7.2 Why memorization does not scale gracefully

Several mechanisms explain the plateau:

- **Capacity versus corpus size.** Every document must be stored somewhere in the weights. With atomic IDs, the output embedding matrix *is* a dense index, of size N x d. With short semantic IDs, many documents share prefixes, and the model must separate near-duplicates using only a few tokens.
- **Beam search pruning.** Left-to-right decoding commits to a prefix early. If the right document sits under a prefix that falls out of the beam at step 1, it can never be recovered. PAG's "planning ahead" was designed to fix exactly this [13].
- **Generalization to unseen phrasing.** Out-of-distribution analyses found GR models brittle under query variation and corpus shift [37].
- **Docid-relevance mismatch.** Clustering by content similarity is not the same as clustering by "would be retrieved by the same queries". RIPOR therefore builds docids from a *relevance-trained* dense encoder [12].

### 7.3 Closing the gap: RIPOR and PAG (UMass, 2024)

**RIPOR** (WWW 2024) targets MS MARCO's 8.8M passages [12][38] in three phases:

```
 phase 1: train T5 as a dense encoder (relevance-based)
 phase 2: residual-quantize its doc vectors -> RQ docids (relevance-aware prefixes)
 phase 3: seq2seq training with prefix-oriented ranking loss:
          at every prefix length, rank relevant docid prefixes above negatives
          mined from constrained beam search, + dense-retriever training data
```

It reports a 30.5% MRR improvement on MS MARCO Dev over the previous best GR model and parity with popular dense retrievers. It was also evaluated on TREC DL 2019 and 2020 [38]. Treat the absolute MRR@10 as approximately low-30s; see [12].

**PAG** (SIGIR 2024) adds a *set-based lexical identifier* (a bag of salient terms) alongside the sequential RQ identifier. It first scores documents with the set-based identifier, using a simultaneous, non-autoregressive pass, and then uses those scores to guide the autoregressive decoding of the sequential ID. This plan-ahead step protects the right prefixes from early pruning. The authors report about 15.6% MRR improvement over the prior state of the art on MS MARCO and a 22x query-latency speedup. Training used 8x A100-40GB and inference a single A100-80GB [13][39].

### 7.4 Generative retrieval *is* (mostly) dense retrieval

Two analyses changed how the field thinks about GR:

- **"Generative Retrieval as Dense Retrieval"** (Nguyen and Yates, 2023) shows that with atomic docids, GR's final softmax over docid embeddings is exactly a dense retriever in which document embeddings are free parameters. The proposed "tied-atomic" variant makes this explicit [15].
- **"Generative Retrieval as Multi-Vector Dense Retrieval"** (SIGIR 2024) generalizes the argument: decoding a multi-token docid is similar to multi-vector late interaction between the query's decoder states and docid token embeddings [16].

The practical conclusion is that **GR does not escape the representational limits of embedding retrieval for free**. It moves the document vectors into the output layer and replaces MIPS with beam search. The 2025 LIMIT work from Google DeepMind shows that single-vector embedding retrieval has dimension-dependent limits: for any dimension d, some top-k combinations are unrepresentable. Even state-of-the-art embedders struggle on the LIMIT dataset (50K documents, 1,000 queries) [41]. GR designs with expressive multi-token IDs or lexical set IDs are one proposed way around such limits. Whether they actually escape them at scale is an open research question.

### 7.5 Generative retrieval with LLM backbones (2024-2026)

Moving from T5/BART to decoder-only LLMs (Llama-class 7B and larger) changes the trade-offs:

- **Self-Retrieval** (NeurIPS 2024) uses one LLM, Llama-2-7B in the reference code, to internalize the corpus through self-supervised training. It *generates the passage text itself*, constrained by a trie over the corpus, and then uses the same model's logits to judge and rerank relevance [20][21]. Indexing, retrieval and reranking all live in one parameter set.
- **LLM-URL** prompts LLMs to generate URLs as identifiers without any corpus-specific training [6].
- **Few-shot indexing** (2024) has an LLM generate docids for each document from a few examples, with no training-based memorization [42].
- **LLMGR at Alipay** (SIGIR 2024) is an industrial deployment of LLM-based generative retrieval for mini-app/service search [43].

LLM backbones inherit general language understanding, which helps generalization. They make memorization much more expensive, though: 7B+ parameter fine-tuning per corpus, plus beam search at LLM decode speed.

### 7.6 Scaling laws revisited (2025)

Pradeep et al.'s negative scaling result used T5 with assigned docids. Cai et al. ("Exploring Training and Inference Scaling Laws in Generative Retrieval", SIGIR 2025) revisited the question with decoder-only LLaMA-family backbones alongside T5, and with n-gram (SEAL/MINDER-style) identifiers. They report that retrieval quality follows smooth power-law trends in model size, training data and inference compute, that LLaMA backbones scale better than T5, and that n-gram identifiers scale more favorably than assigned numeric IDs [54]. Their experiments were on NQ- and MS MARCO-derived benchmarks rather than the full 8.8M collection (approximately; see [54]), so they refine rather than overturn the earlier finding. The lesson for designers is that the backbone and the identifier type, not raw parameter count, decide whether scaling helps.

### 7.7 Theoretical limits of constrained decoding (2025)

Wu et al. ("Constrained Auto-Regressive Decoding Constrains Generative Retrieval", SIGIR 2025) analyze the corpus-specific constraint itself. Their argument is that a model trained without knowledge of a particular corpus's future constraints incurs an irreducible error when decoding is restricted to that corpus's valid docids, and that greedy or beam search can fail to recover the true top-k under realistic relevance distributions [55]. This gives a formal grounding for the "early beam pruning" failure mode (Section 12) and for designs, such as PAG's plan-ahead scoring [13] and coarse-routing hybrids [17][18], that avoid relying on left-to-right search over a deep identifier tree.

---

## 8. Index updates: the operational Achilles heel

For a vector database, adding a document costs one encoder forward pass and one graph insertion (see Chapter 4). Deletion is a tombstone. For GR, the model has never been trained to emit the new docid, so a new document is unreachable until the weights change.

### 8.1 What happens when you add documents

| Approach | Mechanism | Reported cost/behavior |
|---|---|---|
| Full retrain | Re-index the whole corpus | Gold standard; hours to days on multi-GPU (e.g. MINDER NQ about 4 days on 8 V100) [29] |
| Naive continual fine-tune | Train only on new docs | Catastrophic forgetting of old docs [44] |
| DSI++ (2022) | Continual indexing with sharpness-aware minimization and generative memory replay of pseudo-queries | Reduces forgetting of old docs while indexing new ones [44] |
| IncDSI (ICML 2023) | Constrained optimization: add a new docid vector so that the doc's queries map to it, with minimal changes to the network | About 20-50 ms per document; competitive with full retraining; "orders of magnitude faster" [45] |
| PromptDSI (2024) | Rehearsal-free, prompt-based instance-wise incremental learning | Avoids storing old data [46] |
| CLEVER (CIKM 2023) | Incremental product quantization for new docids plus memory-augmented continual learning | Designed for dynamic corpora [6] |
| CorpusBrain++ (2024) | Continual generative pre-training with task-specific adapters | Keeps KILT ability while adding docs [35] |
| SEAL/MINDER-style | Add text to FM-index; model already decodes n-grams | No new vocabulary; FM-index rebuild or merge offline [4][29] |

IncDSI is instructive. It works by treating the *atomic docid output layer* as a matrix of document vectors and solving for a new row, which is just how you would insert into a dense index. The cheapest GR update paths again turn out to be the ones that look most like dense retrieval [15].

### 8.2 Empirical practicality on dynamic corpora

"Exploring the Practicality of Generative Retrieval on Dynamic Corpora" (Yoon et al., EMNLP 2024) compared GR with dual encoders (DE) on a StreamingQA-derived benchmark in which new time-stamped documents arrive in stages. Its finding cuts against the folklore: in that setup, GR adapted to newly added knowledge *better* than DE and was reported to be cheaper in inference FLOPs, indexing time and storage, while DE remained more robust in other respects [47]. Two caveats keep this from being a general verdict. The corpus was far below MS MARCO scale, and "updating" still meant continued training rather than an insert. Separately, out-of-distribution robustness studies found GR more sensitive to query variation and corpus shift than dense retrievers [37]. The practical reading is that GR's update problem is a matter of cost and engineering, not an absolute barrier, and that it must be measured on the target corpus.

### 8.3 Deletion and right-to-be-forgotten

Removing a document from the *constraint structure* (trie or FM-index) stops it from being returned, which is cheap. Removing it from the *weights* is unsolved in general. Memorized content may still leak through generation, and that matters most for designs like Self-Retrieval that generate passage text directly. Teams with compliance-driven deletion requirements should treat this as a hard blocker for passage-generating GR.

---

## 9. Hybrid architectures: GR plus dense, GR plus rerankers

Pure GR rarely wins on large, changing corpora, so most current designs use it as one component.

### 9.1 GR as a coarse router over a vector index

**MEVI (Model-enhanced Vector Index, NeurIPS 2023)** uses residual-quantization docids. A seq2seq model decodes the first few RQ codes, which name clusters, and dense vector search then runs only inside the selected clusters [17][48]. **GDR (Generative Dense Retrieval, 2024)** takes the same view and argues that "memory can be a burden": the model memorizes only coarse cluster IDs, and fine-grained discrimination goes to a dense retriever inside each cluster [18].

```
query --> small seq2seq --> beam over coarse codes (c1,c2) --> top clusters
                                                              |
query --> dense encoder ------------------------------------> ANN within clusters
                                                              |
                                                     merged candidates --> reranker
```

This uses GR where it is strong (semantic routing, a small memorized vocabulary) and dense retrieval where GR is weak (millions of fine-grained distinctions, cheap updates).

### 9.2 Multiview and ensemble GR

MINDER (titles, substrings, pseudo-queries) [29], PAG (lexical set plus sequential codes) [13] and NCI ensembles [3] all use several views of a document to hedge against beam-search pruning errors.

### 9.3 GR plus LLM rerankers (2024-2026)

The current production-shaped pattern treats GR as a *candidate generator* whose output goes to a cross-encoder or LLM reranker (see Chapter 5):

- **Re3val** (2024) combines generative retrieval with reinforcement learning and a reranking stage, using the reranker to correct GR's page-level errors [49].
- **DGR** distills a reranker into the generative retriever to sharpen its ranking [33].
- **Self-Retrieval** makes the *same* LLM act as retriever and reranker, scoring its own generated passages [20].
- **CorpusLM** and **UniGen** unify retrieval and answer generation in one LLM, so that the retrieval step feeds the generator without a separate index [36][6].
- **Multi-graded GR (GR2, NeurIPS 2024)** moves GR training from binary relevance to graded relevance, aligning it with the objectives rerankers optimize [50].

```
                 +--> dense ANN (bi-encoder) -----------+
query -----------+                                       +--> fuse (RRF) --> LLM reranker
                 +--> generative retriever (beam k) ----+          |
                 |    (titles / RQ codes / n-grams)                v
                 +--> BM25 / SPLADE --------------------+     top-n ctx --> LLM answer
```

In this layout GR helps most when its errors are *uncorrelated* with the dense retriever's. Title- and n-gram-based GR (SEAL, MINDER) often has a lexical-entity bias that complements embeddings on entity-heavy queries.

### 9.4 Direct relevance optimization (2025)

Ranking-aware GR training (LTRGR, RIPOR, GR2) typically relies on extra loss machinery or reinforcement learning. **DDRO** (Mekonnen, Tang and de Rijke, SIGIR 2025) borrows from LLM alignment instead. After supervised docid generation training, it applies a DPO-inspired pairwise objective that raises the likelihood of relevant docids over non-relevant ones relative to a frozen reference model, with no reward model or RL loop [56]. The official repository reports MRR@10 of about 50 (title-URL docids) on its MS MARCO document setting and about 55.5 (PQ docids) on NQ320K [56]. The design matters for the reranker coupling above. The same preference data used to train or distill an LLM reranker can be reused directly to align the generative retriever, which narrows the gap between the candidate generator and the reranker it feeds.

---

## 10. Benchmarks: MS MARCO and NQ in context

Benchmark scale differs greatly across papers, which explains much of the conflicting evidence. The table below collects reported or approximate figures. Always check the original paper before quoting a number.

| System | Benchmark / corpus size | Metric | Reported value | Notes |
|---|---|---|---|---|
| BM25 + DocT5Query | NQ320k (~320K docs) | R@1 / R@10 | 35.43 / 61.83 | from [3] |
| ANCE (MaxP) dense | NQ320k | R@1 / R@10 | 52.63 / 80.38 | from [3] |
| DSI T5-Large | NQ320k | R@1 / R@10 | 35.6 / 62.6 | from [3] |
| DSI-Large naive IDs | NQ10K | Hits@1 / Hits@10 | 0.347 / 0.605 | paper Table 3, via [30] |
| SEAL Large | NQ320k | R@1 / R@10 | 59.93 / 81.24 | from [3] |
| NCI Large | NQ320k | R@1 / R@10 | 66.23 / 85.27 | from [3] |
| NCI ensemble + qg-ft | NQ320k | R@1 / R@10 | 72.78 / 91.76 | from [3] |
| Best GR, Pradeep et al. | MS MARCO 8.8M | MRR@10 | mid-20s (approx.) | T5 up to 11B [11] |
| BM25 | MS MARCO 8.8M dev | MRR@10 | ~18.4 (approx.) | standard baseline |
| RIPOR | MS MARCO 8.8M dev, TREC DL19/20 | MRR@10 | +30.5% vs prior GR; parity with dense (approx. low-30s) | [12][38] |
| PAG | MS MARCO 8.8M dev, TREC DL | MRR@10 | +15.6% vs prior SOTA GR; 22x latency speedup | [13][39] |
| DDRO (PQ docids) | NQ320k | MRR@10 / R@10 | 55.51 / 67.31 | repository [56] |
| DDRO (title-URL docids) | MS MARCO document subset (paper setting) | MRR@10 / R@10 | 50.07 / 74.01 | repository [56]; not the 8.8M passage task |

Patterns worth noting:

1. **At about 320K documents, GR beats older dense baselines.** NCI's R@1 of 66 versus ANCE's 53 is a large margin, but the dense baselines did not use the same synthetic-query augmentation.
2. **At 8.8M passages, vanilla GR trails dense retrieval.** Only relevance-initialized, ranking-trained GR (RIPOR, PAG) reaches dense-retriever quality, and it does so by borrowing a dense encoder for docid construction.
3. **KILT and entity-centric tasks favor title GR** (GENRE, CorpusBrain), because the identifiers are meaningful to the pretrained LM.

---

## 11. When does generative retrieval beat ANN?

### 11.1 Decision table

| Factor | Favors generative retrieval | Favors dense/sparse + ANN |
|---|---|---|
| Corpus size | Up to ~1M docs, or GR used only as a coarse router | Millions to billions of chunks |
| Update rate | Static or slowly changing (Wikipedia snapshot, catalog with batch refresh) | Continuous ingest, frequent deletes |
| Identifier semantics | Corpus has natural unique names (entities, titles, SKUs, URLs) | Arbitrary chunks of enterprise docs |
| Query-doc relation | Many training or synthetic queries per doc; repeated query distribution | Long tail of unseen queries, zero-shot domains |
| Serving stack | Already serving a seq2seq/LLM; want one model | Mature vector DB, cheap CPU ANN |
| Memory | Want to drop an N x d vector store (title/n-gram GR) | Vector compression (PQ) already cheap enough |
| Compliance | No hard-deletion requirement | Right-to-be-forgotten, ACL filtering per user |
| Latency budget | Short IDs (3-4 tokens) with small beam; PAG-style plan-ahead | Single-digit-ms ANN at high QPS |
| Domain | Recommendation (user-history -> item semantic ID) | Open-domain text RAG |

### 11.2 Latency and cost

- **ANN** cost: one encoder pass plus a sub-linear graph search, often a few milliseconds on CPU. It scales to billions with sharding and quantization.
- **GR** cost: an encoder pass plus L decoder steps x beam width B, each a full decoder forward pass, plus constraint lookups (trie or FM-index) at every step. With L = 3-8 and B = 10-100 this is much heavier than ANN, and with LLM backbones it is heavier still. PAG's 22x speedup [13] shows how much decoding strategy matters.
- **Training/indexing** cost: dense indexing is embarrassingly parallel inference. GR indexing is *training*, over synthetic queries (often 10+ per document [22]), repeated for every major corpus change.
- **Memory**: title or n-gram GR can avoid storing per-document vectors, needing only model weights plus a compressed FM-index or trie. Atomic-ID GR stores an N x d output matrix, the same as a dense index.

### 11.3 Where GR clearly wins today

1. **Generative recommendation with semantic IDs.** TIGER, HSTU and GRID show the most robust real-world gains, because the generative model already exists and IDs are content-derived and compact [10][24][25].
2. **Entity linking and Wikipedia-page retrieval** with title IDs (GENRE, CorpusBrain), where the identifier space is meaningful and stable [7][8].
3. **Coarse semantic routing inside a hybrid** (MEVI, GDR), where GR picks clusters and ANN does the rest [17][18].
4. **Single-model systems** where an LLM retrieves from a small, stable corpus and reranks in the same pass (Self-Retrieval) [20].

---

## 12. Failure modes catalog

| Failure mode | Symptom | Mitigation |
|---|---|---|
| Indexing-retrieval gap | Model memorizes docs but fails on real queries | Represent docs by synthetic queries (DSI-QG) [22] |
| Early beam pruning | Correct doc's prefix drops out at step 1 | Larger beams, plan-ahead (PAG) [13], multiview IDs [29]; see the formal analysis in [55] |
| Docid collision | Distinct docs share code tuple | Extra disambiguation token [10][34] |
| Content-vs-relevance mismatch | Semantically similar but irrelevant docs share prefixes | Relevance-based RQ docids (RIPOR) [12] |
| Catastrophic forgetting | New docs indexed, old docs no longer retrieved | DSI++, IncDSI, PromptDSI, CLEVER [44][45][46] |
| Hallucinated IDs | Decoded ID not in corpus | Constrained decoding (trie, FM-index) [5][7] |
| OOD brittleness | Accuracy drops under paraphrase or new domains | Hybrid with dense/BM25; more diverse synthetic queries [37] |
| Undeletable memory | Removed docs still influence generation | Constraint removal plus periodic retraining; avoid passage-generating GR for regulated data |
| Benchmark inflation | Big wins on NQ320k vanish at 8.8M | Evaluate at target corpus size [11] |

---

## Key takeaways for system designers

1. **Treat generative retrieval as a component, not a replacement.** At MS MARCO scale (8.8M passages), vanilla GR trails dual encoders. Only GR that borrows a relevance-trained dense encoder for its docids (RIPOR, PAG) reaches parity [11][12][13].
2. **Synthetic queries are the real index.** Across DSI-QG, NCI and Pradeep et al., doc2query augmentation is the dominant factor. Budget for generating and training on 10+ queries per document [11][22].
3. **Pick the docid by your corpus.** Use titles or URLs for entity-like corpora, FM-index n-grams (SEAL/MINDER) for text corpora that change, relevance-based RQ codes for MS MARCO-style passage ranking, and content-derived semantic IDs for recommendation catalogs.
4. **Price in index updates up front.** Every corpus change is a training event. IncDSI's 20-50 ms per document is the best published incremental path, and it works by acting like a dense-index insert [45]. If you ingest continuously, prefer dense or sparse retrieval, or keep GR to a coarse, rarely changing layer.
5. **Atomic-ID GR is dense retrieval with extra steps.** Its output layer is an N x d document matrix [15]. If you would end up there, a vector index is simpler and cheaper to update.
6. **Constrained decoding is mandatory and is part of the index.** The trie or FM-index guarantees valid outputs, but it adds per-step lookups to latency and must be versioned with the corpus [5][7].
7. **Use GR where its errors are uncorrelated with embeddings.** Title and n-gram GR adds lexical and entity precision to a dense plus BM25 ensemble. Fuse candidates, then let a cross-encoder or LLM reranker decide [29][33][49].
8. **Recommendation is the proven production win.** Semantic-ID generative recommenders (TIGER, HSTU, YouTube semantic IDs, OneRec, GRID) are the clearest evidence that GR works at scale [10][24][25][51][52].
9. **LLM-native GR trades generality for cost.** Self-Retrieval-style systems collapse index, retriever and reranker into one 7B+ model. That is attractive for small, stable, non-regulated corpora, and hard to justify where deletion, ACLs or rapid ingest matter [20].
10. **Always evaluate at deployment scale.** Results on 10K-320K documents do not predict behavior at millions of documents [11]. Newer scaling-law results with LLM backbones and n-gram IDs [54] are encouraging but were not run on the full 8.8M collection.
11. **Align the retriever with the reranker's preferences.** DPO-style objectives (DDRO) and reranker distillation (DGR) let one set of preference data train both stages [33][56].

---

## References

1. Yi Tay et al. (Google Research). "Transformer Memory as a Differentiable Search Index." NeurIPS 2022. http://papers.nips.cc/paper_files/paper/2022/hash/892840a6123b5ec99ebaab8be1530fba-Abstract-Conference.html
2. Yujing Wang et al. (Microsoft Research et al.). "A Neural Corpus Indexer for Document Retrieval." NeurIPS 2022 (Outstanding Paper). https://proceedings.neurips.cc/paper_files/paper/2022/file/a46156bd3579c3b268108ea6aca71d13-Paper-Conference.pdf
3. NCI authors. "Neural-Corpus-Indexer-NCI" official repository (README results table). 2022. https://github.com/solidsea98/Neural-Corpus-Indexer-NCI
4. Michele Bevilacqua, Giuseppe Ottaviano, Patrick Lewis, Wen-tau Yih, Sebastian Riedel, Fabio Petroni (Meta AI). "Autoregressive Search Engines: Generating Substrings as Document Identifiers." NeurIPS 2022. http://papers.nips.cc/paper_files/paper/2022/hash/cd88d62a2063fdaf7ce6f9068fb15dcd-Abstract-Conference.html
5. Meta AI (facebookresearch). "SEAL: Search Engines with Autoregressive LMs" repository. 2022. https://github.com/facebookresearch/SEAL
6. Xiaoxi Li, Jiajie Jin, Yujia Zhou, Yuyao Zhang, Peitian Zhang, Yutao Zhu, Zhicheng Dou (Renmin University). "From Matching to Generation: A Survey on Generative Information Retrieval." arXiv 2024 / ACM TOIS 2025. https://arxiv.org/abs/2404.14851 (paper list: https://github.com/RUC-NLPIR/GenIR-Survey)
7. Nicola De Cao et al. (Meta AI). "Autoregressive Entity Retrieval (GENRE)." ICLR 2021; repository. https://github.com/facebookresearch/GENRE (paper: https://arxiv.org/abs/2010.00904)
8. Jiangui Chen, Ruqing Zhang, Jiafeng Guo, Yiqun Liu, Yixing Fan, Xueqi Cheng (ICT/CAS). "CorpusBrain: Pre-train a Generative Retrieval Model for Knowledge-Intensive Language Tasks." CIKM 2022. https://doi.org/10.1145/3511808.3557271
9. ICT Big Data Lab. "CorpusBrain" repository. 2022. https://github.com/ict-bigdatalab/CorpusBrain
10. Shashank Rajput et al. (Google). "Recommender Systems with Generative Retrieval (TIGER)." NeurIPS 2023. http://papers.nips.cc/paper_files/paper/2023/hash/20dcab0f14046a5c6b02b61da9f13229-Abstract-Conference.html
11. Ronak Pradeep et al. (Google Research / University of Waterloo). "How Does Generative Retrieval Scale to Millions of Passages?" EMNLP 2023. https://aclanthology.org/2023.emnlp-main.83
12. Hansi Zeng et al. (UMass Amherst). "Scalable and Effective Generative Information Retrieval (RIPOR)." WWW 2024. https://doi.org/10.48550/arXiv.2311.09134
13. Hansi Zeng et al. (UMass Amherst). "Planning Ahead in Generative Retrieval: Guiding Autoregressive Generation through Simultaneous Decoding (PAG)." SIGIR 2024. https://doi.org/10.48550/arXiv.2404.14600
14. Yongqi Li et al. "Learning to Rank in Generative Retrieval (LTRGR)." AAAI 2024. https://doi.org/10.48550/arXiv.2306.15222
15. Thong Nguyen, Andrew Yates. "Generative Retrieval as Dense Retrieval." arXiv 2023. https://doi.org/10.48550/arXiv.2306.11397
16. Shiguang Wu et al. "Generative Retrieval as Multi-Vector Dense Retrieval." SIGIR 2024. https://arxiv.org/pdf/2404.00684.pdf
17. Hailin Zhang et al. "Model-enhanced Vector Index (MEVI)." NeurIPS 2023. https://arxiv.org/abs/2309.13335
18. Peiwen Yuan et al. "Generative Dense Retrieval: Memory Can Be a Burden (GDR)." arXiv 2024. https://arxiv.org/abs/2401.10487
19. Ultron authors. "Ultron: An Ultimate Retriever on Corpus with a Model-based Indexer." arXiv 2022. https://arxiv.org/abs/2208.09257
20. Qiaoyu Tang et al. "Self-Retrieval: End-to-End Information Retrieval with One Large Language Model." NeurIPS 2024. https://arxiv.org/abs/2403.00801
21. ICIP-CAS. "SelfRetrieval" repository. 2024. https://github.com/icip-cas/SelfRetrieval
22. Shengyao Zhuang, Houxing Ren, Linjun Shou, Jian Pei, Ming Gong, Guido Zuccon, Daxin Jiang. "Bridging the Gap Between Indexing and Retrieval for Differentiable Search Index with Query Generation (DSI-QG)." arXiv 2022. https://doi.org/10.48550/arXiv.2206.10128
23. Shengyao Zhuang. "DSI-QG" official repository. 2022. https://github.com/ArvinZhuang/DSI-QG
24. Jiaqi Zhai et al. (Meta). "Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations (HSTU)." ICML 2024. https://proceedings.mlr.press/v235/zhai24a.html (code: https://github.com/meta-recsys/generative-recommenders)
25. Clark Mingxuan Ju, Liam Collins, Leonardo Neves, Bhuvesh Kumar, Louis Yufeng Wang, Tong Zhao, Neil Shah (Snap Research). "Generative Recommendation with Semantic IDs: A Practitioner's Handbook (GRID)." CIKM 2025. https://arxiv.org/abs/2507.22224 (code: https://github.com/snap-research/GRID)
26. Weiwei Sun et al. "Learning to Tokenize for Generative Retrieval (GenRet)." NeurIPS 2023. https://doi.org/10.48550/arXiv.2304.04171
27. Sunkyung Lee, Minjin Choi, Jongwuk Lee. "GLEN: Generative Retrieval via Lexical Index Learning." EMNLP 2023. https://aclanthology.org/2023.emnlp-main.477 (code: https://github.com/skleee/GLEN)
28. Yongqi Li et al. "Generative Retrieval with Large Language Models (LLM2GR)." arXiv 2024. https://arxiv.org/abs/2402.17010v2
29. Yongqi Li, Nan Yang, Liang Wang, Furu Wei, Wenjie Li. "Multiview Identifiers Enhanced Generative Retrieval (MINDER)." ACL 2023; repository. https://github.com/liyongqi67/MINDER (paper: https://doi.org/10.18653/v1/2023.acl-long.366)
30. Shengyao Zhuang. "DSI-transformers" (unofficial DSI re-implementation and reproduction notes). 2022. https://github.com/ArvinZhuang/DSI-transformers
31. Yongqi Li. "LTRGR" repository. 2023. https://github.com/liyongqi67/LTRGR
32. "Understanding Differential Search Index for Text Retrieval." Findings of ACL 2023. https://doi.org/10.18653/v1/2023.findings-acl.681
33. "Distillation Enhanced Generative Retrieval (DGR)." arXiv 2024. https://arxiv.org/abs/2402.10769
34. Edoardo Botta. "RQ-VAE-Recommender" (TIGER-style semantic-ID implementation). 2024-2026. https://github.com/EdoardoBotta/RQ-VAE-Recommender
35. Jiafeng Guo et al. "CorpusBrain++: A Continual Generative Pre-Training Framework for Knowledge-Intensive Language Tasks." arXiv 2024. https://arxiv.org/abs/2402.16767
36. Xiaoxi Li et al. "CorpusLM: Towards a Unified Language Model on Corpus for Knowledge-Intensive Tasks." SIGIR 2024. https://arxiv.org/abs/2402.01176
37. Yu-An Liu et al. "On the Robustness of Generative Retrieval Models: An Out-of-Distribution Perspective." arXiv 2023. https://doi.org/10.48550/arXiv.2306.12756
38. Hansi Zeng. "RIPOR" official repository. 2024. https://github.com/HansiZeng/RIPOR
39. Hansi Zeng. "PAG" official repository. 2024. https://github.com/HansiZeng/PAG
40. Peitian Zhang, Zheng Liu, Yujia Zhou, Zhicheng Dou, Zhao Cao. "Term-Sets Can Be Strong Document Identifiers For Auto-Regressive Search Engines (AutoTSG)." arXiv 2023. https://arxiv.org/abs/2305.13859
41. Orion Weller, Michael Boratko, Iftekhar Naim, Jinhyuk Lee (Google DeepMind). "On the Theoretical Limitations of Embedding-Based Retrieval (LIMIT)." arXiv 2025. https://arxiv.org/abs/2508.21038 (code: https://github.com/google-deepmind/limit)
42. "Generative Retrieval with Few-shot Indexing." arXiv 2024. https://arxiv.org/abs/2408.02152
43. "LLMGR: Large Language Model-based Generative Retrieval in Alipay Search." SIGIR 2024. https://dl.acm.org/doi/pdf/10.1145/3626772.3661364
44. Sanket Vaibhav Mehta et al. (Google). "DSI++: Updating Transformer Memory with New Documents." arXiv 2022 / EMNLP 2023. https://arxiv.org/abs/2212.09744
45. Varsha Kishore, Chao Wan, Justin Lovelace, Yoav Artzi, Kilian Q. Weinberger (Cornell). "IncDSI: Incrementally Updatable Document Retrieval." ICML 2023. https://proceedings.mlr.press/v202/kishore23a.html (code: https://github.com/varshakishore/IncDSI)
46. "PromptDSI: Prompt-based Rehearsal-free Instance-wise Incremental Learning for Document Retrieval." arXiv 2024. https://arxiv.org/abs/2406.12593
47. Chanwoong Yoon et al. "Exploring the Practicality of Generative Retrieval on Dynamic Corpora." EMNLP 2024 (arXiv preprint 2023). https://api.semanticscholar.org/CorpusID:258967398
48. MEVI, NeurIPS 2023 proceedings version. https://proceedings.neurips.cc/paper_files/paper/2023/file/ac112e8ffc4e5b9ece32070440a8ca43-Paper-Conference.pdf
49. "Re3val: Reinforced and Reranked Generative Retrieval." arXiv 2024. https://doi.org/10.48550/arXiv.2401.16979
50. Yubao Tang et al. "Generative Retrieval Meets Multi-Graded Relevance (GR2)." NeurIPS 2024. https://arxiv.org/abs/2409.18409
51. Anima Singh et al. (Google / YouTube). "Better Generalization with Semantic IDs: A Case Study in Ranking for Recommendations." RecSys 2024. https://arxiv.org/abs/2306.08121
52. Jiaxin Deng et al. (Kuaishou). "OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment." arXiv 2025. https://arxiv.org/abs/2502.18965
53. Kuaishou OneRec team. "OpenOneRec" repository and technical report (arXiv 2512.24762). 2025. https://github.com/Kuaishou-OneRec/OpenOneRec
54. Hongru Cai et al. "Exploring Training and Inference Scaling Laws in Generative Retrieval." SIGIR 2025. https://arxiv.org/abs/2503.18941
55. Shiguang Wu, Zhaochun Ren, Xin Xin, Jiyuan Yang, Mengqi Zhang, Zhumin Chen, Maarten de Rijke, Pengjie Ren. "Constrained Auto-Regressive Decoding Constrains Generative Retrieval." SIGIR 2025. https://arxiv.org/abs/2504.09935
56. Kidist Amde Mekonnen, Yubao Tang, Maarten de Rijke. "Lightweight and Direct Document Relevance Optimization for Generative Information Retrieval (DDRO)." SIGIR 2025. https://arxiv.org/abs/2504.05181 (code: https://github.com/kidist-amde/ddro)
