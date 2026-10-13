# 01. RAG Foundations and Taxonomy: From REALM and RAG to the Modular Reference Architecture

## Overview

Retrieval-augmented generation (RAG) is the family of designs in which a language model's
output depends on two things: what is stored in its weights (*parametric memory*) and text
fetched at run time from an external store (*non-parametric memory*). Lewis et al. gave the
idea its name in 2020 [1]. The lineage is older, though. Open-domain QA systems paired a
retriever with a reader, kNN-LM interpolated a nearest-neighbour datastore into token
probabilities [4], and REALM backpropagated through a learned retriever during pre-training
a few months before RAG appeared [2].

This chapter covers three things:

1. **The five canonical research systems** that set the design space between 2020 and 2022:
   REALM [2], RAG [1], Fusion-in-Decoder (FiD) [7], RETRO [4] and Atlas [9]. A short
   section follows on the 2023-2024 "frozen LLM" generation (In-Context RALM [21], REPLUG
   [22], RA-DIT [23], Retro++/InstructRetro [24][25], Self-RAG [26]), which carried those
   ideas over to API-served LLMs.
2. **The paradigm taxonomy** that the field now uses: *Naive*, *Advanced* and *Modular*
   RAG, as set out in the Gao et al. survey [12] and the follow-up Modular RAG paper [13],
   extended by the 2025-2026 agentic-RAG literature [31][32].
3. **An end-to-end reference architecture** (ingest, index, retrieve, rerank, generate,
   cite) with public numbers on cost, latency and quality, and a catalogue of failure modes
   [14][15][16].

The other chapters of this atlas take the individual stages further. This chapter sets the
vocabulary they use.

A useful frame from the start: every RAG design answers four questions.

- **Where does retrieval happen?** In pre-training (REALM, RETRO, Atlas), in fine-tuning
  (RAG, FiD), or only at inference (In-Context RALM, most production systems).
- **What is the unit of retrieval?** A 100-word passage (RAG, FiD), a 288-wordpiece block
  (REALM), a 64-token chunk (RETRO), or a section-aligned passage of up to 200 words (Atlas).
- **Where are retrieved items fused with the query?** In the input sequence (RAG, REPLUG),
  in the decoder's cross-attention (FiD, Atlas), in chunked cross-attention inside the
  decoder stack (RETRO), or in the output distribution (kNN-LM, continuous cache) [4].
- **Is the retriever trained, and with what signal?** It may be frozen (RETRO), trained
  query-side only (RAG, Atlas few-shot), trained end to end with a refreshed index (REALM,
  Atlas pre-training), or trained from the LM's own preferences (REPLUG-LSR, RA-DIT).

---

## 1. Pre-history: retrieve-then-read and dense retrieval

### 1.1 Retrieve-then-read open-domain QA

Before 2020, open-domain QA meant two stages. A sparse retriever (TF-IDF/BM25) selected
Wikipedia passages, and an extractive reader picked out an answer span. ORQA (Lee et al.,
2019) made retrieval a latent variable trained by marginal likelihood. REALM extended this
directly [2]. The weakness of an extractive reader is structural: it can only return
strings that appear verbatim in a retrieved passage.

### 1.2 Dense Passage Retrieval (DPR)

DPR [11] is the retriever that RAG and FiD used. It is a bi-encoder: two independent
BERT-base encoders, with the [CLS] vector (d = 768) as the representation and dot-product
similarity. It was trained with **in-batch negatives** (batch size 128) plus one hard BM25
negative per question. Its headline numbers made dense retrieval the default:

- Top-20 retrieval accuracy on Natural Questions was **78.4% against 59.1% for BM25**, and
  top-100 was 85.4% against 73.7% [11].
- At query time, a FAISS in-memory index served **995 questions/second** (top-100 passages)
  on a Xeon server. BM25/Lucene served 23.7 questions/second per CPU thread [11].
- Indexing cost ran the other way. Embedding 21M passages took about **8.8 hours on 8
  GPUs**, and building the FAISS index on a single server took another **8.5 hours**,
  against about **30 minutes** for a Lucene inverted index [11].

**Lesson.** Dense retrieval moves cost from query time to ingest time, and every encoder
change re-incurs that cost. Index staleness, covered in REALM and Atlas below, is the
training-time version of the same problem.

---

## 2. REALM (Google, 2020): retrieval inside pre-training

### Problem

Masked language models store world knowledge implicitly in their weights. REALM asked
whether a model could learn to *look things up* during pre-training, with an unsupervised
signal deciding which documents help [2].

### Architecture

```text
 masked sentence x ──► Embed_input(x) ─┐
                                       │  MIPS over ~13M Wikipedia blocks
 Wikipedia blocks z ─► Embed_doc(z) ───┤  (index rebuilt asynchronously)
                                       ▼
                         top-k docs + null doc ∅
                                       │
                         p(z|x) = softmax(Embed_input(x)·Embed_doc(z))
                                       ▼
            knowledge-augmented encoder p(y|z,x) (BERT over [x ; z])
                                       ▼
            p(y|x) = Σ_z p(y|z,x) p(z|x)   ── gradient flows into retriever
```

- **Retriever.** A BERT-style dual encoder scored by inner product. The knowledge corpus is
  the 20 December 2018 English Wikipedia, greedily split into chunks of up to **288 BERT
  wordpieces**, giving just over **13 million** candidates [2].
- **Objective.** Retrieve, then predict. The document is a latent variable and the loss is
  the marginal likelihood. The retriever gradient raises the score of document z exactly
  when `p(y|z,x) > p(y|x)`, so documents that help predict the masked tokens are rewarded
  [2].
- **Asynchronous index refresh.** Updating the document embedder makes the MIPS index stale
  after every step. REALM runs a trainer job and a separate index-builder job in parallel,
  and the index is refreshed about **every 500 training steps**. Because the stale index is
  used only to *select* the top k, p(z|x) is recomputed with fresh parameters for the
  selected documents [2].
- **Salient span masking.** REALM masks named entities (from a CoNLL-2003 BERT tagger) and
  dates (from a regex) rather than random tokens, so that masked spans need world knowledge
  [2].
- **Null document.** An empty document is added to the top k as a sink for predictions that
  need no retrieval [2].

### Scale and results

Pre-training ran for 200k steps on **64 Google Cloud TPUs** (batch size 512). Document
embedding for the index was parallelized over 16 TPUs, and each example marginalized over 8
candidates including the null document [2]. The open-source README says a typical
single-machine setup lacks the needed resources, "80 TPUs and numerous CPUs", and notes that
the release replaced ScaNN with brute-force matrix multiplication [3]. Fine-tuned for
open-domain QA (top-5 candidates), the 330M-parameter REALM reached **40.4 / 40.7 / 42.9**
exact match (EM) on NQ / WebQuestions / CuratedTrec with CC-News pre-training. ORQA, with
the same fine-tuning setup, scored 33.3 / 36.4 / 30.1 [2].

### Failure modes and lessons

The ablations isolate each ingredient [2]:

| REALM ablation (NQ dev)            | EM   | Gold answer in top-5 at init |
|------------------------------------|------|------------------------------|
| Full REALM                         | 38.2 | 38.5                         |
| Random uniform masking             | 32.3 | 24.2                         |
| Random span masking                | 35.3 | 26.1                         |
| 30x staler MIPS index              | 28.7 | 15.1                         |
| Baseline (ORQA)                    | 31.3 | 13.9                         |

**Lessons.** (a) Index freshness is a first-order concern: a 30x staler index is *worse
than the baseline*. (b) The pretext task must actually require retrieval; salient spans
matter. (c) End-to-end retriever training is expensive enough that REALM used
asynchronous refresh only in pre-training. During fine-tuning it trained only the query
encoder against a fixed index. That compromise reappears in RAG and Atlas.

---

## 3. RAG (Facebook AI / UCL / NYU, 2020): the generative template

### Problem

Extractive readers cannot synthesize an answer, and parametric seq2seq models cannot cite,
update or verify what they know. RAG fine-tunes a pre-trained generator jointly with a
pre-trained retriever, so that any seq2seq task can draw on a dense Wikipedia index [1].

### Architecture

```text
  x ──► DPR query encoder q(x) (BERT-base, trainable)
           │  MIPS (FAISS, HNSW) over 21M x 100-word passages
           ▼  document encoder d(z) FROZEN; index never rebuilt
   top-K passages z_1..z_K with p_eta(z|x) ∝ exp(d(z)·q(x))
           │
           ├─► BART-large( [z_1 ; x] ) ─┐
           ├─► BART-large( [z_2 ; x] ) ─┼─► marginalize over z
           └─► BART-large( [z_K ; x] ) ─┘   (per sequence or per token)
                                              ▼
                                              y
```

- **Components.** A DPR retriever, made of a BERT-base query encoder and a BERT-base
  document encoder with 110M parameters each, and a **BART-large** generator with 406M
  parameters, for 626M parameters in total [1]. The document encoder and index stay
  **frozen**. Only the query encoder and the generator are fine-tuned, which avoids
  re-embedding the corpus [1].
- **Corpus and index.** The December 2018 Wikipedia dump, split into disjoint **100-word
  chunks** for **21M documents**, indexed in FAISS with an HNSW approximation. Training
  used k in {5, 10} [1].
- **Two marginalization schemes.** In written-out notation:
  - *RAG-Sequence* uses one document for the whole output:
    `p(y|x) ≈ Σ_{z∈topK} p(z|x) · Π_i p(y_i | x, z, y_<i)`.
  - *RAG-Token* can switch documents at each token:
    `p(y|x) ≈ Π_i Σ_{z∈topK} p(z|x) · p(y_i | x, z, y_<i)`.
  For classification the two are equivalent [1].
- **Decoding.** RAG-Token uses standard beam search over the marginalized per-token
  distribution. RAG-Sequence needs a beam search *per document* plus extra forward passes to
  score hypotheses that did not appear in every beam ("Thorough Decoding"). A cheaper
  approximation ("Fast Decoding") skips those extra passes [1]. Test-time QA used **15
  documents for RAG-Token** and **50 documents with Thorough Decoding for RAG-Sequence** [1].

### Infrastructure numbers

Training ran on **8 x 32 GB V100 GPUs**. The index lived on CPU and needed about **100 GB
of CPU memory**, which FAISS compression later reduced to **36 GB** [1]. The model was
ported to Hugging Face Transformers, which ships `RagSequenceForGeneration`,
`RagTokenForGeneration` and a `RagRetriever` with a compressed `wiki_dpr` index [33].

### Results

| Model (open-domain QA EM)   | NQ   | TQA (std / wiki) | WQ   | CT   |
|-----------------------------|------|------------------|------|------|
| T5-11B closed-book          | 34.5 | - / 50.1         | 37.4 | -    |
| REALM                       | 40.4 | - / -            | 40.7 | 46.8 |
| DPR (retrieve+rerank+extract)| 41.5 | 57.9 / -        | 41.1 | 50.6 |
| RAG-Token                   | 44.1 | 55.2 / 66.1      | 45.5 | 50.0 |
| RAG-Sequence                | 44.5 | 56.8 / 68.0      | 45.2 | 52.2 |

Source: [1], Table 1.

Beyond QA, RAG reached **72.5% / 89.5%** label accuracy on FEVER 3-way / 2-way without
retrieval supervision. Human raters judged RAG more factual than BART in **42.7%** of
Jeopardy question-generation comparisons, and BART more factual in only **7.1%** [1]. On
NQ, RAG answered **11.8%** of questions correctly even when no retrieved passage contained
the answer string, a case where an extractive reader must score 0 [1].

**Index hot-swapping.** Built against a December 2016 index, the NQ model answered 70% of
"Who is {position}?" questions correctly for 2016 world leaders. The 2018 index gave 68% for
2018 leaders. Mismatched pairs gave 12% and 4% [1]. Knowledge can be updated without
gradient steps, which is the property production RAG depends on.

### Failure modes and lessons

- **Retrieval collapse.** On story generation, the retriever learned to return the same
  documents for every input, the generator learned to ignore them, and RAG degraded to BART
  [1]. Weak or long-horizon supervision does not train a retriever reliably.
- **Sparse beats dense on entity-centric claims.** Swapping in BM25 gave the best FEVER
  result (75.1 vs. 72.9 dev accuracy for the frozen-dense variant) [1]. This is an early
  argument for hybrid retrieval.
- **Neither a reranker nor an extractive reader was necessary** for state-of-the-art results
  in 2020 [1]. Later systems brought reranking back once generators were frozen black boxes
  and context budgets mattered (Section 10.4).

---

## 4. Fusion-in-Decoder (FiD, Facebook AI, 2020-2021): scaling the number of passages

### Problem

RAG marginalizes over K separate generator passes, while early-fusion readers concatenate
every passage into one quadratic-cost encoder input. Neither handles evidence spread over
tens or hundreds of passages efficiently [7].

### Architecture

```text
 question + passage_1 ─► T5 encoder ─► H_1 ─┐
 question + passage_2 ─► T5 encoder ─► H_2 ─┤  concatenate [H_1 ... H_N]
        ...                          ...    ├──► T5 decoder cross-attends to all
 question + passage_N ─► T5 encoder ─► H_N ─┘        └─► answer
 (independent, linear in N)            (joint "fusion" happens only in decoder)
```

Each retrieved passage is concatenated with its title and the question and encoded
**independently**. The decoder attends over the concatenation of all encoded passages.
Encoder cost therefore grows **linearly** with the number of passages, and evidence is
fused **in the decoder** [7].

### Numbers

- T5-base (220M) and T5-large (770M) readers, trained on **64 x V100 32 GB** for 10k steps
  [7].
- FiD-large reached **51.4 EM on NQ** and **67.6 on TriviaQA** [7]. The repository reports
  54.4 NQ test for a large reader trained with retriever distillation, and a distilled
  retriever at R@20 = 84.3 on NQ [8].
- Going from 10 to 100 passages added about **3.5 EM on NQ and 6 EM on TriviaQA**, whereas
  extractive readers typically peaked at 10-20 passages [7].
- Training on fewer passages and then fine-tuning for 1,000 steps with 100 passages reached
  **46.0 EM in 147 GPU-hours, against 425 GPU-hours** for full 100-passage training [7].

### Failure modes and lessons

FiD's inference cost is **not** dominated by the encoder. FiDO (Google, 2022) showed that
the encoder receives most of the FLOPs, while decoder **memory bandwidth** (cross-attention
over thousands of cached encoder states) accounts for most inference time. Two
architectural changes yielded about a **7x inference speedup**, enough to afford a much
larger decoder [20]. **Lesson.** Benchmark RAG latency on memory traffic, not FLOPs. The
same holds for today's long-context prompts, where prefill and KV-cache size dominate.

FiD's cross-attention scores also became a relevance signal: "distilling knowledge from
reader to retriever" trains the retriever to imitate where the reader looks [8]. Atlas
generalizes this idea.

---

## 5. RETRO (DeepMind, 2021): retrieval at trillion-token scale

### Problem

Can retrieval substitute for parameters at the scale of pre-training, so that a 7B model
matches models 25x larger by consulting a database bigger than its training set [4][5]?

### Architecture

```text
 input (2048 tokens) split into 64-token chunks C_1 ... C_32
        │
        │  for chunk C_u: key = mean-pooled FROZEN BERT(C_u)
        ▼  SCaNN approx. kNN over key-value DB (value = neighbour N + continuation F)
  Ret(C_u) = k neighbours x (64 + 64 tokens)
        │
        ▼  small bidirectional encoder over neighbours (conditioned on C_u)
  ┌──────────────────────── decoder stack ─────────────────────────┐
  │ standard blocks: self-attn + FFN                               │
  │ RETRO blocks:   self-attn + CHUNKED CROSS-ATTENTION + FFN      │
  │   tokens of chunk u+1 attend to neighbours retrieved for C_u   │
  └────────────────────────────────────────────────────────────────┘
```

- **Chunked retrieval.** Retrieving per 64-token chunk rather than per token cuts storage
  and compute "by a large linear factor" [4]. Each database value is a neighbour chunk N
  plus its continuation F, each up to 64 tokens [4][6].
- **Frozen retriever.** Keys are frozen BERT embeddings averaged over time, so the database
  never needs re-embedding during training. Neighbours were **precomputed** for all
  training data because on-the-fly retrieval could not keep up with training [4].
- **Latency.** With SCaNN, the 2-trillion-token database can be queried in **10 ms** while
  sampling, a cost amortized over a 64-token chunk [4].
- **Chunked cross-attention (CCA).** Cost is linear in the amount of retrieved data.
  Causality is kept by letting chunk u+1 attend only to neighbours of chunk u, and by
  filtering neighbours from the same source document during training [4]. Per Alammar's
  walkthrough, RETRO blocks sit at every third layer from layer 9 in the large models [6].
- **Data.** Multilingual MassiveText (over 5T tokens). The training-time retrieval DB holds
  600B tokens and the evaluation DB **1.75T tokens**. Training documents with 13-gram
  Jaccard similarity of 0.8 or more to test documents were removed (MinHash) [4].

### Results and scale

- Gains hold from **150M to 7B** parameters and do not shrink with size. Larger databases
  and more neighbours help even at evaluation time: models trained with 2 neighbours keep
  improving up to 10 neighbours (172M) or 40 neighbours (7B) [4].
- The 7.5B RETRO outperforms Jurassic-1 (178B) and Gopher (280B) on a majority of Pile
  subsets [4]. DeepMind summarized this as performance comparable to GPT-3 with **25x
  fewer parameters** [5].
- **RETROfitting.** Freezing a pre-trained transformer and training only the CCA and
  neighbour-encoder weights (**under 10%** of the weights for 7B) on **6M sequences (3% of
  pre-training)** nearly matches training from scratch, and leaves performance with
  retrieval switched off unchanged [4].
- On NQ, the fine-tuned RETRO 7.5B (with DPR passages) reached **45.5 EM**: competitive
  with RAG (44.5) but below FiD (51.4) and FiD+distillation (54.7). More than 20 neighbours
  did not help [4].

### Failure modes and lessons

- **Leakage.** A retriever with access to the training corpus can copy evaluation text.
  RETRO introduced filtered bits-per-byte as a function of chunk overlap. RETRO exploits
  leakage *more* than baselines (steeper slope), yet it still beats baselines at overlap as
  low as 12.5%, i.e. fewer than 8 shared contiguous tokens [4]. Any RAG evaluation should
  measure overlap between the corpus and the test set.
- **Reproducibility.** There was no official open-source release. NVIDIA later reproduced
  RETRO up to **9.5B** parameters with a **330B-token** retrieval corpus [24]. It then
  continued pre-training a 43B GPT on 100B tokens while retrieving from **1.2T tokens**,
  for about **2.58%** extra GPU-hours, producing Retro 48B. After instruction tuning,
  "InstructRetro" improved over instruction-tuned GPT by on average 7% on short-form QA,
  10% on long-form QA and 16% on summarization, zero-shot. The authors also found that
  the retrieval encoder could be removed with comparable results [25].
- **Industry lesson.** Architecturally integrated retrieval works, but the market moved to
  in-context retrieval with frozen LLMs (Section 8), which needs no custom pre-training.

---

## 6. Atlas (Meta AI, 2022): joint pre-training for few-shot RAG

### Problem

Large LLMs appeared to need huge parameter counts to *store* knowledge for few-shot use.
Atlas asked whether a much smaller, jointly pre-trained retrieval-augmented model could
learn knowledge-intensive tasks from a handful of examples [9].

### Architecture

Atlas pairs a **Contriever** dual-encoder retriever (BERT-base, mean pooling, shared
query/document weights) with a **FiD reader built on T5 v1.1** (770M, 3B and 11B
variants) [9][10]. Retrieval runs **in the training loop**. Each GPU holds 1/N of the
passage embeddings and runs **exact** inner-product search on its shard; FAISS (for
example, IVF-PQ with code size 16) is optional for memory savings [10]. The code was
tested up to **11B-parameter readers and 400M-passage (~40B-word) indices** [10].

**Retriever training objectives** compared in the paper [9]:

| Objective                     | Training signal for the retriever                           |
|-------------------------------|-------------------------------------------------------------|
| Attention Distillation (ADist)| reader cross-attention mass on each passage                 |
| EMDR2                         | EM-style, passages as latent variables (token-level)        |
| Perplexity Distillation (PDist)| how much each passage alone improves LM likelihood (KL)    |
| Leave-one-out (LOOP)          | how much worse the LM gets when one passage is removed      |

Atlas uses **PDist with masked-language-model pre-training** on a Wikipedia plus Common
Crawl mix for its final models [9].

**Index staleness** is handled three ways [9][10]:

1. **Full refresh.** For a 37M-document index, batch 64, 20 retrieved documents, and a
   refresh every 1,000 steps, the overhead is about **30%** [9].
2. **Over-retrieve and rerank.** Retrieve L > K with the stale index and re-embed those L
   with the current retriever. With L = 10K the overhead is about 10% [9].
3. **Query-side fine-tuning.** Freeze the document encoder so there is no refresh at all.
   This was best in few-shot settings (45.0 vs. 44.3 for standard fine-tuning in the 64-shot
   ablation's NQ column) [9].

### Corpus, memory and cost

- The December 2021 Wikipedia index, with infoboxes and lists linearized and passages under
  200 words, holds **37M passages** (78 words on average). The CCNet-filtered Common Crawl
  index holds **350M passages** [9].
- Embeddings stored in fp16 need **49 GB** of GPU memory for Wikipedia and **587 GB** for
  the combined index. Product quantization compresses these to **4 GB** and **50 GB** with
  negligible loss in retrieval precision and EM [9]. Tokenized passages take 11 GB / 130 GB
  but can live on disk or be memory-mapped [9].
- Atlas-11B was trained with about **3.5e22 FLOPs** [9].

### Results

- **64-shot NQ: 42.4% EM** (paper Table 8), ahead of PaLM-540B with about 50x fewer
  parameters [9]. The repository README reports over 45% with a 2018 index, in line with
  the paper's **+3.6%** gain from a temporally matched index [10][9].
- **Full NQ: 60.4% EM**, the state of the art at the time. TriviaQA 74.5 [9]. The README
  quotes 64% for full NQ; the figures differ by setup and version [10].
- **MMLU 5-shot (multi-task): 56.6%**, and 66.0% with full/transfer fine-tuning, for an 11B
  model [9].
- **Leakage control.** Filtering possibly leaked passages reduced MMLU from 56.4 to 55.8%
  [9].

### Lessons

(a) Joint pre-training of retriever and reader pays off most in few-shot regimes. (b) Index
compression is close to free: about 10x with negligible loss. (c) Temporal alignment
between index and benchmark is worth several EM points, so production indexes need
*versioning*, not just updates.

---

## 7. Comparison of the canonical systems

| System | Year / org | Retriever | Reader | Fusion point | Retriever training | Corpus | Headline |
|---|---|---|---|---|---|---|---|
| REALM [2] | 2020 Google | BERT dual-encoder | BERT (extractive) | input concat; marginalize | end-to-end in pre-training; async refresh ~500 steps | ~13M x 288-wp Wiki blocks | NQ 40.4 EM, 330M |
| RAG [1] | 2020 FAIR/UCL/NYU | DPR (BERT-base x2) | BART-large | input concat; marginalize per seq/token | query encoder only; doc index frozen | 21M x 100-word Wiki | NQ 44.5 EM, 626M |
| FiD [7] | 2020 FAIR | BM25 / DPR | T5-base/large | decoder cross-attn over N encodings | none (later: reader-to-retriever distill.) | 100-word Wiki | NQ 51.4 EM; up to 100 passages |
| RETRO [4] | 2021 DeepMind | frozen BERT + SCaNN | decoder-only + CCA | chunked cross-attn every few layers | frozen | 600B train / 1.75T eval tokens | 7.5B rivals 178B-280B LMs on Pile |
| Atlas [9] | 2022 Meta | Contriever | FiD (T5 up to 11B) | decoder cross-attn | joint pre-training (PDist); query-side FT | 37M Wiki + 350M CCNet | 64-shot NQ 42.4; full NQ 60.4 |
| In-Context RALM [21] | 2023 AI21 | BM25 / dense | any frozen LM | prepend to prompt every s tokens | none | Wikipedia etc. | no LM changes |
| REPLUG [22] | 2023 UW/Meta | Contriever | black-box LM (GPT-3, Codex) | prepend; ensemble over docs | LM-supervised (LSR) | Pile | +6.3% GPT-3 LM; +5.1% Codex MMLU |
| RA-DIT [23] | 2023 Meta | dual-encoder | LLaMA 65B | prepend | dual instruction tuning | Wiki + CC | up to +8.9% 0-shot vs. RALM |
| Self-RAG [26] | 2023 UW/AI2/IBM | off-the-shelf dense | Llama2 7B/13B | prepend; adaptive | none; LM trained to critique | Wikipedia | better factuality and citation accuracy |

Cell values come from each system's paper. Where a cell is qualitative, the paper's
headline claim is quoted rather than a number.

---

## 8. The "frozen LLM" turn (2023-2024)

Once instruction-tuned LLMs became API products, end-to-end training like REALM's and
Atlas's no longer fit. The research frontier moved to designs that leave the LLM untouched:

- **In-Context RALM** (AI21) prepends retrieved documents to the input of an unmodified LM.
  It re-retrieves every *s* tokens (the retrieval stride) and shows that smaller strides
  perform better at a runtime cost [21]. Secondary summaries put the recommended stride
  at about 4 tokens, against RETRO's 64; treat that figure as approximate [21].
- **REPLUG** treats the LM as a black box: retrieved documents are prepended separately and
  the output distributions are ensembled. **REPLUG-LSR** trains the retriever from the LM's
  own likelihoods. The authors report **+6.3%** on GPT-3 (175B) language modelling and
  **+5.1%** on Codex 5-shot MMLU [22].
- **RA-DIT** (Meta, ICLR 2024) retrofits an LLM with two lightweight fine-tuning stages:
  one teaches the LM to use retrieved text, the other teaches the retriever what the LM
  prefers. RA-DIT 65B gained up to **+8.9%** (0-shot) and **+1.4%** (5-shot) on average
  over in-context RALM baselines [23].
- **Self-RAG** trains one LM to emit **reflection tokens**: whether to retrieve, whether a
  passage is relevant (ISREL), whether the generation is supported by the passage (ISSUP),
  and overall utility. These tokens make retrieval adaptive and decoding controllable [26].
- **CRAG** adds a lightweight T5-large *retrieval evaluator*. It classifies a retrieval as
  {Correct, Incorrect, Ambiguous}. It then triggers decompose-then-recompose knowledge
  refinement or a fallback to web search [27].

These designs mark the move from a fixed retrieve-then-read pipeline to *control flow*, the
defining property of Modular RAG.

---

## 9. Taxonomy: Naive, Advanced and Modular RAG

The most widely used taxonomy comes from Gao et al.'s survey (Tongji/Fudan, Dec 2023,
updated Mar 2024) [12]. It describes three paradigms as nested special cases: "Advanced RAG
is a special case of Modular RAG, while Naive RAG is a special case of Advanced RAG" [12].

### 9.1 Naive RAG ("Retrieve-Read")

```text
 docs ─► parse ─► fixed-size chunks ─► embed ─► vector DB
                                                   │
 query ─► embed ─► top-K similarity ───────────────┘──► prompt(query + K chunks) ─► LLM
```

Indexing turns PDF, HTML, Word and Markdown into plain text, chunks it, embeds it and stores
it. Retrieval embeds the query with the same model and takes the top-K most similar chunks.
Generation builds a prompt from the query and chunks [12]. The survey lists the drawbacks
[12]:

- **Retrieval:** low precision and recall (misaligned or irrelevant chunks, missed facts).
- **Generation:** hallucination unsupported by the context; irrelevance, toxicity, bias.
- **Augmentation:** disjointed outputs, redundancy across chunks, a single retrieval that
  is insufficient for complex questions, and over-reliance that merely echoes the retrieved
  text.

### 9.2 Advanced RAG (pre- and post-retrieval optimization)

Still a linear chain, but with work added on both sides of retrieval [12]:

- **Pre-retrieval / indexing:** sliding windows, finer segmentation, metadata, alignment,
  mixed (hybrid) retrieval.
- **Pre-retrieval / query:** rewriting, transformation, expansion. Examples include
  Rewrite-Retrieve-Read [35], HyDE, which embeds an LLM-written *hypothetical answer
  document* instead of the query [28], and multi-query fusion.
- **Post-retrieval:** **reranking**, including moving the most relevant chunks to the edges
  of the prompt (a response to "lost in the middle"), and **context compression** or
  selection [12][15].

### 9.3 Modular RAG (reconfigurable modules and flows)

Modular RAG adds new modules: Search (across search engines, databases and knowledge graphs
via generated code or query languages), RAG-Fusion multi-query, Memory, Routing, Predict
(LLM-generated context) and Task Adapter. It also adds new patterns: Rewrite-Retrieve-Read,
Generate-Read, Recite-Read, Demonstrate-Search-Predict [34], ITER-RETGEN's
Retrieve-Read-Retrieve-Read loop, and adaptive retrieval (FLARE, Self-RAG) [12].

The follow-up paper "Modular RAG: Transforming RAG Systems into LEGO-like Reconfigurable
Frameworks" (Gao et al., July 2024) formalizes this. It decomposes systems into modules,
sub-modules and operators, and names four flow patterns: **linear, conditional, branching
and looping** [13].

### 9.4 Beyond modular: agentic RAG (2025-2026)

Agentic RAG puts an LLM agent in charge of the loop. The agent plans, chooses tools and
indexes, reflects on evidence, and decides when to stop. Singh et al. (Jan 2025, revised
through 2026) classify systems as single-agent ("router"), multi-agent and hierarchical
[31]. A 2026 systematization-of-knowledge paper classifies by *agent topology* and adds an
explicit planner-executor class [32]. Later chapters of this atlas cover these designs. In
the taxonomy here, agentic RAG is Modular RAG whose looping and branching operators are
chosen at run time by a model rather than fixed by an engineer.

### 9.5 Paradigm comparison

| Dimension | Naive RAG | Advanced RAG | Modular RAG | Agentic RAG |
|---|---|---|---|---|
| Control flow | fixed linear | fixed linear, extra stages | linear / conditional / branching / looping [13] | model-decided loops, tools [31] |
| Query handling | raw query | rewrite, expand, HyDE [28][35] | routing, multi-query, sub-queries | planning and decomposition |
| Index | one dense index | metadata, hybrid, finer chunks | multiple stores (vector, KG, SQL, web) | any tool-accessible source |
| Post-retrieval | none | rerank, compress | fusion, evaluation (CRAG [27]) | self-critique, retry |
| When to retrieve | always, once | always, once | adaptive (FLARE, Self-RAG [26]) | agent decides |
| Training | none | optional embedder fine-tuning | optional joint tuning (RA-DIT [23]) | optional RL / tool training |
| Typical failure | misses, noise | latency and cost of extra stages | orchestration complexity | unbounded loops, cost variance |

---

## 10. The end-to-end reference architecture

Production systems from this period converge on six stages, plus an evaluation loop that
runs across all of them. The diagram uses the chapter's vocabulary; the numbers attached to
each stage come from the cited sources.

```text
                   OFFLINE (ingest + index)                     ONLINE (query path)
 ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌──────────────┐
 │ 1 INGEST │─►│  chunk   │─►│  enrich / │─►│   2 INDEX    │
 │ parse,   │  │ size,    │  │ contextual│  │ dense (ANN)  │
 │ clean,   │  │ overlap, │  │ -ize, add │  │ + sparse BM25│
 │ ACLs     │  │ structure│  │ metadata  │  │ + metadata   │
 └──────────┘  └──────────┘  └───────────┘  └──────┬───────┘
                                                    │
 user ─► query understanding ─► 3 RETRIEVE ◄────────┘
         (rewrite, HyDE,        hybrid top-N (e.g. N=150), filters
          route, decompose)          │
                                     ▼
                                4 RERANK  cross-encoder / late interaction
                                     │    top-K (e.g. K=20)
                                     ▼
                                5 GENERATE  prompt assembly, ordering,
                                     │      compression, LLM call
                                     ▼
                                6 CITE      span-level attribution,
                                     │      support checking
                                     ▼
                                  answer + sources ──► logs ──► EVALUATE
                                                         (faithfulness, recall)
```

### 10.1 Ingest

**What it does.** Parse heterogeneous formats into clean text, preserve structure (titles,
sections, tables), attach metadata (source, page, timestamp, author, permissions), and chunk
the result.

**Evidence and numbers.**

- The canonical research systems used 100-word disjoint passages (RAG, DPR, FiD) [1][11][7],
  288-wordpiece blocks (REALM) [2], or **section-aligned passages under 200 words, with
  linearized infoboxes and lists** (Atlas) [9]. Including infoboxes and tables as text was
  a deliberate choice in Atlas.
- The survey lists fixed token sizes (100/256/512) as the most common method, notes the
  trade-off (larger chunks bring more context but more noise and cost), and recommends
  metadata attachments plus artificial metadata such as summaries or hypothetical questions
  ("Reverse HyDE") [12].
- **Contextual Retrieval** (Anthropic, Sept 2024) shows how much chunking discards. A chunk
  like "The company's revenue grew by 3%..." loses its company and quarter. An LLM (Claude 3
  Haiku in the post) writes 50-100 tokens of chunk-specific context from the whole document,
  prepended before embedding and before BM25 indexing. With prompt caching, the one-time
  cost was quoted at **$1.02 per million document tokens** (assuming 800-token chunks and
  8k-token documents) [16]. Generic document summaries added to chunks gave "very limited
  gains" [16].
- Barnett et al. found that adding **the file name and chunk number** to retrieved context
  helped the reader extract answers [14].

### 10.2 Index

**What it does.** Build one or more access paths: an ANN index over embeddings (HNSW, IVF,
IVF-PQ), an inverted index for lexical retrieval, and metadata filters.

**Evidence and numbers.**

- **Memory.** RAG's 21M-passage Wikipedia index used about 100 GB of CPU RAM, or 36 GB
  compressed [1]. Atlas's indexes went from 49 GB to 4 GB and from 587 GB to 50 GB with PQ
  at negligible quality cost [9].
- **Build cost.** DPR needed 8.8 h on 8 GPUs to embed 21M passages plus 8.5 h to build
  FAISS, against about 30 minutes for Lucene [11]. Changing the embedding model means
  re-embedding the whole corpus, so plan for it.
- **Freshness.** Updating an index changes answers without training (RAG's 2016/2018 swap
  [1]; Atlas's temporal experiments [9]). Staleness *within* a learned system is harmful:
  REALM's 30x-staler-index ablation fell below the baseline [2].
- **Hybrid.** Dense and lexical indexes are complementary. DPR+BM25 beat either alone on
  most datasets [11], BM25 won FEVER inside RAG [1], and Anthropic found that
  "Embeddings+BM25 is better than embeddings on their own", especially for exact
  identifiers such as error codes [16].
- **Small corpora.** If the knowledge base is under about 200k tokens (about 500 pages),
  Anthropic suggests skipping retrieval and putting it all in the prompt with prompt
  caching. Caching was quoted as cutting latency by more than 2x and cost by up to 90% [16].

### 10.3 Retrieve

**What it does.** Turn the user request into one or more queries, search each access path,
and fuse the candidate lists (for example with reciprocal rank fusion).

**Evidence and numbers.**

- **Query transformation:** Rewrite-Retrieve-Read trains a small rewriter with LM feedback
  [35]. HyDE embeds a generated hypothetical document for zero-shot dense retrieval [28].
  DSP composes multi-hop search and predict programs [34].
- **Recall budgets.** Anthropic retrieved the top 150 candidates before reranking. Measured
  as 1 - recall@20, contextual embeddings cut retrieval failures by **35% (5.7% to 3.7%)**,
  and contextual embeddings plus contextual BM25 by **49% (5.7% to 2.9%)** [16].
- **Throughput.** A dense ANN search over tens of millions of vectors is cheap: DPR served
  about 1k queries/s on CPU [11], and RETRO queried 2T tokens in about 10 ms [4]. In
  production, latency is dominated by the query-embedding call, reranking and generation,
  not by the ANN search.

### 10.4 Rerank

**What it does.** Re-score a few hundred candidates with a more expensive model that sees
query and document together (a cross-encoder) or uses token-level late interaction
(ColBERT), then keep the top K.

**Evidence and numbers.**

- ColBERT's late interaction gave **over 170x speedup and 14,000x fewer FLOPs** than
  BERT-based cross-encoder reranking on MS MARCO, while beating every non-BERT baseline.
  The paper also notes that adding as little as 100 ms of latency affects user experience
  [19].
- Reranking the contextual-retrieval candidates (Cohere reranker, top 150 down to 20)
  raised the reduction in retrieval failures to **67% (5.7% to 1.9%)** [16].
- Atlas's "over-retrieve and rerank with the fresh encoder" is the same pattern used
  during training [9].
- **Trade-off.** Reranking adds latency even though candidates are scored in parallel. Cost
  grows with N [16].

### 10.5 Generate

**What it does.** Assemble a prompt (instructions, conversation, ordered evidence with
source markers) and call the generator. Optionally compress or filter the context, and
decide whether retrieval was needed at all.

**Evidence and numbers.**

- **Position matters.** Liu et al. found a U-shaped accuracy curve: relevant information
  at the start or end of the context is used best. GPT-3.5-Turbo's multi-document QA
  accuracy fell by more than 20 points when the answer sat in the middle. In some 20-30
  document settings it fell *below* the closed-book score of **56.1%** [15].
- **More is not always better.** Anthropic found top-20 chunks better than top-10 or
  top-5, but warned that extra context can distract [16]. FiD-style readers improved up to
  100 passages [7], whereas RETRO stopped improving past 20 neighbours on NQ [4].
- **Long context vs. RAG.** Li et al. (EMNLP 2024 industry track) found that long-context
  LLMs given enough resources outperform RAG on average, while RAG is far cheaper. Their
  **Self-Route** sends each query to RAG or to long context based on the model's
  self-reflection [30]. Per secondary summaries, token savings reached about 65% in some
  settings; treat this as approximate [30].
- **Adaptive retrieval.** Self-RAG and CRAG decide whether and how to use retrieved text
  instead of always conditioning on it [26][27]. Barnett et al. report that a larger context
  window (8K vs. 4K) gave more accurate answers in their case studies [14].

### 10.6 Cite

**What it does.** Attach verifiable pointers from claims in the answer to source spans, and
optionally check that each claim is supported.

**Evidence and practice.**

- The original RAG paper already presented provenance and interpretability as a benefit of
  non-parametric memory [1]. Atlas emphasized inspecting retrieved items [9].
- **API-level citations.** Anthropic's Citations feature chunks supplied documents into
  sentences (or uses caller-defined content blocks). It returns each claim with pointers
  (character, page or block ranges) into the source. Because `cited_text` is extracted by
  the API, citations are "guaranteed to contain valid pointers to the provided documents",
  and quoted text does not count as output tokens [17]. **Search-result content blocks**
  let a custom RAG tool return results that Claude cites with the source and title the
  caller provides [18].
- **Model-level support checking.** Self-RAG's ISSUP token predicts whether a passage
  supports a generated segment, and the decoder can weight it more heavily [26].
- **Evaluation.** RAGAS defines reference-free *faithfulness* (the share of answer
  statements inferable from the context), *answer relevance* and *context relevance*, each
  scored by an LLM [29].

### 10.7 The cross-cutting loop: evaluate, monitor, recalibrate

Barnett et al.'s experience report (three case studies, 15,000 documents, 1,000 questions)
reaches two conclusions. RAG validation is only feasible during operation, and robustness
"evolves rather than [is] designed in" [14]. Among their lessons: RAG systems need
continuous calibration of chunk size, embedding strategy, chunking strategy, retrieval
strategy, consolidation strategy, context size and prompts. A semantic cache pre-filled
with FAQs cut cost and latency. Jailbreaks can bypass RAG grounding. Open-source embedding
models matched closed ones on small texts [14].

### 10.8 Failure-point map

Barnett et al.'s seven failure points [14] map onto the reference stages:

| # | Failure point [14] | Stage where it originates | Typical mitigation (sources) |
|---|---|---|---|
| FP1 | Missing content | Ingest / corpus scope | coverage audits; "I don't know" behaviour; CRAG web fallback [27] |
| FP2 | Missed top-ranked documents | Index / retrieve | hybrid retrieval, contextual chunks, larger N [16][11] |
| FP3 | Not in context (consolidation limits) | Rerank / context assembly | rerank to top-K; adjust K [16] |
| FP4 | Not extracted (noise, contradiction) | Generate | ordering at edges; compression; fewer, better chunks [15][12] |
| FP5 | Wrong format | Generate | structured output instructions; validation |
| FP6 | Incorrect specificity | Query understanding / generate | query rewriting; routing [35][12] |
| FP7 | Incomplete | Retrieve / generate | decompose multi-part questions; iterative retrieval [14][12] |

Two failure modes from the research systems complete the map. **Retrieval collapse**: a
jointly trained retriever learns to ignore its input [1]. **Leakage**: retrieval can copy
answers from evaluation data, which inflates scores [4][9].

---

## 11. Design lessons drawn across the canon

1. **Decouple the document encoder from training.** RAG, REALM fine-tuning, Atlas few-shot
   and RETRO all freeze the document side to avoid re-indexing [1][2][9][4]. In production
   the equivalent rule is to treat the embedding model as a versioned dependency, because
   changing it means re-embedding the corpus.
2. **The fusion point sets cost and quality.** Input concatenation (RAG, in-context RALM)
   is simple and model-agnostic. Decoder fusion (FiD/Atlas) scales to 100 passages but is
   bound by memory bandwidth [7][20]. Chunked cross-attention (RETRO) scales to trillions of
   tokens but needs architectural surgery [4].
3. **Retrieval quality is mostly decided before the LLM is called.** In Anthropic's
   experiment, contextual chunking, hybrid search and reranking together cut retrieval
   failure by about two thirds [16]. DPR's top-20 gain over BM25 (78.4 vs. 59.1) mattered
   more than reader changes in its era [11].
4. **Index currency is a feature and a risk.** Swapping the index updates knowledge without
   training [1]. Temporal mismatch costs accuracy [9], and stale indexes within training
   hurt badly [2].
5. **More context is not free.** The U-shaped position effect [15], FiDO's bandwidth
   bottleneck [20] and Self-Route's cost analysis [30] all point the same way: budget
   context deliberately.

---

## Key takeaways for system designers

- **Start from the six-stage reference pipeline** (ingest, index, retrieve, rerank,
  generate, cite) and instrument each stage separately. Most production failures (FP1-FP3)
  happen before generation [14].
- **Default to hybrid retrieval** (dense plus BM25, fused) and add a reranker over a wide
  candidate set (about 100-150 candidates down to about 20). This is the best-documented
  quality gain for the cost: 49% and 67% reductions in retrieval failure in Anthropic's
  tests [16].
- **Fix chunk context at ingest.** Contextualized chunks, section-aware passages,
  linearized tables and infoboxes, and file and chunk metadata are cheap one-time costs
  with measurable recall gains [16][9][14].
- **Treat the embedding model and the corpus snapshot as versioned artifacts.** Re-embedding
  is hours of GPU time even at Wikipedia scale [11]. Temporal alignment moves accuracy by
  several points [9]. PQ compression is close to free (about 10x) [9].
- **Order and trim context deliberately.** Put the strongest evidence at the edges, cap K
  by evaluation rather than habit, and consider routing between RAG and long context per
  query [15][30].
- **Make citations structural, not stylistic.** Prefer API- or model-level attribution
  with verifiable spans [17][18][26] over asking the model to "cite sources", and measure
  faithfulness continuously [29].
- **Pick the paradigm by need, not fashion.** Naive RAG suits small, homogeneous corpora.
  Advanced RAG is the production baseline. Modular or agentic flows earn their complexity
  only when queries need routing, multiple hops or multiple tools [12][13][31].
- **Fine-tuning and RAG are complements.** RA-DIT [23], RETROfitting [4] and InstructRetro
  [25] all show that lightweight training to *use* retrieved evidence adds gains on top of
  retrieval itself.
- **Guard evaluations against leakage.** Measure n-gram overlap between corpus and test
  set, as RETRO and Atlas did [4][9]. Otherwise retrieval quietly memorizes the benchmark.

---

## References

1. P. Lewis, E. Perez, A. Piktus, et al. "Retrieval-Augmented Generation for
   Knowledge-Intensive NLP Tasks." Facebook AI Research / UCL / NYU, NeurIPS 2020.
   https://arxiv.org/abs/2005.11401
2. K. Guu, K. Lee, Z. Tung, P. Pasupat, M.-W. Chang. "REALM: Retrieval-Augmented Language
   Model Pre-Training." Google Research, ICML 2020. https://arxiv.org/abs/2002.08909
3. Google Research. "REALM" code release README (google-research/language). 2020.
   https://github.com/google-research/language/tree/master/language/realm
4. S. Borgeaud, A. Mensch, J. Hoffmann, et al. "Improving Language Models by Retrieving
   from Trillions of Tokens" (RETRO). DeepMind, ICML 2022 (arXiv Dec 2021).
   https://arxiv.org/abs/2112.04426
5. DeepMind. "Improving language models by retrieving from trillions of tokens" (blog).
   2021. https://deepmind.google/discover/blog/improving-language-models-by-retrieving-from-trillions-of-tokens/
6. J. Alammar. "The Illustrated Retrieval Transformer." 2022.
   http://jalammar.github.io/illustrated-retrieval-transformer/
7. G. Izacard, E. Grave. "Leveraging Passage Retrieval with Generative Models for Open
   Domain Question Answering" (Fusion-in-Decoder). Facebook AI Research, EACL 2021.
   https://arxiv.org/abs/2007.01282
8. Facebook Research. "FiD: Fusion-in-Decoder and reader-to-retriever distillation" code
   repository README. 2020-2021. https://github.com/facebookresearch/FiD
9. G. Izacard, P. Lewis, M. Lomeli, et al. "Atlas: Few-shot Learning with Retrieval
   Augmented Language Models." Meta AI, JMLR 2023 (arXiv Aug 2022).
   https://arxiv.org/abs/2208.03299
10. Facebook Research. "Atlas" code repository README. 2022-2023.
    https://github.com/facebookresearch/atlas
11. V. Karpukhin, B. Oguz, S. Min, et al. "Dense Passage Retrieval for Open-Domain Question
    Answering." Facebook AI, EMNLP 2020. https://arxiv.org/abs/2004.04906
12. Y. Gao, Y. Xiong, X. Gao, et al. "Retrieval-Augmented Generation for Large Language
    Models: A Survey." Tongji University / Fudan University, 2023-2024.
    https://arxiv.org/abs/2312.10997
13. Y. Gao, Y. Xiong, M. Wang, H. Wang. "Modular RAG: Transforming RAG Systems into
    LEGO-like Reconfigurable Frameworks." 2024. https://arxiv.org/abs/2407.21059
14. S. Barnett, S. Kurniawan, S. Thudumu, Z. Brannelly, M. Abdelrazek. "Seven Failure Points
    When Engineering a Retrieval Augmented Generation System." Deakin University, CAIN 2024.
    https://arxiv.org/abs/2401.05856
15. N. F. Liu, K. Lin, J. Hewitt, et al. "Lost in the Middle: How Language Models Use Long
    Contexts." Stanford et al., TACL 2024 (arXiv 2023). https://arxiv.org/abs/2307.03172
16. Anthropic. "Introducing Contextual Retrieval." 2024.
    https://www.anthropic.com/news/contextual-retrieval
17. Anthropic. "Citations" (Claude Platform documentation). Accessed 2026.
    https://docs.claude.com/en/docs/build-with-claude/citations
18. Anthropic. "Search results" (Claude Platform documentation). Accessed 2026.
    https://docs.claude.com/en/docs/build-with-claude/search-results
19. O. Khattab, M. Zaharia. "ColBERT: Efficient and Effective Passage Search via
    Contextualized Late Interaction over BERT." Stanford, SIGIR 2020.
    https://arxiv.org/abs/2004.12832
20. M. de Jong, Y. Zemlyanskiy, J. Ainslie, et al. "FiDO: Fusion-in-Decoder Optimized for
    Stronger Performance and Faster Inference." USC / Google Research, Findings of ACL 2023.
    https://arxiv.org/abs/2212.08153
21. O. Ram, Y. Levine, I. Dalmedigos, et al. "In-Context Retrieval-Augmented Language
    Models." AI21 Labs, TACL 2023. https://arxiv.org/abs/2302.00083
22. W. Shi, S. Min, M. Yasunaga, et al. "REPLUG: Retrieval-Augmented Black-Box Language
    Models." University of Washington / Meta et al., NAACL 2024.
    https://arxiv.org/abs/2301.12652
23. X. V. Lin, X. Chen, M. Chen, et al. "RA-DIT: Retrieval-Augmented Dual Instruction
    Tuning." Meta FAIR, ICLR 2024. https://arxiv.org/abs/2310.01352
24. B. Wang, W. Ping, P. Xu, et al. "Shall We Pretrain Autoregressive Language Models with
    Retrieval? A Comprehensive Study." NVIDIA et al., EMNLP 2023.
    https://arxiv.org/abs/2304.06762
25. B. Wang, W. Ping, L. McAfee, et al. "InstructRetro: Instruction Tuning post
    Retrieval-Augmented Pretraining." NVIDIA, 2023. https://arxiv.org/abs/2310.07713
26. A. Asai, Z. Wu, Y. Wang, A. Sil, H. Hajishirzi. "Self-RAG: Learning to Retrieve,
    Generate, and Critique through Self-Reflection." UW / AI2 / IBM, ICLR 2024.
    https://arxiv.org/abs/2310.11511
27. S.-Q. Yan, J.-C. Gu, Y. Zhu, Z.-H. Ling. "Corrective Retrieval Augmented Generation."
    2024. https://arxiv.org/abs/2401.15884
28. L. Gao, X. Ma, J. Lin, J. Callan. "Precise Zero-Shot Dense Retrieval without Relevance
    Labels" (HyDE). CMU / Waterloo, ACL 2023. https://arxiv.org/abs/2212.10496
29. S. Es, J. James, L. Espinosa-Anke, S. Schockaert. "RAGAS: Automated Evaluation of
    Retrieval Augmented Generation." EACL 2024 demo (arXiv 2023).
    https://arxiv.org/abs/2309.15217
30. Z. Li, C. Li, M. Zhang, Q. Mei, M. Bendersky. "Retrieval Augmented Generation or
    Long-Context LLMs? A Comprehensive Study and Hybrid Approach" (Self-Route). Google
    DeepMind / U. Michigan, EMNLP 2024 Industry. https://arxiv.org/abs/2407.16833
31. A. Singh, A. Ehtesham, S. Kumar, T. Talaei Khoei. "Agentic Retrieval-Augmented
    Generation: A Survey on Agentic RAG." 2025 (revised 2026).
    https://arxiv.org/abs/2501.09136
32. "SoK: Agentic Retrieval-Augmented Generation (RAG): Taxonomy, Architectures, Evaluation,
    and Research Directions." 2026. https://arxiv.org/abs/2603.07379
33. Hugging Face. "RAG" model documentation (Transformers). 2020-2026.
    https://huggingface.co/docs/transformers/model_doc/rag
34. O. Khattab, K. Santhanam, X. L. Li, et al. "Demonstrate-Search-Predict: Composing
    Retrieval and Language Models for Knowledge-Intensive NLP." Stanford, 2022.
    https://arxiv.org/abs/2212.14024
35. X. Ma, Y. Gong, P. He, H. Zhao, N. Duan. "Query Rewriting for Retrieval-Augmented Large
    Language Models" (Rewrite-Retrieve-Read). EMNLP 2023. https://arxiv.org/abs/2305.14283
