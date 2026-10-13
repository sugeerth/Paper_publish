# 10. Multimodal RAG: Visual Document Retrieval, Cross-Modal Embeddings, and Video and Audio Retrieval

## Overview

Most of this atlas treats the corpus as text. Real corpora are not text. An investor deck
carries its key numbers in bar charts. A 10-K puts its comparisons in tables. A maintenance
manual explains with exploded diagrams. A lecture archive is hours of video, and its spoken
explanation exists only in the audio. A text-only RAG pipeline first converts these
artefacts into text by parsing, OCR, captioning or transcription. Whatever the conversion
drops (the shape of a trend line, the row a number sits in, the slide a speaker points at)
is gone before retrieval starts. Weaviate's 2026 write-up on chart-heavy PDFs puts it
directly: charts and tables "are *primary information* that gets destroyed in extraction"
[29].

Multimodal RAG covers the designs that keep non-text signal available to retrieval and to
generation. Since 2023 the field has split into four architectural families:

1. **Convert-to-text (parse, caption, transcribe).** Every modality is turned into text or
   structured text, which is then indexed with the text machinery of Chapters 2-7. Chapter 2
   covers the parsers. This chapter covers the multimodal extensions: element-level
   captioning, table interpretation, ASR, and multimodal knowledge graphs (RAG-Anything,
   VideoRAG, Video-RAG) [33][47][49].
2. **Single-vector screenshot embedding.** A page, slide or screenshot is rendered as an
   image and encoded by a vision-language model (VLM) into one dense vector. Examples are
   DSE, vdr-2b, VisRAG-Ret, VDocRetriever and UniSE, and API models such as Gemini
   Embedding 2 [8][10][11][14][17][46].
3. **Multi-vector late interaction over image patches.** This is the ColPali family. Each
   patch of a page image becomes a 128-dimensional vector, and query tokens are scored
   against patches with ColBERT's MaxSim. It is the current state of the art on
   visual-document benchmarks, and it is also the most expensive to serve [1][2][27].
4. **Universal or omni-modal embedding spaces.** CLIP, ImageBind, CLAP, UniIR, VLM2Vec,
   Qwen3-VL-Embedding and ColQwen-Omni map text, images, audio and video into one space, so
   that any modality can retrieve any other [28][37][38][39][40][43].

In practice the generator side has converged. Retrieved pages, frames or clips go to a VLM
as images, together with any extracted text. This chapter is organised by retrieval
design. It covers the ColPali lineage in depth, then single-vector screenshot retrievers,
benchmarks, serving at scale, end-to-end visual RAG systems, convert-to-text multimodal
pipelines, universal embedding spaces, video RAG and audio RAG. It ends with a comparison
table and takeaways.

```
                       MULTIMODAL RAG: FOUR RETRIEVAL FAMILIES
 source artefact                index representation                     generator
+----------------+   (1) parse/OCR/caption/ASR -> text chunks (+KG)   +--------------+
| PDF page       |---------------------------------------------------->|              |
| slide, chart   |   (2) render -> VLM -> 1 dense vector per page      |   VLM / LLM  |
| table, figure  |---------------------------------------------------->|  (page img,  |
| screenshot     |   (3) render -> VLM -> ~700-1,030 patch vectors     |   frames,    |
| video, audio   |---------------------------------------------------->|  transcript, |
|                |   (4) any modality -> shared omni-modal space       |   captions)  |
+----------------+---------------------------------------------------->+--------------+
```

## 10.1 ColPali: Late Interaction over Page Images

### Problem

Before 2024, PDF RAG meant a brittle chain of PDF parsing, OCR, layout detection, table
recognition, figure captioning, chunking and text embedding. The ColPali authors measured
the cost of that chain. In their benchmark the parse, OCR and captioning pipeline took
roughly 7.2 seconds of offline processing per page, against roughly 0.4 seconds per page to
embed the page image directly (approximate, as reported in the paper's latency figure,
which Hugging Face reproduces in its 2025 VLM survey) [1][35]. Even after all that work,
the parsed pipelines still lost to direct visual retrieval on visually rich pages [1].

### Architecture

ColPali (Faysse, Sibille, Wu et al., ILLUIN Technology / CentraleSupélec, 2024) takes the
PaliGemma-3B VLM and turns it into a ColBERT-style retriever [1][2]:

- **Vision encoder.** SigLIP-So400m resizes each page to 448x448 pixels and cuts it into
  14x14-pixel patches. That gives a 32x32 grid of 1,024 patches [5].
- **Language model.** Gemma-2B contextualises the patch features together with a short
  instruction prompt.
- **Projection.** A linear layer maps every output token to a 128-dimensional,
  L2-normalised vector. The page embedding is **1,030 vectors x 128 dims**: 1,024 patch
  vectors plus 6 vectors from the instruction and special tokens [5][18].
- **Query side.** The same model encodes the text query into one 128-dim vector per token.
- **Scoring.** MaxSim, from ColBERT [31]. For each query token, take the maximum dot
  product over all document vectors, then sum those maxima over the query tokens.

```
 OFFLINE (per page)                               ONLINE (per query)
 PDF -> rasterise -> 448x448 image                "which year has highest outlay?"
            |                                                 |
   SigLIP-So400m (32x32 = 1,024 patches)            Gemma tokenizer + LM
            |                                                 |
   Gemma-2B contextualisation (+ prompt)            linear proj -> n_q x 128
            |                                                 |
   linear proj -> 1,030 x 128 (bf16/fp32)                     |
            |                                                 v
   multi-vector store ---------------------->  MaxSim(Q,D) = sum_i max_j <q_i, d_j>
                                                              |
                                               top-k page IDs -> page images -> VLM
```

Training follows ColBERT's in-batch contrastive objective on query-page pairs. The pairs
come from academic DocVQA-style sets plus synthetic queries that a VLM generated over
web-crawled PDFs [1]. The model lineage shows how much the recipe matters. ColPali v1.3
keeps the same backbone as v1.1 but trains with an effective batch size of 256 for 3
epochs, and its ViDoRe score rises from 81.5 to 84.8 [2].

### Interpretability

MaxSim is a sum of per-query-token maxima, so every point of a page's score can be
attributed to one query token and one patch. ColPali's "similarity maps" overlay this on
the page as a heatmap. In the Sentence Transformers example, the query token "water"
lights up the "Water Resources & Power" bar of a 1971 budget chart [2][27]. This is the
main explainability advantage that late interaction has over single-vector retrievers,
and production demos (Vespa's ColPali app, Weaviate's Query Agent) surface it to users
[23][29].

### Failure modes and lessons

- **Storage blow-up.** 1,030 x 128 float32 values is 512 KB per page, so 1M pages need
  about 512 GB before compression. A single 768-dim vector needs 3 KB per page, which
  makes ColPali roughly 170x larger [20].
- **Index-construction blow-up.** HNSW assumes a symmetric, query-independent distance.
  MaxSim is neither, so vector databases index the individual patch vectors. Inserting
  one ~700-vector page with `ef_construct = 100` takes on the order of 49M vector
  comparisons [19][21].
- **Fixed resolution.** At 448x448, dense small print is downsampled. Later models moved
  to dynamic resolution (Section 10.2).
- **Special-token contribution.** Non-content tokens contribute to MaxSim. In one text
  example, three special tokens made up 22.7% of the score [27]. Qdrant kept the 6 special
  vectors in its pooled representation for this reason [18].
- **Lesson.** Page-as-chunk removes chunking decisions, but the page becomes the unit of
  retrieval and of generator context. Weaviate notes that when the answer is one paragraph
  of a dense contract, a whole page "may be more context than you want" [29].

## 10.2 The ColVision Family: 2024-2026 Evolution

After ColPali, a family of "Col*" retrievers swapped the backbone, the resolution policy,
the attention direction or the training data. The colpali-engine repository and the
Qdrant multi-vector course document the main variants [2][6]:

| Model | Backbone | Params | Notable design choice | ViDoRe score* | License |
|---|---|---|---|---|---|
| ColPali v1.0 / v1.3 | PaliGemma-3B | ~3B | fixed 448 px, 1,030 vecs/page | 81.3 / 84.8 | Gemma |
| ColQwen2 v1.0 | Qwen2-VL-2B | ~2B | dynamic resolution, up to 768 patches | 89.3 | Apache 2.0 |
| ColQwen2.5 v0.2 | Qwen2.5-VL-3B | ~3B | dynamic res, 768 patches in training | 89.4 | Apache 2.0 |
| tomoro-colqwen3-embed-4b | Qwen3-VL | ~4B | 320-dim vectors; 8B sibling handles short video | 90.6 | Apache 2.0 |
| colqwen3.5-4.5B-v3 | Qwen3.5-4B (hybrid GatedDeltaNet) | 4.5B | LoRA-trained, 320-dim | 90.9 | Apache 2.0 |
| ColSmol-256M / 500M | SmolVLM | 256M / 500M | edge/browser deployments | 80.1 / 82.3 | Apache 2.0 |
| ColFlor | Florence-2 | 174M | 17x smaller, up to 9.8x faster than ColPali | n/a | n/a |
| ColModernVBERT | ModernVBERT (bidirectional) | ~250M | bidirectional attention encoder | n/a | n/a |
| ColNetraEmbed | Gemma-3-4B | ~4B | 22 languages | 86.4 | Gemma |
| ColNomic-Embed-Multimodal | (Nomic) | 3B / 7B | open, commercial-use multilingual | n/a | open |
| Llama-NeMoRetriever-ColEmbed-3B | SigLIP-2 + Llama 3.2-3B | ~3B | multilingual; non-commercial license | n/a | NVIDIA NC |

*Scores are as listed in the colpali-engine README's ViDoRe column [2]. Other sources do
not report comparable numbers, so those rows show n/a. ColFlor, ColNomic and NVIDIA
details are from the Qdrant course [6].

Four design lessons stand out in this lineage:

1. **Dynamic resolution beats a fixed grid.** ColQwen2 caps a page at 768 patches but keeps
   its aspect ratio, and it gained about 5 points over ColPali v1.2 on the same benchmark.
   In Qdrant's measurements it typically produces about 700 vectors per page instead of
   1,030 [2][19].
2. **Bidirectional encoders are more parameter-efficient retrievers.** ModernVBERT
   (Teiletche, Macé, Conti, Loison, Viaud, Colombo, Faysse, 2025) argues that causal
   decoders are a poor fit for embeddings. ColModernVBERT has about 250M parameters, more
   than 10x fewer than ColPali, and scores only slightly lower on ViDoRe [6][7].
3. **Batch size and compute matter as much as architecture.** ColQwen2 v1.0 differs from
   v0.1 mainly in "more powerful GPUs" and an effective batch of 256 instead of 32. The
   result was +2.0 points [2]. Fused MaxSim kernels (`late-interaction-kernels`) avoid
   materialising the `[B, B, Lq, Ld]` score tensor, and they doubled the largest trainable
   batch on an 80 GB H100 from 64 to 128 [2].
4. **Ecosystem consolidation.** In 2026, ILLUIN deprecated `colpali-engine` in favour of
   Sentence Transformers v6, whose `MultiVectorEncoder` loads ColPali-family, PyLate and
   Stanford ColBERT checkpoints behind one `encode_query` / `encode_document` /
   `similarity` API [2][27]. Byaldi offered the earlier RAGatouille-style wrapper, with
   uncompressed indexes [51].

## 10.3 Single-Vector Screenshot Retrievers

### Document Screenshot Embedding (DSE)

DSE (Ma, Lin, Li, Chen, Lin, University of Waterloo, 2024) proposed the simplest possible
unification: treat every document, whether a Wikipedia page, a slide or a PDF page, as a
**screenshot**, and encode it with a VLM into **one** vector by end-of-sequence pooling
[8]. The Tevatron recipe fine-tunes Phi-3-vision with LoRA, and later Qwen2-VL-2B, on
`wiki-ss-nq` screenshot/question pairs. It uses temperature 0.02, `train_group_size 2`
(one positive and one hard negative), `query_max_len 128` and `passage_max_len 4096`
tokens for the image side [9]. DSE avoids every multi-vector cost: one vector per page fits
any ANN index. The price is that a chart, a table and three paragraphs must share a
single vector [27][35].

### vdr-2b-multi (LlamaIndex, 2025)

`vdr-2b-multi-v1` starts from `dse-qwen2-2b-mrl-v1` and is a useful case study in data
engineering for visual retrievers [10]:

- **Corpus construction.** Topic-driven search queries per language found about 50k PDFs.
  A layout-analysis model labelled each page as text-only, visual-only or mixed, and about
  100k pages were sampled evenly across those three types.
- **Query synthesis.** Gemini-1.5-Pro and Qwen2-VL-72B each wrote a "specific" and a
  "general" question per page. Only the specific question was kept, but forcing the
  contrast produced sharper questions.
- **Filtering.** A specific query was kept only if its paired general query appeared in
  the top 100 results of a voyage-3 index. This removed about 40% of queries per language.
  Hard negatives were mined with voyage-3 at a 0.75 similarity threshold.
- **Result.** The dataset holds 496,167 pages, 280,679 of them with queries. The model
  averages +2.3% nDCG@5 over its base across languages, and +6.33% on German visual-only
  pages. At 768 image tokens the English model beats the base model run at 2,560 patches,
  which the authors report as about 3x faster inference. Matryoshka truncation to one
  third of the dimensions keeps about 98% of quality [10].

### VisRAG-Ret, VDocRetriever and UniSE

- **VisRAG-Ret** (OpenBMB/Tsinghua, 2024) builds on MiniCPM-V 2.0 (SigLIP plus
  MiniCPM-2B). It is trained on 362,110 query-document pairs: 34% from academic datasets
  and 66% synthetic GPT-4o queries over web-crawled PDFs [11][12].
- **VDocRetriever** (NTT, CVPR 2025) adds two self-supervised pre-training tasks, RCR and
  RCG, that "compress visual information into dense token representations while aligning
  them with textual content". It is released with OpenDocVQA, a unified open-domain
  DocVQA collection [14].
- **UniSE / VisIR** (BAAI, 2025) generalises the screenshot idea to "Visualized
  Information Retrieval". Text, images, tables and charts are all represented as
  screenshots. The work releases the VIRA training data and the MVRB benchmark, and its
  queries can themselves be screenshot-plus-text [17].

### API-hosted multimodal embedders

Google's **Gemini Embedding 2** (public preview, March 2026) maps text, images, video,
audio and PDFs into one space. Its input limits are: up to 6 PNG/JPEG images per request,
MP4/MOV video up to 128 s, MP3/WAV audio up to 80 s, PDFs up to 6 pages, and 8,192 input
tokens. Output dimensions run from 128 to 3,072 via Matryoshka learning. Qdrant recommends
a two-pass design: retrieve on a truncated vector, then rescore with the full 3,072 dims
[46].

## 10.4 Benchmarks: From ViDoRe V1 to Pipeline-Level Evaluation

**ViDoRe V1** (2024) combined academic sets (DocVQA, InfoVQA, TAT-DQA, ArxivQA, TabFQuAD)
with synthetic "practical" sets (energy, government, healthcare, AI, Shift Project). These
quickly saturated: `vdr-2b` scores 95-98 nDCG@5 on several of them [10]. **ViDoRe V2**
(Macé, Loison, Faysse, 2025) "raised the bar" with harder, more realistic queries [3].
**ViDoRe V3** has more than 26,000 pages, 3,099 queries in 6 languages and 10 datasets
(8 public and 2 private). It covers HR, finance (English and French), industrial,
pharmaceutical, computer-science, energy and physics documents, all with human-verified
annotations [6]. Single-model evaluation has moved into MTEB. The `vidore-benchmark`
repository now evaluates **whole pipelines** (OCR, retriever, reranker, agent) and records
indexing and search times [4].

The public V3 pipeline submissions are the best evidence available on the trade-offs
between OCR, visual and agentic designs. The table below summarises them. nDCG@10 is the
mean of the "overall" scores across the 8 public English splits. Per-query search time is
total search time on the HR split divided by its 318 queries. All runs used a single
A100-80GB GPU, but CPU and RAM differed between submitters. The figures were computed by
this chapter from the submitted JSON files [4].

| Pipeline (submitter) | Modality indexed | nDCG@10 (mean of 8) | Index time, HR | Search time/query |
|---|---|---|---|---|
| mxbai-edge-colbert-v0-32m (Mixedbread) | OCR text, late interaction | 0.440 | 5 s | ~4 ms |
| llama-nemotron-embed-vl-1b-v2 (NVIDIA) | page image, 1 vector | 0.524 | 382 s | ~3 ms |
| qwen3-embedding-8B | OCR text, 1 vector | 0.535 | 177 s | ~40 ms |
| llama-nemotron-embed-vl-1b-v2 (NVIDIA) | image + OCR text | 0.558 | 261 s | ~4 ms |
| nemotron-embed-vl-1b + rerank-vl-1b (NVIDIA) | page image | 0.578 | 201 s | ~5.2 s |
| jina-embeddings-v4 + jina-reranker-m0 | page image | 0.601 | 549 s | ~17 s |
| llama-nemotron-colembed-vl-3b-v2 (NVIDIA) | image, late interaction | 0.609 | 671 s | ~0.46 s |
| nemotron-embed-vl-1b + rerank-vl-1b (NVIDIA) | image + OCR text | 0.620 | 274 s | ~6.2 s |
| nemotron-colembed-vl-8b-v2 (NVIDIA) | image, late interaction | 0.644 | 492 s | ~1.1 s |
| jina-embeddings-v4 (text) + zerank-2 | OCR text | 0.656 | 161 s | ~9.8 s |
| NeMo Retriever Agentic: colembed-8b + Claude Opus 4.5 | image, agentic loop | 0.692 | 68 s** | ~102 s |

**The agentic run's indexing time is reported as submitted. Its configuration preloads
the corpus to the GPU [4].

Five observations follow from these numbers:

1. **Late interaction beats single-vector at equal modality.** The 3B ColEmbed (0.609)
   beats the 1B single-vector image embedder (0.524) by about 8.5 points. Its search cost
   is two orders of magnitude higher, though still sub-second.
2. **Adding OCR text to image embeddings helps.** Image plus text beats image alone both
   without a reranker (0.558 vs 0.524) and with one (0.620 vs 0.578). The modalities are
   complementary, not substitutes.
3. **A strong text pipeline is still competitive.** jina-v4 on transcribed text with the
   zerank-2 reranker (0.656) beats the same embedder on page images with a visual
   reranker (0.601), and edges out the 8B visual ColEmbed (0.644). On V3's text-heavy
   enterprise documents, "OCR is dead" is not supported.
4. **Rerankers are expensive.** The 1B VL reranker adds about 5-6 s per query on the HR
   set for 5-6 points.
5. **Agentic retrieval wins on quality at about 100x the latency.** The NVIDIA agent
   rewrites queries, calls the ColEmbed-8B retriever (top-500) repeatedly, and has
   Claude Opus 4.5 select and rank the final 10 pages. It reaches the best score, 0.692,
   at roughly 100 s per query, against 1.1 s for the same retriever alone [4][53]. This
   is the visual counterpart of the agentic-RAG trade-offs in Chapter 8.

## 10.5 Serving Late-Interaction Visual Retrieval at Scale

The central engineering problem of ColPali-style RAG is cost. Its storage and compute are
proportional to the number of vectors per page. Every production design attacks one of
three terms: **vectors per page**, **bytes per vector**, or **pages scored per query**.

```
 cost ~ (#pages scored) x (#vectors/page) x (bytes/vector) x (#query tokens)
          |                    |                  |
          |                    |                  +-- scalar int8 (4x), binary (32x),
          |                    |                      PLAID residuals [20][27][32]
          |                    +-- token pooling: hierarchical (3x), row/col mean (32x),
          |                        k-means [2][18][26]
          +-- multi-stage: MUVERA / pooled / dense prefetch -> full MaxSim rerank
              on 100-1,000 candidates [19][22][24]
```

### Reducing vectors per page: token pooling

- **Hierarchical token pooling** (Clavié, Chaffin, Adams, 2024) clusters a document's
  vectors with Ward linkage on cosine distance and replaces each cluster with its mean
  [26][27]. It is "CRUDE-compliant": each document is pooled independently, so additions
  and deletions need no global retraining. For ColPali, the ILLUIN team found that a
  pool factor of 3 removed 66.7% of the vectors while keeping 97.8% of retrieval quality
  [2]. LightOn later trained models with a hierarchical-pooling regulariser and reports
  99.4% retention at 5x compression on text [27].
- **Grid pooling** uses the spatial layout of patches. Mean-pooling each row of
  ColPali's 32x32 grid gives 32 vectors per page (plus the 6 special vectors, 38 in
  total). That is 8 KB per page instead of 256 KB in float16. Column pooling captures
  vertical structure such as table borders. Using both gives 64 vectors [18][20].
- **Lesson.** Mean pooling preserves ranking and max pooling does not. In Qdrant's
  experiment, mean-pooled first-stage retrieval followed by full-resolution reranking
  reached NDCG@20 = 0.952 and Recall@20 = 0.917 relative to unpooled ColPali. With max
  pooling the figures were 0.759 and 0.656 [18].

### Reducing bytes per vector: quantisation

Qdrant's course works through the arithmetic for 1M pages of ~1,024 x 128 vectors. Float32
needs 512 GB. Scalar int8 cuts that by 4x to 128 GB with less than 1% accuracy loss.
Binary quantisation cuts it by 32x, to 16 KB per page [20]. Vespa's ColPali sample app
goes all the way to binary. Each patch vector is stored as `tensor<int8>(patch{}, v[16])`,
which is 128 sign bits packed into 16 bytes, with Hamming distance and an HNSW index. The
query is sent twice: as binarised tokens for a cheap Hamming MaxSim in the first phase
(`1/(1+hamming)` summed over the per-token maxima), and as float tokens for an exact
MaxSim against `unpack_bits(embedding)` in a second phase limited to `rerank-count: 10`
[23]. PLAID (ColBERTv2) stores a centroid ID plus a quantised residual per vector. A
608,414-vector text index shrinks from 311.5 MB (float32) to 92 MB as a fast-plaid index
[27][32]. The same residual compression took ColBERT's MS MARCO index from 154 GB to
16-25 GB (1- and 2-bit residuals) [30].

### Reducing pages scored: multi-stage retrieval and MUVERA

- **Pooled or dense prefetch, then MaxSim rerank.** Qdrant's recommended design builds HNSW
  only over the pooled vectors, prefetches about 200 candidates, and reranks them with the
  original multi-vectors. On a 20,000-page corpus (ViDoRe, UFO and DocVQA) it reports
  **13x faster** retrieval and an order-of-magnitude faster indexing [18][19]. Oversample
  in the prefetch: if both stages fetch 10, the reranker can never recover the 11th
  document [22].
- **MUVERA** (Google Research) turns a multi-vector set into one **fixed-dimensional
  encoding (FDE)**. It SimHash-partitions the token vectors into buckets using k_sim random
  hyperplanes, aggregates each bucket, applies random +/-1 projections to d_proj dims, and
  concatenates the results [24][25]. A standard single-vector HNSW then retrieves on the
  FDE, and exact MaxSim reranks. For 1M documents of 100 vectors each, the HNSW graph is
  1% of the multi-vector graph's size. Weaviate 1.31 reports about 70% lower memory and
  import times falling from over 20 minutes to 3-6 minutes, with "some loss in recall"
  that can be recovered by raising HNSW `ef` [24]. Qdrant's FastEmbed (0.7.2+) ships
  MUVERA as a post-processor for any late-interaction model, including ColModernVBERT
  [22].

### Database support (2024-2026)

Vespa (tensor ranking expressions), Qdrant (multivector with `MAX_SIM` comparator), Weaviate
(multi-vector plus MUVERA, and the hosted `multi2multivec-weaviate` vectoriser), Milvus,
Elasticsearch, LanceDB and AstraDB all have documented ColPali integrations [2][19][23][29].
Weaviate's 2026 drag-and-drop pipeline renders each PDF page to a 2,000-pixel PNG, stores
it as a BLOB and vectorises it server-side. Its 92-page NVIDIA earnings demo imported in
about 90 seconds. The team found no strong case for rendering below 1,500 px or above
2,500 px [29].

## 10.6 End-to-End Visual RAG Systems

### M3DocRAG (UNC Chapel Hill and Bloomberg, 2024)

M3DocRAG targets *open-domain*, multi-page, multi-document question answering, where the
evidence may sit only in a figure [15][16]. It has three stages: (1) ColPali v1.2 embeds
every page, (2) MaxSim retrieves the top-K pages, with approximate page indices for
open-domain speed, and (3) Qwen2-VL-7B answers over the retrieved page images. The authors
built **M3DocVQA**, which covers more than 3,000 PDFs and 40,000 pages, and report
state-of-the-art results on MP-DocVQA [15][16]. The design lesson: a multimodal retriever
lets a VLM with limited context "handle single or many documents while preserving visual
information", instead of stuffing whole documents into context.

### VisRAG and EVisRAG (OpenBMB, 2024-2026)

VisRAG pairs VisRAG-Ret (Section 10.3) with a VLM generator. The paper used MiniCPM-V 2.0,
MiniCPM-V 2.6 and GPT-4o, and studied how to feed several retrieved page images to the
generator [11][12]. **EVisRAG (VisRAG 2.0)**, accepted at ACM MM 2026, attacks the next
bottleneck: multi-image reasoning. The VLM first "linguistically observes" each retrieved
image to collect per-image evidence, then reasons over that evidence. It is trained with
**Reward-Scoped GRPO**, which binds token-level rewards to perception tokens and to
reasoning tokens separately. 3B and 7B checkpoints on Qwen2.5-VL are released [12][13].

### VDocRAG (NTT, CVPR 2025)

VDocRAG pairs VDocRetriever with VDocGenerator, both Phi-3-vision with LoRA, so that a
single pipeline runs "purely from visual features". Pages are resized to 1,344x1,344.
Queries carry an instruction prefix ("I'm looking for an image that answers the
question") [14].

### Production patterns: Vespa, Weaviate and NVIDIA

```
 PDF/slide decks -> rasterise (1,500-2,500 px) -> page store (BLOB/object storage)
                         |
                         v
              Col* / DSE embedder (GPU)  -> [pooled|FDE|binary] ANN index
                                         -> full-precision multivectors (rerank store)
 query -> query encoder -> stage 1: ANN prefetch (100-1,000) -> stage 2: MaxSim rerank
       -> optional VL reranker / LLM-agent selection (top 5-10)
       -> VLM generator(page images + query) -> answer + page-image citations + heatmaps
```

- **Vespa's ColPali demo app** combines BM25 over extracted title and text with binary
  MaxSim and full-precision reranking in "hybrid" rank profiles. It returns per-token
  similarity scores for heatmaps, and generates Gemini answers grounded in the retrieved page images [23].
- **Weaviate's Query Agent** wraps multi-vector page retrieval in multi-step reasoning. It
  returns a synthesised answer whose numerical claims are "anchored to a specific page in a
  specific PDF, with the page image right there for verification" [29].
- **NVIDIA's NeMo Retriever agentic pipeline** has an LLM agent with think, retrieve and
  select-and-rerank tools over a ColEmbed index (Section 10.4) [4][53].

### Failure modes

- **Generator context cost.** Each page image consumes hundreds to thousands of VLM input
  tokens, so top-k must stay small (5-10). Selection quality, whether from a reranker or an
  agent, then dominates answer quality.
- **Multi-image reasoning.** Generators confuse evidence across several pages. This is the
  motivation for EVisRAG [12].
- **Domain-specific visual conventions.** Weaviate warns that highly stylised internal
  templates should be validated before rollout [29].
- **Text-dominated corpora.** For contracts, transcripts or logs, "a text embedding model
  will still be cheaper and just as accurate" [29].

## 10.7 Convert-to-Text Multimodal RAG: Parsing, Captioning and Multimodal Graphs

Visual retrieval does not make parsing obsolete. Section 10.4 showed OCR-text pipelines
matching visual ones on ViDoRe V3. Many systems also need text for keyword search,
filtering, citations at sub-page granularity, or downstream LLMs that cannot read images.
The convert-to-text family has moved from "drop the images" to element-aware
interpretation.

### RAG-Anything (HKUDS, 2025)

RAG-Anything extends LightRAG (Chapter 7) into an "all-in-one" multimodal document RAG
system [33][34]. Its pipeline:

1. **Parsing.** MinerU decomposes PDFs and Office files into text blocks, images, tables
   and equations, and keeps the document hierarchy.
2. **Modality routing.** Each element type goes to a dedicated analyser. A *visual content
   analyser* (a VLM) writes context-aware captions and spatial relations. A *structured
   data interpreter* reads tables for trends and cross-table relations. A *math parser*
   handles LaTeX. Text and multimodal pipelines run concurrently.
3. **Multimodal knowledge graph.** Significant multimodal elements become entities.
   Cross-modal edges link text entities to figures and tables, `belongs_to` chains
   preserve section hierarchy, and relation types carry weights.
4. **Retrieval.** Vector-graph fusion combines embedding search with graph traversal and
   adds modality-aware ranking.

```
 PDF/DOCX/PPTX --MinerU--> {text | image | table | equation} + hierarchy
      text  ---------------------------------> chunks + entities --+
      image --VLM caption + spatial relations-> entity nodes ------+--> multimodal KG
      table --structured interpretation-------> entity nodes ------+    + vector index
      eqn   --LaTeX parse + concept mapping---> entity nodes ------+
 query -> vector search + graph traversal -> modality-aware rerank -> LLM/VLM answer
```

The trade-off is the reverse of ColPali's. Indexing is LLM- and VLM-heavy and
error-prone, because each caption is a lossy, possibly hallucinated summary. In exchange,
retrieval runs on cheap text and graph structures, and answers can cite individual
elements.

### Tables and charts as text

Modern VLM-based OCR models (olmOCR, PaddleOCR-VL and others) either keep charts as images
with coordinates, caption them inline, or convert a bar chart into a Markdown table or JSON
[36]. Converting charts to data makes the numbers searchable and quotable, but it discards
visual salience such as trends, outliers and annotations. Weaviate's hybrid
recommendation is pragmatic: "identify pages with charts to implement as multi-vectors and
the remaining corpus as text, and then use an RRF-style approach to merge results at query
time" [29]. Chapter 2 covers parser choice in depth.

## 10.8 Universal Cross-Modal Embedding Spaces

### Contrastive dual encoders: CLIP, ImageBind, CLAP

- **CLIP** (OpenAI, 2021) aligned image and text encoders contrastively. It is the
  default for natural-image retrieval and the ancestor of every model in this section
  [37].
- **ImageBind** (Meta FAIR, CVPR 2023) learns one embedding space over **six modalities**:
  images, text, audio, depth, thermal and IMU. It binds each modality to *frozen* image
  representations, so audio-to-text retrieval emerges without paired audio-text data
  [38][50]. VideoRAG uses ImageBind for video-segment features [48].
- **CLAP** (LAION, ICASSP 2023) applies the CLIP recipe to audio and text, with feature
  fusion and keyword-to-caption augmentation. It is the standard audio-text retriever
  [39].

### Instruction-following universal retrievers

- **UniIR / M-BEIR** (TIGER-Lab, ECCV 2024) trains one retriever with instructions to
  handle heterogeneous queries (text, image, or both) against a pool of millions of
  mixed-modality candidates. It tests CLIP and BLIP backbones with score fusion and
  feature fusion [40].
- **VISTA / Visualized-BGE** (BAAI, 2024) injects image-token embeddings into the BGE text
  embedder for hybrid queries such as composed image retrieval and image-plus-text
  knowledge retrieval [41]. **BGE-VL** (BAAI) is trained on **MegaPairs**, more than 26M
  synthetic heterogeneous-KNN triplets, and reports an 8.1% mAP@5 gain on CIRCO [42].
- **VLM2Vec** (2024), **VLM2Vec-V2** (2025) and **MMEB** turn VLMs into embedders for
  images, video and visual documents. **MMEB-V3** (2026) adds 111 tasks, for 190 in total.
  The new tasks cover audio, text retrieval and *agent* scenarios (tool, GUI and memory
  retrieval). It also adds OmniSET, a set of semantically equivalent tuples across text,
  image, video and audio, to diagnose whether embedders follow modality-specific
  instructions [43][44][45].
- **Qwen3-VL-Embedding and Qwen3-VL-Reranker.** Sentence Transformers v5.4 (April 2026)
  supports these directly. The 2B models need about 8 GB of VRAM and the 8B about 20 GB.
  Multimodal cross-encoders score mixed text+image pairs [28].

### The modality gap

Cross-modal similarities are systematically lower than within-modality ones. In the
Sentence Transformers examples, the best text-image matches score only 0.51-0.67. This is
the "modality gap" [28][52]. The relative ordering within one modality pair is preserved,
so retrieval works. But **a single absolute threshold or a naive score merge across
modalities is wrong**. Text-to-text hits will crowd out text-to-image hits. Fuse by rank
(RRF), normalise per modality, or rerank with a multimodal cross-encoder (Chapter 5).

## 10.9 Video RAG

Video multiplies every visual cost by the frame count and adds an audio track. Published
designs differ mainly in *what* they index per time segment.

### VideoRAG (HKUDS, 2025): graph plus segment embeddings

VideoRAG targets "extremely long-context" video, hundreds of hours, on a **single RTX 3090
(24 GB)** [47][48]. Its default configuration, from the code:

- **Segmentation.** 30-second segments [48].
- **Dual-channel indexing.** (a) A *textual knowledge channel*: faster-distil-whisper-
  large-v3 ASR plus MiniCPM-V-2.6 (int4) captions from 5 sampled frames per segment feed a
  LightRAG-style entity/relation graph and text-chunk index, embedded with
  nomic-embed-text (768-d). (b) A *visual channel*: ImageBind-huge segment embeddings
  (1,024-d) for direct query-to-visual retrieval [48].
- **Retrieval and generation.** Retrieval combines chunk search, entity-graph search and
  visual segment search. For retrieved segments the system re-samples 15 frames for
  fine-grained captioning, keyed to timestamps, before a VLM or LLM answers [48].
- **Results.** The authors built the **LongerVideos** benchmark (164 videos, about
  134.6 h, 602 queries across lectures, documentaries and entertainment) and report that
  VideoRAG outperforms existing long-video methods on it. On Video-MME's long-video track,
  VideoRAG reaches 60.2% accuracy, against 56.3% for its MiniCPM-o and MiniCPM-V
  backbones with subtitles [48].

```
 video --split 30 s--> segment_i
    |-- audio  -> Whisper ASR ------------------+
    |-- 5 frames -> VLM caption ----------------+--> text chunks -> entity/relation KG
    |-- frames  -> ImageBind 1,024-d -> visual vector index           + text vectors
 query --> {KG/entity search, chunk search, visual search} -> top segments
       --> re-sample 15 frames -> fine captions w/ timestamps -> LLM/VLM answer
```

### Video-RAG (2024): training-free auxiliary text

Video-RAG takes the opposite stance. It does not build a new index structure. Instead it
**augments any open large video-language model** with three kinds of "visually aligned
auxiliary text", extracted by open tools: OCR (EasyOCR), ASR (Whisper) and open-vocabulary
object detection (APE). The query retrieves the relevant auxiliary text, which is then
added to the frames in the prompt. It is plug-and-play and needs no training. Applied to a
72B open model, the authors report results on Video-MME above Gemini-1.5-Pro [49].

### Embedding video directly

- **ColQwen-Omni** (ILLUIN) encodes short clips with late interaction. Frame sampling is
  the key control. At 1 fps and full resolution, two cooking videos produced 8,426 and
  5,137 vectors with a 20.8 GB VRAM peak. At 0.5 fps and low resolution they produced
  4,240 and 2,446 vectors at 12.5 GB, with an identical ranking [27].
- **tomoro-colqwen3-embed** extends image-trained late interaction to short video. It
  encodes sampled frames and pools the frame embeddings with a per-dimension max before
  MaxSim, without large-scale video training [6].
- **Gemini Embedding 2** embeds MP4/MOV clips of up to 128 s into the same space as text
  and PDFs. Longer videos must be segmented by the application [46].

**Lessons.** For long video, segment at 10-60 s granularity, index *both* a text view
(ASR plus captions) and a visual view, and return timestamps so the generator can re-look
at a fine frame rate. Sample frames aggressively at index time, because the published
VRAM-per-vector numbers make full-rate encoding unaffordable [27][48].

## 10.10 Audio RAG

Three designs are documented:

1. **ASR-first.** Transcribe with Whisper-class models and index the text. This is the
   default inside VideoRAG and Video-RAG [48][49]. It is cheap and keeps everything in
   text, but it loses tone, speaker identity, music and non-speech sounds, and
   transcription errors on names and jargon carry into retrieval.
2. **Audio-text contrastive embeddings.** CLAP and ImageBind embed sound directly, which
   suits sound effects, music and environmental audio [38][39].
3. **Omni-modal late interaction.** ColQwen-Omni (Qwen2.5-Omni backbone) supports text,
   image, audio and video. Although it was trained *only on image-text pairs*, it
   retrieves audio zero-shot. Over 20 recorded conversations (about 28 s each), the query
   "medicine for car nausea" ranked the "carsickness" pharmacy dialogue first by a clear
   margin, with no transcription step. The release guidance is to chunk long audio into
   30-second pieces of about 800 tokens [27]. Gemini Embedding 2 embeds MP3/WAV clips of
   up to 80 s [46].

A practical pattern combines (1) and (3). Keep the transcript for BM25, dense retrieval and
citation, keep a native audio embedding for paralinguistic or non-speech queries, and fuse
the two by rank.

## 10.11 Comparison of Multimodal RAG Designs

| Design | Index unit | Vectors/unit | Strengths | Weaknesses | Representative |
|---|---|---|---|---|---|
| Parse + text embed | chunk | 1 | cheap ANN, keyword/hybrid, fine citations | loses charts/layout; brittle parsers; slow ingest (~7 s/page, approx.) | Ch. 2 pipelines [1] |
| Parse + caption + KG | element/entity | 1 + graph | element-level citation; cross-modal relations | LLM-heavy indexing; caption hallucination | RAG-Anything [33] |
| Single-vector screenshot | page | 1 (MRL-truncatable) | any ANN index; simple ops; ~3 ms search | one vector per page dilutes detail | DSE, vdr-2b, nemotron-embed-vl [4][8][10] |
| Multi-vector late interaction | page | ~700-1,030 x 128 | best visual retrieval; heatmap explainability | ~170x storage; costly HNSW builds | ColPali/ColQwen/ColEmbed [2][20] |
| Late interaction + pooling/MUVERA | page | 32-64 pooled or 1 FDE + rerank | 13x faster; ~70% less memory | small recall loss; two stores | Qdrant, Weaviate, Vespa [18][23][24] |
| Visual + VL reranker | page | 1 + cross-encoder | +5-6 nDCG points | ~5-17 s/query on V3 HR | NVIDIA, Jina [4] |
| Agentic visual retrieval | page | multi-vector + LLM | best V3 score (0.692) | ~100 s/query | NeMo Retriever Agentic [4] |
| Omni-modal embedding | clip/page/image/audio | 1 or multi | any-to-any retrieval | modality gap; input-length caps | ImageBind, Gemini Emb. 2, ColQwen-Omni [27][38][46] |
| Video graph + segments | 30 s segment | text KG + 1,024-d visual | hundreds of hours on 24 GB | heavy captioning at ingest | VideoRAG [48] |
| Auxiliary-text video RAG | frame/segment text | text | training-free, any LVLM | depends on OCR/ASR/detector quality | Video-RAG [49] |

## 10.12 Cross-Cutting Failure Modes

- **Benchmark saturation and leakage.** ViDoRe V1 subsets are near ceiling (95-99
  nDCG@5), and V1 training splits feed many models [10][35]. Prefer V3, private splits, or
  your own documents [4][6].
- **Resolution mismatch.** A fixed 448 px input loses small print. Dynamic-resolution
  models cap tokens (768 patches for ColQwen2), so very dense pages are still downsampled.
  Render at about 1,500-2,500 px and let the processor resize [2][29].
- **Multilingual drift.** Early Col* models were English/French. vdr-2b shows that
  targeted multilingual synthetic data adds up to 6 points on non-English visual pages
  [10].
- **Licensing.** Gemma-licensed ColPali, NVIDIA non-commercial ColEmbed checkpoints, and
  Apache-2.0 ColQwen/ColSmol checkpoints impose different deployment constraints [2][6].
- **Cross-modal score fusion.** The modality gap breaks absolute thresholds [28][52].
- **Truncation.** `document_length` caps silently drop content. A 662-token passage passed
  through LateOn's 300-token cap keeps only 273 vectors [27]. Check the equivalent
  image-token caps for VLM retrievers.

## Key takeaways for system designers

1. **Decide per corpus, not per company.** For chart-, slide- and figure-heavy documents,
   page-image late interaction (ColQwen/ColEmbed class) is the strongest retriever. For
   text-dominated documents, an OCR-text pipeline with a strong reranker can match or beat
   it at lower cost, as the ViDoRe V3 pipeline results show [4][29].
2. **Index both views when you can.** Image plus OCR-text representations beat image-only
   both with and without a reranker on ViDoRe V3 [4]. Route chart pages to multi-vector
   and plain pages to text, then fuse by rank [29].
3. **Budget for multi-vector cost from day one.** Expect ~700-1,030 vectors x 128 dims per
   page (~512 KB float32). Plan pooled or MUVERA first-stage retrieval, binary or int8
   storage, and an exact MaxSim rerank over 100-1,000 candidates. Published results show
   13x speedups and ~70% memory savings for small recall losses [18][20][24].
4. **Use mean pooling, not max pooling.** Hierarchical pooling at factor 2-3 or grid
   mean pooling keeps 95-98% of quality. Max pooling did not [2][18][26].
5. **Keep the page image for generation.** Send the top 5-10 page images to a VLM. Return
   page-image citations and similarity heatmaps so that users can verify numbers [23][29].
6. **Spend reranking or agent compute deliberately.** A VL reranker buys about 5-6 nDCG
   points for seconds per query. An LLM agent buys about 5 more for about 100 s per query.
   Reserve both for high-value queries [4].
7. **Smaller can be enough.** Bidirectional ~250M-parameter retrievers (ColModernVBERT)
   and ColSmol/ColFlor run where 3B VLMs cannot, at a modest quality cost [6][7].
8. **For video, segment and dual-index.** Use 30-second segments, ASR plus captions for a
   text/graph view, a segment embedding for a visual view, and timestamped fine-grained
   re-inspection at answer time. Sample frames sparsely at index time [27][48].
9. **For audio, keep the transcript and add native embeddings.** ASR gives cheap, citable
   text. CLAP or omni-modal embedders cover tone, music and sound [27][39].
10. **Never merge cross-modal scores raw.** The modality gap makes absolute similarities
    incomparable across modality pairs. Use RRF, per-modality normalisation, or a
    multimodal cross-encoder [28].
11. **Evaluate pipelines, not models.** Use pipeline-level benchmarks (ViDoRe V3 pipeline
    track, MMEB-V3) that record indexing and query latency next to nDCG, and validate on
    your own document templates before rollout [4][29][45].

## References

1. M. Faysse, H. Sibille, T. Wu, B. Omrani, G. Viaud, C. Hudelot, P. Colombo. "ColPali:
   Efficient Document Retrieval with Vision Language Models." arXiv, 2024.
   https://arxiv.org/abs/2407.01449
2. ILLUIN Technology. "ColPali / colpali-engine repository (ColVision models, token
   pooling, Sentence Transformers migration)." GitHub, 2024-2026.
   https://github.com/illuin-tech/colpali
3. Q. Macé, A. Loison, M. Faysse. "ViDoRe Benchmark V2: Raising the Bar for Visual
   Retrieval." arXiv, 2025. https://arxiv.org/abs/2505.17166
4. ILLUIN Technology and contributors. "ViDoRe Pipeline Evaluation Framework
   (vidore-benchmark), ViDoRe V3 pipeline results and descriptions." GitHub, 2025-2026.
   https://github.com/illuin-tech/vidore-benchmark
5. Qdrant. "How ColPali Works (Multi-Vector Search course, Module 2)." 2025-2026.
   https://qdrant.tech/course/multi-vector-search/module-2/how-colpali-works/
6. Qdrant. "ColPali Family Overview (Multi-Vector Search course, Module 2)." 2026.
   https://qdrant.tech/course/multi-vector-search/module-2/colpali-family/
7. P. Teiletche, Q. Macé, M. Conti, A. Loison, G. Viaud, P. Colombo, M. Faysse.
   "ModernVBERT: Towards Smaller Visual Document Retrievers." arXiv, 2025.
   https://arxiv.org/abs/2510.01149
8. X. Ma, S.-C. Lin, M. Li, W. Chen, J. Lin. "Unifying Multimodal Retrieval via Document
   Screenshot Embedding." arXiv, 2024. https://arxiv.org/abs/2406.11251
9. Tevatron. "Document Screenshot Embedding example (training/encoding recipes)." GitHub.
   https://github.com/texttron/tevatron/tree/main/examples/dse
10. marco and cheesyFishes (LlamaIndex, guest authors). "Visual Document Retrieval Goes Multilingual
    (vdr-2b-multi-v1)." Hugging Face Blog, January 2025.
    https://huggingface.co/blog/vdr-2b-multilingual
11. OpenBMB. "VisRAG: Vision-based Retrieval-augmented Generation on Multi-modality
    Documents." arXiv, 2024. https://arxiv.org/abs/2410.10594
12. OpenBMB. "VisRAG / EVisRAG repository." GitHub, 2024-2026.
    https://github.com/OpenBMB/VisRAG
13. OpenBMB. "EVisRAG (VisRAG 2.0): Evidence-Guided Multi-Image Reasoning in Visual
    Retrieval-Augmented Generation." arXiv, 2025. https://arxiv.org/abs/2510.09733
14. R. Tanaka, T. Iki, T. Hasegawa, K. Nishida, K. Saito, J. Suzuki. "VDocRAG:
    Retrieval-Augmented Generation over Visually-Rich Documents." CVPR, 2025.
    http://arxiv.org/abs/2504.09795
15. J. Cho, D. Mahata, O. Irsoy, Y. He, M. Bansal. "M3DocRAG: Multi-modal Retrieval is
    What You Need for Multi-page Multi-document Understanding." arXiv, 2024.
    https://arxiv.org/abs/2411.04952
16. Bloomberg / UNC. "M3DocRAG code release." GitHub, 2024.
    https://github.com/bloomberg/m3docrag
17. BAAI. "Any Information Is Just Worth One Single Screenshot: Unifying Search With
    Visualized Information Retrieval (UniSE, VIRA, MVRB)." arXiv, 2025.
    https://arxiv.org/abs/2502.11431
18. E. Sukhodolskaya, S. Aquino (Qdrant). "Optimizing ColPali for Retrieval at Scale, 13x
    Faster Results." Qdrant Blog, November 2024.
    https://qdrant.tech/blog/colpali-qdrant-optimization/
19. Qdrant. "Multivector Document Retrieval (PDF retrieval at scale with ColPali and
    ColQwen2)." Qdrant Documentation, 2024-2026.
    https://qdrant.tech/documentation/tutorials-search-engineering/pdf-retrieval-at-scale/
20. Qdrant. "Quantization Techniques (Multi-Vector Search course, Module 3)." 2026.
    https://qdrant.tech/course/multi-vector-search/module-3/quantization-techniques/
21. Qdrant. "Problems with Multi-Vector Search (Multi-Vector Search course, Module 1)."
    2026. https://qdrant.tech/course/multi-vector-search/module-1/problems-multi-vector/
22. Qdrant. "Multi-Stage Retrieval and MUVERA (Multi-Vector Search course, Module 3)."
    2026. https://qdrant.tech/course/multi-vector-search/module-3/multi-stage-retrieval/
    and https://qdrant.tech/course/multi-vector-search/module-3/muvera/
23. Vespa.ai. "Visual Retrieval ColPali sample application (schema with binary Hamming
    MaxSim and float rerank)." GitHub, 2024-2026.
    https://github.com/vespa-engine/sample-apps/tree/master/visual-retrieval-colpali
24. Weaviate (Roberto, JP). "More efficient multi-vector embeddings with
    MUVERA." Weaviate Blog, June 2025. https://weaviate.io/blog/muvera
25. L. Dhulipala, M. Hadian, R. Jayaram, J. Lee, V. Mirrokni (Google Research). "MUVERA:
    Multi-Vector Retrieval via Fixed Dimensional Encodings." arXiv, 2024.
    https://arxiv.org/abs/2405.19504
26. B. Clavié, A. Chaffin, G. Adams. "Reducing the Footprint of Multi-Vector Retrieval with
    Minimal Performance Impact via Token Pooling." arXiv, 2024.
    https://arxiv.org/abs/2409.14683
27. T. Aarsen, A. Chaffin (NohTow), R. Sourty (raphaelsty). "Multi-Vector (Late
    Interaction) Embedding Models with Sentence Transformers." Hugging Face Blog, August
    2026. https://huggingface.co/blog/multi-vector-encoder
28. T. Aarsen. "Multimodal Embedding & Reranker Models with Sentence Transformers."
    Hugging Face Blog, April 2026. https://huggingface.co/blog/multimodal-sentence-transformers
29. Weaviate (Etienne, Augustas). "How to extract meaning from charts and tables in PDFs."
    Weaviate Blog, September 2026. https://weaviate.io/blog/charts-tables-pdfs
30. Weaviate (Leonie, Danny, Victoria). "An Overview of Late Interaction Retrieval Models:
    ColBERT, ColPali, and ColQwen." Weaviate Blog, April 2025.
    https://weaviate.io/blog/late-interaction-overview
31. O. Khattab, M. Zaharia. "ColBERT: Efficient and Effective Passage Search via
    Contextualized Late Interaction over BERT." SIGIR, 2020.
    https://arxiv.org/abs/2004.12832
32. K. Santhanam, O. Khattab, J. Saad-Falcon, C. Potts, M. Zaharia. "ColBERTv2: Effective
    and Efficient Retrieval via Lightweight Late Interaction." NAACL, 2022.
    https://arxiv.org/abs/2112.01488
33. HKU Data Intelligence Lab (HKUDS). "RAG-Anything: All-in-One RAG Framework"
    (technical report). arXiv, 2025. https://arxiv.org/abs/2510.12323
34. HKUDS. "RAG-Anything repository." GitHub, 2025-2026.
    https://github.com/HKUDS/RAG-Anything
35. Hugging Face. "Vision Language Models (Better, faster, stronger)", section "Multimodal
    RAG: retrievers, rerankers." Hugging Face Blog, May 2025.
    https://huggingface.co/blog/vlms-2025
36. Hugging Face. "Supercharge your OCR Pipelines with Open Models." Hugging Face Blog,
    October 2025. https://huggingface.co/blog/ocr-open-models
37. A. Radford et al. (OpenAI). "Learning Transferable Visual Models From Natural Language
    Supervision (CLIP)." ICML, 2021. https://arxiv.org/abs/2103.00020
38. R. Girdhar, A. El-Nouby, Z. Liu, M. Singh, K. V. Alwala, A. Joulin, I. Misra (Meta
    FAIR). "ImageBind: One Embedding Space To Bind Them All." CVPR, 2023.
    https://arxiv.org/abs/2305.05665
39. Y. Wu, K. Chen et al. (LAION). "Large-Scale Contrastive Language-Audio Pretraining with
    Feature Fusion and Keyword-to-Caption Augmentation (CLAP)." ICASSP, 2023.
    https://arxiv.org/abs/2211.06687
40. C. Wei et al. (TIGER-Lab). "UniIR: Training and Benchmarking Universal Multimodal
    Information Retrievers." ECCV, 2024. https://arxiv.org/abs/2311.17136
41. BAAI. "VISTA: Visualized Text Embedding for Universal Multi-Modal Retrieval
    (Visualized-BGE)." arXiv, 2024. https://arxiv.org/abs/2406.04292
42. BAAI. "MegaPairs: Massive Data Synthesis For Universal Multimodal Retrieval (BGE-VL)."
    arXiv, 2024. https://arxiv.org/abs/2412.14475
43. Z. Jiang, R. Meng, X. Yang, S. Yavuz, Y. Zhou, W. Chen. "VLM2Vec: Training
    Vision-Language Models for Massive Multimodal Embedding Tasks." arXiv, 2024.
    https://arxiv.org/abs/2410.05160
44. R. Meng et al. "VLM2Vec-V2: Advancing Multimodal Embedding for Videos, Images, and
    Visual Documents." arXiv, 2025. https://arxiv.org/abs/2507.04590
45. TIGER-AI-Lab et al. "MMEB-V3: Measuring the Performance Gaps of Omni-Modality
    Embedding Models." arXiv, 2026. https://arxiv.org/abs/2604.23321 (code:
    https://github.com/TIGER-AI-Lab/VLM2Vec)
46. N. Kanungo (Qdrant). "Qdrant Meets Google Gemini Embedding 2." Qdrant Blog, March
    2026. https://qdrant.tech/blog/qdrant-gemini-embedding-2/
47. HKUDS. "VideoRAG: Retrieval-Augmented Generation with Extreme Long-Context Videos."
    arXiv, 2025. https://arxiv.org/abs/2502.01549
48. HKUDS. "VideoRAG / Vimo repository (VideoRAG-algorithm, LongerVideos benchmark)."
    GitHub, 2025-2026. https://github.com/HKUDS/VideoRAG
49. Video-RAG authors. "Video-RAG: Visually-aligned Retrieval-Augmented Long Video
    Comprehension." arXiv, 2024. https://arxiv.org/abs/2411.13093 (code:
    https://github.com/Leon1207/Video-RAG-master)
50. Z. Hasan (Weaviate). "Multimodal Retrieval-Augmented Generation (RAG)." Weaviate Blog,
    December 2023. https://weaviate.io/blog/multimodal-RAG
51. Answer.AI. "Byaldi: RAGatouille's mini sister project for late-interaction multimodal
    models." GitHub, 2024. https://github.com/AnswerDotAI/byaldi
52. W. Liang et al. "Mind the Gap: Understanding the Modality Gap in Multi-modal
    Contrastive Representation Learning." NeurIPS, 2022.
    https://arxiv.org/abs/2203.02053
53. NVIDIA. "NeMo Retriever retrieval-bench agentic pipeline (AgenticRetrievalPipeline)."
    GitHub, 2026.
    https://github.com/NVIDIA/NeMo-Retriever/blob/main/retrieval-bench/src/retrieval_bench/pipelines/agentic.py
