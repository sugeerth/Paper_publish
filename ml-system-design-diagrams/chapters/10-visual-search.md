# 10. Design a Visual Search / Image Similarity System (Pinterest Lens, Google Lens)

> Prompt as usually given: "A user uploads a photo, or crops a region of a Pin. Return the
> most visually similar images, ranked." Variants: "shop the look" (photo to buyable
> products), camera search (Lens), near-duplicate detection, "more like this", and
> text-to-image search (CLIP-style).
>
> Sourcing note: during review, only github.com was reachable from the build sandbox
> (arxiv.org, medium.com, bytebytego.com, igotanoffer.com and research.google failed DNS)
> and the web-search budget was spent. Tags: (fetched) = opened and checked in review;
> (ch-ref) = verified in another chapter of this atlas; (known) = canonical paper cited by
> arXiv ID, not re-fetched; (book) = structure from the published book. Company tags
> separate "built it" (engineering paper or post) from "asked it" (interview report).

## Company tags

| Company | Evidence | Notes | Source |
|---|---|---|---|
| Pinterest | Built it: Visual Search (2015), Visual Discovery / Lens (2017), unified embedding (2019) | The canonical system this prompt mirrors (Lens, Flashlight, Shop the Look) | [16][17][18] |
| Google | Built it: ScaNN ANN library (used for large-scale vector search); Lens product | Google MLE loops include an ML design round; no portal ties this exact prompt to Google | [12][28][34][35] |
| Microsoft (Bing) | Built it: Bing Visual Search (KDD 2018) | Web-scale, responsive latency budget | [19] |
| eBay | Built it: Visual Search at eBay (KDD 2017) | Product image to listing | [20] |
| Alibaba | Built it: Pailitao visual search (KDD 2018) | Detection + category prediction + embedding | [21] |
| Meta | Built it: photo search (listed under Meta in alirezadir's ML-at-companies list) | Meta MLE loops include ML design; image-similarity prompt not confirmed by a portal | [6][13][14] |
| Generic | Chip Huyen exercise 6: "return images associated with a query like in Google Images" (text-to-image variant) | Open question | [3] |

**Portals that cover it**

| Portal | Treatment | Link |
|---|---|---|
| ByteByteGo (Aminian & Xu, MLSDI), ch. 2 "Visual Search System" | The fullest treatment. Representation learning with contrastive loss, embedding + ANN, indexing pipeline, re-ranking | [1][2] |
| Chip Huyen, ML Systems Design (exercises) | Exercise 6: text-query image search (Google Images); open question | [3] |
| alirezadir AIMLInterviews | Template lists "Image/Video search" and "Visual search" under search systems; search write-up lists multimodal (photo) search and a blender that merges visual search results | [4][5][6] |
| Educative "Grokking the ML Interview" / Khang Pham MLSD | Search and recommendation modules; retrieval and ranking cascade reusable here | [7][8] |
| Exponent / Aced, Hello Interview | Generic MLSD framework and delivery timing | [9][10] |
| IGotAnOffer, Patrick Halina | MLSD guides with example prompts and rubric | [11][15] |

## Requirements & scale

| Dimension | Value | Source |
|---|---|---|
| Functional | Input: image (photo or crop/bounding box). Output: top-K similar images, ranked | [1] |
| Functional (opt.) | Detect objects in the photo and let the user pick one; text + image query; shopping filter | [17][21] |
| Out of scope (book version) | Video; personalization (often added as a follow-up) | [1] |
| Non-functional | Low latency, high recall, safe results (no NSFW, no private Pins) | [1] |
| Catalog size | Billions of images (book uses about 100-200 billion for the Pinterest-style prompt) | [1] (typical) |
| Users / QPS | ~100M+ DAU; ~1K-10K visual QPS at peak | typical assumption |
| Latency | p99 under ~200-300 ms end to end; embedding ~20-50 ms on GPU; ANN ~10-30 ms | typical assumption |
| Freshness | New images searchable within minutes to hours (incremental index) | [29] |
| Embedding size | 64-512 dims float, or binarized / PQ-compressed for storage | [18][26] |

## Diagram 1: end-to-end serving architecture

**Diagram 1. Online prediction path (query image -> embedding -> ANN -> filter -> rerank)**

```
 +--------------+  image / crop    +----------------------+
 | Client       |----------------->| API gateway          |
 | (Lens app)   |<-----------------| auth, rate limit     |
 +--------------+   top-K results  +----------+-----------+
                                              | raw bytes
                                              v
                                   +----------------------+
                                   | Preprocess           |
                                   | resize, normalize    |
                                   +----------+-----------+
                                              |
                                              v
 +----------------------+  boxes   +----------------------+  weights  +--------------+
 | User picks a box     |<---------| Object detector (opt)|<----------| Model store  |
 | (or auto: top box)   |          | boxes + categories   |           | (registry,   |
 +----------+-----------+          +----------------------+           |  versioned)  |
            | crop                                                    +------+-------+
            v                                                                | weights
 +----------------------------------------------------------+                |
 | Embedding service (GPU): CNN / ViT -> d-dim vector q     |<---------------+
 +-----------------------------+----------------------------+
                               | q
                               v
 +----------------------------------------------------------+   +----------------+
 | Candidate retrieval: ANN (HNSW / IVF-PQ / ScaNN)         |<--| Index shards   |
 | sharded fan-out + merge, top ~1000 by cosine / dot       |   | (Diagram 1b)   |
 +-----------------------------+----------------------------+   +----------------+
                               | candidate ids + scores
                               v
 +----------------------------------------------------------+   +----------------+
 | Filtering: NSFW, private, deleted, region / policy,      |<--| Feature store  |
 | out-of-stock (shopping)                                  |   | image meta,    |
 +-----------------------------+----------------------------+   | quality, owner |
                               | ~300 candidates                +-------+--------+
                               v                                        |
 +----------------------------------------------------------+           |
 | Ranking (opt): visual sim + quality + engagement         |<----------+
 | + text / category match (+ user features)                |
 +-----------------------------+----------------------------+
                               | scored list
                               v
 +----------------------------------------------------------+
 | Re-rank / policy: near-dup clustering, diversity (MMR),  |
 | business rules (shopping slots), cut to top-K            |
 +-----------------------------+----------------------------+
                               | top-K -> API gateway -> client
                               v
 +----------------------------------------------------------+
 | Logging: impressions, clicks, saves -> event stream      |
 +----------------------------------------------------------+
```

**Diagram 1b. Indexing pipeline (runs beside serving)**

```
 +------------+ new image +-------------------+  vector  +----------------------+
 | Upload /   |---------->| Batch / stream    |--------->| Index builder        |
 | crawl      |  event    | embedding job     |          | (train IVF/PQ, add   |
 +------------+           | (same model ver.) |          |  to HNSW, shard)     |
                          +-------------------+          +----------+-----------+
                                                                    | swap / incr.
                                                                    v
                                                          +----------------------+
                                                          | Serving index shards |
                                                          +----------------------+
```

## Diagram 2: offline training & data pipeline

**Diagram 2. Data, labels, training, evaluation, deployment and feedback**

```
 +------------------+    +-------------------+    +---------------------------+
 | Interaction logs |    | Image store       |    | Human labels (optional)   |
 | query img, shown,|    | (billions of Pins)|    | same-product / same-style |
 | clicked, saved   |    +---------+---------+    +-------------+-------------+
 +--------+---------+              |                            |
          | join on query/result   | self-supervised            |
          v                        v                            v
 +--------------------------------------------------------------------------+
 | Label builder                                                            |
 |  pos: (query, clicked/saved result) | (img, augmented img) | (img, text) |
 |  neg: in-batch randoms + hard negatives (shown-but-not-clicked, ANN)     |
 +----------+---------------------------------------------------------------+
            | (q, p, n1..nk) tuples
            v
 +---------------------+   +----------------------------+   +-----------------+
 | Augment + preproc   |-->| Train encoder (contrastive |-->| Offline eval    |
 | crop, color, flip   |   | / proxy loss), distributed |   | recall@K, mAP,  |
 +---------------------+   +----------------------------+   | MRR, nDCG       |
                                                            +--------+--------+
                                                                     | pass gate
                                                                     v
 +----------------------+  register new version     +-----------------------------+
 | Model registry       |<--------------------------| Eval-gated encoder          |
 | (versioned encoder)  |-------------------------->| Re-index (new vector space) |
 +----------+-----------+  backfill all embeddings  +--------------+--------------+
            | canary / A-B                                         | index swap
            v                                                      v
 +----------------------+   clicks, saves, dwell    +-----------------------------+
 | Online experiment    |<--------------------------| Online serving (Diagram 1)  |
 +----------+-----------+                           +-----------------------------+
            | new logs (feedback loop, watch position bias)
            +-------------------------------> back to Interaction logs
```

## Diagram 3: model architecture

**Diagram 3. Shared-weight (Siamese) image encoder trained with contrastive loss**

```
  query image q          positive p            negatives n1 .. nk
       |                     |                        |
       v                     v                        v
 +-----------+         +-----------+           +-----------+
 | CNN / ViT |=========| CNN / ViT |===========| CNN / ViT |   shared weights
 | backbone  |         | backbone  |           | backbone  |   (ResNet, ViT,
 +-----+-----+         +-----+-----+           +-----+-----+    CLIP, DINOv2)
       | pooled              |                       |
       v                     v                       v
 +-----------+         +-----------+           +-----------+
 | proj head |         | proj head |           | proj head |   MLP -> d dims
 | + L2 norm |         | + L2 norm |           | + L2 norm |
 +-----+-----+         +-----+-----+           +-----+-----+
       | e_q                 | e_p                   | e_n1..e_nk
       +----------+----------+-----------+-----------+
                  v                      v
        +---------------------------------------------+
        | sim_i = cos(e_q, e_i) / temperature         |
        | softmax over {p, n1..nk}                    |
        | loss = cross-entropy(label = p)  (InfoNCE)  |
        +---------------------------------------------+

  Serving: only one tower is used. Index stores e_i for every catalog image;
  query computes e_q once, then ANN finds argmax cos(e_q, e_i).
```

## Diagram 4: the real system (Pinterest unified visual embedding)

**Diagram 4. Pinterest: one multi-task embedding for Lens, Shop the Look and Flashlight [18]**

```
 +-------------------------------+
 | Task data:                    |
 | Lens (camera photo -> Pin)    |--+
 | Shop the Look (scene -> prod.)|--+    +---------------+   +--------------------+
 | Flashlight (crop -> Pin)      |--+--->| shared CNN    |-->| FC -> embedding    |
 +-------------------------------+       | backbone      |   | (binarized for     |
                                         +---------------+   |  storage/serving)  |
                                                             +---------+----------+
                                    +------------------------+         |
                                    | per-task proxy softmax |<--------+
                                    | (training only)        |         |
                                    +------------------------+         | serving
                                                                       v
                                                      +--------------------------------+
 camera photo --> object detector --> crop ---------->| embed -> kNN over billions     |
 (Lens: detection + category + embedding [17])        | -> filter -> blend + rank      |
                                                      +--------------------------------+
```

## ANN index sizing (FAISS guidance)

| Scale | Suggested index | Note | Source |
|---|---|---|---|
| 100M-1B vectors | `IVF1048576_HNSW32` (2^20 lists, HNSW coarse quantizer) | Training slow; plan 30-256 training vectors per centroid | [33] |
| 10M-100M | `IVF262144_HNSW32` | Train on GPU, rest on CPU, or two-level clustering | [33] |
| RAM-bound | OPQ + PQ (4-bit FastScan) or RaBitQ (~1 bit/dim) | All FAISS indexes live in RAM; HNSW costs ~(4d + 8M) bytes/vector | [33] |
| GPU | Plain IVF (IVF+HNSW not on GPU) | Use for batch backfill / re-index | [33] |

## Data & labels

| Source | Positive | Negative | Pros | Cons | Source |
|---|---|---|---|---|---|
| User interactions | (query, clicked/saved result) | Shown-not-clicked, random | Matches product goal | Position bias, noisy, cold start | [1] |
| Self-supervision | (image, augmented view) | Other images in batch | Free, unlimited labels | Learns invariance, not "style" | [1][23] |
| Image-text pairs | (image, caption / Pin text) | Other pairs in batch | Enables text-to-image search | Needs clean text | [22] |
| Human annotation | Same product / same object | Different product | High precision | Costly, slow | [1][21] |
| Product catalog | Scene crop <-> catalog image of same SKU | Other SKUs, same category | Shopping relevance | Needs detection + matching | [18][21] |

## Features

| Group | Examples | Used in |
|---|---|---|
| Raw pixels | Resized 224x224 RGB, crop from detector box | Encoder |
| Visual embedding | d-dim vector (float / binary / PQ code) | Retrieval, ranking |
| Image metadata | Category, detected objects, OCR text, quality, resolution, NSFW score | Filter, rank |
| Engagement | Saves, clicks, CTR of result image | Rank |
| Text | Pin title, board name, alt text (text tower or CLIP) | Rank, hybrid query |
| User (optional) | Recent interests, locale | Personalized re-rank |

## Model choices (baseline -> advanced)

| Stage | Option | Tradeoff | Source |
|---|---|---|---|
| Baseline | Hand-crafted features (SIFT, color histograms) + LSH | Cheap, poor semantic similarity | [16] |
| v1 | ImageNet-pretrained CNN, take penultimate layer | Easy, strong start; not tuned to task | [16][20] |
| v2 | Fine-tuned encoder with contrastive / triplet loss | Task-aligned; needs labels and hard negatives | [1][24] |
| v3 | Multi-task / proxy-based metric learning, one embedding for many products | One model to maintain; task balancing needed | [18] |
| v4 | Foundation encoders (CLIP, DINOv2) + fine-tuning | Zero-shot text<->image; larger serving cost | [22][31] |
| Detector front end | Faster R-CNN / SSD style detector for crops | Better object-level matches; extra latency | [17][21] |
| ANN | Tree / LSH / clustering (IVF) / graph (HNSW) / ScaNN (anisotropic VQ, SOAR) | Recall vs memory vs latency | [1][26][27][28][35] |

## Loss

| Loss | Form | Note | Source |
|---|---|---|---|
| Contrastive (softmax / InfoNCE) | CE over sim(q, p) vs sim(q, n_i), with temperature | Book's main choice; scales with in-batch negatives | [1][25] |
| Triplet | max(0, d(q,p) - d(q,n) + margin) | Sensitive to negative mining | [24] |
| Proxy-based softmax | Classify embedding against learned class proxies | Stable at scale; used by Pinterest | [18] |
| SimCLR NT-Xent | Two augmented views as positives | Self-supervised pretraining | [23] |
| CLIP symmetric CE | Image<->text matching matrix | Multimodal search | [22] |

## Offline metrics

| Metric | What it measures | Note |
|---|---|---|
| Recall@K | Share of relevant items in top K | Main retrieval metric; also measure ANN recall vs exact kNN |
| Precision@K | Share of top K that is relevant | Ignores order |
| MRR | 1 / rank of first relevant result | Good when one right answer (same product) |
| mAP | Mean of average precision over queries | Needs binary relevance; book's choice for binary labels [1] |
| nDCG | Graded relevance, position discounted | Needs graded labels [1] |
| Human side-by-side | Raters compare old vs new results | Used at Pinterest and Bing [16][19] |

## Online metrics / A-B

| Metric | Notes |
|---|---|
| CTR on results | Primary engagement signal [1] |
| Saves / repins per search, add-to-cart (shopping) | Stronger intent than clicks |
| Time spent on suggested images (daily/weekly) | Book's secondary metric [1] |
| Search abandonment, zero-click rate | Guardrail |
| Latency p50/p99, error rate | Guardrail |
| A-B design | User-level split; re-index needed per arm when the embedding space changes (two indexes side by side) |

## Serving & latency budget (typical)

| Step | Budget | Notes |
|---|---|---|
| Upload + preprocess | 20-40 ms | Client-side resize to cut bytes |
| Object detection (opt) | 20-40 ms | GPU; can run on device |
| Embedding | 15-40 ms | GPU batching, FP16/INT8, distilled model |
| ANN retrieval | 10-30 ms | HNSW or IVF-PQ in memory, sharded; fan-out + merge |
| Filter + feature fetch | 10-20 ms | Feature store / KV lookups |
| Rank + re-rank | 10-30 ms | Light GBDT or MLP over ~300 candidates |
| Total | ~100-200 ms server side | Cache embeddings of popular query images |

## Monitoring

| What | Signal |
|---|---|
| Embedding drift | Mean/variance of query embeddings, nearest-neighbor distance distribution |
| Index health | Index size, staleness (time since last build), ANN recall on a probe set |
| Model version mismatch | Query encoder version must equal index encoder version |
| Quality | CTR, zero-result rate, human rating samples per week |
| Safety | NSFW leakage rate in results, takedown latency |
| System | p99 latency per stage, GPU utilization, shard hot spots |

## Failure modes

| Failure | Cause | Mitigation |
|---|---|---|
| Duplicate-heavy results | Many reposts of the same image | Near-dup clustering, dedupe at re-rank |
| Wrong object matched | Cluttered photo, background dominates | Detector + user-selected crop [17] |
| Space mismatch after retrain | Query and index encoders differ | Full backfill, atomic index swap, version pinning |
| Popularity / position bias in labels | Clicks reflect slot, not relevance | Position feature, randomized slots, IPS weighting |
| Unsafe or private results | Missing filters | Pre-filter at index time + post-filter at serve |
| ANN recall loss | Over-compressed PQ, poor graph params | Tune nprobe/efSearch; exact rerank on top 1000 |
| Cold new images | Not yet indexed | Streaming incremental index [29][30] |
| Domain shift (camera vs Pin photos) | Training on clean Pins only | Train on camera-query logs (Lens task) [17][18] |

## Interviewer follow-ups

- How do you build training data without human labels? Clicks vs augmentation vs image-text pairs.
- Why contrastive loss over classification? What is the role of temperature and batch size?
- How do you mine hard negatives, and what goes wrong if negatives are too hard (false negatives)?
- Exact kNN over 10 billion vectors is too slow: compare LSH, IVF-PQ, HNSW and ScaNN on recall, memory and update cost.
- How do you handle retraining when every catalog embedding must be recomputed (backfill cost, dual index)?
- The user's photo has five objects. How do you know which one they mean?
- How do you remove near-duplicates and keep the results diverse?
- How would you add text to the query ("this chair but in blue")?
- How do you personalize visual search without losing visual relevance?
- How do you keep NSFW, private or copyrighted images out of results?
- How do you compress embeddings for billions of images (binarization, PQ, lower dims)?
- How do you evaluate offline when relevance is subjective (style vs exact product)?
- How do you detect that the model degraded after a deployment?
- How do you serve new images within minutes of upload?
- How do you handle positional bias in click labels?
- Your index is 1B x 256-d floats (~1 TB). How do you fit it in RAM per shard (PQ, RaBitQ, binarization) and what recall do you lose? [33]
- How do you blend visual results with text search results for the same query (blender stage)? [5][17]

## Portal comparison

| Portal | Framing | Model | Training data | Retrieval | Metrics |
|---|---|---|---|---|---|
| ByteByteGo [1][2] | Pinterest-like; image or crop in, ranked similar images out; no personalization | CNN/ViT encoder, contrastive (softmax over 1 pos + n neg) | Clicks as positives or self-supervised augmentations | Embedding service + ANN (tree, LSH, clustering); indexing pipeline; re-rank filters | MRR, Recall@K, Precision@K, mAP, nDCG; CTR, time spent |
| Pinterest papers [16][17][18] | Real product: Lens, Flashlight, Shop the Look | Detector + CNN embedding; later one multi-task proxy-loss embedding | Engagement logs + curated product matches per task | Distributed kNN, binarized embeddings | Human relevance judgments + engagement A-B |
| Bing / eBay / Alibaba [19][20][21] | Web / commerce visual search | Category prediction + detection + embedding; multi-stage rerank | Click logs, catalog, human labels | Inverted / ANN index over billions, multi-stage | Precision, relevance ratings, conversion |
| alirezadir / Educative [4][5][7][8] | Generic retrieval + ranking template | Two-stage: retrieve then rank | Logs with graded labels | ANN or inverted index | nDCG, CTR, conversion |
| Chip Huyen [3] | Open exercise (text-to-image variant) | Left to candidate | Left to candidate | Left to candidate | Emphasis on data and serving trade-offs |

## References

1. A. Aminian & A. Xu, "Machine Learning System Design Interview" (ByteByteGo; ch. 2 Visual Search System), book overview, 2023. https://opencourser.com/book/xop33f/machine-learning-system-design-interview (ch-ref, book)
2. ByteByteGo, "Visual Search System" (MLSDI course chapter), bytebytego.com, 2023. https://bytebytego.com/courses/machine-learning-system-design-interview/visual-search-system (book; URL follows the pattern of chapter URLs verified in this atlas, not fetched)
3. C. Huyen, "Machine Learning Systems Design: Exercises", GitHub, 2019. https://github.com/chiphuyen/machine-learning-systems-design/blob/master/content/exercises.md (fetched)
4. alirezadir, "ML System Design" template, GitHub, 2025. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/ml-system-design.md (fetched)
5. alirezadir, "Search system design" (mlsd-search.md), GitHub, 2025. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/mlsd-search.md (fetched)
6. alirezadir, "ML at Companies" (ml-companies.md), GitHub, 2025. https://github.com/alirezadir/AIMLInterviews/blob/main/src/MLSD/ml-companies.md (fetched)
7. Educative, "Grokking the Machine Learning Interview", educative.io, 2024. https://www.educative.io/courses/grokking-the-machine-learning-interview (ch-ref)
8. Educative / Khang Pham, "Machine Learning System Design", educative.io, 2021. https://educative.io/courses/machine-learning-system-design/introduction (ch-ref)
9. Exponent, "Introduction to the ML System Design Interview", tryexponent.com, 2025. https://www.tryexponent.com/courses/ml-system-design/mlsd-intro (ch-ref)
10. Hello Interview, "ML System Design in a Hurry: Delivery Framework", hellointerview.com, 2025. https://www.hellointerview.com/learn/ml-system-design/in-a-hurry/delivery (ch-ref)
11. IGotAnOffer, "Machine Learning System Design Interview", igotanoffer.com, 2025. https://igotanoffer.com/en/advice/machine-learning-system-design-interview (ch-ref)
12. IGotAnOffer, "Google Machine Learning Engineer Interview", igotanoffer.com, 2025. https://igotanoffer.com/blogs/tech/google-machine-learning-engineer-interview (ch-ref)
13. IGotAnOffer, "Meta Machine Learning Engineer Interview", igotanoffer.com, 2025. https://igotanoffer.com/blogs/tech/facebook-machine-learning-engineer-interview (ch-ref)
14. Meta Engineering, "Under the hood: Photo Search", engineering.fb.com. https://engineering.fb.com/ml-applications/under-the-hood-photo-search/ (link listed in [6]; not fetched, host unreachable in review)
15. P. Halina, "ML Systems Design Interview Guide", patrickhalina.com, 2021. http://patrickhalina.com/posts/ml-systems-design-interview-guide/ (ch-ref)
16. Y. Jing et al., "Visual Search at Pinterest", KDD 2015. https://arxiv.org/abs/1505.07647 (known)
17. A. Zhai et al., "Visual Discovery at Pinterest", WWW 2017. https://arxiv.org/abs/1702.04680 (known)
18. A. Zhai, H.-Y. Wu, E. Tzeng, D. H. Park, C. Rosenberg, "Learning a Unified Embedding for Visual Search at Pinterest", KDD 2019. https://arxiv.org/abs/1908.01707 (ch-ref)
19. H. Hu et al., "Web-Scale Responsive Visual Search at Bing", KDD 2018. https://arxiv.org/abs/1802.04914 (known)
20. F. Yang et al., "Visual Search at eBay", KDD 2017. https://arxiv.org/abs/1706.03154 (known)
21. Y. Zhang et al., "Visual Search at Alibaba", KDD 2018. https://arxiv.org/abs/2102.04674 (known)
22. A. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (CLIP), 2021. https://arxiv.org/abs/2103.00020 (ch-ref)
23. T. Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations" (SimCLR), 2020. https://arxiv.org/abs/2002.05709 (known)
24. F. Schroff, D. Kalenichenko, J. Philbin, "FaceNet: A Unified Embedding for Face Recognition and Clustering" (triplet loss), 2015. https://arxiv.org/abs/1503.03832 (known)
25. A. van den Oord, Y. Li, O. Vinyals, "Representation Learning with Contrastive Predictive Coding" (InfoNCE), 2018. https://arxiv.org/abs/1807.03748 (known)
26. J. Johnson, M. Douze, H. Jegou, "Billion-scale similarity search with GPUs" (FAISS), 2017. https://arxiv.org/abs/1702.08734 (ch-ref)
27. Y. Malkov, D. Yashunin, "Efficient and robust approximate nearest neighbor search using HNSW graphs", 2016. https://arxiv.org/abs/1603.09320 (ch-ref)
28. R. Guo et al., "Accelerating Large-Scale Inference with Anisotropic Vector Quantization" (ScaNN), ICML 2020 (arXiv 2019). https://arxiv.org/abs/1908.10396 (ch-ref; venue confirmed via [35])
29. Pinterest Engineering, "Manas HNSW Streaming Filters", Medium, 2022. https://medium.com/pinterest-engineering/manas-hnsw-streaming-filters-351adf9ac1c4 (ch-ref)
30. Pinterest Engineering, "Evolving Pinterest's Embedding Retrieval Platform", Medium, 2023. https://medium.com/pinterest-engineering/evolving-pinterests-embedding-retrieval-platform-aede4e831e01 (ch-ref)
31. M. Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", 2023. https://arxiv.org/abs/2304.07193 (known)
32. Meta Engineering, "Faiss: A library for efficient similarity search", engineering.fb.com, 2017. https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/ (ch-ref)
33. FAISS wiki, "Guidelines to choose an index", GitHub. https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index (fetched)
34. Google Research, "Announcing ScaNN: Efficient Vector Similarity Search", research.google, 2020. https://research.google/blog/announcing-scann-efficient-vector-similarity-search/ (ch-ref)
35. Google Research, "ScaNN: Scalable Nearest Neighbors" README (cites anisotropic VQ, ICML 2020, and SOAR, NeurIPS 2023), GitHub. https://github.com/google-research/google-research/tree/master/scann (fetched)
