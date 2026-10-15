# 11. Design Multimodal Video Search (YouTube Search over Text + Visual + Audio)

> Prompt as usually given: "A user types a text query. Return the most relevant videos
> from a catalog of about 1B, ranked." The multimodal version asks you to search what is
> *inside* the video (frames, speech, on-screen text), not only the title and tags.
> Variants: in-video moment search ("find the scene where..."), podcast/audio search,
> query-by-image or query-by-clip, and personalised search.
>
> Sourcing note: during compilation and review the shared web-search budget was used up
> and most portal hosts, company blogs and arxiv.org could not be reached from the build
> sandbox. GitHub pages were fetched and read (re-checked in review: [3][4][5][6][7][8][11]). Each reference carries a tag. (fetched) means the page was
> read for this chapter. (ch-ref) means the URL was verified in another chapter of this
> atlas. (known) means a canonical paper cited by its arXiv ID. (book) means the structure
> is taken from the published book. Company tags separate "built it" (a paper or
> engineering blog exists) from "asked it" (an interview report). The "asked it" evidence
> is thin, so treat it as indicative only.

## Company tags

| Company | Evidence | Notes | Source |
|---|---|---|---|
| Google / YouTube | Built it: no public paper on YouTube *search* itself; adjacent public work is YouTube recs [18][37], YouTube-8M video understanding [38], VideoPrism video encoder [39], Google Search [23]. The book chapter is "YouTube Video Search". Asked it (indicative): Google MLE loops include ML design | Canonical framing of the problem | [1][2][14][18][23][37][38][39] |
| Meta (FB / IG video, Reels) | Built it: photo search, Rosetta (text in images and video), embedding-based retrieval in FB search. Asked it (indicative): Meta MLE loops include ML design | Multimodal content understanding at billions scale | [6][15][19][20][21] |
| Netflix | Built it: in-video search (2023), listed in [7] | Mirrors the "moment search" variant | [7][22] |
| Spotify | Built it: natural-language podcast episode search (2022), listed in [7] | Mirrors the audio/ASR branch (dense + lexical hybrid) | [7][24] |

TikTok, Amazon Prime Video and similar tags were removed in review: no interview report or
public system write-up could be confirmed.

**Portals that cover it**

| Portal | Treatment | Link |
|---|---|---|
| ByteByteGo (Aminian & Xu, MLSDI), ch. 4 "YouTube Video Search" | The fullest treatment. Visual search (two-tower, contrastive) plus text search (inverted index), fusion, re-ranking | [1][2] |
| alirezadir AIMLInterviews, "Multimodal video search" case | Book-style summary: 1B videos, 50M DAU, 10M labelled pairs, BERT + ViT/ViViT, MRR, watch time. Its MLSD template lists "Image/Video search" and "Multimodal search" as sample questions | [3][4] |
| alirezadir, "Image search" case | Same contrastive / ANN template for the visual tower | [5] |
| Chip Huyen, ML Systems Design exercises | Ex. 6 "return images associated with a query like in Google Images"; Ex. 14 trigger-word audio detection (audio branch) | [8] |
| Educative "Grokking the ML Interview" / Khang Pham MLSD | Search ranking (staged funnel, LTR) in [9]; Khang Pham's repo lists Airbnb search ranking and YouTube recs cases, not video search | [9][10][11] |
| Exponent, Hello Interview, IGotAnOffer, Patrick Halina | Generic MLSD framework, timing, rubric. Use them for delivery, not for this problem's content | [12][13][14][16][17] |

## Requirements & scale

| Dimension | Value | Source |
|---|---|---|
| Functional | Text query in, ranked list of videos out. Match title/desc/tags *and* visual content | [1][3] |
| Extended functional | Speech (ASR) and on-screen text (OCR) searchable; optional moment timestamps | assumption, [19][22] |
| Personalisation | Not required in the book version; add as a follow-up | [3] |
| Users | 50M DAU | [3] |
| Catalog | 1B videos with title, description, tags | [3] |
| Training data | 10M annotated (video, text query) pairs | [3] |
| Uploads | ~500 hours of video per minute (YouTube-scale), so indexing must be streaming | assumption |
| QPS | 50M DAU x ~5 searches/day = ~3k QPS avg, ~10k peak | assumption |
| Latency | p99 under 200-300 ms end to end; ANN under 20 ms per shard | assumption |
| Freshness | New upload searchable by text in minutes, by visual/ASR embedding within ~1 h | assumption |
| Languages | Open question in the source. Assume multilingual text encoder | [3] |
| Business goal | Higher CTR and total watch time from search | [3] |

## DIAGRAM 1: Online serving path

**Figure 11.1 - Query-time path: hybrid retrieval (lexical + dense multimodal), fusion,
ranking, policy re-rank.**

```
 +--------+  text query   +-------------------------------+
 |  User  |-------------->| Query understanding           |
 +--------+               | spell-fix, lang-ID, intent,   |
     ^                    | safe-search, query rewrite    |
     |                    +---------------+---------------+
     |                                    | q (normalised)
     |                  +-----------------+------------------+
     |                  |                                    |
     |                  v                                    v
     |       +---------------------+             +-----------------------+
     |       | Query text encoder  |             | Lexical retrieval     |
     |       | (BERT / CLIP text,  |             | inverted index (ES):  |
     |       |  distilled, cached) |             | title, desc, tags,    |
     |       +----------+----------+             | ASR + OCR text, BM25  |
     |                  | q_emb (d=256-768)      +-----------+-----------+
     |      +-----------+-----------+                        |
     |      |           |           |                        |
     |      v           v           v                        |
     |  +-------+   +-------+   +-------+                    |
     |  | ANN   |   | ANN   |   | ANN   |                    |
     |  | visual|   | audio/|   | text  |                    |
     |  | frames|   | ASR   |   | meta  |                    |
     |  +---+---+   +---+---+   +---+---+                    |
     |      | ~1k       | ~500      | ~500                   | ~1k
     |      +-----------+-----+-----+------------------------+
     |                        | ~2-3k candidates, deduped
     |                        v
     |              +-------------------+
     |              | Fusion: RRF or    |
     |              | weighted sum      |      +----------------------------+
     |              +---------+---------+      | Feature store (online):    |
     |                        | ~500           | video stats, (q,v) CTR,    |
     |                        v                | quality, freshness, user   |
     |              +-------------------+      +-------------+--------------+
     |              | Filter: removed,  |                    |
     |              | private, geo, age,|                    |
     |              | safe-search, dups |                    |
     |              +---------+---------+                    |
     |                        | ~400                         | features
     |                        v                              |
     |              +-------------------+                    |
     |              | Ranker: GBDT LTR  |<-------------------+
     |              | or cross-encoder  |      +----------------+
     |              |                   |<-----| Model registry |
     |              +---------+---------+      +----------------+
     |                        | top ~100
     |                        v
     |              +-------------------+
     |              | Re-rank / policy: |
     |              | diversity, fresh- |
     |              | ness, creator cap,|
     |              | borderline demote |
     |              +---------+---------+
     |   top 20 results       |
     +------------------------+-----> impression / click / watch log --> Kafka
```

Sources: two-path design (visual ANN plus Elasticsearch text search), fusion by weighted
sum or re-rank model, then business-rule re-rank [1][3]. ASR and audio branches added per production systems [19][22][24].

| Stage | Input -> output | Typical budget | Notes |
|---|---|---|---|
| Query understanding | raw text -> normalised q, lang, intent | 10-20 ms | Spell-fix, synonym, safe-search intent |
| Query encode | q -> 256-768-d vector | 5-15 ms (GPU/CPU distilled) | Cache top-1M head queries |
| Lexical retrieval | q -> ~1k docs | 20-40 ms | BM25 over title/desc/tags/ASR/OCR [1][3] |
| Dense ANN (x3 indexes) | q_emb -> ~500-1k each | 10-20 ms each, parallel | HNSW / ScaNN / FAISS-IVFPQ [25][26][27] |
| Fusion + filter | ~3k -> ~400 | 5-10 ms | RRF or learned weights [1][28] |
| Ranker | ~400 -> top 100 | 30-60 ms | GBDT or small cross-encoder; position-debiased labels [36] |
| Re-rank / policy | 100 -> 20 | 5 ms | Rules [1][3] |
| Total | | ~150-250 ms p99 | assumption |

## DIAGRAM 2: Offline indexing, training and data pipeline

**Figure 11.2 - Two loops: (a) per-upload indexing of every modality, (b) periodic
training of encoders and ranker from labels and search logs.**

```
 (a) INDEXING PIPELINE (streaming, per upload)

 +--------------+  video  +---------------------+
 | Upload svc   |-------->| Decode + sample     |
 | title, desc, |         | ffmpeg, shot detect |
 | tags         |         +----+-----------+----+
 +------+-------+        audio |           | frames (1 fps / per shot)
        |                      |           +-------------+
        |                      v           |             |
        |              +---------------+   v             v
        |              | ASR (Whisper- | +-----------+ +---------------+
        |              |  style)       | | OCR       | | Video encoder |
        |              +-------+-------+ | (Rosetta- | | ViT / ViViT + |
        |                      |         |  style)   | | audio encoder |
        |                      |         +-----+-----+ +-------+-------+
        | meta text            | transcript    | screen text   | v_emb, a_emb
        v                      v               v               |
 +-------------------------------------------------------+     |
 | Text docs: meta + ASR + OCR, then text encoder        |     |
 +-------------+----------------------------+------------+     |
               | tokens                     | t_emb            |
               v                            v                  v
 +-----------------------------+  +----------------------------------------+
 | Inverted index (ES, BM25)   |  | ANN indexes: visual, audio, text-meta  |
 | searchable in minutes       |  | IVF-PQ / ScaNN / HNSW, within ~1 h     |
 +-----------------------------+  +----------------------------------------+

 (b) TRAINING LOOP (daily ranker, weekly-to-monthly encoders)

 +--------------------+  impressions, clicks,  +-----------------------------+
 | Search logs        |  watch, skips, reform. | Label builder               |
 | (Kafka -> DW)      |----------------------->| click + watch > X s = pos;  |
 +---------^----------+                        | position-bias correction    |
           |                                   +--------------+--------------+
           |   +--------------------+  graded 0-4             |
           |   | Human rater pool   |------------------------>|  10M pairs [3]
           |   +--------------------+                         v
           |                                   +-----------------------------+
           |                                   | Training datasets, point-in-|
           |                                   | time features from store    |
           |                                   +--------------+--------------+
           |                                                  |
           |                                                  v
           |                                   +-----------------------------+
           |                                   | Train: dual encoders        |
           |                                   | (contrastive), LTR ranker,  |
           |                                   | fusion weights              |
           |                                   +--------------+--------------+
           |                                                  | offline eval: MRR,
           |                                                  v recall@k, nDCG
           |  +---------------------+  pass gates +-----------------------------+
           |  | Deploy: shadow ->   |<------------| Model registry + eval report|
           |  | canary -> A/B       |             +--------------+--------------+
           |  +----------+----------+                            | new encoder
           |             | serves Figure 11.1                    v version
           +-------------+ (new logs)             +-----------------------------+
                                                  | Backfill (batch GPU):       |
                                                  | re-embed 1B videos, build   |
                                                  | new index, dual-write, swap |
                                                  +-----------------------------+
```

Sources: preprocessing steps (decode, sample, resize, normalise; text normalise, tokenise)
[3]; indexing and ANN pipeline [5]; OCR for text in video [20]; ASR [32].

## DIAGRAM 3: Model architecture

**Figure 11.3 - Multimodal dual encoder (retrieval) plus learning-to-rank head.**

```
       QUERY TOWER                               VIDEO TOWER(S)
        QUERY TOWER                           VIDEO TOWERS (offline, batch)
 +----------------------+   +----------------+ +---------------+ +--------------+
 | "how to fix a bike   |   | frames f1..fN  | | audio / ASR   | | metadata:    |
 |  chain"              |   | (8-32 sampled) | | transcript    | | title, tags  |
 +----------+-----------+   +-------+--------+ +-------+-------+ +------+-------+
            | tokens                |                  |                |
            v                       v                  v                v
 +----------------------+   +----------------+ +---------------+ +--------------+
 | Text encoder (BERT / |   | ViT per frame  | | Text enc on   | | Text encoder |
 | CLIP text, multiling)|   | (CLIP image)   | | ASR, or audio | | (shared with |
 +----------+-----------+   +-------+--------+ | encoder       | |  query)      |
            | [CLS]                 | N x d    +-------+-------+ +------+-------+
            v                       v                  |                |
 +----------------------+   +----------------+         |                |
 | Projection + L2 norm |   | Temporal pool /|         |                |
 +----------+-----------+   | attention (mean|         |                |
            |               | or 4L Xfmr)    |         |                |
            |               +-------+--------+         |                |
            |                       | v_vis            | v_aud          | v_txt
            |                       v                  v                v
            |               +-------------------------------------------------+
            |               | Per-modality proj + L2 norm (separate ANN each) |
            |               | or gated fusion -> single v (one ANN index)     |
            |               +------------------------+------------------------+
            | q in R^d                               | v in R^d
            +------------>  sim = cos(q, v) / tau  <-+
                                     |
                                     v
             InfoNCE, symmetric (q->v, v->q), in-batch + mined hard negatives
 ============================== RANKING STAGE ==============================
   features: sim_vis, sim_aud, sim_txt, BM25 per field, (q,v) CTR, watch time,
             freshness, quality, language match, (opt.) user features
                                     |
                                     v
                     +-------------------------------+
                     | LambdaMART / DNN ranker, or   |---> relevance score
                     | cross-encoder on top-50 (q,v) |     (+ P(click), E[watch])
                     +-------------------------------+
```

Sources: two-tower with text encoder (BERT) and video encoder (frame-level ViT
aggregated vs video-level ViViT) [1][3]; contrastive training [3][5]; CLIP image-text
pretraining [29]; CLIP4Clip temporal pooling variants [30]; ViViT [31].

## DIAGRAM 4: Variant - in-video moment search (Netflix / long-form)

**Figure 11.4 - Moment search: index shots, not whole videos; return (video, t_start,
t_end).**

```
 +----------+  shot boundary  +-------------+  shot clips  +------------------------+
 | Long     |---------------->| Shot / scene|------------->| Clip encoder (CLIP-    |
 | video    |  detection      | segmenter   |              | style) + ASR window    |
 +----------+                 +-------------+              +-----------+------------+
                                                                       | (vid, t0,t1,
                                                                       |  emb, text)
                                                                       v
 +--------+  "car chase at  +-------------+  q_emb   +------------------------------+
 | Editor |---------------->| Text encoder|--------->| ANN over shot embeddings     |
 | / user |   night"        +-------------+          | (100x more rows than videos) |
 +---^----+                                          +--------------+---------------+
     |                                                              | top-K shots
     |        +-----------------------+    group by video, merge    |
     +--------| Results: video +      |<----------------------------+
              | timestamps + thumbs   |    adjacent shots, re-rank
              +-----------------------+
```

Sources: Netflix in-video search [7][22]; segment-level text-video retrieval [30][33].

## Data & labels

| Source | Label | Pros | Cons | Source |
|---|---|---|---|---|
| Human-annotated (query, video) pairs | Relevant / graded 0-4 | Clean, measures relevance | Expensive; 10M pairs in book setup | [1][3] |
| Search logs | Click + watch > X s = positive; skip / quick-back = negative | Free, huge, fresh | Position bias, clickbait, popularity bias | [3] |
| Title/desc as weak caption | (title, video) pair | Billions of pairs | Noisy, SEO spam | [29][33] |
| ASR transcript windows | (speech, clip) pair | Self-supervised alignment | Speech often not about visuals | [33][34] |
| Query reformulation | q1 -> q2 within session | Negative for q1 results | Ambiguous | assumption |
| Debiased clicks | Clicks reweighted by inverse examination propensity | Removes position bias from log labels | Needs propensity estimates (randomisation or click model) | [36] |

## Features

| Group | Features | Used at |
|---|---|---|
| Query | Tokens, language, length, intent class, query embedding, query popularity | Retrieval + rank |
| Video content | Frame embedding, audio/ASR embedding, OCR text, title/desc/tags, duration, language | Indexing + rank |
| Query-video | BM25 per field, cosine per modality, historical CTR / watch for (q, v) | Rank |
| Video quality | Views, like ratio, upload age, channel authority, policy flags | Rank + policy |
| User (optional) | Language, region, watch history embedding, subscriptions | Personalised rank |
| Context | Device, time, safe-search setting | Filter + rank |

## Model choices

| Component | Baseline | Better | Advanced | Tradeoff | Source |
|---|---|---|---|---|---|
| Text matching | BoW / TF-IDF / BM25 | word2vec avg | BERT-family encoder | Semantics vs cost | [1][3] |
| Video encoder | Frame-level ViT + mean pool | Frame ViT + temporal transformer (CLIP4Clip) | Video-level ViViT | Temporal info vs train/serve cost | [3][30][31] |
| Audio | ASR text into lexical index | ASR text embedding | Audio encoder in joint space | Coverage vs index size | [32][35] |
| Retrieval | Inverted index only | + dense two-tower ANN | Per-modality ANN + learned fusion | Recall vs memory | [1][25] |
| Fusion | Weighted sum of scores | RRF | Learned LTR on all scores | Simplicity vs quality | [1][28] |
| Ranker | Hand-tuned weights | GBDT LambdaMART | Cross-encoder on top-50 | Latency vs precision | assumption |
| Pretraining | ImageNet ViT | CLIP image-text | Video-text (HowTo100M, MIL-NCE, VideoCLIP) | Data domain match | [29][33][34] |

## Loss functions

| Model | Loss | Notes | Source |
|---|---|---|---|
| Dual encoder | InfoNCE / softmax contrastive, in-batch negatives, temperature tau | Symmetric (q->v and v->q) as in CLIP | [3][29] |
| Dual encoder (noisy ASR pairs) | MIL-NCE | Several nearby clips count as positives | [34] |
| Hard-negative mining | Same loss with mined near-misses | Mine from ANN results of the previous model | assumption |
| Dual encoder, in-batch negatives | Sampled softmax with logQ correction | Popular videos appear as negatives too often; subtract log sampling prob | [37] |
| Ranker | LambdaRank / listwise softmax on graded labels | Optimises nDCG directly | assumption |
| Multi-objective rank | BCE(click) + regression(watch time) | Combine with weights at serve time | [14] |

## Offline metrics

| Metric | Why | Source |
|---|---|---|
| MRR | Book choice: eval set is (video, text) pairs with one known positive | [1][3] |
| Recall@k (k=100-1000) | Retrieval stage: did the right video survive? | [3] |
| Precision@k, mAP | Considered, less fit for single-positive sets | [3] |
| nDCG@10 | Graded human labels at rank stage | [5] |
| Per-modality ablation | Recall@k with visual / audio / text index off | assumption |
| Public benchmarks | MSR-VTT, HowTo100M-style text-video retrieval R@1/5/10 | [30][33] |

## Online metrics & A/B

| Metric | Verdict in source | Note |
|---|---|---|
| CTR | Rejected as primary: rewards clickbait, not relevance | [3] |
| Video completion rate | Rejected: partial watch can still be relevant | [3] |
| Total watch time from search | Chosen primary | [3] |
| Successful-session rate, zero-result rate, query reformulation rate | Add as guardrails | assumption |
| Latency p99, abandonment | Guardrails | assumption |
| Human side-by-side relevance ratings | Pre-launch gate | Rater guideline style as in [40] |

## Serving & latency

| Concern | Choice | Source |
|---|---|---|
| Index size | 1B x 256-d fp32 = ~1 TB per modality; PQ to 32-64 B/vec = 32-64 GB | [25][26] |
| ANN algorithm | IVF-PQ (FAISS), ScaNN, or HNSW; shard by video id, replicate for QPS | [25][26][27] |
| Encoder serving | Query tower online (distilled, cached); video towers offline batch only | [3] |
| Encoder upgrade | Re-embed full catalog, dual-index, atomic swap; query + video towers must match | assumption |
| Cold start (new upload) | Lexical index in minutes; embeddings within ~1 h; freshness boost in re-rank | assumption |
| Cost control | Embed only sampled frames / shots; skip re-embedding low-view long tail | assumption |

## Monitoring

| Signal | Alert on | Source |
|---|---|---|
| Zero-result and low-confidence query rate | Spike after index or model swap | assumption |
| Embedding drift | Mean cosine of query vs top-1 falling; distribution shift per language | assumption |
| Index health | Index lag behind uploads, shard recall vs brute-force sample | [26] |
| Online metrics | Watch time from search, CTR, reformulation rate by query segment | [3] |
| Policy | Rate of removed / borderline videos in top-10 | assumption |

## Failure modes

| Failure | Symptom | Mitigation |
|---|---|---|
| Clickbait thumbnails/titles | High CTR, short watch | Optimise watch time; quality features [3] |
| Keyword stuffing / tag spam | Lexical path returns irrelevant videos | Field weights, spam classifier, dense path cross-check |
| Visual-text mismatch | Video ranked for words only spoken, never shown (or vice versa) | Per-modality scores into ranker |
| ASR errors / low-resource languages | Missed matches | Multilingual encoders, phonetic fuzzy match |
| Stale index after encoder change | Recall collapse | Version-locked towers, dual-write, shadow eval |
| Head-query bias | Long-tail queries poor | Hard negatives, query rewriting, human labels on tail |
| Harmful / policy content | Unsafe results for benign queries | Filter + demotion stage, safe-search |

## What production systems add beyond the portal answer

| Addition | Why it matters | Source |
|---|---|---|
| Position-bias correction on click/watch labels | Log labels otherwise teach the ranker its own past order | [36] |
| logQ sampling-bias correction for in-batch negatives | Corrects popularity skew in two-tower training at YouTube scale | [37] |
| Pretrained video foundation encoder (YouTube-8M, VideoPrism) | Video tower is pretrained on large YouTube-derived data, then tuned for retrieval | [38][39] |
| Shot-level index for moments | Netflix indexes moments, not whole titles | [22] |
| Hybrid dense + lexical for spoken content | Spotify added semantic episode search alongside term matching | [24] |
| Text in frames (OCR) as a searchable field | Rosetta extracts text from images and video frames | [20] |
| Human rater guidelines and side-by-side evals | Launch gate independent of engagement metrics | [40] |
| Query autocomplete and spell correction | Many sessions start from suggestions; cuts zero-result queries | assumption |

## Interviewer follow-ups

- Why not just use Elasticsearch on titles? What does the visual tower add, and for which queries?
- Frame-level vs video-level encoder: how do you pick, and what does each cost to serve 1B videos? [3]
- How do you fuse lexical and dense scores? Why RRF vs a learned weight? [1][28]
- Why MRR offline and watch time online? Why not CTR? [3]
- How do you get 10M labelled pairs? What if you only have click logs (position bias)?
- How do you index audio: ASR text, audio embeddings, or both?
- The video encoder changes. How do you re-embed 1B videos without downtime?
- How do you make a video uploaded 5 minutes ago searchable?
- One embedding with fused modalities vs separate ANN indexes per modality: tradeoffs?
- How do you add personalisation without hurting relevance for new users?
- How would you return timestamps (moment search) instead of whole videos? [22]
- How do you handle multilingual queries against videos in another language?
- How do you measure and reduce index memory (PQ, dimension reduction)? [25]
- How do you stop the system surfacing harmful content for innocent queries?
- How would you use an LLM or VLM here (query rewriting, captioning frames into text)?
- Click logs are position-biased. How do you build unbiased labels? [36]
- In-batch negatives over-penalise popular videos. How do you correct it? [37]
- Would you pretrain the video tower yourself or start from a video foundation model? [38][39]

## Portal comparison

| Portal | Retrieval | Encoders | Fusion / rank | Metrics | Distinctive point |
|---|---|---|---|---|---|
| ByteByteGo ch. 4 [1][2] | Visual two-tower ANN + text inverted index | BERT text; ViT frames or ViViT | Weighted fusion, then business re-rank | MRR offline; watch time online | Explicit rejection of CTR and completion rate |
| alirezadir AIMLInterviews [3] | Same as book (summary); ANN for visual, Elasticsearch for text | Same; video encoder choice left open | Weighted sum or re-rank model | P@k, mAP, R@k, MRR; watch time | Gives scale numbers (1B, 50M DAU, 10M pairs); deployment and scaling/monitoring sections left empty |
| alirezadir image search [5] | Embedding + ANN (FAISS; tree, LSH, clustering), VQ/PQ | CNN / ViT, contrastive | Business-rule re-rank | MRR, R@k, nDCG; CTR, time spent | Positive sources: human, clicks, augmentation |
| Chip Huyen exercises [8] | Open question (Google Images) | n/a | n/a | n/a | Text query to images; separate audio trigger-word exercise |
| Educative / Khang Pham [9][10][11] | Staged funnel (retrieval -> rank) | Generic search features | LTR | nDCG, engagement | Strong on LTR and data generation; weak on visual content |
| Production (Netflix, Meta, Spotify) [19][21][22][24] | Hybrid lexical + embedding | CLIP-style, OCR, ASR | Learned | Internal | Moment-level index; text-in-video; NL podcast search |

## References

1. A. Aminian & A. Xu, "Machine Learning System Design Interview" (ByteByteGo; ch. 4 YouTube Video Search), book overview, 2023. https://opencourser.com/book/xop33f/machine-learning-system-design-interview (ch-ref, book)
2. ByteByteGo, "YouTube Video Search" (MLSDI course chapter), bytebytego.com, 2023. https://bytebytego.com/courses/machine-learning-system-design-interview/youtube-video-search (book; URL matches the chapter-URL pattern used across this atlas; host unreachable in review)
3. alirezadir, "Multimodal Video Search System" (mlsd-mm-video-search.md), GitHub, 2023. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/mlsd-mm-video-search.md (fetched)
4. alirezadir, "ML System Design" template (lists image/video, multimodal and visual search), GitHub, 2025. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/ml-system-design.md (fetched)
5. alirezadir, "Image Search System" (mlsd-image-search.md), GitHub, 2023. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/mlsd-image-search.md (fetched)
6. alirezadir, "ML at Companies" (ml-companies.md), GitHub, 2025. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/ml-companies.md (fetched)
7. Engineer1999, "A Curated List of ML System Design Case Studies", GitHub, 2024. https://github.com/Engineer1999/A-Curated-List-of-ML-System-Design-Case-Studies (fetched)
8. C. Huyen, "Machine Learning Systems Design: Exercises", GitHub, 2019. https://github.com/chiphuyen/machine-learning-systems-design/blob/master/content/exercises.md (fetched)
9. Educative, "Grokking the Machine Learning Interview", educative.io, 2024. https://www.educative.io/courses/grokking-the-machine-learning-interview (ch-ref)
10. Educative / K. Pham, "Machine Learning System Design", educative.io, 2021. https://educative.io/courses/machine-learning-system-design/introduction (ch-ref)
11. K. Pham, "machine-learning-interview", GitHub, 2021. https://github.com/khangich/machine-learning-interview (fetched; lists Airbnb search ranking and YouTube recs)
12. Exponent, "Introduction to the ML System Design Interview", tryexponent.com, 2025. https://www.tryexponent.com/courses/ml-system-design/mlsd-intro (ch-ref)
13. Hello Interview, "ML System Design in a Hurry: Delivery Framework", hellointerview.com, 2025. https://www.hellointerview.com/learn/ml-system-design/in-a-hurry/delivery (ch-ref)
14. IGotAnOffer, "Google Machine Learning Engineer Interview", igotanoffer.com, 2025. https://igotanoffer.com/blogs/tech/google-machine-learning-engineer-interview (ch-ref)
15. IGotAnOffer, "Meta Machine Learning Engineer Interview", igotanoffer.com, 2025. https://igotanoffer.com/blogs/tech/facebook-machine-learning-engineer-interview (ch-ref)
16. IGotAnOffer, "Machine Learning System Design Interview", igotanoffer.com, 2025. https://igotanoffer.com/en/advice/machine-learning-system-design-interview (ch-ref)
17. P. Halina, "ML Systems Design Interview Guide", patrickhalina.com, 2020. http://patrickhalina.com/posts/ml-systems-design-interview-guide/ (ch-ref)
18. P. Covington, J. Adams, E. Sargin, "Deep Neural Networks for YouTube Recommendations", RecSys / Google Research, 2016. https://research.google/pubs/pub45530/ (ch-ref)
19. Meta Engineering, "Under the hood: Photo search", engineering.fb.com, 2017. https://engineering.fb.com/ml-applications/under-the-hood-photo-search/ (listed in [6])
20. Meta AI, "Rosetta: Understanding text in images and videos with machine learning", ai.facebook.com, 2018. https://ai.facebook.com/blog/rosetta-understanding-text-in-images-and-videos-with-machine-learning/ (listed in [6])
21. J.-T. Huang et al., "Embedding-based Retrieval in Facebook Search", KDD, 2020. https://arxiv.org/abs/2006.11632 (known)
22. Netflix Technology Blog, "Building In-Video Search", 2023. https://netflixtechblog.com/building-in-video-search-936766f0017c (listed in [7])
23. Google, "How Google Search works", google.com. https://www.google.com/search/howsearchworks/ (listed in [6])
24. Spotify Engineering, "Introducing Natural Language Search for Podcast Episodes", 2022. https://engineering.atspotify.com/2022/03/introducing-natural-language-search-for-podcast-episodes/ (listed in [7])
25. J. Johnson, M. Douze, H. Jegou, "Billion-scale similarity search with GPUs" (FAISS), 2017. https://arxiv.org/abs/1702.08734 (ch-ref)
26. R. Guo et al., "Accelerating Large-Scale Inference with Anisotropic Vector Quantization" (ScaNN), ICML, 2020. https://arxiv.org/abs/1908.10396 (ch-ref)
27. Y. Malkov, D. Yashunin, "Efficient and robust approximate nearest neighbor search using HNSW graphs", 2016. https://arxiv.org/abs/1603.09320 (ch-ref)
28. G. Cormack, C. Clarke, S. Buettcher, "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods", SIGIR, 2009. https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf (known)
29. A. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (CLIP), 2021. https://arxiv.org/abs/2103.00020 (ch-ref)
30. H. Luo et al., "CLIP4Clip: An Empirical Study of CLIP for End to End Video Clip Retrieval", 2021. https://arxiv.org/abs/2104.08860 (known)
31. A. Arnab et al., "ViViT: A Video Vision Transformer", ICCV, 2021. https://arxiv.org/abs/2103.15691 (known)
32. A. Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision" (Whisper), 2022. https://arxiv.org/abs/2212.04356 (known)
33. A. Miech et al., "HowTo100M: Learning a Text-Video Embedding by Watching Hundred Million Narrated Video Clips", ICCV, 2019. https://arxiv.org/abs/1906.03327 (known)
34. A. Miech et al., "End-to-End Learning of Visual Representations from Uncurated Instructional Videos" (MIL-NCE), CVPR, 2020. https://arxiv.org/abs/1912.06430 (known)
35. R. Girdhar et al., "ImageBind: One Embedding Space To Bind Them All", CVPR, 2023. https://arxiv.org/abs/2305.05665 (known)
36. T. Joachims, A. Swaminathan, T. Schnabel, "Unbiased Learning-to-Rank with Biased Feedback", WSDM, 2017. https://arxiv.org/abs/1608.04468 (known)
37. X. Yi et al., "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations" (YouTube), RecSys, 2019. https://doi.org/10.1145/3298689.3346996 (known)
38. S. Abu-El-Haija et al., "YouTube-8M: A Large-Scale Video Classification Benchmark", Google, 2016. https://arxiv.org/abs/1609.08675 (known)
39. L. Zhao et al., "VideoPrism: A Foundational Visual Encoder for Video Understanding", Google, ICML, 2024. https://arxiv.org/abs/2402.13217 (known)
40. Google, "Search Quality Rater Guidelines", guidelines.raterhub.com. https://static.googleusercontent.com/media/guidelines.raterhub.com/en//searchqualityevaluatorguidelines.pdf (known)
