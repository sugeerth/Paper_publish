# 33. Design Semantic Search and Duplicate Question Detection (Quora, Stack Overflow)

> Prompt as usually given: "When a user types a new question on Quora or Stack Overflow,
> show similar existing questions and flag likely duplicates before they post." Variants:
> "merge duplicate questions offline", "semantic search over a Q&A corpus", and "related
> questions" in the side panel. Chip Huyen's exercise #11 asks it almost verbatim [7].
>
> Sourcing note: in the review pass the web-search budget was used up and arxiv.org,
> stackoverflow.blog, quora.com, kaggle.com and huggingface.co were unreachable from the
> sandbox. GitHub pages were fetched and read (exercise numbers re-checked against the raw
> file). Each reference carries a tag.
> (fetched) means the page was read for this chapter. (ch-ref) means the URL was verified
> in another chapter of this atlas. (known) means a canonical paper or post cited by its
> stable URL but not re-read for this chapter. The "asked it" evidence for company tags is
> thin, so treat it as indicative only.

## Company tags

| Company | Evidence | Notes | Source |
|---|---|---|---|
| Quora | Built it: the Question Pairs dataset (2017), a Kaggle competition, and an engineering post on deep-learning question matching | The canonical version of this problem | [10][11][12] |
| Stack Overflow / Stack Exchange | Built it: "similar questions" while you type, the "close as duplicate" workflow, and semantic search (2023). Asked it: Chip Huyen exercise #11 frames it as "StackOverflow similar questions" | Moderator duplicate closures give free labels | [7][13] |
| Meta | Built it: embedding-based retrieval in Facebook Search. Asked it (indicative): MLE loops cover search and retrieval design | The same two-tower and ANN template | [16][24] |
| Google, Amazon, LinkedIn | Asked it (indicative): text search and semantic search appear in MLSD question banks for these loops. No interview report was confirmed in this run | Treat as *likely*, not confirmed | [2][5][25][26] |
| Reddit, Zendesk-style support desks, Discourse forums | Same pattern: deflecting a ticket or post to an existing answer. No interview report was confirmed | Variant: "suggest a help-centre article" | assumption |

**Portals that cover it**

| Portal | Treatment | Link |
|---|---|---|
| Chip Huyen, ML Systems Design exercises | #11 "StackOverflow similar questions" and #8 "Quora answer ranking". Prompts only, no solution | [7] |
| alirezadir AIMLInterviews | Lists "text query search (full-text and semantic)". The search case study gives the staged funnel, LTR and NDCG | [1][2] |
| Educative "Grokking the ML Interview" (Search Ranking) | Document selection, then staged ranking, then NDCG. Use it as the generic template | [3] |
| Khang Pham, ML Design Interview / repo | No dedicated case. Search ranking (Airbnb) and embedding papers are the nearest | [4] |
| ByteByteGo (Aminian & Xu) | No dedicated chapter. "YouTube Video Search" and "Visual Search" supply the dual-encoder and ANN pattern | [5][6] |
| Exponent, Hello Interview, IGotAnOffer, Patrick Halina | Generic MLSD framework and delivery advice | [25][26][27][28] |
| sentence-transformers docs (practitioner reference) | Bi-encoder retrieval, cross-encoder rerank, paraphrase mining. Includes a Quora-duplicates training recipe | [8][9][29] |

## Requirements & scale

| Dimension | Value | Source |
|---|---|---|
| Functional (online) | As the user types a title or body, return the top-k similar questions (k is about 5-10) | [7] |
| Functional (dup flag) | Show a "possible duplicate" banner when P(dup) is at least a threshold. Moderators can close or merge | [13], assumption |
| Functional (offline) | Mine duplicate clusters across the corpus and merge them into a canonical question | [9] |
| Functional (search) | Natural-language search box over questions and answers, using lexical and semantic retrieval | [13] |
| Corpus | Stack Overflow has about 24M questions, Quora has hundreds of millions | assumption (typical) |
| Labelled pairs | Quora QQP: 404,290 labelled pairs [8]; about 37% positive | [8][10][11] |
| QPS | Search plus typeahead-similar: about 1-5k QPS average, about 15k peak | assumption |
| Latency | p99 under 150 ms for the similar-questions panel. Can be debounced to every 300-500 ms of typing | assumption |
| Freshness | New question searchable within a minute or two through a streaming embed-and-upsert | assumption |
| Precision need | The dup banner needs high precision, about 0.9 or higher. A false "duplicate" frustrates new users | assumption |
| Languages | Usually English-first. The multilingual follow-up uses a multilingual encoder | assumption |
| Business goal | Fewer duplicate posts, more answers reused, faster time-to-answer, less moderator load | assumption |

## DIAGRAM 1: Online serving path

**Figure 33.1 - Query-time path: debounce, then hybrid retrieval (BM25 + ANN), then
filtering, cross-encoder ranking, and the duplicate decision and policy.**

```
 +---------+ title+body (debounced) +-------------------+
 | Asker / |----------------------->| Query service     |  normalise, lang-id,
 | Searcher|                        | (clean, spell, QU)|  tag predict, PII strip
 +---------+                        +----+---------+----+
      ^                                  | q text  | q text
      |                                  v         v
      |                  +---------------+--+   +--+----------------+
      |                  | Bi-encoder       |   | Lexical retrieval |
      |                  | SBERT d=384-768  |   | BM25 / Elastic    |
      |                  +--------+---------+   +---------+---------+
      |                           | q_emb                 | top ~200
      |                           v                       |
      |                  +------------------+             |
      |                  | ANN index (HNSW/ |             |
      |                  | IVF-PQ, FAISS)   |             |
      |                  +--------+---------+             |
      |                           | top ~200              |
      |                           v                       v
      |                  +--------------------------------------+
      |                  | Merge + RRF fusion -> ~300 cands     |
      |                  +------------------+-------------------+
      |                                     |
      |                                     v
      |   +----------------+   +--------------------------------------+
      |   | Feature store  |-->| Filters: deleted/closed, NSFW, lang, |
      |   | (votes, ans    |   | same-site/tag scope, age, dedup ids  |
      |   |  cnt, accepted,|   +------------------+-------------------+
      |   |  views, tags)  |                      | ~100
      |   +----------------+                      v
      |                       +--------------------------------------+
      |   +----------------+  | Ranker: cross-encoder P(dup|q,c)     |
      |   | Model store    |->| + GBDT blend (quality, answered,     |
      |   | (bi/cross enc, |  |   votes, recency)                    |
      |   |  GBDT, thresh) |  +------------------+-------------------+
      |   +----------------+                     | scored top-20
      |                                          v
      |                       +--------------------------------------+
      |  top-k similar +      | Policy: P(dup) >= tau -> dup banner; |
      +-----------------------| canonical redirect; diversity; show  |
         dup banner           | answered/accepted first              |
                              +------------------+-------------------+
                                                 | impressions, clicks,
                                                 | "yes, my q is answered",
                                                 v posted anyway, closed-as-dup
                              +--------------------------------------+
                              | Event log (Kafka) -> warehouse       |
                              +--------------------------------------+
```

## DIAGRAM 2: Offline training and data pipeline

**Figure 33.2 - Labels come from moderator closures, user clicks and merges. Two models
are trained (bi-encoder and cross-encoder), then the corpus is re-indexed and the
duplicate threshold is calibrated.**

```
 +-------------------+   +--------------------+   +--------------------------+
 | Q&A corpus (DB)   |   | Moderation log:    |   | Serving logs: panel      |
 | questions,answers |   | closed-as-dup(a,b),|   | impressions, clicks,     |
 | tags, edits       |   | merges, reopen     |   | "answered my q", abandon |
 +---------+---------+   +---------+----------+   +--------+-----------------+
           |                       | strong +              | weak +/- (clicks,
           |                       | (reopen => -)         | skip-above)
           v                       v                       v
 +--------------------------------------------------------------------------+
 | Label builder: positives = dup links + transitive closure (A~B, B~C);    |
 | negatives = random, BM25-hard, ANN-hard, reopened; human audit sample    |
 +-------------------------------------+------------------------------------+
                                       | pairs (q, c, y), group-split by
                                       v cluster id (no leak across splits)
 +-------------------+     +---------------------------+     +----------------+
 | Feature pipeline  |---->| Training                  |---->| Evaluation     |
 | text clean, tags, |     | (a) bi-encoder: MNRL +    |     | Recall@k, MRR, |
 | votes, ans stats  |     |     hard negatives        |     | nDCG@10, PR-AUC|
 | -> feature store  |     | (b) cross-encoder: BCE    |     | P@tau, slices  |
 +-------------------+     | (c) GBDT blend, calibrate |     | (tag, lang,new)|
                           +---------------------------+     +-------+--------+
                                                                     | pass gates
                                                                     v
 +--------------------+   +------------------------+   +--------------------------+
 | Corpus re-embed    |<--| Model registry         |-->| Deploy: shadow ->        |
 | (batch, GPU) ->    |   | versions bi/cross/GBDT |   | canary -> A/B; threshold |
 | new ANN index,     |   | + tau per surface      |   | tau set from target      |
 | blue/green swap    |   +------------------------+   | precision on fresh labels|
 +---------+----------+                                +------------+-------------+
           |                                                        |
           +------------------> serving (Fig 33.1) <----------------+
                                       |
                                       | new logs, new closures
                                       +----------> feedback loop (weekly retrain)
```

## DIAGRAM 3: Model architecture

**Figure 33.3 - Bi-encoder (Siamese, shared weights) for retrieval and a cross-encoder for
the precise duplicate score. The cross-encoder's logit plus metadata go into a GBDT that
outputs a calibrated P(dup).**

```
        RETRIEVAL: BI-ENCODER (offline corpus side, online query side)

  question A (title [SEP] body[:256])      question B (title [SEP] body[:256])
            |                                          |
            v                                          v
  +---------------------+    shared weights   +---------------------+
  | Transformer encoder |<------------------->| Transformer encoder |
  | (MiniLM/MPNet/E5)   |                     | (MiniLM/MPNet/E5)   |
  +----------+----------+                     +----------+----------+
             | mean-pool                                 | mean-pool
             v                                           v
       +-----------+                               +-----------+
       | u (d=384) |                               | v (d=384) |
       +-----+-----+                               +-----+-----+
             |            cos(u, v) / temperature        |
             +--------------------+----------------------+
                                  v
            loss: MNRL (in-batch softmax over B-1 negatives + hard negs)

        PRECISION: CROSS-ENCODER + BLEND

  "[CLS] q_title q_body [SEP] c_title c_body [SEP]"   (<= 512 tokens)
                       |
                       v
  +--------------------------------------------+
  | Transformer (BERT/DeBERTa), full attention |
  | across both questions                      |
  +----------------------+---------------------+
                         | [CLS] -> linear -> logit s_ce
                         v
  +--------------------------------------------+   meta: tag Jaccard, cos(u,v),
  | GBDT / LR blend                            |<--BM25 score, c has accepted ans,
  | -> calibrated P(dup) (Platt / isotonic)    |   votes, age gap, same author
  +----------------------+---------------------+
                         v
            P(dup) -> rank score + dup banner if P >= tau
```

## DIAGRAM 4: Variant, offline duplicate mining and Quora's 2017 approach

**Figure 33.4 - (a) Batch paraphrase mining for merge and cleanup [9], and (b) the model
family from Quora's 2017 engineering post [12] and the QQP-era literature [17][18].**

```
 (a) OFFLINE MINING (nightly, corpus-wide)

 +------------+  embed   +-------------+ kNN (k=10) +---------------+ pairs with
 | all Qs     |--------->| corpus emb  |----------->| candidate     | cos >= t1
 | (N ~ 10^7) |  (GPU)   | matrix      | ANN/chunked| pairs ~N*k    |-----------+
 +------------+          +-------------+            +---------------+           |
                                                                                v
 +--------------------+  edges P>=tau   +------------------+  score  +-------------+
 | Merge queue:       |<----------------| Graph: connected |<--------| Cross-enc   |
 | canonical = most   |  clusters       | components /     |         | rescoring   |
 | answers/views;     |                 | cap cluster size |         +-------------+
 | moderator review   |                 +------------------+
 +--------------------+

 (b) QUORA-STYLE PAIR CLASSIFIER (2017)

  q1 --> embed (GloVe) --> LSTM/attention --> h1 --+
                                                   +-> [h1, h2, |h1-h2|, h1*h2]
  q2 --> embed (GloVe) --> LSTM/attention --> h2 --+        |
                                                            v
                         hand features (word overlap,  --> MLP / RF --> P(dup)
                         TF-IDF cos, length diff)
  Baseline: random forest on hand features. Later: LSTM and decomposable attention.
  BiMPM became a strong reference on QQP.
```

## Data & labels

| Source | Label type | Volume / note | Pitfall |
|---|---|---|---|
| Moderator "closed as duplicate" links | Strong positive (a, b) | Stack Overflow-scale, millions; used as labels in early forum work [15] | Biased to popular canonical targets |
| Reopened after dup-close | Negative or hard negative | Small | Noisy, depends on the moderator |
| Quora QQP | Binary human label | 404k pairs, about 37% positive | Sampling bias. Not random pairs [11] |
| Panel click plus "this answers my question" | Weak positive | Very large | Position bias. Needs debiasing or IPS |
| Posted anyway after the panel was shown | Weak negative | Large | User may not have read it |
| Random pairs | Easy negative | Unlimited | Too easy on its own |
| BM25 and ANN top-k non-dups | Hard negative | Mined per epoch | False negatives (unlabelled dups) |
| Human raters | Gold eval set | 5-20k pairs, stratified | Cost. Keep a fresh monthly sample |

## Features

| Group | Features | Used by |
|---|---|---|
| Text (dense) | Bi-encoder cos(u, v), cross-encoder logit | Retrieval, ranker |
| Text (lexical) | BM25, TF-IDF cos, word or char n-gram Jaccard, edit distance, length diff | Blend, baseline |
| Entity / code | Shared tags, library names, error strings, code-block overlap (Stack Overflow) | Blend |
| Candidate quality | Has accepted answer, answer count, score, views, closed or locked | Blend, policy |
| Temporal | Candidate age, age gap, tag-version drift (for example Python 2 vs 3) | Blend |
| User / context | Asker reputation, site, language, entry surface (ask page vs search) | Policy, threshold |

## Model choices

| Stage | Baseline | Advanced | Trade-off |
|---|---|---|---|
| Retrieval | BM25 over title and body | Fine-tuned bi-encoder plus ANN, hybrid with BM25 through RRF | Dense covers paraphrase. BM25 covers exact error strings and code [14][16][22][32] |
| Query understanding | Lower-case, strip code noise | Spell-check, normalisation, expansion, tag/intent prediction | Front of the funnel in the portal search template [1] |
| Pair scoring | LR or RF on lexical features | Cross-encoder (BERT/DeBERTa) fine-tuned on dup pairs | About 100x the compute per pair. Limit to the top ~100 [23][29] |
| Late interaction | n/a | ColBERT (token-level MaxSim) | Better than a bi-encoder, but a larger index [20] |
| Blend | Cross-encoder score alone | GBDT on CE logit plus quality and meta | Lets you push answered or accepted canonical questions up |
| LLM | n/a | LLM judge for borderline P in [0.4, 0.7], or a rewrite of the query | Latency and cost. Use offline or async |
| Clustering | Threshold on pairs | Connected components with cluster-size caps | Transitive chains drift. Cap size and re-verify |

## Loss functions

| Model | Loss | Notes |
|---|---|---|
| Bi-encoder | MultipleNegativesRankingLoss (in-batch softmax, InfoNCE) | Needs positives only. Larger batches help [8][19] |
| Bi-encoder (pairs) | OnlineContrastiveLoss (margin 0.5) | Uses labelled 0/1 pairs and focuses on hard pairs [8] |
| Multi-task | MNRL plus ContrastiveLoss, alternating batches | One for ranking, one for classification [8] |
| Cross-encoder | Binary cross-entropy on (q, c, y) | Calibrate afterwards with Platt or isotonic |
| Blend ranker | LambdaMART (nDCG) or logloss | Pairwise if graded labels exist |
| Distillation | MSE or KL from cross-encoder to bi-encoder scores | Improves recall without serving cost |

## Offline metrics

| Stage | Metric | Target example |
|---|---|---|
| Retrieval | Recall@100 of the known dup target | 0.95 or higher |
| Ranking | MRR, nDCG@10, Hit@5 | Report by slice |
| Dup classifier | PR-AUC, precision at the threshold, recall at precision 0.9 | Precision at tau of 0.9 or higher |
| Calibration | ECE or a reliability plot | Needed for a stable tau |
| Benchmarks | QQP accuracy and F1; BEIR Quora (10k queries, 523k docs) and CQADupStack (13,145 queries, 457k docs) nDCG@10 | Sanity check vs public models [11][21][33] |
| Slices | New tags, low-resource languages, code-heavy questions, very short titles | No slice regresses more than 2% |

## Online metrics & A/B

| Metric | Direction | Note |
|---|---|---|
| Duplicate post rate (later closed as dup) | Down | Primary north star |
| Ask-flow deflection: user leaves satisfied after clicking a similar Q | Up | Guard against it suppressing genuinely new questions |
| Time to first useful answer | Down | Includes deflected sessions |
| Similar-panel CTR, search success rate | Up | Session-level |
| Successful-session rate, dwell on the clicked question | Up | CTR alone can reward clickbait titles [1] |
| Wrongly closed-as-dup or reopen rate | Down (guardrail) | Hurts new users |
| New-question volume and answerer engagement | Not down (guardrail) | Too much deflection starves the community |
| Latency p99, error rate | Within SLO | Infrastructure guardrail |

## Serving & latency

| Step | Budget (p99) | Notes |
|---|---|---|
| Debounce and query cleaning | about 5 ms | Client debounce of 300-500 ms is not counted |
| Bi-encoder query embed (CPU/ONNX or GPU) | about 10-15 ms | Distilled MiniLM, int8 |
| ANN (HNSW, sharded) plus BM25 in parallel | about 15-20 ms | k about 200 each |
| Fusion, filters, feature fetch | about 15 ms | Batched feature-store read |
| Cross-encoder on ~50-100 pairs (GPU, batched) | about 40-60 ms | Truncate bodies. Cache popular candidates |
| GBDT, policy, response | about 10 ms | |
| Total | about 120-150 ms | Search path. The dup check at submit can take longer |

## Monitoring

| Signal | Alert on |
|---|---|
| Embedding drift (mean cos to centroid, new-tag share) | Shift beyond a baseline band |
| P(dup) score distribution and banner rate | Sudden jump or drop (threshold or calibration broke) |
| Reopen rate after an auto-suggested dup | Rise above the guardrail |
| ANN recall vs exact kNN, measured on a sample (HNSW / IVF-PQ) [30][31][34] | Recall@100 below 0.95 |
| Index freshness lag (post to searchable) | More than 5 min |
| Latency p99 and GPU queue depth | SLO breach |

## Failure modes

| Failure | Example | Mitigation |
|---|---|---|
| Near-miss paraphrase | "How to sort a list in Python" vs "...in Java" | Entity and tag features, cross-encoder, hard negatives |
| Version drift | Python 2 vs 3 answers now wrong | Recency and version features. Prefer canonical questions with recent edits |
| Over-deflection | New edge-case questions get marked dup | High tau, "not a dup" button, guardrail metric |
| Transitive chain drift | A~B~C~D but A is not D | Cap cluster size, re-verify edges with the CE |
| Label bias | Moderators favour popular targets | Human audit set, IPS on click labels |
| Lexical-only blind spot | Exact error string missed by dense retrieval | Hybrid BM25 plus dense with RRF |
| Spam and SEO copies | Copied questions cluster with real ones | Spam classifier upstream, author features |
| Cold index after model swap | New embeddings are not comparable with the old index | Full re-embed, blue/green index swap |
| Asymmetric neighbours in mining | B is A's top match but A is not B's, so (A,B),(B,C) chains appear | Symmetrise pairs, re-score with the CE, then cluster [9] |
| Quadratic mining cost | All-pairs compare is impractical beyond about 10k questions | Chunked top-k mining (`query_chunk_size`, `corpus_chunk_size`, `max_pairs`) or ANN [9] |

## Interviewer follow-ups

- Is "similar" the same as "duplicate"? How do the two surfaces differ in threshold and metric?
- Where do labels come from, and how do you handle false negatives among mined hard negatives?
- Bi-encoder vs cross-encoder: why use both, and where does each run?
- How do you set and maintain the duplicate threshold tau? What happens when the calibration drifts?
- How do you prevent leakage when splitting pairs (the same question in train and test)?
- How do you handle code snippets, stack traces and exact error messages?
- How do you re-index 20M+ questions when the embedding model changes, without downtime?
- How do you make a new question searchable within a minute?
- How do you evaluate offline vs online? Which metric is the north star?
- How do you stop over-deflection that discourages new questions?
- How do you handle multilingual sites (for example, Stack Overflow in Portuguese)?
- How would you mine and merge duplicate clusters offline? Why not use pure transitive closure?
- Where would an LLM help (query rewrite, borderline judging, answer synthesis), and at what cost?
- How do you debias click labels from the similar-questions panel?
- How do you scale ANN to 10^8 vectors (PQ, sharding, memory per vector)?
- Paraphrase mining returns asymmetric pairs: how do you turn them into stable clusters? [9]
- How many candidates should the cross-encoder see (about 100 is the common default), and why? [29]
- Which public benchmark would you sanity-check on (BEIR Quora, CQADupStack)? [33]

## Portal comparison

| Portal | Framing | Retrieval | Scoring model | Metrics emphasis |
|---|---|---|---|---|
| Chip Huyen exercises [7] | #11 "StackOverflow similar questions", prompt only | Left to the candidate | Left to the candidate | Open-ended |
| Educative Grokking, Search Ranking [3] | Generic search ranking | Inverted index plus embeddings (document selection) | Staged LR to LambdaMART or DNN | nDCG, session success |
| alirezadir search case [1] | Generic text search funnel | Inverted index, weighted linear score | Pointwise stage 1 then pairwise LambdaMART | P@k, MRR, nDCG, CTR |
| ByteByteGo video and visual search [5][6] | Text-to-item search | Dual encoder plus ANN, fused with text search | Contrastive dual encoder | MRR, CTR, watch time |
| sentence-transformers docs [8][9][29] | Practitioner semantic search and dup mining | Bi-encoder plus ANN, optional BM25 | Cross-encoder rerank | IR metrics (Recall, MRR, nDCG) |
| Quora engineering 2017 [12] | Pair classification (dup or not) | Not covered (pairwise) | RF baseline, then LSTM and attention | Accuracy and F1 on QQP |

## References

1. alirezadir, "Search System" (mlsd-search.md), GitHub, 2023. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/mlsd-search.md (fetched)
2. alirezadir, "ML System Design" template (lists text query search: full-text and semantic), GitHub, 2025. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/ml-system-design.md (fetched)
3. Educative, "Grokking the Machine Learning Interview", "Document selection" lesson (Search Ranking), educative.io, n.d. https://www.educative.io/courses/grokking-the-machine-learning-interview/document-selection (ch-ref)
4. K. Pham, "machine-learning-interview", GitHub, 2021. https://github.com/khangich/machine-learning-interview (fetched)
5. ByteByteGo, "YouTube Video Search" (MLSDI course chapter), bytebytego.com, 2023. https://bytebytego.com/courses/machine-learning-system-design-interview/youtube-video-search (ch-ref)
6. ByteByteGo, "Visual Search System" (MLSDI course chapter), bytebytego.com, 2023. https://bytebytego.com/courses/machine-learning-system-design-interview/visual-search-system (ch-ref)
7. C. Huyen, "Machine Learning Systems Design: Exercises" (#8 Quora answer ranking, #11 StackOverflow similar questions), GitHub, 2019. https://github.com/chiphuyen/machine-learning-systems-design/blob/master/content/exercises.md (fetched)
8. UKPLab, sentence-transformers, "Quora Duplicate Questions" training example, GitHub, 2024. https://github.com/UKPLab/sentence-transformers/tree/master/examples/sentence_transformer/training/quora_duplicate_questions (fetched)
9. UKPLab, sentence-transformers, "Paraphrase Mining" application, GitHub, 2024. https://github.com/UKPLab/sentence-transformers/tree/master/examples/sentence_transformer/applications/paraphrase-mining (fetched)
10. Quora, "First Quora Dataset Release: Question Pairs", quoradata.quora.com, 2017. https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs (known)
11. Kaggle, "Quora Question Pairs" competition, kaggle.com, 2017. https://www.kaggle.com/c/quora-question-pairs (known)
12. Quora Engineering, "Semantic Question Matching with Deep Learning", quoraengineering.quora.com, 2017. https://quoraengineering.quora.com/Semantic-Question-Matching-with-Deep-Learning (known)
13. Stack Overflow Blog, "Ask like a human: Implementing semantic search on Stack Overflow", stackoverflow.blog, 2023. https://stackoverflow.blog/2023/07/31/ask-like-a-human-implementing-semantic-search-on-stack-overflow/ (known)
14. N. Reimers, I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks", EMNLP, arXiv, 2019. https://arxiv.org/abs/1908.10084 (known)
15. D. Bogdanova et al., "Detecting Semantically Equivalent Questions in Online User Forums", CoNLL, ACL Anthology, 2015. https://aclanthology.org/K15-1013/ (known)
16. J.-T. Huang et al., "Embedding-based Retrieval in Facebook Search", KDD, arXiv, 2020. https://arxiv.org/abs/2006.11632 (ch-ref)
17. Z. Wang, W. Hamza, R. Florian, "Bilateral Multi-Perspective Matching for Natural Language Sentences" (BiMPM, evaluated on QQP), arXiv, 2017. https://arxiv.org/abs/1702.03814 (known)
18. A. Parikh et al., "A Decomposable Attention Model for Natural Language Inference", arXiv, 2016. https://arxiv.org/abs/1606.01933 (known)
19. M. Henderson et al., "Efficient Natural Language Response Suggestion for Smart Reply" (in-batch negatives), arXiv, 2017. https://arxiv.org/abs/1705.00652 (known)
20. O. Khattab, M. Zaharia, "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT", SIGIR, arXiv, 2020. https://arxiv.org/abs/2004.12832 (known)
21. N. Thakur et al., "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of IR Models" (includes Quora, CQADupStack), arXiv, 2021. https://arxiv.org/abs/2104.08663 (known)
22. V. Karpukhin et al., "Dense Passage Retrieval for Open-Domain Question Answering", arXiv, 2020. https://arxiv.org/abs/2004.04906 (known)
23. R. Nogueira, K. Cho, "Passage Re-ranking with BERT", arXiv, 2019. https://arxiv.org/abs/1901.04085 (known)
24. IGotAnOffer, "Meta Machine Learning Engineer Interview", igotanoffer.com, 2025. https://igotanoffer.com/blogs/tech/facebook-machine-learning-engineer-interview (ch-ref)
25. Exponent, "Introduction to the ML System Design Interview", tryexponent.com, 2025. https://www.tryexponent.com/courses/ml-system-design/mlsd-intro (ch-ref)
26. IGotAnOffer, "Machine Learning System Design Interview", igotanoffer.com, 2025. https://igotanoffer.com/en/advice/machine-learning-system-design-interview (ch-ref)
27. Hello Interview, "ML System Design in a Hurry: Delivery Framework", hellointerview.com, 2025. https://www.hellointerview.com/learn/ml-system-design/in-a-hurry/delivery (ch-ref)
28. P. Halina, "ML Systems Design Interview Guide", patrickhalina.com, 2020. http://patrickhalina.com/posts/ml-systems-design-interview-guide/ (ch-ref)
29. UKPLab, sentence-transformers, "Retrieve & Re-Rank" application, GitHub, 2024. https://github.com/UKPLab/sentence-transformers/tree/master/examples/sentence_transformer/applications/retrieve_rerank (fetched)
30. J. Johnson, M. Douze, H. Jegou, "Billion-scale similarity search with GPUs" (FAISS), arXiv, 2017. https://arxiv.org/abs/1702.08734 (ch-ref)
31. Y. Malkov, D. Yashunin, "Efficient and robust approximate nearest neighbor search using HNSW graphs", arXiv, 2016. https://arxiv.org/abs/1603.09320 (ch-ref)
32. G. Cormack, C. Clarke, S. Buettcher, "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods", SIGIR, 2009. https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf (ch-ref)
33. BEIR authors, "BEIR: A Heterogeneous Benchmark for Information Retrieval" (dataset table incl. Quora, CQADupStack), GitHub, 2021. https://github.com/beir-cellar/beir (fetched)
34. Meta FAIR, "Faiss" library repository, GitHub, 2024. https://github.com/facebookresearch/faiss (fetched)
