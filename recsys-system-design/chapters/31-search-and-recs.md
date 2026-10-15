# 31. Search Meets Recommendation: Learning-to-Rank, Personalized and Semantic Search, Hybrid Retrieval and Unified Models

Search and recommendation are usually built by different teams, but they solve the same problem. Given a context, return a short ranked list of items from a very large corpus. In search the context contains an explicit query. In recommendation the query is implicit: who the user is, what they did recently, and the page they are on. Eugene Yan's widely cited "System Design for Discovery" draws both as one two-stage system, with offline training and indexing and online retrieval followed by ranking [40]. Amazon's research line goes further and treats product search as a personalized ranking problem, where "every interaction on the site is a recommendation" [9][10].

This chapter follows the parts of the stack where the two fields meet:

1. **Learning-to-rank (LTR).** RankNet, LambdaRank and LambdaMART, the losses that most production search rankers still descend from, and the unbiased-LTR corrections for click logs [1]-[6].
2. **Query understanding and semantic matching.** From DSSM's letter-trigram towers to BERT-class query encoders [7][41][42].
3. **Embedding-based retrieval (EBR) in production search.** Facebook, Amazon, Etsy, Taobao, JD, Airbnb and Pinterest [8]-[21][38][39].
4. **Hybrid lexical plus vector retrieval.** Boolean filters inside ANN indexes, score fusion (RRF, normalization), learned sparse models, late interaction [8][17][23]-[29].
5. **Personalized search.** Session and long-term signals in the ranker, from Bing to Airbnb [12]-[16][30].
6. **Unified search and recommendation.** Shared embedding spaces, joint behavior models and single generative models that serve both tasks (Pinterest, Kuaishou, Spotify, Amazon) [20][31]-[37].

**A note on sourcing.** This chapter was written in an environment where the live web-search budget had already been used up and most company blogs and arXiv could not be reached directly. Primary sources that could be fetched (Microsoft Research pages, the Burges LambdaMART report, GitHub repositories and documentation for LightGBM, TF-Ranking, Vespa, ColBERT, SPLADE, SPTAG, DiskANN, the Elasticsearch LTR plugin, Amazon's ESCI dataset, UniSAR and a curated generative-recommendation paper list) were read directly. Facts about Airbnb, Facebook, Etsy, Amazon, Pinterest, Uber and Kuaishou come from primary-source excerpts that were retrieved and cross-checked for sibling chapters of this atlas (Chapters 2, 6, 22, 24 and 26). References marked "not re-fetched" are well-known papers cited from their stable arXiv or publisher URLs. Numbers marked approximate should be checked against the original paper before reuse.

---

## 31.1 The common skeleton

Every system in this chapter fits the same skeleton. The differences are what goes into the "query" box and which retrieval paths exist.

```
  explicit query (text, filters)     implicit context (user, session, page, geo)
             \                                 /
              v                               v
   +-------------------- QUERY UNDERSTANDING --------------------+
   | normalize, spell-correct, segment, classify (category,      |
   | intent, locale), extract attributes, rewrite / expand,      |
   | encode -> dense vector and/or sparse terms                  |
   +-------------------------------------------------------------+
              |                 |                    |
              v                 v                    v
   lexical / boolean     dense EBR (ANN)      behavior / graph sources
   (inverted index,      (two-tower, IVF,     (co-click, item-to-item,
    BM25, filters)        HNSW, SPANN)         history, trending)
              \                 |                    /
               v                v                   v
   +--------- MERGE: union / RRF / quotas / dedup (10^3 - 10^4) -----+
                                |
                                v
   first-pass ranker (GBDT/LambdaMART or small DNN, in-index or near data)
                                |
                                v
   second-pass ranker (deep multi-task model, personalization, cross-features)
                                |
                                v
   re-rank: diversity, business rules, ads blending -> page
                                |
                                v
   logs: impressions, clicks, add-to-cart, bookings -> labels, propensities
```

Two observations hold across all the case studies.

- **Search is precision-constrained; recommendation is not.** A query such as "iphone 15 case" defines a relevance floor that personalization must not break. A home feed has no such floor. This is why search systems keep lexical retrieval and hard filters even after EBR works, while feed systems can be purely embedding- or graph-based [8][9][11].
- **Recommendation is search with an empty or implicit query.** Once the query is just another feature (text embedding, category, attributes), the same retrieval and ranking stack can serve both. The unified models in Section 31.7 make this explicit.

---

## 31.2 Learning-to-rank: RankNet, LambdaRank, LambdaMART

### 31.2.1 RankNet

Burges' report "From RankNet to LambdaRank to LambdaMART: An Overview" (Microsoft Research, MSR-TR-2010-82) is the canonical self-contained description [1]. RankNet works on pairs of documents for the same query. The model (any differentiable scorer; Burges notes neural nets and also boosted trees) produces scores s_i and s_j. The probability that document i should rank above j is a sigmoid of the score difference:

- P_ij = 1 / (1 + exp(-sigma * (s_i - s_j)))

The cost is cross-entropy against the target probability. With S_ij in {-1, 0, +1} for the label order, this becomes

- C = 0.5 * (1 - S_ij) * sigma * (s_i - s_j) + log(1 + exp(-sigma * (s_i - s_j)))

Two properties matter in practice. Equal scores for differently labelled documents still cost log 2, so the loss contains a margin. The cost is linear for badly ordered pairs and goes to zero for correct ones [1].

**Factorization trick.** The gradient for each pair can be written as a scalar "lambda" times the difference of score gradients. Summing the lambdas per document before back-propagating turns the training cost from roughly quadratic in pairs to close to linear in documents. Burges notes this speed-up "results from the way the problem factorizes, not from using mini-batch alone" and that it laid the groundwork for LambdaRank [1].

### 31.2.2 LambdaRank

IR metrics such as NDCG and ERR depend only on sorted order, so they are flat or discontinuous as functions of the scores. LambdaRank sidesteps this: instead of defining a smooth cost, it **defines the gradient directly**. The RankNet lambda for each pair is multiplied by |delta NDCG|, the change in NDCG that swapping the two documents would cause. Burges reports that models trained this way empirically optimize NDCG directly, and that any other IR metric can be targeted by swapping delta NDCG for the change in that metric [1].

### 31.2.3 LambdaMART

MART (gradient-boosted regression trees) fits each new tree to the gradients of the current ensemble. LambdaRank specifies exactly those gradients. LambdaMART is "the marriage of the two": each boosting round fits a regression tree to the lambdas, and leaf values are set with a Newton step that uses second derivatives [1]. An ensemble of LambdaMART rankers won Track 1 of the 2010 Yahoo! Learning to Rank Challenge [1].

```
  per query q, documents d1..dn with graded labels l_i
       |
       v
  current ensemble F_m(x) --> scores s_i
       |
       v
  for each pair (i, j) with l_i > l_j:
        rho_ij   = 1 / (1 + exp(sigma * (s_i - s_j)))
        lambda_ij = sigma * |delta NDCG(i <-> j)| * rho_ij
        lambda_i += lambda_ij ;  lambda_j -= lambda_ij
       |
       v
  fit regression tree m+1 to {(x_i, lambda_i)}
  leaf value = sum(lambda) / sum(second derivative)   (Newton step)
       |
       v
  F_{m+1} = F_m + eta * tree_{m+1}
```

### 31.2.4 LTR in today's libraries

LightGBM exposes the same ideas as configuration [2]:

- `objective=lambdarank` with a default `label_gain` of 0, 1, 3, 7, 15, ... (2^label - 1), so grades map to exponential NDCG gains.
- `lambdarank_truncation_level` (default 30) limits the pairs considered to the top of the list. The documentation advises setting it slightly above the target NDCG@k cutoff (for example k + 3).
- `lambdarank_norm` (default true) normalizes lambdas per query, which helps when queries have very different numbers of results.
- `rank_xendcg` (XE-NDCG, Bruch 2019) is documented as faster than lambdarank with similar quality [2][43].

TF-Ranking (Google, archived in 2025) brought the same family to neural networks: pointwise, pairwise and listwise losses, LambdaLoss for direct metric optimization, groupwise scoring functions, and unbiased LTR from biased feedback [3]. In search engines, the OpenSource Connections Elasticsearch LTR plugin stores features as query templates, logs feature values to build training sets, and stores RankLib, XGBoost or linear models that re-rank the top results inside the engine [6].

### 31.2.5 From graded judgments to clicks: unbiased LTR

LTR was born on human relevance judgments (graded 0-4). Production search mostly trains on clicks, which are biased by position. Google's "Learning to Rank with Selection Bias in Personal Search" (Wang et al., SIGIR 2016) addressed this for email and personal search, where editorial judgments are impossible for privacy reasons [4]. Joachims et al. (WSDM 2017) formalized **inverse propensity scoring** for LTR: weight each click by 1 / P(examined at position k), with propensities estimated from randomized interventions [5] (not re-fetched).

Airbnb took a simpler route in its deep ranker: **position as an input feature with dropout**, set to a constant at inference time. Too little dropout lets the model lean on position. Too much stops it from separating position from quality. The tuned version gave a significant booking gain [14].

### 31.2.6 Airbnb: a LambdaRank-style DNN in production

Airbnb's "Applying Deep Learning to Airbnb Search" is the best public log of moving a LambdaMART-era ranker to a neural network [13]:

- A GBDT ranker with heavy hand-made features had plateaued.
- A simple network with one hidden layer of 32 ReLU units reached about parity.
- Switching to a **pairwise LambdaRank-style loss** (booked versus not-booked listings in the same search, weighted by the NDCG change of a swap) helped.
- The final model had 2 hidden layers, about 195 features and was trained on the order of 1.7 billion pairs (approximate).
- Listing-ID embeddings overfit (a listing can be booked at most about 365 times per year), and a multi-task "long view" head raised views but not bookings.
- Offline NDCG wins that did not reproduce online taught the team to distrust offline gains [13].

**Lesson.** The loss family (pairwise, metric-weighted) survived the move from trees to networks. What changed was feature engineering effort and the amount of data the model could use.

---

## 31.3 Query understanding and semantic matching

### 31.3.1 What query understanding does

Daniel Tunkelang's "Query Understanding" series frames it as everything that happens to the query before retrieval: language identification, tokenization, spelling correction, segmentation, entity and attribute recognition, query classification (category, intent), relaxation, expansion and rewriting [42] (not re-fetched). In commerce search, the most valuable outputs are usually:

- **Category or product-type prediction**, used both as a retrieval filter and as a ranking feature.
- **Attribute extraction** (brand, color, size, model number), turned into structured filters.
- **Rewrites and relaxations** for null or low-recall queries.
- **A dense query embedding**, which is the input to EBR.

### 31.3.2 DSSM: the first two-tower search model

"Learning Deep Structured Semantic Models for Web Search using Clickthrough Data" (Huang, He, Gao, Deng, Acero and Heck, Microsoft, CIKM 2013) maps queries and documents into a shared low-dimensional space and measures relevance as the distance between them. It is trained to maximize the likelihood of clicked documents given a query, and uses **word hashing** to scale to web-size vocabularies (letter-trigram hashing in the paper). The best model beat earlier latent semantic models on a real web ranking dataset [7]. DSSM is the direct ancestor of every two-tower model in this atlas.

### 31.3.3 Transformer query encoders

Google announced in 2019 that BERT was being used in Search ranking and featured snippets, helping with roughly 1 in 10 US English queries, especially longer, conversational queries where prepositions change the meaning [41] (not re-fetched). BERT-class cross-encoders are too slow to score thousands of candidates, so production systems use them in three ways: as **teachers** distilled into two-tower or small rankers, as **late-stage re-rankers** over tens of documents, or as **offline labelers** for relevance data.

Two research lines try to keep transformer quality at retrieval speed:

- **Late interaction (ColBERT).** Each passage is encoded into a matrix of token embeddings. At query time, each query token takes the MaxSim over passage tokens, and the sums give the score. ColBERTv2 adds residual compression (2-bit examples in the repository), and the PLAID engine adds centroid-based pruning with knobs such as `ncells`, `centroid_score_threshold` and `ndocs` [26].
- **Learned sparse retrieval (SPLADE).** The BERT MLM head produces sparse expansions of queries and documents, with a sparsity regularizer. Because the output is sparse, it runs on an ordinary inverted index (Anserini, PISA). Later work reports latency close to BM25, and the authors report better out-of-domain generalization on BEIR [27].

**Design implication.** Learned sparse models let a team get "semantic" recall without a separate vector stack, at the cost of a bigger posting-list index. Late interaction gives the best quality per query, at the cost of much larger storage.

---

## 31.4 Embedding-based retrieval in production search

### 31.4.1 Facebook Search: EBR inside the inverted index (KDD 2020)

**Problem.** Facebook Search ran on Boolean term matching in Unicorn, its inverted-index engine. Social search also needs the searcher's own context (location, social graph), which text matching cannot express [8].

**Architecture.** A **unified embedding** two-tower model. The query side encodes query text plus searcher context. The document side encodes text plus document context. Documents are quantized with FAISS, and the result is folded back into Unicorn: the coarse IVF cluster ID becomes an index **term** and the PQ-coded residual becomes the posting **payload**. One Unicorn query can then mix Boolean constraints with nearest-neighbor matching [8].

```
  query text + searcher ctx --> query tower --> q (e.g. 128-d)
                                                 |
       Unicorn:  (nn q :radius r)  AND  (term:location ...)  OR  (lexical terms)
                                                 |
  index:  term = IVF centroid id ; payload = PQ(residual)
                                                 |
  hybrid candidate set --> ranking (embedding similarity also a ranking feature)
```

**Training lessons** [8]:

- Impressed-but-not-clicked results are **bad negatives** for retrieval. Using them caused about a 55% absolute recall regression for the people model versus random negatives.
- **Online hard negatives** (the most similar other positives in the batch) gave +8.38% recall for people, +7% for groups and +5.33% for events.
- Unified (text + context) embeddings beat text-only embeddings by +18% recall for events and +16% for groups.

**ANN lessons.** PQ size around d/4 bytes, re-tune nprobe and cluster counts on every major model change, and choose final settings online. In one 128-d test, PCA+PQ16 reached 51.62% 1-Recall@10 versus 74.81% for a flat index [8].

**Lesson.** The cheapest way to add semantic retrieval to an existing search engine is to make vectors look like terms. Filters, sharding and serving then come for free.

### 31.4.2 Amazon Semantic Product Search (KDD 2019)

Nigam et al. built a neural matcher to complement Amazon's lexical product search [9]. Key choices (via the sibling Chapter 24 summary):

- A symmetric two-tower model with a **shared token embedding table** for queries and products.
- Tokens are word unigrams, bigrams and character trigrams. Out-of-vocabulary tokens are **hashed** into extra buckets, which mattered because product queries are full of model numbers, brands and misspellings.
- Simple **average pooling** was competitive with heavier encoders.
- A hinge-style loss separates purchased, impressed-not-purchased and random products.
- Semantic retrieval runs **alongside** lexical matching. Lexical stays precise for exact-attribute queries; the semantic path adds recall for vocabulary-gap queries.

**Evaluation data.** Amazon later released the Shopping Queries (ESCI) dataset. The larger version has 130,652 queries and about 2.62 million judgments in English, Spanish and Japanese. The reduced version has 48,300 queries and 1,118,011 judgments. Each (query, product) pair is labelled Exact, Substitute, Complement or Irrelevant, and the KDD Cup 2022 tasks were query-product ranking, 4-way classification and substitute identification [10]. Graded, typed relevance is the commerce analogue of the 0-4 web judgments that LambdaMART was built for.

### 31.4.3 Etsy Unified Embedding (KDD 2023)

Etsy's long tail of one-of-a-kind items and vague aesthetic queries ("boho wedding decor") defeats pure lexical retrieval. Jha et al. describe a two-tower **Unified Embedding** model [11]:

- **Product tower** fuses transformer text embeddings, term-based (bag-of-tokens) embeddings that keep lexical precision, and graph embeddings from query-product interaction data.
- **Joint query-user tower** encodes the query plus personalization features from recent behavior, so **retrieval itself is personalized**, not just ranking.
- **ANN-based boosting.** Quality signals such as price and seller quality are appended to the vectors as extra dimensions. The inner product then adds a tuned quality boost inside the ANN search. The weights were tuned with black-box optimization.
- Results are unioned with lexical retrieval before ranking.
- Online A/B: **+5.58% search purchase rate** and **+2.63% site-wide conversion** [11].

```
 query + user recent behavior --> query/user tower -----------+
                                                             | cosine
 listing text (transformer) --+                              |
 listing tokens (term emb)  --+--> product tower --> [v ; w * quality] --> ANN
 query-listing graph emb    --+                              |
                                                             v
                                lexical retrieval  --union-->  ranker
```

**Lesson.** Folding quality into the vector is a neat way to stop ANN from returning relevant but low-quality items, which a post-filter cannot fix once they have used up the top-K.

### 31.4.4 Taobao and JD: semantic retrieval with relevance control

Two Chinese e-commerce papers describe the same pattern at very large scale (both not re-fetched; see also Chapter 27):

- **Taobao, "Embedding-based Product Retrieval in Taobao Search" (Li et al., KDD 2021)** [38]. A multi-grained query tower combines several granularities of query tokens with the user's real-time, short-term and long-term behavior. Training uses a softmax over a large candidate set with a temperature. A key point is **relevance control**: because personalized EBR can drift away from the literal query, Boolean term constraints are applied to EBR results so that they still match key query terms. The design is a direct statement of the search-specific "precision floor".
- **JD, "Towards Personalized and Semantic Retrieval: An End-to-End Solution for E-commerce Search via Embedding Learning" (DPSR, Zhang et al., SIGIR 2020)** [39]. A query tower with multiple heads, each producing a different embedding, captures different intents and personalization. The paper covers the full system: offline training, an online indexing and serving layer, and its integration as a retrieval path next to the lexical engine.

### 31.4.5 Airbnb: from embeddings-as-features to EBR

Airbnb used embeddings in search for seven years before using them for retrieval.

1. **2018, embeddings as ranking features** [12]. Skip-gram listing embeddings were trained on about 800 million click sessions (approximate), 32 dimensions. Two tricks: the **booked listing is a global context** for every listing in a booking session, and **negatives are sampled from the same market**. Real-time Kafka streams of clicked, skipped, wishlisted and contacted listings produce "similarity to clicked/skipped listings" features for the then-GBDT ranker.
2. **2020, "ideal listing" two-tower ranker** [14]. A query/guest tower produces an "ideal listing" vector and the score depends on distance to each listing vector. This removed a learned "cheaper is always better" bias.
3. **2023, Journey Ranker** [15]. Multi-task heads along the booking funnel (click, listing page, payment page, request, booking), a twiddler for negative outcomes (cancellations, host rejections), context-dependent weights and monotonic constraints.
4. **2022-2023, learning to rank diversely** [16]. A listing's score depends on the listings ranked above it, applied greedily.
5. **2025-2026, EBR** [17][18]. A two-tower model whose listing tower runs in a daily batch and whose query tower runs per search. It is trained with **session-based hard negatives** (seen but not booked). HNSW gave slightly better recall, but Airbnb chose **IVF**, because frequent price and availability updates made HNSW's memory overhead painful. With IVF, a listing's cluster becomes an ordinary filter in the existing search system. Significant booking gains are reported [18].

```
 guest query (place, dates, guests, filters) + guest context
        |
        v
 retrieval: geo + availability filters  AND  IVF cluster ids of query vector [17]
        |  (order of thousands of listings, approximate)
        v
 base ranker: two-tower / multi-task journey DNN, pairwise loss [14][15]
        |
        v
 second pass: diversity given listings above [16], twiddlers [15]
        |
        v
 interleaving for fast screening (~50x sensitivity vs A/B, approximate) [19]
```

**Lesson.** Facebook and Airbnb both reached the same conclusion: in search, the ANN index should fit into the filter engine. For Facebook that meant cluster IDs as terms; for Airbnb, IVF clusters as filters.

### 31.4.6 Pinterest OmniSearchSage (WWW 2024)

Pinterest search must match a text query to Pins, products and other entities. OmniSearchSage learns a **single query embedding** that is trained jointly against several entity types (Pins and products) and several tasks, in the same space as Pinterest's existing entity embeddings [20]. Item-side text is enriched with signals such as board titles, engaged queries and generated captions (reported in the abstract; details not re-fetched). Pinterest reports gains in offline relevance and online search engagement; check [20] for the exact figures.

The serving side runs on Manas, Pinterest's in-house ANN platform [21]. Its 2026 evolution post benchmarks a 100-million-embedding set: HNSW 121 GB at 93.72% Recall@100, HNSW with scalar quantization 50 GB at 92.92%, IVF 97 GB at 91.69%, and IVF with scalar quantization 25 GB at 95.71%. PQ compressed more but lost much more recall. Manas also implements SPANN for SSD-resident indexes, with an in-memory HNSW over centroids [21]. Manas' streaming filters reuse inverted-index posting lists (bitmap and array-backed) as filter stores, so lexical-style constraints apply during ANN search.

**Lesson.** One query embedding that lives in the same space as the item embeddings used by recommendation turns search and related-item retrieval into the same ANN problem over the same index.

---

## 31.5 Hybrid lexical plus vector retrieval

### 31.5.1 Why hybrid wins

All the production systems above kept lexical retrieval [8][9][11][17][38]. The reasons are consistent:

- **Exact-match queries** (SKUs, model numbers, names, rare tokens) are where dense models are weakest and lexical is near perfect.
- **Hard filters** (availability, price, location, category, permissions) are native to inverted indexes.
- **Explainability and debuggability.** A lexical match shows why a result was returned.
- **Cold start.** A new item is lexically searchable the moment it is indexed, before any embedding refresh.

Dense retrieval adds recall for vocabulary-gap queries, synonyms, multilingual matches, and personalization through user features in the query tower.

### 31.5.2 Five integration patterns

| Pattern | How it works | Example | Strengths | Weaknesses |
|---|---|---|---|---|
| Parallel union | Run lexical and ANN separately, union, rank | Etsy [11], Amazon [9] | Simple; independent scaling | Duplicate infra; ranker must calibrate both |
| Vectors as terms | IVF cluster ID as an index term, PQ residual as payload | Facebook Unicorn [8] | One engine, one query language, filters free | Recall limited by IVF and PQ settings |
| Clusters as filters | IVF assignment stored as a filterable attribute | Airbnb [17] | Handles frequent updates; fits existing filters | Coarse; more candidates to re-score |
| In-engine hybrid ranking | BM25 and closeness both computed per hit; fused in rank phases | Vespa [23][24], Elasticsearch LTR [6] | Score fusion and LTR near data | Engine lock-in; tuning per dataset |
| Learned sparse | One sparse model gives lexical + expansion | SPLADE [27] | One inverted index | Bigger postings; less personalizable |

### 31.5.3 Score fusion

When two retrieval paths return ranked lists, they must be combined before (or as input to) the ranker.

- **Reciprocal rank fusion (RRF).** Cormack, Clarke and Buettcher (SIGIR 2009) score each document as the sum over lists of 1 / (k + rank), with k = 60 [25] (not re-fetched). RRF needs no score calibration, which makes it a safe default.
- **Normalized linear combination.** Map each score to [0, 1] (min-max over the candidate set, or a squashing function such as arctan for BM25) and add with weights.

Vespa's hybrid-search tutorial is a useful reality check [23]. On NFCorpus test queries (nDCG@10): BM25 alone 0.3210, dense (a 384-d embedding model) 0.3077, closeness x (1 + BM25) 0.3330, raw sum 0.3244 (BM25's unbounded scale dominates), atan-normalized BM25 plus cosine 0.3410, RRF over the top 100 in a global phase 0.3233, and linear normalization in a global phase 0.3423. The tutorial's conclusion is that "the effectiveness of the hybrid model depends on the dataset", so the fusion method must be tested on your own data [23]. The tutorial also shows a cost trick: use the `rank()` operator to retrieve with one operator and compute the other only as a ranking feature, which avoids building an HNSW index at all [23].

Vespa's phased ranking documents the in-engine cascade that makes this cheap: a first phase on every matched document on content nodes, a second phase on the top hits per node, and a global phase on the merged top hits in the container. Global normalization and RRF need the global phase, because they need the full candidate set across nodes [24].

```
  query --> [text() OR nearestNeighbor()]  on each content node
                 |
           first-phase: cheap expression (e.g. bm25 + closeness) on all hits
                 |
           second-phase: GBDT / small NN on top-N per node
                 |
   container: global-phase over merged top-K: RRF or normalize, cross-encoder
                 |
               results
```

### 31.5.4 Filtered ANN

Hybrid search almost always means ANN under predicates. The options seen in production:

- **Post-filtering** with over-fetch (early Manas): simple, but recall collapses for selective filters [21].
- **Pre-filter through index structure**: cluster IDs as terms or filters (Facebook, Airbnb) [8][17].
- **Filter-aware ANN**: Microsoft's DiskANN library now exposes attribute-filter hooks, range filters and diversity-aware top-k alongside real-time updates (IP-DiskANN, Fresh-DiskANN), and its legacy branch implements Filtered-DiskANN [29]. SPTAG, released by Microsoft Research and Bing, combines balanced k-means or kd trees with a relative neighborhood graph and supports online insertion and deletion; its SPANN and SPFresh papers target billion-scale, SSD-resident and updatable indexes [28].

### 31.5.5 Uber Eats: the hidden cost of hybrid retrieval

Uber's 2026 Eats search post shows what happens when many retrieval paths (lexical, embedding/ANN, geo) all feed one ranker [22]. Retrieval returned tens of thousands of candidates, and all of them were fully hydrated before the first-pass ranker discarded most. Removing low-yield retrieval paths saved about 120 ms. Splitting hydration into a light phase (ranking features only) and a full phase (presentation data for shown results) saved more than 100 ms more. The first-pass ranker must stay fast "while maintaining a high recall metric" [22].

**Lesson.** Each extra retrieval path costs more than its own latency: it inflates hydration and first-pass ranking downstream. Measure each path's marginal contribution to final results and prune.

---

## 31.6 Personalized search

### 31.6.1 Short-term versus long-term signals

Bennett, White, Chu, Dumais, Bailey, Borisyuk and Cui (Microsoft/Bing, SIGIR 2012) studied how within-session and historical behavior each help web search personalization [30]. Their findings still shape feature design:

- Historical (long-term) behavior helps most at the **start** of a session.
- Within-session behavior accounts for most gains in **longer** sessions.
- Combining both beats either alone, and their relative value shifts during the session [30].

Airbnb's 2018 real-time features are a direct industrial version: two-week Kafka streams of clicks and skips produce per-candidate similarity features, while booking-trained "type" embeddings capture long-term taste [12].

### 31.6.2 Where to personalize: retrieval or ranking

| System | Personalization in retrieval | Personalization in ranking | Guardrail for relevance |
|---|---|---|---|
| Facebook Search [8] | Searcher context in query tower | Embedding similarity as feature | Boolean terms in same Unicorn query |
| Etsy [11] | Joint query-user tower | Downstream ranker | Union with lexical; quality boost dims |
| Taobao [38] | Real-time + long-term behavior in query tower | Downstream ranker | Boolean relevance control on EBR output |
| JD DPSR [39] | Multi-head personalized query tower | Downstream ranker | Lexical engine in parallel |
| Airbnb [12][14][17] | Query tower over guest + query (EBR) | Session embedding features, ideal-listing tower | Hard availability and geo filters |
| Bing (study) [30] | n/a | Session + history re-ranking features | Base relevance ranker |

The pattern: **personalize retrieval only with a relevance guardrail**. A purely personalized query tower drifts toward "what this user likes" and away from "what this user asked for", which is acceptable in a feed but is a bug in search.

### 31.6.3 Practical feature families for personalized search rankers

From the case studies above, the features that recur are:

- **Query-item relevance**: BM25 and field matches, semantic similarity (two-tower cosine), category-match probability.
- **User-item affinity**: similarity of the candidate to recently clicked or skipped items [12], to purchased categories, to long-term type embeddings.
- **User-query context**: is this a refinement of the previous query; time since session start [30].
- **Item quality and marketplace signals**: ratings, price normalized within the result set (as in Expedia-style LTR), availability, seller quality [11].
- **Position and presentation**: used at training time only, with dropout or propensity weighting [4][5][14].

---

## 31.7 Unified search and recommendation

There are four levels of unification, in increasing order of ambition.

```
 L1 shared infra      same ANN platform, feature store, ranking service
                      (Manas, Vespa, Unicorn)
 L2 shared space      one embedding space for queries, users and items
                      (OmniSearchSage, Etsy, Facebook unified embedding)
 L3 joint behavior    search history feeds recommendation and rec history feeds
                      search (UniSAR, Bing short/long-term)
 L4 one model         a single generative model serves both tasks
                      (Spotify GR study, Amazon SynerGen, Kuaishou OneSearch)
```

### 31.7.1 L2: shared embedding spaces

Facebook's unified embedding [8], Etsy's joint query-user tower [11] and Pinterest's OmniSearchSage [20] share one idea: the query vector lives in the same space as item vectors that recommendation surfaces also use. Benefits: one index per entity type, transfer of engagement signal from recommendation into search (and vice versa), and cheap "search-like" recommendation modules (for example, related items for the current query).

### 31.7.2 L3: modeling transitions between search and recommendation

UniSAR ("Modeling User Transition Behaviors between Search and Recommendation", Shi et al., SIGIR 2024) models the user's interleaved stream of search and recommendation actions. Its official PyTorch implementation trains on the Amazon dataset and **KuaiSAR**, a Kuaishou dataset that logs both search and recommendation behavior for the same users [31]. The paper's premise is that what a user searched for is a strong signal of recommendation intent, and what they watched from recommendations disambiguates their next query. In industry this shows up as **cross-surface features**: recent queries as features in the feed ranker, recent feed engagement as features in the search ranker. The Bing study shows the same session-dynamics effect inside search alone [30].

### 31.7.3 L4: one generative model for both tasks

Generative retrieval represents each item as a short sequence of discrete **semantic ID** tokens (Chapter 6) and trains a sequence model to emit them. A query, a user history, or both can be the input, so search and recommendation become the same decoding problem.

**Spotify: does one task help the other?** Penha et al. (Spotify, RecSys 2024) trained generative retrieval models jointly on search (query to item) and recommendation (history to item) and asked whether multi-task training helps [32] (not re-fetched). The paper reports that the joint model can outperform task-specific models, and attributes gains to two effects: joint training regularizes item popularity estimates, and it regularizes item latent representations through signals from both tasks. Gains depend on how much the two tasks' item distributions overlap.

**Amazon: SynerGen (2025).** "SynerGen: Contextualized Generative Recommender for Unified Search and Recommendation" is a single generative model that conditions on context and serves both search and recommendation [36] (listed in [37]; details not re-fetched).

**Kuaishou: OneSearch (2025) and OneSearch-V2 (2026).** OneSearch applies Kuaishou's "One" end-to-end generative recipe (Chapter 6's OneRec) to e-commerce search [33]. It replaces the cascade (lexical/semantic recall, pre-rank, rank) with a model that takes the query and user behavior and decodes item semantic IDs. The tokenizer uses **3 residual-quantization layers (4096/1024/512 codes) plus 2 OPQ codes (2 x 256)** to separate near-duplicate products (via the Chapter 6 summary of [33]). The paper also adds keyword-aware item encoding so that tokens respect query terms, multi-view user behavior inputs, and a preference-aware reward for post-training (abstract-level; details not re-fetched). OneSearch-V2 (March 2026) adds latent reasoning and self-distillation [34]. The same team built **OneSug** (2025) for query suggestion, an end-to-end generative model that unifies the multi-stage suggestion cascade [35].

```
  OneSearch-style unified generative search (simplified)

  query text --> keyword-aware encoding --+
  user behavior (search + rec history) --+--> encoder
                                          |
                                          v
                               decoder (beam search over
                               valid semantic-ID prefixes)
                                          |
             SID tokens:  [RQ1 4096][RQ2 1024][RQ3 512][OPQ 256][OPQ 256]
                                          |
                                          v
                    items  --> optional light re-rank --> results
  training: SFT on (query, user) -> purchased/clicked item SIDs,
            then preference / reward alignment
```

### 31.7.4 What unification buys and costs

| Dimension | Separate stacks | Shared space (L2) | Joint behavior (L3) | One generative model (L4) |
|---|---|---|---|---|
| Infra cost | 2x indexes, rankers | 1 index per entity | 2 stacks, shared features | 1 model, 1 serving path |
| Cold-start items | Lexical covers search | Content towers help both | Same as base | SID from content helps |
| Relevance guarantee | Strong (lexical) | Needs lexical union | Strong | Weakest; needs constrained decoding and filters |
| Cross-task transfer | None | Through embeddings | Through features | Through all parameters [32] |
| Debuggability | High | Medium | Medium | Low |
| Public scale evidence | Everywhere | Facebook, Etsy, Pinterest [8][11][20] | Research + features [30][31] | Kuaishou, Amazon, Spotify [32][33][36] |

The main risk of L4 in search is the precision floor. A generative model can emit a fluent but wrong item, and SKU-like exact queries are where semantic IDs are weakest. Production designs therefore keep constrained decoding over valid ID prefixes, filter checks after decoding, and often a lexical fallback path.

---

## 31.8 Reference architecture: a unified discovery stack

Combining the lessons, a modern design that serves search, related items and home feed from one platform looks like this:

```
                       +------------------- shared feature store -------------------+
                       | user long/short-term seqs (search + rec), item stats,      |
                       | query stats, real-time session events                      |
                       +------------------------------------------------------------+
  request(surface, query?, user, ctx)
        |
        v
  query understanding (only if query): spell, classify, attributes, rewrite, encode
        |
        v
  retrieval portfolio (per surface quotas)
   - lexical / learned sparse (search only; also relevance guardrail)
   - unified two-tower ANN (query+user tower; empty query for feed)
   - item-to-item / graph (related items; expansion from top search hits)
   - generative SID decoding (optional, constrained)
        |
        v
  merge: RRF or learned blend; dedup; filter (availability, policy)
        |
        v
  first pass (in-index or near data): GBDT LambdaMART / small DNN
        |
        v
  second pass: multi-task DNN (click, purchase/booking, negative outcomes),
               surface-specific heads, position debiasing
        |
        v
  re-rank: diversity, ads, business rules  --> page
        |
        v
  logs + interleaving + A/B
```

Design rules embedded in this diagram:

1. The **query tower accepts an empty query**. Feed requests and search requests use one model, and the surface is a feature.
2. **Lexical retrieval is a guardrail as well as a source.** Its matches can gate or boost semantic results for search [38].
3. **Per-surface heads** let search optimize relevance-constrained conversion while feed optimizes engagement.
4. **One feature store** guarantees that search history is visible to recommendation and vice versa [30][31].

---

## 31.9 Key takeaways for system designers

1. **LambdaMART is still the baseline to beat.** It is fast, robust and directly targets NDCG. Start with LightGBM `lambdarank`, a sensible `lambdarank_truncation_level` and exponential label gains, then move to neural rankers when features and data justify it [1][2]. Airbnb kept the pairwise, metric-weighted loss when it switched to a DNN [13].
2. **Fix the labels before the model.** Click logs are position-biased. Use propensity weighting or position-as-feature-with-dropout [4][5][14]. Typed relevance labels (ESCI) beat binary clicks for commerce [10].
3. **Keep lexical retrieval.** Every production search system here kept it for exact matches, filters and cold start. Add EBR next to it, not instead of it [8][9][11][17].
4. **Make the ANN index speak the filter engine's language.** Cluster IDs as terms (Facebook) or as filters (Airbnb), or streaming filters over posting lists (Pinterest), beat post-filtering [8][17][21].
5. **Negatives define retrieval quality.** Random plus in-batch hard negatives work; impressed-not-clicked results are poor retrieval negatives (about 55% recall loss at Facebook). Session-based hard negatives work for Airbnb because they are true alternatives within the same intent [8][17][18].
6. **Personalize retrieval only with a relevance guardrail.** Boolean constraints, lexical union, or quality and relevance dimensions in the vector [11][38].
7. **Fusion is dataset-dependent.** RRF is the safe default. Normalized linear fusion was best on Vespa's benchmark, but by small margins. Always measure [23][25].
8. **Count the downstream cost of each retrieval path.** Uber saved 120 ms by removing low-yield paths and over 100 ms more with two-phase hydration [22].
9. **Use session and long-term signals together.** Long-term history matters early in a session; in-session behavior dominates later [30].
10. **Unify in steps.** Shared infrastructure and a shared embedding space are proven at scale. Joint behavior features are cheap. A single generative model for search and recommendation is now in production at Kuaishou and studied at Spotify and Amazon, but it needs constrained decoding and a precision guardrail for search [20][32][33][36].
11. **Use interleaving to iterate on rankers.** Airbnb reports about 50x sensitivity over A/B for search ranking changes (approximate), with A/B as the final gate [19].

---

## References

1. Burges, C. J. C. "From RankNet to LambdaRank to LambdaMART: An Overview." Microsoft Research Technical Report MSR-TR-2010-82, 2010. https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/
2. Microsoft. "LightGBM Parameters" (lambdarank, rank_xendcg, lambdarank_truncation_level, label_gain, lambdarank_norm). LightGBM documentation, accessed 2026. https://github.com/microsoft/LightGBM/blob/master/docs/Parameters.rst
3. Google. "TF-Ranking: Learning-to-Rank in TensorFlow" (repository, archived 2025). https://github.com/tensorflow/ranking
4. Wang, X., Bendersky, M., Metzler, D., Najork, M. "Learning to Rank with Selection Bias in Personal Search." Google, SIGIR 2016. https://research.google/pubs/learning-to-rank-with-selection-bias-in-personal-search/
5. Joachims, T., Swaminathan, A., Schnabel, T. "Unbiased Learning-to-Rank with Biased Feedback." Cornell, WSDM 2017 (not re-fetched). https://arxiv.org/abs/1608.04468
6. OpenSource Connections. "Elasticsearch Learning to Rank" plugin (repository and README). https://github.com/o19s/elasticsearch-learning-to-rank
7. Huang, P.-S., He, X., Gao, J., Deng, L., Acero, A., Heck, L. "Learning Deep Structured Semantic Models for Web Search using Clickthrough Data." Microsoft Research, CIKM 2013. https://www.microsoft.com/en-us/research/publication/learning-deep-structured-semantic-models-for-web-search-using-clickthrough-data/
8. Huang, J.-T., et al. "Embedding-based Retrieval in Facebook Search." Facebook, KDD 2020. https://arxiv.org/abs/2006.11632
9. Nigam, P., Song, Y., Mohan, V., et al. "Semantic Product Search." Amazon, KDD 2019. https://arxiv.org/abs/1907.00937
10. Reddy, C. K., et al. "Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product Search." Amazon, 2022. Dataset repository: https://github.com/amazon-science/esci-data ; paper: https://arxiv.org/abs/2206.06588
11. Jha, R., Subramaniyam, S., Benjamin, E., Taula, T. "Unified Embedding Based Personalized Retrieval in Etsy Search." Etsy, KDD 2023. https://arxiv.org/abs/2306.04833
12. Grbovic, M., Cheng, H. "Real-time Personalization using Embeddings for Search Ranking at Airbnb." Airbnb, KDD 2018. https://dl.acm.org/doi/10.1145/3219819.3219885
13. Haldar, M., Abdool, M., Ramanathan, P., et al. "Applying Deep Learning to Airbnb Search." Airbnb, KDD 2019. https://arxiv.org/abs/1810.09591
14. Haldar, M., Ramanathan, P., Sax, T., et al. "Improving Deep Learning for Airbnb Search." Airbnb, KDD 2020. https://arxiv.org/abs/2002.05515
15. Tan, C. H., et al. "Optimizing Airbnb Search Journey with Multi-task Learning." Airbnb, KDD 2023 (not re-fetched). https://arxiv.org/abs/2305.18431
16. Abdool, M., Haldar, M., et al. "Learning To Rank Diversely At Airbnb." Airbnb, 2022 (CIKM 2023). https://arxiv.org/abs/2210.07774
17. Airbnb Engineering. "Embedding-Based Retrieval for Airbnb Search." Airbnb Tech Blog, 2025. https://airbnb.tech/ai-ml/embedding-based-retrieval-for-airbnb-search/
18. Airbnb. "Applying Embedding-Based Retrieval to Airbnb Search." arXiv, 2026. https://arxiv.org/pdf/2601.06873
19. Airbnb Engineering. "Beyond A/B Test: Speeding up Airbnb Search Ranking Experimentation through Interleaving." Airbnb Tech Blog, 2022. https://medium.com/airbnb-engineering/beyond-a-b-test-speeding-up-airbnb-search-ranking-experimentation-through-interleaving-7087afa09c8e
20. Agarwal, P., et al. "OmniSearchSage: Multi-Task Multi-Entity Embeddings for Pinterest Search." Pinterest, WWW 2024 companion. https://arxiv.org/abs/2404.16260
21. Pinterest Engineering. "Evolving Pinterest's Embedding Retrieval Platform." Pinterest Engineering Blog, 2026. https://medium.com/pinterest-engineering/evolving-pinterests-embedding-retrieval-platform-aede4e831e01 (summary: InfoQ, "From Memory-Hungry HNSW to Quantized SPANN", 2026, https://www.infoq.com/news/2026/09/pinterest-search/ ; streaming filters: https://medium.com/pinterest-engineering/manas-hnsw-streaming-filters-351adf9ac1c4)
22. Uber Engineering. "Uber Eats Search Pipeline." Uber Blog, 2026. https://www.uber.com/us/en/blog/uber-eats-search-pipeline/
23. Vespa.ai. "Hybrid Text Search Tutorial." Vespa documentation (source on GitHub), accessed 2026. https://github.com/vespa-engine/documentation/blob/master/en/learn/tutorials/hybrid-search.md
24. Vespa.ai. "Phased Ranking." Vespa documentation, 2025-2026. https://docs.vespa.ai/en/ranking/phased-ranking.html
25. Cormack, G. V., Clarke, C. L. A., Buettcher, S. "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods." University of Waterloo, SIGIR 2009 (not re-fetched). https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
26. Khattab, O., Zaharia, M., Santhanam, K., et al. "ColBERT / ColBERTv2 / PLAID" (repository; papers arXiv 2004.12832, 2112.01488, 2205.09707). Stanford, 2020-2022. https://github.com/stanford-futuredata/ColBERT
27. Formal, T., Lassance, C., Piwowarski, B., Clinchant, S. "SPLADE: sparse neural search" (repository; papers arXiv 2107.05720, 2109.10086, 2205.04733). NAVER Labs Europe, 2021-2022. https://github.com/naver/splade
28. Microsoft Research and Bing. "SPTAG: A library for fast approximate nearest neighbor search" (repository; SPANN NeurIPS 2021, SPFresh SOSP 2023). https://github.com/microsoft/SPTAG
29. Microsoft. "DiskANN" (repository: DiskANN3, Fresh/IP-DiskANN, Filtered-DiskANN). Accessed 2026. https://github.com/microsoft/DiskANN
30. Bennett, P. N., White, R. W., Chu, W., Dumais, S. T., Bailey, P., Borisyuk, F., Cui, X. "Modeling the Impact of Short- and Long-Term Behavior on Search Personalization." Microsoft, SIGIR 2012. https://www.microsoft.com/en-us/research/publication/modeling-the-impact-of-short-and-long-term-behavior-on-search-personalization/
31. Shi, T., et al. "UniSAR: Modeling User Transition Behaviors between Search and Recommendation." Renmin University of China / Kuaishou data, SIGIR 2024. Official implementation: https://github.com/TengShi-RUC/UniSAR
32. Penha, G., et al. "Bridging Search and Recommendation in Generative Retrieval: Does One Task Help the Other?" Spotify, RecSys 2024 (not re-fetched). https://arxiv.org/abs/2410.16823
33. Kuaishou. "OneSearch: A Preliminary Exploration of the Unified End-to-End Generative Framework for E-commerce Search." 2025. https://arxiv.org/abs/2509.03236
34. Kuaishou. "OneSearch-V2: The Latent Reasoning Enhanced Self-distillation Generative Search Framework." 2026. https://arxiv.org/abs/2603.24422
35. Kuaishou. "OneSug: The Unified End-to-End Generative Framework for E-commerce Query Suggestion." 2025. https://arxiv.org/abs/2506.06913
36. Amazon. "SynerGen: Contextualized Generative Recommender for Unified Search and Recommendation." 2025. https://arxiv.org/abs/2509.21777
37. Huang, Y., et al. "Awesome-Generative-Recommendation" (curated industrial paper list). 2025-2026. https://github.com/uestc-huangyw/Awesome-Generative-Recommendation
38. Li, S., et al. "Embedding-based Product Retrieval in Taobao Search." Alibaba, KDD 2021 (not re-fetched). https://arxiv.org/abs/2106.09297
39. Zhang, H., et al. "Towards Personalized and Semantic Retrieval: An End-to-End Solution for E-commerce Search via Embedding Learning." JD.com, SIGIR 2020 (not re-fetched). https://arxiv.org/abs/2006.02282
40. Yan, E. "System Design for Recommendations and Search." eugeneyan.com, 2021. https://eugeneyan.com/writing/system-design-for-discovery/
41. Nayak, P. "Understanding searches better than ever before." Google, The Keyword blog, 2019 (not re-fetched). https://blog.google/products/search/search-language-understanding-bert/
42. Tunkelang, D. "Query Understanding" (essay series). 2017 onward (not re-fetched). https://queryunderstanding.com/
43. Bruch, S. "An Alternative Cross Entropy Loss for Learning-to-Rank" (XE-NDCG). 2019 (linked from LightGBM docs; not re-fetched). https://arxiv.org/abs/1911.09798
