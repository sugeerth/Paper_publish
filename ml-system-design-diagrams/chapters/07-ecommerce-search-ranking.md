# 07. Design Product Search Ranking for an E-commerce Site (Amazon Search), Including Query Understanding

> Prompt as usually given: "A user types a query into the Amazon (or Walmart, Etsy,
> Flipkart) search box. Design the ML system that returns the most relevant products,
> ranked to maximise purchases." Variants: "design a search ranking system", "improve
> product search recall", "design Airbnb search ranking", "design typeahead / query
> autocomplete" (a sub-problem).

## Company tags

| Company (reported) | Where it was reported | Notes |
|---|---|---|
| Amazon | Community-compiled Amazon MLE interview process [22] | Sample ML system design prompt: "Design a search retrieval and ranking pipeline, including online experiments and position-bias correction" [22]; Stores "search and personalization" team owns product search ranking [22] |
| Airbnb | Khang Pham case list [5]; Chip Huyen exercise 9 [7] | "Display top 10 results when a user searches for rental listings in a certain location" [7] |
| Google (adjacent, not confirmed) | Chip Huyen exercises 5-6 use Google-style "related searches" and Google Images text-to-image search [7] | Web/image-search framing of the same retrieve-then-rank funnel |
| Walmart, Etsy, eBay | Not confirmed as interview prompts; they publish the production systems this question mirrors [14] [15] [19] | Use their papers as "real system" talking points |

**Portals and sources that cover it**

| Portal | What it covers | Link |
|---|---|---|
| alirezadir / AIMLInterviews | Full e-commerce text-search case: QU, retrieval, 2-stage LTR, blender, filter [1] [2] | [1] |
| Educative, Grokking the ML Interview | "Search Ranking" chapter: layered funnel, document selection, LambdaMART, NDCG [3] [4] | [3] |
| Khang Pham (Educative / mlengineer.io) | "Airbnb Search ranking" case study plus Amazon/Airbnb reading list [5] [6] | [5] |
| Chip Huyen, ML Systems Design | Exercises: Airbnb search ranking, related searches, image search, autocomplete [7] | [7] |
| ByteByteGo (Aminian & Xu) | No e-commerce chapter; closest are "Visual Search" and "YouTube Video Search" (text + visual retrieval and fusion) [29] | [29] |
| Hello Interview / Exponent | Generic retrieval-then-rank pattern and framework applied to search (not re-verified in this pass) [26] [27] | [26] |

*Research note: in the review pass only github.com pages were reachable (arxiv.org,
amazon.science, hellointerview.com, igotanoffer.com did not resolve). Verified against
the live page: [1] [4] [5] [6] [7] [20] [22] [31] and the ESCI data repo [10]. Paper URLs
were cross-checked against the applied-ml index [20]. Company tags from community lists
are indicative, not confirmed; three unsupported tag sources were removed.*

## Requirements & scale

| Dimension | Value | Source |
|---|---|---|
| Functional | Text query in, ranked list of products out (SERP); filters/facets; ad auction out of scope (sponsored slots only blended in) | [1] |
| Business goal | Higher CTR and conversion (purchases per search) | [1] |
| Personalization | Optional in v1 ("not required" in [1]); add in v2 via user features / personalized EBR [14] | [1] [14] |
| Users | 50M DAU (alirezadir assumption); Amazon-scale 300M+ users (typical assumption) | [1] |
| Catalog | [1] writes "from 100 B to 100k" (likely a typo); 100M-1B products is a typical assumption | [1] |
| Query volume | 5M queries/day in [1]; Amazon-scale ~10k-50k QPS peak (typical assumption) | [1] |
| Latency | End-to-end p99 < 200 ms; ranker < 50 ms; feature fetch < 10 ms [4] | [4] + assumption |
| Funnel sizes | Catalog -> ~100k (retrieval) -> stage-1 -> ~500 (stage-2 output) [1]; [4] uses 500 as the stage-1 window; top 10-50 shown | [1] [4] |
| Freshness | New products searchable in minutes (index); price/stock real time; models retrained daily-weekly | Typical assumption |
| Languages / locales | Multi-locale (ESCI covers US, ES, JP) | [10] |

## DIAGRAM 1: online serving path

**Diagram 1. Query to SERP: query understanding, hybrid retrieval, filtering, two-stage
learning-to-rank, blending and business rules, with feature store, model store and logging
(synthesised from [1] [3] [8] [13])**

```
 +--------+  "nike runing shoes men"   +-----------------------------------------+
 | Client |--------------------------->| Search API / Gateway                    |
 +--------+                            +--------------------+--------------------+
     ^                                                      | raw query + ctx
     |                                                      v
     |             +--------------------------------------------------------------+
     |             | QUERY UNDERSTANDING (QU)                                     |
     |             | normalize -> spell fix ("runing"->"running") -> tokenize     |
     |             | -> NER/attrs {brand:nike, gender:men} -> product-type/       |
     |             | category classifier -> intent -> expansion/relaxation        |
     |             | -> query embedding (query tower)                             |
     |             +-----+-------------------------+--------------------+---------+
     |                   | rewritten terms         | q-embedding        | category
     |                   v                         v                    v
     |   +----------------------+   +----------------------+  +-----------------+
     |   | Lexical retrieval    |   | Semantic retrieval   |  | Category/browse |
     |   | inverted index, BM25 |   | ANN (HNSW / PQ) over |  | node + top      |
     |   | + attribute match    |   | product embeddings   |  | sellers         |
     |   +----------+-----------+   +----------+-----------+  +--------+--------+
     |              |  ~10k                    |  ~10k                 | ~1k
     |              +------------+-------------+-----------------------+
     |                           v  union + dedupe (~20-100k)
     |              +---------------------------------------------+
     |              | FILTERING: in-stock, ships-to-region,       |
     |              | adult/policy, compliance, user facets       |
     |              +----------------------+----------------------+
     |                                     v
     |  +---------------+    features   +---------------------------------------+
     |  | Feature store |-------------->| STAGE-1 RANKER (light: GBDT / LR)     |
     |  | online KV:    |               | pointwise, ~20k -> ~500, < 10 ms      |
     |  | product stats |               +-------------------+-------------------+
     |  | query stats   |                                   v
     |  | user history  |-------------->+---------------------------------------+
     |  +---------------+               | STAGE-2 RANKER (heavy: LambdaMART or  |
     |  +---------------+   weights     | multi-task DNN / cross-encoder)       |
     |  | Model store / |-------------->| ~500 -> top 50, < 40 ms               |
     |  | registry      |               +-------------------+-------------------+
     |  +---------------+                                   v
     |              +---------------------------------------------+
     |              | RE-RANK / POLICY: blend sponsored + organic |
     |              | slots, diversity (brand/seller), dedupe     |
     |              | variants, exploration, business boosts      |
     |              +----------------------+----------------------+
     |      SERP (top 48)                  |
     +-------------------------------------+
                                           | impressions, positions, scores, features
                                           v
                             +-------------------------------+
                             | Logging (Kafka): query, SERP, |
                             | clicks, ATC, purchase         |
                             +-------------------------------+
```

**Funnel and latency (typical assumption, numbers from [1] [4] where cited)**

| Stage | In -> out | Model | p99 budget |
|---|---|---|---|
| Query understanding | 1 query -> rewritten query + attrs + embedding | Spell model, NER, classifier, query tower (cached for head queries) | 10-15 ms |
| Retrieval | 100M-100B -> ~100k [1] | BM25 + ANN + category | 20-30 ms |
| Filtering | ~100k -> ~20k | Rules + lookups | 5 ms |
| Stage-1 rank | ~20k -> ~500 (500 per [4]) | LR / small MART, pointwise, AUC [1] | 10 ms |
| Stage-2 rank | ~500 -> 50 | LambdaMART / LambdaRank NN [1] | 30-40 ms |
| Re-rank + blend | 50 -> 48 slots | Rules + diversity | 5 ms |

## DIAGRAM 2: offline training and data pipeline

**Diagram 2. Search logs to labels (graded engagement + human ESCI judgements), feature
joins with point-in-time correctness, LTR training, offline NDCG gate, registry and
interleaving/A-B rollout (synthesised from [1] [3] [9] [10] [11])**

```
 +-----------------+    +----------------------+    +-------------------------+
 | Search logs     |    | Product catalog      |    | Human relevance         |
 | query, SERP,    |    | title, brand, attrs, |    | judgements (ESCI:       |
 | position, click,|    | category, price,     |    | Exact/Substitute/       |
 | ATC, purchase   |    | images, reviews      |    | Complement/Irrelevant)  |
 +--------+--------+    +----------+-----------+    +------------+------------+
          | sessionize             | daily snapshot              | sampled queries
          v                        |                             |
 +---------------------+           |                             |
 | LABEL BUILDER       |           |                             |
 | graded label per    |           |                             |
 | (query, product):   |           |                             |
 | 0 imp, 1 click,     |           |                             |
 | 2 dwell>T, 3 ATC,   |           |                             |
 | 4 purchase          |           |                             |
 | + position-bias     |           |                             |
 |   weights (IPW)     |           |                             |
 +----------+----------+           |                             |
            v                      v                             v
 +-------------------------------------------------------------------------+
 | POINT-IN-TIME FEATURE JOIN (features as of the query timestamp)         |
 | query feats | product feats | query-product feats | user/context feats  |
 +------------------------------------+------------------------------------+
                                      v
 +----------------------+   +-------------------------+   +-------------------+
 | Sampling: group by   |-->| Temporal split: 2 wks   |-->| TRAIN             |
 | query, keep top-k +  |   | train / 1 wk val+test   |   | - retrieval: two- |
 | random negs (page    |   | (seasonality) [1]       |   |   tower, contrast.|
 | 10+) [1]             |   +-------------------------+   | - stage-1: GBDT   |
 +----------------------+                                 | - stage-2: Lambda |
                                                          |   MART / DNN      |
                                                          +---------+---------+
                                                                    v
 +-------------------------+    fail   +------------------------------------+
 | Back to features/labels |<----------| OFFLINE EVAL: NDCG@10, MRR, recall |
 +-------------------------+           | @k (retrieval), ESCI-judged NDCG,  |
                                       | slices: head/torso/tail, locale    |
                                       +------------------+-----------------+
                                                          | pass
                                                          v
 +------------------+   +-------------------+   +------------------------------+
 | Model registry   |-->| Shadow + inter-   |-->| A/B test: conversion/search, |
 | (versioned, with |   | leaving (fast     |   | revenue/search, zero-result  |
 | feature schema)  |   | ranking compare)  |   | rate, latency guardrails     |
 +------------------+   +-------------------+   +--------------+---------------+
                                                               | ramp 1%->50%->100%
                                                               v
       +---------------------------------------------------------------+
       | Production -> new logs (only shown items get labels:          |
       | feedback loop; add exploration/randomized slots) -------------+--> Search logs
       +---------------------------------------------------------------+
 +---------------------------------------------------------------------+
 | Product embedding job: re-embed catalog nightly + on new listing,   |
 | rebuild ANN index (HNSW/PQ), hot-swap behind the retrieval service  |
 +---------------------------------------------------------------------+
```

## DIAGRAM 3: model architecture

**Diagram 3a. Retrieval: Amazon "Semantic Product Search" style two-tower model [8]
(shared token embeddings over word unigrams, bigrams and character trigrams; average
pooling; trained on purchase / impression / random negatives with a 3-part hinge loss)**

```
        QUERY TOWER                                   PRODUCT TOWER
 "nike running shoes men"                 title + brand + attributes + category
            |                                              |
            v                                              v
 +-----------------------+                     +-----------------------+
 | tokenize: unigrams,   |                     | tokenize: unigrams,   |
 | bigrams, char-3grams; |                     | bigrams, char-3grams; |
 | OOV -> hashed bins    |                     | OOV -> hashed bins    |
 +-----------+-----------+                     +-----------+-----------+
             v                                             v
 +-----------------------+    shared weights   +-----------------------+
 | token embedding table |<------------------->| token embedding table |
 +-----------+-----------+                     +-----------+-----------+
             v  average pool                               v  average pool
 +-----------------------+                     +-----------------------+
 | dense layer(s) + norm |                     | dense layer(s) + norm |
 +-----------+-----------+                     +-----------+-----------+
             | q (d=256)                                   | p (d=256)
             +-------------------+   +---------------------+
                                 v   v
                          +-----------------+
                          | cosine(q, p)    |
                          +--------+--------+
                                   v
       +-------------------------------------------------------------+
       | 3-part hinge loss [8]:                                      |
       |  purchased      -> push score above high threshold          |
       |  impressed, not purchased -> push into middle band          |
       |  random negative -> push below low threshold                |
       +-------------------------------------------------------------+
  Serving: p precomputed for all products -> ANN index; q computed per query
  (cached for head queries). Upgrade path: BERT-style towers, hard negatives,
  personalized user tower (Etsy [14], Facebook EBR [13]).
```

**Diagram 3b. Ranking: multi-task learning-to-rank DNN with LambdaRank-style listwise
objective (synthesised from [1] [11] [17]; Airbnb's GBDT -> NN -> deep NN path [11])**

```
 +-------------+ +--------------+ +----------------+ +-------------+ +----------+
 | Query feats | | Product feats| | Query-product  | | User / ctx  | | Retrieval|
 | category,   | | price, rating| | BM25, attr     | | history emb,| | scores:  |
 | intent, len,| | #reviews,    | | match, q-cat   | | device,     | | cosine,  |
 | head/tail,  | | sales rank,  | | CTR, CTR/CVR of| | locale,     | | BM25     |
 | q-embedding | | stock, ship  | | (q,p) pair     | | time        | |          |
 +------+------+ +------+-------+ +-------+--------+ +------+------+ +----+-----+
        |               |                 |                 |             |
        +---------------+---------+-------+-----------------+-------------+
                                  v
           +----------------------------------------------+
           | Shared bottom: concat -> cross layers (DCN)  |
           | -> MLP 512-256-128                           |
           | inputs: sparse ids -> embeddings; dense ->   |
           | log/normalize (NN needs it [11])             |
           +-------+-----------------+---------------+----+
                   |                 |               |
              +----v----------+  +---v----------+  +-v-------------+
              | head: P(click)|  | head: P(ATC) |  | head: P(buy)  |
              +-------+-------+  +------+-------+  +-------+-------+
                      +-----------------+------------------+
                                        v
               score = w1*P(click) + w2*P(ATC) + w3*P(buy)   (or P(buy) only)
                                        v
          +----------------------------------------------------------------+
          | Loss per query group: LambdaRank (NDCG-weighted pairwise) on   |
          | graded labels 0-4 [1] [17] + BCE per head; position fed as a   |
          | feature at train time, fixed to slot 1 at serving (bias fix)   |
          +----------------------------------------------------------------+
```

**Diagram 3c. Query understanding stack (synthesised from [1] [10] [18] [23]; rewrite step learned from reformulation logs in [23])**

```
 raw query --> +-----------+   +-----------+   +---------------+   +--------------+
               | normalize |-->| spell     |-->| segmentation  |-->| NER / slot   |
               | lowercase,|   | correct   |   | + tokenize    |   | tagging:     |
               | unicode,  |   | (noisy-   |   | ("iphone15"   |   | brand, color,|
               | units     |   | channel / |   |  -> iphone 15)|   | size, gender |
               +-----------+   | seq2seq)  |   +---------------+   +------+-------+
                               +-----------+                              |
       +------------------------------------------------------------------+
       v
 +---------------+   +----------------+   +------------------+   +--------------+
 | product-type /|-->| intent: navig. |-->| rewrite: synonym |-->| outputs:     |
 | category clf  |   | (brand page) / |   | expansion,       |   | terms, attr  |
 | (multi-label, |   | specific /     |   | relaxation (drop |   | filters, cat |
 | from clicks)  |   | broad / ESCI   |   | terms on 0 hits) |   | priors, q-emb|
 +---------------+   +----------------+   +------------------+   +--------------+
```

## DIAGRAM 4: the real systems this mirrors

**Diagram 4. Production references: Amazon (GBDT LTR + semantic retrieval), Airbnb (NN
LTR evolution), Facebook/Etsy/Walmart (embedding-based retrieval) [8] [9] [11] [13] [14] [15]**

```
 AMAZON SEARCH (Sorokina & Cantu-Paz 2016 [9]; Nigam et al. 2019 [8])
 +-------------+   +-----------------------+   +----------------------------------+
 | query       |-->| lexical match + Sem.  |-->| GBDT LTR models per category /   |
 |             |   | Product Search ANN [8]|   | locale, trained on purchases and |
 +-------------+   +-----------------------+   | behavioral features [9]          |
                                               +----------------------------------+
 AIRBNB SEARCH (Haldar et al. 2019 [11]; diversity [12])
 +-------------+   +-----------------------+   +----------------------------------+
 | GBDT        |-->| simple NN (1 hidden,  |-->| deep NN, LambdaRank pairwise on  |
 | baseline    |   | 32 units) -> LambdaRk |   | bookings, NDCG of booked listing |
 +-------------+   +-----------------------+   +----------------------------------+
 EMBEDDING-BASED RETRIEVAL (Facebook [13], Etsy [14], Walmart [15])
 +-------------+   +-----------------------+   +----------------------------------+
 | unified     |-->| hard-negative mining, |-->| hybrid: ANN + inverted index,    |
 | query/user  |   | HNSW + PQ quantization|   | then existing ranker             |
 | tower       |   | [14]                  |   |                                  |
 +-------------+   +-----------------------+   +----------------------------------+
```

## Data & labels

| Source | Label / use | Caveats |
|---|---|---|
| Search logs (implicit) | Graded: impression 0, click 1, dwell > T 2, add-to-cart 3, purchase 4 [1] | Position bias, presentation bias, only shown items labelled |
| Purchases | Strongest positive; Amazon trains on purchases [8] [9] | Sparse, delayed; some purchases look irrelevant to the query [28] |
| Impressed-not-purchased | Hard-ish negatives / middle band [8] | Not truly irrelevant |
| Random products | Easy negatives (or page 10+ results) [1] [8] | Too easy alone; add hard negatives [13] |
| Human judgements | ESCI 4-class (Exact, Substitute, Complement, Irrelevant): 130k queries, 2.6M judgements, 3 locales [10] | Costly; [1] estimates 100k queries x 10 results x 10 raters = 10M and rejects it as not scalable |
| Catalog | Product tower inputs, attribute tags for QU | Noisy seller-entered text |
| Window | >= 1 week to cover weekly cycle; temporal split [1] | Holiday seasonality |

## Features

| Group | Examples | Freshness |
|---|---|---|
| Query | Length, category distribution, head/torso/tail bucket, intent, historical CTR, embedding [1] [4] | Daily (head cached) |
| Product | Price, rating, #reviews, sales rank, stock, delivery speed, title/image embeddings, age | Price/stock real time; rest daily |
| Query-product | BM25 / TF-IDF unigram+bigram, attribute match (brand, color, size), query-category match, historical (q,p) CTR/CVR, embedding cosine [1] [4] | Hourly-daily |
| User | Past purchases/views in category, brand affinity, price sensitivity, Prime | Near real time |
| Context | Device, locale, time, previous queries in session [1] | Request time |
| Position (train only) | Displayed rank | Fixed at serving |

## Model choices (baseline -> advanced)

| Stage | Baseline | Advanced | Tradeoff |
|---|---|---|---|
| QU | Rules + dictionaries + edit-distance spell | Neural query rewriting from reformulation logs (Amazon QUEEN [23]), BERT NER, LLM rewrites (offline for head queries) | Quality vs latency; cache head |
| Retrieval | Inverted index + TF-IDF/BM25, weighted linear relevance (term match 0.5, popularity 0.125, intent 0.0625) [1]; same signals unweighted in [4] | Two-tower EBR [8], personalized EBR [13] [14], hybrid with lexical [15] | Recall on tail vs precision on exact attrs |
| Stage-1 | LR / small GBDT pointwise, AUC [1] [4] | Distilled small NN | Microseconds per item [1] |
| Stage-2 | LambdaMART (GBDT) [1] [17] [31] | Multi-task DNN with LambdaRank loss [11] [25]; Etsy moved search ranking from GBDT to DNN [19]; cross-encoder (BERT) on top-50 | DNN needs feature normalization and more data [11] |
| Re-rank | Business rules, dedupe | Diversity model [12], explore/exploit | Relevance vs diversity |
| Seasonality | Recency-weighted retrain | Seasonal-relevance model as ranking feature (Amazon [24]) | Extra model to maintain |

## Loss

| Model | Loss | Why |
|---|---|---|
| Two-tower | In-batch softmax / contrastive; 3-part hinge with purchase/impression/random [8] | Learn semantic match at scale |
| Stage-1 | Binary cross-entropy (pointwise) [1] | Cheap, calibrated |
| Stage-2 | LambdaRank / LambdaMART: pairwise with delta-NDCG weights [17] | Optimises ranking metric directly |
| Multi-task heads | Sum of BCE per head (click, ATC, buy) | Dense click signal helps sparse purchase |
| Debiasing | Inverse propensity weighting; unbiased LambdaMART [33] (XGBoost `lambdarank_unbiased` [31]) | Remove position bias |

## Offline metrics

| Metric | Stage | Note |
|---|---|---|
| Recall@k (k=100, 1000) | Retrieval | Did the purchased item enter the funnel? |
| NDCG@10 | Ranking | Graded labels 0-4; chosen because relevance is non-binary [1] [4] |
| MRR | Ranking | Position of first purchased item [1] |
| AUC / log loss | Stage-1, heads | Pointwise quality [1] [4] |
| ESCI-judged NDCG / precision | Whole SERP | Human relevance, catches "buyable but irrelevant" [10] |
| Slices | All | Head vs tail queries, locale, new products, category |

## Online metrics / A-B

| Metric | Type | Note |
|---|---|---|
| Conversion rate (purchases / search) | Primary | Business goal [1] |
| Revenue / GMV per search | Primary | Watch price-mix shift |
| CTR | Secondary | Prone to clickbait [1] |
| Successful session rate (dwell > T or ATC) | Secondary | [1] [4] |
| Zero-result rate, reformulation rate, time-to-first-click | Guardrail | QU quality signals |
| Latency p99, error rate | Guardrail | SLA |
| Interleaving | Fast pre-test | Higher sensitivity for ranking comparisons |

## Serving & latency budget

| Component | Choice | Budget |
|---|---|---|
| QU | Cached results for head queries (~top 1M queries); online models for tail | 10-15 ms |
| ANN | HNSW / PQ-quantized index, sharded by product id [14] | 10-20 ms |
| Inverted index | Sharded Lucene/Elasticsearch-style | 10-20 ms (parallel with ANN) |
| Feature fetch | Online KV (Redis/DynamoDB) [4] | < 10 ms [4] |
| Stage-1 / Stage-2 | CPU GBDT; GPU or CPU-optimised DNN, batch per request | 10 ms / 30-40 ms |
| Total | p99 | < 200 ms (assumption) |

## Monitoring

| What | Signal | Action |
|---|---|---|
| Input drift | Query distribution, new tokens, category mix (holidays) | Refresh QU dictionaries, retrain |
| Feature health | Null rates, staleness of price/stock | Alert, fallback defaults |
| Quality | Zero-result rate, NDCG on daily human-judged sample [10] | Rollback |
| Train-serve skew | Logged serving features vs offline recomputed | Log features at serve time |
| Index | ANN recall vs brute force on sample, index age | Rebuild |
| Business | Conversion per search by slice | Alert on drops |

## Failure modes

| Failure | Cause | Mitigation |
|---|---|---|
| Accessories outrank main product ("iphone" -> cases) | Click/purchase popularity beats relevance | ESCI relevance model as feature/gate [10]; product-type match |
| Inappropriate items on SERP | Offensive/adult products match query | P(inappropriate) filter model (LR/MART/DNN) before the blender [1] |
| Zero results on tail queries | Lexical mismatch, typos | Spell correct, query relaxation, semantic retrieval [8] [15] |
| Attribute violations ("red dress" shows blue) | Semantic retrieval ignores exact attrs | Hybrid lexical + attribute filters [15] [16] |
| Rich-get-richer | Feedback loop, position bias | IPW, exploration slots, position feature [33] |
| Cold-start products | No engagement stats | Content features, embedding retrieval, boost new items |
| Seasonality | Holiday query shift ("costume" in October) | Short retrain cycle, seasonal-relevance feature [24] |
| Spam / keyword stuffing | Sellers gaming titles | Spam classifier, quality features |
| Latency blow-up on broad queries | Huge candidate sets | Cap per-source candidates, early termination |

## Interviewer follow-ups

- How do you build relevance labels? Clicks vs purchases vs human ESCI judgements, and how do you combine them? [1] [10]
- How do you correct position bias in click logs (IPW, position-as-feature, randomized swaps)? [22] [33]
- Walk through query understanding for "apple watch band 41mm black": which models, in what order?
- Why two-stage ranking instead of one heavy model? Where does the cutoff (500) come from? [1] [4]
- Lexical vs semantic retrieval: when does each fail, and how do you merge them? [8] [15]
- How do you handle tail queries and zero-result searches?
- Pointwise vs pairwise vs listwise loss: why LambdaMART for stage 2? [17]
- Why can NDCG go up offline while conversion goes down online?
- How do you rank a brand-new product with no history?
- How would you add personalization without hurting relevance? [13] [14]
- How do you blend sponsored products with organic results?
- How do you design the A/B test; why might interleaving be used first?
- Real-time features for product search (price, stock): architecture and skew?
- How would you learn query rewrites from reformulation sessions? [23]
- How do you handle seasonal queries whose intent shifts over the year? [24]
- How would you blend text-search results with visual search in one SERP? [1]
- How would you use an LLM here (query rewriting, relevance labelling, cross-encoder distillation) within latency?
- How do you keep the ANN index fresh as millions of products change daily?

## Portal comparison

| Portal | Framing | Retrieval | Ranking | Metrics | Distinctive |
|---|---|---|---|---|---|
| alirezadir [1] | E-commerce text search, 50M DAU, CTR + conversion goal | Inverted index + expansion; weighted linear score | Stage-1 pointwise LR/MART -> stage-2 LambdaMART | NDCG offline; conversion, CTR, success session online | Graded 0-4 labels, blender with visual search, P(inappropriate) filter |
| Educative Grokking [3] [4] | General web search (scope, scale, personalization) | Document selection over inverted index | Layered funnel; stage-1 AUC, stage-2 LambdaMART | NDCG; CTR, successful session rate | Feature taxonomy; < 10 ms feature store (Redis/DynamoDB) [4] |
| Khang Pham [5] [6] | Airbnb search ranking (booking likelihood) | Location-based candidates | Binary classifier -> NN | Conversion; NDCG | Paid case study; free reading list of Amazon/Airbnb search papers [6] |
| Chip Huyen [7] | Open-ended exercise (Airbnb top-10, related searches) | Left to candidate | Left to candidate | Left to candidate | Emphasis on tradeoffs, compute cost |
| ByteByteGo [29] | Visual / video search (no e-commerce chapter) | Text + visual embeddings, ANN | Fusion and re-rank | MRR, recall@k | Multimodal fusion pattern reusable for product images |
| Hello Interview / Exponent [26] [27] | Generic retrieval -> rank pattern | Two-tower / ANN | Heavy ranker | Online/offline split | Framework, delivery timing |

## References

1. A. Dirafzoon, "Search System" (mlsd-search.md), alirezadir/Machine-Learning-Interviews, GitHub, 2023. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/mlsd-search.md
2. alirezadir, "ML System Design" template (lists text query search, typeahead), GitHub, 2025. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/ml-system-design.md
3. Educative, "Grokking the Machine Learning Interview" (Search Ranking module; "Document selection" lesson), educative.io, n.d. https://www.educative.io/courses/grokking-the-machine-learning-interview/document-selection
4. janofsun, "Search Ranking" (study notes on Educative Grokking ML system design), GitHub, 2023. https://github.com/janofsun/janofsun.github.io/blob/master/source/_posts/Search%20Ranking.md
5. K. Pham, "machine-learning-interview" (Airbnb Search ranking case; offers from Google, LinkedIn, Coupang, Snap, Stitch Fix), GitHub, 2021. https://github.com/khangich/machine-learning-interview
6. K. Pham, "appliedml.md" (Amazon and Airbnb search reading list), GitHub, 2021. https://github.com/khangich/machine-learning-interview/blob/master/appliedml.md
7. C. Huyen, "Machine Learning Systems Design - Exercises", GitHub, 2019. https://github.com/chiphuyen/machine-learning-systems-design/blob/master/content/exercises.md
8. P. Nigam, Y. Song, et al., "Semantic Product Search", KDD (Amazon), 2019. https://arxiv.org/abs/1907.00937
9. D. Sorokina, E. Cantu-Paz, "Amazon Search: The Joy of Ranking Products", SIGIR, amazon.science, 2016. https://www.amazon.science/publications/amazon-search-the-joy-of-ranking-products
10. C. K. Reddy et al., "Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product Search", arXiv (Amazon), 2022 (130,652 queries, ~2.6M judgements, US/ES/JP). https://arxiv.org/abs/2206.06588 (data: https://github.com/amazon-science/esci-data)
11. M. Haldar et al., "Applying Deep Learning To Airbnb Search", KDD, arXiv, 2018. https://arxiv.org/abs/1810.09591
12. Airbnb, "Managing Diversity in Airbnb Search", arXiv, 2020. https://arxiv.org/abs/2004.02621
13. J.-T. Huang et al., "Embedding-based Retrieval in Facebook Search", KDD, arXiv, 2020. https://arxiv.org/abs/2006.11632
14. R. Jha, S. Subramaniyam, E. Benjamin, T. Taula, "Unified Embedding Based Personalized Retrieval in Etsy Search", arXiv, 2023. https://arxiv.org/abs/2306.04833
15. A. Magnani et al., "Semantic Retrieval at Walmart", KDD, ACM DL, 2022. https://dl.acm.org/doi/abs/10.1145/3534678.3539164
16. Choi et al., "Semantic Product Search for Matching Structured Product Catalogs", arXiv, 2020. https://arxiv.org/abs/2008.08180
17. C. J. C. Burges, "From RankNet to LambdaRank to LambdaMART: An Overview", Microsoft Research MSR-TR-2010-82, 2010. https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/
18. Uber Engineering, "Food Discovery with Uber Eats: Building a Query Understanding Engine", eng.uber.com, 2018. https://eng.uber.com/uber-eats-query-understanding/
19. Etsy Engineering, "Deep Learning for Search Ranking at Etsy", Code as Craft, 2022. https://www.etsy.com/codeascraft/deep-learning-for-search-ranking-at-etsy
20. E. Yan, "applied-ml" (Search & Ranking section), GitHub, 2020. https://github.com/eugeneyan/applied-ml
21. E. Yan, "System Design for Recommendations and Search", eugeneyan.com, 2021. https://eugeneyan.com/writing/system-design-for-discovery/
22. kevin-2023-code, "Amazon Machine Learning Engineer interview process" (community-compiled; lists the search retrieval and ranking prompt), GitHub, 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/amazon/interview-process-machine-learning-engineer.md
23. Amazon Science, "QUEEN: Neural query rewriting in e-commerce", 2021. https://www.amazon.science/publications/queen-neural-query-rewriting-in-e-commerce
24. Amazon Science, "Seasonal relevance in e-commerce search", 2021. https://www.amazon.science/publications/seasonal-relevance-in-e-commerce-search
25. M. Haldar et al., "Improving Deep Learning for Airbnb Search", arXiv, 2020. https://arxiv.org/abs/2002.05515
26. Hello Interview, "Recommendation Systems (ML System Design pattern)", hellointerview.com, 2025. https://www.hellointerview.com/learn/ml-system-design/patterns/recommendation-systems
27. Exponent, "Introduction to the ML System Design Interview" (course), tryexponent.com, 2025. https://www.tryexponent.com/courses/ml-system-design/mlsd-intro
28. Amazon Science, "Why Do People Buy Seemingly Irrelevant Items in Voice Product Search?", 2020. https://www.amazon.science/publications/why-do-people-buy-irrelevant-items-in-voice-product-search
29. A. Aminian & A. Xu, "Machine Learning System Design Interview" (ByteByteGo; chapters incl. Visual Search, YouTube Video Search), book overview, 2023. https://opencourser.com/book/xop33f/machine-learning-system-design-interview
30. P. Halina, "ML Systems Design Interview Guide", patrickhalina.com, 2020. http://patrickhalina.com/posts/ml-systems-design-interview-guide/
31. XGBoost, "Learning to Rank" tutorial (LambdaMART objectives, position debiasing), GitHub, 2024. https://github.com/dmlc/xgboost/blob/master/doc/tutorials/learning_to_rank.rst
32. V. Ha-Thuc et al., "Learning to Rank Personalized Search Results in Professional Networks" (LinkedIn), arXiv, 2016. https://arxiv.org/abs/1605.04624
33. Z. Hu et al., "Unbiased LambdaMART: An Unbiased Pairwise Learning-to-Rank Algorithm", WWW, ACM DL, 2019. https://dl.acm.org/doi/10.1145/3308558.3313447
