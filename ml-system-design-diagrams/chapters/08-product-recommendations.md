# 08. Design Product Recommendations for Amazon ("Customers Also Bought" + Homepage Recs)

> Prompt as usually given: "Design the recommendation system behind Amazon's product detail
> page ('customers who bought this also bought') and the personalized homepage."
> Variants: "similar items" carousel, "frequently bought together", replacement items for
> out-of-stock products (Instacart, Chip Huyen ex. 3), cart add-on recs, recs in email.
>
> Sourcing note: at review time the web-search budget was exhausted and several hosts
> (amazon.science, arxiv.org, bytebytego.com, igotanoffer.com, hellointerview.com) did not
> resolve. (fetched) = read during review; (ch-ref) = verified in an earlier chapter of this
> atlas; (known) = canonical paper cited by DOI/arXiv ID. Company tags for this exact prompt
> are thinner than for feed/video prompts; treat them as indicative.

## Company tags

| Company | What is reported | Source |
|---|---|---|
| Amazon | MLE loop includes ML design. Product recs is the house problem: item-to-item CF was invented there [1][2][3] | [11] |
| Instacart | "Replacement product recommendation" is in the AIMLInterviews MLSD question list | [8] |
| Generic e-commerce | "Suggest replacement items for out-of-stock products" (Chip Huyen exercise 3) | [10] |
| Meta (Marketplace) | Marketplace recs named as a Meta MLSD prompt | [12] |

**Portals that cover it**

| Portal | Treatment | Link |
|---|---|---|
| ByteByteGo (Aminian & Xu, MLSDI) | No Amazon chapter. The closest is "Similar Listings on Vacation Rental Platforms" (session-based item embeddings), plus "Video Recommendation" for the homepage half | [13][14] |
| Educative "Grokking the ML Interview" | Recommendation-system module (Netflix-style). Candidate gen with CF and embeddings, then a ranker | [15] |
| Educative / Khang Pham "ML System Design" | Case studies plus Wide & Deep and YouTube DNN reading | [9][16] |
| alirezadir AIMLInterviews | Recsys template (CF, two-tower, LTR, metrics) and the Instacart prompt | [8] |
| Hello Interview | Recommendation-systems pattern page (retrieval, ranking, re-ranking) | [17] |
| Exponent / Aced | MLSD course and guide; the recsys framework applies directly | [18][19] |
| Chip Huyen | Exercises 3 and 21 (replacement items, faster checkout) | [10][20] |
| IGotAnOffer, Patrick Halina, DesignGurus | Amazon MLE guide, generic MLSD guides | [11][21][22] |

## Requirements & scale

| Dimension | Value | Source |
|---|---|---|
| Surfaces | (a) Detail page "also bought/viewed" and "frequently bought together" (complements), item-anchored. (b) Homepage, user-anchored. (c) Cart and email | [1][2][26] |
| Business goal | Incremental revenue/GMV and conversion, not just CTR | [2][3] |
| Functional | Top-K (about 10-50) items per widget, in stock, eligible, deduped, diversified | assumption |
| Catalog | Hundreds of millions of SKUs (long tail). Plan for 10^8-10^9 items | assumption |
| Users | About 300M active customers. Most are sparse, many are logged out | assumption |
| QPS | 10^5+ widget requests/s at peak (several widgets per page view, Prime Day spikes) | assumption |
| Latency | Under 100 ms p99 per widget. Model scoring about 20-40 ms | assumption |
| Freshness | Session signals within seconds. Item-item tables daily to hourly. New SKUs within hours | assumption |
| Key constraint | Item-to-item similarity is computed offline; the online step is a lookup, so it scales independently of user count [1] | [1] |

## Diagram 1: End-to-end architecture (online serving path)

**Figure 8.1: Online serving. Item-anchored ("also bought") and user-anchored (homepage)
requests share one candidate-filter-rank-policy pipeline.**

```
 +-------------+  page ctx: user_id, anchor ASIN (detail page), session, device
 |   Client    |---------------------------------------------+
 |  web / app  |<-----------------------------------+        |
 +-------------+   final K items (rendered widget)  |        v
                                              +-----+--------+--------------+
                                              | Recommendation API /        |
                                              | orchestrator (per widget)   |
                                              +--------------+--------------+
                                                             | fan-out, parallel
      +-----------------+------------------+-----------------+   (~25 ms)
      v                 v                  v                 v
 +------------+  +--------------+  +---------------+  +---------------+
 | Item-to-   |  | Two-tower    |  | Session / seq |  | Trending /    |
 | item KV    |  | ANN (HNSW)   |  | model: recent |  | bestsellers   |
 | anchor ->  |  | user emb ->  |  | views -> next |  | per category  |
 | top-N sims |  | top-500      |  | item          |  | (fallback)    |
 +-----+------+  +------+-------+  +-------+-------+  +-------+-------+
       |                |                  |                  |
       +----------------+--------+---------+------------------+
                                 | union + dedupe: ~1-2k candidates
                                 v
 +--------------------------------------------+     +--------------------+
 | Filters: in-stock, ship-to region, owned   |<----| Inventory service, |
 | non-consumables, adult/recall/policy,      |     | order history      |
 | same-variant (size/color) dedupe           |     +--------------------+
 +----------------------+---------------------+
                        | ~500
                        v
 +--------------------------------------------+     +--------------------+
 | Ranker: multi-task DNN                     |<----| Online feature     |
 | P(click), P(add-to-cart), P(buy|click)     |     | store (user, item, |
 | score = combined heads x price/margin      |<-+  | cross, context)    |
 +----------------------+---------------------+  |  +--------------------+
                        | top ~100               |  +--------------------+
                        v                        +--| Model store        |
 +--------------------------------------------+     | (versioned ranker, |
 | Re-rank / policy: diversity (MMR), brand & |     | ANN index, KV tbl) |
 | category caps, sponsored slots, explore    |     +--------------------+
 | (~5%), business rules                      |
 +----------------------+---------------------+
                        | final K items (returned via API to client)
                        v
 +--------------------------------------------+
 | Impression + request logger: req_id,       |--> Kafka / Kinesis
 | candidates, scores, positions, model ver   |    --> data lake (Diagram 2)
 +--------------------------------------------+
```

## Diagram 2: Offline training & data pipeline

**Figure 8.2: Offline pipeline. Logged impressions are joined with delayed purchase labels.
Item-item tables, embeddings and the ranker are rebuilt on separate cadences.**

```
 +-------------+   +--------------+   +---------------+   +-------------------+
 | Impressions |   | Clicks, ATC, |   | Orders/returns|   | Catalog: title,   |
 | log (req_id)|   | views (stream|   | (delayed up to|   | category, brand,  |
 +------+------+   +------+-------+   | ~7 days)      |   | price, images     |
        |                 |           +-------+-------+   +---------+---------+
        +--------+--------+-------------------+                     |
                 v                                                  |
 +-------------------------------+  attribution window (e.g. 24h    |
 | Label join on req_id + ASIN   |  click-to-buy, 7d view-to-buy)   |
 | label = {click, ATC, purchase}|  minus returns / cancellations   |
 +---------------+---------------+                                  |
                 |                                                  |
     +-----------+-------------+-----------------------+            |
     v                         v                       v            v
 +----------------+  +-------------------+  +-------------------------------+
 | Co-occurrence  |  | Retrieval data:   |  | Feature pipelines (batch +    |
 | job (Spark):   |  | (user, bought     |  | streaming): user aggregates,  |
 | baskets ->     |  | item) positives,  |  | item stats (CTR, conv, rating)|
 | item-item sims |  | in-batch + hard   |  | text/image embeddings (PIT-   |
 | (daily)        |  | negatives         |  | correct snapshots)            |
 +-------+--------+  +---------+---------+  +---------------+---------------+
         |                     |                            |
         |                     v                            v
         |           +-------------------+      +----------------------+
         |           | Train two-tower   |      | Train multi-task     |
         |           | (sampled softmax) |      | ranker (daily, warm  |
         |           +---------+---------+      | start; hourly fine-  |
         |                     |                | tune)                |
         |                     v                +----------+-----------+
         |           +-------------------+                 |
         |           | Item-tower embed  |                 |
         |           | all SKUs -> build |                 |
         |           | ANN index         |                 |
         |           +---------+---------+                 |
         v                     v                           v
 +---------------------------------------------------------------------+
 | Offline eval: recall@K (retrieval), NDCG/AUC per head, calibration, |
 | slices (new users, new items, categories), replay / counterfactual  |
 +-----------------------------------+---------------------------------+
                                     | pass gates
                                     v
 +----------------------+   +--------------------+   +---------------------+
 | Model registry +     |-->| Shadow -> canary   |-->| A/B test (GMV/visit,|
 | KV table / index ver |   | (latency, errors)  |   | conversion, guards) |
 +----------------------+   +--------------------+   +----------+----------+
                                                                | ramp to 100%
                                                                v
         +----------------------------------------------------------------------+
         | Feedback loop: new impressions and purchases flow back into the logs |
         | (top of this figure). Watch position bias and popularity feedback;   |
         | keep logging explore traffic for unbiased training/eval data.        |
         +----------------------------------------------------------------------+
```

## Diagram 3: Model architecture

**Figure 8.3: Two-stage model. A two-tower retriever and item-item CF feed a multi-task
ranker. Its heads are combined into one expected-value score.**

```
     RETRIEVAL: TWO-TOWER (sampled softmax, logQ correction)
 +---------------------------+           +-----------------------------+
 | User/query tower          |           | Item tower                  |
 | user_id emb, last-N views |           | ASIN id emb (hashed), brand,|
 | & buys (avg/attn pool),   |           | category path, price bucket,|
 | search terms, anchor ASIN |           | title text emb, image emb   |
 | (detail page), ctx (dev,  |           |                             |
 | locale, time)             |           |                             |
 +-------------+-------------+           +--------------+--------------+
               | MLP -> u (d=128)                       | MLP -> v (d=128)
               +------------------+  dot(u,v)  +--------+
                                  v            v
                          +--------------------------+
                          | score = u . v; ANN top-N |
                          +--------------------------+

     RANKING: MULTI-TASK DNN (DLRM / Wide&Deep / DIN-style)
 +----------+  +-------------+  +------------------+  +-------------------+
 | Sparse   |  | Dense       |  | User behaviour   |  | Cross features    |
 | ids:user,|  | price, avg  |  | seq (last 50     |  | i2i sim(anchor,   |
 | item,    |  | rating, CTR,|  | items) -> target-|  | cand), bought-same|
 | brand,cat|  | conv, stock |  | attention (DIN)  |  | -category-30d     |
 +----+-----+  +------+------+  +--------+---------+  +---------+---------+
      | emb tables    |                  |                      |
      +-------+-------+--------+---------+----------+-----------+
              v                v                    v
         +--------------------------------------------------+
         | Feature interaction (dot / DCN cross layers)     |
         +-------------------------+------------------------+
                                   v
                 +-----------------------------------+
                 | Shared bottom / MMoE experts      |
                 +----+------------+-------------+---+
                      v            v             v
                 +--------+   +---------+   +----------+
                 | pClick |   | pATC    |   | pBuy|clk |  (+ pReturn head)
                 +---+----+   +----+----+   +----+-----+
                     +-------------+-------------+
                                   v
          final = pClick * pBuy|clk * margin_or_price^a  + b * pATC
                  (weights tuned online; a, b are business knobs)
```

## Diagram 4: Real system (Amazon item-to-item collaborative filtering)

**Figure 8.4: Amazon's item-to-item CF, as described in Linden, Smith & York (2003) and
Smith & Linden (2017). The expensive similar-items table is built offline. Online work only
looks up and merges the neighbours of the items the user has interacted with [1][2][3].**

```
 OFFLINE (batch, daily)                         ONLINE (per request)
 +-------------------------------+              +-------------------------------+
 | Purchase / view histories     |              | User's recent items (bought,  |
 | for every customer            |              | viewed, in cart) or anchor    |
 +---------------+---------------+              +---------------+---------------+
                 | for each item i, each customer c             | for each item j
                 | who bought i, each item k bought by c:       v
                 | count co(i,k)                +-------------------------------+
                 v                              | lookup SimTable[j] -> top-N   |
 +-------------------------------+      +------>| neighbours                    |
 | Similarity(i,k): cosine on    |      |       +---------------+---------------+
 | customer vectors (2003), or   |      |                       | merge, weight by
 | observed vs EXPECTED co-buys  |      |                       | recency, drop owned
 | under a random-buyer baseline |  +---+---+                   v
 | (2017; corrects popularity),  |->| Sim   |   +-----------------------------------+
 | time order, relation type     |  | Table |   | top-K "Customers who bought this  |
 +-------------------------------+  | (KV)  |   | also bought" / "Recommended for   |
  cost ~ O(items x avg basket^2);   +-------+   | you"                              |
  most items have few customers                 +-----------------------------------+
                                                latency independent of #users
```

## Data & labels

| Signal | Use as label? | Notes |
|---|---|---|
| Purchase (net of returns) | Primary positive | Sparse, delayed up to days. Use an attribution window [2] |
| Add-to-cart | Auxiliary positive | Denser than purchases, strong intent |
| Click on rec | Auxiliary positive | Densest. Rewards clickbait if used alone |
| Impression without click | Negative (ranker) | Only items that were actually shown. Watch position bias |
| Detail view / dwell | Feature + weak positive | Drives the "also viewed" widget |
| Rating / review | Feature | Too sparse and biased to use as a label |
| Co-purchase in the same basket or session | Item-item pair positive | Basis for "also bought" / "frequently bought together" [1] |
| Random catalog items | Negatives (retrieval) | In-batch and sampled negatives, with logQ correction [5] |

## Features

| Group | Examples |
|---|---|
| User | Hashed user id, purchase categories (30/90/365 d), price sensitivity, Prime flag, locale |
| Session / context | Anchor ASIN, last-N viewed and searched, device, time, page type, cart contents |
| Item | ASIN id, brand, category path, price, rating, review count, stock, delivery speed |
| Item content | Title and description text embedding, image embedding (for cold start) |
| Item stats | Rec CTR, conversion rate, return rate (smoothed, by age bucket) |
| User x item | Category affinity, brand affinity, already-owned flag, i2i score to recent items |
| Anchor x candidate | Co-purchase lift, complement vs substitute type, price ratio |

## Model choices (baseline -> advanced)

| Stage | Baseline | Advanced | Tradeoff |
|---|---|---|---|
| "Also bought" | Co-occurrence counts | Item-to-item CF with an expected-vs-observed correction [1][2] | Cheap and explainable. Weak on new items |
| "Frequently bought together" | Co-purchase pairs minus co-view pairs (removes substitutes) | Complement model with type-aware, diversified outputs (Amazon P-Companion) [26] | Separates complements from substitutes; handles cold-start anchors via content |
| Item similarity | Content (text/category) kNN | Item2vec / session embeddings (ByteByteGo similar-listings style) [13] | Covers cold start. Needs ANN |
| Homepage retrieval | Popular by category | Matrix factorization, then two-tower [4][5] | Two-tower personalizes and scales via ANN, but is costly to train |
| Sequential | Last-viewed neighbours | Transformer next-item model (e.g. SASRec) [24] | Captures intent shifts. Higher latency |
| Ranker | Logistic regression or GBDT on hand-made features | Wide & Deep / DLRM / DIN with multi-task heads [6][7][23][25] | Better calibration and accuracy. Needs a GPU/CPU serving budget |
| Re-rank | Business rules | MMR diversity plus a constrained optimizer (sponsored slots, margins) | Trades a little relevance for diversity and revenue |

## Loss functions

| Model | Loss |
|---|---|
| Two-tower | Sampled softmax / in-batch softmax with logQ correction [5]. Optionally a hard-negative margin loss |
| Item embeddings | Skip-gram with negative sampling over sessions or baskets |
| Ranker heads | Per-task binary cross-entropy, summed with task weights (or uncertainty weighting). pBuy trained on clicked items, or as an ESMM-style pClick x pBuy chain to avoid selection bias [27] |
| Re-rank (optional) | Listwise (LambdaRank / softmax over the slate) |

## Offline metrics

| Stage | Metric |
|---|---|
| Retrieval | Recall@K / hit-rate@K on next purchase, catalog coverage |
| Ranking | AUC and log loss per head, NDCG@K, calibration (predicted vs. observed) |
| Slate | Intra-list diversity, novelty, share of long-tail items |
| Evaluation | Time-based split (train on days 1..T, test on T+1). Counterfactual / IPS replay on exploration logs |
| Slices | New users, new items, categories, logged-out, mobile vs desktop |

## Online metrics / A-B

| Type | Metric |
|---|---|
| Primary | Revenue or GMV per visit attributed to the widget, conversion rate |
| Secondary | Widget CTR, add-to-cart rate, units per order |
| Guardrails | Return rate, page latency p99, sponsored revenue, complaints, sitewide sales (cannibalization) |
| Design | User-level randomization. 1-2 weeks to cover the purchase delay. Holdout for long-term effect. Interleaving for fast ranker comparisons |

## Serving & latency budget (assumption, per widget, p99)

| Step | Budget | How |
|---|---|---|
| Request parse + user/session fetch | 5 ms | Online feature store |
| Candidate gen (parallel) | 15-25 ms | KV lookup of the item-item table (about 1 ms); ANN HNSW top-500 |
| Filters | 5 ms | Bloom filter of owned items, inventory cache |
| Feature assembly | 10 ms | Batched multi-get |
| Ranking ~500 items | 20-30 ms | CPU/GPU batched inference, quantized embeddings |
| Re-rank + policy | 5 ms | In-process |
| Total | under 80-100 ms | Fallback: cached precomputed recs per user, or bestsellers |

## Monitoring

| What | Signal |
|---|---|
| Data | Feature null rates and drift (PSI), label delay, event volume per surface |
| Model | Calibration per head, score distribution, online AUC from delayed labels |
| Business | Widget CTR and GMV per visit by surface, anomaly alerts during events (Prime Day) |
| System | p99 latency, fallback rate, ANN recall vs exact search, KV staleness |
| Ecosystem | Concentration (Gini of impressions), share of new and long-tail items |

## Failure modes

| Failure | Mitigation |
|---|---|
| Popular items dominate "also bought" (e.g. batteries everywhere) | Expected-vs-observed similarity, IDF-style damping [2] |
| Recommending items the user just bought (non-consumables) | Owned-item filter. Allow consumables (replenishment) |
| Substitutes on the post-purchase page | Separate complement and substitute models by page context [2][26] |
| New SKU or new user cold start | Content-tower embeddings, category bestsellers, exploration slots |
| Out-of-stock or undeliverable recs | Real-time inventory filter; substitute recs (Instacart / Huyen prompt) [8][10] |
| Feedback loop / position bias | Exploration traffic, position feature at train time (set to a constant at serve time), IPS |
| Seasonal shift (holidays, Prime Day) | Faster retrains, recency-weighted features, short-window trending source |
| Training-serving skew | Point-in-time feature snapshots, logging the features used at serving |

## Interviewer follow-ups

- How would you compute item-item similarity for 10^8 items without O(N^2) work?
- "Also bought" vs "also viewed" vs "frequently bought together": should these be one model or three? How do you tell complements from substitutes?
- Why might cosine similarity on co-purchase counts over-recommend bestsellers? How do you fix it?
- Purchases arrive days late. How do you build labels and train daily?
- Click, add-to-cart or purchase: which do you optimize? How do you combine multi-task heads into one score?
- How do you handle a brand-new product with no interactions, or a logged-out user?
- How do you build "frequently bought together" so it shows a phone case for a phone, not another phone? What data separates complements from substitutes?
- How do you avoid recommending a second TV to someone who just bought one, while still recommending detergent?
- How do you rebuild the ANN index and item-item table without downtime? How stale is acceptable?
- How do you blend sponsored products with organic recs without hurting trust?
- How do you measure incrementality (would the user have bought it anyway)?
- Diversity: how would you stop the carousel showing 10 phone cases of the same type?
- How do you serve recs under 100 ms during Prime Day traffic spikes? What is the fallback?
- How do you evaluate offline when the logging policy biased what users saw?
- How would you add an LLM (for example, generating "why recommended" text or a semantic retrieval source)?

## Portal comparison

| Portal | Framing | Candidate gen | Ranker | Distinctive emphasis |
|---|---|---|---|---|
| Amazon papers [1][2][3] | Item-anchored, scale-first | Item-to-item CF table | Light heuristics (in 2003) | Offline table, online lookup; expected co-buy correction |
| ByteByteGo [13][14] | Similar listings (session-based) + video home feed | Session item embeddings + ANN; multi-source | Two-tower / DNN | Embeddings from co-occurrence in sessions; hybrid filtering |
| Educative Grokking [15] | Netflix-style personalized recs | CF/MF + content + embeddings | Pointwise DNN | Data-generation and staged funnel |
| alirezadir AIMLInterviews [8] | Generic recsys template; Instacart replacement prompt | User/item CF, MF, two-tower, content | Pointwise / pairwise / listwise LTR | 9-step formula, metrics catalogue |
| Hello Interview [17] | Recsys pattern | Multi-source retrieval | Multi-stage ranking + re-ranking | Interview delivery and tradeoff talk |
| Chip Huyen [10][20] | Out-of-stock replacements, checkout speed | Open-ended | Open-ended | Business framing, data and deployment issues |

## References

1. G. Linden, B. Smith, J. York, "Amazon.com Recommendations: Item-to-Item Collaborative Filtering", IEEE Internet Computing, 2003. https://doi.org/10.1109/MIC.2003.1167344 (known)
2. B. Smith, G. Linden, "Two Decades of Recommender Systems at Amazon.com", IEEE Internet Computing, 2017. https://doi.org/10.1109/MIC.2017.72 (known)
3. Amazon Science, "The history of Amazon's recommendation algorithm", amazon.science, 2019. https://www.amazon.science/the-history-of-amazons-recommendation-algorithm (known; host unreachable at review)
4. P. Covington, J. Adams, E. Sargin, "Deep Neural Networks for YouTube Recommendations", RecSys / Google Research, 2016. https://research.google/pubs/pub45530/ (ch-ref)
5. X. Yi et al., "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations", RecSys / Google Research, 2019. https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/ (ch-ref)
6. H.-T. Cheng et al., "Wide & Deep Learning for Recommender Systems", arXiv, 2016. https://arxiv.org/abs/1606.07792 (ch-ref)
7. M. Naumov et al., "Deep Learning Recommendation Model for Personalization and Recommendation Systems (DLRM)", arXiv, 2019. https://arxiv.org/abs/1906.00091 (known; ID confirmed via fetched github.com/facebookresearch/dlrm)
8. alirezadir, "ML System Design" (recsys template; Instacart replacement-product prompt), AIMLInterviews, GitHub, 2025. https://github.com/alirezadir/AIMLInterviews/blob/main/src/MLSD/ml-system-design.md (fetched)
9. K. Pham, "machine-learning-interview", GitHub, 2021. https://github.com/khangich/machine-learning-interview (fetched)
10. C. Huyen, "Machine Learning Systems Design: Exercises" (ex. 3 replacement items, ex. 21 faster purchase), GitHub, 2019. https://github.com/chiphuyen/machine-learning-systems-design/blob/master/content/exercises.md (fetched)
11. IGotAnOffer, "Amazon Machine Learning Engineer Interview", igotanoffer.com, 2025. https://igotanoffer.com/blogs/tech/amazon-machine-learning-engineer-interview (ch-ref)
12. IGotAnOffer, "Meta Machine Learning Engineer Interview", igotanoffer.com, 2025. https://igotanoffer.com/blogs/tech/facebook-machine-learning-engineer-interview (ch-ref)
13. A. Aminian & A. Xu, "Similar Listings on Vacation Rental Platforms" (Machine Learning System Design Interview), ByteByteGo, 2023. https://bytebytego.com/courses/machine-learning-system-design-interview/similar-listings-on-vacation-rental-platforms (ch-ref; same URL used in ch. 09; host unreachable at review)
14. A. Aminian & A. Xu, "Video Recommendation System" (Machine Learning System Design Interview, ch. 6), ByteByteGo, 2023. https://bytebytego.com/courses/machine-learning-system-design-interview/video-recommendation-system (ch-ref)
15. Educative, "Grokking the Machine Learning Interview" (recommendation system module), educative.io, n.d. https://www.educative.io/courses/grokking-the-machine-learning-interview (ch-ref)
16. Educative / K. Pham, "Machine Learning System Design" (course), educative.io, 2021. https://educative.io/courses/machine-learning-system-design/introduction (ch-ref)
17. Hello Interview, "Recommendation Systems (ML System Design pattern)", hellointerview.com, 2025. https://www.hellointerview.com/learn/ml-system-design/patterns/recommendation-systems (ch-ref)
18. Exponent, "Introduction to the ML System Design Interview" (course), tryexponent.com, 2025. https://www.tryexponent.com/courses/ml-system-design/mlsd-intro (ch-ref)
19. Aced (formerly Exponent), "Machine Learning System Design Interview (2026 Guide)", aced.io, 2026. https://www.aced.io/blog/machine-learning-system-design-interview-guide (ch-ref)
20. C. Huyen, "Introduction to Machine Learning Interviews Book", huyenchip.com, 2021. https://huyenchip.com/ml-interviews-book/ (ch-ref)
21. P. Halina, "ML Systems Design Interview Guide", patrickhalina.com, 2020. http://patrickhalina.com/posts/ml-systems-design-interview-guide/ (ch-ref)
22. DesignGurus, "The Grokking series", designgurus.io, 2025. https://www.designgurus.io/grokking (ch-ref)
23. G. Zhou et al., "Deep Interest Network for Click-Through Rate Prediction" (Alibaba), arXiv, 2017. https://arxiv.org/abs/1706.06978 (known; ID confirmed via fetched github.com/zhougr1993/DeepInterestNetwork)
24. W.-C. Kang, J. McAuley, "Self-Attentive Sequential Recommendation (SASRec)", ICDM, 2018. https://arxiv.org/abs/1808.09781 (known; venue confirmed via fetched github.com/kang205/SASRec)
25. J. Ma et al., "Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts", KDD, 2018. https://doi.org/10.1145/3219819.3220007 (known)
26. J. Hao et al., "P-Companion: A Principled Framework for Diversified Complementary Product Recommendation" (Amazon), CIKM, 2020. https://doi.org/10.1145/3340531.3412732 (known)
27. X. Ma et al., "Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate (ESMM)" (Alibaba), SIGIR, 2018. https://arxiv.org/abs/1804.07931 (known)

