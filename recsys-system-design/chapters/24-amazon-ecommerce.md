# 24. Amazon and E-commerce: From Item-to-Item Collaborative Filtering to Query-Aware Retrieval

E-commerce is where industrial recommendation started. Amazon's item-to-item collaborative filtering went into production in the late 1990s, was patented in 2001 and described publicly in 2003. It is still the clearest example of a recommender design that won because it fitted the system constraints of its time: a catalog of millions of items, tens of millions of customers, and a page that had to render in well under a second. Twenty years on, e-commerce recommenders look different. They use two-tower retrieval, sequence models, graph embeddings and LLM-derived item representations. But the problem has stayed the same, and so has its structure. Commerce recommenders differ from media and social feeds in several ways:

- **The conversion signal is sparse and delayed but very valuable.** One purchase is worth thousands of impressions, so training labels are imbalanced and multi-level (impression, click, add-to-cart, purchase).
- **The catalog is long-tailed and churns.** New SKUs, third-party listings and one-of-a-kind items (eBay, Etsy) arrive constantly. Many items have no interaction history.
- **Item relationships are typed.** "Similar" (substitute) and "goes with" (complement) are different products with different data sources. Mixing them up is the classic failure.
- **Search and recommendation merge.** Most commerce sessions start with a query, so the best recommenders are *query-aware*. Product search is itself a personalized ranking problem.
- **Inventory, price and availability are hard constraints.** An item that is out of stock, undeliverable or (for grocery) unavailable at the shopper's store has to be filtered out at serving time.

This chapter starts with Amazon's original algorithm and its later refinements. It then covers Amazon's deep-learning infrastructure, its product-search and complementary-item research, and Amazon Personalize, the managed service that packages Amazon-style recipes for other companies. After that come the marketplace and retailer designs (eBay, Etsy, Zalando, Alibaba-style graph embeddings, prod2vec) and query-aware recommendation patterns.

**A note on sourcing.** For this chapter the live web-search budget was used up, and the writing environment could not reach arXiv, amazon.science or AWS documentation directly. The chapter therefore relies on well-established primary sources (papers, patents, official documentation) whose content and URLs are widely cited and stable. Numbers that could not be re-checked are marked *approximate*. Wayfair, Shopify, Target, Walmart and Instacart have all published recommender and search work on their engineering blogs. Their specific architectures and numbers could not be verified against primary sources in this pass, so Section 9 covers them only at the level of patterns, without specific figures. Readers should consult those companies' engineering blogs directly.

---

## 1. Amazon item-to-item collaborative filtering (1998-2003)

### 1.1 The problem

In 2003 Linden, Smith and York described the constraints Amazon faced [1]. The catalog held several million items and there were more than 29 million customers. Recommendations had to be computed in real time, within about half a second, for every page view. They also had to react immediately to new information: an item added to the cart should change the recommendations on the next page. New customers had very little history, and long-time customers could have thousands of purchases.

The paper looked at the three families of methods available at the time and rejected each of them for this setting:

| Approach | How it works | Why it failed at Amazon scale [1] |
|---|---|---|
| User-based CF | Find similar customers (cosine over M-dim item vectors), recommend what they bought | Worst case O(MN) per request; dimensionality reduction or sampling hurts quality |
| Cluster models | Assign users to segments, recommend segment favourites | Cheap online, but coarse; recommendations are poor for an individual |
| Search/content-based | Query the catalog for same author, artist, keywords | Too narrow: recommends what the customer already knows |
| **Item-to-item CF** | Precompute similar-items table; at request time look up items in the user's history | Online cost depends only on the user's history size, not on M or N |

(M = number of customers, N = number of catalog items.)

### 1.2 The algorithm

The key inversion is to compute similarity between **items** rather than between users. This can be done entirely offline. The paper's pseudocode is [1]:

```
For each item I1 in catalog:
    For each customer C who purchased I1:
        For each item I2 purchased by customer C:
            record that a customer purchased I1 and I2
    For each item I2:
        compute similarity(I1, I2)      # cosine over customer vectors
```

Each item is a vector over customers, and similarity is the cosine of the angle between two item vectors. The worst-case offline cost is O(N^2 M). In practice it is close to O(NM), because most customers buy very few items. Precomputing on a sample of customers, or ignoring the heaviest buyers, cuts the cost further [1]. The patent filed by Linden, Jacobi and Benson describes the same similarity-mapping design and how it applies to shopping carts and purchase histories [2]. Sarwar et al. published a closely related academic study of item-based CF in the same period [3].

At request time the system works like this:

```
 user history / cart / page context
        |
        v
 +---------------------+     lookup per item      +---------------------------+
 | seed items (k)      | ------------------------> | similar-items table       |
 | purchased, rated,   |                           | item -> [(item, score)]   |
 | carted, viewed      | <------------------------ | precomputed offline       |
 +---------------------+   top-n lists per seed    +---------------------------+
        |
        v
 merge + aggregate scores, drop owned/seen items, apply business filters
        |
        v
 "Customers who bought this also bought", "Your Recommendations", cart recs
```

Three properties made it a good production design:

1. **Latency is independent of catalog and customer counts.** The online cost is a few table lookups and a merge.
2. **It is reactive.** Because the inputs are the current cart or the page item, the recommendations change immediately without any retraining.
3. **It is explainable.** "Because you purchased X" falls straight out of the algorithm, and that builds trust and helps users debug the recommendations.

Smith and Linden later noted that the 2003 paper was chosen by *IEEE Internet Computing* as the paper from its first 20 years that had best stood the test of time [4]. Amazon's own retrospective also tells the story of how the algorithm spread across the site and beyond [5].

### 1.3 Refinements: "Two Decades of Recommender Systems at Amazon.com" (2017)

The 2017 follow-up by Smith and Linden is one of the most useful design documents in the field. It explains why naive co-occurrence fails and how Amazon corrected it [4].

**The popularity problem.** If you rank related items by raw co-purchase count, every item's "customers also bought" list is dominated by bestsellers. The authors' example is a Harry Potter book. To correct this, compute the *expected* number of co-purchases of X and Y under independence and compare the observed count with it. For each customer c who bought X, the probability that c also bought Y by chance is about 1 - (1 - P_Y)^|c|, where P_Y is Y's purchase probability and |c| is the number of items c has bought. Summing over all buyers of X gives the expected count E_XY. A good relatedness score is then N_XY - E_XY, normalised by something like sqrt(E_XY) (a chi-square-style statistic) [4]. In this formulation, heavy buyers contribute less evidence per co-purchase, which is the same intuition as down-weighting very active users in modern CF.

**Other lessons from the 2017 paper** [4]:

- **Time and direction matter.** Someone who buys a camera wants a memory card next, not another camera. Directional "bought X then Y" statistics work better than symmetric co-occurrence for complements.
- **Use the right signal for the right relation.** Co-views suggest substitutes ("people who viewed this also viewed"). Co-purchases suggest complements or bundle-mates ("frequently bought together").
- **Items are not all equal.** Different product categories need different treatment. Consumables, books in a series and durable goods have different repeat-purchase and complementarity patterns.
- **The vision.** The authors argue that recommendation should become more like a conversation with a knowledgeable friend: every interaction on the site, including search, should be a recommendation. This anticipates today's query-aware and conversational shopping systems.

---

## 2. Amazon's deep-learning recommendation stack

### 2.1 DSSTNE: sparse deep learning for recommendations (2016)

In 2016 Amazon open-sourced DSSTNE (Deep Scalable Sparse Tensor Network Engine). It was built for training and serving neural recommenders on very sparse, high-dimensional input such as one-hot purchase histories over a large catalog [6]. Its design points are typical of commerce recommenders of that time:

- Input layers are extremely wide (one unit per item) and very sparse. DSSTNE uses sparse kernels so that it only computes over non-zero inputs.
- **Model parallelism across GPUs.** Wide layers are split across devices, so the model does not have to fit on one GPU. Most frameworks of the day only did data parallelism.
- Models are specified declaratively (JSON network definitions). The typical use case is an autoencoder-style "history in, purchase-probability-over-catalog out" recommender.

DSSTNE's lesson is still relevant: in commerce the *output* side (scoring the whole catalog) and the *embedding* side are the expensive parts. Modern systems solve this with two-tower retrieval and ANN search rather than full-softmax layers.

### 2.2 Product-to-product relations: substitutes and complements

Academic work using Amazon data formalised the substitute/complement split. McAuley, Pandey and Leskovec (KDD 2015) learned to infer "substitutable" and "complementary" product networks from review text, using Amazon's "also viewed" and "also bought" graphs as supervision [7]. McAuley et al. (SIGIR 2015) showed that visual style alone can predict which items are bought together, using CNN image features projected into a learned "style space" [8]. These datasets (the Amazon review corpora) became a standard benchmark. The 2023 Amazon Reviews release contains on the order of 570 million reviews over about 48 million items (*approximate*). It came with BLaIR, a family of language models pretrained to align review and metadata text with item identity for retrieval and recommendation [9].

### 2.3 P-Companion: diversified complementary recommendation (CIKM 2020)

P-Companion is an Amazon framework for "frequently bought together"-style complementary recommendation. It deals with three problems that raw co-purchase statistics cannot solve [10]:

1. **Noisy labels.** Co-purchase pairs mix complements with substitutes (people buy two phone cases to compare) and with unrelated items. P-Companion builds *distant supervision*: it keeps co-purchased pairs and drops pairs that are also frequently co-viewed, since co-viewed pairs are likely substitutes.
2. **Cold start.** New products have no co-purchases. Products are represented by a product embedding learned from content (catalog attributes and text) together with behaviour through a graph neural network. So a new item can get complements from its content alone.
3. **Diversity.** A camera should get a lens, a bag *and* a memory card, not five memory cards. The model first predicts a set of complementary **product types** for the query item's type, then retrieves items within each predicted type.

```
 query item q
    |
    v
 [product embedding (content + GNN over behaviour graph)]
    |
    +--> [complementary type transition]  type(q) -> {t1, t2, ..., tk}
    |                                       (diverse set of complement types)
    v
 for each ti: [type-conditioned item embedding] --> ANN within type ti
    |
    v
 merge per-type top items -> diversified "frequently bought together" list
```

The paper reports offline gains over co-purchase and embedding baselines, and online A/B improvements in production at Amazon (specific lift figures are *approximate* and should be checked in [10]). The design pattern of **type first, then item** reappears in many later complementary-recommendation systems.

### 2.4 Session and multi-locale shopping data

In 2023 Amazon released Amazon-M2 for the KDD Cup. It is a multilingual, multi-locale shopping-session dataset (millions of sessions across six locales, *approximate*) with tasks for next-product prediction, cross-locale transfer for low-resource locales, and next-product title generation [11]. The task design shows what Amazon cares about: session-based next-item prediction, transfer from data-rich to data-poor marketplaces, and text generation over product titles as a step toward LLM-based recommenders.

---

## 3. Amazon product search as a recommender

### 3.1 Semantic Product Search (KDD 2019)

Nigam et al. describe a neural matching model that Amazon deployed to complement lexical (inverted-index) product search [12]. It is an early, influential commerce two-tower model, and it was built with an unusually explicit eye on production constraints.

**Architecture.** It is a symmetric two-tower model with a *shared* token embedding table for queries and products. Each side is tokenised into word unigrams, word bigrams and character trigrams. Out-of-vocabulary tokens are **hashed** into a fixed number of bins instead of being dropped. Embeddings are combined by **average pooling**: the authors found that simple pooling beat more complex encoders (RNN/CNN variants) on their data and is much cheaper. Product text comes from the title and other catalog attributes. Relevance is the cosine similarity between query and product vectors, so retrieval works with approximate nearest-neighbour (ANN) search.

```
 query text ---tok--> [uni|bi|char-3gram|hashed OOV] --embed--> avg pool --+
                                                                          |
                         shared embedding table                          cos
                                                                          |
 product text -tok--> [uni|bi|char-3gram|hashed OOV] --embed--> avg pool --+
 (title, attributes)
                      offline: embed catalog -> ANN index; online: embed query -> kNN
```

**Loss.** A three-part hinge loss uses graded relevance from behaviour. *Purchased* products should score above a high threshold. *Impressed-but-not-purchased* products should fall in a middle band. *Random* negatives should score below a low threshold. The impressed-not-purchased class is important: those items are often relevant but not chosen, and treating them as hard negatives would teach the model to push down relevant items [12].

**Lessons from the paper** [12]:

- Hashing OOV tokens gave large gains, because product search is full of model numbers, brands and misspellings.
- Average pooling with rich n-gram tokenisation beat heavier encoders, and is also cheaper at billion-product inference.
- Semantic retrieval is used *alongside* lexical matching, not instead of it. Lexical matching stays precise for exact attribute queries, and the semantic model adds recall for vocabulary-gap queries.

### 3.2 Shopping Queries Dataset (ESCI, 2022)

Amazon's Shopping Queries Dataset contains about 130 thousand difficult search queries and about 2.6 million human-labelled (query, product) judgments in English, Japanese and Spanish (*approximate*). It was released for KDD Cup 2022 [13]. Its label scheme is a useful design artefact in its own right. Each pair is **E**xact, **S**ubstitute, **C**omplement or **I**rrelevant (ESCI). This makes explicit that a search engine for shopping is also a recommender: substitutes and complements are valid, useful results for a query, but they are different from exact matches and should be ranked and presented differently.

### 3.3 Query-aware recommendation as a design pattern

In e-commerce, a "query-aware" recommender conditions candidates or scores on the shopper's current query (or the query that led to the current page), not only on their long-term profile. The pieces above add up to a recognisable architecture:

```
           query q, user u, context (page, cart, locale, device)
                     |
     +---------------+-----------------+----------------------+
     |                                 |                      |
 lexical retrieval              semantic two-tower       item-to-item / complement
 (inverted index)               (query emb -> ANN)       expansion from top results
     |                                 |                      |
     +---------------+-----------------+----------------------+
                     v
       candidate union, filters (stock, delivery, region, policy)
                     v
       ranker: f(q, u, item, context) with ESCI-style relevance + behaviour labels
                     v
       slotting: exact matches first; substitutes; complements / "related" widgets
```

The parts that are specific to commerce are: (a) **typed relevance** (ESCI) rather than binary relevance; (b) expanding from the top results through item-to-item graphs, so complements reach the page without being literally retrieved for the query; and (c) treating personalization as a re-ranking signal that must not override relevance. A personalised ranker should never promote an irrelevant item because the user likes the brand.

---

## 4. Amazon Personalize: Amazon's recommender as a managed service

Amazon Personalize was announced at re:Invent 2018 and became generally available in 2019. It packages Amazon-style recommendation pipelines as "recipes" (algorithm templates) and "recommenders" (use-case-specific, preconfigured endpoints) [14]. Its design shows what a general-purpose commerce recommender has to support.

### 4.1 Data model

- **Interactions dataset** (required): user_id, item_id, timestamp, event_type (for example click or purchase) and optional event_value and context fields.
- **Items dataset**: item metadata (categorical, numerical and, in later versions, unstructured text fields).
- **Users dataset**: user metadata.
- **Real-time events**: an event tracker with a PutEvents API streams interactions into the deployed solution, so recommendations react within a session without retraining. This is the same "react to the cart now" property that Amazon's original design had.

### 4.2 Recipe families

| Recipe type | Representative recipes | Purpose |
|---|---|---|
| USER_PERSONALIZATION | aws-user-personalization, aws-user-personalization-v2; legacy aws-hrnn, aws-hrnn-metadata, aws-hrnn-coldstart | Top-N items for a user |
| PERSONALIZED_RANKING | aws-personalized-ranking (and v2) | Re-rank a supplied candidate list (search results, curated lists) for a user |
| RELATED_ITEMS | aws-similar-items, aws-sims, aws-semantic-similarity | Item-to-item ("similar to X") |
| POPULAR_ITEMS | aws-popularity-count, aws-trending-now | Popularity and trending baselines |
| USER_SEGMENTATION | aws-item-affinity, aws-item-attribute-affinity | Find users likely to engage with an item or attribute (marketing) |
| Next best action | aws-next-best-action | Recommend actions (sign up, enrol) rather than items |

Sources: AWS Personalize developer guide [14]. The exact set of recipes has changed over time, and some legacy recipes are no longer offered to new customers.

Key design notes:

- **HRNN lineage.** The original user-personalization recipe was a *hierarchical RNN*. It models short-term within-session behaviour with a session-level GRU and carries a user-level state across sessions. This follows the GRU4Rec line of session-based models [15][16]. The "-metadata" and "-coldstart" variants add item and user features and the ability to recommend items with no interactions.
- **aws-sims** is classical item-to-item CF on co-occurrence, the direct descendant of Section 1. **aws-similar-items** combines co-occurrence with item metadata so that cold items get neighbours [14].
- **Exploration is a first-class parameter.** User-Personalization exposes an exploration weight and an item-age cutoff. It reserves part of the slate for new or less-exposed items, which is a bandit-style answer to catalog churn [14].
- **The v2 recipes** (released around 2024) moved to a transformer-based architecture and raised catalog limits, to the order of millions of items (*approximate*; see [14]).
- **Use-case "recommenders"** for e-commerce domain dataset groups wrap these recipes behind business-named endpoints: "Customers who viewed X also viewed", "Frequently bought together", "Best sellers", "Most viewed" and "Recommended for you" [14].
- **Filters and promotions.** Filter expressions exclude or include items by metadata or interaction history (for example out-of-stock items or already-purchased items). Promotions reserve a share of slots for a business-defined subset. Every commerce recommender eventually needs both.

### 4.3 Architecture of a Personalize-based deployment

```
 clickstream / orders ----batch (S3 import)-----> datasets --> solution version (train)
         |                                                        |
         +--real-time PutEvents--> event tracker ------------+    |
                                                             v    v
 app request (user, item, ctx) --> campaign / recommender endpoint
                                    - candidate scoring (recipe model)
                                    - filters (stock, owned, category)
                                    - promotions / exploration slots
                                   <-- ranked item ids (+ scores)
 batch inference jobs --> S3 --> email, push, offline segments
```

The main lesson of Personalize as a product is that most commerce recommendation needs fit into a small set of tasks: user top-N, re-rank a list, item-to-item, popularity, and user segmentation. These are served with real-time event ingestion, filtering and exploration. The algorithm inside each task can change (HRNN to transformer) without changing the contract with the application.

---

## 5. Embedding lineage for product recommendations

Before covering marketplaces, it helps to look at the representation-learning ideas that most commerce systems share.

### 5.1 prod2vec (Yahoo, KDD 2015)

Grbovic et al. trained word2vec-style skip-gram embeddings on sequences of purchases extracted from e-mail receipts. Products are treated as "words" and a user's purchase sequence as a "sentence" [17]. Variants included **bagged-prod2vec**, which treats each receipt (a basket) as a bag of co-purchased items, and **user2vec**, which embeds users in the same space. The system recommended products in an e-mail inbox setting. The lasting lesson is that *sequential co-purchase context* produces embeddings whose neighbours are good "next purchase" candidates, and they are cheap to train.

### 5.2 Graph embeddings with side information (Alibaba EGES, KDD 2018)

Alibaba's "billion-scale commodity embedding" builds an item graph from users' session behaviour sequences. It runs weighted random walks over the graph and trains skip-gram on the walks (as in DeepWalk) [18]. To deal with cold start and sparsity, it adds **side information** (category, brand, shop, price bucket). The final item embedding is a learned, attention-weighted average of the item-ID embedding and the side-information embeddings (EGES: Enhanced Graph Embedding with Side information). Items with no interactions can still be embedded from their side information. This "ID embedding + attribute embeddings, gated" pattern is now standard in commerce item towers. Alibaba's systems are covered in more depth in the China e-commerce chapter.

### 5.3 Zalando Fashion DNA (2016)

Zalando's Fashion DNA work learns a dense item representation for fashion articles from three inputs: curated attributes, product images and sales data [19]. The network is trained to predict customer-article purchase from these inputs. Its intermediate layer becomes the "fashion DNA" vector, which is reused for similarity, search and personalization. Because images and attributes are available on day one, new articles get useful vectors before they have any sales. Fashion is a good example of a domain where most of the catalog turns over every season, so content-derived embeddings are essential.

---

## 6. eBay: recommendations without a stable catalog

### 6.1 The problem

eBay's catalog is very different from Amazon's. A large share of listings are unique items or short-lived auctions, often listed by individual sellers with free-text titles. Pure item-to-item CF works poorly, because most listings never build up enough co-interactions before they expire. eBay's recommendation work has therefore leaned on **content-based and embedding-based** methods that generalise across listings: title text, category, aspects (structured key-value attributes) and images.

### 6.2 Personalized embedding-based recommendations (2021)

Wang et al. describe eBay's deployment of a personalized embedding-based recommendation system [20]. The design is a two-tower model:

- **Item tower.** It encodes listing content (title text plus categorical features such as category and aspects), so new listings can be embedded immediately without interaction history.
- **User tower.** It encodes the user's recent sequence of interacted listings (views, purchases, watches and similar events).
- **Serving.** Item embeddings are computed offline and in near-real time as listings are created, then indexed for ANN retrieval. User embeddings are computed from the live event stream at request time, so the candidates reflect in-session intent.

```
 new listing --> item tower (title tokens, category, aspects) --> item emb --> ANN index
                                                                              ^
 user events (recent N) --> user tower (sequence encoder) --> user emb --kNN--+
                                                                              |
                                          candidates --> downstream ranker --> module
```

The paper reports online gains from A/B tests over the previous production recall system (exact figures *approximate*; see [20]). The lesson for designers: **when items are ephemeral, put the generalisation in the item tower** (content features) and keep the user side behaviour-driven.

### 6.3 Lessons from eBay's setting

- Item-to-item tables keyed on listing IDs go stale fast. Key similarity on content or on canonical product clusters instead.
- Structured aspects (brand, size, colour, condition) are powerful features and filters. In a marketplace they double as a way to normalise messy seller titles.
- Recommendation modules on the view-item page must handle the "already ended" state. If the item a user is looking at has sold, the "similar items" module becomes the main conversion path.

---

## 7. Etsy: unified embedding-based personalized retrieval

### 7.1 The problem

Etsy's marketplace has a long tail of handmade and vintage items, and most items are one-of-a-kind or come in small quantities. Search queries are often vague or aesthetic ("boho wedding decor"). Pure lexical retrieval misses many relevant items, and pure behavioural CF fails on the tail.

### 7.2 Unified Embedding (KDD 2023)

Jha et al. describe Etsy's "Unified Embedding" system for personalized retrieval in search [21]. It brings several representation types into one two-tower model:

- **Product tower.** It combines (a) **transformer-based** text embeddings of the listing, (b) **term-based** embeddings (bag-of-tokens, which keep the precision of lexical matching), and (c) **graph-based** embeddings learned from query-listing interaction graphs.
- **Query/user tower.** It encodes the query and **personalisation features** from the user's recent behaviour, so retrieval itself is personalised rather than only the ranking.
- **Quality-aware retrieval.** Etsy blends a product-quality signal into ANN retrieval, so the nearest-neighbour stage does not surface low-quality listings that the ranker would only demote later.
- **Serving.** Product embeddings are indexed for ANN search. The embedding retrieval path runs alongside the existing lexical retrieval, and the results are unioned before ranking.

```
 query + user features --> query/user tower ----------------------------+
                                                                         | cosine
 listing --> [transformer text emb] + [term emb] + [graph emb] -> fuse --+
                                         |
                       ANN index (+ quality-aware scoring) --> union with lexical
                                                         --> learning-to-rank
```

Etsy reported site-wide conversion gains from the launch. The paper cites figures of a few percent in search purchase rate and site-wide conversion (*approximate*; see [21]). The paper also discusses practical lessons: negative sampling strategies, the cost of keeping ANN indexes fresh as inventory changes, and how lexical-style token embeddings inside a neural model help recover precision on specific queries.

### 7.3 Takeaways from Etsy

- **Fuse the three views** (text semantics, exact terms, interaction graph). Each one covers the others' blind spots.
- **Personalise at retrieval, not just at ranking.** For vague queries, the set of candidates is where personalisation has the most effect.
- **Put quality into retrieval.** A purely semantic ANN stage will find relevant but low-quality items, which wastes ranking capacity.

---

## 8. Comparison of the major e-commerce designs

| System | Year | Core idea | Signals | Cold-start strategy | Serving pattern |
|---|---|---|---|---|---|
| Amazon item-to-item CF [1][4] | 2003/2017 | Precomputed item similarity, online lookup | Purchases, cart, views | Content fallback; similarity on sparse data | Table lookup + merge |
| DSSTNE [6] | 2016 | Sparse wide neural nets, model-parallel | Purchase histories | Not addressed by engine | Batch/online scoring |
| Semantic Product Search [12] | 2019 | Shared-vocab two-tower, avg pooling, hinge loss | Purchases, impressions | Text-only product tower | ANN + lexical hybrid |
| P-Companion [10] | 2020 | Type-then-item complement retrieval | Co-purchase minus co-view, content | GNN/content product embedding | Per-type ANN |
| Amazon Personalize [14] | 2019- | Recipe-based managed service | Interactions, items, users, real-time events | Metadata recipes, exploration weight | Campaign/recommender endpoint |
| prod2vec [17] | 2015 | Skip-gram on purchase sequences | E-mail receipts | None (ID-based) | kNN on product vectors |
| Alibaba EGES [18] | 2018 | Random-walk graph embeddings + side info | Session sequences, attributes | Side-info embeddings | ANN candidate generation |
| Zalando Fashion DNA [19] | 2016 | Item vectors from attributes + images + sales | Purchases, images | Image/attribute-driven vectors | Embedding similarity |
| eBay personalized EBR [20] | 2021 | Content item tower, sequence user tower | Recent user events, listing text | Content item tower | ANN, real-time user emb |
| Etsy Unified Embedding [21] | 2023 | Fused text+term+graph two-tower, personalised | Queries, clicks, purchases | Text/term towers | ANN + lexical union |

---

## 9. Other retailers: Wayfair, Shopify, Walmart, Target, Instacart

These companies have published recommendation and search work through engineering blogs and conference talks. Their specific architectures and numbers could not be checked against primary sources in this pass, so this section describes only the *design problems* that are specific to each business model and that their public work is known to address. Readers should treat it as an orientation and go to each company's engineering blog for details.

- **Wayfair (home goods).** Products are bulky, visually driven and bought rarely, so purchase-based CF signals are very sparse per user. This pushes the design toward visual similarity ("shop the look", visual search), style embeddings and session-based intent rather than long-term purchase history. The same reasoning motivates Zalando's Fashion DNA and Amazon's visual-style work [8][19].
- **Shopify (platform for millions of independent merchants).** The recommender has to serve many small stores, each with little data of its own. This favours models that are pretrained across merchants and use shared content representations (product text and images, product taxonomy), adapted per store. The structure is similar to a managed service like Amazon Personalize, with one important difference: the platform owns the cross-store representation.
- **Walmart and Target (omnichannel general retail).** Recommendations and search have to respect store-level inventory and fulfilment options (ship, pickup, delivery). The candidate set is filtered by availability at the shopper's location, not just by global stock. Hybrid lexical-plus-semantic retrieval (as in Sections 3 and 7) is the norm for search, and complement recommendations ("goes with") matter in grocery and household categories.
- **Instacart (grocery delivery across many retailers).** Grocery is dominated by **repeat purchase** ("buy it again") and **replacement** (substitutes when an item is out of stock at the chosen store). Availability is uncertain and changes by the hour per store, so availability prediction acts as a hard filter or as a score multiplier. Basket completion (complements) and substitution (substitutes) are the ESCI distinction again, applied to fulfilment.

All five share a pattern: the **business constraint layer** (inventory, fulfilment, merchant boundaries, availability) becomes as important as the relevance model. It has to sit inside the serving path, not in an offline post-filter.

---

## 10. A reference architecture for e-commerce recommendation

Putting the published designs together gives a typical modern commerce recommender, as sketched here:

```
                         +---------------------------------------------+
  events (view, cart,    |  streaming layer: session features,          |
  purchase, search) ---->|  real-time user embedding, cart state        |
                         +---------------------------------------------+
                                     |
  request(user, page, query?, cart)  v
  ------------------------------> candidate generation (union)
          - item-to-item tables (co-view: substitutes; co-buy: complements)
          - two-tower ANN (content-rich item tower; sequence user tower)
          - query-conditioned semantic + lexical retrieval
          - complement-type expansion (type -> items)
          - popularity / trending / exploration slots
                                     |
                                     v
                business filters: stock, delivery region, store availability,
                already-owned, policy, merchant/seller constraints
                                     |
                                     v
                ranker: multi-task (click, add-to-cart, purchase, return?)
                features: user x item x query x context; price, deal, rating
                                     |
                                     v
                re-rank: diversity (by product type), dedupe variants,
                promotions, explanations ("because you bought X")
                                     |
                                     v
                widgets: "Frequently bought together", "Customers also viewed",
                "Buy it again", "Recommended for you", search results
```

Design choices that recur across the published systems:

1. **Separate item-relation types.** Substitutes (co-view, same type) and complements (co-buy minus co-view, different type) need different data and different widgets [4][7][10][13].
2. **Correct for popularity.** Use expected-co-occurrence or lift-style normalisation, otherwise bestsellers take over every list [4].
3. **Generalise through content in the item tower.** Churn and one-of-a-kind items make ID-only models fragile [12][18][19][20][21].
4. **Make retrieval hybrid.** Lexical matching for precision, semantic matching for recall, graph or behaviour data for popularity and taste [12][21].
5. **Be reactive.** The cart and current session should change recommendations immediately, through lookups or real-time user embeddings [1][14][20].
6. **Diversify by type.** For complements, predict a diverse set of product types first [10].
7. **Explore.** Reserve slots for new items, because commerce catalogs churn [14].

---

## 11. Key takeaways for system designers

- **Fit the algorithm to the serving constraints.** Item-to-item CF did not win because it was the most accurate model in 2003. It won because its online cost does not depend on catalog or customer counts, it reacts immediately to the cart, and it explains itself [1].
- **Normalise co-occurrence by its expectation.** Raw co-purchase counts mostly measure popularity. Smith and Linden's expected-count correction is a cheap, high-value fix that still applies to embedding training (popularity-corrected negatives) [4].
- **Model relation types explicitly.** "Similar" and "goes with" are different products. Use co-view for substitutes and co-purchase-minus-co-view for complements, and present them in different widgets. ESCI labels make this distinction explicit in search [10][13].
- **Search is your largest recommender surface.** A query-aware system that combines lexical, semantic and personalised retrieval, and then expands through item graphs, covers more commerce intent than any home-page module [12][13][21].
- **Simple text encoders with good tokenisation go a long way.** N-grams, character trigrams and hashed OOV tokens with average pooling were competitive with heavier encoders for product search at Amazon scale [12].
- **Cold start is the default.** Marketplaces (eBay, Etsy) and seasonal catalogs (fashion) need content-driven item towers or side-information embeddings, so new items are retrievable on day one [18][19][20][21].
- **Personalise retrieval, not just ranking**, especially for vague queries [21].
- **Treat business constraints as first-class serving components.** Inventory, availability, fulfilment and merchant boundaries belong in the serving path, and services like Personalize expose filters and promotions as core APIs [14].
- **Separate the task contract from the model.** Personalize keeps stable task types (user top-N, re-rank, related items) while the model behind them moved from HRNN to transformers. Your internal platform should be built the same way [14].

---

## References

1. Linden, G., Smith, B., York, J. "Amazon.com Recommendations: Item-to-Item Collaborative Filtering." IEEE Internet Computing, 2003. https://www.cs.umd.edu/~samir/498/Amazon-Recommendations.pdf
2. Linden, G. D., Jacobi, J. A., Benson, E. A. "Collaborative recommendations using item-to-item similarity mappings." US Patent 6,266,649, Amazon.com, 2001. https://patents.google.com/patent/US6266649B1/en
3. Sarwar, B., Karypis, G., Konstan, J., Riedl, J. "Item-based Collaborative Filtering Recommendation Algorithms." WWW 2001, University of Minnesota. https://dl.acm.org/doi/10.1145/371920.372071
4. Smith, B., Linden, G. "Two Decades of Recommender Systems at Amazon.com." IEEE Internet Computing, 2017. https://doi.org/10.1109/MIC.2017.72
5. Amazon Science. "The history of Amazon's recommendation algorithm." Amazon Science, 2019. https://www.amazon.science/the-history-of-amazons-recommendation-algorithm
6. Amazon. "DSSTNE: Deep Scalable Sparse Tensor Network Engine." GitHub (amzn/amazon-dsstne), 2016. https://github.com/amzn/amazon-dsstne
7. McAuley, J., Pandey, R., Leskovec, J. "Inferring Networks of Substitutable and Complementary Products." KDD 2015. https://arxiv.org/abs/1506.08839
8. McAuley, J., Targett, C., Shi, Q., van den Hengel, A. "Image-based Recommendations on Styles and Substitutes." SIGIR 2015. https://arxiv.org/abs/1506.04757
9. Hou, Y., Li, J., He, Z., Yan, A., Chen, X., McAuley, J. "Bridging Language and Items for Retrieval and Recommendation" (BLaIR, Amazon Reviews 2023). UCSD, 2024. https://arxiv.org/abs/2403.03952
10. Hao, J., Zhao, T., Li, J., Dong, X. L., Faloutsos, C., Sun, Y., Wang, W. "P-Companion: A Principled Framework for Diversified Complementary Product Recommendation." Amazon, CIKM 2020. https://dl.acm.org/doi/10.1145/3340531.3412732
11. Jin, W., et al. "Amazon-M2: A Multilingual Multi-locale Shopping Session Dataset for Recommendation and Text Generation." Amazon, NeurIPS Datasets and Benchmarks 2023 / KDD Cup 2023. https://arxiv.org/abs/2307.09688
12. Nigam, P., Song, Y., Mohan, V., Lakshman, V., Ding, W., Shingavi, A., Teo, C. H., Gu, H., Yin, B. "Semantic Product Search." Amazon, KDD 2019. https://arxiv.org/abs/1907.00937
13. Reddy, C. K., Marquez, L., Valero, F., Rao, N., Zaragoza, H., Bandyopadhyay, S., Biswas, A., Xing, A., Subbian, K. "Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product Search." Amazon, 2022. https://arxiv.org/abs/2206.06588
14. Amazon Web Services. "Amazon Personalize Developer Guide: Choosing a recipe / predefined recipes." AWS Documentation, 2019-2025. https://docs.aws.amazon.com/personalize/latest/dg/working-with-predefined-recipes.html
15. Quadrana, M., Karatzoglou, A., Hidasi, B., Cremonesi, P. "Personalizing Session-based Recommendations with Hierarchical Recurrent Neural Networks." RecSys 2017. https://arxiv.org/abs/1706.04148
16. Hidasi, B., Karatzoglou, A., Baltrunas, L., Tikk, D. "Session-based Recommendations with Recurrent Neural Networks" (GRU4Rec). ICLR 2016. https://arxiv.org/abs/1511.06939
17. Grbovic, M., Radosavljevic, V., Djuric, N., Bhamidipati, N., Savla, J., Bhagwan, V., Sharp, D. "E-commerce in Your Inbox: Product Recommendations at Scale." Yahoo Labs, KDD 2015. https://arxiv.org/abs/1606.07154
18. Wang, J., Huang, P., Zhao, H., Zhang, Z., Zhao, B., Lee, D. L. "Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba." Alibaba, KDD 2018. https://arxiv.org/abs/1803.02349
19. Bracher, C., Heinz, S., Vollgraf, R. "Fashion DNA: Merging Content and Sales Data for Recommendation and Article Mapping." Zalando Research, 2016. https://arxiv.org/abs/1609.02489
20. Wang, T., Brovman, Y. M., Madhvanath, S. "Personalized Embedding-based e-Commerce Recommendations at eBay." eBay, 2021. https://arxiv.org/abs/2102.06156
21. Jha, R., Subramaniyam, S., Benjamin, E., Taula, T. "Unified Embedding Based Personalized Retrieval in Etsy Search." Etsy, KDD 2023 (applied track). https://arxiv.org/abs/2306.04833
