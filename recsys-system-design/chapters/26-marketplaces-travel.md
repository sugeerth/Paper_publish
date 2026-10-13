# 26. Marketplace and Travel Recommenders: Airbnb, Booking.com, Uber Eats, DoorDash and Others

Marketplace recommenders differ from media recommenders in a few ways that shape the whole
design. The "item" is a physical supply unit that someone else owns: a listing, a hotel
room, a restaurant, a driver. Supply is limited, tied to a place, and often tied to a date.
A recommendation is only useful if it can actually be fulfilled: the listing is free for
those dates, the restaurant is open and within delivery range, the host will accept. The
label we most care about (a booking or an order) is rare, comes late, and can be undone by
cancellations or host rejections. Every ranking change also moves demand between sellers,
which creates fairness concerns and makes A/B tests interfere with each other.

This chapter covers the publicly documented designs from these companies. Airbnb has the
richest public record: its KDD 2018 listing embeddings, the KDD 2019 and 2020 papers on deep
learning for search, the multi-task "journey" ranker, learning to rank diversely, and
embedding-based retrieval in 2025-2026. Booking.com's "150 successful models" paper is the
best-known statement of what goes wrong between offline metrics and business value. Uber
Eats contributes GNN features, a two-tower homefeed retriever and the Michelangelo platform.
DoorDash contributes store embeddings, explore/exploit on the homepage and its feature
platforms (Riviera, Fabricator, a Redis feature store). We also briefly cover Lyft, Expedia,
Zillow, Yelp and Instacart, mostly to describe the patterns they share.

**A note on sourcing.** This chapter was written in an environment where the web-search
budget was used up and outbound fetches to arXiv, Medium and company blogs were blocked by
network policy. Facts are therefore taken from (a) sources that were fetched and checked
while sibling chapters of this atlas were researched, and (b) well-established primary papers
whose content is widely cited. References that could not be re-fetched are marked "not
re-verified" in the reference list. Numbers from those sources are given as approximate and
should be checked against the original before you quote them.

---

## 26.1 What makes marketplace recommenders different

| Dimension | Media feed (e.g. video) | Marketplace / travel |
|---|---|---|
| Item | Unlimited copies of a piece of content | Unique, capacity-limited supply unit |
| Validity | Almost always valid | Availability, dates, delivery radius, opening hours |
| Primary label | Watch time, engagement (dense) | Booking / order (sparse, delayed, reversible) |
| Session length | Minutes | Days to weeks (travel); minutes (food) |
| Geo constraint | Weak | Hard (city, market, delivery zone) |
| Other side | Creators, loosely coupled | Hosts / merchants / drivers who can reject |
| Experiment interference | Low | High: treatment and control compete for supply |
| Repeat behaviour | Varies | Low for travel; very high for food and grocery |

Three consequences come up again and again in the systems below:

1. **Geography partitions the problem.** Retrieval is almost always geo-filtered first, and
   negatives must be sampled from the same market, or the model only learns geography.
2. **The label is a funnel.** Click, then detail view, then checkout, then booking, then not
   cancelled. Multi-task and staged models that use the denser upper-funnel signals are
   standard.
3. **Offline metrics lie more often.** Booking.com and Airbnb both report many cases where
   offline gains did not show up as business gains. Online experiments, often with
   interleaving or switchbacks, are the final judge.

---

## 26.2 Airbnb search ranking

Airbnb search takes a query (location, dates, number of guests, filters) and returns a
ranked list of listings that are available for those dates. The public record covers the
system end to end, from the GBDT era to a multi-stage deep-learning stack.

### 26.2.1 Overall architecture

```
 guest query (place, dates, guests, filters)
        |
        v
 +--------------------------+     availability, price, geo filters
 | Retrieval                |---> (inverted index; since 2025 also
 |  - geo + availability    |      two-tower EBR with IVF clusters as filters [8][9])
 +--------------------------+
        |  candidates (order of thousands, approximate)
        v
 +--------------------------+     features: listing, query, guest, real-time
 | First-pass / base ranker |     session features from embeddings [1]
 |  DNN (pairwise/listwise) |     (journey ranker: multi-task base module [5])
 +--------------------------+
        |
        v
 +--------------------------+
 | Second pass / context    |     diversity given listings above [6],
 | twiddlers, business rules|     cancellation / rejection penalties [5]
 +--------------------------+
        |
        v
  ranked results  ---> logs (impressions, clicks, bookings) ---> training
                         \--> interleaving experiments [10]
```

### 26.2.2 Real-time personalization with listing embeddings (KDD 2018)

**Problem.** Ranking should react to what the guest is doing in the current session. A guest
who clicks on several cheap, modern apartments near the city centre should see more of
those, right away, without retraining a model.

**Design.** Grbovic and Cheng trained word2vec-style skip-gram embeddings on search click
sessions [1]. Each session is a sequence of listing IDs that a user clicked, and sessions are
split when there is a gap of more than 30 minutes between clicks [1][7]. Figures reported in
secondary summaries of the paper (approximate): about 800 million click sessions, about 4.5
million active listings, and 32-dimensional embeddings [1][11].

Two changes to plain skip-gram matter most:

- **Booked listing as global context.** In sessions that end in a booking, the booked
  listing is added as a context item for every listing in the session, whether or not it
  falls inside the sliding window. This pulls the embedding towards "what people who looked
  at this eventually booked", not only towards "what was clicked next to it".
- **Market-aware negatives.** Random negatives almost always come from another city, which
  makes them trivially easy. Airbnb adds negatives sampled from the same market as the
  central listing, so the model must learn differences within a market (style, price,
  neighbourhood), not just geography.

**Cold start.** A new listing has no clicks. Its embedding is set to the average of the
embeddings of about three nearby listings of the same type and similar price, as described
in the paper (approximate).

**Long-term interest: type embeddings.** Bookings are too sparse to learn per-user and
per-listing embeddings from booking sequences directly. Airbnb therefore maps users and
listings to coarse **types**, built from bucketed attributes such as market, price, room
type and capacity for listings, and similar buckets for users. User-type and listing-type
embeddings are learned in the same vector space from booking sequences. Host rejections are
used as explicit negatives in this training, so the embedding also captures "this kind of
guest tends to be rejected by this kind of listing".

**How it is used.**

- *Similar listings carousel.* K nearest neighbours by cosine similarity, filtered by
  availability. Secondary summaries report about a 21% CTR increase over the previous
  similar-listings algorithm [11].
- *Real-time ranking features.* Kafka streams record the user's clicked listings, skipped
  listings, long clicks, wishlisted listings and contacted listings over roughly the last
  two weeks. For each candidate, features such as "similarity to clicked listings" and
  "similarity to skipped listings" are computed online and fed to the GBDT search ranker
  that was in production at the time. The paper reports a booking gain from these features
  in an online test, and the embedding similarity features rank high in feature importance.

**Lessons.** (1) Cheap embeddings and simple online similarity features can give most of
the benefit of session personalization, with no online model training. (2) Change the
objective to match the business event (booking as global context). (3) Sample negatives
from the same constraint set the user actually chooses within.

### 26.2.3 Moving to deep learning (KDD 2019)

"Applying Deep Learning to Airbnb Search" [2] is unusually candid. It reads as a log of what
the team tried, what failed and why.

**The trajectory** (as described in the paper):

1. A GBDT model with heavy hand-made features had become hard to improve.
2. A simple neural network with one hidden layer (32 ReLU units) reached about parity with
   GBDT.
3. Training with a **LambdaRank-style pairwise loss** (booked listing versus not-booked
   listings in the same search, weighted by the NDCG change of a swap) helped.
4. An intermediate step combined GBDT and factorization-machine outputs as inputs to the
   neural network.
5. The final step was a **deeper network with 2 hidden layers** and about 195 features,
   trained on the order of 1.7 billion pairs (figures approximate, as reported in [2]).
   This model let the team stop much of the manual feature engineering.

**Things that failed** (and the reasons the paper gives):

- **Listing ID embeddings.** They overfit badly. Even a popular listing can be booked at
  most about 365 times a year, so there is very little data per ID compared with words in
  a language model.
- **Multi-task learning with long views.** Adding a "long view" (dwell) task made views go
  up, but bookings did not follow. The model learned to promote listings that were nice to
  look at, such as expensive or unusual homes, that people did not book.
- **Features that are proxies for the label.** Several offline NDCG wins did not reproduce
  online, which taught the team to be suspicious of offline gains.

**Engineering practices that worked:**

- Normalize features so they look roughly normal: log-transform heavy-tailed features such
  as price, and express geo coordinates relative to the centre of the map viewport rather
  than as raw latitude and longitude.
- Check feature distributions and plot model output as a function of single features, to
  catch features that the network uses in surprising ways.
- Accept that the deep model is harder to explain, and invest in debugging tools.

**Lesson.** Moving from GBDT to DNN was not one model change. It was a journey of many
small steps, and each needed an online test. The biggest gains came from scale (more
training data, fewer hand-built features) once the basics (normalization, a pairwise loss)
were right.

### 26.2.4 Architecture, bias and cold start (KDD 2020)

"Improving Deep Learning for Airbnb Search" [3] groups its changes under three headings.

**Architecture: a two-tower "ideal listing" model.** The team found that the DNN had learned
a strong "cheaper is better" bias. Many guests have a preference for a price that is not
the lowest. The new design has a **query/user tower** that produces an "ideal listing"
vector from the query and the guest, and a **listing tower** that produces a vector for each
listing. The score depends on the distance between the two. This makes the model favour
listings close to what this guest wants, rather than ones that maximize a single monotone
direction such as low price. The paper reports a booking gain online.

**Bias: position as a feature, with dropout.** Bookings depend strongly on rank position.
Airbnb adds the listing's position as an input feature during training and applies dropout
to that feature. At inference time the position is set to a constant (zero), so every
listing is scored as if shown in the same position. The dropout rate trades off two
failures. With too little dropout the model leans on position. With too much it cannot
separate the effect of position from the effect of quality. The tuned version gave a
statistically significant booking gain [3].

**Cold start: estimate engagement for new listings.** New listings have missing or noisy
engagement features (for example historical booking rate), so the DNN ranks them low,
which means they get fewer impressions and stay data-poor. Airbnb replaces the defaults
with **estimates computed from similar nearby listings** (same geography, similar capacity
and so on). The paper reports a large relative booking gain for new listings (roughly
double-digit percent; around 14% as commonly quoted, approximate) and a small but
significant overall booking gain [3].

```
   query + guest features            listing features (+ position at train time)
            |                                   |
     [ query tower ]                    [ listing tower ]
            |                                   |
       "ideal listing"  ----- distance -----  listing vector
                                |
                       score -> pairwise loss
   at serving: position feature := 0 ; new-listing engagement := neighbour estimate
```

### 26.2.5 Learning to rank diversely (2022, CIKM 2023)

**Problem.** A ranker that scores each listing on its own puts similar listings next to each
other: same price range, same neighbourhood, same style. A guest whose taste differs from
the average then finds nothing to book on the first page.

**Design.** Abdool et al. [6] change training and inference so that a listing's score
depends on the listings already ranked above it. In effect, the model learns "what is the
chance this listing gets booked, given what the guest has already seen above". Listings
that look like the ones above are worth less. Ranking is done greedily, position by
position. The paper reports an online booking gain.

**Lesson.** In a marketplace, diversity is not a nice-to-have. It is a way to cover the
spread of guest preferences when the ranker cannot read them from the query.

### 26.2.6 Journey ranker: multi-task learning along the booking funnel (KDD 2023)

**Problem.** A booking is the end of a long journey: search, click, view the listing page,
go to the payment page, request or instantly book, the host accepts, and the trip is not
cancelled. Training only on bookings throws away the denser signals before it. Training on
clicks alone optimizes the wrong thing.

**Design** (Tan et al., "Optimizing Airbnb Search Journey with Multi-task Learning" [5]).
The Journey Ranker is built from modules:

- **Base module.** A multi-task network that predicts each step of the positive funnel
  (click, listing page, payment page, request, booking). The final score combines the
  conditional probabilities along the funnel, so dense early steps help the model learn
  while the target is still the booking.
- **Twiddler module.** Predicts negative milestones such as cancellations and host
  rejections, and uses them to adjust the score. This keeps the ranker from sending guests
  to listings that tend to fall through.
- **Context module.** Lets the combination weights depend on context, for example how far
  ahead the trip is or what kind of guest is searching.
- **Monotonic module.** Enforces monotonic relationships for some features (for example,
  that a better value on a quality signal should never lower the score).

The paper reports online booking gains and says the design was reused for other Airbnb
search surfaces (approximate; see [5]).

**Lesson.** Model the funnel explicitly, including the negative outcomes that only the
marketplace sees (rejections, cancellations). A booking that is later cancelled is not a
success.

### 26.2.7 Embedding-based retrieval (2025-2026)

Airbnb's most recent public work adds a learned retrieval stage [8][9].

- **Model.** Two-tower. The listing tower is run offline in a daily batch. The query tower
  runs online for each search.
- **Training data.** Built from guest search journeys, with **session-based hard negatives**
  (listings the guest saw in the session but did not book) instead of random negatives.
- **ANN choice.** HNSW gave slightly better recall in evaluation, but Airbnb chose **IVF**.
  Prices and availability change all the time, and HNSW's memory overhead on frequent
  updates was a problem. With IVF, a listing's cluster assignment becomes an ordinary
  filter in the existing search system, so retrieval is "match the query to clusters, then
  filter by cluster plus the usual availability filters" [8].
- **Result.** The paper reports significant booking gains [9].

**Lesson.** When items change often and retrieval must respect hard filters, an index that
fits into the existing filter engine (IVF clusters) can beat a slightly more accurate graph
index (HNSW).

### 26.2.8 Experimentation for search: interleaving

Airbnb adapted team-draft interleaving to a two-sided marketplace with sparse, delayed
bookings [10]. Only "competitive pairs", the positions where the two rankers actually
differ, are used for attribution, and searches where both rankers agree are dropped. Airbnb
reports roughly a 50x gain in sensitivity compared with A/B tests, and interleaving results
agreed in direction with later A/B tests in most paired comparisons (approximate) [10].
Interleaving is used to screen ranking ideas quickly; A/B tests still make the final call.

### 26.2.9 Data platform: Zipline to Chronon

Airbnb's feature platform Zipline was rebuilt and open-sourced as **Chronon**, which is
maintained jointly with Stripe and is described as powering all major ML applications at
Airbnb [12]. It lets engineers define features in Python (`GroupBy`, `Join`) once. The same
definition is used for point-in-time-correct backfills in Spark and for online serving with
streaming updates. This removes the main source of train-serve skew for real-time features
such as the session embedding similarities in 26.2.2. Chapter 8 covers Chronon in detail.

---

## 26.3 Booking.com: 150 models and 6 lessons

Bernardi, Mavridis and Estevez's KDD 2019 paper [4] is not about one model. It summarizes
what Booking.com learned from about 150 customer-facing ML models in production. It is one
of the most quoted papers on applied recommender systems.

### 26.3.1 Model families

The paper groups the models into families (paraphrased):

| Family | What it predicts | Example use |
|---|---|---|
| Traveller preferences | How flexible the user is about dates or destination | Show alternative dates or places |
| Traveller context | Trip type (family, business, couple) | Adapt filters and messaging |
| Item space navigation | Which items or destinations the user will engage with | Recommendations, sorting |
| User interface optimization | Best layout, font size, number of items | UI parameters |
| Content curation | Which reviews, photos or descriptions to show | Review highlights |
| Content augmentation | Derived facts about properties | "Great value today", price trend |

Most of these are not "rank the hotels" models. They are small predictions that each shape
part of the experience, and their combined effect is the product.

### 26.3.2 The six lessons

1. **ML projects deliver value.** Across families, models gave measurable business gains in
   randomized experiments. Even "semantic" models of hidden concepts (such as how flexible a
   user is) worked as building blocks for many features.
2. **Model performance is not business performance.** Better offline metrics often gave no
   business gain. Reasons the paper gives: the value is already saturated, the model
   optimizes a proxy that diverges from conversion, and an "uncanny valley" effect where a
   prediction that is too accurate feels intrusive to users.
3. **Be on the lookout for the problem to solve.** Framing the problem well (for example as
   regression versus classification, or picking the right label) mattered more than the
   learner.
4. **Prediction serving latency matters.** In a controlled experiment, adding about 30% to
   latency cost roughly 0.5% in conversion (approximate, as reported). Tricks the paper
   lists: model redundancy (many replicas near the caller), in-house linear and Naive Bayes
   prediction, sparse models, precomputing and caching, bulking requests, and keeping
   feature transformation small.
5. **Get early feedback on model quality.** Labels arrive late or never. Booking.com uses
   **Response Distribution Analysis**: look at the shape of the model's output distribution
   on live traffic. A healthy binary classifier shows a clear split; a flat or one-peaked
   histogram is an early warning.
6. **Test business impact with randomized controlled trials.** Everything goes through A/B
   tests. The paper describes designs for cases where the model only applies to part of the
   traffic, for example triggering analysis that compares only users who would have been
   affected, and a three-arm design to separate the effect of the model from the effect of
   showing any change at all.

### 26.3.3 Lessons for system designers

- Treat offline metrics as a health check, not as the goal.
- Latency is a business metric: budget it like conversion.
- Build monitoring that works without labels.

---

## 26.4 Expedia and the hotel-ranking problem

Expedia's best-known public contribution is a benchmark rather than a production system
write-up. The 2013 Kaggle competition "Personalize Expedia Hotel Searches" released search
logs with hotel features, user history, price competitiveness, position and click/booking
labels [13]. Its structure is still a good model of the hotel-ranking problem:

- Each search returns a list of hotels. The labels are graded (booked > clicked > shown),
  and the metric is NDCG@38.
- The data contains both "random" and normally ranked result pages, which allowed position
  bias to be studied.
- Winning solutions were mostly gradient-boosted LambdaMART models with heavy feature
  engineering: price normalized within the search, hotel historical CTR and booking rate,
  and the difference between the user's past average price or star rating and the hotel's.

The patterns it shows (graded labels, normalizing features within the query, comparing the
item to the user's history) also appear in Airbnb's GBDT era and in Booking.com's work.
Expedia Group has published further work on search ranking and travel recommendations on
its engineering blog, but those posts could not be verified for this chapter.

---

## 26.5 Uber Eats

Uber Eats is a three-sided marketplace (eaters, merchants, couriers) with very strong
locality: an eater can only order from restaurants in delivery range. Roughly 95% of users
start their journey on the Homefeed, according to an Uber talk [16], so homefeed ranking is
central.

### 26.5.1 Graph learning for dish and restaurant recommendations (2019)

**Problem.** Orders are sparse and local. Each eater orders from a small set of nearby
restaurants, so standard collaborative filtering has little to work with [14].

**Design** [14]:

- **Graphs.** Bipartite user-dish and user-restaurant graphs, built **per city**. Delivery
  range makes edges across cities almost meaningless. Edge weights encode how often the
  user ordered.
- **Model.** GraphSAGE with sampled neighbourhoods and node features (dish and restaurant
  descriptors).
- **Loss.** A max-margin ranking loss, extended with "low-rank positives". A dish ordered
  many times should score above a dish ordered once, which should score above a negative.
  The edge weights are therefore used in the loss as well as in sampling.
- **Pipeline.** Embeddings are trained and generated offline every day on Uber's ML
  platform. Similarity between user and dish or restaurant embeddings is fed as **features**
  into the downstream ranking model and also used for candidate retrieval.

**Results (approximate).** The blog reports a clear offline AUC gain for the ranker when the
graph features were added, with the graph similarity among the most important features,
and CTR gains in online tests in the first cities [14].

```
 orders (per city) -> weighted bipartite graphs (user-dish, user-restaurant)
      -> GraphSAGE (sampled 2-hop, max-margin + low-rank positives), daily
      -> user / dish / restaurant embeddings
      -> similarity features ---> homefeed ranker ---> carousels and store list
```

**Lesson.** Splitting the graph by city both scales the training and encodes a real
constraint. A GNN does not have to be the recommender: its embeddings used as features in an
existing ranker are often the safest way to ship.

### 26.5.2 Two-tower embeddings for homefeed retrieval

**Problem.** The earlier first-pass ranker was a Deep Matrix Factorization model trained
**per city**. As Uber expanded, this meant thousands of weekly Spark jobs and did not scale
[15].

**Design.** A single global two-tower model [15]:

- The **store tower** runs offline. Store embeddings are indexed in Uber's search platform
  (SIA) for approximate nearest neighbour search.
- The **eater tower** runs in real time for every Homefeed request, followed by an ANN
  lookup against the store index, with location filters.
- Cost drops from O(q x M) online inference (q requests, M stores) to O(M) offline plus O(q)
  online.

A Feature Store Summit talk gives the latency target: find the best stores out of thousands
in about **50 ms** [16].

### 26.5.3 Eats search pipeline: hydration is the hidden cost

A 2026 Uber post on the Eats search pipeline [17] reports that retrieval was returning tens
of thousands of candidates, all of which were fully hydrated (all their data loaded) before
the ranker discarded most of them. Removing low-yield retrieval paths saved about 120 ms.
Splitting hydration into a light phase for ranking and a full phase for the results that
are actually shown saved more than 100 ms more. The first-pass ranker has to stay fast
"while maintaining a high recall metric" [17].

```
 eater request
    |
    v
 retrieval paths (lexical, embedding/ANN, geo)  --> tens of thousands of candidates
    |                                                 (before the fix)
    v
 light hydration (ranking features only)
    |
    v
 first-pass ranker (high recall, low latency)
    |
    v
 second-pass ranker (multi-objective)
    |
    v
 full hydration (presentation data) only for shown results
```

### 26.5.4 Michelangelo

Michelangelo, Uber's ML platform [18], is the infrastructure under these models. Its 2017
description covers the full lifecycle: manage data, train, evaluate, deploy, predict and
monitor. Two ideas from it are now everywhere:

- A **feature store** with an offline store (Hive) for training and an online store
  (Cassandra) for serving, fed by batch (Spark) and streaming (Kafka plus Samza, later
  Flink) pipelines. Features are shared across teams. The 2017 post mentions about 10,000
  features in the store (approximate).
- **Transformations inside the model artifact** (a DSL for selecting and transforming
  features), so the same code runs at training and at serving time.

Michelangelo **Palette** (2019) [19] extended the feature store into a feature engineering
platform with a catalog, ownership, and support for batch, near-real-time and RPC feature
sources.

### 26.5.5 Experimentation under marketplace interference

Uber's experimentation platform supports A/B tests, multi-armed bandits, **switchback**
experiments and causal inference methods [20]. Switchbacks are the key tool for
marketplaces: instead of splitting users, the whole market (a city for a time window) is
switched between treatment and control, so treatment and control do not compete for the
same couriers or restaurants. The platform runs on the order of 1,000 or more concurrent
experiments (approximate) and uses delta-method statistics, sequential tests and CUPED-style
variance reduction [20].

---

## 26.6 DoorDash

DoorDash's public writing focuses on the homepage store feed, explore/exploit, and the
feature and prediction infrastructure behind them.

### 26.6.1 Store embeddings for the personalized store feed (2018)

DoorDash's early store-feed personalization trained **store2vec**: word2vec-style embeddings
where each "sentence" is the sequence of stores a user viewed in a session [21]. A user
vector is built by averaging the vectors of stores the user ordered from. The cosine
similarity between the user vector and a candidate store becomes a feature in the store
ranking model. This is the same pattern as Airbnb's listing embeddings (26.2.2): cheap
sequence embeddings, used as features, not as the ranker itself. The post reports gains in
click-through and conversion in experiments (details approximate) [21].

### 26.6.2 Homepage ranking with exploitation and exploration (2022)

**Problem.** A pure exploitation ranker shows the same familiar stores and cuisines. New or
rarely shown merchants never collect enough data to rank well, and users who would like
something different never see it.

**Design** [22], as described:

- A personalized ranking model predicts how likely the user is to engage with each store or
  cuisine (exploitation).
- An **upper-confidence-bound (UCB) style bonus** is added for items with few impressions
  for this user, so uncertain items get some exposure (exploration).
- The two terms are combined into the final score, with a weight that sets how much to
  explore.

The post reports engagement gains in A/B tests and better discovery of new merchants
(approximate).

### 26.6.3 Feature and prediction infrastructure

- **Sibyl** is DoorDash's prediction service. It serves many models with batched feature
  fetches and is built to keep latency predictable [23].
- **Gigascale feature store on Redis (2020).** DoorDash benchmarked Redis against other
  stores and used binary serialization and compression to store billions of features
  cheaply, with large savings in memory and CPU (figures approximate) [24].
- **Riviera (2021)** is a declarative framework for **real-time** features on Flink:
  features are defined in configuration and SQL, and the framework builds the streaming job
  [25].
- **Fabricator (2022)** is a declarative framework for **batch** feature engineering. A
  feature definition (sources, SQL, schedule, sinks) generates the pipelines, the backfills
  and the upload to the online store, which shortens the time from a feature idea to
  production [26].

```
 events (orders, views, ETAs) --Kafka--> Riviera (Flink, real-time features) --+
 warehouse tables -----------> Fabricator (batch, declarative)  ----------------+
                                                                                v
                                                    online feature store (Redis)
                                                                                |
 homepage request --> candidate stores (geo, open now) --> Sibyl prediction -----+
                                                           (store ranking + UCB)
                                                                  |
                                                                  v
                                               carousels + store list on homepage
```

---

## 26.7 Lyft

Lyft's public ML work is mostly about dispatch, pricing and ETA, not recommendation lists,
but its feature serving design is a typical marketplace one. Lyft describes a feature
serving system that serves both batch-computed and streaming features with low latency, and
that keeps training and serving consistent by logging served values [27]. For marketplace
recommenders the lesson is that ranking, pricing and dispatch draw on the same real-time
supply and demand features, so a shared feature platform pays off across all of them.

---

## 26.8 Zillow, Yelp and Instacart: patterns only

These three companies have published recommender and search work on their engineering
blogs. The specific posts could not be fetched for this chapter, so this section covers
only the design problems their domains create, without company-specific numbers.

- **Real estate (Zillow).** Each home is unique and sells once. Interaction data per item is
  very thin, so content features (location, price, size, photos) and user preference
  profiles (saved searches, favourites) carry most of the weight. Items leave the catalogue
  quickly when they sell, which favours models that generalize from attributes over models
  that memorize IDs, and makes freshness of the index important.
- **Local business discovery (Yelp).** Ranking blends text relevance, distance, quality
  (ratings, review volume) and personal taste, and must avoid letting advertising reduce
  user trust. Review text is a rich content signal for both search and recommendations.
- **Grocery and retail media (Instacart).** Grocery is dominated by **repeat purchase**
  ("buy it again") and **replacements** when an item is out of stock. Availability changes
  by the hour per store, so predicted availability acts as a hard filter or as a score
  multiplier. Sponsored products are ranked inside organic results, which needs an auction
  that combines bids with predicted relevance and protects user experience. Chapter 16
  covers ads ranking in general.

---

## 26.9 Cross-company comparison

| System | Stage | Model | Key trick | Reported result | Ref |
|---|---|---|---|---|---|
| Airbnb listing embeddings (2018) | Features, similar listings | Skip-gram, d=32 | Booked listing as global context; in-market negatives | ~21% CTR, similar listings (approx.) | [1][11] |
| Airbnb DL search (2019) | Ranking | 2-layer DNN, pairwise loss | Feature normalization; drop ID embeddings | Bookings gain vs GBDT | [2] |
| Airbnb ABA (2020) | Ranking | Two-tower "ideal listing" | Position feature + dropout; new-listing priors | Bookings gain; new listings ~+14% (approx.) | [3] |
| Airbnb diversity (2022) | Re-ranking | Context-aware DNN | Score depends on listings above | Bookings gain | [6] |
| Airbnb Journey Ranker (2023) | Ranking | Multi-task modules | Funnel milestones + cancel/reject twiddler | Bookings gain | [5] |
| Airbnb EBR (2025-26) | Retrieval | Two-tower | Session hard negatives; IVF clusters as filters | Bookings gain | [8][9] |
| Booking.com (2019) | Whole product | ~150 models | Response distribution analysis; latency budget | ~30% latency -> ~0.5% conversion (approx.) | [4] |
| Uber Eats GNN (2019) | Features, retrieval | GraphSAGE per city | Weighted max-margin with low-rank positives | Offline AUC up; CTR up | [14] |
| Uber Eats TTE | Retrieval | Global two-tower | Store tower offline; SIA ANN | ~50 ms over thousands of stores | [15][16] |
| Uber Eats search | Pipeline | Multi-stage | Two-phase hydration | ~120 ms + 100+ ms saved | [17] |
| DoorDash store2vec (2018) | Features | Skip-gram on sessions | User vector = mean of ordered stores | CTR/conversion up (approx.) | [21] |
| DoorDash homepage (2022) | Ranking | Ranker + UCB | Exploration bonus for low-impression items | Engagement and discovery up (approx.) | [22] |

Platform view:

| Company | Feature platform | Serving | Experiments |
|---|---|---|---|
| Airbnb | Zipline -> Chronon (OSS, with Stripe) [12] | Search backend with filters + EBR | ERF, interleaving [10] |
| Uber | Michelangelo, Palette [18][19] | Michelangelo online prediction, SIA ANN | XP: A/B, switchbacks, bandits [20] |
| DoorDash | Riviera, Fabricator, Redis store [24][25][26] | Sibyl [23] | A/B and switchbacks (general) |
| Booking.com | In-house; latency-first serving [4] | Replicated, precomputed models | RCTs with triggering, 3-arm designs [4] |
| Lyft | Feature serving with logged values [27] | LyftLearn (general) | Switchbacks (general) |

---

## 26.10 Recurring design patterns

### Sequence embeddings as features first

Airbnb (listing2vec), DoorDash (store2vec) and Uber Eats (GraphSAGE) all started by
learning item embeddings from sessions or order graphs and feeding **similarities as
features** into an existing ranker. This is low-risk: the ranker keeps its other signals,
and the embedding can be refreshed daily without touching serving code.

### Constrained negative sampling

Airbnb's in-market negatives, its session hard negatives in EBR, and Uber's per-city graphs
are the same idea: negatives must come from the set the user really chose from. Otherwise
the model learns geography or availability, which filters already handle.

### Funnel-aware and outcome-aware labels

The booking is sparse, so the systems borrow signal from the funnel above it (clicks,
detail views, checkout) and correct for outcomes below it (cancellations, host rejections).
Airbnb's Journey Ranker is the clearest example [5]. Airbnb's 2019 failure with long-view
multi-task learning [2] is the warning: an upper-funnel task must be tied to the final
outcome, or the model will learn to show attractive things that do not convert.

### Cold start of supply is a business problem

New hosts and new merchants need impressions to survive. Airbnb estimates engagement from
similar nearby listings and removes position from scoring [3]. DoorDash adds UCB
exploration [22]. Both protect the supply side of the marketplace as well as improving
ranking quality.

### Filters before learning

Availability, dates, delivery radius and opening hours are hard constraints. The successful
designs keep them as filters in the retrieval engine, and choose ANN structures (Airbnb's
IVF) and partitions (Uber's per-city graphs, geo-filtered ANN) that work with those filters.

### Measure carefully, because the market is shared

Interleaving (Airbnb [10]), switchbacks (Uber [20]) and triggering and three-arm designs
(Booking.com [4]) all address the same issue: marketplace experiments are noisy and biased
by interference, and offline metrics are not reliable enough to decide alone.

---

## 26.11 Key takeaways for system designers

1. **Model the transaction, not the click.** Use booking or order as the target, the denser
   funnel steps as auxiliary tasks, and cancellations and rejections as negatives. Check
   that every auxiliary task actually moves the final outcome online.
2. **Geography and availability come first.** Partition by market or city, keep hard
   constraints as filters, and sample negatives inside the same constraints.
3. **Start with embeddings as features.** Session or graph embeddings fed into an existing
   GBDT or DNN ranker are cheap, fast to ship and easy to debug.
4. **Do not memorize IDs when data per item is thin.** Airbnb's listing ID embeddings
   overfit because each listing can be booked at most a few hundred times a year. Prefer
   attributes, types and content.
5. **Debias position and help new supply.** Position as a feature with dropout and
   constant-at-serving is a simple, effective trick. Estimate missing engagement for new
   items from similar neighbours, and keep an explicit exploration budget.
6. **Diversify the page.** When the query cannot tell you the guest's taste, cover the
   space. Make each listing's score depend on what is already above it.
7. **Pick the ANN index that fits the update pattern and filter engine,** not only the one
   with the best recall. Fast-changing prices and availability favoured IVF at Airbnb.
8. **Hydrate late.** Load only ranking features for the large candidate set and fetch
   presentation data only for the items you will show. Uber saved over 200 ms this way.
9. **Treat latency as a business metric.** Booking.com measured that added latency costs
   conversion. Precompute, cache, replicate and keep models small where possible.
10. **Trust online experiments, and design them for marketplaces.** Use interleaving to
    screen, switchbacks or market-level randomization for interference, label-free
    monitoring (response distribution analysis) for early warnings, and A/B tests for the
    final decision.
11. **Invest in one feature platform.** Chronon, Michelangelo/Palette and
    Riviera/Fabricator exist because real-time marketplace features (session clicks, open
    stores, courier supply) are where train-serve skew is most likely.

---

## References

1. Grbovic, M., Cheng, H. "Real-time Personalization using Embeddings for Search Ranking at
   Airbnb." KDD 2018 (Applied Data Science best paper), Airbnb, 2018.
   https://dl.acm.org/doi/10.1145/3219819.3219885
2. Haldar, M., Abdool, M., Ramanathan, P., et al. "Applying Deep Learning to Airbnb Search."
   KDD 2019, Airbnb, 2018/2019. https://arxiv.org/abs/1810.09591
3. Haldar, M., Ramanathan, P., Sax, T., et al. "Improving Deep Learning for Airbnb Search."
   KDD 2020, Airbnb, 2020. https://arxiv.org/abs/2002.05515
4. Bernardi, L., Mavridis, T., Estevez, P. "150 Successful Machine Learning Models: 6
   Lessons Learned at Booking.com." KDD 2019, Booking.com, 2019.
   https://dl.acm.org/doi/10.1145/3292500.3330744
5. Tan, C. H., et al. "Optimizing Airbnb Search Journey with Multi-task Learning." KDD 2023,
   Airbnb, 2023 (not re-verified). https://arxiv.org/abs/2305.18431
6. Abdool, M., Haldar, M., et al. "Learning To Rank Diversely At Airbnb." Airbnb, 2022 (CIKM
   2023). https://arxiv.org/abs/2210.07774
7. Airbnb Engineering. "Listing Embeddings in Search Ranking." Airbnb Tech Blog (Medium),
   2018 (not re-verified).
   https://medium.com/airbnb-engineering/listing-embeddings-for-similar-listing-recommendations-and-real-time-personalization-in-search-601172f7603e
8. Airbnb Engineering. "Embedding-Based Retrieval for Airbnb Search." Airbnb Tech Blog,
   2025. https://airbnb.tech/ai-ml/embedding-based-retrieval-for-airbnb-search/
9. Airbnb. "Applying Embedding-Based Retrieval to Airbnb Search." arXiv, 2026.
   https://arxiv.org/pdf/2601.06873
10. Airbnb Engineering. "Beyond A/B Test: Speeding up Airbnb Search Ranking Experimentation
    through Interleaving." Airbnb Tech Blog, 2022.
    https://medium.com/airbnb-engineering/beyond-a-b-test-speeding-up-airbnb-search-ranking-experimentation-through-interleaving-7087afa09c8e
11. BestPractice.AI. Case study of "Real-time Personalization using Embeddings for Search
    Ranking at Airbnb" (secondary summary), n.d.
    https://bestpractice.ai/ai-use-cases/case-studies/consumer-retail/airbnb-increased-similar-property-listing-recommendations-click-through-rates-by-21-with-machine-learning-embeddings
12. Airbnb and Stripe. "Chronon" (repository and documentation), 2024.
    https://github.com/airbnb/chronon
13. Expedia and Kaggle. "Personalize Expedia Hotel Searches - ICDM 2013" (competition and
    dataset), 2013 (not re-verified). https://www.kaggle.com/c/expedia-personalized-sort
14. Jain, A., Liu, I., Sarda, A., Molino, P. "Food Discovery with Uber Eats: Using Graph
    Learning to Power Recommendations." Uber Engineering Blog, 2019.
    https://www.uber.com/blog/uber-eats-graph-learning/
15. Uber Engineering. "Innovative Recommendation Applications Using Two Tower Embeddings at
    Uber." Uber Engineering Blog, 2023 (approx.).
    https://eng.uber.com/blog/innovative-recommendation-applications-using-two-tower-embeddings/
16. Kurra, D. D. "Uber - Embeddings." Feature Store Summit 2024 talk, Uber, 2024.
    https://content.hopsworks.ai/hubfs/Feature%20Store%20Summit%202024/FS%20Summit%2024%20-%20Uber%20-%20Embeddings.pdf
17. Uber Engineering. "Uber Eats Search Pipeline." Uber Blog, 2026.
    https://www.uber.com/us/en/blog/uber-eats-search-pipeline/
18. Hermann, J., Del Balso, M. "Meet Michelangelo: Uber's Machine Learning Platform." Uber
    Engineering Blog, 2017 (not re-verified).
    https://eng.uber.com/michelangelo-machine-learning-platform/
19. Uber. "Michelangelo Palette: A Feature Engineering Platform at Uber." InfoQ presentation,
    2019 (not re-verified). https://www.infoq.com/presentations/michelangelo-palette-uber/
20. Uber Engineering. "Under the Hood of Uber's Experimentation Platform." Uber Engineering
    Blog, 2018. https://www.uber.com/blog/xp/
21. DoorDash Engineering. "Personalized Store Feed with Vector Embeddings." DoorDash
    Engineering Blog, 2018 (not re-verified).
    https://doordash.engineering/2018/04/02/personalized-store-feed-with-vector-embeddings/
22. DoorDash Engineering. "Homepage Recommendation with Exploitation and Exploration."
    DoorDash Engineering Blog, 2022 (not re-verified).
    https://doordash.engineering/2022/10/05/homepage-recommendation-with-exploitation-and-exploration/
23. DoorDash Engineering. "Meet Sibyl - DoorDash's New Prediction Service." DoorDash
    Engineering Blog, 2020 (not re-verified).
    https://doordash.engineering/2020/06/29/doordashs-new-prediction-service/
24. DoorDash Engineering. "Building a Gigascale ML Feature Store with Redis, Binary
    Serialization, String Hashing, and Compression." DoorDash Engineering Blog, 2020 (not
    re-verified). https://doordash.engineering/2020/11/19/building-a-gigascale-ml-feature-store-with-redis/
25. DoorDash Engineering. "Building Riviera: A Declarative Real-Time Feature Engineering
    Framework." DoorDash Engineering Blog, 2021 (not re-verified).
    https://doordash.engineering/2021/03/04/building-a-declarative-real-time-feature-engineering-framework/
26. DoorDash Engineering. "Introducing Fabricator: A Declarative Feature Engineering
    Framework." DoorDash Engineering Blog, 2022 (not re-verified).
    https://doordash.engineering/2022/01/11/introducing-fabricator-a-declarative-feature-engineering-framework/
27. Lyft Engineering. "ML Feature Serving Infrastructure at Lyft." Lyft Engineering Blog,
    2021 (not re-verified). https://eng.lyft.com/ml-feature-serving-infrastructure-at-lyft-d30bf2d3c32a
