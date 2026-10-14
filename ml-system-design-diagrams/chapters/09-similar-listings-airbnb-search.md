# 09. Design Similar Listings and Search Ranking for Airbnb (Listing Embeddings, Two-Sided Marketplace)

> Prompt as usually given: "Design the 'similar listings' carousel on an Airbnb listing page",
> or "Design the system that returns the top 10 listings when a user searches for rentals in a
> location on Airbnb" (Chip Huyen, exercise 9 [5]).
> Variants: hotel or vacation-rental search ranking (Booking.com, Expedia, Vrbo style),
> "rental recommendation", Airbnb Experiences ranking [18], real-time in-session
> personalization.
>
> Sourcing note (review pass, Oct 2026): medium.com, arxiv.org, doi.org and bytebytego.com
> did not resolve from the build sandbox and the web-search budget was spent. (fetched) =
> read for this chapter. (listed) = title, year and URL confirmed in a curated GitHub list
> (eugeneyan/applied-ml, alirezadir, ByteByteGoHq, wzhe06/Reco-papers); text not re-read.
> (ch-ref) = verified in an earlier chapter. Check paper numbers against the PDF before
> quoting.

## Company tags

| Company | What is reported | Source |
|---|---|---|
| Airbnb | Chip Huyen's exercise list has "top 10 results for rental listings in a location on Airbnb" (ex. 9) | [5] |
| Airbnb | AIMLInterviews lists "Rental recommendation (Airbnb)" as a sample MLSD question | [7] |
| Airbnb | Khang Pham's ML design course has an "Airbnb Search ranking" case study | [8] |
| Airbnb-style marketplaces | ByteByteGo ch. 9, "Similar Listings on Vacation Rental Platforms". Its only Airbnb reference is the "Listing embeddings in search ranking" blog | [6][10][17] |

**Portals that cover it**

| Portal | Treatment | Link |
|---|---|---|
| ByteByteGo (Aminian & Xu, MLSDI ch. 9) | Similar listings: session-based skip-gram listing embeddings, ANN retrieval and re-ranking. Chapter references also cite positional bias, random walks (with restart) and seasonality | [6][10][17] |
| Khang Pham (Educative / course) | "Airbnb Search ranking" case study (LTR, features, metrics) | [8][9] |
| Chip Huyen (MLSD exercises) | Open prompt, ex. 9 (Airbnb top-10 rental search) | [5] |
| alirezadir AIMLInterviews | "Rental recommendation (Airbnb)" in the question bank. Its company list has three Airbnb items, incl. "Applying Deep Learning To Airbnb Search" | [7][11] |
| Educative "Grokking the ML Interview" | Search ranking and recommendation modules (generic funnel, not Airbnb-specific) | [12] |
| Hello Interview | Recommendation-systems pattern (retrieval, ranking, re-ranking); not Airbnb-specific | [13] |
| Community notes | junfanz1 "Awesome-AI-Review" summary of ByteByteGo ch. 8/9 (Airbnb) | [14] |

## Requirements & scale

| Dimension | Value | Source |
|---|---|---|
| Surfaces | (a) "Similar listings" carousel on the listing detail page. (b) Search results for location + dates + guests. (c) Real-time re-ranking within a session | [1][2] |
| Business goal | Bookings (conversion), not clicks. The marketplace has two sides, so the host must also accept the booking (host rejection counts as a negative) | [1][3][16] |
| Functional | Return the top K (about 10-20 per page, roughly 18 per search page) listings that are available for the dates, match the filters and are bookable | assumption |
| Catalog | About 4.5M active listings when the paper was written (2018), now more than 7M. Inventory is spread across about 100k+ cities or markets | [1] / assumption |
| Training data | About 800M search click sessions for listing embeddings | [1] |
| Users | Hundreds of millions of guests. Most book once or twice a year, so per-user history is sparse | [1] / assumption |
| QPS | About 10^4-10^5 search and carousel requests per second at peak | assumption |
| Latency | Under 200 ms p99 end to end for search. The ranking model has about 30-50 ms for roughly 1-2k candidates | assumption |
| Freshness | Clicks and skips within the session take effect within seconds. Embeddings retrain daily. Availability and price must be real-time | [1] |
| Key constraints | Listings are unique items with limited availability; a listing booked for one guest's dates is gone. Queries are geo-bound. Cold start: new listings appear every day | [1][2] |

## Diagram 1: End-to-end architecture (online serving path)

**Figure 9.1: Online serving. Search and the similar-listings carousel share retrieval,
availability filtering, ranking and a marketplace re-ranker. Session signals feed real-time
personalization features.**

```
+-----------+  query: location, dates, guests, filters | anchor listing_id (carousel)
|  Client   |---------------------------------------------------------+
| web / app |<-------------------------------------------+            |
+-----------+         top-K listings + logging ids        |            v
      |                                            +------+-------------------+
      | clicks, skips, wishlists,                  |   Search / Recs API      |
      | contacts, bookings (events)                |   (orchestrator)         |
      v                                            +--+----------+--------+---+
+-------------+                                       |          |        |
| Event stream|---> session store (last N clicks,     |          |        |
| (Kafka)     |     skips, wishlists; TTL ~2 weeks)   |          |        |
+-------------+            |                          v          |        |
                           |               +--------------------+|        |
                           |               | Candidate retrieval||        |
                           |               | geo index (S2/H3)  ||        |
                           |               | + ANN on listing   ||        |
                           |               |   embeddings (k-NN)||        |
                           |               | + popularity/new   ||        |
                           |               +---------+----------+|        |
                           |                         | ~2-5k ids |        |
                           |                         v           |        |
                           |               +--------------------+|        |
                           |               | Filters: dates     ||        |
                           |               | availability, price||        |
                           |               | guests, host rules,||        |
                           |               | trust & policy     ||        |
                           |               +---------+----------+|        |
                           |                         | ~1-2k     |        |
                           v                         v           |        |
+-------------------------------+        +---------------------+ |        |
| Feature store                 |------->| Ranker (GBDT -> DNN)| |        |
| online: listing stats, price, | feats  | P(book), P(click),  | |        |
|  host accept rate, reviews    |        | P(host accept)      |<+        |
| real-time: EmbClickSim,       |        +----------+----------+ model    |
|  EmbSkipSim, EmbWishlistSim   |                   | scores     version  |
+-------------------------------+                   v                     |
          ^                              +---------------------+          |
          | listing emb (32-d)           | Re-rank / policy:   |          |
+---------+---------+                    | diversity (location,|          |
| Embedding store + |                    | price), dedupe host,|          |
| ANN index (daily) |                    | new-listing boost,  |          |
+-------------------+                    | fairness to hosts   |          |
+-------------------+                    +----------+----------+          |
| Model registry    |--> ranker weights             | top-K + request_id  |
| (versioned)       |    (hot-swapped)              +---------------------+
+-------------------+
          impressions + positions + features (logged at serve time) --> logs
```

## Diagram 2: Offline training & data pipeline

**Figure 9.2: Offline pipeline. Two models train from the same logs. The listing embeddings
(skip-gram over click sessions) are retrained daily. The booking ranker trains on
impressions labelled with how far each one got toward a booking.**

```
+------------------+   +-------------------+   +-------------------------------+
| Serving logs     |   | Event logs        |   | Listing DB / host DB          |
| impressions,     |   | clicks, dwell,    |   | price, type, geo, amenities,  |
| positions, feats |   | wishlist, contact,|   | reviews, accept/reject,       |
| (request_id)     |   | booking, rejection|   | calendar                      |
+--------+---------+   +---------+---------+   +---------------+---------------+
         |                       |                             |
         +-----------+-----------+-----------------------------+
                     | join on request_id / user_id / listing_id (Spark, daily)
                     v
   +-------------------------------+         +-------------------------------+
   | A. Session builder            |         | B. Ranking label builder      |
   | click sessions split on 30 min|         | per impression label:         |
   | gap; drop clicks with dwell   |         | book=1, contact=0.25,         |
   | < 30 s; tag booked sessions   |         | click=0.01, impr=0,           |
   +---------------+---------------+         | host reject=-0.4 [1]          |
                   |                         +---------------+---------------+
                   v                                         |
   +-------------------------------+                         v
   | Skip-gram listing embeddings  |         +-------------------------------+
   | window m=5, d=32, + booked    |-------->| Feature gen: listing stats,   |
   | listing as global context,    | emb sims| query-listing, user-listing,  |
   | + same-market negatives [1]   |         | EmbClickSim / EmbSkipSim ...  |
   +---------------+---------------+         +---------------+---------------+
                   |                                         |
                   v                                         v
   +-------------------------------+         +-------------------------------+
   | Cold-start fill: avg emb of 3 |         | Train ranker (pairwise        |
   | nearest same-type, same-price |         | LambdaRank GBDT -> DNN)       |
   | listings within 10 miles      |         | split by time, not by random  |
   +---------------+---------------+         +---------------+---------------+
                   |                                         |
                   v                                         v
   +-------------------------------+         +-------------------------------+
   | Offline eval: avg rank of the |         | Offline eval: NDCG on booking |
   | eventually-booked listing     |         | (NDCU), by market/segment     |
   +---------------+---------------+         +---------------+---------------+
                   | pass                                    | pass
                   v                                         v
   +-------------------------------+         +-------------------------------+
   | Embedding store + ANN rebuild |         | Model registry -> shadow ->   |
   | (daily, versioned)            |         | interleaving -> A/B test      |
   +---------------+---------------+         +---------------+---------------+
                   |                                         |
                   +--------------------+--------------------+
                                        v
                         online serving (Figure 9.1) --> new logs
                         (feedback loop: position bias; exploration slots)
```

## Diagram 3: Model architecture

**Figure 9.3a: Listing embedding model (KDD 2018 [1]). Skip-gram with negative sampling
over click sessions. The booked listing is a global context in every window, and some
negatives come from the same market.**

```
 click session s = ( l1, l2, l3, [l4], l5, l6, ..., lN )  ---> booked listing lb
                          window m=5 around central l4
                                    |
                                    v
                          +--------------------+
                          |  input embedding   |  v_l4  (d = 32)
                          |  table V (|L| x d) |
                          +---------+----------+
                                    |
        +---------------------------+------------------------------+
        |                           |                              |
        v                           v                              v
+----------------+        +-------------------+          +--------------------+
| positives D_p  |        | random negatives  |          | global context lb  |
| context l2..l6 |        | D_n (whole corpus)|          | (booked sessions   |
| sigma(v'_c.v)  |        | sigma(-v'_n.v)    |          | only) sigma(v'_b.v)|
+-------+--------+        +---------+---------+          +---------+----------+
        |                           |                              |
        |               +-----------+-----------+                  |
        |               | market negatives D_mn |                  |
        |               | same city as l4,      |                  |
        |               | sigma(-v'_mn.v)       |                  |
        |               +-----------+-----------+                  |
        +---------------------------+------------------------------+
                                    v
              loss = - sum log sigma(+) - sum log sigma(-)   (SGD, daily)
   output: 32-d listing vectors; cosine similarity --> similar listings / k-NN
   also: user_type and listing_type embeddings from booking sessions, trained
         in one space with host-rejection negatives (two-sided signal) [1]
```

**Figure 9.3b: Search ranker. Real-time embedding-similarity features join the classic
features in a booking-optimized LambdaRank model (GBDT in 2018, later a DNN [1][2]).**

```
 query ctx: location, dates,   user: past bookings,      listing l: price, type,
 guests, filters, device       user_type emb, session    reviews, geo, host accept
        |                      clicks / skips             rate, listing emb
        |                             |                          |
        |              +--------------+---------------+          |
        |              | real-time similarity feats   |<---------+
        |              | EmbClickSim = cos(v_l, centroid of clicked, per market)
        |              | EmbSkipSim, EmbLongClickSim, EmbWishlistSim,
        |              | EmbInqSim, EmbLastLongClickSim, UserTypeListingTypeSim
        |              +--------------+---------------+
        |                             |
        +-------------+---------------+---------------------------+
                      v
         +---------------------------+   2018: GBDT, LambdaRank, ~100+ feats
         | scorer f(query,user,l)    |   2019: DNN, 2 hidden ReLU layers [2]
         | pairwise loss over        |   2020: two towers, query/user vs listing
         | (booked > not booked)     |         + position as feature with dropout [3]
         +-------------+-------------+
                       v
           score_l --> sort --> diversity re-rank [4] --> top-K
```

## Diagram 4: Real company system (Airbnb Journey Ranker, KDD 2023)

**Figure 9.4: Multi-task "journey" ranking [16]. Positive milestones toward a booking and
negative milestones such as cancellation or host rejection are each modelled as a task,
then combined into one score.**

```
                 query + user + listing features (+ listing embeddings)
                                     |
                                     v
                     +-------------------------------+
                     | shared base network           |
                     +---------------+---------------+
          +-----------+--------------+---------------+-------------+
          v           v              v               v             v
     +---------+ +----------+ +-------------+ +------------+ +--------------+
     | P(click)| | P(detail | | P(request / | | P(booking) | | P(cancel or  |
     |         | |  long    | |  payment)   | |            | |  host reject)|
     |         | |  view)   | |             | |            | | (twiddler)   |
     +----+----+ +----+-----+ +------+------+ +-----+------+ +------+-------+
          |           |              |              |               |
          +-----------+--------------+------+-------+---------------+
                                            v
                 combination module: chain of milestone probabilities,
                 minus a penalty for negative outcomes, + context weights
                                            v
                                   final score --> rank
```

## Data & labels

| Data | Use | Label / note |
|---|---|---|
| Click sessions (more than 30 s dwell, split on a 30-min gap) | Listing embeddings | Co-occurrence within window m=5 [1] |
| Booked sessions | Global context for embeddings | Booked listing = positive for every window [1] |
| Booking sessions (users x listing types) | User-type / listing-type embeddings | Long-term preference. Host rejections are explicit negatives [1] |
| Search impressions | Ranker | book=1, host contact=0.25, click=0.01, impression=0, host reject=-0.4 [1] |
| Skipped listings (shown above a click but not clicked) | EmbSkipSim feature, negatives | Implicit negative [1] |
| Reviews, ratings, calendar, host response | Features | Quality and trust signals |

## Features

| Group | Examples | Freshness |
|---|---|---|
| Listing | Price per night, type, capacity, amenities, review count/score, booking rate, photo quality | Daily / hourly |
| Query | Location, distance to the query center, check-in date, length of stay, guests, lead time | Per request |
| Query x listing | Distance to the map center, price relative to market median, fit to the date range | Per request |
| User / guest | Past booked price, trip type, user_type embedding | Daily |
| Real-time session | EmbClickSim, EmbSkipSim, EmbLongClickSim, EmbWishlistSim, EmbInqSim, EmbLastLongClickSim [1] | Seconds |
| Host / marketplace | Host acceptance rate, instant-book flag, response time, superhost, cancellation history | Daily |
| Context | Device, locale, seasonality, holidays | Per request |

## Model choices (baseline -> advanced)

| Stage | Option | Pros | Cons |
|---|---|---|---|
| Similar listings | Rule: same city + price band + type | Simple, no training | Ignores taste and style |
| Similar listings | Item CF (co-click/co-book matrix) | Uses behaviour | Sparse, poor cold start |
| Similar listings | Skip-gram embeddings + booked global context + market negatives [1] | Captures style and price within a market; cheap ANN | Clicks are not bookings; needs cold-start fill |
| Similar listings | GNN or two-tower with content (photos, text) | Better for new and rare listings | Heavier to train and serve |
| Ranking | Hand-tuned scoring function | Transparent | Plateaus quickly |
| Ranking | GBDT LambdaRank [1] | Strong on tabular data | Manual feature crosses |
| Ranking | DNN, then two-tower DNN [2][3] | Scales with data, learns crosses | Hard to debug; many failed attempts reported [2] |
| Ranking | Multi-task journey ranker [16] | Optimizes the full booking funnel and penalizes bad outcomes | Weighting the tasks is hard |
| Re-rank | Greedy MMR / diversity network [4] | Variety of location and price | Can lower per-item relevance |
| Re-rank | Learned diverse ranking: score each slot given the listings already placed above it [19] | Diversity learned from bookings, not hand-tuned | Sequential scoring costs latency |

## Loss functions

| Model | Loss | Note |
|---|---|---|
| Listing embedding | Skip-gram negative-sampling log loss, plus booked-global and market-negative terms | [1] |
| User/listing type embedding | Same, with host-rejection negatives | Two-sided signal [1] |
| Ranker | Pairwise LambdaRank (booked > not-booked within a search) | Optimizes NDCG directly [1][2] |
| Multi-task | Sum of per-milestone BCE terms (weighted), plus a negative-outcome head | [16] |

## Offline metrics

| Metric | Model | Note |
|---|---|---|
| Average rank of the eventually-booked listing | Embeddings | Rank the candidates by similarity to the last clicked listing [1] |
| NDCG@K / NDCU on booking | Ranker | Main gate for promotion [1][2] |
| Recall@K of ANN vs exact k-NN | Retrieval | Index quality |
| Coverage / new-listing share | Retrieval and re-rank | Cold start health |
| Calibration per market | Ranker | Required for multi-task combination |

## Online metrics / A-B

| Metric | Type | Note |
|---|---|---|
| Bookings per search user, booking conversion | Primary | North star |
| Similar-carousel CTR and bookings from the carousel | Surface | 2018 paper: about +21% carousel CTR and about +4.9% more guests finding their booked listing in the carousel [1] |
| Host acceptance rate, cancellations | Guardrail | Two-sided health [16] |
| GMV / nights booked, price mix | Business | Watch for a drift toward cheaper or pricier listings |
| Search latency, zero-result rate | Guardrail | |
| Interleaving (team-draft) wins | Fast pre-screen | Airbnb uses interleaving before A/B [15] |

## Serving & latency budget (assumption, search, p99)

| Step | Budget | Note |
|---|---|---|
| Parse query + geo lookup | 10 ms | Geocode, map bounds |
| Retrieval (geo index + ANN) | 30 ms | 2-5k candidates |
| Availability/price filter | 30 ms | Calendar service, the largest cost |
| Feature fetch (batch) | 30 ms | Online store + session store |
| Ranker inference | 40 ms | 1-2k listings, CPU/GPU batch |
| Re-rank + policy | 10 ms | Diversity, dedupe, boosts |
| Network/serialization slack | 40 ms | Total under 200 ms |
| Carousel (similar listings) | under 50 ms | Precomputed k-NN lists per listing, then filter by dates |

## Monitoring

| Signal | Why |
|---|---|
| Embedding drift (neighbour overlap from day to day) | Daily retrain can reshuffle the similar lists |
| Feature freshness (session store lag) | Real-time features silently going stale |
| Score distribution per market | Calibration breaks in new or seasonal markets |
| New-listing impression share | Cold-start starvation |
| Host rejection / cancellation of ranked bookings | Marketplace quality |
| Latency and calendar service error rate | Availability errors lead to empty or wrong results |

## Failure modes

| Failure | Mitigation |
|---|---|
| Click-optimized embeddings surface listings people click but do not book | Booked listing as global context; rank on the booking label [1] |
| Similar listings from another city (cross-market leakage) | Market negatives in training [1]; geo filter at serving |
| New listings never shown | Cold-start embedding averaging [1]; exploration slots; new-listing boost |
| Position bias in logged labels | Position as a training feature with dropout; set to 0 at serving [3] |
| Many near-duplicate results (same building or host) | Diversity re-rank [4][19], host dedupe |
| Popular listings get booked out, so rankings go stale | Real-time availability filter; demand-aware features |
| Hosts reject the guest | P(host accept) head; host preference features [1][16] |
| Seasonality shift (holidays) | Time-based splits, daily retrain, seasonal features [10] |
| Cold-start listing with no clicks yet | Random walk with restart on the listing co-click graph, or content embedding, to seed neighbours [10] |

## Interviewer follow-ups

- Why train embeddings on click sessions rather than booking sessions? What do the booked
  global context and the market negatives each fix?
- How do you get an embedding for a listing created today (cold start)?
- How do you evaluate embeddings offline before an A/B test?
- How do you combine long-term preference (user_type) with in-session intent (EmbClickSim)?
- What is the label for ranking, and why use graded labels (book, contact, click, host
  reject) instead of a single click label?
- How do you handle the host side: rejections, acceptance rate, fairness to new hosts?
- How do you correct position bias in the training data?
- How do you enforce date availability and price filters without blowing the latency budget?
- How would you add diversity (location, price) without losing bookings?
- Pointwise vs pairwise vs listwise ranking: which one, and why?
- How do you trade off guest conversion against host revenue and platform GMV?
- How do you speed up ranking experiments? (interleaving, then A/B)
- What changes for map search, flexible dates, or "I'm flexible" queries with no location?
- How do you serve the similar-listings carousel: precomputed k-NN or online ANN?
- How do you detect that a daily embedding retrain broke similarity?
- How would you learn diversity from data instead of a hand-tuned MMR penalty? [19]
- How would ranking change for Airbnb Experiences, which have far less traffic? [18]

## Portal comparison

| Portal | Framing | Retrieval | Ranking | Distinctive emphasis |
|---|---|---|---|---|
| Airbnb papers [1][2][3][16][19] | Search + similar listings, booking-optimized | Geo + embedding similarity | GBDT, then DNN, then two-tower, then multi-task; learned diversity | Graded labels, real-time embedding features, two-sided negatives |
| ByteByteGo ch. 9 [6][10][14][17] | Session-based recs for similar listings | Skip-gram embeddings + ANN index | Light re-ranking (seen, filters) | Booked global context, same-region negatives, avg rank of booked listing, indexing pipeline; positional bias, seasonality |
| Khang Pham [8][9] | Airbnb search ranking | Candidate set by location | LTR (pointwise vs pairwise) | Features and metrics walkthrough |
| Chip Huyen [5] | Open prompt: top-10 rental search | Open-ended | Open-ended | Business framing, data problems |
| alirezadir AIMLInterviews [7][11] | Rental recommendation in the question bank | Generic recsys/search templates | Pointwise, pairwise or listwise LTR | Links to Airbnb papers in its company list |
| Educative / Hello Interview [12][13] | Generic search/recsys funnel | Multi-source retrieval | Multi-stage | Interview delivery and tradeoffs |

## References

1. M. Grbovic, H. Cheng, "Real-time Personalization using Embeddings for Search Ranking at Airbnb", KDD (Applied Data Science track), 2018. https://doi.org/10.1145/3219819.3219885 (listed)
2. M. Haldar et al., "Applying Deep Learning To Airbnb Search", KDD 2019 (arXiv 2018). https://arxiv.org/abs/1810.09591 ; https://doi.org/10.1145/3292500.3330658 (listed: eugeneyan/applied-ml, alirezadir)
3. M. Haldar et al., "Improving Deep Learning For Airbnb Search", arXiv / KDD, 2020. https://arxiv.org/abs/2002.05515 (listed: eugeneyan/applied-ml)
4. M. Abdool et al., "Managing Diversity in Airbnb Search", arXiv / KDD, 2020. https://arxiv.org/abs/2004.02621 (listed: eugeneyan/applied-ml)
5. C. Huyen, "Machine Learning Systems Design: Exercises" (ex. 9, Airbnb rental search), GitHub, 2019. https://github.com/chiphuyen/machine-learning-systems-design/blob/master/content/exercises.md (fetched)
6. A. Aminian & A. Xu, "Similar Listings on Vacation Rental Platforms" (Machine Learning System Design Interview, ch. 9), ByteByteGo, 2023. https://bytebytego.com/courses/machine-learning-system-design-interview/similar-listings-on-vacation-rental-platforms (ch-ref; host unreachable in review)
7. alirezadir, "ML System Design" (sample questions: "Rental recommendation (Airbnb)"), GitHub, 2025. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/ml-system-design.md (fetched)
8. K. Pham, "machine-learning-interview" (ML design case 6: Airbnb Search ranking), GitHub, 2021. https://github.com/khangich/machine-learning-interview (fetched)
9. Educative / K. Pham, "Machine Learning System Design" (course), educative.io, 2021. https://educative.io/courses/machine-learning-system-design/introduction (ch-ref)
10. ByteByteGo, "ml-bytebytego: Reference materials for ML System Design Interview" (ch. 9 references), GitHub, 2023. https://github.com/ByteByteGoHq/ml-bytebytego (fetched; ch. 9 lists 8 references incl. the Airbnb listing-embeddings blog, positional bias, random walk with restart, seasonality)
11. alirezadir, "ML at Companies" (Airbnb papers and blogs), GitHub, 2025. https://github.com/alirezadir/AIMLInterviews/blob/main/src/MLSD/ml-companies.md (fetched)
12. Educative, "Grokking the Machine Learning Interview", educative.io, n.d. https://www.educative.io/courses/grokking-the-machine-learning-interview (ch-ref)
13. Hello Interview, "Recommendation Systems (ML System Design pattern)", hellointerview.com, 2025. https://www.hellointerview.com/learn/ml-system-design/patterns/recommendation-systems (ch-ref)
14. junfanz1, "ML System Design Interview" notes (section "Airbnb: Similar Listings on Vacation Rental"), GitHub, n.d. https://github.com/junfanz1/Awesome-AI-Review/blob/main/System%20Design/ML%20System%20Design%20Interview.md (fetched)
15. Airbnb Engineering, "Beyond A/B Test: Speeding up Airbnb Search Ranking Experimentation through Interleaving", Medium, 2022. https://medium.com/airbnb-engineering/beyond-a-b-test-speeding-up-airbnb-search-ranking-experimentation-through-interleaving-7087afa09c8e (listed: eugeneyan/applied-ml)
16. C. H. Tan et al., "Optimizing Airbnb Search Journey with Multi-task Learning", KDD, 2023. https://github.com/wzhe06/Reco-papers/blob/master/Industry%20Recommender%20System/%5BAirbnb%5D%20Optimizing%20Airbnb%20Search%20Journey%20with%20Multi-task%20Learning.pdf (listed: PDF present in wzhe06/Reco-papers)
17. Airbnb Engineering, "Listing Embeddings in Search Ranking", Medium, 2018. https://medium.com/airbnb-engineering/listing-embeddings-for-similar-listing-recommendations-and-real-time-personalization-in-search-601172f7603e (listed: ByteByteGo ch. 9 refs, eugeneyan/applied-ml)
18. Airbnb Engineering, "Machine Learning-Powered Search Ranking of Airbnb Experiences", Medium, 2019. https://medium.com/airbnb-engineering/machine-learning-powered-search-ranking-of-airbnb-experiences-110b4b1a0789 (listed: eugeneyan/applied-ml)
19. Airbnb Engineering, "Learning To Rank Diversely", Medium, 2022. https://medium.com/airbnb-engineering/learning-to-rank-diversely-add6b1929621 (listed: eugeneyan/applied-ml)
