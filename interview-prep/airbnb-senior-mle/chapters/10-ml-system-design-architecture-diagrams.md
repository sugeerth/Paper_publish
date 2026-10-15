# 10. Whiteboard-Ready ML System Design Diagrams: Search Ranking, Embeddings, Pricing, Trust, LLM Support Routing and AI Search

> **Research note (read first).** Written on 2026-10-14 and reviewed the same day. The shared web-search budget for this run was already used up, so **no new search queries could run**, in the first pass or in the review. In the review, only GitHub (through raw.githubusercontent.com) could be reached. arxiv.org, medium.com, dl.acm.org, airbnb.tech, careers.airbnb.com, YouTube, zenml.io and letsdatascience.com were blocked by the proxy. trueinterview.io, prachub.com, glassdoor.com, interviewquery.com, igotanoffer.com and tryexponent.com did not resolve. The review re-fetched and re-read the candidate-report and official pages: the TrueInterview-derived Airbnb MLE loop summary [1], the dated Airbnb question index [2], the Airbnb interview-process page [3], two translated 1point3acres onsite posts [4][5], Airbnb's Chronon README [6], and the Airbnb Software Engineer process page [26], which is new in the review. It also scanned 91 other translated 1point3acres Airbnb posts from the same mirror; none of them contained an ML design prompt. Two curated indexes, applied-ml [7] and an ML case-study list [31], were fetched to confirm the titles and URLs of Airbnb papers and blog posts. A general ML-design template [8] was also read. Glassdoor, Blind, LeetCode Discuss, Reddit, IGotAnOffer, Exponent, InterviewQuery and Prepfully could not be searched or opened. Before the onsite, cross-check this chapter against those sites.
>
> **Review changes.** Several [Reported] tags claimed more than the source says, and were retagged [Likely] or reworded: questions 1 (first probe), 36, 101 and 124. A job posting that could not be opened (reference 26 in the previous version) was replaced by the fetched team list in [1]. Two uncited and unreachable references were removed: a PracHub report and the Glassdoor page. The review added 30 questions, and the whole chapter was renumbered. Ten are [Reported] from the fetched index [2] and the process pages [1][3][26]. The rest are [Likely] and tied to confirmed Airbnb posts.
>
> **About the diagrams.** No source reports that interviewers hand out diagrams. The diagrams below are **study scaffolds**: one plausible whiteboard layout per core Airbnb prompt. They are built from the reported round skeleton [1][3] and from Airbnb's public papers and posts [9]-[29]. They are **not** Airbnb's actual production architecture. Every latency number is an **illustrative budget to defend or change**, not a published figure. Use each diagram to practise drawing the boxes in the first 10 minutes. Then work through the questions underneath it, which are what interviewers probe.

## How the tags work

Every question carries exactly one tag. The same scheme is used in chapters 01-09, so the whole pack reads the same way in the PDF.

- **[Reported] [n]**: a fetched source [n] says this prompt, or a close variant, was asked in an Airbnb interview. Most are MLE loops. [2] does not always give the role.
- **[Likely]**: a well-grounded inference from the Senior MLE role, the reported round structure and Airbnb's published systems. A citation next to [Likely] points to the paper or post behind the topic. It does **not** mean a candidate reported the question.
- **Follow-up probes** are the indented bullets under a question. They carry no tag of their own. A probe inherits its parent's tag only if the source names that probe; where a source does name a probe, it is cited inline.
- **Reference status** (see References): **fetched** = read on 2026-10-14; **URL confirmed** = title and URL match the applied-ML index [7] or the case-study index [31], but the page itself was not opened; **unverified** = carried over from earlier chapters, host unreachable. Check unverified links before relying on them.

## Format notes

- **Number of rounds.** MLE candidates get "one or two ML system design rounds" in place of the standard system design round [3]. Design is "filtered by design depth, not by code" [1]. Two 1point3acres MLE loops each had two ML design rounds: customer LTV plus host pricing [4], and listing recommendation plus price prediction [5]. In [5], the scheduled "ML experience" round also turned into an unplanned ML design question.
- **Length and medium.** Around 45-60 minutes over video with a shared doc or whiteboard (chapter 03, from [1]). The rounds are "conversations without code, and the first minutes of scoping are graded" [1]. Being able to draw a clean block diagram fast frees time for depth.
- **Interviewer.** Usually a senior or staff MLE from an ML-heavy team such as Search, Pricing, Trust, Customer Support/GenAI or ML platform [Likely]. Interviewers steer hard. Reported examples: an LTV round moved from regression to bucketized classification, a relisting round went into geohashing, and search-ranking rounds "pressed on latency, freshness, and experimentation" [1].
- **What is graded** [1][3]: framing (the business decision and success metric) before any model; the label and its time window; concrete marketplace features; how offline metrics link to online results through an A/B test; and production behaviour: candidate generation, latency, freshness, cold start for new listings, and monitoring. "Problem and label before model" [3]. One ML rejection was blamed on "jumping straight to model ideas" [3].
- **Senior (G9) bar.** At Senior and above, design "carries more weight". A strong coding result does not offset a weak design, and a borderline design signal can add another design round [1]. An added late design round is "a warning sign" [3].
- **Reported prompts covered here:** end-to-end search ranking [1][3]; ranking home results without a text query [1][2]; home page search + availability + ranking [2]; personalized search ranking [2]; user-embedding semantic search [2]; listing recommendation [5]; host pricing / price prediction [4][5]; relisting detection [1][2]; and generative-AI support retrieval and ranking, including turning company policies into customer-support instructions [1][2]. The fraud/ATO and natural-language AI search prompts are [Likely].
- **Adjacent prompts from the standard design round** [2][3][26], each reused below with an ML layer: a homepage backend "with ranking signals"; booking with availability search, pagination and indexing; a support ticketing system "with agent claim rules by language and location and first-response metrics"; a host dashboard for 100+ listings; ad-click event aggregation; and a time-and-geo activity query system. The base prompt was reported. The ML extension is this chapter's addition, and the tags say which part is which.

## Diagram 0. The skeleton every diagram follows

Draw this first in any round. Then specialise the boxes. The flow follows the reported grading skeleton [1][3] and a standard nine-step template [8].

```
 (1) DECISION + METRIC         (2) LABEL + WINDOW             (3) DATA
 what action changes? -------> booking? 7-day? censored? ---> logs, listings, hosts, payments
 north-star + guardrails                                         |
                                                                 v
 +------------------------------ OFFLINE --------------------------------------+
 |  raw logs (impressions, clicks, bookings, CS tickets) --> Hive / data lake  |
 |        |                                                                    |
 |        v                                                                    |
 |  Chronon GroupBy / Join --> point-in-time backfill --> training set         |
 |        |                                         |                          |
 |        |                                         v                          |
 |        |                         train (GBDT / NN) -> offline eval (NDCG,   |
 |        |                         AUC, calibration, slice metrics) -> model  |
 |        |                                                         registry   |
 +--------|---------------------------------------------------------|---------+
          | batch upload + streaming (Kafka) updates                 | deploy
          v                                                          v
 +------------------------------ ONLINE ---------------------------------------+
 |  request -> feature fetch (Chronon KV) -> model server -> post-process ->   |
 |  response        [p99 latency budget written on the board]                 |
 +--------|--------------------------------------------------------------------+
          | log served features + predictions + impressions
          v
 FEEDBACK LOOP: outcomes join back as labels (with delay) --> retrain cadence
 EXPERIMENT:    A/B (or interleaving for ranking) --> launch decision
 MONITORING:    online/offline feature consistency, drift, calibration,
                latency/error SLOs, business guardrails, fallback model
```

### Questions on the skeleton

1. In the first five minutes, which business decision, success metric and label would you write on the board before drawing any model box? [Reported] [1][3]
   - You catch yourself naming model architectures before the label is agreed. How do you recover? [Likely] (One ML rejection was blamed on "jumping straight to model ideas" [3].)
   - How do you restate scope after the interviewer redirects you, so that the new scope is explicit? [Reported] [1]
2. Where in the diagram does point-in-time correctness matter, and what goes wrong if the training join uses today's feature values? [Likely] [6]
   - How would you measure training-serving skew once the model is live, not just prevent it? [Likely] [6]
3. Which boxes are batch, which are streaming, and which are request-time? How do you decide for each feature? [Likely] [6]
4. What do you log at serving time so that the next model can be trained from it without a fresh backfill? [Likely] [6]
5. Where do the offline metric and the online metric live in your diagram, and how do you show the link between them? [Reported] [1]
6. What is your fallback path if the model server or the feature store times out? Draw it. [Likely]

## Diagram 1. Stays search ranking (end to end)

Reported as "end-to-end search ranking", with probes on latency, freshness and experimentation [1][3]. Also reported in the variants "rank home search results without a text query", "home page: search + availability + ranking" and "personalized search ranking" [2]. The layout is drawn from Airbnb's search papers: the deep-learning ranker [9][10], diversity re-ranking [12][13], the multi-stage journey ranker [21], maps ranking [22], embedding-based retrieval [20] and location retrieval [29].

```
 GUEST QUERY: location text / map box, dates, guests, filters, (maybe nothing)
      |
      v
 [Query understanding]  geocode + location retrieval model -> search area     ~10 ms
      |                 intent: flexible dates? category? group size?
      v
 [Candidate retrieval]  (a) geo index: listings inside area                   ~30 ms
      |                 (b) ANN over listing embeddings (EBR two-tower)
      |                 HARD FILTERS: availability calendar for the dates,
      |                 guest count, price filter, instant-book, policy
      |                 -> 1k-10k candidates
      v
 [Feature fetch]        Chronon online KV: listing aggregates (CTR, booking   ~15 ms
      |                 rate, reviews, host response), guest history, session
      |                 embeddings (streaming), query x listing crosses
      v
 [Stage-1 ranker]       light GBDT / small NN -> top ~500                      ~20 ms
      v
 [Stage-2 ranker]       deep NN (pairwise / multi-task: click, wishlist,      ~40 ms
      |                 book, host-accept, no-cancel), price as a feature
      v
 [Re-rank]              diversity (location, price, type), business rules,   ~10 ms
      |                 new-listing exploration slots, map vs list layout
      v
 RESULTS PAGE (list + map pins)              total p99 target: write it, e.g. <200 ms
      |
      v
 [Logging]  impressions with position, served features, model version,
            experiment bucket -> Kafka -> Hive
      |
      v
 LABELS: click / long view / wishlist / booking request / host accept /
         completed stay (delayed days-to-weeks) --> daily retrain
 EXPERIMENT: interleaving to screen rankers fast -> A/B on bookings [15]
 MONITORING: NDCG on logged searches, calibration per market, feature
            null rates, availability-filter misses, latency, booking guardrails
```

### Framing, labels and retrieval

7. Design Airbnb stays search ranking end to end. Draw the full pipeline before going deep on any box. [Reported] [1][3]
   - Which box would you deep-dive first if you had only 20 minutes left, and why?
   - What is your single online success metric, and which guardrails sit beside it?
8. Rank home search results when the guest gives no text query (just location and dates, or nothing at all). How does the diagram change? [Reported] [2]
   - With no query, what does query understanding output, and what does the retrieval stage key on?
9. Design the home page as one system covering search, availability and ranking. Where does the availability check sit, and what does it cost if it sits in the wrong place? [Reported] [2]
   - If the availability calendar is eventually consistent, how do you avoid ranking a listing that was booked a minute ago?
   - Would you filter on availability before or after ANN retrieval? What happens to recall in each case?
10. Design a personalized search ranking system. Which boxes in the diagram carry personalization, and which stay global? [Reported] [2]
11. Why split retrieval from ranking at all, given that a city may have only a few thousand listings? [Likely] [20]
    - When does embedding-based retrieval beat a pure geo-filter, for example for flexible-location or category searches? [Likely] [20]
12. How do you train the two-tower retrieval model, and what are its negatives? [Likely] [20]
    - How do you keep the ANN index fresh when hosts change prices, photos and calendars many times a day?
13. How would you build the query-understanding step that maps "near the beach in Portugal" to a search area? [Likely] [29]
    - What is the label for location retrieval, and how do you evaluate it offline?

### Ranker, re-rank and latency

14. Would you use GBDT or a neural network as the main ranker? Defend it in the context of Airbnb's own move from GBDT to deep NN. [Likely] [9][10]
    - What would make you keep a GBDT stage in the cascade?
15. Write the latency budget on the board for each stage. Which stage do you cut first if p99 doubles? [Reported] [1]
    - What do you precompute offline, and what must run at request time?
    - How do you handle a feature-store timeout inside the latency budget?
16. How does the ranker stay fresh: real-time session features, daily retraining, or both? [Reported] [1]
    - Which features must be streaming, and which are fine as daily batch?
17. How do you combine click, wishlist, booking and host-acceptance signals: one weighted label, a multi-task model, or separate stages? [Likely] [21]
18. Where do you put diversity: in the loss, in a re-rank stage, or both? What does diversity mean for Airbnb results? [Likely] [12][13]
    - How do you measure whether diversity helped bookings rather than only changing the page?
19. How do new listings with no history get exposure without hurting booking rate? Where does exploration sit in the diagram? [Reported] [1]
20. How does ranking pins on the map differ from ranking the list, and does it need its own model? [Likely] [22]
21. Price is both a feature and something the guest filters on. How do you stop the ranker from simply learning "cheaper is better"? [Likely] [9][10]
22. You are moving the production ranker from GBDT to a neural network. Draw the migration path (shadow scoring, feature parity checks, interleaving, ramp) and say how you would stop the first NN from losing to the tuned GBDT. [Likely] [9][42]

### Adjacent prompts on the same diagram

23. Design the Airbnb homepage backend "with ranking signals". Show where the ML ranker sits relative to the backend fan-out, the cache and the page-module layout. [Reported] [26]
   - Which homepage sections can be precomputed per guest offline, and which must be ranked per request?
24. Design booking with availability search, pagination and indexing. Then plug the two-stage ranker into it. How do you keep page 2 consistent with page 1 when scores are recomputed between requests? [Reported] [26]
25. How would the search diagram change for Airbnb Experiences, where inventory is small, sold in time slots, and bookings are sparse? [Likely] [14]
26. Design category browsing ("Beachfront", "Treehouses"). How do categories assigned by ML and checked by humans feed candidate retrieval, and how do you rank inside a category? [Likely] [33]
27. Which home attributes (wifi, parking, a workspace) should a listing card show to a given guest? Where does that model sit in Diagram 1, and what is its label? [Likely] [38]

### Logging, feedback and monitoring

28. What exactly do you log per impression so that you can correct for position bias later? [Likely] [10]
    - How would you estimate position bias from Airbnb's logs?
29. Bookings arrive days after the search. How does that delay change your training-data pipeline and your retraining cadence? [Likely] [9]
30. How do you test a new ranker: offline replay, interleaving, then A/B? What decision does each stage make? [Reported] [1][15]
    - When does interleaving disagree with the A/B result, and which do you trust?
31. What would you put on the search-ranking monitoring dashboard on day one after launch? [Reported] [1]
    - A drop in bookings hits a single market only. How do you narrow it down to data, model or product causes?

## Diagram 2. Listing-embedding personalization (similar listings and real-time session signals)

Reported in the forms "design user embedding semantic search" (Apr 2026) [2] and "listing recommendation" [5]. The layout follows Airbnb's listing-embedding work [11] and the later embedding-based retrieval work [20].

```
 OFFLINE (daily)
 click sessions (30-min gap split) ----+
 booking sessions (user-type/listing- |--> skip-gram w/ negative sampling
   type tokens for sparse bookers)    |    + booked listing as global context
 same-market negatives  --------------+    + in-market negatives
                                           |
                                           v
                          listing vectors (d ~ 32-128) + user-type/listing-type vectors
                                           |
            +------------------------------+----------------------------+
            v                                                           v
 [ANN index: listing_id -> vec]                          [Cold start: average vectors of
  (similar-listing carousel,                              k nearest listings by geo/type/price]
   retrieval for search)
            |
 ONLINE (per request)
 session clicks / skips / wishlists (streaming, Chronon) --> short-term user vector
            |
            v
 features for ranker: cos(candidate, recent clicks), cos(candidate, skips),
                      cos(candidate, long-term booked type)          ~5-10 ms
            |
            v
 Ranker (Diagram 1) or "Similar listings" carousel
            |
 FEEDBACK: clicks / bookings on carousel + search --> new sessions --> retrain
 MONITORING: embedding drift between versions, nearest-neighbour sanity sets,
            coverage of new listings, carousel CTR and bookings vs control
```

### Questions

32. Design a user-embedding semantic search system for Airbnb. [Reported] [2]
    - What are the user tower's inputs, and how do you represent a guest with one past booking or none?
    - How do you evaluate the embedding before any online test?
33. Design a listing recommendation system ("similar listings" or homepage recommendations). [Reported] [5]
    - Where does the candidate set come from, and how is the final list ranked?
34. Why train listing embeddings on click sessions rather than on listing attributes? What does each capture? [Likely] [11]
    - How do you split a browsing log into sessions, and why does that choice matter?
35. Bookings are far sparser than clicks. How would you learn embeddings that capture long-term booking preference? [Likely] [11]
36. How do you give a brand-new listing an embedding on day one? [Likely] [1][11] (Cold start for new listings is a graded area [1], but no source reports this embedding variant.)
    - How long until the cold-start vector is replaced, and how do you detect that it was a bad guess?
37. How do you turn in-session signals (clicks, skips, wishlists) into ranker features within the latency budget? [Likely] [11]
    - Where does the session state live, and what happens when the guest switches device?
38. Embeddings are retrained daily, and vectors from different runs are not aligned. How do you keep downstream features and the ANN index consistent? [Likely]
39. Which ANN index would you use, and what recall or latency trade-off would you accept? [Likely] [20]
    - How do you apply hard filters (dates, guests) to ANN results without losing recall?
40. How would you add text and photo signals to listing embeddings, and when is that worth the cost? [Likely] [7]
41. How would you check whether personalization is narrowing guests into a filter bubble, and what would you monitor? [Likely] [12]
42. A photo classifier tags each listing photo by room type (bedroom, kitchen, exterior). Draw it as an upstream model that feeds both cover-photo choice and listing-quality features. How do you version it so that downstream models do not break? [Likely] [35]
43. Amenities detected from photos and amenities entered by the host can disagree. How do vision-derived amenities enter the feature store and the embeddings, and which source wins when they conflict? [Likely] [36][37]

## Diagram 3. Smart Pricing and demand forecasting

Reported as "design a system to help landlords set prices" [4] and "price prediction" [5]. The layout follows Airbnb's published dynamic-pricing work [19] and home-value prediction post [32].

```
 DATA: listing attributes, calendar (booked / open by night), price history,
       market supply, search demand (searches for dates x area), events, seasonality
      |
      v
 +------------------------------- OFFLINE (daily) -----------------------------+
 | [Demand model]   P(book night d | price p, lead time, features)  (GBDT/NN) |
 |       |          label: booked by check-in? (censored for future nights)   |
 | [Market forecast] demand per market x date (time series + event signals)   |
 |       |                                                                    |
 | [Price model]    suggested price per (listing, night); custom loss that    |
 |                  penalises over- vs under-pricing differently              |
 |       |                                                                    |
 | backtest: would suggestion have booked? revenue vs host-set price           |
 +-------|--------------------------------------------------------------------+
         v
 [Price store: listing x night -> suggested price, min/max host bounds]
         |                                   ^
         | read by host app + calendar        | host overrides (feedback)
         v                                   |
 HOST UI: "Smart Pricing on" / tips -----> host accepts or edits
         |
         v
 SEARCH sees the new price -> guest demand responds -> bookings
 FEEDBACK: bookings + host overrides + price-change elasticity --> retrain
 EXPERIMENT: host-level or market-level randomisation (interference!)
 MONITORING: forecast error by market and lead time, occupancy, host revenue,
            override rate, price shock alerts (events, outages)
```

### Questions

44. Design a system that helps hosts set nightly prices. [Reported] [4]
    - What does the host want to maximise: revenue, occupancy, or something else? How do you agree on that with the interviewer?
45. Design a price prediction model for Airbnb listings. What exactly is the target? [Reported] [5]
    - Why is "the price the host set" a weak label?
46. Draw the split between a demand (booking probability) model and a price-suggestion model. Why separate them? [Likely] [19]
47. Future nights are not yet booked. How do you build training labels for demand at different lead times? [Likely] [19]
48. Why might a custom asymmetric loss beat plain regression for price suggestions? [Likely] [19]
49. How often do you recompute suggested prices, and which inputs must be near real time (for example a newly announced concert)? [Likely]
50. Changing prices for many hosts in one market changes the demand that every other host sees. How do you design the experiment? [Likely] [24]
    - Would you randomise by host, by listing or by market? What does each cost in power?
51. How do you estimate price elasticity from observational data where hosts chose their own prices? [Likely]
52. Hosts override suggestions often. How do overrides enter the feedback loop, and what bias do they create? [Likely]
53. What would you monitor to catch a pricing model that is quietly pushing prices too low in one city? [Likely]
54. How does demand forecasting feed other systems in the diagram (search ranking, host tips, supply growth)? [Likely]
55. How does Smart Pricing suggest a price for a brand-new listing with no booking history? Draw the cold-start path, and say when it hands over to the listing's own demand model. [Likely] [19][32]
56. Hosts choose whether to turn Smart Pricing on, so the adopters are not a random sample. How would you estimate its effect on host revenue without a randomised test? Where would an ML-based counterfactual estimate sit in the diagram? [Likely] [40]
57. Build a panel dataset in SQL and run a causal regression on it, for example bookings on price at the listing-week level. Which confounders must the panel control for? [Reported] [2]
58. Design the host listings page for a host with 100+ listings so that it shows suggested prices and demand forecasts for every listing. What do you precompute, and how do you aggregate across listings fast enough? [Reported] [2][3] (The base prompt, a host listings page or dashboard for 100+ listings, was reported. The pricing layer is this chapter's addition.)

## Diagram 4. Trust: payment fraud, account takeover (ATO) and off-platform payment

Fraud and ATO are not reported as prompts in the fetched sources; they are [Likely] because Airbnb's Trust and Safety MLE team works on "fraud defense, reservation screening, party-risk prediction and property-damage risk" [1], and Airbnb has published on targeted friction for fraud [34]. **Relisting detection** is the reported trust-style prompt, including geohashing and its link to online metrics [1][2][3].

```
 EVENTS: login, device / IP / geo, password reset, payout-method change,
         booking + payment, messages, listing create / edit
      |
      v
 [Real-time features]  Chronon streaming: velocity (logins / cards / payouts
      |                per device, IP, account in 1h / 24h), new-device flag,
      |                geo distance from last login, account age
      |                + batch: graph features (shared device / card / phone)
      v
 [Risk models]   ATO model (login + sensitive actions)              ~20-50 ms
      |          payment-fraud model (checkout)                     sync path
      |          off-platform model: message text classifier
      |             ("pay me by bank transfer", emails, phone numbers)
      v
 [Decision layer]  score + rules + cost matrix ->
      |   allow | targeted friction (2FA, ID check, 3DS) | hold payout | block
      |   | send to human review queue
      v
 [Human review] --> labels (fraud / not fraud), appeals --> label store
      |
 DELAYED LABELS: chargebacks (30-90 days), account-recovery claims, user reports
      |
      v
 retrain (weekly) with label-delay-aware windows
 MONITORING: precision at review capacity, friction rate on good users,
            chargeback rate, attack-pattern drift, rule-hit dashboards
 RELISTING: geohash buckets -> candidate pairs (photo / text / host-graph
            similarity) -> pair classifier -> review -> enforcement
```

### Questions

59. Design a system that detects relisting: a removed listing that comes back, possibly under a new account. Draw candidate generation and the pair scorer. [Reported] [1][2][3]
    - How does geohashing generate candidate pairs, and what geohash precision would you start with? [Reported] [1][3]
    - How do the model's outputs connect to online metrics when confirmed labels arrive late and only for flagged cases? [Reported] [1]
60. Design an account-takeover detection model. Where does it run in the login and payout flows, and with what latency budget? [Likely] [1]
    - Which actions after login are the riskiest, and should they be re-scored?
61. Design payment-fraud scoring at checkout. How does the decision layer turn a score into allow, friction or block? [Likely] [34]
    - How do you set thresholds when a false positive loses a real booking and a false negative costs a chargeback?
62. Chargebacks arrive 30-90 days later. How do you build training data and evaluate a model trained before the labels mature? [Likely]
63. Design a detector for hosts or guests who try to take payment off the platform through messages. [Likely] [17]
    - How do you handle obfuscated text ("p a y p @ l"), images and multiple languages?
    - What is the action: warn, block the message, or send to review?
64. Only reviewed cases get labels, and the model chooses which cases are reviewed. How do you correct for that selection bias? [Likely]
65. How would you use a graph of shared devices, cards and phone numbers? Is it a feature source or a separate model? [Likely]
66. Fraudsters adapt to the model. What monitoring tells you an attack pattern has shifted, and how fast can you ship a fix? [Likely]
67. How would you run an A/B test on a fraud model when the "treatment" lets some fraud through? [Likely] [24]
68. How do you measure the cost of friction on good users and include it in the objective? [Likely] [34]
69. Design reservation screening: score a booking request before it is confirmed, for example for party risk. Where does it sit between checkout and host acceptance, and what actions can it take? [Likely] [1]
   - The label (a party actually happened) is rare, delayed and mostly unobserved. What proxy labels would you use, and how do you check them?
70. Design a property-damage risk model at booking time. What is the label, how rare is it, and which action does the score drive (deposit, coverage, extra checks, block)? [Likely] [1]
71. Account deduplication, where accounts that share any attribute are merged, has been reported as a coding question with a follow-up that grows from two attributes to any number [2][26]. How would you turn that entity-resolution step into a feature source for the trust models in this diagram? [Likely] [2][26]
72. Chronon's own quickstart uses a fraud model that is called when checkout starts [6]. Define the GroupBys and the Join you would write for Airbnb's checkout payment-fraud model, including the windows. [Likely] [6]

## Diagram 5. LLM customer-support routing and assistance

Reported: "recent senior loops added ranking and retrieval prompts around generative-AI support tools" [1], and "Generate customer-support instructions from company policies" (Sep 2026) [2]. The Community Support AI team works on "LLM fine-tuning, retrieval, evaluation and guardrails for customer-support assistants" [1]. The layout follows Airbnb's message-intent classification [17], its Intelligent Automation Platform [18][27], its text-generation models for support [30] and its LLM-friendly knowledge representation work [23].

```
 GUEST / HOST CONTACT (chat, email, voice transcript) + reservation context
      |
      v
 [Pre-processing]  PII redaction, language detect, attach booking state      ~20 ms
      v
 [Intent + urgency classifier]  (small fine-tuned model; safety intents        ~30 ms
      |                         e.g. safety emergency -> human immediately)
      v
 [Retrieval]  policy / help-centre KB chunks, past resolved tickets,          ~100 ms
      |       reservation facts (tools / APIs) -> top-k -> re-ranker
      v
 [LLM]  draft answer or self-serve action (cancel, refund within policy)      ~1-3 s
      |  grounded on retrieved policy; structured output
      v
 [Guardrails]  policy-compliance check, hallucination check, refund limits,
      |        confidence threshold
      v
 ROUTER:  auto-resolve | suggest reply to agent | route to queue (skill,
          language, tier, priority) | escalate to safety team
      |
 FEEDBACK: resolution, re-contact within 7 days, CSAT, agent edits to drafts,
           escalations --> eval sets + fine-tuning data
 OFFLINE EVAL: golden ticket set, LLM-as-judge + human rubric per policy area
 MONITORING: auto-resolve rate, wrong-refund rate, re-contact, latency, cost
            per ticket, drift in intent mix (e.g. after a storm or outage)
```

### Questions

73. Design retrieval and ranking for a generative-AI customer-support tool. [Reported] [1]
    - What is the retrieval unit (policy chunk, past ticket, macro), and how do you label relevance?
    - How do you evaluate the retriever separately from the generator?
74. Design a system that generates customer-support instructions from company policies. [Reported] [2]
    - How do you keep generated instructions consistent with policies that change?
    - How would you evaluate instructions before agents use them?
75. Design the routing layer that decides whether a ticket is auto-resolved, drafted for an agent or sent to a specialist queue. [Likely] [18]
    - Which intents must never be auto-resolved, and how is that enforced?
76. How would you classify contact intent across many languages with limited labelled data? [Likely] [17]
77. Where do you redact PII in the diagram, and what does redaction do to model quality? [Likely]
78. What is the latency budget for chat vs email vs voice, and how does it change the model choice? [Likely] [27]
79. How do you stop the LLM from granting a refund that policy does not allow? Draw the guardrail. [Likely]
80. Which online metric shows success: deflection, resolution, re-contact rate or CSAT? Which is the guardrail? [Likely]
81. How do agent edits to LLM drafts become training data, and what bias does that carry? [Likely]
82. A storm cancels thousands of trips in one region. How does the system cope with the spike and the shift in intent mix? [Likely]
83. How would you A/B test a new support model when tickets from the same reservation can hit both arms? [Likely] [24]
84. Design a customer-support ticketing system with agent claim rules by language and location, and first-response metrics. Then show where an ML router would replace or augment the hand-written claim rules. [Reported] [2][3][26] (The ticketing system was reported in Jun 2026 [2] and as an added design round that decided the outcome [26]. The ML router is this chapter's addition.)
   - Which first-response or resolution metric would the router optimise, and which would you keep as a guardrail?
85. Would you use one generative model for intent detection, answer retrieval and reply suggestion, or separate models? Defend the choice on latency, cost and failure isolation. [Likely] [30]
86. When would you fine-tune a model for support, and when would you prompt a base model with retrieval? What evaluation result would change your mind? [Likely] [1][30]

## Diagram 6. Natural-language AI search ("a quiet cabin near Lake Tahoe for 6 with a hot tub")

No fetched source reports this prompt by name. It is [Likely] because "design user embedding semantic search" is reported [2] and Airbnb has publicly moved towards AI-powered search [28].

```
 NL QUERY (+ chat history)
      |
      v
 [LLM query parser]  -> structured: {location: "Lake Tahoe", guests: 6,          ~300-800 ms
      |                 amenities: [hot_tub], vibe: "quiet", dates: ?}         (cache common queries)
      |                 + clarifying question if dates missing
      v
 [Hybrid retrieval]  structured filters (geo, availability, guests, amenities)
      |              + semantic ANN on text / photo listing embeddings ("quiet")
      |              + lexical match on reviews / descriptions
      v
 [Ranker]  Diagram-1 ranker + query-listing semantic match features          ~50 ms
      v
 [Explain / summarise]  optional LLM "why this fits" per result (async, streamed)
      v
 RESULTS + follow-up chat refinements
      |
 FEEDBACK: reformulations, refinements, clicks, bookings, abandons
 OFFLINE EVAL: parse accuracy on labelled queries, retrieval recall on
              judged sets, end-to-end LLM-judge relevance
 MONITORING: parse failure rate, empty-result rate, hallucinated amenity
            rate, LLM cost and latency, bookings vs classic search
```

### Questions

87. Design natural-language search for stays. Draw where the LLM sits and where it must not sit. [Likely] [2][28]
    - Would you let the LLM rank listings directly? Why or why not?
88. How do you evaluate the query parser, and what happens when it extracts a wrong constraint, such as the wrong city? [Likely]
89. Which constraints become hard filters and which become soft ranking signals ("quiet", "good for kids")? [Likely]
90. How would you keep latency acceptable when an LLM call sits in front of retrieval? [Likely]
    - What would you cache, distil or run in parallel?
91. How would you make sure explanations do not claim an amenity the listing does not have? [Likely]
92. How do you compare NL search with classic search in an experiment when the two attract different kinds of query? [Likely] [24]
93. How do you build a judged relevance set for NL queries at Airbnb's scale? [Likely] [33]
94. How would a knowledge graph of places, activities and listing attributes help the parser and the retriever handle "quiet cabin near Lake Tahoe"? [Likely] [25]
95. The parser returns one guess for an ambiguous place name ("Springfield", "the Cape"). How do you detect the ambiguity, and do you ask, show a mix, or pick the most likely? [Likely] [29]
96. Which parts of NL search can run offline, such as listing summaries and attribute extraction from descriptions and reviews? How do you keep them fresh when hosts edit listings? [Likely] [37]

## Diagram 7. Experimentation loop for ML launches

The role includes experimentation and A/B testing, and "how an A/B test confirms" offline gains is graded [1]. "Design an A/B testing platform" was reported in Apr 2026 [2]. The layout follows Airbnb's posts on interleaving [15] and experiment guardrails [24].

```
 new model candidate
      |
      v
 offline gate: NDCG / AUC / calibration vs prod on the same logged data
      |
      v
 shadow traffic: latency, error rate, score-distribution sanity
      |
      v
 interleaving (ranking only): team-draft, small traffic, days   [15]
      |
      v
 A/B on bookings (unit: guest or search; market-level if interference)
      |  guardrails: cancellations, CS contacts, host earnings, latency  [24]
      v
 ramp 1% -> 10% -> 50% -> 100%  with auto-rollback on guardrail breach
      |
      v
 holdout kept for long-term effects; post-launch monitoring
```

### Questions

97. Design an A/B testing platform that ML teams at Airbnb would use. [Reported] [2]
    - How does assignment work for logged-out guests who later log in?
98. Design robust primary and guardrail metrics, for example for a search-ranking launch. [Reported] [2]
99. Design an A/B test with causal inference for a feature where randomisation is imperfect. [Reported] [2]
100. Where in the launch funnel do you place interleaving, and what can interleaving not tell you? [Likely] [15]
101. Offline NDCG went up but the A/B on bookings is flat. Walk through the diagram to find why. [Likely] [1] (The offline-to-online link is a graded area [1]. This scenario is not reported.)
102. Which unit of randomisation fits each diagram in this chapter (search, pricing, trust, support), and why? [Likely]
103. How do you measure a long-term effect, such as repeat bookings, when the A/B runs only for two weeks? [Likely] [24]
104. Design a network-aware experiment for a "Wi-Fi" badge on listings. [Reported] [2]
   - Treated and control listings compete for the same guests. Which unit and design would you choose (listing, market cluster, switchback), and what bias does each leave?
105. Design an event aggregation pipeline for clicks (reported in its ad-click form). Then make the same pipeline feed both the experiment metrics and the ranker's training labels. How do you handle late, duplicate and bot events? [Reported] [2]
106. Two teams' launches trade bookings now against host retention later. How would one shared "future value" metric settle the trade-off, and where does it enter the launch decision in this diagram? [Likely] [39]
107. Draw the assignment, exposure-logging and metric-computation services of an experiment platform that runs hundreds of concurrent search experiments. How do you detect interaction between two experiments? [Likely] [41]

## Diagram 8. The Chronon feature path (zoom-in for any diagram)

Chronon is Airbnb's open-source feature platform. It defines features once as GroupBys (keyed, windowed aggregations) and Joins, produces point-in-time-correct backfills for training, serves the latest values from a key-value store, can update features from Kafka streams, and measures online/offline consistency by replaying logged fetches [6]. Its predecessor, Zipline, is described in two Airbnb talks [16].

```
 sources: Hive tables (batch) + Kafka topics (streaming events)
      |
      v
 GroupBy  key=listing_id   aggs: count(bookings, 7d/30d), avg(rating), ...
 GroupBy  key=guest_id     aggs: last_k(clicked_listing_ids, 10), ...
      |
      v
 Join  (left = search / checkout events with timestamps)
      |------------------------------+-----------------------------------+
      v                              v                                   v
 BACKFILL (point-in-time)     BATCH UPLOAD -> online KV           STREAMING updates
 -> training table            (daily snapshot)                    (Kafka -> KV, seconds)
                                     |                                   |
                                     +-------------> fetch API <---------+
                                                      |  (low ms, high fan-out)
                                                      v
                                            model server request
      |
      v
 CONSISTENCY: log online fetches -> replay as backfill -> compare -> alert
```

### Questions

108. Draw how a feature like "listing bookings in the last 7 days" flows from raw events to both training data and the online model. [Likely] [6]
109. What does point-in-time correctness mean for a search-ranking training example, and how would you test that your backfill has it? [Likely] [6]
110. A feature's online value differs from its backfilled value 3% of the time. How do you find the cause? [Likely] [6]
111. Which features in the search ranker need streaming updates, and what does streaming cost you in complexity? [Likely] [6]
112. How does high fan-out (hundreds of candidate listings per request) shape the feature-fetch design and the latency budget? [Likely] [6]
113. When would you log features at serving time instead of backfilling them? [Likely] [6]
114. You are asked to design a platform feature, such as "model cumulative booking metrics and first and last booking times". How would you express it in Chronon terms? [Reported] [2]
115. Design a query system over user activity filtered by time window and geography. Then reuse it to compute windowed geo features, such as searches per area in the last hour, for pricing and ranking. [Reported] [2] (The base query system was reported in Nov 2025. The feature reuse is this chapter's addition.)
116. A streaming feature in the ranker stops updating, but the online store keeps serving its last value. How do you detect stale features, and what should the model or the serving layer do when a feature is stale? [Likely] [6]

## Cross-cutting probes (any diagram)

117. Estimate the listing lifetime value. Show where the LTV model sits in a diagram, and what decision consumes it. [Reported] [1][2][3]
    - The interviewer moves you from regression to bucketized classification. Redraw the output layer and the evaluation. [Reported] [1]
118. Design a system that calculates customer (guest) lifetime value. [Reported] [4]
    - The interviewer has no time for serving. Which part of the diagram do you drop, and which must stay? [Reported] [4]
119. Choose a listing's cover photo to lift bookings. Draw the system, including cold start. [Reported] [1]
120. Design a model that identifies high-quality listings. Where does the label come from in your diagram? [Reported] [1]
121. Which of the diagrams in this chapter has the worst feedback loop (the model shaping its own training data)? How do you break it? [Likely]
122. For each diagram, name the one metric that would page you at 3 a.m. [Likely]
123. Which pieces of these six systems would you build once as shared platform (feature store, embedding service, ANN, experiment platform), and which stay product-specific? [Likely] [6][32]
124. Your scheduled ML experience round turns into an unplanned ML design question. How do you switch to this skeleton and get a diagram on the board within five minutes? [Reported] [5]
125. A phone screen ends with two short ML design prompts. Give a two-minute opener for each: the target variable, a cold-start plan and a one-line architecture. [Reported] [1]
126. Redraw one ML system you shipped in this chapter's skeleton. Mark the boxes you decided yourself, the alternatives you rejected and the metric the system moved. [Reported] [1]

## References

Status: **fetched** = read on 2026-10-14; **URL confirmed** = title and URL match the applied-ML index [7] or the case-study index [31] (both fetched), but the page itself was not opened; **unverified** = carried over from chapters 03 and 09, host blocked by the proxy on 2026-10-14. No web searches could run for this chapter (search budget exhausted). Removed in review: a PracHub report and the Glassdoor MLE page (uncited and unreachable), and an Airbnb Careers job posting (unreachable; its claim is now cited to [1]).

1. "Airbnb Machine Learning Engineer interview process" (TrueInterview-derived summary of 3 candidate reports, mostly 2026 Q1-Q2). kevin-2023-code/Tech-Interview-Questions, GitHub, 2026. Fetched. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/airbnb/interview-process-machine-learning-engineer.md
2. "Airbnb" question index with report dates (Rank Home Search Results Without a Text Query, Sep 2026; Generate Customer-Support Instructions from Company Policies, Sep 2026; Model Cumulative Booking Metrics, Sep 2026; Listing Lifetime Value, May 2026; Design User Embedding Semantic Search, Apr 2026; Design an A/B Testing Platform, Apr 2026; Home Page Search + Availability + Ranking, Mar 2026; undated: personalized search ranking, relisting detection, primary and guardrail metrics, A/B test with causal inference). kevin-2023-code/Tech-Interview-Questions, GitHub, 2026. Fetched. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/airbnb.md
3. "Airbnb interview process" (one or two ML system design rounds for MLEs; "Problem and label before model"). kevin-2023-code/Tech-Interview-Questions, GitHub, 2026. Fetched. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/airbnb/interview-process.md
4. 1point3acres post #40, English mirror (ML design: customer lifetime value; helping hosts set prices). FlowSight/coding, GitHub. Fetched. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/40.md
5. 1point3acres post #79, English mirror (ML design: listing recommendation; price prediction; ML experience round turned into ML design). FlowSight/coding, GitHub. Fetched. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/79.md
6. "Chronon: A data platform for AI/ML" (GroupBy, Join, point-in-time backfills, online serving, streaming, consistency measurement). airbnb/chronon, GitHub. Fetched. https://github.com/airbnb/chronon
7. Yan, "applied-ml: Papers & tech blogs by companies sharing their work on data science & machine learning in production." eugeneyan/applied-ml, GitHub. Fetched. https://github.com/eugeneyan/applied-ml
8. Dirafzoon, "ML System Design" (nine-step template; lists "Rental recommendation (Airbnb)"). alirezadir/AIMLInterviews, GitHub. Fetched. https://github.com/alirezadir/AIMLInterviews/blob/main/src/MLSD/ml-system-design.md
9. Haldar et al., "Applying Deep Learning To Airbnb Search," arXiv 1810.09591, 2018 (KDD 2019). URL confirmed. https://arxiv.org/abs/1810.09591
10. Haldar et al., "Improving Deep Learning For Airbnb Search," arXiv 2002.05515, 2020. URL confirmed. https://arxiv.org/abs/2002.05515
11. Grbovic, "Listing Embeddings in Search Ranking," Airbnb Tech Blog, 2018. URL confirmed. https://medium.com/airbnb-engineering/listing-embeddings-for-similar-listing-recommendations-and-real-time-personalization-in-search-601172f7603e
12. Abdool et al., "Managing Diversity in Airbnb Search," arXiv 2004.02621, 2020. URL confirmed. https://arxiv.org/abs/2004.02621
13. "Learning To Rank Diversely," Airbnb Tech Blog, 2022. URL confirmed. https://medium.com/airbnb-engineering/learning-to-rank-diversely-add6b1929621
14. Grbovic, "Machine Learning-Powered Search Ranking of Airbnb Experiences," Airbnb Tech Blog, 2019. URL confirmed. https://medium.com/airbnb-engineering/machine-learning-powered-search-ranking-of-airbnb-experiences-110b4b1a0789
15. "Beyond A/B Test: Speeding up Airbnb Search Ranking Experimentation through Interleaving," Airbnb Tech Blog, 2022. URL confirmed. https://medium.com/airbnb-engineering/beyond-a-b-test-speeding-up-airbnb-search-ranking-experimentation-through-interleaving-7087afa09c8e
16. Simha, "Zipline: Airbnb's Machine Learning Data Management Platform" (2018) and "Zipline - A Declarative Feature Engineering Framework" (2020), talks, YouTube. URL confirmed. https://www.youtube.com/watch?v=Tg5VEMEsC-0 ; https://www.youtube.com/watch?v=LjcKCm0G_OY
17. "Discovering and Classifying In-app Message Intent at Airbnb," Airbnb Tech Blog, 2019. URL confirmed. https://medium.com/airbnb-engineering/discovering-and-classifying-in-app-message-intent-at-airbnb-6a55f5400a0c
18. "Intelligent Automation Platform: Empowering Conversational AI and Beyond at Airbnb," Airbnb Tech Blog, 2022. URL confirmed. https://medium.com/airbnb-engineering/intelligent-automation-platform-empowering-conversational-ai-and-beyond-at-airbnb-869c44833ff2
19. Ye et al., "Customized Regression Model for Airbnb Dynamic Pricing," KDD 2018. Unverified. https://dl.acm.org/doi/10.1145/3219819.3219830
20. "Embedding-Based Retrieval for Airbnb Search," Airbnb Tech Blog, and "Applying Embedding-Based Retrieval to Airbnb Search," arXiv 2601.06873, 2026. Unverified. https://airbnb.tech/ai-ml/embedding-based-retrieval-for-airbnb-search/ ; https://arxiv.org/abs/2601.06873
21. Tan et al., "Optimizing Airbnb Search Journey with Multi-task Learning," arXiv 2305.18431, 2023 (KDD 2023). Unverified. https://arxiv.org/abs/2305.18431
22. Haldar et al., "Learning to Rank for Maps at Airbnb," arXiv 2407.00091, 2024 (KDD 2024). Unverified. https://arxiv.org/abs/2407.00091
23. "LLM-Friendly Knowledge Representation for Customer Support," arXiv 2510.10331, 2025. Unverified. https://arxiv.org/abs/2510.10331
24. "Designing Experimentation Guardrails," Airbnb Tech Blog, 2021. URL confirmed. https://medium.com/airbnb-engineering/designing-experimentation-guardrails-ed6a976ec669
25. "Contextualizing Airbnb by Building Knowledge Graph," Airbnb Tech Blog, 2019. URL confirmed. https://medium.com/airbnb-engineering/contextualizing-airbnb-by-building-knowledge-graph-b7077e268d5a
26. "Airbnb Software Engineer interview process" (TrueInterview-derived; the standard design round prompts: homepage backend "with ranking signals", booking with availability search, pagination and indexing, a host dashboard for 100+ listings, and an added design round on a support ticketing system "with agent claim rules by language and location and first-response metrics"; account deduplication follow-up from two attributes to any number). kevin-2023-code/Tech-Interview-Questions, GitHub, 2026. Fetched. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/airbnb/interview-process-software-engineer.md
27. "Automation Platform v2: hybrid LLM conversational platform" (third-party summary of a 2024 Airbnb Tech Blog post), ZenML LLMOps database. Unverified. https://www.zenml.io/mlops-database/airbnb-chronon-internal-dataai-app-platform-conversational-ai-platform-automation-platform-v2-hybrid-llm-conversational
28. "Airbnb Adds AI-Powered Search and Social Features," Let's Data Science, 2026. Unverified. https://letsdatascience.com/news/airbnb-adds-ai-powered-search-and-social-features-dd4ce7ea
29. "Transforming Location Retrieval at Airbnb," arXiv 2408.13399, 2024. Unverified. https://arxiv.org/abs/2408.13399
30. "How AI Text Generation Models Are Reshaping Customer Support at Airbnb," Airbnb Tech Blog. URL confirmed (via [31]). https://medium.com/airbnb-engineering/how-ai-text-generation-models-are-reshaping-customer-support-at-airbnb-a851db0b4fa3
31. "A Curated List of ML System Design Case Studies" (300+ case studies, including 12 Airbnb entries). Engineer1999/A-Curated-List-of-ML-System-Design-Case-Studies, GitHub. Fetched. https://github.com/Engineer1999/A-Curated-List-of-ML-System-Design-Case-Studies
32. "Using Machine Learning to Predict Value of Homes On Airbnb," Airbnb Tech Blog, 2017. URL confirmed. https://medium.com/airbnb-engineering/using-machine-learning-to-predict-value-of-homes-on-airbnb-9272d3d4739d
33. "Building Airbnb Categories with ML and Human-in-the-Loop," Airbnb Tech Blog, 2022. URL confirmed. https://medium.com/airbnb-engineering/building-airbnb-categories-with-ml-and-human-in-the-loop-e97988e70ebb
34. "Fighting Financial Fraud with Targeted Friction," Airbnb Tech Blog, 2018. Unverified; article URL not confirmed, so only the blog index is given. https://medium.com/airbnb-engineering
35. "Categorizing Listing Photos at Airbnb," Airbnb Tech Blog, 2018. URL confirmed. https://medium.com/airbnb-engineering/categorizing-listing-photos-at-airbnb-f9483f3ab7e3
36. "Amenity Detection and Beyond: New Frontiers of Computer Vision at Airbnb," Airbnb Tech Blog, 2019. URL confirmed. https://medium.com/airbnb-engineering/amenity-detection-and-beyond-new-frontiers-of-computer-vision-at-airbnb-144a4441b72e
37. "WIDeText: A Multimodal Deep Learning Framework," Airbnb Tech Blog, 2020. URL confirmed. https://medium.com/airbnb-engineering/widetext-a-multimodal-deep-learning-framework-31ce2565880c
38. "Prioritizing Home Attributes Based on Guest Interest," Airbnb Tech Blog. URL confirmed (via [31]). https://medium.com/airbnb-engineering/prioritizing-home-attributes-based-on-guest-interest-3c49b827e51a
39. "How Airbnb Measures Future Value to Standardize Tradeoffs," Airbnb Tech Blog, 2021. URL confirmed. https://medium.com/airbnb-engineering/how-airbnb-measures-future-value-to-standardize-tradeoffs-3aa99a941ba5
40. "Artificial Counterfactual Estimation (ACE): Machine Learning-Based Causal Inference at Airbnb," Airbnb Tech Blog, 2022. URL confirmed. https://medium.com/airbnb-engineering/artificial-counterfactual-estimation-ace-machine-learning-based-causal-inference-at-airbnb-ee32ee4d0512
41. "Scaling Airbnb's Experimentation Platform," Airbnb Tech Blog, 2017. URL confirmed. https://medium.com/airbnb-engineering/https-medium-com-jonathan-parks-scaling-erf-23fd17c91166
42. "Improving Deep Learning for Ranking Stays at Airbnb," Airbnb Tech Blog. URL confirmed (via [31]). https://medium.com/airbnb-engineering/improving-deep-learning-for-ranking-stays-at-airbnb-959097638bde
