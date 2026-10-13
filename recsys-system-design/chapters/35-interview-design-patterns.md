# 35. The Recommender System Design Interview Playbook: Feeds, PYMK, Ads, E-commerce and Notifications

> Research note for this chapter: the shared web-search quota for this run was used up
> before this chapter started, and most blog hosts (eugeneyan.com, huyenchip.com,
> engineering.fb.com, research.google) did not resolve from the research sandbox. The
> sources read directly for this chapter are GitHub-hosted: the ByteByteGo "Machine
> Learning System Design Interview" reference repository [6], Chip Huyen's booklet and
> book repositories [3][4], Eugene Yan's design-doc template and applied-ml list
> [13][56], the community interview guide by Alireza Dirafzoon and contributors
> [7]-[12], and the open-sourced X/Twitter code [22][23]. Production facts (funnel sizes,
> parameter counts, latencies) are taken from the company-specific chapters of this Atlas
> (Chapters 1, 2, 16, 17, 19, 21, 23, 24, 32), whose authors read the primary sources.
> Those sources are cited directly here. All back-of-envelope figures are this chapter's
> own arithmetic, labelled as illustrative, and are not claims about any company.

## Overview

The "design a recommender" question is the most common ML system design interview
prompt. The ByteByteGo / Aminian-and-Xu book *Machine Learning System Design Interview*
gives six of its ten case studies to recommendation-shaped problems: video
recommendation, event recommendation, ad click prediction on social platforms, similar
listings on vacation-rental platforms, personalized news feed and People You May Know
[6]. The widely used open-source guide by Dirafzoon et al. lists the same families
(video/movie recs, PYMK, newsfeed ranking, ads serving and click prediction) as the core
of its question bank [7]. Chip Huyen's ML systems design booklet ends with 27 open-ended
design questions and organizes the answer around project setup, data pipeline, modeling
and serving [3].

This chapter turns those resources and the production designs documented elsewhere in
this Atlas into one playbook. It covers:

1. A reusable answer framework, with the industry references behind each step.
2. Back-of-envelope arithmetic for QPS, candidate counts, embedding memory and logs.
3. Six worked designs: YouTube/TikTok video feed, social news feed, People You May Know,
   ad click prediction, e-commerce recommendations, and notifications.
4. A comparison table, the follow-up questions interviewers ask most often, and the
   answers production systems give.

The guiding observation comes from Eugene Yan's survey of discovery systems: nearly every
industrial recommender and search system fits a 2x2 grid of offline versus online and
candidate retrieval versus ranking [1][2]. A good interview answer is a tailored
instance of that grid, with numbers.

## 35.1 The answer framework

### 35.1.1 Nine steps, mapped to time

The Dirafzoon guide breaks an ML design answer into nine steps [7]. Yan's design-doc
template [13] and Huyen's four-stage booklet [3] cover the same ground in different
words. The table below aligns them and suggests a 45-minute time split (the timings are
this chapter's suggestion, not from a source).

| Step | Guide [7] | Design doc [13] | Huyen [3] | Minutes |
|---|---|---|---|---|
| 1 | Problem formulation, clarify scope | Overview, motivation, requirements | Project setup | 5 |
| 2 | Metrics (offline and online) | Success metrics, guardrails | Project setup | 4 |
| 3 | Architecture (MVP) | High-level design | Serving | 6 |
| 4 | Data and labels | Methodology: data | Data pipeline | 5 |
| 5 | Features | Methodology: techniques | Data pipeline | 5 |
| 6 | Model development, offline eval | Techniques, validation | Modeling | 8 |
| 7 | Prediction service | Performance, infra | Serving | 5 |
| 8 | Online testing, deployment | Experimentation and validation | Serving | 3 |
| 9 | Scaling, monitoring, updates | Monitoring, risks | Serving | 4 |

Practical points from the sources:

- **Translate the business goal into an ML objective.** The guide's examples: input
  user_id, output a ranked list; "relevance" must be defined, for example a like or
  playing half of the video [8]. Huyen notes that recommender labels are *natural* (the
  user's reaction) but delayed, and the delay ("feedback loop length") drives how fast
  you can retrain [4].
- **Start simple and iterate**: heuristics, then logistic regression, then GBDT, then
  neural networks [7]. This mirrors Google's Rules of ML, which tells teams to launch a
  simple model and get the pipeline right first [42].
- **Batch, online or hybrid prediction** is an explicit decision, with Netflix's mix of
  batch-computed titles and online-composed rows as the stock example [7]. Huyen frames
  the trade-off as responsiveness to changing preferences versus inference latency [4].
- **Write down what you are not doing.** Yan's template has an explicit scope section and
  an "alternatives considered" appendix [13]. Saying "I'm deferring ads blending and
  creator-side fairness" is a strong signal in an interview.

### 35.1.2 The canonical architecture to draw first

Every case in this chapter starts from the same skeleton. Production systems (YouTube
[14], Instagram [17][18], Facebook Feed [20], X [21]) differ in the number of stages and
the model at each stage, not in the shape.

```
                OFFLINE                                ONLINE (per request)
 +-------------------------------------+   +-------------------------------------+
 | logs -> label join -> training data |   | request(user, context)              |
 |   |                                 |   |   |                                 |
 |   v                                 |   |   v                                 |
 | train retrieval (two-tower, graph)  |   | candidate generation (parallel):    |
 | train rankers (MTL DNN)             |   |   ANN(user emb) | graph | trending  |
 |   |                                 |   |   | follows | rules  -> 10^3-10^4 |
 |   v                                 |   |   v                                 |
 | item embeddings -> ANN index -------+-->| filter: seen, blocked, policy       |
 | batch features -> feature store ----+-->| light ranker            -> 10^2-10^3|
 | stream features (Kafka/Flink) ------+-->| heavy ranker (MTL)      -> scores   |
 +-------------------------------------+   | value fn + re-rank (diversity, biz) |
                ^                          |   -> top K (10^1)                   |
                |                          +-----------------|-------------------+
                +--- impression / engagement logs <----------+
```

Funnel sizes from public systems are a useful anchor. Google's course describes
candidate generation that "reduces billions of videos down to hundreds or thousands"
[43]. Instagram Explore retrieves up to 1,500 media, keeps the top 100 after early-stage
ranking, then runs late-stage ranking [19]. Facebook Feed starts from more than 1,000
candidate posts per visit and prunes to about 500 before heavy ranking [20]. X's 2023
pipeline draws about 1,500 candidates (approximate) from about 500 million daily tweets
[21]. Chapter 1 summarises the pattern: retrieval leaves 10^3-10^4 candidates,
pre-ranking 10^2-10^3, and ranking 10^1-10^2.

## 35.2 Back-of-envelope numbers

Interviewers want sizing that drives decisions, not precision. The numbers below are
illustrative arithmetic, with the assumptions shown.

### 35.2.1 Traffic

- Seconds per day: about 86,400, so 1 billion requests per day is about 11.6K QPS on
  average. Use 2-3x for peak (an assumption, not a sourced constant).
- The Dirafzoon newsfeed case assumes 2B DAU, each refreshing twice a day, with a 200 ms
  latency target [9]. That is 4B feed requests per day, or about 46K QPS average and
  roughly 100K+ at peak.
- The video case assumes 100M DAU and 100M videos [8]. At 10 feed loads per user per
  day that is 1B requests per day, about 12K QPS.
- X's 2023 blog reports its For You pipeline runs about 5 billion times a day and
  completes in under 1.5 s on average [21]. That is about 58K pipeline runs per second
  on average.

### 35.2.2 Scoring load

- Heavy-ranker load = QPS x candidates per request. At 12K QPS and 500 candidates per
  request, that is 6M item scorings per second. This explains why heavy rankers see only
  hundreds of items and why a light ranker exists.
- Instagram reported in 2019 that Explore "extracts 65 billion features and makes 90
  million model predictions every second" [17]. That is a real anchor for "how big can
  this get."
- X's 2023 heavy ranker was a neural network of about 48M parameters using about 6,000
  features and predicting about 10 engagement labels [21][22]. A single pipeline
  execution consumed about 220 s of CPU time [21], so feature hydration and model
  inference, not ANN search, dominate cost.

### 35.2.3 Memory

- Item embeddings: 100M items x 128 dims x 4 bytes (fp32) is about 51 GB. fp16 halves
  it to about 26 GB and int8 to about 13 GB. A single large host can hold that index in
  memory, but a sharded ANN service is needed once you add replicas and freshness.
- User embeddings computed on request (two-tower user tower) need no storage. Precomputed
  user vectors for 1B users at 128 dims fp16 come to about 256 GB, which belongs in a
  key-value store.
- Sparse ID embedding tables dominate ranking-model size. Google's 2013 ads system found
  memory, not FLOPs, was the binding constraint. It used Poisson and Bloom-filter
  feature admission and 16-bit weights to save roughly 40-75% of RAM at negligible
  AucLoss [24].

### 35.2.4 Logs and training data

- Impressions per day = requests x items shown. 1B requests x 20 items gives 20B
  impressions per day. At about 1 KB per logged impression with features, that is about
  20 TB per day.
- Negative downsampling is standard. Facebook's ads paper keeps negatives at rate w and
  recalibrates with q = p / (p + (1 - p) / w) [25].
- YouTube's 2016 ranker trained on hundreds of billions of examples with roughly one
  billion parameters (approximate figures from the paper and summaries) [14].

### 35.2.5 Latency budget

A common split of a 200-300 ms p99 budget (a heuristic from Chapter 10, not one company's
numbers):

| Stage | Share of p99 | Dominant cost |
|---|---|---|
| User/query hydration | 5-15% | KV lookups, user sequence fetch |
| Candidate generation (parallel) | 15-25% | ANN, slowest source |
| Feature hydration | 20-35% | fan-out RPCs, tail latency |
| Light ranking | 5-10% | CPU, small model, many items |
| Heavy ranking | 20-35% | GPU compute, batching delay |
| Re-rank, filters, mixing | 5-10% | rules, diversity, ad auction |

Uber Eats found that hydration was the real hotspot. Retrieval returned tens of
thousands of candidates that were fully hydrated before ranking discarded most of them.
Removing low-yield retrieval paths saved about 120 ms [50]. This makes a good interview
point: candidate *count* is a latency lever, not just a recall lever.

## 35.3 Design 1: YouTube / TikTok video recommendation feed

### Problem and clarifying questions

Homepage or For You feed recommendations, personalised per user, worldwide and
multilingual, aimed at engagement [8]. Ask about: home feed versus watch-next, long-form
versus short-form, whether search history is available, the latency target (the guide
suggests about 200 ms), and whether creator-side goals matter [8].

### Metrics

- Offline: precision@k, mAP and diversity for the list. Recall@k for retrieval. AUC or
  log loss per head for the ranker [8].
- Online: CTR, completed videos, total watch time, explicit feedback [8]. Guardrails:
  dislikes, "not interested", reports, session abandonment.
- Long-term: Google found short-term ad experiments mislead without long-term holdouts
  [47]. Netflix similarly writes about recommending for long-term member satisfaction
  rather than immediate engagement [55]. Mentioning user-level retention or satisfaction
  surveys sets a senior answer apart.

### Architecture

```
 user_id + context
      |
      v
 +------------ CANDIDATE GENERATION (parallel, ~1-2K total) ------------+
 | two-tower ANN (recent watches) | co-watch / item-to-item | subs      |
 | trending / fresh (explore)     | search-history sources  | multi-int |
 +----------------------------------------------------------------------+
      |
      v   filter: watched, blocked, region/age policy, dedupe
 light ranker (distilled two-tower or small DNN)  ->  ~100-500
      |
      v
 heavy ranker: multi-task DNN (MMoE) -> p(click), E[watch time], p(like), p(share),
               p(dislike), p(survey satisfied); shallow position-bias tower
      |
      v
 value function  sum_i w_i * pred_i  ->  re-rank: diversity (DPP / MMR), freshness,
                                         creator caps, policy  ->  top ~10-20
```

### Models and the key tricks

- **Candidate generation as extreme multiclass classification.** YouTube 2016 averages
  embeddings of watched videos and search tokens, feeds them into an MLP, and trains a
  sampled softmax over the corpus. At serving time it uses nearest-neighbour search on
  the softmax output vectors [14]. Two refinements matter in interviews. The **example
  age** feature lets the model learn a freshness preference. Training uses watches from
  everywhere (not just recommendations) with a fixed number of examples per user, so
  heavy users do not dominate [14].
- **Two-tower retrieval with in-batch negatives and logQ correction.** In-batch softmax
  over-penalises popular items. Subtracting log p_j (estimated with streaming frequency
  counts) corrects the bias. Google deployed this on a YouTube corpus of tens of millions
  of videos [15].
- **Ranking on watch time, not clicks.** YouTube trained weighted logistic regression
  with positives weighted by watch time, so the odds approximate expected watch time
  [14]. This answers "how do you avoid clickbait?"
- **Multi-task ranking with position debiasing.** The 2019 Watch Next ranker uses MMoE
  for engagement and satisfaction objectives. A shallow tower takes position and device
  features and adds its logit, which absorbs selection bias. At serving time position is
  set to missing [16].
- **Diversity.** Google's course lists freshness, diversity and fairness as re-ranking goals
  [38]. YouTube re-ranks with a determinantal point process over quality scores
  and embedding distances [48].
- **Beyond supervised imitation.** A REINFORCE candidate generator with top-K off-policy
  correction was one of YouTube's largest single launches in its period, with roughly
  0.8-0.9% more viewing time (approximate) [49].
- **Short-video specifics (TikTok, Kuaishou).** Signals arrive in seconds and interests
  shift within a session, so real-time training matters. ByteDance's Monolith uses
  collisionless, expirable hash embedding tables and minute-level parameter sync from
  online training to serving [44]. Kuaishou's OneRec goes further and replaces the
  retrieve-then-rank cascade with one generative model [45]. In an interview, treat it
  as a "what's next" follow-up, not the baseline.

### Cold start and exploration

New users get profile and context features through the user tower. New videos get a
small, randomised exposure budget, then fine-tuning on the collected data, with bandits
for explore/exploit [8]. Content embeddings (title, thumbnail, audio) let the item tower
place a new video near similar ones before any interaction data exists.

### Lessons

Optimise watch time or satisfaction rather than clicks. Correct sampling and position
bias explicitly. Keep the item corpus fresh through example age, streaming updates and
exploration slots. Most of the gain sits in retrieval coverage plus ranker objectives,
not in model depth.

## 35.4 Design 2: Social news feed (Facebook, LinkedIn, X)

### Problem

Show unseen recent posts from connections and (increasingly) recommended accounts,
ranked by predicted value, within about 200 ms. The guide's sizing is 2B DAU, two
refreshes a day each, with privacy constraints [9].

### Architecture: Facebook's four passes

Meta's 2021 engineering post describes Feed ranking as passes run on serving machines
it calls *predictors* [20]:

```
 inventory: all eligible posts for viewer (> 1,000 on average)
   -> pass 0: integrity + eligibility filters
   -> pass 1: lightweight model -> ~500 posts
   -> pass 2: multitask NNs predict P(like), P(comment), P(share), P(hide),
              P("worth your time" survey) ... -> one value score per post
   -> pass 3: contextual re-ranking (content-type diversity, no runs of similar posts)
```

The value function is a weighted sum, value(post) = sum_i w_i * P(event_i). The weights
are a product decision informed by surveys about what people find meaningful, not a
learned output [20]. The guide's interview version ranks weights as share > comment >
like > click and adds heads for dwell time and P(skip) to serve passive users [9].

### X's 2023 open-sourced pipeline

Twitter had moved timeline ranking to deep learning by 2017 [53]. Its 2023 Home Mixer is
the most detailed public feed implementation [21][22]:

- Candidate sourcing: about 1,500 candidates (approximate), roughly half in-network
  (Earlybird search index, scored by a light ranker) and half out-of-network (social
  graph walks via UTEG/GraphJet, SimClusters community embeddings, TwHIN embeddings) [21].
- Heavy ranker: about 48M-parameter multi-task network (MaskNet), about 6,000 features,
  about 10 engagement labels, combined with configured weights [21][22].
- Heuristics and filters: author diversity, content balance between in- and
  out-of-network, feedback fatigue, visibility filtering [21].
- Mixing: ads, who-to-follow modules and prompts are inserted into the timeline [22].
- Resilience: Home Mixer keeps a reverse-chronological backup pipeline in case scored
  tweets fail (Chapter 10, [22]).

### Key design decisions to discuss

1. **Fan-out on write versus fan-out on read** for candidate inventory. Precomputed
   per-user inboxes make reads cheap but are expensive for celebrity authors. Pull-time
   retrieval from an index (Earlybird) handles high-fanout authors. A hybrid is usual.
   (This is the classic systems-design half of the question.)
2. **Separate prediction from policy.** Calibrated multi-task heads plus a small,
   versioned weight vector let product change trade-offs without retraining [20][21].
3. **Dwell time and passive users.** LinkedIn's dwell-time work is a standard reference
   for adding implicit signals beyond clicks [52]. The guide's skip head does the same
   [9].
4. **Integrity first.** Filter violating content before heavy ranking, to save compute
   and avoid amplifying it [20].
5. **Freshness.** Feature pipelines are split: static features computed in batch (daily
   or weekly) into a feature store, and dynamic features (recent clicks, post engagement
   velocity) streamed [9].

### Follow-ups specific to feeds

Viral and celebrity posts, cold-start users, position bias, calibration, data leakage
and catastrophic forgetting during continual fine-tuning are all listed as expected
follow-ups [9]. Have one sentence ready for each.

## 35.5 Design 3: People You May Know (friend / follow / connection recommendation)

### Problem

Recommend users to connect with, to maximise connections formed. The guide's sizing is
1B users with an average of 1,000 connections each (its text reads "10000", but its own
candidate arithmetic of 1,000 x 1,000 = 1M friends-of-friends implies 1,000) [10]. The
objective is two-sided: a suggestion is only good if the *other* person accepts.

### Architecture

```
 OFFLINE (daily / hourly)                          ONLINE
 graph snapshot at t ----------------------+      request(user)
   -> FoF expansion (2-hop) ~10^5-10^6     |        -> read precomputed PYMK list
   -> + EBR: GNN embeddings -> ANN         |        -> filter: already connected,
   -> + co-attributes: school, company,    |           blocked, pending, privacy
        contacts, location                 |        -> light re-score with fresh
   -> ranker: p(send) * p(accept)          |           signals (recent views)
   -> top ~100 per user -> KV store -------+------> -> diversity, frequency caps
```

### Candidate generation

- **Triangle closing / friends-of-friends** is the classic generator. It reduces 1B users
  to about 1M FoF candidates per user in the guide's arithmetic [10]. In practice you
  keep the top FoF by weighted mutual-friend counts.
- **Random walks.** Graph walks scale well: Pinterest's Pixie serves random-walk
  recommendations over 3+ billion items in real time [37]. X's Follow Recommendations Service uses SALSA-style personalised
  random walks with restarts over a local bipartite graph built from address book,
  mutual follows and recent follows. It ranks by a weighted sum of p(follow | rec) and
  p(positive engagement | follow) (Chapter 7, [22]).
- **Embedding-based retrieval beyond two hops.** Snap added GNN-embedding EBR for
  friend recommendation across billions of users. It reaches candidates that FoF misses,
  especially for new users in sparse markets, and improved friendships made in A/B
  tests [33]. A follow-up issuing several ANN queries per user (query expansion) gave an
  11% relative gain in reciprocated-communication metrics at more than 800M MAU [34].

### Models

- Pairwise classifier on (user_i, user_j) with features such as mutual connections
  (time-discounted), shared school or company, profile similarity and interaction
  history; or a GNN that produces node embeddings and scores edges by similarity [10].
- Training data: take a graph snapshot at time t and label pairs that connect by t+1 as
  positive [10]. This temporal split avoids leakage.
- LinkedIn reports its LiGNN graph embeddings feed PYMK among other surfaces, with about
  +0.1% weekly active users attributed to people recommendations [36].

### Serving

PYMK is the best example of **batch prediction**. The friend graph changes slowly, so
precompute ranked lists for active users and store them in a key-value store [10], with
a light online re-score for freshness. Fresh signals (a just-accepted connection opens
new triangles) can trigger incremental recomputation for that user.

### Follow-ups

- **Delayed feedback**: invitations are accepted days later, so labels need waiting
  windows or delayed-feedback correction [10][27].
- **Reciprocity and fairness**: optimise p(send) x p(accept), and avoid
  rich-get-richer. LinkedIn has written about optimising PYMK for equity in network
  creation [35].
- **Privacy**: do not reveal sensitive edges, for example contact-book-derived
  suggestions [10].

## 35.6 Design 4: Ad click prediction (pCTR) and ads ranking

### Problem

Predict p(click | user, ad, context) for ads shown on a social feed, to maximise revenue
while protecting user experience [11]. It is pointwise learning-to-rank over
<user, ad> pairs. A click within t seconds of the impression is a positive [11].

### What makes ads different

1. **The output feeds an auction**, so predictions must be **calibrated probabilities**,
   not just well ordered. Ranking typically uses expected value, bid x pCTR (x pCVR for
   conversion objectives) plus quality terms. Every ads paper since 2013 treats
   calibration as a first-class output [24][25].
2. **Metrics**: normalized entropy (NE), which is log loss divided by the entropy of the
   background CTR, plus calibration ratio (sum predicted / sum observed) per slice
   [24][25]. The guide lists cross-entropy and normalized cross-entropy offline, and
   CTR, conversion rate, revenue lift and hide rate online [11].
3. **Freshness is worth money.** Facebook found NE degrades as training data falls days
   behind serving [25]. This motivates online or continual training.

### Architecture

```
 ad request (user, placement)
   -> eligibility: targeting (age, geo, interests), budget/pacing, policy
   -> retrieval: tens of millions of eligible ads -> a few thousand      (Andromeda-like)
   -> light ranker -> hundreds
   -> heavy ranker: pCTR, pCVR, p(hide) heads; calibration layer
   -> auction: rank by bid * pCTR * pCVR + quality; price (GSP/VCG-style)
   -> blend with organic feed (ad load, spacing)
 logs -> online joiner (impression + click within window) -> streaming trainer
```

Secondary coverage of Meta's Andromeda post reports that ads retrieval narrows tens of
millions of eligible ads to a few thousand candidates per impression before ranking and
auction [28]. In 2025 Meta described GEM, an LLM-scale ads foundation model used to
transfer knowledge into the served models rather than to score every candidate inline
[29].

### Models: the expected progression

The guide recommends starting with logistic regression and moving to DCN and DeepFM [11].
The ByteByteGo chapter's references cover the same path: GBDT feature extraction, DCN
and DCN-V2, Deep Crossing, FM and DeepFM, FFM, and delayed-feedback correction [6].

| Model | Why it appears | Weakness raised in interviews |
|---|---|---|
| LR + hand crosses (FTRL) [24] | sparse, fast, online-updatable, calibrated | manual feature crosses |
| GBDT + LR [25] | trees as learned feature transform; >3% NE gain | trees retrain slowly; no embeddings |
| FM / FFM [6] | pairwise interactions on sparse IDs | only second order |
| Wide & Deep [54], DeepFM [6] | memorisation + generalisation | heavier serving |
| DCN-V2 [26] | explicit learned crosses, low-rank for cost | tuning cross depth |
| DIN-style target attention [31] | user-history attention to candidate ad | sequence fetch latency |
| Foundation model + distillation [29] | scale beyond per-request budget | infra cost, transfer design |

GBDT is interpretable but inefficient for continual training and cannot learn
embeddings. Plain DNNs struggle with very sparse features without explicit interaction
structure [11].

### Key tricks

- **Online joiner and label windows.** Join impressions and clicks by request ID inside
  a waiting window. Too short mislabels late clicks as negatives, too long delays
  training [25]. Twitter's delayed-feedback work trains continuously with fake negatives
  and corrects with an importance-weighted or FN-calibrated loss [27].
- **Negative downsampling with recalibration** (formula in 35.2.4) [25].
- **Memory tricks** for huge sparse feature spaces: probabilistic feature admission and
  low-precision weights [24].
- **Conversion prediction (pCVR)** suffers sample-selection bias because conversions are
  only observed after clicks. ESMM models p(click) x p(conversion | click) over the
  entire impression space [32].

### Follow-ups

Calibration after downsampling; leakage from post-impression features; catastrophic
forgetting under continual training; ad fatigue and negative feedback (hide/block);
budget pacing; cold-start ads (use advertiser and campaign-level priors and content
embeddings) [11].

## 35.7 Design 5: E-commerce recommendations

### Problem variants

E-commerce interviews usually narrow to one surface:

- "Customers who viewed/bought X also bought" (item-to-item, product page).
- Personalised homepage ("recommended for you").
- Complementary products / cart add-ons.
- Similar listings (Airbnb-style), which ByteByteGo treats as its own chapter [6].

Business metrics are conversion rate, GMV or revenue per session and add-to-cart rate.
CTR alone is a weak proxy.

### Item-to-item collaborative filtering (the baseline that still wins)

Amazon's 2003 design computes item-item cosine similarity offline over customer purchase
vectors. At request time it only looks up similar items for the user's recent items, so
online cost depends on the size of the user's history, not on catalogue or customer count
[30]. Worst-case offline cost is O(N^2 M), but sparsity makes it close to O(NM) in
practice (Chapter 24).

```
 OFFLINE: orders/views -> item-item co-occurrence/cosine -> similar-items table (KV)
 ONLINE : user history (viewed, carted, bought) -> lookup neighbours -> merge, dedupe,
          remove purchased -> rank (personal DNN, price, availability) -> slate
```

### Session embeddings and similar listings

Airbnb trained listing embeddings word2vec-style on click sessions. It used booked
listings as global context and in-market negatives, and applied them to similar-listing
recommendations and real-time search personalisation [39]. This is ByteByteGo's
reference design for the similar-listings chapter [6]. Interview points: sessions as
"sentences", in-market negative sampling so the model learns within-city distinctions,
and the cold-start rule of averaging embeddings of nearby similar listings.

### Personalised ranking

- **Target attention over behaviour sequences.** Alibaba's DIN weights each past
  behaviour by relevance to the candidate item instead of averaging history [31]. This
  is the standard answer to "users have diverse interests."
- **Multi-objective**: p(click), p(add-to-cart), p(purchase), with ESMM-style modelling
  to handle the click-then-buy funnel [32].
- **Business re-ranking**: in-stock and deliverable filters, price-band diversity,
  sponsored-product slots, and suppressing items already bought (except consumables).

### Follow-ups

Substitutes versus complements (do not recommend a second TV after a TV purchase); cold
start for new SKUs (content embeddings from title, image and attributes); seasonality;
popularity bias; and real-time session intent versus long-term profile.

## 35.8 Design 6: Notification and push recommendation

### Problem

Decide **whether**, **what**, **when** and **how often** to notify a user (Chapter 32).
The user is not present, so there is no request. The system is event-driven (a friend
posted) or schedule-driven (a digest). Every send has a hidden cost: a bad push can lead
to disabled notifications or an uninstall, which are near-permanent channel losses.

### Architecture (from X's open-sourced pushservice)

X's `pushservice` README describes a RefreshForPushHandler flow [23]:

```
 user id -> Target (eligibility, settings) -> query candidate sources
   -> hydrate candidates -> light filters
   -> LIGHT RANKER (pre-selects from the large pool)
   -> HEAVY RANKER (multi-task: p(open), p(engage), negative-feedback risk)
   -> take step: run expensive predicates (quality, fatigue) one candidate at a time
      until ONE passes -> send push + in-app notification
```

The "take" step is the design point worth copying. Expensive checks, such as quality
predicates and fatigue predicates, run lazily on the ranked list until one candidate
passes, which bounds cost per user (Chapter 32, [23]).

### Models and policy

- **Ranker**: multi-task p(open), p(downstream session), p(disable or unsubscribe).
- **Send policy**: send if value(candidate) - cost(fatigue, disable risk) > threshold.
  The threshold can be per user. Pinterest's notification volume control and LinkedIn's
  near-real-time activity-notification optimisation are the standard references
  [40][41] (DOIs not re-verified in this run; see Chapter 32).
- **Volume and timing**: per-user daily budgets, quiet hours, time-zone-aware
  send-time optimisation, and dedup across channels (push, email, in-app).

### Metrics and experimentation

Optimise long-term metrics (DAU/WAU, sessions over weeks) with disables and
unsubscribes as guardrails. Notification effects are heavily confounded, because active
users get more notifications, so randomised volume holdouts are required (Chapter 32).
Short-term open rate is a trap, in the same way short-term ad CTR misleads [47].

## 35.9 Cross-design comparison

| Design | Primary objective | Candidate gen | Ranker | Serving mode | Main pitfall |
|---|---|---|---|---|---|
| Video feed [8][14][16] | watch time, satisfaction | two-tower ANN, co-watch, fresh | MMoE MTL + position tower | online, real-time features | clickbait, popularity bias |
| News feed [9][20][21] | weighted engagement value | in-network index + OON graph/emb | MTL heads + value fn | online, fan-out hybrid | engagement-only weights |
| PYMK [10][33] | connections formed (both sides) | FoF, random walks, GNN EBR | pairwise / GNN edge score | batch precompute + refresh | delayed accept, rich-get-richer |
| Ad click [11][24][25] | revenue (calibrated pCTR) | targeting + ANN ad retrieval | LR/GBDT+LR -> DCN/DIN | online, continual training | miscalibration, label delay |
| E-commerce [30][31][39] | conversion, GMV | item-to-item, session emb | DIN-style, multi-objective | hybrid: batch tables + online | substitutes vs complements |
| Notifications [23] | long-term DAU minus disables | event/schedule sources | light + heavy MTL + policy | async batch/stream | fatigue, confounding |
| Events [12] | registrations / tickets | rules (location, time) | GBDT -> NN | online | perpetual cold start |

## 35.10 Common follow-up questions and strong answers

| Follow-up | Strong answer (with source) |
|---|---|
| How do you handle cold-start items? | Content features in the item tower; exploration budget; bandits [8]; example-age feature [14] |
| How do you handle position bias? | Shallow position tower added to logits, position missing at serving [16]; randomised slots; IPS weighting [46] |
| Your two-tower favours popular items. Why? | In-batch negatives oversample popular items; apply logQ correction [15]; mix in random negatives |
| How do you combine multiple objectives? | Calibrated MTL heads + configured weights [20][21]; tune weights online |
| Model is stale; what now? | Measure NE vs data age [25]; streaming features; continual or online training [44]; Huyen's continual learning [5] |
| How do you prevent feedback loops? | Exploration traffic, off-policy correction [49], diversity re-ranking [48]; Huyen names degenerate feedback loops as a system design issue [4] |
| Offline gain, online loss? | Check train-serving skew and leakage [9][11]; log features at serving time [1]; use A/B tests with guardrails |
| How do you ensure diversity? | DPP or MMR re-rank [48]; author/creator caps [21] |
| Labels arrive late (conversions, accepts)? | Waiting windows, fake-negative correction [27] |
| Scale to 10x traffic? | Cut candidates (hydration dominates) [50]; distil heavy ranker; cache user embeddings; quantise ANN index |
| What would you do with LLMs? | Content embeddings and semantic IDs for cold start; generative retrieval and one-model ranking as the frontier [45][51] |

## 35.11 Anti-patterns interviewers flag

- **Jumping to a model before defining the label.** "Relevance" must be operationalised:
  watched more than half, registered, accepted [8][10][12].
- **One-stage design for a large corpus.** Scoring 100M items with a heavy model per
  request is infeasible. Derive the funnel from the latency budget [43].
- **Optimising CTR only.** Every production source above moves to watch time,
  satisfaction surveys or weighted value functions [14][16][20].
- **Random train/test splits on time-ordered data.** Use time-based splits [7][10].
- **Ignoring calibration** in ads or in any value function that sums probabilities
  [24][25].
- **Ignoring the data pipeline.** Yan's 2x2 grid puts as much weight on offline data,
  embedding and index jobs as on the online path [1].

## Key takeaways for system designers

1. **Draw the funnel first, then size it.** Corpus size and latency budget set the number
   of stages. Public systems cluster at about 1,000-1,500 candidates entering ranking
   (Instagram [19], X [21], Facebook [20]).
2. **Do the arithmetic out loud.** Requests per day / 86,400 = QPS; QPS x candidates =
   scoring load; items x dims x bytes = index memory; impressions x bytes = log volume.
   Each number should justify a design choice.
3. **Separate prediction from policy.** Calibrated multi-task heads plus a configured
   value function is the dominant production pattern for feeds, ads and notifications
   [20][21][23].
4. **Choose serving mode per problem.** PYMK and item-to-item tables are batch-friendly
   [10][30]. Feeds and ads need online scoring with streaming features [9][25]. Notifications
   are asynchronous decision problems [23].
5. **Name the biases and fix them.** Sampling bias (logQ) [15], position bias (shallow
   tower) [16], selection bias in conversions (ESMM) [32] and feedback loops
   (exploration, off-policy correction) [49].
6. **Plan for freshness and delayed labels.** Online joiners, continual training and
   delayed-feedback correction are core, not extras [25][27][44].
7. **Measure the long term.** Use holdouts and long-term metrics, because short-term CTR
   and open-rate wins can hurt retention [47][55].
8. **Know the frontier but lead with the baseline.** Generative recommenders (HSTU,
   OneRec) are strong follow-up material [45][51]. The interview answer should still
   start from retrieval + ranking + re-ranking.

## References

1. Eugene Yan. "System Design for Recommendations and Search." eugeneyan.com, 2021. https://eugeneyan.com/writing/system-design-for-discovery/
2. Eugene Yan. "System Design for Discovery (RecSys and Search)." ApplyingML. https://applyingml.com/resources/discovery-system-design/
3. Chip Huyen. "Machine Learning Systems Design" (booklet source; four steps: project setup, data pipeline, modeling, serving; 27 open-ended questions). GitHub, 2019-2020. https://github.com/chiphuyen/machine-learning-systems-design
4. Chip Huyen. "Designing Machine Learning Systems" (book repository, chapter summary). O'Reilly / GitHub, 2022. https://github.com/chiphuyen/dmls-book
5. Chip Huyen. "Real-time machine learning: challenges and solutions." huyenchip.com, 2022 (cited from [6]). https://huyenchip.com/2022/01/02/real-time-machine-learning-challenges-and-solutions.html
6. ByteByteGo (Ali Aminian and Alex Xu). "Machine Learning System Design Interview" chapter reference list. GitHub, ByteByteGoHq/ml-bytebytego, 2023. https://github.com/ByteByteGoHq/ml-bytebytego
7. Alireza Dirafzoon et al. "ML System Design" (9-step framework and question bank), Machine-Learning-Interviews. GitHub. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/ml-system-design.md
8. Alireza Dirafzoon et al. "Video Recommendation System" (mlsd-video-recom). GitHub. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/mlsd-video-recom.md
9. Alireza Dirafzoon et al. "Newsfeed ranking" (mlsd-newsfeed). GitHub. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/mlsd-newsfeed.md
10. Alireza Dirafzoon et al. "People You May Know" (mlsd-pymk). GitHub. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/mlsd-pymk.md
11. Alireza Dirafzoon et al. "Ads click prediction / ads ranking" (mlsd-ads-ranking). GitHub. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/mlsd-ads-ranking.md
12. Alireza Dirafzoon et al. "Event Recommendation System" (mlsd-event-recom). GitHub. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/mlsd-event-recom.md
13. Eugene Yan. "ml-design-docs: A template for ML design docs." GitHub. https://github.com/eugeneyan/ml-design-docs
14. Paul Covington, Jay Adams, Emre Sargin (Google). "Deep Neural Networks for YouTube Recommendations." RecSys 2016. https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/
15. Xinyang Yi et al. (Google). "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations." RecSys 2019. https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/
16. Zhe Zhao et al. (Google). "Recommending What Video to Watch Next: A Multitask Ranking System." RecSys 2019. https://dl.acm.org/doi/10.1145/3298689.3346997
17. Ivan Medvedev, Haotian Wu, Taylor Gordon (Instagram). "Powered by AI: Instagram's Explore recommender system." Instagram Engineering, 2019. https://instagram-engineering.com/powered-by-ai-instagrams-explore-recommender-system-7ca901d2a882
18. Vladislav Vorotilov, Ilnur Shugaepov (Meta). "Scaling the Instagram Explore recommendations system." Engineering at Meta, 2023. https://engineering.fb.com/2023/08/09/ml-applications/scaling-instagram-explore-recommendations-system/
19. Meta Transparency Center. "Instagram Explore AI system." Accessed 2026. https://transparency.meta.com/features/explaining-ranking/ig-explore/
20. Akos Lada, Meihong Wang, Tak Yan (Meta). "How machine learning powers Facebook's News Feed ranking algorithm." Engineering at Meta, 2021. https://engineering.fb.com/2021/01/26/ml-applications/news-feed-ranking/
21. Twitter Engineering. "Twitter's Recommendation Algorithm." Twitter (X) Engineering Blog, 2023. https://blog.x.com/engineering/en_us/topics/open-source/2023/twitter-recommendation-algorithm
22. Twitter. "the-algorithm" repository (home-mixer, follow-recommendations-service READMEs). GitHub, 2023. https://github.com/twitter/the-algorithm
23. Twitter. "pushservice" README (RefreshForPushHandler, light and heavy rankers). GitHub, 2023. https://github.com/twitter/the-algorithm/tree/main/pushservice
24. H. Brendan McMahan et al. (Google). "Ad Click Prediction: a View from the Trenches." KDD 2013. https://research.google/pubs/ad-click-prediction-a-view-from-the-trenches/
25. Xinran He et al. (Facebook). "Practical Lessons from Predicting Clicks on Ads at Facebook." ADKDD 2014. https://quinonero.net/Publications/predicting-clicks-facebook.pdf
26. Ruoxi Wang et al. (Google). "DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems." WWW 2021. https://arxiv.org/abs/2008.13535
27. Sofia Ira Ktena et al. (Twitter). "Addressing Delayed Feedback for Continuous Training with Neural Networks in CTR Prediction." RecSys 2019. https://arxiv.org/abs/1907.06558
28. Meta Engineering. "Meta Andromeda: Supercharging Advantage+ automation with the next-gen personalized ads retrieval engine." 2024 (figures via secondary coverage, https://www.jonloomer.com/meta-andromeda/). https://engineering.fb.com/2024/12/02/production-engineering/meta-andromeda-advantage-automation-next-gen-personalized-ads-retrieval-engine/
29. Engineering at Meta. "Meta's Generative Ads Model (GEM): The central brain accelerating ads recommendation AI innovation." 2025. https://engineering.fb.com/2025/11/10/ml-applications/metas-generative-ads-model-gem-the-central-brain-accelerating-ads-recommendation-ai-innovation/
30. Greg Linden, Brent Smith, Jeremy York (Amazon). "Amazon.com Recommendations: Item-to-Item Collaborative Filtering." IEEE Internet Computing, 2003. https://www.cs.umd.edu/~samir/498/Amazon-Recommendations.pdf
31. Guorui Zhou et al. (Alibaba). "Deep Interest Network for Click-Through Rate Prediction." KDD 2018. https://arxiv.org/abs/1706.06978
32. Xiao Ma et al. (Alibaba). "Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate." SIGIR 2018. https://arxiv.org/abs/1804.07931
33. Snap Research. "Embedding Based Retrieval in Friend Recommendation." SIGIR 2023. https://research.snap.com/publications/embedding-based-retrieval-in-friend-recommendation.html
34. Snap Research. "Improving Embedding-Based Retrieval in Friend Recommendation with ANN Query Expansion." https://research.snap.com/publications/improving-embedding-based-retrieval-in-friend-recommendation-with-ann-query-expansion.html
35. LinkedIn Engineering. "Optimizing PYMK for equity in network creation." LinkedIn Engineering Blog, 2021 (cited via [6]). https://engineering.linkedin.com/blog/2021/optimizing-pymk-for-equity-in-network-creation
36. Fedor Borisyuk et al. (LinkedIn). "LiGNN: Graph Neural Networks at LinkedIn." KDD 2024. https://arxiv.org/abs/2402.11139
37. Chantat Eksombatchai et al. (Pinterest). "Pixie: A System for Recommending 3+ Billion Items to 200+ Million Users in Real-Time." WWW 2018. https://arxiv.org/pdf/1711.07601
38. Google for Developers. "Re-ranking: freshness, diversity and fairness" (Recommendation Systems course, cited via [6]). https://developers.google.com/machine-learning/recommendation/dnn/re-ranking
39. Mihajlo Grbovic et al. (Airbnb). "Listing Embeddings in Search Ranking." Airbnb Engineering, 2018 (cited via [6]). https://medium.com/airbnb-engineering/listing-embeddings-for-similar-listing-recommendations-and-real-time-personalization-in-search-601172f7603e
40. Yan Gao et al. (LinkedIn). "Near Real-time Optimization of Activity-based Notifications." KDD 2018 (DOI not re-verified). https://dl.acm.org/doi/10.1145/3219819.3219880
41. Bo Zhao et al. (Pinterest). "Notification Volume Control and Optimization System at Pinterest." KDD 2018 (DOI not re-verified). https://dl.acm.org/doi/10.1145/3219819.3219906
42. Martin Zinkevich (Google). "Rules of Machine Learning: Best Practices for ML Engineering." https://developers.google.com/machine-learning/guides/rules-of-ml
43. Google for Developers. "Recommendation systems overview." Machine Learning: Recommendation Systems course. https://developers.google.com/machine-learning/recommendation/overview/types
44. Zhuoran Liu et al. (ByteDance). "Monolith: Real Time Recommendation System With Collisionless Embedding Table." 2022. https://arxiv.org/abs/2209.07663
45. Kuaishou OneRec Team. "OneRec Technical Report." 2025. https://arxiv.org/abs/2506.13695
46. Eugene Yan. "How to Measure and Mitigate Position Bias." eugeneyan.com (cited via [6]). https://eugeneyan.com/writing/position-bias/
47. Henning Hohnhold, Deirdre O'Brien, Diane Tang (Google). "Focusing on the Long-term: It's Good for Users and Business." KDD 2015. https://dl.acm.org/doi/10.1145/2783258.2788583
48. Mark Wilhelm et al. (Google/YouTube). "Practical Diversified Recommendations on YouTube with Determinantal Point Processes." CIKM 2018. https://dl.acm.org/doi/10.1145/3269206.3272018
49. Minmin Chen et al. (Google/YouTube). "Top-K Off-Policy Correction for a REINFORCE Recommender System." WSDM 2019. https://arxiv.org/abs/1812.02353
50. Uber Engineering. "Uber Eats search pipeline" (retrieval, hydration and ranking latency). Uber Blog, 2026. https://www.uber.com/us/en/blog/uber-eats-search-pipeline/
51. Jiaqi Zhai et al. (Meta). "Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations." ICML 2024. https://arxiv.org/abs/2402.17152
52. LinkedIn Engineering. "Understanding dwell time to improve LinkedIn feed ranking." 2020 (cited via [6]). https://engineering.linkedin.com/blog/2020/understanding-feed-dwell-time
53. Twitter Engineering. "Using Deep Learning at Scale in Twitter's Timelines." 2017 (cited via [6]). https://blog.twitter.com/engineering/en_us/topics/insights/2017/using-deep-learning-at-scale-in-twitters-timelines
54. Heng-Tze Cheng et al. (Google). "Wide & Deep Learning for Recommender Systems." DLRS 2016. https://arxiv.org/abs/1606.07792
55. Netflix Technology Blog. "Recommending for Long-Term Member Satisfaction at Netflix." 2024. https://netflixtechblog.com/recommending-for-long-term-member-satisfaction-at-netflix-ac15cada49ef
56. Eugene Yan et al. "applied-ml: Papers and tech blogs by companies sharing their work on data science and machine learning in production." GitHub. https://github.com/eugeneyan/applied-ml
