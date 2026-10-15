# 38. Design Exploration for Cold-Start Items and Users (Bandits) in a Recommender

> How the prompt is usually worded: "Netflix shows one artwork per title. Pick the best image
> for each member" [1], or "New creators and new items never get impressions. Fix the cold
> start" [16][17]. It also comes up as a follow-up to any recommendation question: "How do you
> balance exploration and exploitation?" [16][17][18].
> Variants: Spotify Home shelves (BaRT) [3], DoorDash homepage explore/exploit [5], best
> merchant image [6], Twitter deep Bayesian bandits [7], Expedia cascade ranking [8].
>
> Sourcing note: written and reviewed after the shared web-search budget ran out; non-GitHub
> hosts returned 403 through the proxy. (fetched) = read from GitHub for this chapter.
> (list) = URL, title and company confirmed in a fetched curated list [19][22] but the page
> itself not re-read. (known) = canonical paper cited from memory. Details from (list) or
> (known) sources are flagged as such.

## Company tags

| Company | What is reported | Source |
|---|---|---|
| Netflix | The original product: per-member artwork chosen with contextual bandits. Netflix also published slate-bandit research | [1][2] (list) |
| Spotify | BaRT bandits for Home shelves and explanations. RL used to shift listening toward diverse content | [3][4] (list) |
| DoorDash | Homepage explore/exploit for new or rarely seen stores (UCB-style). Exploration to pick each merchant's image | [5][6] (list) |
| Expedia, Stitch Fix | Cascade bandits for ranking; a bandit experimentation platform | [8][9] (list) |
| Twitter | Meta-learning for cold-start items. Deep Bayesian bandits (dropout / bootstrap uncertainty) for exploration in ad recommendations | [10][7] (list) |
| Yahoo! | LinUCB news-article bandit and replay evaluation, the classic references | [11][12] (known) |
| Google/YouTube | Top-K off-policy REINFORCE recommender. Zero-shot transfer from recsys to cold-start search | [27] (known), [31] (list) |
| ByteDance/TikTok | For You feed deliberately mixes in videos outside a user's shown interests | [32] (list) |
| Coveo | Content-based inference of embeddings for new and rare products | [33] (list) |

Meta, LinkedIn and others ask cold start and explore/exploit as generic recsys follow-ups
(alirezadir and ByteByteGo templates [16][17][21]), but no company-specific source was
verified, so they are not tagged.

**Portals and sources that cover it**

| Portal / source | Treatment | Link |
|---|---|---|
| alirezadir / Machine-Learning-Interviews | The video and game recommendation templates each have a "Cold start problem" and an "Exploration exploitation trade-off (multi-armed bandit)" section. The MLSD template lists bandits under online testing | [16][17][18] (fetched) |
| ByteByteGo (Aminian & Xu) | Ch. 6 (video recs) cites the multi-armed bandit as its explore/exploit reference. Ch. 10 (news feed) cites cold start | [21] (fetched) |
| Chip Huyen, Designing ML Systems | Ch. 9 "Continual Learning and Test in Production" has a contextual-bandits section; its resource list cites deep Bayesian bandits [7] | [22] (fetched: resources list) |
| E. Yan applied-ml list | Curates Netflix, Spotify, DoorDash, Expedia, Stitch Fix and Twitter cold-start and bandit posts | [19] (fetched) |
| Exponent, Hello Interview, Educative, IGotAnOffer | Generic recsys or MLSD frameworks with cold start as a follow-up. No bandit-specific write-up could be verified | [23][24][25][26] |

## Requirements & scale

| Dimension | Value | Basis |
|---|---|---|
| Functional | Pick an arm (artwork, new item, new creator) for each user and slot. Give new items a guaranteed but bounded number of impressions. Learn quickly from feedback | [1][5] |
| Arms per decision | Artwork: about 5-20 images per title. Item exploration: 10^2-10^4 cold candidates | assumption |
| Catalog | 10^4 titles (Netflix-like) up to 10^8 items or creators (UGC) | assumption |
| Users | 10^8 MAU. Interactions arrive in a stream | assumption |
| QPS | 10^5 page requests/s at peak, each needing 10-100 bandit decisions | assumption |
| Latency | Policy step p99 under 10-20 ms inside a 150-200 ms page budget | assumption |
| Freshness | Posteriors and counts updated in minutes, and the model retrained daily | assumption |
| Exploration budget | 1-10% of impressions, or a bounded regret target. Guardrails on watch time and retention | assumption |
| Non-functional | Log the propensity for every decision. Assign arms per session without flicker. Stay fair to new creators | [1][12] |

## Diagram 1: End-to-end architecture (online serving path)

**Figure 38.1 - Serving path with a cold-start candidate source and a bandit policy layer**

```
 +---------+  request(user,ctx)  +----------------+
 | Client  |-------------------->| Rec API / page |
 +---------+<--------------------| assembler      |
      ^        slate + artwork   +-------+--------+
      |                                  | user ctx
      |              +-------------------+-------------------+
      |              v                   v                   v
      |      +---------------+   +---------------+   +----------------+
      |      | Warm retrieval|   | Cold-item pool|   | Popular /      |
      |      | two-tower ANN |   | new < 7 days, |   | trending prior |
      |      | (exploit)     |   | content embed |   | (new users)    |
      |      +-------+-------+   +-------+-------+   +--------+-------+
      |              +-------------------+--------------------+
      |                                  v  ~1-2k candidates
      |                         +------------------+
      |                         | Filters: seen,   |
      |                         | policy, geo, age |
      |                         +--------+---------+
      |                                  v
      |  +--------------+  feats  +------------------+  model  +------------+
      |  | Feature store|-------->| Ranker (MTL p()) |<--------| Model store|
      |  | online+stats |         | + uncertainty    |         +------------+
      |  +------^-------+         +--------+---------+
      |         |                          v  mu, sigma per candidate
      |         |                 +------------------+  posteriors  +----------+
      |         |                 | Bandit policy    |<-------------| Param    |
      |         |                 | TS / UCB / e-grd |              | store    |
      |         |                 | explore slots,   |              | (a,b) or |
      |         |                 | impression caps  |              | (A,b)/arm|
      |         |                 +--------+---------+              +----^-----+
      |         |                          v  arm + propensity p(a|x)    |
      |         |                 +------------------+                   |
      +---------|-----------------| Re-rank: divers. |                   |
        slate   |                 | dedupe, artwork  |                   |
                |                 +--------+---------+                   |
                |                          v  served slate + p(a|x)      |
                |                 +------------------+  stream  +--------+-----+
                +-----------------| Impression+reward|--------->| Near-RT      |
                  item counters   | logger (Kafka)   |          | updater      |
                                  +------------------+          +--------------+
```

## Diagram 2: Offline training & data pipeline

**Figure 38.2 - Logged bandit feedback to off-policy evaluation and deployment**

```
 +-------------------+   (x, a, p(a|x), r, ts)    +----------------------+
 | Decision logs     |--------------------------->| Join impressions +   |
 | context, arm,     |                            | delayed rewards      |
 | propensity        |                            | (play > N min, save) |
 +-------------------+                            +----------+-----------+
                                                             v
 +-------------------+   content embeds           +----------------------+
 | New item / creator|--------------------------->| Feature builder      |
 | metadata, images, |   (CLIP, text, audio)      | user, item, arm,     |
 | creator profile   |                            | context crosses      |
 +-------------------+                            +----------+-----------+
                                                             v
               +---------------------+         +--------------------------+
               | Train reward model  |<--------| Train/val split by time  |
               | IPS/DR-weighted     |         | keep only explore traffic|
               | + cold-start tower  |         | for unbiased eval        |
               +----------+----------+         +--------------------------+
                          v
               +---------------------+  pass  +-------------+ canary +-----------+
               | Off-policy eval     |------->| Registry    |------->| A/B or    |
               | replay, IPS, SNIPS, |        | model+policy|        | bandit-vs-|
               | doubly robust       |        | config      |        | bandit    |
               +----------+----------+        +-------------+        +-----+-----+
                          | fail                                           |
                          v                                                |
                 back to features                                          |
                                                                           |
   +-----------------------------------------------------------------------+
   |  new policy serves --> new logs (with propensities) --> loop
   v
 +-----------------------------------------------+
 | Near-RT updater: Beta(a,b) or Bayesian linear |
 | (A, b) per arm, every few minutes             |
 +-----------------------------------------------+
```

## Diagram 3: Model architecture

**Figure 38.3 - Neural contextual bandit with a content tower for cold arms**

```
  USER / CONTEXT                       ARM (item, artwork, creator)
 +--------------------+              +-------------------------------+
 | user id emb (or    |              | id emb  (zero / dropped out   |
 | onboarding prefs   |              |          when item is new)    |
 | for new users)     |              | content: image CLIP, text,    |
 | history pooled     |              | genre, creator stats          |
 | device, time, row  |              | cold flag, age, #impressions  |
 +---------+----------+              +---------------+---------------+
           v                                         v
   +---------------+                       +------------------+
   | User tower    |                       | Item tower       |
   | MLP -> u (64) |                       | MLP -> v (64)    |
   +-------+-------+                       +---------+--------+
           +-------------------+---------------------+
                               v  [u, v, u*v, ctx]
                     +-----------------------+
                     | Shared MLP trunk      |
                     +-----+-----------+-----+
                           v           v
                +--------------+   +------------------------+
                | Reward head  |   | Uncertainty head       |
                | p(play),     |   | last-layer Bayesian    |
                | E[watch]     |   | linear, MC dropout,    |
                +------+-------+   | or ensemble -> sigma   |
                       |           +-----------+------------+
                       +-----------+-----------+
                                   v
                 +-----------------------------------+
                 | Acquisition: TS sample  mu+s*eps  |
                 |  or UCB  mu + alpha*sigma         |
                 |  -> argmax arm, log p(a|x)        |
                 +-----------------------------------+
```

## Diagram 4: Variant - the real systems (Netflix artwork, DoorDash, Spotify BaRT)

**Figure 38.4 - Published production patterns (cold-start tower: [10][13])**

```
 NETFLIX ARTWORK [1]                       DOORDASH HOMEPAGE E&E [5]
 +-------------------------+               +----------------------------+
 | title -> pool of images |               | store candidates           |
 | member context, history |               | score = exploit pCTR       |
 | contextual bandit picks |               |       + c*sqrt(ln N / n_i) |
 | image; randomized share |               | (UCB bonus for rarely seen |
 | of traffic -> logs      |               |  stores)                   |
 | replay metric offline   |               +----------------------------+
 +-------------------------+
 SPOTIFY BaRT [3]                          COLD-START TOWER (Twitter, DropoutNet)
 +-------------------------+               +----------------------------+
 | shelf x card x reason   |               | learn item embedding from  |
 | factorization-machine   |               | content so new items score |
 | reward, epsilon-greedy  |               | on day 0; bandit refines   |
 | explore; counterfactual |               | it once data accrues       |
 | training with propens.  |               +----------------------------+
 +-------------------------+
```

## Data & labels

| Item | Choice |
|---|---|
| Unit of logging | (context x, arm a, propensity p(a given x), reward r, slot, timestamp) [1][12] |
| Reward | Artwork: play within the session or quality plays (exclude quick aborts) [1]. Items: click, then long dwell, save or follow |
| Delayed reward | Join within an attribution window (for example 24 h). Count late rewards as positives when joined |
| Negatives | Impressed but not engaged. Skipped arms count only if actually viewed (viewport) |
| Explore data | Keep a uniformly randomized slice for unbiased evaluation [1][11][12] |
| Bias | Position and presentation bias. Trust only data that carries logged propensities |

## Features

| Group | Examples | Cold-start relevance |
|---|---|---|
| User | Embedding, genre affinity, recency, device | New user: onboarding picks, geo/locale, device, popular prior |
| Item / arm content | Image embedding, text, genre, cast, creator follower count | Only signal on day 0 [10][13] |
| Item stats | Impressions, CTR posterior (alpha, beta), age | Low counts mean high uncertainty and a bigger bonus |
| Context | Time, row/shelf, page position, country | Artwork effect varies by context [1] |
| Cross | User genre x artwork theme, user x creator category | Personalized explore |

## Model choices (baseline -> advanced)

| Level | Model | Pros | Cons |
|---|---|---|---|
| 0 | Popularity + fixed new-item boost | Simple | Not personalized; can be gamed |
| 1 | Epsilon-greedy over the ranker | Easy, gives propensities | Wastes traffic uniformly [3] |
| 2 | Beta-Bernoulli Thompson sampling per arm [14] | Efficient for few arms | No context; slow for long tail |
| 3 | LinUCB / linear TS, contextual [11] | Personalized, closed-form update | Linear; feature engineering |
| 4 | Neural bandit (NN + Bayesian last layer, ensembles) [15] | Rich features, shared learning | Calibration of sigma, cost |
| 4b | Deep Bayesian bandit at production scale: MC dropout or bootstrapped heads on the ranker [7] | Reuses the existing ranker; cheap sigma | Uncertainty is approximate; needs tuning |
| 5 | Slate / cascade bandits, top-K off-policy RL [2][8][27] | Handles whole page | Complex, high variance |
| Cold-start add-on | Content tower, DropoutNet, meta-learning, content-to-embedding inference [10][13][33] | Day-0 scores | Needs good content features |
| Cold-start transfer | Zero-shot transfer of recsys embeddings to a new surface with no logs [31] | Launch a surface without data | Domain mismatch |

## Loss functions

| Component | Loss |
|---|---|
| Reward model | Binary cross-entropy on r, weighted by 1/p(a given x) (IPS), clipped |
| Watch time | MSE or weighted logistic on log-duration |
| Cold tower | BCE plus an alignment loss: content embed close to learned ID embed (DropoutNet-style) [13] |
| Policy (RL variant) | REINFORCE with off-policy correction and a top-K correction [27] |
| Objective | Minimize cumulative regret, the sum over t of r(best arm) - r(chosen arm) |

## Offline metrics

| Metric | Use |
|---|---|
| Replay / rejection-sampling estimate | Unbiased on uniformly random logs. Used for Netflix artwork and Yahoo! news [1][12] |
| IPS, SNIPS, doubly robust | Off-policy value with logged propensities [28] |
| AUC / log loss of reward model | Sanity check, sliced by cold vs warm |
| Calibration of sigma | Coverage of credible intervals |
| Simulated regret | Bandit simulator on historical data before launch |

## Online metrics / A-B

| Metric | Goal |
|---|---|
| Take rate / play rate per impression | Primary for artwork [1] |
| New-item coverage | Percent of new items reaching N impressions within T hours |
| Time to first signal | Hours until a new item's posterior is confident |
| Creator retention | Percent of new creators posting again (supply side) |
| Diversity / filter bubble | Share of impressions outside a user's top interests; catalog coverage [4][32] |
| Guardrails | Session watch time, retention, complaints, latency |
| Test design | Bandit vs control A/B with user-level randomization. Keep an always-on random-explore bucket [9] |

## Serving & latency budget (assumption, page p99 about 200 ms)

| Stage | Budget |
|---|---|
| Retrieval (warm ANN + cold pool + popular) | 30 ms |
| Feature fetch | 20 ms |
| Ranker with uncertainty head | 50 ms |
| Bandit sampling (TS draw per arm, caps) | 5-10 ms |
| Re-rank and artwork selection | 10 ms |
| Logging (async) | 0 ms on the critical path |
| Param store reads | Cached in-process and refreshed every 1-5 min |

## Monitoring

| Signal | Alert on |
|---|---|
| Exploration rate | Drift from the target band (for example 5% plus or minus 1%) |
| Propensity logging | Missing or zero propensities (breaks OPE) |
| Posterior health | Arms stuck at the prior; variance collapse |
| Cold-item funnel | Drop in new items reaching the impression threshold |
| Reward delay | Rising join latency; reward-pipeline lag |
| Business guardrails | Watch-time regression in explore buckets |

## Failure modes

| Failure | Mitigation |
|---|---|
| Over-exploration hurts users | Cap the explore share; explore only in low-cost slots; use TS, not uniform |
| Clickbait artwork wins on clicks | Reward quality plays, not clicks [1] |
| Feedback loop / rich-get-richer | Guaranteed impression budget for new items; diversity re-rank [4] |
| Non-stationarity (trends, novelty) | Discounted posteriors or sliding windows |
| Artwork flicker across sessions | Sticky per-member assignment for a period |
| Spam creators farming the boost | Quality and integrity gates before cold-pool entry |
| Delayed rewards bias the posterior | Pending-reward handling; batch updates after the window |
| Biased offline evaluation | Log propensities; keep a random bucket; use doubly robust estimators [28] |

## Interviewer follow-ups

- Why use a bandit instead of an A/B test for choosing artwork? What regret does A/B pay? [1][22]
- Thompson sampling vs UCB vs epsilon-greedy: which do you pick and why? [11][14]
- How do you evaluate a new policy offline without deploying it (replay, IPS, DR)? [12][28]
- What do you log so that off-policy evaluation is possible later?
- How do you get a sigma (uncertainty) out of a deep ranker? [15]
- A new creator uploads a video. Walk through its first 24 hours of impressions.
- A new user has no history. What do you show first, and how fast do you personalize?
- How do you pick the explore budget, and who pays for it (which users or slots)?
- Rewards arrive hours later. How do you update the bandit?
- How do you stop clickbait thumbnails from winning? [1]
- How do you explore over a slate or whole page rather than one slot? [2][8]
- How do you handle non-stationary arm rewards (seasonality, trending)?
- How would you balance fairness to new creators against user engagement? [4]
- When would you move from bandits to full RL (long-term value)? [27]
- Your ranker is a deep net. How do you add exploration without a separate bandit model (dropout or bootstrap heads)? [7]
- You launch a new surface or market with no logs. How do you bootstrap it? [31]
- How do you keep users out of a filter bubble without hurting engagement? [4][32]

## Portal comparison

| Source | Framing | Cold-start answer | Exploration answer | Gap |
|---|---|---|---|---|
| alirezadir video/game recs [16][17] | Section inside the recsys template | Two-tower with profile features; show new items to random users, then fine-tune | Multi-armed bandit, defined briefly | No policy, logging or OPE detail |
| ByteByteGo ch. 6 / ch. 10 [21] | Follow-up in video and feed recs | Content features and popular prior | MAB reference only | No serving design for bandits |
| Chip Huyen DMLS ch. 9 [22] | Testing in production | n/a | Bandits vs A/B; contextual bandits as exploration | Not a full recsys design |
| Netflix artwork blog [1] | Real production system | Per-title image pool | Contextual bandit, randomized explore, replay metric | Not an interview script |
| DoorDash / Spotify blogs [3][5] | Real production systems | UCB bonus for rarely seen stores | Epsilon-greedy (BaRT), UCB (DoorDash) | Company-specific constraints |

## References

1. Netflix Technology Blog (A. Chandrashekar, F. Amat, J. Basilico, T. Jebara), "Artwork Personalization at Netflix", netflixtechblog.com, 2017. https://netflixtechblog.com/artwork-personalization-c589f074ad76 (list)
2. Netflix, "Marginal Posterior Sampling for Slate Bandits", IJCAI, 2019. https://www.ijcai.org/proceedings/2019/308 (list)
3. J. McInerney et al. (Spotify), "Explore, Exploit, and Explain: Personalizing Explainable Recommendations with Bandits", RecSys, 2018. https://dl.acm.org/doi/10.1145/3240323.3240354 (list)
4. Spotify Research, "Shifting Consumption towards Diverse Content via Reinforcement Learning", research.atspotify.com, 2022. https://research.atspotify.com/shifting-consumption-towards-diverse-content-via-reinforcement-learning/ (list)
5. DoorDash Engineering, "Homepage Recommendation with Exploitation and Exploration", doordash.engineering, 2022. https://doordash.engineering/2022/10/05/homepage-recommendation-with-exploitation-and-exploration/ (list)
6. DoorDash Engineering, "Selecting the Best Image for Each Merchant Using Exploration and Machine Learning", doordash.engineering, 2023. https://doordash.engineering/2023/01/04/selecting-the-best-image-for-each-merchant-using-exploration-and-machine-learning/ (list)
7. D. Guo, S. I. Ktena, F. Huszar, P. K. Myana, W. Shi, A. Tejani (Twitter), "Deep Bayesian Bandits: Exploring in Online Personalized Recommendations", RecSys, 2020. https://arxiv.org/abs/2008.00727 (list: [22] resources)
8. Expedia Group Tech, "How to Optimise Rankings with Cascade Bandits", medium.com, 2022. https://medium.com/expedia-group-tech/how-to-optimise-rankings-with-cascade-bandits-5d92dfa0f16b (list)
9. Stitch Fix, "Multi-Armed Bandits and the Stitch Fix Experimentation Platform", multithreaded.stitchfix.com, 2020. https://multithreaded.stitchfix.com/blog/2020/08/05/bandits/ (list)
10. M. Vartak et al. (Twitter), "A Meta-Learning Perspective on Cold-Start Recommendations for Items", NeurIPS, 2017. https://papers.nips.cc/paper/7266-a-meta-learning-perspective-on-cold-start-recommendations-for-items (list)
11. L. Li, W. Chu, J. Langford, R. Schapire, "A Contextual-Bandit Approach to Personalized News Article Recommendation", WWW, 2010. https://arxiv.org/abs/1003.0146 (known)
12. L. Li, W. Chu, J. Langford, X. Wang, "Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms", WSDM, 2011. https://arxiv.org/abs/1003.5956 (known)
13. M. Volkovs, G. Yu, T. Poutanen, "DropoutNet: Addressing Cold Start in Recommender Systems", NeurIPS, 2017. https://papers.nips.cc/paper/7081-dropoutnet-addressing-cold-start-in-recommender-systems (known)
14. O. Chapelle, L. Li, "An Empirical Evaluation of Thompson Sampling", NeurIPS, 2011. https://papers.nips.cc/paper/4321-an-empirical-evaluation-of-thompson-sampling (known)
15. C. Riquelme, G. Tucker, J. Snoek, "Deep Bayesian Bandits Showdown", ICLR, 2018. https://arxiv.org/abs/1802.09127 (known)
16. alirezadir, "Video Recommendation System" (mlsd-video-recom.md), GitHub, 2024. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/mlsd-video-recom.md (fetched)
17. alirezadir, "Game Recommendation System" (mlsd-game-recom.md), GitHub, 2024. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/mlsd-game-recom.md (fetched)
18. alirezadir, "ML System Design" template (ml-system-design.md), GitHub, 2025. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/ml-system-design.md (fetched)
19. E. Yan, "applied-ml" (curated papers and tech blogs), GitHub, 2024. https://github.com/eugeneyan/applied-ml (fetched)
20. khangich, "machine-learning-interview", GitHub, 2021. https://github.com/khangich/machine-learning-interview (fetched)
21. ByteByteGo, "ml-bytebytego: Reference materials for ML System Design Interview" (ch. 6 ref 9, ch. 10 ref 16), GitHub, 2023. https://github.com/ByteByteGoHq/ml-bytebytego (fetched)
22. C. Huyen, "Designing Machine Learning Systems" (ch. 9), O'Reilly, 2022. https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/ ; chapter resources: https://github.com/chiphuyen/dmls-book/blob/main/resources.md (fetched)
23. Exponent, "Introduction to the ML System Design Interview" (course), tryexponent.com, 2025. https://www.tryexponent.com/courses/ml-system-design/mlsd-intro (ch-ref)
24. Hello Interview, "Recommendation Systems (ML System Design pattern)", hellointerview.com, 2025. https://www.hellointerview.com/learn/ml-system-design/patterns/recommendation-systems (ch-ref)
25. Educative, "Grokking the Machine Learning Interview", educative.io, 2024. https://www.educative.io/courses/grokking-the-machine-learning-interview (ch-ref)
26. IGotAnOffer, "Machine Learning (ML) System Design Interview", igotanoffer.com, 2025. https://igotanoffer.com/en/advice/machine-learning-system-design-interview (ch-ref)
27. M. Chen et al. (Google/YouTube), "Top-K Off-Policy Correction for a REINFORCE Recommender System", WSDM, 2019. https://arxiv.org/abs/1812.02353 (known)
28. Y. Saito et al., "Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation", NeurIPS Datasets, 2021. https://arxiv.org/abs/2008.07146 (known)
29. Wikipedia, "Multi-armed bandit" (cited by ByteByteGo ch. 6). https://en.wikipedia.org/wiki/Multi-armed_bandit
30. Wikipedia, "Cold start (recommender systems)" (cited by ByteByteGo ch. 10). https://en.wikipedia.org/wiki/Cold_start_(recommender_systems)
31. Google, "Zero-Shot Heterogeneous Transfer Learning from Recommender Systems to Cold-Start Search Retrieval", CIKM, 2020. https://arxiv.org/abs/2008.02930 (list)
32. TikTok Newsroom, "How TikTok recommends videos #ForYou", newsroom.tiktok.com, 2020. https://newsroom.tiktok.com/en-us/how-tiktok-recommends-videos-for-you (list)
33. Coveo, "The Embeddings That Came in From the Cold: Improving Vectors for New and Rare Products with Content-Based Inference", RecSys, 2020. https://dl.acm.org/doi/10.1145/3383313.3411477 (list)
