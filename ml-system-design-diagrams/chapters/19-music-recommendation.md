# 19. Design Music Recommendation and Playlist Generation (Spotify Discover Weekly and Home)

> How the prompt is usually worded: "Design Spotify's 'Discover Weekly' playlist" [1][2], or
> "Design the Spotify Home page: which shelves and which playlists or tracks each user sees".
> Variants: automatic playlist continuation ("add songs to this playlist") [22][23], radio or
> autoplay (next-track), "Daily Mix", podcast recommendations on Home [11][12], and
> conversational music recommendation (Amazon Music) [26].
>
> Sourcing note: this chapter was compiled and reviewed after the shared web-search budget
> had run out. Only GitHub hosts could be reached. engineering.atspotify.com, dl.acm.org,
> hackernoon.com and amazon.science did not resolve when re-checked in review.
> Sources marked (fetched) were read for this chapter through GitHub. Sources marked
> (ch-ref) were verified in earlier chapters of this atlas. Sources marked (known) are
> canonical papers or posts whose URLs were confirmed through curated GitHub lists [24][25]
> but whose text could not be re-read here. Numbers taken from them are flagged.

## Company tags

| Company | What is reported | Source |
|---|---|---|
| Spotify | The problem is Spotify's own product. "Design Spotify's 'Discover Weekly' playlist" appears in community MLSD question lists as a standard recommendation prompt | [1][2][3] |
| Meta | A company-tagged question bank lists "Design Spotify" (system design, medium) under Meta. This is the classic (non-ML) version; an open-source "Design Spotify" answer adds a `GET /api/recommendations/discover-weekly` endpoint, so the recommender is a natural deep-dive | [4][35] |
| Amazon (Amazon Music), Apple (Apple Music), YouTube Music, Pandora, Deezer, SoundCloud | These companies run the same kind of product. No direct question report could be verified for this chapter, so treat this row as an inference | assumption |
| General MLE loops | One community guide asserts that almost every MLSD round includes a prompt such as "Design Spotify's Discover Weekly" (opinion, no data) | [2] |

**Portals and sources that cover it**

| Portal / source | Treatment | Link |
|---|---|---|
| Community MLSD notes (sharayu134, mdrijwan123, Gaurav14cs17) | [1] lists "Design Spotify's 'Discover Weekly' playlist" as a sample prompt; [2] cites Discover Weekly as a recsys impact example; [3] lists "Spotify: Discover Weekly hybrid approach" as a case study | [1][2][3] |
| Chip Huyen (DMLS book resources) | Lists the Ciocca Discover Weekly article (CF + NLP + audio models) as reading. Her MLSD booklet [34] is generic and has no music content | [5][6] |
| ByteByteGo (Aminian & Xu) | No music chapter. Use the video-recommendation chapter (two-tower retrieval plus ranker) as the template | [28] |
| alirezadir AIMLInterviews | Video and game recommendation templates. No music file | [27] |
| Hello Interview | Generic recommendation-systems pattern (retrieval, ranking, re-ranking); no music write-up. See also E. Yan on discovery system design | [30][32] |
| Exponent / Aced, Educative | Generic MLSD frameworks. No music-specific write-up could be verified | [29][31] |
| Spotify engineering and research (the real system) | Discover Weekly hybrid (CF + NLP + audio), BaRT bandits on Home, CoSeRNN contextual user embeddings, Annoy then Voyager ANN, in-house experimentation platform | [6]-[20], [36] |

## Requirements & scale

| Dimension | Value | Basis |
|---|---|---|
| Functional | (a) Discover Weekly: 30 new-to-user tracks per user, refreshed every Monday. (b) Home: rank shelves (cards) and the items inside each shelf. (c) Explain the slot ("Because you listened to X") | [6][8] (known) |
| Novelty constraint | Discover Weekly must exclude tracks the user has already played. Home can mix familiar and new content | [6] (known) |
| Users | Hundreds of millions of MAU. Assume 600M+ users and 100M+ tracks, plus podcasts and audiobooks | typical assumption |
| Weekly batch | 600M users x 30 tracks = about 18B rows of slate per week. Spread over a 2-3 day batch window | assumption |
| Home QPS | About 50k QPS peak for Home loads, with 10-20 shelves per page | assumption |
| Latency | Home p99 under 200 ms server-side. Discover Weekly is offline (precomputed), with a single key-value read at request time | assumption |
| ANN scale | Spotify's Voyager is "queried hundreds of millions of times per day" | [19] (fetched) |
| Historical reach | Discover Weekly "reaches over 30 million users every week" (older figure quoted by a community guide) | [2] (fetched) |
| Non-functional | Freshness (new releases, new users), diversity and artist fairness, explainability, cost of weekly batch, privacy | [8][16] (known) |

## Diagram 1: End-to-end architecture (online serving path)

**Figure 19.1 - Hybrid serving: Discover Weekly is precomputed in batch; Home is ranked online.**

```
  Client (app open, Monday or any time)
        |
        v
 +-----------------------+   (A) Discover Weekly tab   +---------------------------+
 | API gateway /         |---------------------------->| Playlist KV store         |
 | Home orchestrator     |<----------------------------| user -> 30 track ids      |
 +-----------------------+   one KV read, < 10 ms      | (written by Monday batch) |
        |                                              +---------------------------+
        | (B) Home: user_id + ctx (hour, device, country)
        v
 +-----------------------+                             +---------------------------+
 | 1. Candidate gen      |<----- user/ctx features ----| Feature store             |
 |  (parallel sources)   |                             | taste emb, last N plays,  |
 |  - ANN on user emb    |                             | shelf CTR, artist affin.  |
 |    (Voyager / HNSW)   |                             +---------------------------+
 |  - item-item co-listen|                             +---------------------------+
 |  - editorial, new rel.|<----- ANN index files ------| Model + ANN index store   |
 |  - "because you X"    |                             | (weights, Voyager/Annoy   |
 +-----------------------+                             |  files, mmap-shared)      |
        | ~1-5k items per shelf                        +---------------------------+
        v                                                            |
 +-----------------------+                                           |
 | 2. Filter             |  licence/region, blocked artists,         |
 |                       |  explicit setting, dedupe track/album     |
 +-----------------------+                                           |
        v                                                            |
 +-----------------------+                                           |
 | 3. Multi-task ranker  |<-------------- ranker weights ------------+
 |  p(stream>=30s),      |
 |  p(save), p(add),     |
 |  p(skip)              |
 +-----------------------+
        v
 +-----------------------+
 | 4. Page policy        |  shelf order via contextual bandit (BaRT),
 |  (re-rank)            |  artist/genre diversity, explanation text
 +-----------------------+
        |  page JSON -> client
        v
 +-----------------------+   impressions, plays, skips, saves,
 | Event logging         |   + propensity of each shown item  ---> training (Fig 19.2)
 | (Pub/Sub / Kafka)     |
 +-----------------------+
```

## Diagram 2: Offline training & data pipeline

**Figure 19.2 - Weekly batch for Discover Weekly plus daily retraining for Home rankers.**

```
 +-------------+   +------------------+   +-------------------+   +------------------+
 | Client logs |-->| Sessionize &     |-->| Label join        |-->| Training tables  |
 | play, skip, |   | dedupe (Dataflow |   | stream>=30s = pos |   | (user, item, ctx,|
 | save, pl-add|   |  / Beam, BQ)     |   | skip<30s  = neg   |   |  label, propens.)|
 +-------------+   +------------------+   | impression no-play|   +------------------+
                                          | = weak neg        |            |
 +-------------+   +------------------+   +-------------------+            |
 | Catalog:    |-->| Content features |                                    |
 | audio, meta,|   | audio CNN embed, |---- item features ---------------->|
 | web text,   |   | NLP artist/track |                                    v
 | playlists   |   | vectors (word2vec|                          +-------------------+
 +-------------+   |  on playlists)   |                          | Train             |
                   +------------------+                          | - implicit MF/ALS |
                                                                 | - two-tower       |
 +-------------------+   +-------------------+   +------------+  | - MT ranker       |
 | Registry / canary |<--| Offline eval      |<--| Candidates |<-| - bandit (off-pol)|
 | model + ANN index |   | recall@k, NDCG,   |   | / scores   |  +-------------------+
 +-------------------+   | R-precision,      |   +------------+
        |                | novelty, coverage |
        | deploy         +-------------------+
        v
 +--------------------+   Monday batch: score all users  +---------------------+
 | Batch scorer       |--------------------------------->| Playlist KV store   |
 | (Discover Weekly)  |                                  | (user -> 30 tracks) |
 +--------------------+                                  +---------------------+
        |
        v           +---------------------+   A/B (exp. platform)   +----------------+
 Online services -->| Home ranker + ANN   |------------------------>| Metrics: DW    |
 (daily refresh)    | indexes             |                         | streams, saves,|
                    +---------------------+                         | retention      |
                              |                                     +----------------+
                              | new impressions / plays (feedback loop)      |
                              +------------------> back to Client logs <-----+
```

## Diagram 3: Model architecture

**Figure 19.3 - Hybrid retrieval (CF + content + contextual user tower) feeding a multi-task ranker.**

```
 USER / QUERY SIDE                                  ITEM (TRACK) SIDE
 +---------------------------+                      +------------------------------+
 | long-term taste: avg of   |                      | track_id emb (from MF / w2v) |
 | top-N played track embs   |                      | artist_id emb, genre, year   |
 | recent session sequence   |                      | audio CNN emb (cold start):  |
 |  [t1, t2, ... tn] (+skips)|                      |  mel-spectrogram 599x128     |
 | context: hour, weekday,   |                      |  -> 4x conv1d -> global pool |
 |  device, platform         |                      |  -> 2x dense 2048 -> 40-d    |
 +---------------------------+                      |  (regresses CF factors)      |
              |                                     | NLP emb (web text, pl names) |
              v                                     +------------------------------+
 +---------------------------+                                    |
 | Sequence encoder          |                                    v
 | (RNN like CoSeRNN, or     |                      +------------------------------+
 |  transformer)             |                      | Item tower MLP -> 128-d      |
 +---------------------------+                      +------------------------------+
              |                                                   |
              v                                                   v
 +---------------------------+   dot product / cosine   +----------------------------+
 | User-context emb (128-d)  |------------------------->| ANN index (Voyager / HNSW) |
 +---------------------------+                          +----------------------------+
                                                                   |
                              top ~1k candidates (+ other sources) |
                                                                   v
 +---------------------------------------------------------------------------------+
 | Multi-task ranker (GBDT baseline -> DNN with shared bottom / MMoE)              |
 |  inputs: user emb, item emb, user x artist affinity, novelty (played before?),  |
 |          popularity, release age, context, position of shelf                   |
 |  heads:  p(stream >= 30s)   p(save / like)   p(add to playlist)   p(skip)       |
 |  score = w1*p_stream + w2*p_save + w3*p_add - w4*p_skip  (+ novelty boost)      |
 +---------------------------------------------------------------------------------+
                                       |
                                       v
                  Slate builder: 30 tracks, max 2 per artist, sequenced for flow
```

## Diagram 4: Variant - the real Spotify systems

**Figure 19.4 - Discover Weekly as an ensemble of three model families, and Home ordered by a contextual bandit (BaRT).**

```
 DISCOVER WEEKLY (weekly batch) [5][6][7][15]
 +----------------------+  +----------------------+  +-------------------------+
 | Collaborative filter |  | NLP models           |  | Audio models            |
 | implicit MF over     |  | word2vec-like vectors|  | CNN on raw audio:       |
 | user x track plays   |  | from web text and    |  | predicts CF latent      |
 | and playlists        |  | playlist co-occurr.  |  | factors for new tracks  |
 +----------------------+  +----------------------+  +-------------------------+
            \                        |                          /
             +-----------------------+-------------------------+
                                     v
                 +-----------------------------------------+
                 | Candidate union -> remove played tracks |
                 | -> rank -> 30-track slate per user      |
                 +-----------------------------------------+

 HOME (online) [8][10][11]
 +-------------+    +-------------------+    +-----------------------------------+
 | Shelves &   |--->| BaRT: reward model|--->| epsilon-greedy over (card, shelf, |
 | cards pool  |    | p(stream | card,  |    | explanation). Logs propensities;  |
 | (playlists, |    |  explanation, ctx)|    | retrain off-policy (IPS)          |
 | podcasts)   |    +-------------------+    +-----------------------------------+
 +-------------+
```

## Data & labels

| Signal | Label use | Notes |
|---|---|---|
| Stream >= 30 s | Positive | A common "real listen" threshold. Treat as an assumption in interviews |
| Skip < 30 s | Negative (strong) | Position-dependent: skips late in a long session are weaker |
| Save / like / add to playlist | Strong positive, separate head | Sparse but high intent. Key Discover Weekly success signal |
| Impression with no play | Weak negative | Needs propensity logging to de-bias [8] |
| User playlists (co-occurrence) | Item-item similarity, CF training | The Million Playlist Dataset (RecSys Challenge 2018) was built from this signal [22] |
| Audio and web text | Content features for cold start | [7][15] |

## Features

| Group | Examples |
|---|---|
| User | Long-term taste embedding, top artists and genres, listening diversity, account age, country |
| Session / context | Last N tracks (with skips), time of day, weekday, device (car, phone, speaker) [9] |
| Item | Track, artist and album embeddings, audio embedding, release date, popularity, language |
| User x item | Artist affinity, played-before flag, genre match, co-listen score, days since last artist play |
| Shelf (Home) | Shelf type, historical CTR, explanation type, position [8] |

## Model choices (baseline -> advanced)

| Stage | Baseline | Advanced | Tradeoff |
|---|---|---|---|
| Retrieval | Item-item co-occurrence; implicit MF (ALS) + Annoy [18] | Two-tower with sequence user tower (CoSeRNN-like) + Voyager/HNSW [9][19][23][33] | MF is cheap and strong. The sequence tower adds context but costs more |
| Cold start (items) | Metadata and genre | Audio CNN regressing CF factors [7] | The audio model may recommend "sound-alike" tracks that are not good fits |
| Cold start (users) | Onboarding artist picks, popular in region | Bandit exploration | Exploration costs short-term engagement |
| Ranking | GBDT on hand features | Multi-task DNN (MMoE) | MTL needs weight tuning across objectives |
| Slate / page | Rule-based (diversity caps) | Contextual bandit (BaRT) [8], RL for diversity [16] | Bandits need propensity logging and careful off-policy evaluation |

## Loss functions

| Model | Loss |
|---|---|
| Implicit MF | Weighted squared loss with confidence c = 1 + alpha * play_count (ALS) |
| Two-tower | Sampled softmax with in-batch negatives and logQ correction [21] |
| Audio CNN | MSE to unit-normalised 40-d CF latent vectors; unit norm reduces popularity leakage, since factor norms correlate with popularity [7] (fetched) |
| Ranker heads | Binary cross-entropy per head; weighted sum |
| Bandit reward model | Logistic loss on stream given impression; IPS-weighted for off-policy [8] |

## Offline metrics

| Metric | Use |
|---|---|
| Recall@k / HitRate@k | Retrieval on held-out next-week plays |
| NDCG@k, MAP | Ranking quality |
| R-precision, NDCG, "recommended songs clicks" | Playlist continuation (RecSys Challenge 2018) [22] (fetched) |
| Novelty, diversity, catalog coverage, artist Gini | Discover Weekly quality beyond accuracy [16] |
| Counterfactual (IPS / SNIPS) reward | Evaluate Home bandit policy offline [8] |

## Online metrics / A-B

| Metric | Notes |
|---|---|
| Discover Weekly: share of users who stream, streams per user, saves per user, tracks added to library | Primary success metrics (assumption) |
| Home: shelf CTR, consumption minutes from Home, time to first stream | [10][11] (known) |
| Long-term: weekly active retention, return to Discover Weekly next Monday | Guardrail |
| Guardrails: skip rate, latency, crash rate, diversity and new-artist exposure | Spotify runs these on an in-house experimentation platform [20][36] |
| Latency guardrail as a distribution | Compare p90/p99 between arms, not only means (quantile tests at scale) [37] |

## Serving & latency budget (assumption, Home p99 about 200 ms)

| Step | Budget |
|---|---|
| Feature fetch (user + context) | 15 ms |
| Candidate generation (parallel ANN + co-listen + editorial) | 30 ms |
| Filtering and dedupe | 5 ms |
| Ranking (~1-2k items, batched) | 60 ms |
| Bandit and diversity re-rank, explanations | 20 ms |
| Network, serialisation, slack | 70 ms |
| Discover Weekly read | Single KV lookup, under 10 ms |

## Monitoring

| What | How |
|---|---|
| Data freshness | Lag of play-event ingestion; missing Monday batch alarms |
| Feature drift | PSI on user and item embeddings and on popularity distribution |
| Embedding/index health | ANN recall vs brute force on a sample; index build time [19] |
| Index distribution | Build ANN index once offline, ship static files, mmap them so many serving processes share one copy (Annoy design) [18] |
| Prediction drift | Mean p(stream) vs observed stream rate per shelf and country |
| Content health | Share of new artists, top-artist concentration, explicit-content leaks |

## Failure modes

| Failure | Mitigation |
|---|---|
| Popularity bias / filter bubble | Novelty boost, diversity caps, RL for diverse consumption [16] |
| Cold-start tracks never surface | Audio and NLP content embeddings [7][15]; exploration slots |
| Shared accounts or kids' listening pollute taste | Session clustering; context-aware user embeddings [9] |
| Feedback loop (Home only shows what it already shows) | Logged propensities, epsilon exploration [8] |
| Sound-alike but wrong (covers, intros, remixes) | Dedupe by work id; CF signal outweighs audio when data exist [7] |
| Weekly batch fails or partially completes | Keep last week's list; incremental re-scoring |

## Interviewer follow-ups

- Discover Weekly must contain only tracks the user has not heard. Where do you enforce this, and how do you keep a played-set for 600M users?
- Why batch-compute Discover Weekly instead of serving it online? What changes if it must refresh daily?
- How do you recommend a track released yesterday with zero plays (audio CNN, NLP, artist prior)? [7]
- How do you handle a new user with no history?
- Is a skip negative? How do you treat skips in a workout session vs a focus session?
- How do you order the 30 tracks for flow (tempo, key, energy), not just relevance?
- How do you trade off familiarity vs discovery? What metric captures "discovery"?
- How would you rank shelves on Home and also pick the item inside each shelf? (BaRT) [8]
- How do you evaluate a new Home policy offline when logs come from the old policy? (IPS)
- How do you mix music, podcasts and audiobooks in one ranking? [11][12]
- How do you keep artist fairness or avoid over-exposing top labels? [16]
- Family plans and shared devices: how do you detect and separate listeners?
- Which ANN library and index parameters, and how do you measure recall loss? [18][19]
- How do you ship a new ANN index to thousands of serving hosts without a memory spike? (static, mmap-shared index files) [18]
- The audio model regresses CF factors: why normalise the targets, and what happens to popular tracks if you do not? [7]
- How would you add an LLM ("AI DJ"-style commentary, or text prompts for playlists) on top of this stack?

## Portal comparison

| Source | Framing | Retrieval | Ranking | Distinctive point |
|---|---|---|---|---|
| Community MLSD notes [1][2][3] | "Design Discover Weekly" | CF + content | Generic ranker | Emphasise CF vs content and cold start |
| Chip Huyen DMLS resources [5] / Ciocca [6] | Case study | CF + NLP + audio ensemble | Not detailed | Three model families |
| ByteByteGo video rec (template) [28] | YouTube-style | Two-tower + ANN | Multi-task NN | Template to adapt; no music specifics |
| Hello Interview pattern [30] | Generic recsys | Multi-source retrieval | Ranker + re-rank | Stage-by-stage delivery |
| Spotify research [8][9] | Home + sessions | Contextual user embedding (CoSeRNN) | Bandit (BaRT) for cards | Explanations and exploration as first-class |
| RecSys Challenge 2018 [22][23] | Playlist continuation | MF, item-item, two-tower | Ensembles | R-precision and clicks metrics; public MPD data |

## References

1. sharayu134, "Notes: Machine Learning system design question types" (system..md), GitHub, 2025 (fetched). https://github.com/sharayu134/Notes/blob/main/Machine%20Lerning/system..md
2. mdrijwan123, "AI Engineering Guide: Recommendation Systems", GitHub, 2025 (fetched). https://github.com/mdrijwan123/AI_enginnering_guide/blob/main/Phase_2_Advanced_Systems_Jul_Sep_year_01/Month_06_September/Recommendation_Systems.md
3. Gaurav14cs17, "ml_system_design: 01 Recommendation Systems", GitHub, 2025 (fetched). https://github.com/Gaurav14cs17/ml_system_design/blob/main/01_recommendation_systems/README.md
4. kevin-2023-code, "Tech-Interview-Questions: Meta" (company question bank listing "Design Spotify"), GitHub, 2025 (fetched). https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/meta.md
5. C. Huyen, "Designing Machine Learning Systems: resources.md", GitHub, 2022 (fetched). https://github.com/chiphuyen/dmls-book/blob/main/resources.md
6. S. Ciocca, "Spotify's Discover Weekly: How machine learning finds your new music", HackerNoon, 2017 (known via [5]). https://hackernoon.com/spotifys-discover-weekly-how-machine-learning-finds-your-new-music-19a41ab76efe
7. S. Dieleman, "Recommending Music on Spotify with Deep Learning", benanne.github.io, 2014 (fetched via GitHub source). https://benanne.github.io/2014/08/05/spotify-cnns.html (source: https://github.com/benanne/benanne.github.io/blob/master/_posts/2014-08-05-spotify-cnns.md)
8. J. McInerney et al., "Explore, Exploit, and Explain: Personalizing Explainable Recommendations with Bandits", RecSys, 2018 (known). https://dl.acm.org/doi/10.1145/3240323.3240354
9. C. Hansen et al., "Contextual and Sequential User Embeddings for Large-Scale Music Recommendation", RecSys, 2020 (known). https://dl.acm.org/doi/10.1145/3383313.3412248
10. Spotify Engineering, "For Your Ears Only: Personalizing Spotify Home with Machine Learning", engineering.atspotify.com, 2020 (known). https://engineering.atspotify.com/2020/01/16/for-your-ears-only-personalizing-spotify-home-with-machine-learning/
11. Spotify Engineering, "The Rise (and Lessons Learned) of ML Models to Personalize Content on Home (Part I)", engineering.atspotify.com, 2021 (known). https://engineering.atspotify.com/2021/11/the-rise-and-lessons-learned-of-ml-models-to-personalize-content-on-home-part-i/
12. Spotify Engineering, "The Rise (and Lessons Learned) of ML Models to Personalize Content on Home (Part II)", engineering.atspotify.com, 2021 (known). https://engineering.atspotify.com/2021/11/the-rise-and-lessons-learned-of-ml-models-to-personalize-content-on-home-part-ii/
13. Spotify Engineering, "Reach for the Top: How Spotify Built Shortcuts in Just Six Months", engineering.atspotify.com, 2020 (known). https://engineering.atspotify.com/2020/04/15/reach-for-the-top-how-spotify-built-shortcuts-in-just-six-months/
14. B. Carterette, "Music Recommendation at Spotify" (slides), ACM SIGIR AFIRM, 2019 (known). http://sigir.org/afirm2019/slides/16.%20Friday%20-%20Music%20Recommendation%20at%20Spotify%20-%20Ben%20Carterette.pdf
15. E. Bernhardsson, "How Music Recommendation Works - And Doesn't Work", notes.variogram.com, 2012 (known). https://notes.variogram.com/2012/12/11/how-music-recommendation-works-and-doesnt-work/
16. Spotify Research, "Shifting Consumption towards Diverse Content via Reinforcement Learning", WSDM, 2021 (known). https://research.atspotify.com/shifting-consumption-towards-diverse-content-via-reinforcement-learning/ (paper: https://dl.acm.org/doi/10.1145/3437963.3441775)
17. Arize, "Embeddings at Spotify's Scale - How Hard Could It Be?", arize.com, 2023 (known). https://arize.com/resource/embeddings-at-scale-spotify-recsys/
18. Spotify, "Annoy: Approximate Nearest Neighbors Oh Yeah", GitHub, 2013- (fetched). https://github.com/spotify/annoy
19. Spotify, "Voyager: approximate nearest-neighbor search library (HNSW)", GitHub, 2023 (fetched). https://github.com/spotify/voyager
20. Spotify Engineering, "Spotify's New Experimentation Platform (Part 1)", engineering.atspotify.com, 2020 (known). https://engineering.atspotify.com/2020/10/29/spotifys-new-experimentation-platform-part-1/
21. X. Yi, J. Yang, L. Hong et al., "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations", RecSys / Google Research, 2019 (ch-ref). https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/
22. S. Antenucci, S. Boglio, E. Chioso, T. Scarlatti et al. (Creamy Fireflies, Politecnico di Milano), "Spotify RecSys Challenge 2018 solution" (MPD, automatic playlist continuation, metrics), GitHub, 2018 (fetched). https://github.com/tmscarla/spotify-recsys-challenge
23. J. Wortz, "Building Two Tower Models on Google Cloud from Spotify Data" (MPD two-tower + ANN), GitHub, 2022 (fetched). https://github.com/jswortz/spotify_mpd_two_tower
24. E. Yan, "applied-ml: Papers & tech blogs by companies sharing their work on data science & ML in production", GitHub, 2024 (fetched). https://github.com/eugeneyan/applied-ml
25. Engineer1999, "A Curated List of ML System Design Case Studies", GitHub, 2024 (fetched). https://github.com/Engineer1999/A-Curated-List-of-ML-System-Design-Case-Studies
26. Amazon Science, article on Amazon Music recommendation and conversational ML, amazon.science, 2022 (not re-verified; host unreachable in review, title unconfirmed). https://www.amazon.science/latest-news/how-amazon-music-uses-recommendation-system-machine-learning
27. alirezadir, "AIMLInterviews: MLSD video recommendation", GitHub, 2025 (ch-ref). https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/mlsd-video-recom.md
28. A. Aminian & A. Xu, "Machine Learning System Design Interview: Video Recommendation System", ByteByteGo, 2023 (ch-ref). https://bytebytego.com/courses/machine-learning-system-design-interview/video-recommendation-system
29. Educative, "The 6-step ML SD framework" (Grokking the ML System Design Interview), educative.io, 2025 (ch-ref). https://www.educative.io/courses/grokking-the-machine-learning-system-design-interview/the-6-step-ml-sd-framework
30. Hello Interview, "ML System Design Patterns: Recommendation Systems", hellointerview.com, 2025 (ch-ref). https://www.hellointerview.com/learn/ml-system-design/patterns/recommendation-systems
31. Aced (formerly Exponent), "Machine Learning System Design Interview (2026 Guide)", aced.io, 2026 (ch-ref). https://www.aced.io/blog/machine-learning-system-design-interview-guide
32. E. Yan, "System Design for Recommendations and Search", eugeneyan.com, 2021 (ch-ref). https://eugeneyan.com/writing/system-design-for-discovery/
33. P. Covington, J. Adams, E. Sargin, "Deep Neural Networks for YouTube Recommendations", RecSys / Google Research, 2016 (ch-ref). https://research.google/pubs/pub45530/
34. C. Huyen, "machine-learning-systems-design" booklet, GitHub, 2019 (fetched; generic, no Spotify content). https://github.com/chiphuyen/machine-learning-systems-design
35. evgenyvinnik, "llm-driven-system-design: Design Spotify" (includes a Discover Weekly endpoint), GitHub, 2025 (fetched). https://github.com/evgenyvinnik/llm-driven-system-design/blob/main/spotify/README.md
36. Spotify Engineering, "Spotify's New Experimentation Platform (Part 2)", engineering.atspotify.com, 2020 (URL confirmed via [24]). https://engineering.atspotify.com/2020/11/02/spotifys-new-experimentation-platform-part-2/
37. Spotify Engineering, "Comparing quantiles at scale in online A/B-testing", engineering.atspotify.com, 2022 (URL confirmed via [24]). https://engineering.atspotify.com/2022/03/comparing-quantiles-at-scale-in-online-a-b-testing
