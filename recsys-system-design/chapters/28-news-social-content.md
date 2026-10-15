# 28. News, Social and Content Platforms: From Click Co-visitation to Open Feed Protocols

## Overview

This chapter covers recommenders for platforms where content is perishable, abundant and often user-generated: news portals (Google News, Yahoo! JAPAN, Microsoft News, The New York Times), discussion and social platforms (Reddit, Quora, Medium, Snapchat), decentralized social networks (Bluesky, Mastodon), and community or streaming services (Substack, Discord, Twitch, Hulu, Disney+, Roku).

These systems share three structural problems that set them apart from the catalog recommenders in Chapters 18 (Netflix) and 24 (Amazon):

1. **Extreme item churn.** A news story or a social post is mostly worthless after hours or days. Item-ID collaborative filtering never collects enough interactions before the item dies. Content encoders such as text, entity and image models therefore do most of the work, and freshness is a primary ranking signal rather than a tie-breaker.
2. **A popularity/recency prior is a strong baseline.** Reddit's "hot" formula and Mastodon's trending scores are closed-form functions of votes and age. They still serve as candidate sources, cold-start fallbacks and sanity baselines, even in platforms with heavy ML.
3. **Governance and control matter.** Mastodon deliberately keeps a reverse-chronological home timeline. Bluesky moves ranking out of the platform into third-party "feed generators". Both are architecture decisions about who owns the ranking function, and they change the system design as much as any model does.

A note on sourcing for this edition. Network access during research was limited to GitHub-hosted material. Every Bluesky, Mastodon, Reddit-ranking and MIND fact below was verified against primary source code or dataset documentation fetched from GitHub [1]-[12]. Facts about older published papers (Google News, Yahoo! JAPAN, NRMS, DKN, NYT, Hulu) come from well-known peer-reviewed publications and are cited at their canonical DOIs or URLs [13]-[20], but were not re-fetched in this session. Those claims are kept to the papers' headline contributions, and any numbers from them are marked approximate. For several platforms (Snapchat Spotlight/Discover, Quora, Medium, Substack, Discord, Twitch, Disney+, Roku), primary engineering detail could not be retrieved. Those sections say so explicitly instead of filling gaps.

```
           Common shape of a news / social / content recommender
  +----------------+     +--------------------+     +---------------------+
  | ingest stream  | --> | content encoders   | --> | item index (ANN /   |
  | (articles,     |     | (text, entities,   |     | inverted / time-    |
  |  posts, votes) |     |  images, author)   |     | bucketed store)     |
  +----------------+     +--------------------+     +----------+----------+
                                                               |
  +----------------+     +--------------------+                v
  | user activity  | --> | user encoder       | --> +---------------------+
  | (clicks, dwell,|     | (history attention |     | candidate sources:  |
  |  follows)      |     |  / RNN / counts)   |     | follow graph, ANN,  |
  +----------------+     +--------------------+     | trending, co-visit  |
                                                    +----------+----------+
                                                               v
                                    +---------------------------------------+
                                    | ranker (CTR/dwell/quality, recency,   |
                                    | dedup, diversity, policy filters)     |
                                    +---------------------------------------+
```

## 28.1 Google News: Scalable Online Collaborative Filtering (2007) and Click-Behavior Profiles (2010)

### Problem
Google News aggregates stories from thousands of publishers. Its 2007 personalization paper by Das, Datar, Garg and Rajaram [13] set out the problem that still defines news recommendation: very large numbers of users and items, an item set that changes by the minute, and a need to produce recommendations in real time from implicit click feedback only.

### Architecture and models
The paper combines three collaborative-filtering signals into one score per candidate story [13]:

- **MinHash clustering.** Users are represented by the sets of stories they clicked. MinHash, a locality-sensitive hash for Jaccard similarity, puts users with overlapping click sets into the same buckets. Clustering runs as a batch MapReduce job.
- **PLSI (Probabilistic Latent Semantic Indexing).** A latent-class model of user-story co-occurrence, also fitted offline with MapReduce.
- **Co-visitation counts.** These are updated online. Two stories are "co-visited" when the same user clicks both within a time window. The counts give an item-to-item signal that adapts within minutes as new stories gain traffic.

At serving time, the scores for a candidate story from the user's clusters (MinHash and PLSI) and from co-visitation with the user's recent clicks are blended linearly [13]. Cluster memberships are batch artifacts. The click counts per cluster and the co-visitation counts are kept in a large online key-value store (Bigtable in the paper), so new stories get a score as soon as clicks arrive.

```
   offline (MapReduce)                    online (per request)
  +---------------------+         +-----------------------------------+
  | user click sets     |         | candidate stories (fresh)         |
  |  -> MinHash buckets |-------->| score = w1*minhash_cluster_clicks |
  |  -> PLSI classes    |         |       + w2*plsi_class_clicks      |
  +---------------------+         |       + w3*covisit(recent_clicks) |
  +---------------------+         +-----------------+-----------------+
  | click stream        |--updates-> cluster click counts / covisit   |
  +---------------------+         |  counts in a large KV store        |
                                  +-----------------------------------+
```

### Follow-up: content-based click-behavior profiles (2010)
Liu, Dolan and Pedersen [14] analyzed Google News click logs. They found that users' topical interests drift over time and are partly driven by the general news trend in their location. They built a Bayesian model of each user's "genuine" interest, separated from the local trend, and combined it with the existing CF method. The paper reports a substantial live click-through improvement over CF alone (roughly 30%; approximate, from the paper's live experiment) [14].

### Lessons
- **Split the pipeline by how fast each part must change.** Slow structure (clusters) is computed in batch. Fast statistics (co-visitation, counts per cluster) are updated online. This is the template for most later news systems.
- **CF alone fails on fresh items and new users.** Even in 2010, Google moved to a hybrid with a content profile [14].

## 28.2 Yahoo! JAPAN: Embedding-Based News Recommendation for Millions of Users (KDD 2017)

### Problem
Yahoo! JAPAN's homepage serves news to very large audiences, and article lifetimes are short. Okura et al. [15] replaced ID-based methods with learned representations, with a strict serving budget in mind.

### Architecture
The paper's design has three parts [15]:

1. **Article encoder.** A variant of the denoising autoencoder learns distributed representations of articles from their content. It is trained so that articles in the same category end up closer together. Because the representation comes from content, a brand-new article gets an embedding the moment it is published.
2. **User encoder.** A recurrent neural network (the paper compares RNN variants, with a GRU-based model performing best) reads the user's browsing history as a sequence of article embeddings and outputs a user vector.
3. **Scoring.** Relevance is the inner product of user and article vectors. That makes serving a dot product over fresh candidates, and lets user vectors be computed ahead of time and cached.

```
  article text --> [denoising AE] --> a_j  (computed at publish time)
  browse history a_1..a_t --> [GRU] --> u  (refreshed as the user reads)
  score(u, a_j) = <u, a_j>   -> top-k fresh articles, then business rules
```

### Lessons
The paper's main contribution is a reusable split: precompute item vectors from content, update user vectors incrementally, and score with an inner product. That gives cheap serving with no cold start for items. It is the ancestor of the two-tower retrieval used everywhere today (see Chapter 2). The authors report that the deployed system improved engagement metrics over the previous production method in online tests [15]; the exact figures are omitted here.

## 28.3 Microsoft News and the MIND Ecosystem

Microsoft News (MSN News) is the most openly documented news recommender. That is mostly because Microsoft Research released a production-scale benchmark and a family of open models built on it.

### 28.3.1 The MIND dataset
MIND (MIcrosoft News Dataset) was collected from anonymized behavior logs of the Microsoft News website. It contains **about 160k English news articles and more than 15 million impression logs generated by 1 million users** [1]. Every article has title, abstract, body, category and entities. Each impression log holds the clicked and non-clicked items plus the user's click history before that impression [1]. The companion paper is Wu et al., "MIND: A Large-scale Dataset for News Recommendation", ACL 2020 [1][16].

Data construction details from the official documentation [2]:

- **Users:** about 1M, sampled at random from users with at least 5 news clicks. **MIND-small** has 50,000 users.
- **Time span:** six weeks, 12 October to 22 November 2019. The test set is the last week. The training set is the fifth week, with click histories built from the first four weeks. Validation is the last day of the fifth week.
- **Files:** `behaviors.tsv` (impression ID, user ID, time, ordered click history, and the impression list with `-1`/`-0` click labels in shuffled order). `news.tsv` (ID, category, subcategory, title, abstract, URL, and title and abstract entities linked to Wikidata). Also 100-dimensional **TransE** entity and relation embeddings from a knowledge graph.

The impression-log format matters for system design. It records **the slate the user actually saw**, not just positives. That allows impression-aware negative sampling, where the negatives are items shown but not clicked, and it lets evaluation use per-impression ranking metrics (AUC, MRR, nDCG@5, nDCG@10) instead of global AUC.

### 28.3.2 The model family: DKN, NPA, NAML, LSTUR, NRMS
Microsoft's open-source `recommenders` repository ships five news models, all marked as invented or contributed by Microsoft, each with a MIND quick-start notebook [3]:

| Model | Core idea (per repo) | User modeling | Item modeling |
|---|---|---|---|
| DKN | Knowledge-graph entities fused with article text | Attention over clicked news | Word + entity (KG) CNN |
| NPA | Personalized attention | User-ID-conditioned attention | CNN over title words |
| NAML | Attentive multi-view learning | Attention over clicked news | Title, body, category views |
| LSTUR | Long- and short-term user representations | GRU (short) + user-ID embedding (long) | CNN over title |
| NRMS | Multi-head self-attention | Self-attention over clicked news | Self-attention over title words |

DKN was originally published for Bing News [17]. NRMS (Wu et al., EMNLP-IJCNLP 2019) [18] became the default baseline in the field.

**NRMS architecture** (as documented in the reference implementation [4]):

```
 title words (<=30) -> GloVe 300d -> multi-head self-attn (20 heads x 20 dim)
                    -> additive attention (hidden 200) -> news vector r_i
 clicked news r_1..r_50 -> multi-head self-attn -> additive attention -> user u
 candidate news r_c  -> score = <u, r_c>
 training: 1 positive + 4 sampled negatives from same impression, softmax CE
```

Reference hyperparameters from the notebook [4]: `title_size=30`, `word_emb_dim=300`, `his_size=50`, `npratio=4`, `head_num=20`, `head_dim=20`, `attention_hidden_dim=200`, `dropout=0.2`, Adam with learning rate 1e-4, batch size 32, 5 epochs. On the small MINDdemo split (5k users), validation group AUC rises from 0.479 untrained to 0.613 after 5 epochs, and nDCG@10 from 0.270 to 0.363 [4]. These are toy-scale numbers. They show the shape of the learning curve, not the model's full-data quality.

**Key trick: impression-level "npratio" training.** The model is trained as a (1 + K)-way softmax. The K negatives are sampled from the same impression, so they are items the user saw and skipped. This is a listwise approximation that matches the serving task, which is ordering one slate.

### 28.3.3 Pre-trained language models and the MIND leaderboard
Later Microsoft work replaced GloVe/CNN news encoders with pre-trained language models (the "PLM-NR" line). The public training code (`PLM4NewsRec`) uses a BERT-based news encoder, distributed with Horovod across 4 GPUs, with MIND's `behaviors.tsv` sharded per GPU [5]. Efficient-Transformer user encoders (Fastformer, and "UniUM" universal user modeling) then pushed the public leaderboard.

The live MIND leaderboard (fetched October 2026) [1]:

| Rank | Entry | Date | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|---|---|
| 1 | zetik | Jun 2026 | 0.7326 | 0.3780 | 0.4175 | 0.4735 |
| 2 | newsbang | May 2026 | 0.7316 | 0.3768 | 0.4162 | 0.4722 |
| 3 | UniUM-Fastformer-Pretrain | Oct 2021 | 0.7304 | 0.3770 | 0.4180 | 0.4718 |
| 4 | MINER | Sep 2021 | 0.7275 | 0.3724 | 0.4102 | 0.4661 |
| 6 | UniUM-Fastformer | Aug 2021 | 0.7268 | 0.3745 | 0.4151 | 0.4684 |
| 29 | UNBERT | Feb 2021 | 0.7207 | 0.3677 | 0.4041 | 0.4602 |
| 111 | MIND_Organizer | -- | 0.7012 | 0.3482 | 0.3808 | 0.4377 |

The leaderboard has plateaued. A 2021 pretrained Fastformer entry is still within 0.003 AUC of the 2026 leaders. On MIND, gains now come from bigger PLMs and pretraining on user behavior, not from new user-encoder architectures.

### 28.3.4 Lessons from the MIND ecosystem
- **Content encoders are the backbone.** In news, the item tower carries most of the information, and upgrading it (CNN to PLM) gives larger gains than upgrading the user tower.
- **Cache news vectors and compute users online.** News vectors from a heavy PLM are computed once at publish time. Only the lightweight user aggregator runs per request, which keeps PLM-based news recommendation affordable.
- **Log impressions, not just clicks.** The MIND format (shown slate plus click labels plus history) is a good template for any feed's training logs.

## 28.4 The New York Times: Collaborative Topic Modeling

The New York Times R&D/data team described its recommender evolution in a 2015 engineering post [19]. The early system was content-based, matching keyword and topic tags of articles a reader had read to new articles. Its weakness was over-reliance on tags. The team moved to **Collaborative Topic Modeling (CTM)** (Wang and Blei, KDD 2011). CTM represents each article by LDA topic proportions learned from its text, then adjusts those proportions with offsets learned from readership, so an article's position reflects both what it says and who reads it [19]. Readers are placed in the same topic space according to what they read, and recommendations are articles close to the reader's position.

Design points worth carrying forward:

- **The content prior handles cold start.** A new article starts at its pure text-topic position, so it can be recommended at once. Readership data then moves it.
- **Interpretable latent space.** Topics are human-readable, which matters to an editorial organization that has to explain and audit what the algorithm promotes.

## 28.5 Reddit: Closed-Form Ranking at the Core

### Problem
Reddit ranks posts inside communities (subreddits), on the home feed and in comment threads. Its original ranking functions are open source, and they remain a reference design for vote-driven ranking.

### "Hot" ranking for posts
From the open-source `_sorts.pyx` [6]:

```
score   = ups - downs
order   = log10(max(|score|, 1))
sign    = +1 / 0 / -1 by sign of score
seconds = epoch_seconds(post_time) - 1134028003
hot     = round(sign * order + seconds / 45000, 7)
```

Design properties:
- **Log-compressed votes.** The first 10 net votes count as much as the next 90.
- **Time is additive and never decays a post's own score.** Each newer post gets a fixed bonus: 45,000 seconds (12.5 hours) of recency equals a 10x change in net votes. So the score can be computed once per vote and stored, and the sort index stays stable. There is no periodic rescoring job, which is a key systems property.

### "Best" (confidence) ranking for comments
Comments use the **lower bound of the Wilson score interval** at about 80% confidence (z = 1.281551565545) on the upvote fraction [6]. A comment with 5 of 5 upvotes does not outrank one with 95 of 100. The ranking therefore gives credit for both quality and evidence, with no time term, since comment threads are short-lived contexts.

### "Controversial"
`magnitude ** balance`, where magnitude = ups + downs and balance is the minority/majority vote ratio [6]. Large, evenly split vote totals score highest.

### The modern ML home feed
Reddit's later home feed, built on an ML platform with feature stores, candidate generators and learned rankers, has been described in posts on the company's engineering subreddit, r/RedditEng. Those primary posts were not reachable for this edition, so no architecture or scale numbers are given here. The closed-form scores above are still part of the system design story. Votes and age are the strongest features of any learned ranker on Reddit, and a hot-style score is a natural candidate source and fallback.

## 28.6 Bluesky: Recommendation as an Open Protocol Service

### Problem
Bluesky, built on the AT Protocol, separates **hosting** (Personal Data Servers, PDS), **indexing** (AppView) and **ranking**. Anyone can publish a **custom feed**: a ranking algorithm users can subscribe to. This is a "marketplace of algorithms" in place of one platform ranker.

### Architecture
From the official feed-generator starter kit [7]:

```
   user client                PDS                     Feed Generator (did:web)
  +-----------+  getFeed   +------+  getFeedSkeleton  +----------------------+
  | app view  | ---------> | PDS  | ----------------> | index built from the |
  |           |            |      |  (JWT signed by   | firehose; returns    |
  |           | <--------- |      |   user's key)     | [post URIs + reason] |
  +-----------+  hydrated  +------+ <---------------- +----------^-----------+
                 posts       hydrate: post content,              |
                             author, like/repost counts   com.atproto.sync.
                                                          subscribeRepos (firehose)
```

1. A feed generator is a service identified by a **DID**. Each algorithm it hosts is declared by a record in the creator's repo and addressed by that record's **at-uri** [7].
2. The user's PDS resolves the at-uri to the generator's DID document and calls `app.bsky.feed.getFeedSkeleton` at the listed endpoint. The request carries a **JWT signed with the user's repo signing key**, so the generator knows who is asking and can personalize [7].
3. The generator returns only a **skeleton**: a list of post URIs, each with an optional reason. The PDS (and eventually an AppView) **hydrates** the posts with content, author info and aggregate counts [7].
4. Generators build their own index by subscribing to the **firehose** (`com.atproto.sync.subscribeRepos`). They index only what their algorithm needs, and data older than about 48 hours can usually be discarded [7].

### Protocol contract details (lexicons)
- `getFeedSkeleton` takes `feed` (at-uri), `limit` (1-100, default 50) and an opaque `cursor`. It returns `feed`, an optional `cursor` and an optional `reqId` (at most 100 chars, unique per request) [8].
- Each `skeletonFeedPost` has `post`, an optional `reason` (a repost reason or a pin reason) and an optional **`feedContext`** string of up to 2,000 characters. The client passes `feedContext` through, and it returns to the generator together with interactions [9].
- **Feedback loop:** `app.bsky.feed.sendInteractions` sends interactions with feed items "back to the feed generator that served them" [10]. Defined events include `requestMore`, `requestLess`, `clickthroughItem`, `clickthroughAuthor`, `clickthroughReposter`, `clickthroughEmbed`, `interactionSeen`, `interactionLike`, `interactionRepost`, `interactionReply`, `interactionQuote` and `interactionShare` [9].
- A generator advertises `acceptsInteractions` and a `contentMode` (`contentModeUnspecified` or `contentModeVideo`) [9].

The `feedContext` plus `reqId` design gives third-party rankers what in-house teams get from request logging. A generator can tag each served item with its own context string (candidate source, model version, experiment arm). It receives that tag back with the user's explicit "show more/less" or implicit "seen/clicked" events, so it can join impressions to outcomes without storing all client state.

### Cursor design
The sample algorithm orders by `indexedAt DESC, cid DESC`, encodes the cursor as the millisecond timestamp of the last item, and filters `indexedAt < cursor` on the next page [11]. The docs recommend that cursors be **unique per item**, for example timestamp plus CID [7]. The minimal sample encodes only the timestamp, so posts sharing a millisecond can be skipped across pages. Feed authors are expected to fix this in production.

### Lessons
- **Separate ranking from hydration.** Returning IDs only keeps ranking services cheap, stateless toward the client, and swappable.
- **Put the feedback channel in the protocol.** Without `sendInteractions`, third-party rankers would be blind to engagement.
- **Accept fragmentation of the ranking function as a deliberate product choice.** Moderation, spam filtering and quality become the responsibility of the shared hydration and labeling layers, not of each feed.

## 28.7 Mastodon: Chronological Fan-Out and Trend Detection

Mastodon is the instructive counterexample: a large social network whose **home timeline is not ranked by ML at all**. Its value for system designers is in how it does fan-out, filtering and trend detection.

### Home timeline: push fan-out into Redis
From `FeedManager` [12]:

- Each home feed is a Redis sorted set, `feed:home:<account_id>`, of status IDs **scored by status ID**. Status IDs are time-ordered (Snowflake-like), so the sort is chronological.
- `MAX_ITEMS = 800` items per feed. `trim` uses `zremrangebyrank` to keep the newest 800.
- **Fan-out on write:** `push_to_home` inserts a new status into each follower's feed. It skips recipients who have not signed in recently, so storage and work for dormant accounts are not wasted. A streaming update is pushed only if `subscribed:timeline:<id>` shows a connected client.
- **Boost (reblog) dedup:** if the original or a boost of it is already in the newest `REBLOG_FALLOFF = 80` items, later boosts are held back in a per-original set. When the shown boost is deleted, the oldest held-back one is promoted.
- **Rebuild on demand:** `populate_home` rebuilds a feed from scratch. It takes up to 400 of the user's own posts and up to 400 per followed account, and skips accounts whose `last_status_at` is older than the feed's oldest item. `merge_into_home` on a new follow pulls up to 200 of that account's posts [12].
- **Rule-based filtering** (`filter_from_home`) drops posts involving blocks or mutes (author, booster or mentioned accounts), posts in languages the user did not select for that author, replies to people the user does not follow, and boosts from accounts whose boosts are hidden [12].

Federated and local timelines are separate views. The "federated" view shows only what the user's server knows about, because "there is no global shared state between all servers" [21].

### Trending: a chi-square-like spike detector
Mastodon's only "algorithmic" surface is Trends. Both scorers use a squared-deviation test that rewards **acceleration** over an expected baseline.

**Trending links** [22]:
```
expected = accounts sharing the link one day earlier (min 1)
observed = accounts sharing it now
score    = 0                          if observed < 5 or expected > observed
         = (observed - expected)^2 / expected   otherwise
max_score kept with cooldown 2 days; decayed = max_score * 0.5^(age / 8h)
trend if decayed >= 1
```

**Trending statuses** [23]: `observed` = reblogs + favourites, `expected` = 1, threshold 5, score = (observed - 1)^2, **half-life 1 hour**, kept only while the decayed score is at least 0.3. Eligibility rules exclude replies, sensitive or content-warned posts, non-discoverable and silenced accounts, and invalid locales. Items whose score beats the rank-3 trend in their language, but which have not yet been approved, trigger a **human review request** [22][23].

### Lessons
- **Score against a baseline, not just counts.** Absolute counts favor the largest accounts. A per-item expected value (yesterday's level) catches spikes.
- **Different content needs different half-lives.** Posts use 1 hour and links use 8 hours.
- **Human review gates promotion.** Trends are allowed only after moderator approval or account trust. That is cheap safety for a small operator.

## 28.8 Snapchat: Embedding-Based Retrieval, GNNs and Generative Recommendation

### What is verifiable
Snap's engineering blog describes embedding-based retrieval (EBR) with two-tower models for Spotlight, its short-video feed, and other surfaces. That blog was not reachable for this edition, so no Spotlight-specific numbers are given. Snap's open-source and published work shows the stack that powers its recommendation surfaces:

- **GiGL (Gigantic Graph Learning)** is Snap's open-source library for GNN training and inference at "very large (billion) scale", built for horizontal scaling across preprocessing, training and inference [24]. Its pipeline has six components: Config Populator, Data Preprocessor, Subgraph Sampler, Split Generator, Trainer and Inferencer [24]. It supports node classification and link prediction, runs on Vertex AI/Kubeflow with Dataflow/Dataproc for data work, and ships an example of link prediction with tabularized subgraph sampling and a contrastive loss [25]. The accompanying paper is "GiGL: Large-Scale Graph Neural Networks at Snapchat" (KDD 2025) [24].
- **Friend-recommendation EBR.** Chapter 2 of this Atlas documents Snap's GNN-embedding EBR for friend recommendation and its ANN query expansion. That source reaches beyond friends-of-friends and reports gains in reciprocated communication for more than 800M MAU.
- **GRID** is Snap Research's open framework for generative recommendation with semantic IDs [26]. LLM text embeddings of items are quantized into hierarchical semantic IDs with Residual K-means, RVQ or RQ-VAE, and a TIGER-style transformer generates the next item's ID tokens. It is described in "Generative Recommendation with Semantic IDs: A Practitioner's Handbook" (CIKM 2025) [26].

```
 Snap-style retrieval stack (composite of [24]-[26] and Chapter 2)
 +----------------+   +-------------------+   +-------------------------+
 | social graph + |-->| GiGL: subgraph    |-->| user/item/node embeds    |
 | engagement     |   | sampling, GNN     |   | -> ANN index (EBR)       |
 +----------------+   +-------------------+   +------------+------------+
 +----------------+   +-------------------+                v
 | item text/LLM  |-->| semantic IDs      |-->  generative retrieval (GRID)
 +----------------+   +-------------------+                v
                                              multi-source candidates -> ranker
```

### Lessons
At Snap the social graph is a first-class training signal. GNN embeddings serve as user and item representations across friend, content and possibly ads surfaces, and Snap invests in reusable graph infrastructure (GiGL) instead of building one model per surface.

## 28.9 Streaming and Long-Tail Content Platforms: Hulu, Disney+, Roku, Twitch

### Hulu
Hulu's research group published **CF-NADE** (Zheng et al., ICML 2016), a neural autoregressive collaborative-filtering model [20]. It models a user's ratings as a product of conditionals, shares parameters across rating levels, and reported state-of-the-art results on MovieLens and Netflix benchmarks at the time [20]. In production, Hulu's recommendations have to handle TV series as a unit (episode versus show), live TV, and a mix of subscription tiers. Detailed production architecture was not available from primary sources for this edition.

### Disney+, Roku and Twitch
No primary engineering source on the recommender architecture of Disney+, Roku or Twitch could be retrieved for this edition, so no design claims are made about them. From a system-design view, the problem shapes differ:

| Platform | Dominant problem shape | Implication for design |
|---|---|---|
| Disney+ | Small, curated, franchise-heavy catalog | Row/page assembly and editorial control dominate (Ch. 15) |
| Roku | Cross-app aggregator on a TV device | Recs span third-party channels; few, coarse signals per user |
| Twitch | Live items exist only while streaming | Candidate set depends on availability; repeat consumption |
| Hulu | Episodic TV, live TV, mixed tiers | Show-level vs episode-level modeling; continue-watching |

This table is analysis from the problem structure. It does not report the companies' disclosed designs.

## 28.10 Quora, Medium, Substack and Discord

These platforms have described parts of their feeds publicly over the years, including Quora's ML-ranked question feed, Medium's topic- and follow-based feed, Substack's network recommendations between publications, and Discord's server discovery. Their primary engineering posts were not reachable for this edition, so this Atlas gives no architectural specifics for them. Readers should go to each company's engineering blog. The general pattern from the systems above applies: follow-graph candidates, plus content-similarity candidates, plus a learned or rule-based ranker with freshness. That pattern is a reasonable starting hypothesis, not a reported fact.

## 28.11 Cross-Platform Comparison

| System | Year | Item signal | User signal | Candidate generation | Ranking | Freshness handling | Source |
|---|---|---|---|---|---|---|---|
| Google News | 2007 | Click sets | Click history | MinHash, PLSI clusters, co-visitation | Linear blend of CF scores | Online counts, co-visit window | [13] |
| Google News | 2010 | Topic of content | Bayesian interest vs local trend | CF + content profile | Hybrid score | Trend modeling | [14] |
| Yahoo! JAPAN | 2017 | Denoising-AE article vector | GRU over history | Inner product over fresh set | Dot product + rules | Item vectors at publish | [15] |
| Microsoft News | 2019-21 | Self-attention / PLM title encoder | Attention over 50 clicks | Dot product | NRMS-style softmax over impression | Cached news vectors | [3][4][18] |
| NYT | 2015 | LDA topics + reader offsets | Topic position of reader | Nearest in topic space | Similarity | Content prior for new items | [19] |
| Reddit (classic) | 2005+ | Votes | None (per-community) | Subreddit sets | Closed-form hot / Wilson | Additive time term | [6] |
| Bluesky | 2023+ | Anything (generator-defined) | Requester DID via JWT | Generator's own firehose index | Generator-defined | ~48h index window typical | [7]-[11] |
| Mastodon | 2016+ | None (chronological) | Follow graph | Fan-out on write | Status ID (time) | 800-item cap; trends decay | [12][22][23] |
| Snap | 2023-25 | GNN / LLM / semantic IDs | Graph + engagement | EBR, generative | Learned ranker | Not reported here | [24]-[26] |

## Key Takeaways for System Designers

1. **Content encoders beat IDs when items are perishable.** Every successful news system here (Google 2010, Yahoo! JAPAN, Microsoft, NYT) moved to content representations so new items can be scored at publish time. Compute item vectors once, cache them, and keep the per-request user computation light.
2. **Split by rate of change.** Batch-learned structure (clusters, topic models, encoders) plus online counters (co-visitation, vote totals, trend windows) is the durable architecture, from Google News 2007 to Mastodon trends.
3. **Closed-form scores are infrastructure, not legacy.** Reddit's hot score can be computed once per vote with no rescoring jobs. Mastodon's spike detector compares to a baseline with tunable half-lives. Both make cheap candidate sources, fallbacks and features for learned rankers.
4. **Train on impressions.** MIND's format (shown slate, click labels, prior history) enables impression-aware negatives and per-impression metrics. Log what was shown, with request IDs.
5. **Benchmark plateaus point to where gains come from.** On MIND, 2021 pretrained-Fastformer entries are within about 0.3 AUC points of the 2026 leaders. Larger language-model item encoders and behavior pretraining are now the levers, not new attention variants.
6. **Protocol-level ranking needs protocol-level feedback.** Bluesky's `feedContext` + `reqId` + `sendInteractions` gives third-party rankers attribution and feedback loops. Anyone designing an open or multi-tenant ranking system needs the same three pieces: an opaque per-item context, a request ID, and an event return channel.
7. **Fan-out with guardrails.** Mastodon's push model caps storage per feed (800), skips inactive users, deduplicates boosts in the recent window, and rebuilds feeds on demand. That is a fine pattern even beneath an ML ranker, as the in-network source.
8. **Different content types need different decay.** Mastodon decays posts with a 1-hour half-life and links with 8 hours. Reddit's hot score trades 12.5 hours for 10x votes. Tune recency per surface and item type, not globally.
9. **Governance is architecture.** Chronological-only (Mastodon), marketplace-of-feeds (Bluesky), and centralized ML (Snap, Microsoft) are different allocations of control. They change where moderation, spam defense and quality signals must live.
10. **Shared graph infrastructure pays off.** Snap's GiGL shows the value of one billion-scale GNN platform producing embeddings for many retrieval surfaces, instead of one model per product.

## References

1. Microsoft. "MIND: MIcrosoft News Dataset" (homepage and leaderboard), 2020-2026. https://msnews.github.io/ (fetched via https://raw.githubusercontent.com/msnews/msnews.github.io/master/index.html)
2. Microsoft. "MIND dataset introduction" (format and construction documentation), 2020. https://github.com/msnews/msnews.github.io/blob/master/assets/doc/introduction.md
3. Microsoft / Recommenders Team (Linux Foundation AI & Data). "Recommenders" repository README (news algorithms DKN, LSTUR, NAML, NPA, NRMS). https://github.com/recommenders-team/recommenders
4. Recommenders Team. "NRMS: Neural News Recommendation with Multi-Head Self-Attention" quick-start notebook on MIND. https://github.com/recommenders-team/recommenders/blob/main/examples/00_quick_start/nrms_MIND.ipynb
5. Wu, C. "PLM4NewsRec" (code for PLM-empowered news recommendation). https://github.com/wuch15/PLM4NewsRec
6. Reddit. `r2/r2/lib/db/_sorts.pyx` (hot, confidence and controversy sorts), reddit open-source archive. https://github.com/reddit-archive/reddit/blob/master/r2/r2/lib/db/_sorts.pyx
7. Bluesky Social. "ATProto Feed Generator" starter kit README. https://github.com/bluesky-social/feed-generator
8. Bluesky Social. Lexicon `app.bsky.feed.getFeedSkeleton`. https://github.com/bluesky-social/atproto/blob/main/lexicons/app/bsky/feed/getFeedSkeleton.json
9. Bluesky Social. Lexicon `app.bsky.feed.defs` (skeletonFeedPost, interaction events, generatorView). https://github.com/bluesky-social/atproto/blob/main/lexicons/app/bsky/feed/defs.json
10. Bluesky Social. Lexicon `app.bsky.feed.sendInteractions`. https://github.com/bluesky-social/atproto/blob/main/lexicons/app/bsky/feed/sendInteractions.json
11. Bluesky Social. Sample feed algorithm `whats-alf.ts`. https://github.com/bluesky-social/feed-generator/blob/main/src/algos/whats-alf.ts
12. Mastodon gGmbH. `app/lib/feed_manager.rb` (home feed fan-out). https://github.com/mastodon/mastodon/blob/main/app/lib/feed_manager.rb
13. Das, A., Datar, M., Garg, A., Rajaram, S. "Google News Personalization: Scalable Online Collaborative Filtering." Google, WWW 2007. https://dl.acm.org/doi/10.1145/1242572.1242610 (not re-fetched this session)
14. Liu, J., Dolan, P., Pedersen, E. R. "Personalized News Recommendation Based on Click Behavior." Google, IUI 2010. https://dl.acm.org/doi/10.1145/1719970.1719976 (not re-fetched this session)
15. Okura, S., Tagami, Y., Ono, S., Tajima, A. "Embedding-based News Recommendation for Millions of Users." Yahoo! JAPAN, KDD 2017. https://dl.acm.org/doi/10.1145/3097983.3098108 (not re-fetched this session)
16. Wu, F., Qiao, Y., Chen, J.-H., Wu, C., Qi, T., Lian, J., Liu, D., Xie, X., Gao, J., Wu, W., Zhou, M. "MIND: A Large-scale Dataset for News Recommendation." Microsoft, ACL 2020. https://aclanthology.org/2020.acl-main.331/
17. Wang, H., Zhang, F., Xie, X., Guo, M. "DKN: Deep Knowledge-Aware Network for News Recommendation." Microsoft Research Asia, WWW 2018. https://arxiv.org/abs/1801.08284 (not re-fetched this session)
18. Wu, C., Wu, F., Ge, S., Qi, T., Huang, Y., Xie, X. "Neural News Recommendation with Multi-Head Self-Attention." Microsoft / Tsinghua, EMNLP-IJCNLP 2019. https://aclanthology.org/D19-1671/ (not re-fetched this session)
19. Spangher, A. "Building the Next New York Times Recommendation Engine." The New York Times (NYT Open blog), 2015. https://open.nytimes.com/ (post index; exact article URL not verified this session, locate by title)
20. Zheng, Y., Tang, B., Ding, W., Zhou, H. "A Neural Autoregressive Approach to Collaborative Filtering." Hulu, ICML 2016. https://arxiv.org/abs/1605.09477 (not re-fetched this session)
21. Mastodon. User documentation, `content/en/user/network.md` (local and federated timelines). https://github.com/mastodon/documentation/blob/main/content/en/user/network.md
22. Mastodon gGmbH. `app/models/trends/links.rb` (trending links scorer). https://github.com/mastodon/mastodon/blob/main/app/models/trends/links.rb
23. Mastodon gGmbH. `app/models/trends/statuses.rb` (trending statuses scorer). https://github.com/mastodon/mastodon/blob/main/app/models/trends/statuses.rb
24. Snap Research. "GiGL: Gigantic Graph Learning" repository and paper "GiGL: Large-Scale Graph Neural Networks at Snapchat," KDD 2025. https://github.com/snap-research/GiGL ; https://arxiv.org/abs/2502.15054
25. Snapchat. GiGL (active development repository). https://github.com/Snapchat/GiGL
26. Ju, C., Collins, L., Neves, L., Kumar, B., Wang, L. Y., Zhao, T., Shah, N. "Generative Recommendation with Semantic IDs: A Practitioner's Handbook" (GRID). Snap Research, CIKM 2025. https://github.com/snap-research/GRID ; https://arxiv.org/abs/2507.22224
